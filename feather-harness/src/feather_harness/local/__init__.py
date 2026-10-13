"""Assemble one local Feather run from the shared loop and tools."""

import asyncio
import json
import logging
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path

from ..core.compaction import CompactionConfig
from ..core.messages import SystemMessage, UserMessage
from ..core.model import ChatCompletionsModel, ModelSettings
from ..core.runner import FeatherAgent, RunRequest, RunResult
from ..production import (
    Profile,
    build_production_instructions,
    production_compaction,
    production_tools,
    validate_profile,
)
from ..prompt import build_instructions, compose_additional_instructions, prompt_metadata
from ..scripts.install_skills import install_production_skills, install_skills, install_task_skills
from ..scripts.save_transcript import save_transcript
from ..subagents import PARENT_GUIDANCE, AgentSession, SubagentConfig, Subagents
from ..tools import Tool
from ..tools.agents import subagent_tools
from ..tools.files import FileTools
from ..tools.mcp import McpServerSpec, run_with_mcp
from ..tools.process import ProcessTools
from ..tools.todo import TodoList
from ..tools.web import WebProvider, web_fetch_tool, web_search_tool
from .filesystem import LocalFileRuntime
from .process import LocalShellRuntime

logger = logging.getLogger(__name__)


async def run_local(
    prompt: str,
    *,
    model: ChatCompletionsModel,
    workspace: Path,
    task_python: Path | None = None,
    max_turns: int = 100,
    model_settings: ModelSettings = ModelSettings(),
    compaction: CompactionConfig = CompactionConfig(),
    mcp_servers: tuple[McpServerSpec, ...] = (),
    skills_dir: Path | None = None,
    profile: Profile = "prod",
    enable_image_tools: bool | None = None,
    subagents: SubagentConfig | None = None,
    additional_instructions: str | None = None,
    eval_prompt_output: Path | None = None,
    web_provider: WebProvider | None = None,
) -> RunResult:
    validate_profile(profile)
    if subagents is not None and profile != "prod":
        raise ValueError("Subagents are only available in the prod profile")
    if enable_image_tools is None:
        enable_image_tools = profile == "prod"
    workspace = LocalFileRuntime(workspace).workspace
    # Bundled skills are harness-owned and replaced at every start, so a workspace reused
    # across profiles carries this run's manuals rather than another profile's copies.
    install_skills(workspace, overwrite=True)
    if profile == "prod":
        install_production_skills(workspace)
    if skills_dir is not None:
        install_task_skills(workspace, skills_dir)
    _exclude_harness_paths(workspace)
    if profile == "prod":
        instructions = build_production_instructions(
            skills_root=workspace / ".agents",
            additional_instructions=additional_instructions,
            workspace=str(workspace),
        )
        compaction = production_compaction(compaction)
    else:
        instructions = build_instructions(skills_root=workspace / ".agents")
        instructions = compose_additional_instructions(
            instructions, additional_instructions, workspace=str(workspace)
        )
    parent_instructions = instructions
    if subagents is not None:
        parent_instructions += (
            f"\n\n{PARENT_GUIDANCE}\n"
            f"Limits: {subagents.max_concurrent} concurrently running children, "
            f"{subagents.max_agents} children total, and {subagents.max_turns} turns per "
            "child across its lifetime, including follow-ups."
        )
    if eval_prompt_output is not None:
        eval_prompt_output.parent.mkdir(parents=True, exist_ok=True)
        temporary = eval_prompt_output.with_suffix(".tmp")
        temporary.write_text(json.dumps(
            prompt_metadata(parent_instructions, profile=profile), indent=2
        ))
        temporary.replace(eval_prompt_output)
    request = RunRequest(
        messages=(
            SystemMessage(content=parent_instructions),
            UserMessage(content=prompt),
        ),
        max_turns=max_turns,
        model_settings=model_settings,
    )

    async def run_child(agent_id: str, serve: AgentSession) -> RunResult:
        return await _with_local_agent(
            serve,
            model=model,
            workspace=workspace,
            task_python=task_python,
            state_directory=workspace / ".feather/agents" / agent_id,
            compaction=compaction,
            mcp_servers=mcp_servers,
            profile=profile,
            enable_image_tools=enable_image_tools,
            web_provider=web_provider,
        )

    async def save_result(agent_id: str, run_number: int, result: RunResult) -> str:
        path = workspace / ".feather/agents" / agent_id / f"run-{run_number}.json"
        await asyncio.to_thread(
            save_transcript,
            path,
            result.model_dump_json(),
        )
        return str(path)

    agents = (
        Subagents(
            run_child,
            instructions=instructions,
            config=subagents,
            model_settings=model_settings,
            save_result=save_result,
        )
        if subagents is not None
        else None
    )

    async def run_parent(agent: FeatherAgent) -> RunResult:
        if agents is None:
            return await agent.run(request)
        try:
            async with agents:
                return await agent.run(request)
        finally:
            try:
                await asyncio.to_thread(
                    save_transcript,
                    workspace / ".feather/agents/manifest.json",
                    json.dumps(
                        {"agents": [status.model_dump(mode="json") for status in agents.list()]}
                    ),
                )
            except OSError:
                logger.warning("Could not save the subagent status manifest", exc_info=True)

    return await _with_local_agent(
        run_parent,
        model=model,
        workspace=workspace,
        task_python=task_python,
        state_directory=workspace / ".feather",
        compaction=compaction,
        mcp_servers=mcp_servers,
        profile=profile,
        enable_image_tools=enable_image_tools,
        agents=agents,
        web_provider=web_provider,
    )


async def _with_local_agent(
    serve: AgentSession,
    *,
    model: ChatCompletionsModel,
    workspace: Path,
    state_directory: Path,
    compaction: CompactionConfig,
    mcp_servers: tuple[McpServerSpec, ...],
    profile: Profile,
    enable_image_tools: bool,
    task_python: Path | None = None,
    agents: Subagents | None = None,
    web_provider: WebProvider | None = None,
) -> RunResult:
    file_transport = LocalFileRuntime(workspace)
    files = FileTools(file_transport, enable_image_reads=enable_image_tools)
    todos = TodoList()

    def compaction_state() -> tuple[str, ...]:
        return (*todos.compaction_state(), *(agents.compaction_state() if agents else ()))

    async def publish_transcript(content: str) -> None:
        await asyncio.to_thread(save_transcript, state_directory / "transcript.jsonl", content)

    async with LocalShellRuntime(
        workspace, log_directory=state_directory / "tasks", task_python=task_python,
    ) as transport:
        processes = ProcessTools(transport)
        builtins = (
            processes.bash_tool(),
            processes.task_list_tool(),
            processes.task_output_tool(),
            processes.task_stop_tool(),
            files.read_tool(),
            files.edit_tool(),
            files.write_tool(),
            files.grep_tool(),
            files.glob_tool(),
            todos.tool(),
        )
        if web_provider is not None:
            builtins = (*builtins, web_search_tool(web_provider), web_fetch_tool(web_provider))
        if profile == "prod":
            builtins = production_tools(builtins, image_reads=enable_image_tools)
        if task_python is not None:
            builtins = tuple(
                replace(tool, spec=tool.spec.model_copy(update={
                    "description": tool.spec.description +
                    ' python and python3 use the selected task Python. Use "$FEATHER_PYTHON" '
                    'for Paper packages installed with Feather.',
                })) if tool.spec.name == "bash" else tool
                for tool in builtins
            )
        if agents is not None:
            builtins = (*builtins, *subagent_tools(agents))

        async def run(mcp_tools: tuple[Tool, ...]) -> RunResult:
            agent = FeatherAgent(
                model=model,
                tools=(*mcp_tools, *builtins),
                compaction=compaction,
                compaction_tool_state=compaction_state,
                save_transcript=publish_transcript,
                response_recovery=profile == "prod",
            )
            return await serve(agent)

        return await run_with_mcp(
            mcp_servers,
            run,
            reserved_names=frozenset(tool.spec.name for tool in builtins),
            enable_image_tools=enable_image_tools,
            file_transport=file_transport,
        )


_HARNESS_PATHS = (".agents/", ".feather/")


def _exclude_harness_paths(workspace: Path) -> None:
    """Keep harness-owned files out of `git add -A` so patch-graded tasks see only agent edits."""
    if shutil.which("git") is None:
        return
    # The agent user may not own the checkout; this read-only probe must not trip safe.directory.
    probe = subprocess.run(
        ["git", "-c", "safe.directory=*", "rev-parse", "--show-prefix", "--git-path", "info/exclude"],
        cwd=workspace,
        capture_output=True,
        text=True,
        check=False,
    )
    if probe.returncode != 0:
        return
    prefix, exclude = probe.stdout.splitlines()
    exclude_path = workspace / exclude
    existing = exclude_path.read_text() if exclude_path.exists() else ""
    entries = [f"/{prefix}{path}" for path in _HARNESS_PATHS]
    missing = [entry for entry in entries if entry not in existing.splitlines()]
    if not missing:
        return
    exclude_path.parent.mkdir(parents=True, exist_ok=True)
    separator = "\n" if existing and not existing.endswith("\n") else ""
    with exclude_path.open("a") as handle:
        handle.write(separator + "\n".join(missing) + "\n")
