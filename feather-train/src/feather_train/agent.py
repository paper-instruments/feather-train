"""Run the shared Feather agent against an already-provisioned E2B sandbox."""

from contextlib import nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING

from feather_harness import FileTools
from feather_harness.core.compaction import CompactionConfig
from feather_harness.core.messages import SystemMessage, UserMessage
from feather_harness.core.model import ChatCompletionsModel, ModelSettings
from feather_harness.core.runner import FeatherAgent, RunRequest, RunResult
from feather_harness.production import (
    Profile,
    build_production_instructions,
    production_compaction,
    production_tools,
    validate_profile,
)
from feather_harness.prompt import (
    SKILLS_ROOT,
    build_instructions,
    compose_additional_instructions,
    prompt_metadata,
)
from feather_harness.tools import Tool
from feather_harness.tools.mcp import McpServerSpec, run_with_mcp
from feather_harness.tools.process import ProcessTools
from feather_harness.tools.todo import TodoList
from feather_harness.tools.web import WEB_TOOL_NAMES, WebProvider, web_fetch_tool, web_search_tool

from feather_train.sandbox.transport import E2BRuntime

if TYPE_CHECKING:
    from feather_train.observability.braintrust import BraintrustRollout

PROCESS_TOOLS = ("bash", "task_list", "task_output", "task_stop")
DEFAULT_TOOLS = (*PROCESS_TOOLS, "read", "edit", "write", "grep", "glob", "todo_write")
SUPPORTED_TOOLS = (*DEFAULT_TOOLS, *WEB_TOOL_NAMES)


@dataclass
class ToolMetrics:
    known_calls: int = 0
    invalid_calls: int = 0

    def wrap(self, tool: Tool) -> Tool:
        async def invoke(arguments: str):
            self.known_calls += 1
            result = await tool.invoke(arguments)
            if result.error == "invalid_arguments":
                self.invalid_calls += 1
            return result

        return Tool(tool.spec, invoke)


async def run_e2b(
    prompt: str,
    *,
    model: ChatCompletionsModel,
    transport: E2BRuntime,
    max_turns: int = 100,
    model_settings: ModelSettings = ModelSettings(),
    compaction: CompactionConfig = CompactionConfig(),
    mcp_servers: tuple[McpServerSpec, ...] = (),
    enable_image_tools: bool = False,
    enable_image_reads: bool | None = None,
    profile: Profile = "train",
    instructions: str | None = None,
    additional_instructions: str | None = None,
    tools: tuple[str, ...] = DEFAULT_TOOLS,
    web_provider: WebProvider | None = None,
    metrics: ToolMetrics | None = None,
    rollout: "BraintrustRollout | None" = None,
) -> RunResult:
    validate_profile(profile)
    validate_tool_selection(tools)
    image_reads = enable_image_tools if enable_image_reads is None else enable_image_reads
    if image_reads and not enable_image_tools:
        raise ValueError("enable_image_reads requires enable_image_tools")
    if profile == "prod":
        system_instructions = build_production_instructions(
            skills_root=SKILLS_ROOT,
            task_instructions=instructions,
            additional_instructions=additional_instructions,
            workspace=transport.workspace if additional_instructions is not None else ".",
        )
        compaction = production_compaction(compaction)
    else:
        system_instructions = build_instructions(instructions)
        if additional_instructions is not None:
            system_instructions = compose_additional_instructions(
                system_instructions, additional_instructions,
                workspace=transport.workspace,
            )
    if rollout is not None:
        rollout.add_metadata(prompt_metadata(system_instructions, profile=profile))
    request = RunRequest(
        messages=(
            SystemMessage(content=system_instructions),
            UserMessage(content=prompt),
        ),
        max_turns=max_turns,
        model_settings=model_settings,
    )
    files = FileTools(transport, enable_image_reads=image_reads)
    processes = ProcessTools(transport)
    todos = TodoList()
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
    if set(tools) & set(WEB_TOOL_NAMES):
        if web_provider is None:
            raise ValueError("Web tools require a web provider")
        builtins = (*builtins, web_search_tool(web_provider), web_fetch_tool(web_provider))
    builtins = tuple(tool for tool in builtins if tool.spec.name in tools)
    if profile == "prod":
        builtins = production_tools(builtins, image_reads=image_reads)

    async def run(mcp_tools: tuple[Tool, ...]) -> RunResult:
        selected = (*mcp_tools, *builtins)
        if rollout is not None:
            selected = tuple(rollout.wrap_tool(tool) for tool in selected)
        if metrics is not None:
            selected = tuple(metrics.wrap(tool) for tool in selected)
        with (
            rollout.trace_agent(model)
            if rollout is not None
            else nullcontext(model) as traced_model
        ):
            agent = FeatherAgent(
                model=traced_model,
                tools=selected,
                compaction=compaction,
                compaction_tool_state=todos.compaction_state,
                save_transcript=transport.save_transcript,
                response_recovery=profile == "prod",
            )
            return await agent.run(request)

    async with transport:
        return await run_with_mcp(
            mcp_servers,
            run,
            reserved_names=frozenset(tool.spec.name for tool in builtins),
            enable_image_tools=enable_image_tools,
        )


def validate_tool_selection(tools: tuple[str, ...]) -> None:
    unknown = set(tools) - set(SUPPORTED_TOOLS)
    if unknown:
        raise ValueError(f"Unsupported tools: {sorted(unknown)}")
    process_tools = set(PROCESS_TOOLS) & set(tools)
    if process_tools and process_tools != set(PROCESS_TOOLS):
        raise ValueError("Bash and background-task tools must be selected together")
