"""Noninteractive local Feather entrypoint."""

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

from openai import AsyncOpenAI
from pydantic import Field, model_validator

from .core.compaction import CompactionConfig
from .core.messages import AssistantMessage, FrozenModel
from .core.model import ChatCompletionsModel, ModelSettings, RetryPolicy
from .core.runner import AgentRunError, RunResult
from .core.token_budget import PromptTokenBudget
from .local import run_local
from .production import Profile
from .subagents import SubagentConfig
from .tools.mcp import McpServerSpec
from .tools.web.exa import ExaWebProvider

logger = logging.getLogger(__name__)


class LocalConfig(FrozenModel):
    model: str = Field(min_length=1)
    base_url: str | None = None
    workspace: Path = Path(".")
    python: Path | None = None
    profile: Profile = "prod"
    enable_image_tools: bool | None = None
    enable_web_tools: bool = False
    max_turns: int = Field(default=100, ge=1)
    model_settings: ModelSettings = Field(default_factory=ModelSettings)
    model_retry: RetryPolicy = Field(default_factory=RetryPolicy)
    prompt_token_budget: PromptTokenBudget | None = None
    compaction: CompactionConfig = Field(default_factory=CompactionConfig)
    mcp_servers: tuple[McpServerSpec, ...] = ()
    skills_dir: Path | None = None
    subagents: SubagentConfig | None = None
    additional_instructions: str | None = None
    eval_prompt_output: Path | None = None

    @model_validator(mode="after")
    def validate_subagent_profile(self) -> "LocalConfig":
        if self.subagents is not None and self.profile != "prod":
            raise ValueError("Subagents are only available in the prod profile")
        return self


def inspect_trace(path: Path) -> dict[str, object]:
    """Summarize a saved run for CLI inspection without re-executing the agent."""
    document = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(document, dict) and document.get("schema_version") == "tldc.feather-agent-trace.v2":
        document = document["trace"]
    if not isinstance(document, dict) or document.get("schema_version") != "feather.run.v1":
        raise ValueError("Expected feather.run.v1 or tldc.feather-agent-trace.v2")
    result = RunResult.model_validate(document)
    tools: dict[str, int] = {}
    for message in result.execution_messages():
        if isinstance(message, AssistantMessage):
            for call in message.tool_calls:
                tools[call.name] = tools.get(call.name, 0) + 1
    return {
        "schema_version": result.schema_version,
        "stop_reason": result.stop_reason,
        "stop_message": result.stop_message,
        "num_turns": result.num_turns,
        "compactions": result.compactions,
        "recoveries": result.recoveries,
        "final_answer": result.final_answer,
        "usage": result.usage().model_dump(mode="json", exclude_none=True),
        "tools": dict(sorted(tools.items())),
    }


async def run(config: LocalConfig, prompt: str) -> int:
    web_provider = (
        ExaWebProvider(api_key=os.environ.get("EXA_API_KEY", ""))
        if config.enable_web_tools
        else None
    )
    # Feather owns request retries so they back off like the training sampler.
    async with AsyncOpenAI(base_url=config.base_url, max_retries=0) as client:
        try:
            result = await run_local(
                prompt,
                model=ChatCompletionsModel(
                    client=client,
                    model=config.model,
                    prompt_token_budget=config.prompt_token_budget,
                    stream=True,
                    retry=config.model_retry,
                ),
                workspace=config.workspace,
                task_python=config.python,
                max_turns=config.max_turns,
                model_settings=config.model_settings,
                compaction=config.compaction,
                mcp_servers=config.mcp_servers,
                skills_dir=config.skills_dir,
                profile=config.profile,
                enable_image_tools=config.enable_image_tools,
                subagents=config.subagents,
                additional_instructions=config.additional_instructions,
                eval_prompt_output=config.eval_prompt_output,
                web_provider=web_provider,
            )
        except AgentRunError as exc:
            # Preserve the loop's partial transcript when a model or tool fails.
            sys.stdout.write(exc.result.model_dump_json() + "\n")
            logger.error("%s", exc)
            return 1
    sys.stdout.write(result.model_dump_json() + "\n")
    return 0 if result.stop_reason == "completed" else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Feather on one task in a local workspace.")
    parser.add_argument(
        "prompt",
        nargs="?",
        help="Task instructions; use '-' to read them from stdin.",
    )
    parser.add_argument("--config", type=Path, help="JSON configuration file.")
    parser.add_argument("--model", help="Model identifier at the Chat Completions endpoint.")
    parser.add_argument(
        "--base-url", help="OpenAI-compatible API base URL, including /v1 when needed."
    )
    parser.add_argument(
        "--workdir",
        dest="workspace",
        help="Existing workspace directory (default: current directory).",
    )
    parser.add_argument("--max-turns", type=int, help="Maximum agent turns (default: 100).")
    parser.add_argument(
        "--python", type=Path,
        help="Python executable for task shell commands; Feather keeps its own runtime.",
    )
    parser.add_argument("--profile", choices=("train", "prod"), help="Startup profile (default: prod).")
    parser.add_argument(
        "--inspect",
        type=Path,
        metavar="TRACE",
        help="Print a JSON summary of a saved RunResult and exit.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    if args.inspect is not None:
        if args.prompt is not None:
            parser.error("--inspect does not take a prompt")
        try:
            sys.stdout.write(json.dumps(inspect_trace(args.inspect)) + "\n")
            return 0
        except Exception as exc:
            logger.error("%s: %s", type(exc).__name__, exc)
            return 1
    if args.prompt is None:
        parser.error("the following arguments are required: prompt")
    try:
        values = json.loads(args.config.read_text()) if args.config else {}
        overrides = {
            name: getattr(args, name)
            for name in ("model", "base_url", "workspace", "python", "max_turns", "profile")
            if getattr(args, name) is not None
        }
        config = LocalConfig.model_validate({**values, **overrides})
        prompt = sys.stdin.read() if args.prompt == "-" else args.prompt
        if not prompt.strip():
            raise ValueError("Task instructions must not be empty")
        return asyncio.run(run(config, prompt))
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        logger.error("%s: %s", type(exc).__name__, exc)
        return 1
