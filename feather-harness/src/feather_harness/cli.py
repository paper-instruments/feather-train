"""Noninteractive local Feather entrypoint."""

import argparse
import asyncio
import json
import logging
import os
import sys
from importlib.metadata import version
from pathlib import Path

from openai import AsyncOpenAI
from pydantic import Field, model_validator

from .core.compaction import CompactionConfig
from .core.messages import FrozenModel
from .core.model import ChatCompletionsModel, ModelSettings, RetryPolicy
from .core.runner import AgentRunError
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
        "--version", action="version", version=f"feather {version('feather-harness')}"
    )
    parser.add_argument("prompt", help="Task instructions; use '-' to read them from stdin.")
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
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
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
