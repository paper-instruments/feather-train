"""Install Feather in Harbor's sandbox; Harbor owns lifecycle and grading."""

import hashlib
import json
import re
import shlex
import tempfile
from importlib.metadata import version
from pathlib import Path

from feather_harness import McpServerSpec, RunResult
from feather_harness.cli import LocalConfig
from feather_harness.core.token_budget import PromptTokenBudget
from feather_harness.production import Profile
from harbor.agents.installed.base import (
    BaseInstalledAgent,
    EnvVar,
    PackageSpec,
    with_prompt_template,
)
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from harbor.models.trajectories.agent import Agent
from pydantic import JsonValue

from feather_eval.atif import write_atif

INSTALL_DIR = "/installed-agent/feather"
PYTHON = f"{INSTALL_DIR}/venv/bin/python"
UV_VERSION = "0.11.14"


class FeatherHarborAgent(BaseInstalledAgent):
    SUPPORTS_ATIF = True
    ENV_VARS = [EnvVar(kwarg="api_key", env="OPENAI_API_KEY", env_fallback="OPENAI_API_KEY")]
    SYSTEM_PACKAGES = {
        **BaseInstalledAgent.SYSTEM_PACKAGES,
        "file": PackageSpec.standard("file"),
        "libreoffice": PackageSpec(
            commands=("libreoffice",),
            packages={
                manager: ("libreoffice-calc", "libreoffice-writer", "libreoffice-impress")
                for manager in ("apt-get", "dnf", "yum", "apk")
            },
            # An existing launcher does not establish that all three components are installed.
            always_install=True,
        ),
    }

    def __init__(
        self,
        *args,
        wheel: str,
        base_url: str | None = None,
        max_turns: int = 100,
        model_settings: dict[str, JsonValue] | None = None,
        prompt_token_budget: PromptTokenBudget | dict[str, JsonValue] | None = None,
        compaction: dict[str, JsonValue] | None = None,
        workspace: str = ".",
        atif_mode: str = "segments",
        profile: Profile = "prod",
        enable_image_tools: bool = False,
        enable_web_tools: bool = False,
        additional_instructions: str | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.wheel = Path(wheel).resolve()
        if atif_mode not in {"segments", "grader"}:
            raise ValueError("atif_mode must be segments or grader")
        self.atif_mode = atif_mode
        self.config = LocalConfig(
            model=self.model_name,
            base_url=base_url,
            max_turns=max_turns,
            prompt_token_budget=prompt_token_budget,
            model_settings=model_settings if model_settings is not None else {},
            compaction=compaction if compaction is not None else {},
            workspace=workspace,
            mcp_servers=tuple(_harbor_mcp_spec(server) for server in self.mcp_servers),
            skills_dir=Path(self.skills_dir) if self.skills_dir else None,
            profile=profile,
            enable_image_tools=enable_image_tools,
            enable_web_tools=enable_web_tools,
            additional_instructions=additional_instructions,
            eval_prompt_output=Path(self.environment_logs_dir) / "eval-prompt.json",
        )

    @staticmethod
    def name() -> str:
        return "feather"

    def get_version_command(self) -> str:
        return shlex.join(
            [
                PYTHON,
                "-c",
                "from importlib.metadata import version; print(version('feather-harness'))",
            ]
        )

    async def install(self, environment: BaseEnvironment) -> None:
        if _WHEEL_FILENAME.match(self.wheel.name) is None:
            raise ValueError(
                f"Feather harness wheel filename must include a PEP 427 version: {self.wheel.name}"
            )
        with self.wheel.open("rb") as wheel:
            wheel_sha256 = hashlib.file_digest(wheel, "sha256").hexdigest()
        condition = self.config.model_dump(
            mode="json", include={"profile", "enable_image_tools", "enable_web_tools"}
        )
        if condition["enable_image_tools"] is None:
            condition["enable_image_tools"] = self.config.profile == "prod"
        if self.config.prompt_token_budget is not None:
            condition["prompt_token_budget"] = self.config.prompt_token_budget.model_dump(mode="json")
        condition["wheel_sha256"] = wheel_sha256
        (self.logs_dir / "harness-condition.json").write_text(json.dumps(condition, indent=2))
        manager = await self._get_system_package_manager(environment)
        if manager is None:
            raise ValueError("Feather's native eval toolkit requires apt-get, dnf, yum or apk")
        await self.ensure_system_dependencies(
            environment, ("curl", "bash", "ca_certificates", "ripgrep", "file", "libreoffice")
        )
        await self.exec_as_root(environment, command=f"mkdir -p {INSTALL_DIR}")
        remote_wheel = f"{INSTALL_DIR}/{self.wheel.name}"
        await environment.upload_file(self.wheel, remote_wheel)
        wheel_requirement = remote_wheel + ("[token-count]" if self.config.prompt_token_budget else "")
        requirements = [wheel_requirement]
        if condition["enable_image_tools"]:
            requirements.append("Pillow>=12.1.1,<13")
        # Python's managed installation must be outside /root so the task user can execute it.
        await self.exec_as_root(
            environment,
            command=(
                f"curl -LsSf https://astral.sh/uv/{UV_VERSION}/install.sh | sh && "
                f"{INSTALL_DIR}/bin/uv venv --python 3.12 {INSTALL_DIR}/venv && "
                f"{INSTALL_DIR}/bin/uv pip install --python {PYTHON} {shlex.join(requirements)}"
            ),
            env={
                "UV_INSTALL_DIR": f"{INSTALL_DIR}/bin",
                "UV_NO_MODIFY_PATH": "1",
                "UV_PYTHON_INSTALL_DIR": f"{INSTALL_DIR}/python",
            },
        )

    @with_prompt_template
    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        logs = str(self.environment_logs_dir)
        # No cwd override: Harbor's task image chooses the workspace via WORKDIR.
        with tempfile.TemporaryDirectory() as temporary:
            config = Path(temporary) / "config.json"
            config.write_text(self.config.model_dump_json())
            prompt = Path(temporary) / "instruction.txt"
            prompt.write_text(instruction)
            await environment.upload_file(config, f"{INSTALL_DIR}/config.json")
            await environment.upload_file(prompt, f"{INSTALL_DIR}/instruction.txt")
        command = shlex.join(
            [
                f"{INSTALL_DIR}/venv/bin/feather",
                "--config",
                f"{INSTALL_DIR}/config.json",
                "-",
            ]
        )
        env = {**self.extra_env, **self.resolve_env_vars()}
        if not env.get("OPENAI_API_KEY"):
            raise RuntimeError(
                "FeatherHarborAgent requires OPENAI_API_KEY in the Harbor agent env"
            )
        if self.config.enable_web_tools and not env.get("EXA_API_KEY"):
            raise RuntimeError(
                "FeatherHarborAgent enable_web_tools requires EXA_API_KEY in the Harbor agent env"
            )
        await self.exec_as_agent(
            environment,
            command=(
                f"{command} < {INSTALL_DIR}/instruction.txt "
                f"> {shlex.quote(logs + '/run.json')} 2> {shlex.quote(logs + '/stderr.txt')}"
            ),
            env=env,
        )

    def populate_context_post_run(self, context: AgentContext) -> None:
        # Harbor skips this callback if run() has already populated the context.
        self._populate_condition(context)
        output = self.logs_dir / "run.json"
        # CLI setup failures or Harbor timeouts can leave redirected output empty.
        if not output.exists() or output.stat().st_size == 0:
            return
        result = RunResult.model_validate_json(output.read_text())
        write_atif(
            result,
            self.logs_dir,
            agent=Agent(
                name=self.name(), version=version("feather-harness"), model_name=self.model_name
            ),
            session_id=self.logs_dir.parent.name,
            grader=self.atif_mode == "grader",
        )
        usage = result.usage()
        context.n_input_tokens = usage.prompt_tokens
        context.n_cache_tokens = usage.cached_tokens
        context.n_output_tokens = usage.completion_tokens
        context.cost_usd = usage.cost_usd
        context.metadata = {
            **(context.metadata or {}),
            "stop_reason": result.stop_reason,
            "num_turns": result.num_turns,
            "compactions": result.compactions,
            "recoveries": result.recoveries,
            "final_answer": result.final_answer,
        }
        if usage.model_calls:
            context.metadata["usage"] = usage.model_dump(mode="json", exclude_none=True)

    def _populate_condition(self, context: AgentContext) -> None:
        condition = self.logs_dir / "harness-condition.json"
        if condition.exists():
            recorded = json.loads(condition.read_text())
            prompt_metadata = self.logs_dir / "eval-prompt.json"
            if prompt_metadata.exists():
                recorded.update(json.loads(prompt_metadata.read_text()))
                condition.write_text(json.dumps(recorded, indent=2))
            context.metadata = {
                **(context.metadata or {}),
                "feather": recorded,
            }


_WHEEL_FILENAME = re.compile(r"^[\w.]+\-[0-9][^-]*\-(?:[^-]+\-)?[^-]+\-[^-]+\-[^-]+\.whl$")


def _harbor_mcp_spec(server) -> McpServerSpec:
    if server.transport == "stdio":
        return McpServerSpec(
            name=server.name,
            transport="stdio",
            command=server.command,
            args=tuple(server.args),
        )
    return McpServerSpec(name=server.name, transport=server.transport, url=server.url)
