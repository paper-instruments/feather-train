import asyncio
import hashlib
import json
import os
import shlex
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from feather_harness import (
    AssistantMessage,
    McpServerSpec,
    ProviderUsage,
    RunResult,
    ToolCall,
    ToolMessage,
    TraceSegment,
    UserMessage,
)
from harbor.agents.installed.base import NonZeroAgentExitCodeError
from harbor.environments.base import ExecResult
from harbor.models.agent.context import AgentContext
from harbor.models.task.config import MCPServerConfig
from harbor.trial.trial import Trial
from pydantic import ValidationError

from feather_eval.harbor import FeatherHarborAgent


class LocalEnvironment:
    """Execute the agent locally; stand in for Linux root-level package installation."""

    def __init__(self, root: Path):
        self.install_dir = root / "installed-agent" / "feather"
        self.workspace = root / "workspace"
        self.workspace.mkdir()

    async def upload_file(self, source, target):
        target = Path(target.replace("/installed-agent/feather", str(self.install_dir)))
        shutil.copyfile(source, target)

    async def exec(self, command, *, env=None, cwd=None, user=None, timeout_sec=None):
        if user == "root":
            return ExecResult(
                return_code=0, stdout="apt-get" if "for manager" in command else ""
            )
        command = command.replace("/installed-agent/feather", str(self.install_dir))
        process = await asyncio.create_subprocess_exec(
            "bash",
            "-c",
            command,
            cwd=cwd or self.workspace,
            env={**os.environ, **(env or {})},
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        return ExecResult(
            return_code=process.returncode, stdout=stdout.decode(), stderr=stderr.decode()
        )


@pytest.mark.parametrize(
    "condition",
    [
        {},
        {"profile": "prod", "enable_image_tools": True},
        {"profile": "train", "additional_instructions": "Complete the benchmark task autonomously."},
        {"profile": "train", "enable_image_tools": True},
        {"profile": "prod", "enable_web_tools": True},
        {
            "profile": "prod",
            "max_turns": 250,
            "model_settings": {"temperature": 1.0, "top_p": 0.95, "max_tokens": 131072},
            "compaction": {
                "enabled": True,
                "context_window_tokens": 131072,
                "min_turns_before_compaction": 0,
                "keep_recent_turns": 2,
            },
        },
    ],
)
async def test_cli_receives_literal_instruction_and_records_its_condition(tmp_path, condition):
    environment = LocalEnvironment(tmp_path)
    executable = environment.install_dir / "venv" / "bin" / "feather"
    executable.parent.mkdir(parents=True)
    executable.write_text(
        f"#!{sys.executable}\n"
        + """
import argparse
import json
import os
import sys
from pathlib import Path
from feather_harness import AssistantMessage, RunResult, TraceSegment, UserMessage

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=Path, required=True)
parser.add_argument('prompt', choices=['-'])
args = parser.parse_args()
instruction = sys.stdin.read()
Path('received.json').write_text(json.dumps({
    'instruction': instruction,
    'config': json.loads(args.config.read_text()),
    'api_key': os.environ['OPENAI_API_KEY'],
    'exa_api_key': os.environ.get('EXA_API_KEY'),
    'path': os.environ['PATH'],
    'workspace': os.getcwd(),
}))
print(RunResult(
    stop_reason='completed', num_turns=1, compactions=0, final_answer='Final',
    segments=(TraceSegment(
        kind='execution', input=(UserMessage(content=instruction),),
        messages=(AssistantMessage(content='Final'),),
    ),),
).model_dump_json())
print('probe diagnostic', file=sys.stderr)
"""
    )
    executable.chmod(0o755)
    logs = tmp_path / "agent logs"
    logs.mkdir()
    wheel = tmp_path / "feather_harness-0.1.0-py3-none-any.whl"
    wheel.write_bytes(b"selected wheel")
    agent = FeatherHarborAgent(
        logs_dir=logs,
        environment_logs_dir=logs,
        model_name="provider/model",
        wheel=str(wheel),
        base_url="http://endpoint/v1",
        workspace="/filesystem",
        extra_env={"OPENAI_API_KEY": "private-api-key", "EXA_API_KEY": "private-exa-key"},
        mcp_servers=[
            MCPServerConfig(
                name="local", transport="stdio", command="server", args=["--token=private-mcp-key"]
            )
        ],
        **condition,
    )
    await agent.install(environment)
    installed_wheel = environment.install_dir / wheel.name
    wheel.write_bytes(b"rebuilt later")

    instruction = "Quotes ' \" and $(touch unwanted) café\n" + "x" * 140_000 + "\n\n"
    context = AgentContext()
    await agent.run(instruction, environment, context)
    Trial._populate_agent_context(SimpleNamespace(user_agent=None, agent=agent), context)

    received = json.loads((environment.workspace / "received.json").read_text())
    assert received["instruction"] == instruction
    assert received["config"]["additional_instructions"] == condition.get("additional_instructions")
    assert received["config"]["eval_prompt_output"] == str(logs / "eval-prompt.json")
    assert received["config"]["model"] == "provider/model"
    assert received["config"]["base_url"] == "http://endpoint/v1"
    assert received["config"]["workspace"] == "/filesystem"
    assert received["config"]["profile"] == condition.get("profile", "prod")
    settings = received["config"]["model_settings"]
    assert settings["temperature"] == condition.get("model_settings", {}).get("temperature")
    assert settings["top_p"] == condition.get("model_settings", {}).get("top_p")
    assert settings["max_tokens"] == condition.get("model_settings", {}).get("max_tokens")
    assert received["config"]["max_turns"] == condition.get("max_turns", 100)
    compaction = received["config"]["compaction"]
    assert compaction["enabled"] == condition.get("compaction", {}).get("enabled")
    for key, value in condition.get("compaction", {}).items():
        assert compaction[key] == value
    assert received["config"]["enable_image_tools"] == condition.get("enable_image_tools", False)
    assert received["config"]["enable_web_tools"] == condition.get("enable_web_tools", False)
    assert received["api_key"] == "private-api-key"
    assert received["exa_api_key"] == "private-exa-key"
    assert received["path"] == os.environ["PATH"]
    assert Path(received["workspace"]) == environment.workspace
    assert not (environment.workspace / "unwanted").exists()
    assert (agent.logs_dir / "stderr.txt").read_text() == "probe diagnostic\n"
    recorded = json.loads((logs / "harness-condition.json").read_text())
    expected = {
        "profile": received["config"]["profile"],
        "enable_image_tools": received["config"]["enable_image_tools"],
        "enable_web_tools": received["config"]["enable_web_tools"],
        "wheel_sha256": hashlib.sha256(installed_wheel.read_bytes()).hexdigest(),
    }
    assert recorded == expected
    assert context.metadata == {
        "feather": recorded,
        "stop_reason": "completed",
        "num_turns": 1,
        "compactions": 0,
        "recoveries": 0,
        "final_answer": "Final",
    }
    trajectory = json.loads((logs / "trajectory.json").read_text())
    assert trajectory["steps"][-1]["message"] == "Final"


async def test_nonzero_cli_exit_fails_the_harbor_run(tmp_path):
    environment = LocalEnvironment(tmp_path)
    executable = environment.install_dir / "venv" / "bin" / "feather"
    executable.parent.mkdir(parents=True)
    executable.write_text('#!/bin/bash\nprintf "provider failed\\n" >&2\nexit 7\n')
    executable.chmod(0o755)
    logs = tmp_path / "agent logs"
    logs.mkdir()
    wheel = tmp_path / "feather_harness-0.1.0-py3-none-any.whl"
    wheel.write_bytes(b"selected wheel")
    agent = FeatherHarborAgent(
        logs_dir=logs,
        environment_logs_dir=logs,
        model_name="test",
        wheel=str(wheel),
        extra_env={"OPENAI_API_KEY": "test"},
    )
    await agent.install(environment)

    context = AgentContext()
    with pytest.raises(NonZeroAgentExitCodeError):
        await agent.run("task", environment, context)

    assert (agent.logs_dir / "stderr.txt").read_text() == "provider failed\n"
    Trial._populate_agent_context(SimpleNamespace(user_agent=None, agent=agent), context)
    assert context.metadata == {
        "feather": json.loads((logs / "harness-condition.json").read_text())
    }


async def test_unicode_task_text_survives_a_windows_default_encoding(tmp_path, monkeypatch):
    environment = LocalEnvironment(tmp_path)
    executable = environment.install_dir / "venv" / "bin" / "feather"
    executable.parent.mkdir(parents=True)
    executable.write_text(
        f"#!{sys.executable}\n"
        + """
import json, sys
from feather_harness import AssistantMessage, RunResult, TraceSegment, UserMessage

config = json.loads(open(sys.argv[2], encoding='utf-8').read())
answer = sys.stdin.read() + config['additional_instructions']
print(RunResult(
    stop_reason='completed', num_turns=1, compactions=0, final_answer=answer,
    segments=(TraceSegment(
        kind='execution', input=(UserMessage(content='task'),),
        messages=(AssistantMessage(content=answer),),
    ),),
).model_dump_json())
""",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    original_read_text = Path.read_text
    original_write_text = Path.write_text

    def windows_read_text(self, encoding=None, errors=None):
        return original_read_text(self, encoding or "cp1252", errors)

    def windows_write_text(self, data, encoding=None, errors=None, newline=None):
        return original_write_text(self, data, encoding or "cp1252", errors, newline)

    monkeypatch.setattr(Path, "read_text", windows_read_text)
    monkeypatch.setattr(Path, "write_text", windows_write_text)
    logs = tmp_path / "agent logs"
    logs.mkdir()
    wheel = tmp_path / "feather_harness-0.1.0-py3-none-any.whl"
    wheel.write_bytes(b"selected wheel")
    agent = FeatherHarborAgent(
        logs_dir=logs,
        environment_logs_dir=logs,
        model_name="test",
        wheel=str(wheel),
        extra_env={"OPENAI_API_KEY": "test"},
        additional_instructions=" Résumé – 工作",
    )
    await agent.install(environment)

    context = AgentContext()
    await agent.run("Rédigez le rapport — 报告", environment, context)
    Trial._populate_agent_context(SimpleNamespace(user_agent=None, agent=agent), context)

    answer = "Rédigez le rapport — 报告 Résumé – 工作"
    assert context.metadata["final_answer"] == answer
    trajectory = json.loads((logs / "trajectory.json").read_bytes().decode("utf-8"))
    assert trajectory["steps"][-1]["message"] == answer


async def test_web_tools_without_exa_key_fail_before_feather_starts(tmp_path):
    environment = LocalEnvironment(tmp_path)
    logs = tmp_path / "agent logs"
    logs.mkdir()
    agent = FeatherHarborAgent(
        logs_dir=logs,
        environment_logs_dir=logs,
        model_name="test",
        wheel=str(tmp_path / "feather_harness-0.1.0-py3-none-any.whl"),
        extra_env={"OPENAI_API_KEY": "test"},
        enable_web_tools=True,
    )
    environment.install_dir.mkdir(parents=True)

    with pytest.raises(RuntimeError, match="EXA_API_KEY"):
        await agent.run("task", environment, AgentContext())

    assert not (logs / "run.json").exists()


@pytest.mark.parametrize("mode,root_answer", [("segments", ""), ("grader", "Final")])
def test_context_reports_outcome_and_selects_the_requested_atif_view(tmp_path, mode, root_answer):
    agent = FeatherHarborAgent(
        logs_dir=tmp_path, model_name="test", wheel=str(tmp_path / "feather.whl"), atif_mode=mode
    )
    result = RunResult(
        stop_reason="completed",
        num_turns=2,
        compactions=1,
        recoveries=1,
        final_answer="Final",
        segments=(
            TraceSegment(
                kind="execution",
                input=(UserMessage(content="task"),),
                messages=(
                    AssistantMessage(
                        tool_calls=(ToolCall(id="read", name="read", arguments="{}"),)
                    ),
                    ToolMessage(tool_call_id="read", content="file contents"),
                ),
                calls=(_billed("agent", reasoning=900),),
            ),
            TraceSegment(
                kind="compaction",
                input=(UserMessage(content="Summarize"),),
                messages=(AssistantMessage(content="summary"),),
                calls=(_billed("compaction"),),
            ),
            TraceSegment(
                kind="execution",
                input=(UserMessage(content="summary"),),
                messages=(AssistantMessage(content="Final"),),
                calls=(_billed("agent", reasoning=100),),
            ),
        ),
    )
    (tmp_path / "run.json").write_text(result.model_dump_json())
    context = AgentContext()
    Trial._populate_agent_context(SimpleNamespace(user_agent=None, agent=agent), context)
    root = json.loads((tmp_path / "trajectory.json").read_text())
    assert root["steps"][-1]["message"] == root_answer
    assert ("continued_trajectory_ref" in root) == (mode == "segments")
    if mode == "segments":
        # The Braintrust Harbor plugin makes LLM spans only from single-call steps with usage.
        produced = [step for step in root["steps"] if step["source"] == "agent"]
        assert [(step["llm_call_count"], step["metrics"]["extra"]["reasoning_tokens"])
                for step in produced] == [(1, 900)]
        assert root["final_metrics"]["total_completion_tokens"] == 3000
        continued = json.loads((tmp_path / root["continued_trajectory_ref"]).read_text())
        assert [step.get("metrics", {}).get("cost_usd") for step in continued["steps"]] == [
            None,
            pytest.approx(0.11),
        ]
    # Summary calls are billed too, so totals cover all three calls in either ATIF view.
    assert (context.n_input_tokens, context.n_cache_tokens, context.n_output_tokens) == (
        3000,
        1500,
        3000,
    )
    assert context.cost_usd == pytest.approx(0.33)
    usage = context.metadata.pop("usage")
    assert (usage["reasoning_tokens"], usage["model_calls"], usage["byok_calls"]) == (1000, 3, 3)
    assert context.metadata == {
        "stop_reason": "completed",
        "num_turns": 2,
        "compactions": 1,
        "recoveries": 1,
        "final_answer": "Final",
    }


def _billed(kind, *, reasoning=None):
    usage = {
        "prompt_tokens": 1000,
        "completion_tokens": 1000,
        "prompt_tokens_details": {"cached_tokens": 500},
        "cost": 0.01,
        "is_byok": True,
        "cost_details": {"upstream_inference_cost": 0.1},
    }
    if reasoning is not None:
        usage["completion_tokens_details"] = {"reasoning_tokens": reasoning}
    return ProviderUsage(kind=kind, usage=usage)


def test_missing_result_does_not_mask_an_install_failure(tmp_path):
    agent = FeatherHarborAgent(
        logs_dir=tmp_path, model_name="test", wheel=str(tmp_path / "feather.whl")
    )
    context = AgentContext()
    agent.populate_context_post_run(context)
    assert context.is_empty()


def test_adapter_maps_harbor_stdio_http_and_skills_dir(tmp_path):
    agent = FeatherHarborAgent(
        logs_dir=tmp_path,
        model_name="test",
        wheel=str(tmp_path / "feather.whl"),
        mcp_servers=[
            MCPServerConfig(name="mail", transport="streamable-http", url="http://mail/mcp/"),
            MCPServerConfig(
                name="local",
                transport="stdio",
                command="/usr/bin/mcp-server",
                args=["--stdio"],
            ),
        ],
        skills_dir="/harbor/skills",
    )
    assert agent.config.mcp_servers == (
        McpServerSpec(name="mail", transport="streamable-http", url="http://mail/mcp/"),
        McpServerSpec(
            name="local",
            transport="stdio",
            command="/usr/bin/mcp-server",
            args=("--stdio",),
        ),
    )
    assert agent.config.skills_dir == Path("/harbor/skills")


def test_resolved_symlink_keeps_the_versioned_wheel_filename(tmp_path) -> None:
    wheel = tmp_path / "feather_harness-0.1.0-py3-none-any.whl"
    wheel.write_bytes(b"x")
    alias = tmp_path / "feather_harness.whl"
    alias.symlink_to(wheel)
    agent = FeatherHarborAgent(logs_dir=tmp_path, model_name="test", wheel=str(alias))
    assert agent.wheel.name == "feather_harness-0.1.0-py3-none-any.whl"


async def test_install_rejects_a_versionless_wheel_filename(tmp_path) -> None:
    wheel = tmp_path / "feather_harness.whl"
    wheel.write_bytes(b"x")
    agent = FeatherHarborAgent(logs_dir=tmp_path, model_name="test", wheel=str(wheel))
    with pytest.raises(ValueError, match="PEP 427"):
        await agent.install(LocalEnvironment(tmp_path))


@pytest.mark.parametrize(
    "settings",
    [
        {"model_settings": {"temperature": 1.0, "topp": 0.95}},
        {"compaction": {"enabled": True, "context_window_token": 131072}},
    ],
)
def test_misspelled_setting_fails_before_install(tmp_path, settings):
    with pytest.raises(ValidationError, match="extra_forbidden"):
        FeatherHarborAgent(
            logs_dir=tmp_path,
            model_name="z-ai/glm-5.3-flash",
            wheel=str(tmp_path / "feather_harness-0.1.0-py3-none-any.whl"),
            **settings,
        )


def test_effective_prompt_hash_is_retained_without_a_successful_result(tmp_path):
    agent = FeatherHarborAgent(
        logs_dir=tmp_path, model_name="test", wheel=str(tmp_path / "absent.whl")
    )
    (tmp_path / "harness-condition.json").write_text(json.dumps({"wheel_sha256": "c" * 64}))
    metadata = {
        "profile": "train",
        "system_prompt_sha256": "d" * 64,
    }
    (tmp_path / "eval-prompt.json").write_text(json.dumps(metadata))
    context = AgentContext()
    agent.populate_context_post_run(context)
    expected = {"wheel_sha256": "c" * 64, **metadata}
    assert context.metadata["feather"] == expected
    assert json.loads((tmp_path / "harness-condition.json").read_text()) == expected


async def test_budget_installs_optional_counter_and_records_pinned_revision(tmp_path):
    budget = {
        "processor_name": "zai-org/GLM-5.3-Flash",
        "processor_revision": "eb9eb208eb0d988989d07a6a12d0fdeb5f52574a",
        "context_window_tokens": 131072,
    }
    wheel = tmp_path / "feather_harness-0.1.0-py3-none-any.whl"
    wheel.write_bytes(b"wheel")
    commands = []

    class Environment(LocalEnvironment):
        async def exec(self, command, **kwargs):
            commands.append(command)
            return await super().exec(command, **kwargs)

    agent = FeatherHarborAgent(
        logs_dir=tmp_path, model_name="glm", wheel=str(wheel), prompt_token_budget=budget
    )
    environment = Environment(tmp_path)
    environment.install_dir.mkdir(parents=True)
    await agent.install(environment)
    assert any(".whl[token-count]" in command for command in commands)
    assert agent.config.prompt_token_budget.model_dump(mode="json") == budget
    condition = json.loads((tmp_path / "harness-condition.json").read_text())
    assert condition["prompt_token_budget"] == budget


@pytest.mark.parametrize("enable_image_tools", [False, True])
@pytest.mark.parametrize("manager,install", [
    ("apt-get", "apt-get update && apt-get install -y "),
    ("dnf", "dnf install -y "),
    ("yum", "yum install -y "),
    ("apk", "apk add --no-cache "),
])
async def test_installer_supplies_office_binaries_and_image_dependency_without_changing_task_python(
    tmp_path, enable_image_tools, manager, install
):
    wheel = tmp_path / "feather_harness-0.1.0-py3-none-any.whl"
    wheel.write_bytes(b"wheel")
    commands = []

    class Environment(LocalEnvironment):
        async def exec(self, command, **kwargs):
            commands.append((command, kwargs))
            if "for manager" in command:
                return ExecResult(return_code=0, stdout=manager, stderr="")
            return await super().exec(command, **kwargs)

    agent = FeatherHarborAgent(
        logs_dir=tmp_path, model_name="glm", wheel=str(wheel),
        enable_image_tools=enable_image_tools,
    )
    environment = Environment(tmp_path)
    environment.install_dir.mkdir(parents=True)
    await agent.install(environment)

    system_command, system_kwargs = next(
        (command, kwargs) for command, kwargs in commands if install in command
    )
    packages = shlex.split(system_command.split(install, 1)[1])
    assert {"file", "libreoffice-calc", "libreoffice-writer", "libreoffice-impress"} <= set(packages)
    assert system_kwargs["user"] == "root"
    assert system_kwargs["env"] == ({"DEBIAN_FRONTEND": "noninteractive"} if manager == "apt-get" else None)
    pip_command, pip_kwargs = next(
        (command, kwargs) for command, kwargs in commands if "uv pip install" in command
    )
    arguments = shlex.split(pip_command.split("uv pip install ", 1)[1])
    assert arguments[:2] == ["--python", "/installed-agent/feather/venv/bin/python"]
    assert arguments[2:] == [f"/installed-agent/feather/{wheel.name}"] + (
        ["Pillow>=12.1.1,<13"] if enable_image_tools else []
    )
    assert "PATH" not in pip_kwargs["env"]
    assert not any("paper-" in command or "pip install --python python" in command for command, _ in commands)


async def test_missing_package_manager_fails_before_installation(tmp_path):
    wheel = tmp_path / "feather_harness-0.1.0-py3-none-any.whl"
    wheel.write_bytes(b"wheel")
    commands = []

    class Environment(LocalEnvironment):
        async def exec(self, command, **kwargs):
            commands.append(command)
            return ExecResult(return_code=0, stdout="", stderr="")

        async def upload_file(self, source, target):
            pytest.fail("An unsupported sandbox must fail before uploading the harness")

    agent = FeatherHarborAgent(logs_dir=tmp_path, model_name="glm", wheel=str(wheel))
    with pytest.raises(ValueError, match="requires apt-get, dnf, yum or apk"):
        await agent.install(Environment(tmp_path))
    assert len(commands) == 1
