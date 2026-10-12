from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from feather_train.errors import NonRetryableHarnessError
from feather_train.harness import _needs_agent_trajectory_artifact
from feather_train.sandbox.lifecycle import ComposeSandbox
from feather_train.tasks.harbor_spec import ArtifactSpec


@pytest.mark.parametrize(
    ("environment_mode", "artifacts", "expected"),
    [
        ("separate", ("/logs/agent/trajectory.json",), True),
        (
            "separate",
            (ArtifactSpec(source="/logs/agent/trajectory.json", service="main"),),
            True,
        ),
        (
            "separate",
            (ArtifactSpec(source="/logs/agent/trajectory.json", service="world"),),
            False,
        ),
        ("separate", ("/logs/world/final_snapshot.tar.gz",), False),
        ("shared", ("/logs/agent/trajectory.json",), False),
    ],
)
def test_only_separate_agent_trajectory_artifact_needs_preisolation_staging(
    environment_mode, artifacts, expected
):
    task = SimpleNamespace(environment_mode=environment_mode, artifacts=artifacts)
    assert _needs_agent_trajectory_artifact(task) is expected


@pytest.mark.parametrize("exit_code", [0, 1])
async def test_stages_trace_without_command_line_payload_and_cleans_up(
    monkeypatch, exit_code
):
    written = AsyncMock()
    monkeypatch.setattr("feather_train.sandbox.lifecycle.write_file", written)
    run = AsyncMock(
        return_value=SimpleNamespace(exit_code=exit_code, stdout="", stderr="")
    )
    remove = AsyncMock()
    sandbox = SimpleNamespace(
        commands=SimpleNamespace(run=run),
        files=SimpleNamespace(remove=remove),
    )
    world = ComposeSandbox(sandbox, runtime=None)
    payload = b'{"sensitive":"trace body"}\n'

    if exit_code:
        with pytest.raises(NonRetryableHarnessError, match="trajectory artifact"):
            await world.stage_agent_trace(payload)
    else:
        await world.stage_agent_trace(payload)

    written.assert_awaited_once()
    assert written.await_args.args[0] is sandbox
    upload_path = written.await_args.args[1]
    assert upload_path.startswith("/tmp/feather-agent-trace-")
    assert written.await_args.args[2] == payload
    command = run.await_args.args[0]
    assert "docker exec -u root agent sh -c" in command
    assert "docker cp" in command
    assert upload_path in command
    assert "trace body" not in command
    remove.assert_awaited_once_with(upload_path, user="root", request_timeout=120)
