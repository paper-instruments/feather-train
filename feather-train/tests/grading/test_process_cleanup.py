import asyncio
import json
from types import SimpleNamespace

import pytest

from feather_train.errors import NonRetryableHarnessError
from feather_train.grading.grader import JudgeAttemptFailure, _run_harbor_grader_attempt
from feather_train.sandbox.collector import Collector
from feather_train.sandbox.lifecycle import SandboxIO


@pytest.mark.parametrize("failure", ["timeout", "cancel", "lost-recovery"])
async def test_failed_grader_cannot_overwrite_retry_reward(sandbox, monkeypatch, failure):
    session = SandboxIO(sandbox, container=sandbox.container, host_runtime=sandbox.host_runtime)

    async def wait_ready():
        async with asyncio.timeout(5):
            while not (await session.exec("test", "-f", "/filesystem/ready")).ok():
                await asyncio.sleep(0.02)

    if failure == "lost-recovery":
        sandbox.commands.drop_before_reply = True

        async def failed_inspect(*args, **kwargs):
            await wait_ready()
            raise ConnectionError("saved result unavailable")

        monkeypatch.setattr(Collector, "inspect", failed_inspect)

    command = (
        "(touch /filesystem/ready; "
        "while [ ! -f /filesystem/release ]; do sleep 0.02; done; "
        "echo stale > /filesystem/reward) & wait"
    )
    attempt = asyncio.create_task(
        _run_harbor_grader_attempt(
            session, command, attempt=1, timeout=1 if failure == "timeout" else 10, steps=[]
        )
    )
    await wait_ready()
    if failure == "cancel":
        attempt.cancel()
        with pytest.raises(asyncio.CancelledError):
            await attempt
    else:
        assert isinstance(await attempt, JudgeAttemptFailure)

    await session.exec("echo retry > /filesystem/reward; touch /filesystem/release", shell=True)
    await asyncio.sleep(0.3)
    result = await session.exec("cat", "/filesystem/reward")
    assert result.stdout.strip() == "retry"


async def test_lost_launch_request_cannot_start_after_cleanup(sandbox, monkeypatch):
    run = sandbox.commands.run
    delayed = []

    async def lose_launch(command, **kwargs):
        if "remote.py run " in command and not delayed:
            delayed.append(command)
            raise ConnectionError("launch reply lost")
        return await run(command, **kwargs)

    async def failed_inspect(*args, **kwargs):
        raise ConnectionError("launch not yet visible")

    monkeypatch.setattr(sandbox.commands, "run", lose_launch)
    monkeypatch.setattr(Collector, "inspect", failed_inspect)
    session = SandboxIO(sandbox, container=sandbox.container, host_runtime=sandbox.host_runtime)
    result = await _run_harbor_grader_attempt(
        session, "touch /filesystem/stale", attempt=1, timeout=10, steps=[]
    )
    assert isinstance(result, JudgeAttemptFailure)
    late = await run(delayed[0])
    assert json.loads(late.stdout)["result"]["exit_code"] == 125
    assert (await session.exec("test", "!", "-f", "/filesystem/stale")).ok()


@pytest.mark.parametrize("cleanup_failure", [ConnectionError("cleanup lost"), None])
async def test_unconfirmed_cleanup_cannot_become_a_retryable_attempt(monkeypatch, cleanup_failure):
    async def failed_execute(*args, **kwargs):
        raise TimeoutError("grader timed out")

    monkeypatch.setattr(Collector, "execute", failed_execute)

    async def run(command, **kwargs):
        if cleanup_failure:
            raise cleanup_failure
        return SimpleNamespace(exit_code=1, stdout="", stderr="could not stop group")

    session = SandboxIO(SimpleNamespace(commands=SimpleNamespace(run=run)), container="verifier")
    with pytest.raises(NonRetryableHarnessError, match="Cannot confirm grader stopped"):
        await _run_harbor_grader_attempt(session, "grader", attempt=1, timeout=10, steps=[])


async def test_cancellation_waits_for_cleanup_before_propagating(monkeypatch):
    cleanup_started = asyncio.Event()
    release_cleanup = asyncio.Event()

    async def execute(*args, **kwargs):
        return {"result": {"exit_code": 0, "stdout": "", "stderr": ""}}

    async def stop(self, script, state):
        cleanup_started.set()
        await release_cleanup.wait()

    monkeypatch.setattr(Collector, "execute", execute)
    monkeypatch.setattr(SandboxIO, "_stop_grader", stop)
    session = SandboxIO(SimpleNamespace())
    attempt = asyncio.create_task(session.exec_grader("true", timeout=10))
    await cleanup_started.wait()
    attempt.cancel()
    await asyncio.sleep(0)
    assert not attempt.done()
    release_cleanup.set()
    with pytest.raises(asyncio.CancelledError):
        await attempt


@pytest.mark.parametrize("exit_code", [0, 7])
async def test_managed_grader_preserves_output_and_exit_status(sandbox, exit_code):
    session = SandboxIO(sandbox, container=sandbox.container, host_runtime=sandbox.host_runtime)
    result = await session.exec_grader(
        f"printf 'résultat'; printf 'diagnostic' >&2; exit {exit_code}", timeout=10
    )
    assert (result.exit_code, result.stdout, result.stderr) == (exit_code, "résultat", "diagnostic")
