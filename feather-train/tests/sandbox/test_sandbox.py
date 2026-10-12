import asyncio
import base64
import io
import tarfile
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from e2b import AsyncSandbox, CommandExitException
from e2b.exceptions import InvalidArgumentException

from feather_train.errors import NonRetryableHarnessError
from feather_train.sandbox.io import read_remote_bytes
from feather_train.sandbox.lifecycle import ComposeSandbox, SandboxIO


async def test_container_io_transfers_binary_files_and_preserves_exit_status(sandbox):
    files = SandboxIO(sandbox, container=sandbox.container)
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        entry = tarfile.TarInfo("résumé.bin")
        entry.mode, entry.size = 0o600, 7
        archive.addfile(entry, io.BytesIO(b"\x00hello\xff"))
    result = await files.extract_archive(buffer.getvalue(), "/filesystem")
    assert result.ok()
    result = await files.exec(
        "python3",
        "-c",
        "from pathlib import Path; p=Path('/filesystem/résumé.bin'); print(p.read_bytes().hex()); print(oct(p.stat().st_mode & 0o777))",
    )
    assert result.stdout.splitlines() == ["0068656c6c6fff", "0o600"]
    result = await files.extract_archive(b"not a tar", "/filesystem")
    assert not result.ok()
    result = await files.exec("printf out; printf err >&2; exit 7", shell=True)
    assert (result.exit_code, result.stdout, result.stderr) == (7, "out", "err")
    assert not result.ok()


async def test_archive_cleanup_failure_preserves_successful_extraction(caplog):
    sandbox = SimpleNamespace(
        files=SimpleNamespace(
            write=AsyncMock(), remove=AsyncMock(side_effect=ConnectionError("cleanup failed"))
        ),
        commands=SimpleNamespace(
            run=AsyncMock(
                return_value=SimpleNamespace(exit_code=0, stdout="", stderr="tar diagnostic")
            )
        ),
    )

    result = await SandboxIO(sandbox, container="agent").extract_archive(b"archive", "/filesystem")

    assert result.ok()
    assert result.stderr == "tar diagnostic"
    assert "cleanup failed" in caplog.text


async def test_archive_cleanup_failure_preserves_extraction_exception(caplog):
    failure = InvalidArgumentException("extraction rejected")
    sandbox = SimpleNamespace(
        files=SimpleNamespace(
            write=AsyncMock(), remove=AsyncMock(side_effect=ConnectionError("cleanup failed"))
        ),
        commands=SimpleNamespace(run=AsyncMock(side_effect=failure)),
    )

    with pytest.raises(NonRetryableHarnessError, match="extraction rejected") as raised:
        await SandboxIO(sandbox, container="agent").extract_archive(b"archive", "/filesystem")

    assert raised.value.__cause__ is failure
    assert "cleanup failed" in caplog.text


@pytest.mark.parametrize("exit_code", [0, 7])
async def test_command_io_preserves_provider_text_and_exit_status(exit_code):
    stdout, stderr = "résultat ✅\n", "diagnostic échec\n"
    run = AsyncMock(return_value=SimpleNamespace(exit_code=0, stdout=stdout, stderr=stderr))
    if exit_code:
        run.side_effect = CommandExitException(stderr, stdout, exit_code, None)
    session = SandboxIO(SimpleNamespace(commands=SimpleNamespace(run=run)))

    result = await session.exec("command")

    assert (result.exit_code, result.stdout, result.stderr) == (exit_code, stdout, stderr)
    assert result.ok() is (exit_code == 0)
    run.assert_awaited_once()


async def test_remote_file_read_preserves_arbitrary_bytes():
    content = b"\x00\xff\x80" + "café\n".encode()
    run = AsyncMock(
        return_value=SimpleNamespace(
            exit_code=0,
            stdout=base64.b64encode(content).decode("ascii"),
            stderr="",
        )
    )
    session = SandboxIO(SimpleNamespace(commands=SimpleNamespace(run=run)))

    assert await read_remote_bytes(session, "/task/résultat.bin") == content
    run.assert_awaited_once()


async def test_failed_file_read_preserves_text_diagnostic_without_retry():
    from verifiers.v1 import SandboxError

    run = AsyncMock(side_effect=CommandExitException("fichier introuvable — échec\n", "", 1, None))
    session = SandboxIO(SimpleNamespace(commands=SimpleNamespace(run=run)))

    with pytest.raises(SandboxError) as raised:
        await read_remote_bytes(session, "/task/résultat.bin")

    assert (
        str(raised.value)
        == "read_remote_bytes /task/résultat.bin failed: fichier introuvable — échec\n"
    )
    run.assert_awaited_once()


async def test_native_creation_is_destroyed_when_resume_fails(monkeypatch):
    pytest.importorskip("verifiers.v1")
    created = []
    killed = []

    class Commands:
        async def run(self, *args, **kwargs):
            raise ConnectionError("resume failed")

    async def kill(**kwargs):
        killed.append(True)

    async def create(**kwargs):
        created.append(kwargs)
        return SimpleNamespace(commands=Commands(), kill=kill)

    # Substitute the external provider boundary, not any lifecycle internals.
    monkeypatch.setattr(AsyncSandbox, "create", create)
    task = SimpleNamespace(e2b_snapshot_id="baked-world", network_mode="no-network", mcp_servers=[], environment_mode=None)
    from verifiers.v1 import SandboxError

    with pytest.raises(SandboxError, match="resume failed"):
        async with ComposeSandbox.start(task, timeout=7200, max_attempts=1):
            pytest.fail("broken resume must not reach agent execution")
    assert created == [
        {
            "template": "baked-world",
            "timeout": 7200,
            "allow_internet_access": False,
            "request_timeout": 120,
        }
    ]
    assert killed == [True]


async def test_native_creation_is_destroyed_on_cancellation(monkeypatch):
    pytest.importorskip("verifiers.v1")
    started = asyncio.Event()
    killed = asyncio.Event()

    class Commands:
        async def run(self, *args, **kwargs):
            started.set()
            await asyncio.Event().wait()

    async def kill(**kwargs):
        killed.set()

    async def create(**kwargs):
        return SimpleNamespace(commands=Commands(), kill=kill)

    monkeypatch.setattr(AsyncSandbox, "create", create)
    task = SimpleNamespace(e2b_snapshot_id="baked-world", network_mode="public", mcp_servers=[], environment_mode=None)

    async def start():
        async with ComposeSandbox.start(task, max_attempts=1):
            pytest.fail("cancelled resume must not reach agent execution")

    pending = asyncio.create_task(start())
    await started.wait()
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    assert killed.is_set()


async def test_failed_teardown_does_not_replace_resume_failure(monkeypatch, caplog):
    vf = pytest.importorskip("verifiers.v1")

    class Commands:
        async def run(self, *args, **kwargs):
            raise ConnectionError("original resume failure")

    async def kill(**kwargs):
        raise ConnectionError("cleanup failure")

    async def create(**kwargs):
        return SimpleNamespace(commands=Commands(), kill=kill, sandbox_id="test-sandbox")

    monkeypatch.setattr(AsyncSandbox, "create", create)
    task = SimpleNamespace(e2b_snapshot_id="baked-world", network_mode="public", mcp_servers=[], environment_mode=None)
    with pytest.raises(vf.SandboxError, match="original resume failure"):
        async with ComposeSandbox.start(task, max_attempts=1):
            pytest.fail("broken resume must not reach agent execution")
    assert "cleanup failure" in caplog.text


async def test_startup_cleans_failed_sandbox_and_does_not_retry_rollout(monkeypatch):
    pytest.importorskip("verifiers.v1")
    events = []

    class Commands:
        def __init__(self, number):
            self.number = number

        async def run(self, command, **kwargs):
            if self.number == 1:
                raise ConnectionError("first resume failed")
            if command == "cat /opt/apex_world/docker-compose.yaml":
                output = "services:\n  main:\n    build: {context: .}\n    networks: [default]\nnetworks:\n  default: {}\n"
            elif command == "cat /opt/apex_world/compose.train.yaml":
                output = "services:\n  main:\n    container_name: agent\n"
            elif "docker info" in command:
                output = "STACK_LIVE"
            elif "docker inspect" in command and "verifier" in command:
                output = "true 0 0"
            else:
                raise AssertionError(f"Unexpected command: {command}")
            return SimpleNamespace(exit_code=0, stdout=output, stderr="")

    async def create(**kwargs):
        number = sum(event.startswith("create") for event in events) + 1
        events.append(f"create-{number}")

        async def kill(**kwargs):
            events.append(f"kill-{number}")

        return SimpleNamespace(commands=Commands(number), kill=kill, sandbox_id=str(number))

    monkeypatch.setattr(AsyncSandbox, "create", create)
    task = SimpleNamespace(
        e2b_snapshot_id="baked-world",
        network_mode="public",
        mcp_servers=[],
        environment_mode=None,
        task_artifacts=SimpleNamespace(inputs=False),
    )
    with pytest.raises(RuntimeError, match="agent failed"):
        async with ComposeSandbox.start(task):
            events.append("agent")
            raise RuntimeError("agent failed")
    assert events == ["create-1", "kill-1", "create-2", "agent", "kill-2"]


@pytest.mark.parametrize("permanent", [False, True], ids=["connection", "missing-revision"])
async def test_input_download_distinguishes_configuration_from_connection_errors(
    monkeypatch, permanent
):
    import httpx
    import huggingface_hub
    from huggingface_hub.errors import RevisionNotFoundError

    from feather_train.errors import NonRetryableHarnessError
    from feather_train.sandbox.lifecycle import stage_task_inputs

    error = (
        RevisionNotFoundError(
            "unknown revision",
            response=httpx.Response(
                404,
                request=httpx.Request("GET", "https://example.invalid/tasks"),
            ),
        )
        if permanent
        else ConnectionError("disconnected")
    )

    def download(*args, **kwargs):
        raise error

    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    task = SimpleNamespace(
        task_id="task",
        hf_dataset="unused/tasks",
        hf_revision="a" * 40,
        task_artifacts=SimpleNamespace(inputs=True),
    )
    with pytest.raises(NonRetryableHarnessError if permanent else ConnectionError):
        await stage_task_inputs(None, task)
