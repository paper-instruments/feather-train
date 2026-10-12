"""E2B sandbox lifecycle and infrastructure I/O."""

from __future__ import annotations

import asyncio
import io
import logging
import shlex
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import httpx
from e2b.exceptions import SandboxException

if TYPE_CHECKING:
    from feather_train.sandbox.compose import ComposeRuntime
    from feather_train.tasks.taskset import FeatherTaskData

from e2b import AsyncSandbox, CommandExitException
from tenacity import AsyncRetrying, retry_if_exception, stop_after_attempt, wait_exponential

from feather_train.sandbox.paths import WORKSPACE_DIR
from feather_train.sandbox.retries import PERMANENT_E2B_ERRORS, REQUEST_TIMEOUT, write_file

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CommandOutput:
    exit_code: int
    stdout: str
    stderr: str

    def ok(self) -> bool:
        return self.exit_code == 0


@dataclass(frozen=True)
class SandboxIO:
    """Host or verifier-container I/O for setup and grading."""

    sandbox: AsyncSandbox
    container: str | None = None
    host_runtime: str = "/opt/feather-host"

    async def exec(
        self, *args: str, shell: bool = False, timeout: float | None = None
    ) -> CommandOutput:
        command = self._command(args, shell=shell)
        try:
            result = await self.sandbox.commands.run(
                command,
                user="root",
                timeout=timeout if timeout is not None else 300,
                request_timeout=REQUEST_TIMEOUT,
            )
        except CommandExitException as exc:
            result = exc
        except PERMANENT_E2B_ERRORS as exc:
            from feather_train.errors import NonRetryableHarnessError

            raise NonRetryableHarnessError(str(exc)) from exc
        except (SandboxException, ConnectionError, TimeoutError, httpx.TransportError) as exc:
            from verifiers.v1 import SandboxError

            raise SandboxError(f"E2B command failed: {exc}") from exc
        return CommandOutput(result.exit_code, result.stdout, result.stderr)

    async def exec_grader(self, command: str, *, timeout: float) -> CommandOutput:
        """Recover a grader's reply, then stop its group before returning or retrying."""
        from verifiers.v1 import SandboxError

        from feather_train.errors import NonRetryableHarnessError
        from feather_train.sandbox.collector import Collector

        script = Path(__file__).with_name("grader_process.py").read_text()
        token = uuid.uuid4().hex
        state = f"/tmp/feather-grader-{token}"
        collector = Collector(self.sandbox, "grader", self.host_runtime)
        request = {
            "capture": True,
            "docker": [
                "/bin/bash",
                "-lc",
                self._command(("python3", "-c", script, "run", state, command)),
            ],
        }
        try:
            deadline = asyncio.get_running_loop().time() + timeout
            async with asyncio.timeout_at(deadline):
                launch = await collector.prepare_launch(token, request)
            result = await collector.execute(
                token, False, deadline, launch, wait_for_launch_on_cancel=False
            )
            if "failure" in result:
                raise SandboxError(result["failure"])
            return CommandOutput(**result["result"])
        except PERMANENT_E2B_ERRORS as exc:
            raise NonRetryableHarnessError(str(exc)) from exc
        except Exception as exc:
            raise SandboxError(f"E2B grader command failed: {type(exc).__name__}: {exc}") from exc
        finally:
            cleanup = asyncio.create_task(self._stop_grader(script, state))
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError:
                await cleanup
                raise

    async def extract_archive(
        self,
        archive: bytes,
        destination: str,
        *,
        guard_paths: tuple[str, ...] = (),
        timeout: float = 60,
    ) -> CommandOutput:
        upload = f"/tmp/feather-upload-{uuid.uuid4().hex}.tar"
        await write_file(self.sandbox, upload, archive)
        try:
            checks = [f"test ! -L {shlex.quote(path)}" for path in guard_paths]
            command = " && ".join([*checks, f"tar -xf - -C {shlex.quote(destination)}"])
            if self.container is not None:
                command = shlex.join(
                    ["docker", "exec", "-i", "-u", "root", self.container, "sh", "-c", command]
                )
            else:
                command = shlex.join(["sh", "-c", command])
            return await SandboxIO(self.sandbox).exec(
                command + " < " + shlex.quote(upload),
                shell=True,
                timeout=timeout,
            )
        finally:
            try:
                await self.sandbox.files.remove(
                    upload, user="root", request_timeout=REQUEST_TIMEOUT
                )
            except Exception:
                logger.warning("Could not remove temporary archive %s", upload, exc_info=True)

    def _command(self, args: tuple[str, ...], *, shell: bool = False) -> str:
        command = args[0] if shell else shlex.join(args)
        if self.container is not None:
            command = shlex.join(
                ["docker", "exec", "-u", "root", self.container, "/bin/sh", "-c", command]
            )
        return command

    async def _stop_grader(self, script: str, state: str) -> None:
        from feather_train.errors import NonRetryableHarnessError

        try:
            result = await self.exec("python3", "-c", script, "stop", state, timeout=15)
            if not result.ok():
                raise RuntimeError(result.stderr or result.stdout)
        except Exception as exc:
            raise NonRetryableHarnessError(
                "Cannot confirm grader stopped; refusing to retry in this sandbox"
            ) from exc


@dataclass(frozen=True)
class ComposeSandbox:
    sandbox: AsyncSandbox
    runtime: ComposeRuntime
    egress_denylist_installed: bool = False

    @classmethod
    @asynccontextmanager
    async def start(
        cls,
        task: FeatherTaskData,
        *,
        timeout: int = 10800,
        max_attempts: int = 3,
        blocked_egress_domains: tuple[str, ...] = (),
    ):
        from feather_train.errors import NON_RETRYABLE_CAUSES

        async for attempt in AsyncRetrying(
            retry=retry_if_exception(
                lambda error: (
                    isinstance(error, Exception) and not isinstance(error, NON_RETRYABLE_CAUSES)
                )
            ),
            stop=stop_after_attempt(max_attempts),
            wait=wait_exponential(multiplier=1),
            reraise=True,
        ):
            with attempt:
                world = await cls._start(
                    task, timeout=timeout, blocked_egress_domains=blocked_egress_domains
                )
        # Retrying must end before yielding: an agent/grader failure must never
        # repeat a completed rollout inside the sandbox-start policy.
        try:
            yield world
        finally:
            await world.close()

    @classmethod
    async def _start(
        cls, task: FeatherTaskData, *, timeout: int, blocked_egress_domains: tuple[str, ...]
    ) -> ComposeSandbox:
        from feather_train.errors import NonRetryableHarnessError
        from feather_train.sandbox.compose import invoke_state_hook, resume_compose_stack
        from feather_train.sandbox.egress import install_agent_egress_denylist

        if not task.e2b_snapshot_id:
            raise NonRetryableHarnessError("FeatherEnv requires an offline-baked E2B snapshot")
        sandbox = await AsyncSandbox.create(
            template=task.e2b_snapshot_id,
            timeout=timeout,
            allow_internet_access=task.network_mode == "public",
            request_timeout=REQUEST_TIMEOUT,
        )
        try:
            runtime = await resume_compose_stack(
                SandboxIO(sandbox),
                mcp_servers=list(task.mcp_servers),
                require_verifier=task.environment_mode is None,
            )
            await stage_task_inputs(sandbox, task)
            if (
                runtime.state_export is not None
                and runtime.state_export.initialize_path is not None
            ):
                await invoke_state_hook(
                    SandboxIO(sandbox),
                    runtime.state_export,
                    runtime.state_export.initialize_path,
                )
            # Without internet access there is nothing to filter.
            restrict_egress = bool(blocked_egress_domains) and task.network_mode == "public"
            if restrict_egress:
                await install_agent_egress_denylist(SandboxIO(sandbox), blocked_egress_domains)
            return cls(sandbox, runtime, egress_denylist_installed=restrict_egress)
        except BaseException:
            await _close_sandbox(sandbox)
            raise

    async def close(self) -> None:
        await _close_sandbox(self.sandbox)

    async def stage_agent_trace(self, payload: bytes) -> None:
        """Stage Feather evidence before Harbor copies declared agent artifacts."""
        from feather_train.errors import NonRetryableHarnessError

        upload = f"/tmp/feather-agent-trace-{uuid.uuid4().hex}.json"
        await write_file(self.sandbox, upload, payload)
        try:
            prepare = (
                "test ! -L /logs && mkdir -p /logs/agent && "
                "test ! -L /logs/agent && "
                "test ! -e /logs/agent/trajectory.json && "
                "test ! -L /logs/agent/trajectory.json"
            )
            result = await SandboxIO(self.sandbox).exec(
                "docker exec -u root agent sh -c "
                f"{shlex.quote(prepare)} && "
                f"docker cp {shlex.quote(upload)} agent:/logs/agent/trajectory.json",
                shell=True,
                timeout=120,
            )
            if not result.ok():
                raise NonRetryableHarnessError(
                    "Could not stage the declared agent trajectory artifact before grading"
                )
        finally:
            try:
                await self.sandbox.files.remove(
                    upload, user="root", request_timeout=REQUEST_TIMEOUT
                )
            except Exception:
                logger.warning("Could not remove temporary agent trace upload", exc_info=True)

    async def prepare_grading(self, task: FeatherTaskData) -> SandboxIO:
        from feather_train.sandbox.compose import (
            assert_agent_container_survived,
            prepare_compose_grading,
        )
        from feather_train.sandbox.egress import lift_agent_egress_denylist

        host = SandboxIO(self.sandbox)
        await assert_agent_container_survived(host)
        if self.egress_denylist_installed:
            await lift_agent_egress_denylist(host)
        await prepare_compose_grading(
            host,
            self.runtime.state_export,
            verifier_network=self.runtime.verifier_network,
            mount_submission_volume=task.submission_transport == "submit_tool",
            environment_mode=task.environment_mode,
            collect_hooks=tuple(task.collect_hooks),
            artifacts=task.artifacts,
        )
        container = "agent" if task.environment_mode == "shared" else "verifier"
        return SandboxIO(self.sandbox, container=container)


async def stage_task_inputs(sandbox: AsyncSandbox, task: FeatherTaskData) -> None:
    if not task.task_artifacts.inputs:
        return
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import (
        GatedRepoError,
        RepositoryNotFoundError,
        RevisionNotFoundError,
    )

    from feather_train.errors import NonRetryableHarnessError
    from feather_train.tasks.paths import is_safe_task_id

    if not is_safe_task_id(task.task_id):
        raise NonRetryableHarnessError(f"Unsafe task ID: {task.task_id!r}")
    prefix = f"tasks/{task.task_id}/inputs"

    def load() -> dict[str, bytes]:
        try:
            directory = snapshot_download(
                task.hf_dataset,
                repo_type="dataset",
                revision=task.hf_revision,
                allow_patterns=[f"{prefix}/**"],
            )
        except (GatedRepoError, RepositoryNotFoundError, RevisionNotFoundError) as exc:
            raise NonRetryableHarnessError(
                f"Cannot load task inputs from {task.hf_dataset}@{task.hf_revision}: {exc}"
            ) from exc
        source = Path(directory) / prefix
        return {
            path.relative_to(source).as_posix(): path.read_bytes()
            for path in sorted(source.rglob("*"))
            if path.is_file()
        }

    files = await asyncio.to_thread(load)
    if not files:
        raise NonRetryableHarnessError(f"Task {task.task_id!r} declares inputs but has no files")
    await upload_task_inputs(sandbox, files)


async def upload_task_inputs(sandbox: AsyncSandbox, files: dict[str, bytes]) -> None:
    import tarfile
    from pathlib import PurePosixPath

    directories = set()
    for name in files:
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or not path.parts:
            raise ValueError(f"Task input path escapes workspace: {name!r}")
        directories.update(str(parent) for parent in path.parents if str(parent) != ".")
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for name in sorted(directories):
            entry = tarfile.TarInfo(name)
            entry.type, entry.mode = tarfile.DIRTYPE, 0o777
            archive.addfile(entry)
        for name, data in sorted(files.items()):
            entry = tarfile.TarInfo(name)
            entry.size, entry.mode = len(data), 0o666
            archive.addfile(entry, io.BytesIO(data))
    result = await SandboxIO(sandbox, container="agent").extract_archive(
        buffer.getvalue(),
        WORKSPACE_DIR,
        guard_paths=tuple(
            WORKSPACE_DIR + "/" + name for name in sorted(directories | files.keys())
        ),
    )
    if not result.ok():
        raise RuntimeError(f"Task input extraction failed: {result.stderr}")


async def _close_sandbox(sandbox: AsyncSandbox) -> None:
    try:
        await sandbox.kill(request_timeout=REQUEST_TIMEOUT)
    except Exception:
        logger.warning("E2B sandbox cleanup failed for %s", sandbox.sandbox_id, exc_info=True)
