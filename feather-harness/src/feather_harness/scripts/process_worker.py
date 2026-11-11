"""Own Bash processes for one caller; exchange JSON lines on stdin/stdout."""

import asyncio
import json
import logging
import os
import re
import shlex
import signal
import sys
import tempfile
import uuid
from collections.abc import Callable, Sequence
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path

from _files import FileToolError
from _process_output import (
    _BoundedOutput,
    format_process_list,
    format_process_result,
)

logger = logging.getLogger(__name__)


@dataclass
class Task:
    task_id: str
    command: str
    description: str | None
    process: asyncio.subprocess.Process
    token: str
    log_file: Path
    output: _BoundedOutput = field(default_factory=_BoundedOutput)
    state: str = "running"
    exit_code: int | None = None
    log_path: str | None = None
    monitor: asyncio.Task[None] = field(init=False)
    terminating: bool = False


class Processes:
    def __init__(self, config):
        self.workspace = config["workspace"]
        self.grants = config["grants"]
        self.log_directory = Path(config["log_directory"])
        self.namespace = f"run_{uuid.uuid4().hex}"
        self.tasks: dict[str, Task] = {}
        self.foreground: Task | None = None
        self.next_foreground = 1

    async def invoke(self, name: str, args) -> str:
        if name == "bash":
            task = await self.start(args)
            if not args["run_in_background"]:
                try:
                    await asyncio.shield(task.monitor)
                finally:
                    if task.monitor.done():
                        self.foreground = None
            return format_process_result(task)
        if name == "task_list":
            tasks = list(self.tasks.values())
            if args["active_only"]:
                tasks = [task for task in tasks if not task.monitor.done()]
            # A failed monitor is an execution failure, not a successful task result.
            for task in tasks:
                if task.monitor.done():
                    task.monitor.result()
            return format_process_list(
                tuple(tasks[-args["limit"] :]),
                active_only=args["active_only"],
                total=len(tasks),
            )
        task = self.tasks.get(args["task_id"])
        if task is None:
            raise FileToolError(f"Task {args['task_id']} was not found.")
        if name == "task_stop":
            await stop(task)
        elif name == "task_output":
            if args["block"] and not task.monitor.done():
                try:
                    # Inspection timeout must not cancel the process monitor.
                    async with asyncio.timeout(args["timeout"] / 1000):
                        await asyncio.shield(task.monitor)
                except TimeoutError:
                    pass
            if task.monitor.done():
                task.monitor.result()
        else:
            raise ValueError(f"Unknown process operation: {name}")
        if not task.monitor.done():
            persist(task)
        return format_process_result(task)

    async def start(self, args) -> Task:
        background = args["run_in_background"]
        if background:
            active_count = sum(not task.monitor.done() for task in self.tasks.values())
            if active_count >= 32:
                raise FileToolError("At most 32 background tasks may run at once.")
        if background:
            task_id = f"task_{len(self.tasks) + 1}"
            log_key = task_id
        else:
            task_id = "foreground"
            log_key = f"foreground_{self.next_foreground}"
            self.next_foreground += 1
        task = await start_task(
            args,
            workspace=self.workspace,
            grants=self.grants,
            task_id=task_id,
            log_file=self.log_directory / f"{self.namespace}_{log_key}.log",
            token=uuid.uuid4().hex,
        )
        # Register ownership before any further await can be cancelled.
        if background:
            self.tasks[task_id] = task
        else:
            self.foreground = task
        return task

    async def close(self) -> None:
        tasks = [*self.tasks.values()]
        if self.foreground is not None:
            tasks.append(self.foreground)
        results = await asyncio.gather(*(stop(task) for task in tasks), return_exceptions=True)
        for result in results:
            if isinstance(result, BaseException):
                raise result


async def start_task(
    args,
    *,
    workspace: str,
    grants: Sequence[str],
    task_id: str,
    log_file: Path,
    token: str,
    on_update: Callable[[Task], None] | None = None,
) -> Task:
    background = args["run_in_background"]
    command = command_in_workdir(args["command"], args["workdir"], workspace, grants)
    timeout_ms = args["timeout"]
    if not background and not timeout_ms:
        timeout_ms = 120_000
    deadline = None
    if timeout_ms:
        deadline = asyncio.get_running_loop().time() + timeout_ms / 1000
    if background:
        # Fail before launching if the required background log cannot be created.
        write_log(log_file, b"")
    with ExitStack() as stack:
        argv = ["bash", "-c", command]
        pass_fds = ()
        if "\0" in command:
            raise ValueError("embedded null byte")
        if len(command.encode()) >= 128 * 1024:
            # Linux limits a single argv entry to 128 KiB. Let Bash read the
            # script through a separate fd; stdin remains /dev/null for commands.
            script = stack.enter_context(tempfile.TemporaryFile())
            # Store a shell-quoted literal: its closing quote keeps command
            # substitution from stripping meaningful trailing newlines.
            script.write(shlex.quote(command).encode())
            script.seek(0)
            fd = script.fileno()
            argv = ["bash", "-c", f'eval -- "eval -- $(< /dev/fd/{fd})" {fd}<&-']
            pass_fds = (fd,)
        launch = asyncio.create_task(
            asyncio.create_subprocess_exec(
                *argv,
                cwd=workspace,
                pass_fds=pass_fds,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                start_new_session=True,
                env={**os.environ, "FEATHER_PROCESS_TOKEN": token},
            )
        )
        try:
            process = await asyncio.shield(launch)
        except asyncio.CancelledError:
            # Retrieve the handle even if cancellation arrived during process creation.
            process = await launch
            await terminate(process.pid, token)
            await discard_output(process)
            raise
    task = Task(
        task_id=task_id,
        command=args["command"],
        description=args["description"],
        process=process,
        token=token,
        log_file=log_file,
        log_path=str(log_file) if background else None,
    )
    task.monitor = asyncio.create_task(monitor(task, deadline, on_update))
    return task


def command_in_workdir(
    command: str,
    workdir: str | None,
    workspace: str,
    grants: Sequence[str],
) -> str:
    if workdir is None or not workdir.strip():
        return command
    # Match the Bash tool's lexical workdir policy. This is deliberately not
    # the file tools' resolved-path policy; Bash itself can access the OS user’s files.
    raw = workdir.replace("\\", "/")
    path = Path(os.path.normpath(os.path.join(workspace, raw)))
    allowed = path.is_relative_to(workspace) or (
        Path(raw).is_absolute() and any(path.is_relative_to(root) for root in grants)
    )
    if not allowed or re.match(r"^[A-Za-z]:[/\\]", raw):
        raise FileToolError(
            "Working directory must be inside the workspace or an explicitly granted path.",
            "invalid_path",
        )
    # An && prefix would guard only the first command in a multi-command script.
    return f"cd {shlex.quote(str(path))} || exit\n{command}"


async def monitor(
    task: Task,
    deadline: float | None,
    on_update: Callable[[Task], None] | None = None,
) -> None:
    try:
        try:
            async with asyncio.timeout_at(deadline):
                await drain(task, on_update)
                # A shell can exit while a background child has redirected its output.
                # Completion still waits for living members of its process group.
                next_update = asyncio.get_running_loop().time() + 1
                while await managed_groups(task.process.pid, task.token):
                    await asyncio.sleep(0.1)
                    if asyncio.get_running_loop().time() >= next_update:
                        if on_update is not None and task.task_id != "foreground":
                            on_update(task)
                        next_update = asyncio.get_running_loop().time() + 1
            task.state = "exited"
        except (TimeoutError, asyncio.CancelledError) as exc:
            task.terminating = True
            await terminate(task.process.pid, task.token)
            async with asyncio.timeout(1):
                await drain(task, on_update)
            task.state = "timed_out" if isinstance(exc, TimeoutError) else "stopped"
        returncode = await task.process.wait()
        # Match shell exit codes for signal termination (e.g. SIGTERM -> 143).
        task.exit_code = 128 - returncode if returncode < 0 else returncode
        if task.task_id != "foreground":
            persist(task)
        elif task.output.preview()[1]:
            try:
                persist(task)
            except OSError:
                logger.warning("Could not save truncated foreground output", exc_info=True)
        if on_update is not None:
            on_update(task)
    except BaseException:
        task.state = "failed"
        task.terminating = True
        await terminate(task.process.pid, task.token)
        await discard_output(task.process)
        if on_update is not None:
            on_update(task)
        raise


async def discard_output(process: asyncio.subprocess.Process) -> None:
    try:
        async with asyncio.timeout(1):
            while await process.stdout.read(65_536):
                pass
            await process.wait()
    except TimeoutError as exc:
        # A detached child can clear its ownership token and retain stdout.
        # asyncio exposes no public way to close the inherited pipe transport.
        process._transport.close()
        raise RuntimeError(
            "Process cleanup incomplete: stdout remained open after termination"
        ) from exc


async def drain(task: Task, on_update: Callable[[Task], None] | None = None) -> None:
    loop = asyncio.get_running_loop()
    next_flush = loop.time() + 1
    saved_bytes = task.output.total_bytes
    while True:
        try:
            async with asyncio.timeout_at(next_flush):
                chunk = await task.process.stdout.read(65_536)
        except TimeoutError:
            chunk = None
        if chunk == b"":
            if task.task_id != "foreground" and task.output.total_bytes != saved_bytes:
                persist(task)
            break
        if chunk:
            task.output.append(chunk)
        if loop.time() >= next_flush:
            if task.task_id != "foreground":
                if task.output.total_bytes != saved_bytes:
                    persist(task)
                    saved_bytes = task.output.total_bytes
                if on_update is not None:
                    on_update(task)
            next_flush = loop.time() + 1
    await task.process.wait()


async def managed_groups(group: int, token: str) -> set[int]:
    if sys.platform == "linux":
        # A descendant may call setsid(). Its inherited token keeps it owned even
        # after it leaves Bash's process group, matching the pinned E2B helper.
        needle = f"FEATHER_PROCESS_TOKEN={token}".encode()
        groups = set()
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            try:
                text = (entry / "stat").read_text()
                # The parenthesized process name can contain spaces or parentheses.
                fields = text[text.rfind(")") + 2 :].split()
                state = fields[0]
                process_group = int(fields[2])
                if process_group <= 1 or state == "Z":
                    continue
                if process_group == group:
                    groups.add(process_group)
                    continue
                environment = (entry / "environ").read_bytes().split(b"\0")
                if needle in environment:
                    groups.add(process_group)
            except (FileNotFoundError, ProcessLookupError, PermissionError):
                # Processes can exit between directory enumeration and either read.
                continue
        return groups
    # Local macOS development supports ordinary groups; detached descendants
    # require the Linux /proc ownership path used by the release target.
    process = await asyncio.create_subprocess_exec(
        "ps",
        "-eo",
        "pgid=,stat=",
        stdout=asyncio.subprocess.PIPE,
    )
    output, _ = await process.communicate()
    if process.returncode:
        raise RuntimeError("Could not inspect Bash process group")
    for line in output.splitlines():
        process_group, state = line.split()
        if int(process_group) == group and not state.startswith(b"Z"):
            return {group}
    return set()


async def terminate(group: int, token: str) -> None:
    groups = await managed_groups(group, token)
    signal_groups(groups, signal.SIGTERM)
    # Process-discovery time counts toward the half-second grace period.
    loop = asyncio.get_running_loop()
    deadline = loop.time() + 0.5
    while loop.time() < deadline:
        groups = await managed_groups(group, token)
        if not groups:
            return
        await asyncio.sleep(min(0.05, max(0, deadline - loop.time())))
    signal_groups(await managed_groups(group, token), signal.SIGKILL)


def signal_groups(groups: set[int], sig: signal.Signals) -> None:
    for group in groups:
        try:
            os.killpg(group, sig)
        except ProcessLookupError:
            pass


async def stop(task: Task) -> None:
    # Once timeout/stop cleanup starts, another stop must wait rather than cancel it.
    if not task.monitor.done() and not task.terminating and not task.monitor.cancelling():
        task.monitor.cancel()
    await asyncio.shield(task.monitor)


def persist(task: Task) -> None:
    write_log(task.log_file, task.output.bytes())
    task.log_path = str(task.log_file)


def write_log(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Readers must never observe a half-written bounded log.
    temporary = path.with_suffix(".tmp")
    temporary.write_bytes(content)
    temporary.replace(path)


async def serve() -> None:
    reader = asyncio.StreamReader(limit=2**31)
    protocol = asyncio.StreamReaderProtocol(reader)
    await asyncio.get_running_loop().connect_read_pipe(lambda: protocol, sys.stdin)
    first = await reader.readline()
    if not first:
        return
    config = json.loads(first)
    processes = Processes(config)
    requests = asyncio.Queue()

    async def receive() -> None:
        while line := await reader.readline():
            await requests.put(json.loads(line))

    async def execute() -> None:
        while True:
            request = await requests.get()
            try:
                result = {"content": await processes.invoke(request["name"], request["arguments"])}
            except FileToolError as exc:
                result = {"content": str(exc), "error": exc.category}
            sys.stdout.write(json.dumps(result, ensure_ascii=False) + "\n")
            sys.stdout.flush()

    receiving = asyncio.create_task(receive())
    executing = asyncio.create_task(execute())
    try:
        done, _ = await asyncio.wait((receiving, executing), return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            task.result()
    finally:
        receiving.cancel()
        executing.cancel()
        await asyncio.gather(receiving, executing, return_exceptions=True)
        await processes.close()


if __name__ == "__main__":
    asyncio.run(serve())
