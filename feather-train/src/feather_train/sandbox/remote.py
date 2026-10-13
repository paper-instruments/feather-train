"""Per-command result storage on the E2B host, outside the agent container."""

import asyncio
import json
import os
import sys
import time
from pathlib import Path


def save(path: Path, value) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def alive(pid: int, path: Path) -> bool:
    try:
        os.kill(pid, 0)
        if sys.platform != "linux":
            return True
        # A killed collector can remain a zombie until envd reaps it.
        text = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        args = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
        return text[text.rfind(")") + 2 :].split()[0] != "Z" and os.fsencode(path) in args
    except (FileNotFoundError, ProcessLookupError):
        return False


def snapshot(path: Path):
    value = (
        json.loads(path.with_suffix(".result").read_text(encoding="utf-8"))
        if path.with_suffix(".result").exists()
        else None
    )
    if value and value["done"]:
        return value
    pid_file = path.with_suffix(".pid")
    if pid_file.exists() and not alive(int(pid_file.read_text(encoding="utf-8")), path):
        # Publication may have happened between the first read and process exit.
        result = path.with_suffix(".result")
        if result.exists():
            completed = json.loads(result.read_text(encoding="utf-8"))
            if completed["done"]:
                return completed
        raise RuntimeError(f"Tool supervisor died without a completed result: {path.stem}")
    return value


async def inspect(path: Path, wait: bool, wait_ms: int | None = None, fresh: bool = False):
    requested_at = time.monotonic()
    # The launch acknowledgement may have been lost before the collector started.
    # Bound only that ambiguity; a running command has its own tool deadline.
    startup_deadline = requested_at + 10
    wait_until = requested_at + wait_ms / 1000 if wait_ms is not None else None
    while True:
        value = snapshot(path)
        if value is not None and value["done"]:
            return value
        now = time.monotonic()
        wait_expired = wait_until is not None and now >= wait_until
        publication_current = not fresh
        if fresh:
            heartbeat = path.with_suffix(".heartbeat")
            publication_current = (
                heartbeat.exists() and float(heartbeat.read_text(encoding="utf-8")) >= requested_at
            )
        if value is not None and (wait_expired or (not wait and publication_current)):
            # The collector may have replaced the result after our first read
            # but before the heartbeat confirmed a current publication.
            return snapshot(path) if fresh else value
        if fresh and not publication_current and now > startup_deadline:
            raise RuntimeError(f"Tool did not publish fresh output: {path.stem}")
        startup_expired = now > startup_deadline and (
            not wait or not path.with_suffix(".started").exists()
        )
        if value is None and (startup_expired or wait_expired):
            raise RuntimeError(f"Tool result unavailable: {path.stem}")
        await asyncio.sleep(0.1)


async def run(path: Path, payload: str | None = None) -> None:
    save(path.with_suffix(".pid"), os.getpid())
    if payload is None:
        payload = path.read_text(encoding="utf-8")
        path.unlink()  # Large write/edit payloads are no longer needed after loading.
    request = json.loads(payload)
    if request.get("capture"):
        await capture(path, request["docker"])
        return
    emitted = False
    process = await asyncio.create_subprocess_exec(
        *request["docker"],
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        # Image reads return up to 5 MB of base64; leave room for the JSON envelope.
        # Other tools keep the existing bound on supervisor protocol lines.
        limit=16 * 1024 * 1024 if request.get("name") == "read" else 1_000_000,
    )

    async def errors() -> str:
        tail = b""
        while chunk := await process.stderr.read(65536):
            tail = (tail + chunk)[-16000:]
        return tail.decode(errors="replace")

    stderr = asyncio.create_task(errors())
    last = None
    try:
        process.stdin.write(json.dumps(request).encode())
        await process.stdin.drain()
        process.stdin.close()
        while line := await process.stdout.readline():
            update = json.loads(line)
            if update.get("started"):
                path.with_suffix(".started").touch()
                continue
            if update.get("heartbeat"):
                save(path.with_suffix(".heartbeat"), time.monotonic())
                continue
            save(path.with_suffix(".result"), update)
            last = update
            save(path.with_suffix(".heartbeat"), time.monotonic())
            if not emitted and (last["done"] or request["background"]):
                print(json.dumps(last), flush=True)
                emitted = True
        await process.wait()
        error = await stderr
        if last is None or not last["done"]:
            raise RuntimeError(
                f"Container tool exited without a complete result ({process.returncode}): {error}"
            )
    except Exception as exc:
        # Recovery must keep the saved result even if heartbeat or delivery fails.
        if last is not None and last["done"]:
            raise
        failure = {"done": True, "failure": f"{type(exc).__name__}: {exc}"}
        save(path.with_suffix(".result"), failure)
        if not emitted:
            print(json.dumps(failure), flush=True)
    finally:
        # The outer transport owns in-container cleanup. Killing docker exec
        # alone would not reliably stop a tool or its detached descendants.
        if process.returncode is None:
            process.kill()
        await process.wait()
        await stderr


async def capture(path: Path, command: list[str]) -> None:
    """Grader stdout is data, never the tool supervisor's JSON protocol."""
    try:
        process = await asyncio.create_subprocess_exec(
            *command,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        path.with_suffix(".started").touch()
        stdout, stderr = await process.communicate()
        value = {
            "done": True,
            "result": {
                "exit_code": process.returncode,
                "stdout": stdout.decode("utf-8", errors="replace"),
                "stderr": stderr.decode("utf-8", errors="replace"),
            },
        }
    except Exception as exc:
        value = {"done": True, "failure": f"{type(exc).__name__}: {exc}"}
    # Publish before delivery so a lost connection cannot lose a completed command.
    save(path.with_suffix(".result"), value)
    print(json.dumps(value), flush=True)


async def main():
    mode, path = sys.argv[1], Path(sys.argv[2])
    if mode == "run":
        await run(path, sys.argv[3] if len(sys.argv) > 3 else None)
    elif mode == "inspect":
        print(
            json.dumps(
                await inspect(
                    path, sys.argv[3] == "wait", int(sys.argv[4]), fresh=sys.argv[5:] == ["fresh"]
                )
            ),
            flush=True,
        )
    elif mode == "tasks":
        request = json.loads(path.read_text(encoding="utf-8"))
        values = []
        for item in request["paths"]:
            state = await inspect(Path(item), False)
            # Listing tasks must not transfer every task's output back over E2B.
            values.append(
                {key: state[key] for key in ("done", "failure", "list_entry") if key in state}
            )
        print(json.dumps(values), flush=True)
    elif mode == "finish":
        await finish(path)
    elif mode == "started":
        async with asyncio.timeout(10):
            for name in sys.argv[2:]:
                while not Path(name).with_suffix(".started").exists():
                    await asyncio.sleep(0.1)


async def finish(prefix: Path) -> None:
    collectors = [
        (int(item.read_text(encoding="utf-8")), item.with_suffix(".json"))
        for item in prefix.parent.glob(prefix.name + "-*.pid")
    ]
    deadline = asyncio.get_running_loop().time() + 3
    while (
        any(alive(pid, path) for pid, path in collectors)
        and asyncio.get_running_loop().time() < deadline
    ):
        await asyncio.sleep(0.1)
    stalled = [(pid, path) for pid, path in collectors if alive(pid, path)]
    for pid, _ in stalled:
        try:
            os.kill(pid, 9)
        except ProcessLookupError:
            pass
    async with asyncio.timeout(5):
        while any(alive(pid, path) for pid, path in stalled):
            await asyncio.sleep(0.1)
    failures = []
    for _, path in collectors:
        result = path.with_suffix(".result")
        state = json.loads(result.read_text(encoding="utf-8")) if result.exists() else None
        if state is None or not state["done"]:
            failures.append(f"Tool collector ended without a completed result: {path.stem}")
        elif "failure" in state:
            failures.append(state["failure"])
    # No surviving collector can recreate a result after this point.
    for item in prefix.parent.glob(prefix.name + "-*"):
        item.unlink(missing_ok=True)
    if failures:
        raise RuntimeError("; ".join(failures))


if __name__ == "__main__":
    asyncio.run(main())
