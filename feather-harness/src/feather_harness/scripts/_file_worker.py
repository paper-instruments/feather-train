"""Shared subprocess lifecycle for local and sandbox file tools."""

import asyncio
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

SEARCH_TIMEOUT_SECONDS = 30


async def run_file_worker(name: str, payload: bytes) -> tuple[int, bytes, bytes]:
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        str(Path(__file__).with_name(f"{name}_tool.py")),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        # The JSON protocol is UTF-8 even when the host uses a legacy locale.
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        start_new_session=True,
    )
    timeout_seconds = SEARCH_TIMEOUT_SECONDS if name in {"grep", "glob"} else None
    try:
        # Search owns ripgrep children; cap the whole call, including enumeration.
        async with asyncio.timeout(timeout_seconds):
            stdout, stderr = await process.communicate(payload)
    except TimeoutError:
        result = {
            "content": f"{name} search timed out after {timeout_seconds} seconds.",
            "error": "tool_error",
        }
        return 0, json.dumps(result).encode(), b""
    finally:
        if process.returncode is None:
            # Stop the worker and its children together on cancellation or timeout.
            _kill_process_tree(process.pid)
            await process.wait()
    return await process.wait(), stdout, stderr


def _kill_process_tree(pid: int) -> None:
    if sys.platform != "win32":
        os.killpg(pid, signal.SIGKILL)
        return
    # Windows has no process groups for this; taskkill /T ends the worker's
    # ripgrep children too. Absolute path so a planted taskkill can't run.
    taskkill = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "taskkill.exe"
    subprocess.run(
        [str(taskkill), "/PID", str(pid), "/T", "/F"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        check=False,
    )
