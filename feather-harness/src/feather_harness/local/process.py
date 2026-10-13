"""Caller-owned connection to the local managed-process worker."""

from __future__ import annotations

import asyncio
import json
import os
import shlex
import sys
import tempfile
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path

from pydantic import JsonValue

from ..tools import ToolResult


class LocalShellRuntime:
    def __init__(
        self, workspace: Path, *, log_directory: Path, grants: tuple[Path, ...] = (),
        task_python: Path | None = None,
    ):
        self.workspace = Path(os.path.abspath(workspace))
        self.log_directory = Path(os.path.abspath(log_directory))
        self.grants = tuple(Path(os.path.abspath(path)) for path in grants)
        self.task_python = Path(os.path.abspath(task_python)) if task_python is not None else None
        if self.task_python is not None and (
            not self.task_python.is_file() or not os.access(self.task_python, os.X_OK)
        ):
            raise ValueError("Task Python must be an existing executable file")
        if not self.workspace.is_dir():
            raise ValueError("workspace must be a directory")
        if any(path.resolve() == path.resolve().parent for path in self.grants):
            raise ValueError("path grants cannot cover the filesystem root")

    async def __aenter__(self):
        self._contexts = ExitStack()
        environment = self._contexts.enter_context(_task_environment(self.task_python))
        launching = asyncio.create_task(
            asyncio.create_subprocess_exec(
                sys.executable,
                str(Path(__file__).parent.parent / "scripts" / "process_worker.py"),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                # Worker failures belong in the caller's diagnostics, not model output.
                stderr=None,
                limit=1_000_000,
                env=environment,
            )
        )
        try:
            self.process = await asyncio.shield(launching)
        except asyncio.CancelledError:
            while not launching.done():
                try:
                    await asyncio.shield(launching)
                except asyncio.CancelledError:
                    continue
            try:
                self.process = launching.result()
            except BaseException:
                self._contexts.close()
                raise
            await self.__aexit__(asyncio.CancelledError, None, None)
            raise
        except BaseException:
            self._contexts.close()
            raise
        try:
            self.process.stdin.write(
                json.dumps(
                    {
                        "workspace": str(self.workspace),
                        "log_directory": str(self.log_directory),
                        "grants": [str(path) for path in self.grants],
                    }
                ).encode()
                + b"\n"
            )
            await self.process.stdin.drain()
        except BaseException:
            await self.__aexit__(*sys.exc_info())
            raise
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        self.process.stdin.close()
        # Closing stdin lets the worker stop all owned commands and drain their output.
        closing = asyncio.create_task(self.process.wait())
        cancelled = False
        while not closing.done():
            try:
                await asyncio.shield(closing)
            except asyncio.CancelledError:
                cancelled = True
        self._contexts.close()
        if cancelled:
            raise asyncio.CancelledError
        if closing.result() and exc_type is None:
            raise RuntimeError(f"Process worker exited with code {closing.result()}")

    async def run(self, name: str, arguments: dict[str, JsonValue]) -> ToolResult:
        self.process.stdin.write(
            json.dumps({"name": name, "arguments": arguments}).encode() + b"\n"
        )
        try:
            await self.process.stdin.drain()
            response = await self.process.stdout.readline()
        except asyncio.CancelledError:
            await self.__aexit__(*sys.exc_info())
            raise
        if not response:
            code = await self.process.wait()
            raise RuntimeError(f"Process worker exited with code {code}")
        return ToolResult.model_validate_json(response)


@contextmanager
def _task_environment(task_python: Path | None) -> Iterator[dict[str, str]]:
    environment = {**os.environ, "FEATHER_PYTHON": sys.executable}
    if task_python is None:
        yield environment
        return
    for name in ("PYTHONHOME", "PYTHONPATH", "PYTHONNOUSERSITE", "VIRTUAL_ENV"):
        environment.pop(name, None)
    if (task_python.parent.parent / "pyvenv.cfg").is_file():
        environment["VIRTUAL_ENV"] = str(task_python.parent.parent)
    # Preserve the selected executable's venv path, even when it is a symlink. pip is
    # bound too: a selected venv without pip would otherwise fall through PATH.
    python = f"exec {shlex.quote(str(task_python))}"
    shims = {
        "python": f'{python} "$@"', "python3": f'{python} "$@"',
        "pip": f'{python} -m pip "$@"', "pip3": f'{python} -m pip "$@"',
    }
    with tempfile.TemporaryDirectory(prefix="feather-task-python-") as aliases:
        for name, body in shims.items():
            shim = Path(aliases) / name
            shim.write_text(f"#!/bin/sh\n{body}\n")
            shim.chmod(0o700)
        environment["PATH"] = os.pathsep.join(
            (aliases, str(task_python.parent), environment.get("PATH", ""))
        )
        yield environment
