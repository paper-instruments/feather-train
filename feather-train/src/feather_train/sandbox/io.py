"""Command and byte-file operations for Compose setup and the private grader."""

from __future__ import annotations

import base64
import shlex
from typing import TYPE_CHECKING

from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential
from verifiers.v1 import SandboxError

if TYPE_CHECKING:
    from feather_train.sandbox.lifecycle import CommandOutput, SandboxIO


async def exec_bash(session: SandboxIO, cmd: str, *, timeout: float | None = None) -> CommandOutput:
    return await session.exec("bash", "-lc", cmd, timeout=timeout)


async def read_remote_bytes(session: SandboxIO, path: str, *, check: bool = True) -> bytes | None:
    """Read with transport recovery; unchecked command failures return None."""
    # E2B decodes command output as text; base64 preserves arbitrary file bytes.
    # Retry transport failures while reading an existing result, never grader execution.
    async for attempt in AsyncRetrying(
        retry=retry_if_exception_type(SandboxError),
        stop=stop_after_attempt(10),
        wait=wait_exponential(multiplier=1),
        reraise=True,
    ):
        with attempt:
            result = await exec_bash(session, f"base64 -w0 {shlex.quote(path)}", timeout=60)
    if not result.ok():
        if not check:
            return None
        detail = result.stderr or result.stdout
        raise SandboxError(f"read_remote_bytes {path} failed: {detail[:500]}")
    return base64.b64decode(result.stdout)
