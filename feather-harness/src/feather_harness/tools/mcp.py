"""MCP tool conversion and task-affine connection ownership."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import re
from collections import Counter
from collections.abc import Awaitable, Callable
from contextlib import AsyncExitStack
from datetime import timedelta
from typing import Literal

import httpx
from mcp import ClientSession
from mcp.client.sse import sse_client
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.client.streamable_http import streamable_http_client
from mcp.shared.exceptions import McpError
from mcp.types import INTERNAL_ERROR, INVALID_PARAMS, ErrorData
from pydantic import Field, model_validator

from ..core.messages import FrozenModel, ImageBlock, TextBlock
from ..core.runner import RunResult
from ..scripts._files import MAX_FILE_SIZE_BYTES, MAX_OUTPUT_CHARS
from . import Tool, ToolResult, ToolSpec
from .files import FileToolTransport

logger = logging.getLogger(__name__)
_CALL_VALIDATION_ERROR = re.compile(
    r"^[1-9]\d* validation errors? for call\[[^\]\r\n]+\](?:\r?\n|$)"
)


class McpServerSpec(FrozenModel):
    name: str = Field(min_length=1)
    transport: Literal["streamable-http", "streamable_http", "http", "sse", "stdio"]
    url: str | None = None
    command: str | None = None
    args: tuple[str, ...] = ()
    headers: dict[str, str] = Field(default_factory=dict, repr=False)
    raise_on_error_result: bool = False
    request_timeout_seconds: float = Field(
        default=60.0, ge=0, strict=True, allow_inf_nan=False,
        description="MCP response/read timeout in seconds; 0 disables the deadline.",
    )

    @model_validator(mode="after")
    def validate_transport_fields(self) -> McpServerSpec:
        if self.transport == "stdio":
            if not self.command:
                raise ValueError("command is required for stdio transport")
            if self.url:
                raise ValueError("url is not used for stdio transport")
            if self.headers:
                raise ValueError("headers are not used for stdio transport")
            return self
        if not self.url:
            raise ValueError("url is required for HTTP/SSE transport")
        if self.command or self.args:
            raise ValueError("command and args are not used for HTTP/SSE transport")
        return self


class MCPTransportError(RuntimeError):
    pass


async def run_with_mcp(
    servers: tuple[McpServerSpec, ...],
    run: Callable[[tuple[Tool, ...]], Awaitable[RunResult]],
    *,
    reserved_names: frozenset[str] = frozenset(),
    enable_image_tools: bool = False,
    file_transport: FileToolTransport | None = None,
) -> RunResult:
    if not servers:
        return await run(())
    closing = asyncio.Event()
    # AnyIO scopes opened by MCP must close in the task that opened them. Shield
    # that task so transport cancellation cannot masquerade as caller cancellation.
    run_task = asyncio.create_task(_run_and_close(servers, run, reserved_names, closing, enable_image_tools, file_transport))
    try:
        outcome = await asyncio.shield(run_task)
    except asyncio.CancelledError:
        if not closing.is_set():
            run_task.cancel()
        while not run_task.done():
            try:
                await asyncio.shield(run_task)
            except asyncio.CancelledError:
                continue
        raise
    if isinstance(outcome, BaseException):
        if _internal_cancellation(outcome):
            raise MCPTransportError("MCP transport cancelled internally") from outcome
        raise outcome
    return outcome


async def _run_and_close(servers, run, reserved_names, closing, enable_image_tools, file_transport) -> RunResult | BaseException:
    stack = AsyncExitStack()
    try:
        tools = await _connect_tools(stack, servers, reserved_names, enable_image_tools, file_transport)
        outcome = await run(tools)
    except BaseException as exc:
        # Return failures as values so even CancelledError reaches the caller
        # through shield as an outcome, rather than cancelling its await.
        outcome = exc
    closing.set()
    try:
        await stack.aclose()
    except BaseException as exc:
        # Teardown artifacts must not replace a completed run or its primary failure.
        if not isinstance(exc, Exception) and not _internal_cancellation(exc):
            return exc
        logger.warning("MCP cleanup failed: %s: %s", type(exc).__name__, exc)
    return outcome


async def _connect_tools(stack, servers, reserved_names, enable_image_tools=False, file_transport=None) -> tuple[Tool, ...]:
    discovered = []
    for server in servers:
        response_timeout = server.request_timeout_seconds or None
        transport_read_timeout = (
            max(300, response_timeout) if response_timeout is not None else None
        )
        async with asyncio.timeout(60):
            if server.transport == "stdio":
                if server.command is None:
                    raise ValueError("command is required for stdio transport")
                streams = stdio_client(
                    StdioServerParameters(command=server.command, args=list(server.args))
                )
            elif server.transport == "sse":
                streams = sse_client(
                    _http_url(server), headers=server.headers, timeout=60,
                    sse_read_timeout=transport_read_timeout,
                )
            else:
                client = await stack.enter_async_context(
                    httpx.AsyncClient(
                        headers=server.headers,
                        timeout=httpx.Timeout(60, read=transport_read_timeout),
                        follow_redirects=True,
                    )
                )
                streams = streamable_http_client(_http_url(server), http_client=client)
            read, write, *_ = await stack.enter_async_context(streams)
            session = await stack.enter_async_context(
                ClientSession(
                    read, write,
                    read_timeout_seconds=(
                        timedelta(seconds=response_timeout) if response_timeout is not None else None
                    ),
                )
            )
            await session.initialize()
        for attempt in range(3):
            try:
                listing = await session.list_tools()
                break
            except Exception:
                if attempt == 2:
                    raise
                logger.warning("MCP discovery failed for %s; retrying", server.name)
                await asyncio.sleep(attempt + 1)
        discovered.extend((server, session, tool) for tool in listing.tools)
    names = _tool_names(discovered, reserved_names)
    return tuple(
        _bind(server, session, tool, name, enable_image_tools=enable_image_tools, file_transport=file_transport)
        for (server, session, tool), name in zip(discovered, names)
    )


def _bind(server, session, tool, name, *, enable_image_tools=False, file_transport: FileToolTransport | None = None) -> Tool:
    schema = {**tool.inputSchema}
    schema.setdefault("properties", {})
    description = (
        tool.description
        or tool.title
        or (tool.annotations.title if tool.annotations else None)
        or ""
    )

    async def invoke(raw: str) -> ToolResult:
        try:
            arguments = json.loads(raw) if raw else {}
        except ValueError:
            return _invalid_arguments(f"Invalid JSON input for tool {name}")
        if not isinstance(arguments, dict):
            return _invalid_arguments(f"Invalid JSON input for tool {name}: expected a JSON object")
        required = schema.get("required", [])
        missing = (
            [key for key in required if isinstance(key, str) and key not in arguments]
            if isinstance(required, list)
            else []
        )
        if missing:
            return _invalid_arguments(
                f"Failed to call tool '{tool.name}' on MCP server '{server.name}': missing required parameters: {', '.join(sorted(missing))}"
            )
        try:
            result = await session.call_tool(tool.name, arguments)
        except McpError as exc:
            if exc.error.code != INVALID_PARAMS:
                raise
            return _invalid_arguments(str(exc))
        if server.raise_on_error_result and result.isError:
            message = (
                "\n".join(
                    item.text.strip()
                    for item in result.content
                    if item.type == "text" and item.text.strip()
                )
                or "MCP tool returned an unspecified error result"
            )
            if message.startswith("Input validation error:") or _CALL_VALIDATION_ERROR.match(
                message
            ):
                return _invalid_arguments(message)
            raise McpError(ErrorData(code=INTERNAL_ERROR, message=message))
        content: list[TextBlock | ImageBlock] = []
        for item in result.content:
            if item.type == "image" and enable_image_tools:
                if item.mimeType not in {"image/png", "image/jpeg", "image/gif", "image/webp"}:
                    raise ValueError(f"Unsupported MCP image MIME type: {item.mimeType}")
                # The model can recover by requesting a smaller image; MIME/encoding faults cannot.
                if len(item.data) > ((MAX_FILE_SIZE_BYTES + 2) // 3) * 4 or len(base64.b64decode(item.data, validate=True)) > MAX_FILE_SIZE_BYTES:
                    return ToolResult(
                        content="Image exceeds the 10 MiB limit; request a smaller image or page.",
                        error="tool_error",
                    )
                content.append({"type": "image_url", "image_url": {"url": f"data:{item.mimeType};base64,{item.data}"}})
                continue
            if item.type in {"image", "audio"}:
                raise ValueError(f"MCP {item.type} output is not currently supported")
            text = item.text if item.type == "text" else str(item.model_dump(mode="json"))
            content.append({"text": text, "type": "text"})
        # Preserve the pinned Chat Completions representation, including a single block.
        return ToolResult(content=content)

    async def invoke_bounded(raw: str) -> ToolResult:
        result = await invoke(raw)
        if file_transport is None:
            return result
        text = (
            result.content
            if isinstance(result.content, str)
            else "".join(block["text"] for block in result.content if block["type"] == "text")
        )
        if len(text) <= MAX_OUTPUT_CHARS:
            return result
        try:
            preview = await file_transport.run(
                "_save_and_truncate_output", {"content": text, "tool_name": name}
            )
            if preview.error is not None or not preview.metadata.get("saved_path"):
                return result
            head_chars = preview.metadata["head_chars"]
            tail_chars = preview.metadata["tail_chars"]
            head, notice, tail = (
                preview.content[:head_chars],
                preview.content[head_chars : len(preview.content) - tail_chars],
                preview.content[len(preview.content) - tail_chars :],
            )
        except Exception:
            logger.warning("Could not save full %s output", name, exc_info=True)
            return result
        if isinstance(result.content, str):
            content = head + notice + tail
        else:
            content = _project_preview(result.content, len(text), head, notice, tail)
        return result.model_copy(update={"content": content})

    return Tool(spec=ToolSpec(name=name, description=description, parameters=schema), invoke=invoke_bounded)


def _project_preview(
    blocks: list[TextBlock | ImageBlock],
    total: int,
    head: str,
    notice: str,
    tail: str,
) -> list[TextBlock | ImageBlock]:
    content: list[TextBlock | ImageBlock] = []
    offset = 0
    for block in blocks:
        if block["type"] != "text":
            content.append(block)
            continue
        text = block["text"]
        end = offset + len(text)
        kept = text[: max(0, len(head) - offset)]
        if offset <= len(head) < end:
            kept += notice
        kept += text[max(0, total - len(tail) - offset) :]
        if kept:
            content.append({"type": "text", "text": kept})
        offset = end
    return content


def _invalid_arguments(message: str) -> ToolResult:
    return ToolResult(
        content=f"Invalid tool arguments:\n{message}\nRe-send the call with arguments that match the tool's schema.",
        error="invalid_arguments",
    )


def _tool_names(discovered, reserved) -> list[str]:
    # Preserve the SDK's stable 64-character names, including sanitized-name collisions.
    bases = [
        f"mcp__{_safe(server.name, 'server')}__{_safe(tool.name, 'tool')}"
        for server, _, tool in discovered
    ]
    counts = Counter(bases)
    candidates = []
    for index, ((server, _, tool), base) in enumerate(zip(discovered, bases)):
        seed = f"{server.name}\0{tool.name}"
        initial = _shorten(base, seed, counts[base] > 1 or base in reserved)
        candidates.append((initial, seed, index, base))
    used = set(reserved)
    names = [""] * len(discovered)
    for initial, seed, index, base in sorted(candidates):
        name = initial
        collision = 1
        while name in used:
            name = _shorten(base, f"{seed}\0{collision}", True)
            collision += 1
        names[index] = name
        used.add(name)
    return names


def _safe(value: str, fallback: str) -> str:
    return (
        "".join(
            char if char.isascii() and (char.isalnum() or char in "_-") else "_" for char in value
        ).strip("_-")
        or fallback
    )


def _shorten(base: str, seed: str, force_hash: bool) -> str:
    if len(base) <= 64 and not force_hash:
        return base
    suffix = hashlib.sha1(seed.encode()).hexdigest()[:8]
    return f"{base[:55].rstrip('_-') or 'mcp'}_{suffix}"


def _http_url(server: McpServerSpec) -> str:
    if not server.url:
        raise ValueError("url is required for HTTP/SSE transport")
    return server.url


def _internal_cancellation(error: BaseException) -> bool:
    if isinstance(error, asyncio.CancelledError):
        return True
    if isinstance(error, BaseExceptionGroup):
        cancelled, remaining = error.split(asyncio.CancelledError)
        return cancelled is not None and (remaining is None or isinstance(remaining, Exception))
    return False
