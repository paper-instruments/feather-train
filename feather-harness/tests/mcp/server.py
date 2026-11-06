"""A real local MCP server for the HTTP/SSE adapter tests."""

from __future__ import annotations

import asyncio
import socket
import threading
import time
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

import uvicorn
from mcp.server.lowlevel import Server
from mcp.server.sse import SseServerTransport
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from mcp.types import (
    CallToolResult,
    ImageContent,
    ListToolsRequest,
    ListToolsResult,
    TextContent,
    Tool,
)
from starlette.applications import Starlette
from starlette.routing import Mount, Route


@contextmanager
def mcp_server(
    *,
    discovery_failures=0,
    call_delay=0,
    text_blocks=None,
    fail_response_after_write: Path | None = None,
    tool_names: tuple[str, ...] | None = None,
    page_size: int | None = None,
    pagination: Literal["finite", "repeat", "endless"] = "finite",
):
    if page_size is not None and page_size <= 0:
        raise ValueError("page_size must be positive")
    state = SimpleNamespace(
        list_calls=0, calls=[], headers=[], discovery_failures=discovery_failures
    )
    server = Server("fixture")

    @server.list_tools()
    async def list_tools(request: ListToolsRequest):
        state.list_calls += 1
        if state.list_calls <= state.discovery_failures:
            raise RuntimeError("discovery temporarily unavailable")
        if tool_names is not None:
            tools = [
                Tool(
                    name=name,
                    description=name,
                    inputSchema={
                        "type": "object",
                        "properties": {"value": {"type": "string"}},
                        "required": ["value"],
                    },
                )
                for name in tool_names
            ]
            if page_size is None:
                return tools
            if pagination == "repeat":
                return ListToolsResult(tools=[], nextCursor="repeated")
            start = int(request.params.cursor) if request.params and request.params.cursor else 0
            end = start + page_size
            if pagination == "endless":
                return ListToolsResult(tools=[], nextCursor=str(end))
            return ListToolsResult(
                tools=tools[start:end],
                nextCursor=str(end) if end < len(tools) else None,
            )
        return [
            Tool(
                name="echo",
                description="Echo a value",
                inputSchema={
                    "type": "object",
                    "properties": {"value": {"type": "string"}},
                    "required": ["value"],
                },
            ),
            Tool(name="failure", inputSchema={"type": "object"}),
            Tool(name="image", inputSchema={"type": "object"}),
        ]

    @server.call_tool()
    async def call_tool(name, arguments):
        state.calls.append((name, arguments))
        if fail_response_after_write is not None:
            with fail_response_after_write.open("a") as marker:
                marker.write(arguments["value"] + "\n")
        if call_delay:
            await asyncio.sleep(call_delay)
        if name == "failure":
            return CallToolResult(
                content=[
                    TextContent(type="text", text=arguments.get("message", "database unavailable"))
                ],
                isError=True,
            )
        if name == "image":
            return [ImageContent(type="image", data="AA==", mimeType="image/png")]
        blocks = text_blocks if text_blocks is not None else ["Echo:", arguments["value"]]
        return [TextContent(type="text", text=text) for text in blocks]

    manager = StreamableHTTPSessionManager(server, json_response=True)
    sse = SseServerTransport("/messages/")

    class HttpEndpoint:
        async def __call__(self, scope, receive, send):
            state.headers.append(dict(scope.get("headers", ())))

            async def send_response(message):
                # Return an HTTP failure after the mutation has already committed.
                if (
                    fail_response_after_write is not None
                    and state.calls
                    and message["type"] == "http.response.start"
                ):
                    message = {**message, "status": 503}
                await send(message)

            await manager.handle_request(scope, receive, send_response)

    class SseEndpoint:
        async def __call__(self, scope, receive, send):
            state.headers.append(dict(scope["headers"]))
            async with sse.connect_sse(scope, receive, send) as (read, write):
                await server.run(read, write, server.create_initialization_options())

    @asynccontextmanager
    async def lifespan(app):
        async with manager.run():
            yield

    app = Starlette(
        routes=[
            Mount("/mcp", app=HttpEndpoint()),
            Route("/sse", SseEndpoint()),
            Mount("/messages/", app=sse.handle_post_message),
        ],
        lifespan=lifespan,
    )
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    sock.listen()
    port = sock.getsockname()[1]
    http = uvicorn.Server(uvicorn.Config(app, log_level="critical", lifespan="on"))
    thread = threading.Thread(target=lambda: asyncio.run(http.serve(sockets=[sock])), daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 5
        while not http.started:
            if time.monotonic() >= deadline:
                raise TimeoutError("MCP test server did not start")
            time.sleep(0.01)
        yield f"http://127.0.0.1:{port}", state
    finally:
        http.should_exit = True
        thread.join(timeout=5)
        sock.close()
        assert not thread.is_alive(), "MCP server did not shut down"
