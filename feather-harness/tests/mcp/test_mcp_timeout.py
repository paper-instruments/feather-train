from __future__ import annotations

import asyncio
from datetime import timedelta

import pytest
from mcp.shared.exceptions import McpError
from pydantic import ValidationError

from feather_harness import McpServerSpec, run_with_mcp
from feather_harness.core.runner import RunResult
from feather_harness.tools import mcp

from .server import mcp_server


@pytest.mark.parametrize("transport,path", [("http", "/mcp/"), ("sse", "/sse")])
@pytest.mark.parametrize("request_timeout_seconds", [0, 0.05])
async def test_delayed_tool_response_obeys_server_deadline(transport, path, request_timeout_seconds):
    with mcp_server(call_delay=0.2) as (url, server):
        async def run(tools):
            result = await tools[0].invoke('{"value": "delayed"}')
            assert result.content[-1]["text"] == "delayed"
            return RunResult(stop_reason="completed", num_turns=0, compactions=0, segments=())

        spec = McpServerSpec(
            name="slow", transport=transport, url=url + path,
            request_timeout_seconds=request_timeout_seconds,
        )
        async with asyncio.timeout(5):
            if request_timeout_seconds:
                with pytest.raises((McpError, mcp.MCPTransportError)):
                    await run_with_mcp((spec,), run)
            else:
                result = await run_with_mcp((spec,), run)
                assert result.stop_reason == "completed"
        assert server.calls == [("echo", {"value": "delayed"})]


@pytest.mark.parametrize("deadline", [None, 0, 12.5])
@pytest.mark.parametrize("transport,path", [("http", "/mcp/"), ("sse", "/sse")])
async def test_response_timeout_reaches_session_and_transport(monkeypatch, deadline, transport, path):
    session_timeouts = []
    transport_timeouts = []
    session_factory = mcp.ClientSession
    http_factory = mcp.httpx.AsyncClient
    sse_factory = mcp.sse_client

    def session(*args, **kwargs):
        session_timeouts.append(kwargs["read_timeout_seconds"])
        return session_factory(*args, **kwargs)

    def http_client(*args, **kwargs):
        transport_timeouts.append(kwargs["timeout"])
        return http_factory(*args, **kwargs)

    def sse_client(*args, **kwargs):
        transport_timeouts.append((kwargs["timeout"], kwargs["sse_read_timeout"]))
        return sse_factory(*args, **kwargs)

    monkeypatch.setattr(mcp, "ClientSession", session)
    if transport == "http":
        monkeypatch.setattr(mcp.httpx, "AsyncClient", http_client)
    else:
        monkeypatch.setattr(mcp, "sse_client", sse_client)

    with mcp_server() as (url, _):
        async def run(tools):
            return RunResult(stop_reason="completed", num_turns=0, compactions=0, segments=())

        options = {} if deadline is None else {"request_timeout_seconds": deadline}
        spec = McpServerSpec(name="slow", transport=transport, url=url + path, **options)
        await run_with_mcp((spec,), run)
    expected = 60 if deadline is None else deadline or None
    expected_read = max(300, expected) if expected is not None else None
    assert session_timeouts == [timedelta(seconds=expected) if expected is not None else None]
    if transport == "http":
        assert transport_timeouts[0].connect == 60
        assert transport_timeouts[0].read == expected_read
    else:
        assert transport_timeouts == [(60, expected_read)]


@pytest.mark.parametrize("deadline", [-1, float("nan"), float("inf"), True, "60"])
def test_response_timeout_requires_finite_nonnegative_number(deadline):
    with pytest.raises(ValidationError):
        McpServerSpec(
            name="slow", transport="http", url="http://slow/mcp/",
            request_timeout_seconds=deadline,
        )
