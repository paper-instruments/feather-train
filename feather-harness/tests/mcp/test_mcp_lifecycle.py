from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
from mcp.types import ListToolsResult
from mcp.types import Tool as MCPTool

from feather_harness import McpServerSpec, run_with_mcp
from feather_harness.core.runner import RunResult
from feather_harness.tools import mcp as adapter
from feather_harness.tools.mcp import MCPTransportError


def result():
    return RunResult(stop_reason="completed", num_turns=0, compactions=0, segments=())


@pytest.fixture
def connections(monkeypatch):
    # Replace only MCP connections to inject cancellation and teardown failures.
    state = SimpleNamespace(
        owner=None,
        closed=[],
        cleanup_error=None,
        block_cleanup=False,
        closing=asyncio.Event(),
        release=asyncio.Event(),
        fail_connect=None,
    )

    @asynccontextmanager
    async def streams(url, **kwargs):
        owner = asyncio.current_task()
        state.owner = owner
        try:
            yield url, None, None
        finally:
            assert asyncio.current_task() is owner
            state.closed.append(url)
            state.closing.set()
            if state.block_cleanup:
                await state.release.wait()
            if state.cleanup_error:
                raise state.cleanup_error

    class Session:
        def __init__(self, read, write, **kwargs):
            self.url = read

        async def __aenter__(self):
            self.owner = asyncio.current_task()
            return self

        async def __aexit__(self, *args):
            assert asyncio.current_task() is self.owner

        async def initialize(self):
            if self.url == state.fail_connect:
                raise RuntimeError("initialization failed")

        async def list_tools(self, cursor: str | None = None) -> ListToolsResult:
            return ListToolsResult(tools=[MCPTool(name="change", inputSchema={"type": "object"})])

    monkeypatch.setattr(adapter, "streamable_http_client", streams)
    monkeypatch.setattr(adapter, "ClientSession", Session)
    return state


def servers():
    return tuple(
        McpServerSpec(name=name, transport="http", url=name) for name in ("first", "second")
    )


@pytest.mark.parametrize(
    "cleanup_error",
    [
        asyncio.CancelledError(),
        RuntimeError("close failed"),
        BaseExceptionGroup(
            "close failed",
            [
                asyncio.CancelledError(),
                ExceptionGroup("nested", [RuntimeError("socket closed")]),
            ],
        ),
    ],
    ids=["cancelled", "ordinary-error", "nested-cancellation"],
)
async def test_cleanup_failure_preserves_completed_result(connections, cleanup_error):
    connections.cleanup_error = cleanup_error
    expected = result()

    async def run(tools):
        assert asyncio.current_task() is connections.owner
        return expected

    assert await run_with_mcp(servers(), run) is expected
    assert connections.closed == ["second", "first"]


async def test_cleanup_cancellation_preserves_original_failure(connections):
    connections.cleanup_error = asyncio.CancelledError()
    error = ValueError("original failure")

    async def run(tools):
        raise error

    with pytest.raises(ValueError) as caught:
        await run_with_mcp(servers(), run)
    assert caught.value is error
    assert connections.closed == ["second", "first"]


async def test_partial_connection_failure_closes_opened_transports(connections):
    connections.fail_connect = "second"

    async def run(tools):
        pytest.fail("must not run after initialization failure")

    with pytest.raises(RuntimeError, match="initialization failed"):
        await run_with_mcp(servers(), run)
    assert connections.closed == ["second", "first"]


async def test_repeated_caller_cancellation_waits_for_cleanup(connections):
    connections.block_cleanup = True
    running = asyncio.Event()

    async def run(tools):
        running.set()
        await asyncio.Event().wait()

    task = asyncio.create_task(run_with_mcp(servers(), run))
    await running.wait()
    task.cancel()
    await connections.closing.wait()
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    connections.release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert connections.closed == ["second", "first"]


async def test_connection_cancellation_becomes_transport_failure(connections):
    async def run(tools):
        asyncio.current_task().cancel()
        await asyncio.sleep(0)

    with pytest.raises(MCPTransportError, match="cancelled internally"):
        await run_with_mcp(servers(), run)
    await asyncio.sleep(0)  # The caller was not cancelled.


async def test_cleanup_does_not_swallow_process_control_signals(connections):
    connections.cleanup_error = SystemExit(9)

    async def run(tools):
        return result()

    with pytest.raises(SystemExit) as caught:
        await run_with_mcp(servers(), run)
    assert caught.value.code == 9


async def test_cancellation_after_run_finishes_does_not_interrupt_cleanup(connections):
    connections.block_cleanup = True

    async def run(tools):
        return result()

    task = asyncio.create_task(run_with_mcp(servers(), run))
    await connections.closing.wait()
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    connections.release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert connections.closed == ["second", "first"]


async def test_cleanup_group_does_not_swallow_system_exit(connections):
    error = BaseExceptionGroup("close failed", [asyncio.CancelledError(), SystemExit(9)])
    connections.cleanup_error = error

    async def run(tools):
        return result()

    with pytest.raises(BaseExceptionGroup) as caught:
        await run_with_mcp(servers(), run)
    assert caught.value is error
