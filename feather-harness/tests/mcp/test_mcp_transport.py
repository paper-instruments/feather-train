from __future__ import annotations

import asyncio
import json
import re
import sys
from pathlib import Path

import pytest
from mcp.shared.exceptions import McpError
from pydantic import ValidationError

from feather_harness import McpServerSpec, run_with_mcp
from feather_harness.core.runner import RunResult
from feather_harness.tools.mcp import MCPTransportError

from .server import mcp_server

_STDIO_SERVER = Path(__file__).with_name("stdio_server.py")


def completed():
    return RunResult(stop_reason="completed", num_turns=0, compactions=0, segments=())


@pytest.mark.parametrize("transport,path", [("http", "/mcp/"), ("sse", "/sse")])
async def test_http_and_sse_execute_discovered_tools(transport, path):
    with mcp_server() as (url, server):

        async def run(tools):
            assert [tool.spec.name for tool in tools] == [
                "mcp__mail__echo",
                "mcp__mail__failure",
                "mcp__mail__image",
            ]
            for value in ("one", "two"):
                result = await tools[0].invoke(json.dumps({"value": value}))
                assert result.content == [
                    {"text": "Echo:", "type": "text"},
                    {"text": value, "type": "text"},
                ]
            return completed()

        result = await run_with_mcp(
            (McpServerSpec(name="mail", transport=transport, url=url + path),), run
        )
        assert result.stop_reason == "completed"
        assert server.list_calls == 1
        assert server.calls == [("echo", {"value": "one"}), ("echo", {"value": "two"})]


async def test_headers_are_sent_to_server():
    with mcp_server() as (url, server):

        async def run(tools):
            return completed()

        await run_with_mcp(
            (
                McpServerSpec(
                    name="mail",
                    transport="http",
                    url=url + "/mcp/",
                    headers={"x-test": "present"},
                ),
            ),
            run,
        )
        assert server.headers
        assert all(headers[b"x-test"] == b"present" for headers in server.headers)


async def test_structured_only_tool_result_is_visible_to_the_model():
    data = {"count": 2, "items": ["résumé", "report"]}
    with mcp_server(text_blocks=[], structured_content=data) as (url, _):

        async def run(tools):
            result = await tools[0].invoke('{"value":"query"}')
            assert len(result.content) == 1
            assert result.content[0]["type"] == "text"
            assert json.loads(result.content[0]["text"]) == data
            return completed()

        await run_with_mcp((McpServerSpec(name="mail", transport="http", url=url + "/mcp/"),), run)


async def test_structured_result_does_not_duplicate_existing_content():
    with mcp_server(text_blocks=["Two items"], structured_content={"count": 2}) as (url, _):

        async def run(tools):
            result = await tools[0].invoke('{"value":"query"}')
            assert result.content == [{"text": "Two items", "type": "text"}]
            return completed()

        await run_with_mcp((McpServerSpec(name="mail", transport="http", url=url + "/mcp/"),), run)


async def test_discovery_retries_after_transient_failure():
    with mcp_server(discovery_failures=1) as (url, server):

        async def run(tools):
            assert tools[0].spec.name == "mcp__mail__echo"
            return completed()

        await run_with_mcp((McpServerSpec(name="mail", transport="http", url=url + "/mcp/"),), run)
        assert server.list_calls == 2


async def test_discovery_stops_after_three_failures():
    with mcp_server(discovery_failures=3) as (url, server):

        async def run(tools):
            pytest.fail("Agent must not start when tool discovery fails")

        with pytest.raises(McpError, match="discovery temporarily unavailable"):
            await run_with_mcp(
                (McpServerSpec(name="mail", transport="http", url=url + "/mcp/"),), run
            )

        assert server.list_calls == 3


async def test_failed_response_does_not_replay_committed_tool(tmp_path):
    marker = tmp_path / "sent-messages.txt"
    with mcp_server(fail_response_after_write=marker) as (url, _):

        async def run(tools):
            await tools[0].invoke('{"value": "sent once"}')
            pytest.fail("An HTTP failure must fail the run")

        with pytest.raises(MCPTransportError):
            await run_with_mcp(
                (McpServerSpec(name="mail", transport="http", url=url + "/mcp/"),), run
            )

        assert marker.read_text() == "sent once\n"


async def test_discovered_schema_remains_non_strict():
    with mcp_server() as (url, _):

        async def run(tools):
            assert tools[0].spec.strict is False
            assert tools[0].spec.parameters == {
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"],
            }
            assert tools[1].spec.parameters == {"type": "object", "properties": {}}
            return completed()

        await run_with_mcp((McpServerSpec(name="mail", transport="http", url=url + "/mcp/"),), run)


async def test_cancelling_http_tool_call_closes_connections():
    with mcp_server(call_delay=60) as (url, server):

        async def run(tools):
            await tools[0].invoke('{"value": "hello"}')
            return completed()

        servers = tuple(
            McpServerSpec(name=name, transport="http", url=url + "/mcp/")
            for name in ("first", "second")
        )
        existing_tasks = asyncio.all_tasks()
        task = asyncio.create_task(run_with_mcp(servers, run))
        try:
            async with asyncio.timeout(5):
                while not server.calls:
                    await asyncio.sleep(0.01)
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        assert server.calls == [("echo", {"value": "hello"})]
        assert asyncio.all_tasks() <= existing_tasks


async def test_colliding_long_and_reserved_names_dispatch_to_original_tools():
    original_names = ("send.email", "send/email", "read_" + "x" * 80, "echo")
    reserved = frozenset({"mcp__mail__echo"})
    with mcp_server(tool_names=original_names) as (url, server):

        async def run(tools):
            names = [tool.spec.name for tool in tools]
            assert len(set(names)) == len(original_names)
            assert all(0 < len(name) <= 64 for name in names)
            assert all(re.fullmatch(r"[A-Za-z0-9_-]+", name) for name in names)
            assert not reserved.intersection(names)
            for tool in tools:
                # Descriptions identify the original discovery entry without depending on hash spelling.
                result = await tool.invoke(json.dumps({"value": tool.spec.description}))
                assert result.content == [
                    {"text": "Echo:", "type": "text"},
                    {"text": tool.spec.description, "type": "text"},
                ]
            return completed()

        await run_with_mcp(
            (McpServerSpec(name="mail", transport="http", url=url + "/mcp/"),),
            run,
            reserved_names=reserved,
        )
        assert server.calls == [(name, {"value": name}) for name in original_names]


def test_stdio_spec_requires_command_and_rejects_http_fields() -> None:
    with pytest.raises(ValidationError, match="command is required"):
        McpServerSpec(name="local", transport="stdio")
    with pytest.raises(ValidationError, match="url is not used"):
        McpServerSpec(name="local", transport="stdio", command="mcp", url="http://invalid")
    with pytest.raises(ValidationError, match="headers are not used"):
        McpServerSpec(
            name="local",
            transport="stdio",
            command="mcp",
            headers={"x-test": "no"},
        )


def test_http_spec_requires_url_and_rejects_stdio_fields() -> None:
    with pytest.raises(ValidationError, match="url is required"):
        McpServerSpec(name="mail", transport="http")
    with pytest.raises(ValidationError, match="command and args are not used"):
        McpServerSpec(name="mail", transport="http", url="http://mail/mcp/", command="mcp")


async def test_stdio_executes_discovered_tools() -> None:
    async def run(tools):
        assert [tool.spec.name for tool in tools] == ["mcp__mail__echo"]
        result = await tools[0].invoke(json.dumps({"value": "hello"}))
        assert result.content == [{"text": "hello", "type": "text"}]
        return completed()

    result = await run_with_mcp(
        (
            McpServerSpec(
                name="mail",
                transport="stdio",
                command=sys.executable,
                args=(str(_STDIO_SERVER),),
            ),
        ),
        run,
    )
    assert result.stop_reason == "completed"
