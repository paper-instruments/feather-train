"""Trace real loop transitions through run_e2b, without sending logs to Braintrust."""

import asyncio
import json
from contextlib import AbstractAsyncContextManager
from contextvars import ContextVar

import httpx
import pytest
from feather_harness import AgentRunError, ChatCompletionsModel, CompactionConfig
from feather_harness.tools import ToolResult
from feather_harness.tools.todo import TodoList
from openai import AsyncOpenAI

from feather_train import E2BRuntime, run_e2b
from feather_train.observability.braintrust import BraintrustRollout

current_span = ContextVar("test_span", default=None)


@pytest.fixture(autouse=True)
def braintrust_context(monkeypatch):
    monkeypatch.setattr(
        "feather_train.observability.braintrust.braintrust.current_span", current_span.get
    )


class Span:
    def __init__(self, name="rollout", parent=None):
        self.name = name
        self.parent = parent
        self.children = []
        self.closed = False
        self.root_span_id = "test"
        self.logged = {}

    def export(self):
        return "test"

    def permalink(self):
        return "https://example.invalid/test"

    def log(self, **kwargs):
        self.logged.update(kwargs)

    def start_span(self, *, name, **kwargs):
        span = Span(name, self)
        span.logged.update(kwargs)
        self.children.append(span)
        return span

    def __enter__(self):
        self.owner = asyncio.current_task()
        self.token = current_span.set(self)
        return self

    def __exit__(self, *exc):
        assert asyncio.current_task() is self.owner
        assert not self.closed
        self.closed = True
        current_span.reset(self.token)


class ToolRuntime(AbstractAsyncContextManager):
    """Controlled tool completion for span tests that do not test remote execution."""

    async def run(self, name, arguments):
        return ToolResult(content="Created answer.")

    async def save_transcript(self, content):
        pass

    async def __aexit__(self, *exc):
        pass


def completion(content="done", *, write=False):
    message = {"role": "assistant", "content": content}
    if write:
        message["tool_calls"] = [
            {
                "id": "write",
                "type": "function",
                "function": {
                    "name": "write",
                    "arguments": '{"file_path":"answer", "content":"yes"}',
                },
            }
        ]
    return httpx.Response(
        200,
        json={
            "id": "test",
            "object": "chat.completion",
            "created": 0,
            "model": "test",
            "choices": [
                {"index": 0, "message": message, "finish_reason": "tool_calls" if write else "stop"}
            ],
            "usage": {"prompt_tokens": 100, "completion_tokens": 5, "total_tokens": 105},
        },
    )


@pytest.mark.parametrize("compactions", [0, 1, 2])
async def test_context_windows_and_summaries_are_ordered_siblings(sandbox, compactions):
    root = Span()
    rollout = BraintrustRollout(root)
    calls = []
    tool_windows = []
    turns = 0

    class Transport(E2BRuntime):
        async def run(self, name, arguments):
            if name == "write":
                span = current_span.get()
                assert span.name == "write"
                tool_windows.append(span.parent.name)
            return await super().run(name, arguments)

    def respond(request):
        nonlocal turns
        body = json.loads(request.content)
        span = current_span.get()
        assert span.parent is root
        # Every earlier sibling must have closed before the next one starts.
        assert all(s.closed for s in root.children[:-1])
        calls.append((span.name, body["messages"]))
        if span.name.startswith("compaction_"):
            return completion(f"summary {turns // 2}")
        turns += 1
        return completion(write=turns <= 2 * compactions + 1)

    try:
        async with AsyncOpenAI(
            api_key="test",
            max_retries=0,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
        ) as client:
            # MCP also executes in a child task: context tokens must stay with that task.
            result = await asyncio.create_task(
                run_e2b(
                    "solve",
                    model=ChatCompletionsModel(client=client, model="test"),
                    rollout=rollout,
                    transport=Transport(
                        sandbox, container=sandbox.container, host_runtime=sandbox.host_runtime
                    ),
                    tools=("write",),
                    compaction=CompactionConfig(
                        enabled=True, context_window_tokens=30001, min_turns_before_compaction=2
                    ),
                )
            )
        expected = ["context_window_1"]
        for n in range(1, compactions + 1):
            expected.extend([f"compaction_{n}", f"context_window_{n + 1}"])
        assert result.compactions == compactions
        assert [span.name for span in root.children] == expected
        assert [name for name, _ in calls] == [
            name for name in expected for _ in range(2 if name.startswith("context") else 1)
        ]
        assert tool_windows == [
            f"context_window_{n}" for n in range(1, compactions + 1) for _ in range(2)
        ] + [f"context_window_{compactions + 1}"]
        tool_spans = [tool for window in root.children for tool in window.children]
        assert len(tool_spans) == len(tool_windows)
        assert all(tool.closed for tool in tool_spans)
        for tool in tool_spans:
            assert tool.logged["input"] == {"file_path": "answer", "content": "yes"}
            assert tool.logged["output"]
            assert tool.logged["error"] is None
        for n in range(1, compactions + 1):
            assert any(f"summary {n}" in (m.get("content") or "") for m in calls[3 * n][1])
        assert all(span.closed for span in root.children)
        assert current_span.get() is root
        with rollout.child_span(name="judge") as judge:
            assert judge.parent is root
    finally:
        rollout.close()
    assert current_span.get() is None


@pytest.mark.parametrize("failure", ["request", "empty_summary", "cancelled"])
async def test_failed_compaction_closes_spans_without_opening_another_window(failure):
    root = Span()
    rollout = BraintrustRollout(root)
    summarizing = asyncio.Event()

    async def respond(request):
        if current_span.get().name == "context_window_1":
            return completion(write=True)
        summarizing.set()
        if failure == "cancelled":
            await asyncio.Event().wait()
        if failure == "request":
            return httpx.Response(400, json={"error": {"message": "summary failed"}})
        return completion("")

    try:
        async with AsyncOpenAI(
            api_key="test",
            max_retries=0,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
        ) as client:
            task = asyncio.create_task(
                run_e2b(
                    "solve",
                    model=ChatCompletionsModel(client=client, model="test"),
                    rollout=rollout,
                    transport=ToolRuntime(),
                    tools=("write",),
                    compaction=CompactionConfig(
                        enabled=True, context_window_tokens=30001, min_turns_before_compaction=1
                    ),
                )
            )
            if failure == "cancelled":
                async with asyncio.timeout(10):
                    await summarizing.wait()
                task.cancel()
            with pytest.raises(asyncio.CancelledError if failure == "cancelled" else AgentRunError):
                await task
        attempts = 4 if failure == "empty_summary" else 1
        assert [span.name for span in root.children] == [
            "context_window_1", *(["compaction_1"] * attempts)
        ]
        assert all(span.closed for span in root.children)
        assert current_span.get() is root
    finally:
        rollout.close()
    assert current_span.get() is None


async def test_initial_prompt_can_compact_before_the_first_agent_call():
    root = Span()
    rollout = BraintrustRollout(root)

    def respond(request):
        return completion("summary" if current_span.get().name == "compaction_1" else "done")

    try:
        async with AsyncOpenAI(
            api_key="test",
            max_retries=0,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
        ) as client:
            result = await run_e2b(
                "solve",
                model=ChatCompletionsModel(client=client, model="test"),
                rollout=rollout,
                transport=ToolRuntime(),
                tools=(),
                compaction=CompactionConfig(enabled=True, context_window_tokens=30001),
            )
        assert result.compactions == 1
        assert [span.name for span in root.children] == [
            "context_window_1",
            "compaction_1",
            "context_window_2",
        ]
        assert all(span.closed for span in root.children)
        assert current_span.get() is root
    finally:
        rollout.close()


async def test_agent_cancellation_closes_the_active_window():
    root = Span()
    rollout = BraintrustRollout(root)
    generating = asyncio.Event()

    async def respond(request):
        generating.set()
        await asyncio.Event().wait()

    try:
        async with AsyncOpenAI(
            api_key="test",
            max_retries=0,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
        ) as client:
            task = asyncio.create_task(
                run_e2b(
                    "solve",
                    model=ChatCompletionsModel(client=client, model="test"),
                    rollout=rollout,
                    transport=ToolRuntime(),
                    tools=(),
                )
            )
            async with asyncio.timeout(10):
                await generating.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        assert [span.name for span in root.children] == ["context_window_1"]
        assert root.children[0].closed
        assert current_span.get() is root
    finally:
        rollout.close()


@pytest.mark.parametrize("failure", ["exception", "cancelled"])
async def test_tool_failure_closes_tool_and_context_spans(failure):
    root = Span()
    rollout = BraintrustRollout(root)
    executing = asyncio.Event()

    class Transport(ToolRuntime):
        async def run(self, name, arguments):
            executing.set()
            if failure == "cancelled":
                await asyncio.Event().wait()
            raise RuntimeError("worker unavailable")

    try:
        async with AsyncOpenAI(
            api_key="test",
            max_retries=0,
            http_client=httpx.AsyncClient(
                transport=httpx.MockTransport(lambda request: completion(write=True))
            ),
        ) as client:
            task = asyncio.create_task(
                run_e2b(
                    "solve",
                    model=ChatCompletionsModel(client=client, model="test"),
                    rollout=rollout,
                    transport=Transport(),
                    tools=("write",),
                )
            )
            if failure == "cancelled":
                async with asyncio.timeout(10):
                    await executing.wait()
                task.cancel()
            with pytest.raises(asyncio.CancelledError if failure == "cancelled" else AgentRunError):
                await task
        (window,) = root.children
        (tool,) = window.children
        assert tool.name == "write"
        assert tool.closed and window.closed
        assert current_span.get() is root
    finally:
        rollout.close()


async def test_malformed_tool_arguments_are_logged_without_masking_tool_error():
    root = Span()
    rollout = BraintrustRollout(root)
    try:
        result = await rollout.wrap_tool(TodoList().tool()).invoke('{"todos":')
        (tool,) = root.children
        assert result.error == "invalid_arguments"
        assert tool.logged["input"] == '{"todos":'
        assert tool.logged["output"] == result.content
        assert tool.logged["error"] == result.error
        assert tool.closed
    finally:
        rollout.close()
