"""Production recovery from cut-off or empty responses, observed through real request bodies."""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager

import httpx
import pytest
from openai import APIStatusError, AsyncOpenAI

from feather_harness import (
    AgentRunError,
    ChatCompletionsModel,
    FeatherAgent,
    RetryPolicy,
    RunRequest,
    Tool,
    ToolResult,
    ToolSpec,
    UserMessage,
)
from feather_harness.core.compaction import CompactionConfig
from feather_harness.core.messages import AssistantMessage
from feather_harness.core.runner import RECOVERY_FEEDBACK

# A minimal window compacts before every execution turn after the first.
TINY_WINDOW = CompactionConfig(enabled=True, context_window_tokens=30001, keep_recent_turns=0)


def reply(content=None, *, finish="stop", tools=(), reasoning=None, refusal=None):
    message = {"role": "assistant", "content": content, "refusal": refusal}
    if reasoning is not None:
        message["reasoning_content"] = reasoning
    if tools:
        message["tool_calls"] = [
            {
                "id": f"call-{index}",
                "type": "function",
                "function": {"name": name, "arguments": args},
            }
            for index, (name, args) in enumerate(tools)
        ]
    return {
        "id": "test",
        "object": "chat.completion",
        "created": 0,
        "model": "test",
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


@asynccontextmanager
async def scripted(*replies):
    requests = []
    remaining = iter(replies)

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=next(remaining))

    async with AsyncOpenAI(
        api_key="test",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
    ) as client:
        yield ChatCompletionsModel(client=client, model="test"), requests


def counting_tool():
    calls = []

    async def invoke(arguments):
        calls.append(arguments)
        return ToolResult(content="ok")

    spec = ToolSpec(name="work", description="work", parameters={"type": "object"})
    return Tool(spec=spec, invoke=invoke), calls


def task(max_turns=10):
    return RunRequest(messages=(UserMessage(content="task"),), max_turns=max_turns)


async def test_normal_answers_and_tool_calls_are_unchanged_by_recovery():
    sent = {}
    for recovery in (False, True):
        tool, calls = counting_tool()
        async with scripted(
            reply(finish="tool_calls", tools=(("work", "{}"),)), reply("Done.")
        ) as (model, requests):
            result = await FeatherAgent(model=model, tools=(tool,), response_recovery=recovery).run(
                task()
            )
        assert (result.stop_reason, result.final_answer, result.recoveries) == (
            "completed",
            "Done.",
            0,
        )
        assert calls == ["{}"]
        sent[recovery] = requests
    assert sent[True] == sent[False]


@pytest.mark.parametrize(
    "failure, cause, replayed",
    [
        (reply("Draft of the ans", finish="length"), "length", "Draft of the ans"),
        (reply(None, reasoning="Long private reasoning"), "empty", None),
        (reply("   ", finish="stop"), "empty", "   "),
        (reply(None, finish="error"), "error", None),
    ],
)
async def test_one_failed_response_is_recovered_with_matching_feedback(failure, cause, replayed):
    async with scripted(failure, reply("Done.")) as (model, requests):
        result = await FeatherAgent(model=model, response_recovery=True).run(task())
    assert (result.stop_reason, result.final_answer, result.recoveries) == ("completed", "Done.", 1)
    follow_up = requests[1]["messages"]
    assert follow_up[-1] == {"role": "user", "content": RECOVERY_FEEDBACK[cause]}
    if replayed is None:
        # Reasoning-only and empty responses are not resent, so the retry follows the task.
        assert follow_up[:-1] == requests[0]["messages"]
    else:
        assert follow_up[-2] == {"role": "assistant", "content": replayed}
        assert "reasoning_content" not in follow_up[-2]


async def test_second_failure_stops_even_across_compaction():
    async with scripted(
        reply("Summary one."), reply(None), reply("Summary two."), reply("Partial", finish="length")
    ) as (model, requests):
        result = await FeatherAgent(
            model=model, compaction=TINY_WINDOW, response_recovery=True
        ).run(task())
    assert result.stop_reason == "ModelBehaviorError"
    assert (
        result.stop_message == "length response (finish_reason=length) after the run's one recovery"
    )
    assert (result.compactions, result.recoveries, result.final_answer) == (2, 1, "")
    assert result.segments[-1].messages[-1] == AssistantMessage(content="Partial")
    assert len(requests) == 4


async def test_no_feedback_is_added_when_no_turn_remains():
    async with scripted(reply(None)) as (model, requests):
        result = await FeatherAgent(model=model, response_recovery=True).run(task(max_turns=1))
    assert (result.stop_reason, result.recoveries) == ("MaxTurnsExceeded", 0)
    assert not isinstance(result.segments[-1].messages[-1], UserMessage)
    assert len(requests) == 1


@pytest.mark.parametrize("finish", ["length", "error", "content_filter"])
async def test_tool_calls_from_an_interrupted_response_never_run(finish):
    tool, calls = counting_tool()
    async with scripted(reply(finish=finish, tools=(("work", '{"path": "a"}'),))) as (
        model,
        requests,
    ):
        result = await FeatherAgent(model=model, tools=(tool,), response_recovery=True).run(task())
    assert result.stop_reason == "ModelBehaviorError"
    assert result.stop_message.endswith("none of its tool calls were executed")
    assert (calls, len(requests), result.recoveries) == ([], 1, 0)


async def test_refusals_and_filtered_answers_are_never_coached():
    async with scripted(reply(None, refusal="I can't help with that.")) as (model, requests):
        with pytest.raises(AgentRunError, match="refused"):
            await FeatherAgent(model=model, response_recovery=True).run(task())
    assert len(requests) == 1

    async with scripted(reply("Partial", finish="content_filter")) as (model, requests):
        result = await FeatherAgent(model=model, response_recovery=True).run(task())
    assert result.stop_reason == "ModelBehaviorError"
    assert result.stop_message == "content_filter response (finish_reason=content_filter)"
    assert (len(requests), result.recoveries) == (1, 0)


async def test_recovery_off_keeps_empty_and_cut_off_answers_completed():
    for failure in (reply(None), reply("Partial", finish="length")):
        async with scripted(failure) as (model, requests):
            result = await FeatherAgent(model=model).run(task())
        assert result.stop_reason == "completed"
        assert (result.recoveries, len(requests)) == (0, 1)


@pytest.mark.parametrize("recovery", [False, True])
async def test_interrupted_summary_never_replaces_context_after_retry_exhaustion(recovery):
    async with scripted(*[reply("Partial summ", finish="length")] * 4, reply("Done.")) as (
        model,
        requests,
    ):
        agent = FeatherAgent(model=model, compaction=TINY_WINDOW, response_recovery=recovery)
        if recovery:
            with pytest.raises(AgentRunError, match="compaction summary is incomplete") as caught:
                await agent.run(task())
            assert caught.value.result.segments[-1].kind == "compaction"
            assert len(requests) == 4
            assert all(body == requests[0] for body in requests)
            assert caught.value.result.compactions == 0
        else:
            result = await agent.run(task())
            assert (result.final_answer, result.compactions) == ("Partial summ", 1)


@pytest.mark.parametrize("finish", ["length", "error"])
async def test_production_retries_interrupted_summary_without_reusing_partial_text(finish):
    async with scripted(
        reply("partial", finish=finish), reply("valid summary"), reply("Done.")
    ) as (model, requests):
        result = await FeatherAgent(
            model=model, compaction=TINY_WINDOW, response_recovery=True
        ).run(task())

    assert (result.final_answer, result.compactions, result.recoveries) == ("Done.", 1, 0)
    assert requests[0] == requests[1]
    assert "partial" not in json.dumps(requests[2])
    assert [s.messages[0].content for s in result.segments if s.kind == "compaction"] == [
        "partial", "valid summary"
    ]


@pytest.mark.parametrize("failure", [reply("filtered", finish="content_filter"), reply(refusal="No")])
async def test_production_does_not_retry_filtered_or_refused_compaction(failure):
    async with scripted(failure) as (model, requests):
        with pytest.raises(AgentRunError, match="summary is incomplete"):
            await FeatherAgent(
                model=model, compaction=TINY_WINDOW, response_recovery=True
            ).run(task())
    assert len(requests) == 1


@pytest.mark.parametrize("status, attempts", [(400, 1), (503, 2)])
async def test_compaction_does_not_multiply_model_transport_retries(status, attempts):
    requests = []

    def respond(request):
        requests.append(json.loads(request.content))
        return httpx.Response(status, json={"error": {"message": "upstream failure"}})

    async with AsyncOpenAI(
        api_key="test", max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
    ) as client:
        model = ChatCompletionsModel(
            client=client, model="test",
            retry=RetryPolicy(max_attempts=2, base_backoff_seconds=0),
        )
        with pytest.raises(AgentRunError) as caught:
            await FeatherAgent(
                model=model, compaction=TINY_WINDOW, response_recovery=True
            ).run(task())
    assert isinstance(caught.value.cause, APIStatusError)
    assert caught.value.cause.status_code == status
    assert len(requests) == attempts


async def test_cancellation_interrupts_compaction_without_retrying():
    started = asyncio.Event()
    requests = []

    async def respond(request):
        requests.append(request)
        started.set()
        await asyncio.Future()

    async with AsyncOpenAI(
        api_key="test", max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
    ) as client:
        model = ChatCompletionsModel(client=client, model="test", retry=RetryPolicy())
        run = asyncio.create_task(
            FeatherAgent(model=model, compaction=TINY_WINDOW, response_recovery=True).run(task())
        )
        await asyncio.wait_for(started.wait(), 2)
        run.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(run, 2)
    assert len(requests) == 1
