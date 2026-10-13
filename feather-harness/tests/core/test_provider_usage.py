from __future__ import annotations

import json

import httpx
import pytest
from openai import AsyncOpenAI

from feather_harness import (
    AgentRunError,
    AssistantMessage,
    ChatCompletionsModel,
    CompactionConfig,
    FeatherAgent,
    ModelSettings,
    ProviderUsage,
    ToolResult,
    UserMessage,
    summarize_usage,
)
from feather_harness.core.model import ModelCall

from ..helpers import request, tool, work_message

# OpenRouter's usage object for a BYOK call, as returned on the final chunk or the response.
BYOK_USAGE = {
    "prompt_tokens": 50000,
    "completion_tokens": 9000,
    "total_tokens": 59000,
    "prompt_tokens_details": {"cached_tokens": 40000, "cache_write_tokens": 0},
    "completion_tokens_details": {"reasoning_tokens": 8000},
    "cost": 0.01,
    "is_byok": True,
    "cost_details": {"upstream_inference_cost": 0.2},
}


async def _complete(respond, *, stream: bool, replay_reasoning: bool = True) -> ModelCall:
    async with AsyncOpenAI(
        api_key="test",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
    ) as client:
        model = ChatCompletionsModel(
            client=client, model="model", stream=stream, replay_reasoning=replay_reasoning
        )
        return await model.complete([UserMessage(content="task")], settings=ModelSettings())


@pytest.mark.parametrize("stream", [False, True])
async def test_billed_usage_is_kept_verbatim_with_byok_upstream_cost(stream):
    message = {"role": "assistant", "content": "done"}

    async def respond(req):
        common = {"id": "gen-1", "created": 0, "model": "model", "provider": "Azure"}
        if not stream:
            return httpx.Response(
                200,
                json={
                    **common,
                    "object": "chat.completion",
                    "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
                    "usage": BYOK_USAGE,
                },
            )
        chunks = [
            {"choices": [{"index": 0, "delta": message, "finish_reason": "stop"}]},
            {"choices": [], "usage": BYOK_USAGE},
        ]
        body = "".join(
            f"data: {json.dumps({**common, 'object': 'chat.completion.chunk', **chunk})}\n\n"
            for chunk in chunks
        )
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=(body + "data: [DONE]\n\n").encode(),
        )

    usage = (await _complete(respond, stream=stream)).usage

    assert (usage.response_id, usage.provider, usage.kind) == ("gen-1", "Azure", "agent")
    assert (usage.prompt_tokens, usage.cached_tokens, usage.completion_tokens) == (
        50000,
        40000,
        9000,
    )
    assert usage.reasoning_tokens == 8000
    assert usage.cost_usd == pytest.approx(0.21)
    # The per-call row stored with each trace must keep the BYOK upstream charge.
    row = usage.accounting_row()
    assert (row["response_id"], row["is_byok"], row["upstream_cost_usd"]) == ("gen-1", True, 0.2)
    assert row["cost_usd"] == pytest.approx(0.21)


def test_missing_fields_stay_unknown_instead_of_zero():
    reported = ProviderUsage(
        kind="agent",
        usage={"prompt_tokens": 10, "completion_tokens": 2, "cost": 0.5, "is_byok": False},
    )
    silent = ProviderUsage(kind="agent")
    upstream_without_byok = ProviderUsage(
        kind="compaction",
        usage={
            "prompt_tokens": 5,
            "completion_tokens": 1,
            "cost": 0.25,
            "cost_details": {"upstream_inference_cost": 9.0},
        },
    )

    totals = summarize_usage([reported, silent, upstream_without_byok])

    assert (totals.model_calls, totals.calls_without_usage) == (3, 1)
    assert (totals.prompt_tokens, totals.completion_tokens) == (15, 3)
    assert totals.reasoning_tokens is None
    assert totals.cost_usd == pytest.approx(0.75)


async def test_replayed_reasoning_text_is_not_counted_as_hidden():
    async def respond(req):
        return httpx.Response(
            200,
            json={
                "id": "gen-1",
                "object": "chat.completion",
                "created": 0,
                "model": "model",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok", "reasoning": "r" * 300},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 1200,
                    "completion_tokens_details": {"reasoning_tokens": 1000},
                },
            },
        )

    assert (await _complete(respond, stream=False)).unreplayed_reasoning_tokens == 900
    unreplayed = await _complete(respond, stream=False, replay_reasoning=False)
    assert unreplayed.unreplayed_reasoning_tokens == 1000


class _BilledModel:
    """Scripted model whose calls each bill reasoning that never returns to context."""

    def __init__(self, *messages):
        self.messages = iter(messages)
        self.number = 0

    async def complete(self, context, *, settings, tools=(), kind="agent"):
        self.number += 1
        return ModelCall(
            assistant=next(self.messages),
            input_tokens=1000,
            output_tokens=20000,
            unreplayed_reasoning_tokens=19000,
            usage=ProviderUsage(
                kind=kind,
                response_id=f"gen-{self.number}",
                usage={"prompt_tokens": 1000, "completion_tokens": 20000},
            ),
        )


async def _work(arguments):
    return ToolResult(content="ok")


@pytest.mark.parametrize("exclude, compactions", [(False, 1), (True, 0)])
async def test_hidden_reasoning_triggers_compaction_only_when_counted(exclude, compactions):
    # Threshold is min(85% of 40000, 40000 - 30000) = 10000 tokens of context.
    config = CompactionConfig(
        enabled=True, context_window_tokens=40000, exclude_unreplayed_reasoning=exclude
    )
    model = _BilledModel(
        work_message(),
        *([AssistantMessage(content="summary")] if compactions else []),
        AssistantMessage(content="done"),
    )

    result = await FeatherAgent(model=model, tools=(tool(_work),), compaction=config).run(
        request()
    )

    assert result.compactions == compactions
    # Every call is accounted to the segment whose assistant message it produced.
    assert [
        (segment.kind, [call.response_id for call in segment.calls])
        for segment in result.segments
    ] == (
        [
            ("execution", ["gen-1"]),
            ("compaction", ["gen-2"]),
            ("execution", ["gen-3"]),
        ]
        if compactions
        else [("execution", ["gen-1", "gen-2"])]
    )
    assert [call.kind for segment in result.segments for call in segment.calls] == [
        "agent",
        *(["compaction"] if compactions else []),
        "agent",
    ]
    assert result.usage().completion_tokens == 20000 * (2 + compactions)


async def test_failed_compaction_keeps_one_record_per_attempt_for_trace_export():
    config = CompactionConfig(enabled=True, context_window_tokens=40000)
    model = _BilledModel(work_message(), *[AssistantMessage(content="") for _ in range(4)])

    with pytest.raises(AgentRunError) as failure:
        await FeatherAgent(model=model, tools=(tool(_work),), compaction=config).run(request())

    # ATIF attaches usage by position, so every produced message needs its own call record.
    segments = failure.value.result.segments
    assert [(segment.kind, len(segment.messages), len(segment.calls)) for segment in segments] == [
        ("execution", 2, 1),
        *[("compaction", 1, 1)] * 4,
    ]
    assert failure.value.result.usage().model_calls == 5
