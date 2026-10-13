from __future__ import annotations

import json

import httpx
import pytest
from openai import AsyncOpenAI

from feather_harness import (
    AssistantMessage,
    ChatCompletionsModel,
    CompactionConfig,
    FeatherAgent,
    ModelSettings,
    ToolResult,
    UserMessage,
)

from ..helpers import ScriptedModel, request, tool, work_message

SIGNED = {"type": "reasoning.text", "text": "think", "signature": "sig", "index": 0}
ENCRYPTED = {"type": "reasoning.encrypted", "data": "opaque", "id": "rs_1", "index": 1}
SUMMARY = {"type": "reasoning.summary", "summary": "short", "index": 0}


def _sse(*deltas):
    events = [
        {"id": "s", "object": "chat.completion.chunk", "created": 0, "model": "m", **d}
        for d in deltas
    ]
    body = "".join(f"data: {json.dumps(e)}\n\n" for e in events) + "data: [DONE]\n\n"
    return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body.encode())


def _delta(finish_reason=None, **delta):
    return {"choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}]}


async def _two_turns(first, *, stream):
    """Return the first assistant message and the second request's replayed assistant message."""
    requests = []

    async def respond(req):
        requests.append(json.loads(req.content))
        if len(requests) > 1 and stream:
            return _sse(_delta(role="assistant", content="done", finish_reason="stop"))
        if len(requests) > 1:
            return httpx.Response(200, json={
                "id": "2", "object": "chat.completion", "created": 0, "model": "m",
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": "done"}}],
            })
        return first()

    async with AsyncOpenAI(
        api_key="k", max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
    ) as client:
        model = ChatCompletionsModel(client=client, model="m", stream=stream)
        call = await model.complete([UserMessage(content="task")], settings=ModelSettings())
        await model.complete(
            [UserMessage(content="task"), call.assistant, UserMessage(content="next")],
            settings=ModelSettings(),
        )
    return call.assistant, requests[1]["messages"][1]


@pytest.mark.parametrize("stream", [False, True])
async def test_signed_and_encrypted_reasoning_is_replayed_verbatim(stream):
    def first():
        if not stream:
            return httpx.Response(200, json={
                "id": "1", "object": "chat.completion", "created": 0, "model": "m",
                "choices": [{"index": 0, "finish_reason": "stop", "message": {
                    "role": "assistant", "content": "ok", "reasoning": "short",
                    "reasoning_details": [SUMMARY, ENCRYPTED],
                }}],
            })
        # Streamed summaries arrive in pieces; encrypted data and ids arrive whole.
        return _sse(
            _delta(role="assistant", reasoning="sh", reasoning_details=[{**SUMMARY, "summary": "sh"}]),
            _delta(reasoning="ort", reasoning_details=[{**SUMMARY, "summary": "ort"}]),
            _delta(reasoning_details=[ENCRYPTED]),
            _delta(content="ok", finish_reason="stop"),
        )

    assistant, replayed = await _two_turns(first, stream=stream)

    assert assistant.reasoning_details == (SUMMARY, ENCRYPTED)
    assert replayed["reasoning_details"] == [SUMMARY, ENCRYPTED]
    # Plaintext reasoning is not sent alongside the blocks it summarizes.
    assert "reasoning_content" not in replayed


async def test_streamed_signature_completes_the_signed_block():
    def first():
        return _sse(
            _delta(role="assistant", reasoning_details=[{**SIGNED, "text": "thi"}]),
            _delta(reasoning_details=[{"type": "reasoning.text", "text": "nk", "index": 0}]),
            _delta(reasoning_details=[{"type": "reasoning.text", "signature": "sig", "index": 0}]),
            _delta(content="ok", finish_reason="stop"),
        )

    assistant, replayed = await _two_turns(first, stream=True)

    assert replayed["reasoning_details"] == [SIGNED]


async def test_plaintext_only_reasoning_keeps_the_existing_replay():
    unsigned = {"type": "reasoning.text", "text": "think", "index": 0}

    def first():
        return httpx.Response(200, json={
            "id": "1", "object": "chat.completion", "created": 0, "model": "m",
            "choices": [{"index": 0, "finish_reason": "stop", "message": {
                "role": "assistant", "content": "ok", "reasoning": "think",
                "reasoning_details": [unsigned],
            }}],
        })

    assistant, replayed = await _two_turns(first, stream=False)

    assert assistant.reasoning_details == ()
    assert replayed["reasoning_content"] == "think"
    assert "reasoning_details" not in replayed


async def test_preserved_reasoning_does_not_cross_a_compaction():
    async def work(arguments):
        return ToolResult(content="ok")

    signed_turn = work_message().model_copy(update={"reasoning_details": (SIGNED,)})
    model = ScriptedModel(
        signed_turn,
        AssistantMessage(content="summary"),
        AssistantMessage(content="done"),
        tokens=120000,
    )
    config = CompactionConfig(enabled=True, context_window_tokens=131072, keep_recent_turns=2)

    await FeatherAgent(model=model, tools=(tool(work),), compaction=config).run(request())

    assert [call.kind for call in model.calls] == ["agent", "compaction", "agent"]
    for call in model.calls[1:]:
        assert all(
            not message.get("reasoning_details") for message in call.request["messages"]
        ), call.kind
    # The kept recent turn is still present, only without its signed block.
    assert any(
        message.get("tool_calls") for message in model.calls[2].request["messages"]
    )
