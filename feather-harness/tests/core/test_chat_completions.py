from __future__ import annotations

import json

import httpx
import pytest
from openai import (
    APIConnectionError,
    APIError,
    APIResponseValidationError,
    APIStatusError,
    AsyncOpenAI,
)

from feather_harness import (
    AgentRunError,
    ChatCompletionsModel,
    FeatherAgent,
    ModelSettings,
    RetryPolicy,
    RunRequest,
    Tool,
    ToolMessage,
    ToolResult,
    ToolSpec,
    UserMessage,
)


async def test_chat_settings_reasoning_alias_and_provider_response():
    requests = []

    async def respond(req):
        requests.append(json.loads(req.content))
        return httpx.Response(
            200,
            json={
                "id": "response-1",
                "object": "chat.completion",
                "created": 0,
                "model": "model",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "done", "reasoning": "alias"},
                        "finish_reason": "stop",
                        "logprobs": {"content": []},
                    }
                ],
                "usage": {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15},
            },
        )

    async with AsyncOpenAI(
        api_key="test",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
    ) as client:
        result = await ChatCompletionsModel(client=client, model="model").complete(
            [UserMessage(content="hello")],
            settings=ModelSettings(
                temperature=0, parallel_tool_calls=False, top_logprobs=0, extra_body={"seed": 7}
            ),
        )
    assert requests[0]["temperature"] == 0
    assert requests[0]["parallel_tool_calls"] is False
    assert requests[0]["logprobs"] is True
    assert requests[0]["seed"] == 7
    assert not (set(requests[0]) & {"stream", "tool_choice", "tools"})
    assert result.assistant.reasoning_content == "alias"
    assert requests[0]["top_logprobs"] == 0
    assert (result.input_tokens, result.output_tokens) == (12, 3)


@pytest.mark.parametrize(
    "parallel, with_tools, expected",
    [
        (None, True, None),
        (False, False, False),
        (True, False, None),
        (True, True, True),
    ],
)
async def test_parallel_tool_option_only_changes_the_model_request(parallel, with_tools, expected):
    requests = []

    async def respond(req):
        requests.append(json.loads(req.content))
        return httpx.Response(
            200,
            json={
                "id": "r",
                "object": "chat.completion",
                "created": 0,
                "model": "model",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "done"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    async with AsyncOpenAI(
        api_key="test",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
    ) as client:
        await ChatCompletionsModel(client=client, model="model").complete(
            [UserMessage(content="task")],
            tools=(ToolSpec(name="work", description="work", parameters={}),) if with_tools else (),
            settings=ModelSettings(parallel_tool_calls=parallel),
        )
    wire = requests[0]
    if expected is None:
        assert "parallel_tool_calls" not in wire
    else:
        assert wire["parallel_tool_calls"] is expected
    assert ("tools" in wire) == with_tools
    assert all(spec["type"] == "function" for spec in wire.get("tools", []))
    assert all(spec["function"]["strict"] is False for spec in wire.get("tools", []))


@pytest.mark.parametrize("failure", ["connection", "empty_choices", "provider_error"])
async def test_provider_failure_preserves_completed_tool_work_without_replaying_it(failure):
    requests = []
    edits = []
    connection_error = httpx.ConnectError("provider unavailable")

    async def edit(arguments):
        edits.append(arguments)
        return ToolResult(content="File updated.")

    async def respond(req):
        requests.append(json.loads(req.content))
        if len(requests) > 1:
            if failure == "connection":
                raise connection_error
            payload = {"choices": []}
            if failure == "provider_error":
                payload = {"choices": None, "error": {"message": "provider unavailable"}}
            return httpx.Response(200, json=payload)
        return httpx.Response(
            200,
            json={
                "id": "r",
                "object": "chat.completion",
                "created": 0,
                "model": "model",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "edit-1",
                                    "type": "function",
                                    "function": {
                                        "name": "edit",
                                        "arguments": '{"file":"notes.txt"}',
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
            },
        )

    async with AsyncOpenAI(
        api_key="test",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
    ) as client:
        agent = FeatherAgent(
            model=ChatCompletionsModel(client=client, model="model"),
            tools=(
                Tool(
                    spec=ToolSpec(name="edit", description="Edit a file", parameters={}),
                    invoke=edit,
                ),
            ),
        )
        with pytest.raises(AgentRunError) as raised:
            await agent.run(RunRequest(messages=(UserMessage(content="Update notes.txt"),)))

    result = raised.value.result
    if failure == "connection":
        assert isinstance(raised.value.cause, APIConnectionError)
        assert raised.value.cause.__cause__ is connection_error
    else:
        assert isinstance(raised.value.cause, RuntimeError)
        assert "response has no choices" in str(raised.value.cause)
        if failure == "provider_error":
            assert "provider unavailable" in str(raised.value.cause)
    assert edits == ['{"file":"notes.txt"}']
    assert len(requests) == 2
    assert result.stop_reason == "failed"
    assert result.num_turns == 1
    assert len(result.execution_messages()) == 3
    assert result.execution_messages()[-1] == ToolMessage(
        tool_call_id="edit-1", content="File updated."
    )
    assert requests[1]["messages"][-1] == {
        "role": "tool",
        "tool_call_id": "edit-1",
        "content": "File updated.",
    }


def _sse(*chunks):
    events = [
        {"id": "s", "object": "chat.completion.chunk", "created": 0, "model": "model", **c}
        for c in chunks
    ]
    body = "".join(f"data: {json.dumps(event)}\n\n" for event in events) + "data: [DONE]\n\n"
    return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body.encode())


def _delta(finish_reason=None, **delta):
    return {"choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}]}


async def test_streamed_response_is_assembled_like_a_nonstreamed_one():
    requests = []

    async def respond(req):
        requests.append(json.loads(req.content))
        return _sse(
            _delta(role="assistant", reasoning="think "),
            _delta(reasoning="more", content="Editing"),
            _delta(
                tool_calls=[
                    {
                        "index": 0,
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "edit", "arguments": '{"file":'},
                    }
                ]
            ),
            _delta(tool_calls=[{"index": 0, "function": {"arguments": '"a.txt"}'}}]),
            _delta(finish_reason="tool_calls"),
            {
                "choices": [],
                "usage": {"prompt_tokens": 9, "completion_tokens": 4, "total_tokens": 13},
            },
        )

    async with AsyncOpenAI(
        api_key="test",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
    ) as client:
        result = await ChatCompletionsModel(client=client, model="model", stream=True).complete(
            [UserMessage(content="task")], settings=ModelSettings()
        )
    assert requests[0]["stream"] is True
    assert requests[0]["stream_options"] == {"include_usage": True}
    assert result.assistant.content == "Editing"
    assert result.assistant.reasoning_content == "think more"
    assert [(c.id, c.name, c.arguments) for c in result.assistant.tool_calls] == [
        ("c1", "edit", '{"file":"a.txt"}')
    ]
    assert (result.finish_reason, result.input_tokens, result.output_tokens) == ("tool_calls", 9, 4)


async def test_transient_failures_are_retried_and_client_errors_are_not():
    replies = [
        httpx.Response(
            503, headers={"retry-after": "0"}, json={"error": {"message": "scaling up"}}
        ),
        _sse(_delta(content="cut off")),
        _sse(_delta(content="done", finish_reason="stop")),
        httpx.Response(400, json={"error": {"message": "bad request"}}),
    ]
    requests = []

    async def respond(req):
        requests.append(req)
        return replies[len(requests) - 1]

    async with AsyncOpenAI(
        api_key="test",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
    ) as client:
        model = ChatCompletionsModel(
            client=client, model="model", stream=True, retry=RetryPolicy(base_backoff_seconds=0)
        )
        result = await model.complete([UserMessage(content="task")], settings=ModelSettings())
        assert result.assistant.content == "done"
        assert len(requests) == 3
        with pytest.raises(APIStatusError):
            await model.complete([UserMessage(content="task")], settings=ModelSettings())
    assert len(requests) == 4


async def test_stream_ending_in_provider_error_is_retried():
    """OpenRouter's finish_reason "error" once aborted a 475-rubric eval run (2026-10-10)."""
    replies = [
        _sse(_delta(content="partial"), _delta(finish_reason="error")),
        _sse(_delta(content="done", finish_reason="stop")),
    ]
    requests = []

    async def respond(req):
        requests.append(req)
        return replies[len(requests) - 1]

    async with AsyncOpenAI(
        api_key="test",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
    ) as client:
        model = ChatCompletionsModel(
            client=client, model="model", stream=True,
            retry=RetryPolicy(max_attempts=2, base_backoff_seconds=0, max_backoff_seconds=0),
        )
        result = await model.complete([UserMessage(content="task")], settings=ModelSettings())
    assert len(requests) == 2
    assert (result.assistant.content, result.finish_reason) == ("done", "stop")


@pytest.mark.parametrize("kind", ["agent", "compaction"])
@pytest.mark.parametrize("error", [
    {"type": "api_error", "code": "upstream_error", "message": "The model stream was interrupted."},
    {"code": 502, "message": "Network connection lost."},
    {"message": "Network connection lost."},
])
async def test_in_stream_api_errors_discard_partial_output_before_retry(error, kind):
    requests = []
    replies = [
        _sse(_delta(content="partial", reasoning="discard this"), {"error": error}),
        _sse(_delta(content="done", finish_reason="stop")),
    ]

    async def respond(request):
        requests.append(json.loads(request.content))
        return replies[len(requests) - 1]

    async with AsyncOpenAI(
        api_key="test", max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
    ) as client:
        result = await ChatCompletionsModel(
            client=client, model="model", stream=True,
            retry=RetryPolicy(max_attempts=2, base_backoff_seconds=0),
        ).complete([UserMessage(content="task")], settings=ModelSettings(), kind=kind)

    assert requests == [requests[0]] * 2
    assert result.assistant.content == "done"
    assert result.assistant.reasoning_content is None
    assert all(reply.is_closed for reply in replies)


@pytest.mark.parametrize("error", [
    {"code": code, "message": "Permanent failure"} for code in [400, 401, 403, 404, 422]
] + [
    {"type": "invalid_request_error", "message": "Invalid input"},
    {"type": "authentication_error", "message": "Invalid key"},
    {"code": "context_length_exceeded", "message": "Context too long"},
    {"code": "insufficient_quota", "message": "Insufficient balance"},
    {"code": "content_policy_violation", "message": "Refused"},
])
async def test_permanent_in_stream_api_errors_are_not_retried(error):
    requests = []

    async def respond(request):
        requests.append(request)
        return _sse({"error": error})

    async with AsyncOpenAI(
        api_key="test", max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
    ) as client:
        with pytest.raises(APIError):
            await ChatCompletionsModel(
                client=client, model="model", stream=True,
                retry=RetryPolicy(max_attempts=3, base_backoff_seconds=0),
            ).complete([UserMessage(content="task")], settings=ModelSettings())
    assert len(requests) == 1


@pytest.mark.parametrize("retry, attempts", [
    (None, 1), (RetryPolicy(max_attempts=2, base_backoff_seconds=0), 2),
])
async def test_in_stream_api_errors_respect_retry_budget_and_preserve_error(retry, attempts):
    requests = []
    error = {"type": "api_error", "code": "upstream_error", "message": "Interrupted"}

    async def respond(request):
        requests.append(request)
        return _sse({"error": error})

    async with AsyncOpenAI(
        api_key="test", max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
    ) as client:
        with pytest.raises(APIError) as caught:
            await ChatCompletionsModel(
                client=client, model="model", stream=True, retry=retry,
            ).complete([UserMessage(content="task")], settings=ModelSettings())
    assert len(requests) == attempts
    assert type(caught.value) is APIError
    assert caught.value.body == error


async def test_stream_response_schema_errors_are_not_retried():
    requests = []

    async def respond(request):
        requests.append(request)
        return _sse({"choices": "invalid schema"})

    async with AsyncOpenAI(
        api_key="test", max_retries=0, _strict_response_validation=True,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
    ) as client:
        with pytest.raises(APIResponseValidationError):
            await ChatCompletionsModel(
                client=client, model="model", stream=True,
                retry=RetryPolicy(max_attempts=3, base_backoff_seconds=0),
            ).complete([UserMessage(content="task")], settings=ModelSettings())
    assert len(requests) == 1


async def test_in_stream_retry_does_not_replay_completed_or_partial_tool_calls():
    requests, edits = [], []
    replies = [
        _sse(_delta(tool_calls=[{
            "index": 0, "id": "edit-1", "type": "function",
            "function": {"name": "edit", "arguments": '{"file":"notes.txt"}'},
        }]), _delta(finish_reason="tool_calls")),
        _sse(_delta(content="partial", tool_calls=[{
            "index": 0, "id": "partial-edit", "type": "function",
            "function": {"name": "edit", "arguments": '{"file":'},
        }]), {"error": {"code": "upstream_error", "message": "Interrupted"}}),
        _sse(_delta(content="done", finish_reason="stop")),
    ]

    async def edit(arguments):
        edits.append(arguments)
        return ToolResult(content="File updated.")

    async def respond(request):
        requests.append(json.loads(request.content))
        return replies[len(requests) - 1]

    async with AsyncOpenAI(
        api_key="test", max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
    ) as client:
        result = await FeatherAgent(
            model=ChatCompletionsModel(
                client=client, model="model", stream=True,
                retry=RetryPolicy(max_attempts=2, base_backoff_seconds=0),
            ),
            tools=(Tool(
                spec=ToolSpec(name="edit", description="Edit a file", parameters={}),
                invoke=edit,
            ),),
        ).run(RunRequest(messages=(UserMessage(content="Update notes.txt"),)))
    assert result.final_answer == "done"
    assert edits == ['{"file":"notes.txt"}']
    assert len(requests) == 3
    assert requests[1] == requests[2]
    assert requests[2]["messages"][-1] == {
        "role": "tool", "tool_call_id": "edit-1", "content": "File updated.",
    }
