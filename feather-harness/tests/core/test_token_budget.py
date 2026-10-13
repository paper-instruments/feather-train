import json

import httpx
import pytest
from openai import AsyncOpenAI

from feather_harness import (
    AssistantMessage,
    ChatCompletionsModel,
    ModelSettings,
    ToolCall,
    ToolMessage,
    ToolSpec,
    UserMessage,
)
from feather_harness.core import model as model_module
from feather_harness.core import token_budget
from feather_harness.core.token_budget import PromptTokenBudget, PromptTokenBudgetError

BUDGET = PromptTokenBudget(
    processor_name="zai-org/GLM-5.3-Flash",
    processor_revision="eb9eb208eb0d988989d07a6a12d0fdeb5f52574a",
    context_window_tokens=131072,
)


@pytest.mark.parametrize("requested,kind,prompt_tokens,expected", [
    (131072, "agent", 100000, 31072),
    (65536, "compaction", 100000, 31072),
    (131072, "compaction", 100000, 31072),
    (131072, "compaction", 10000, 121072),
    (8, "agent", 100000, 8),
    (None, "agent", 100000, 31072),
])
@pytest.mark.parametrize("reasoning", [{"reasoning": {"effort": "max"}}, {"reasoning_effort": "max"}])
async def test_output_budget_clamps_final_outbound_request(
    monkeypatch, requested, kind, prompt_tokens, expected, reasoning
):
    sent = []
    counted = []

    def count(budget, messages, tools, **kwargs):
        counted.append((messages, tools, kwargs))
        return prompt_tokens

    def respond(request):
        sent.append(json.loads(request.content))
        return httpx.Response(200, json={
            "id": "r", "object": "chat.completion", "created": 0, "model": "glm",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "done"},
                         "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": 1,
                "total_tokens": prompt_tokens + 1,
            },
        })

    monkeypatch.setattr(model_module, "count_prompt_tokens", count)
    tools = (ToolSpec(name="bash", description="Run commands", parameters={"type": "object"}),)
    async with AsyncOpenAI(api_key="test", http_client=httpx.AsyncClient(
        transport=httpx.MockTransport(respond)
    )) as client:
        await ChatCompletionsModel(client=client, model="glm", prompt_token_budget=BUDGET).complete(
            [UserMessage(content="hello")], tools=tools,
            settings=ModelSettings(max_tokens=requested, extra_body=reasoning),
            kind=kind,
        )
    assert sent[0]["max_tokens"] == expected
    assert counted[0][0] == sent[0]["messages"]
    assert counted[0][1] == sent[0]["tools"]
    assert counted[0][2]["reasoning_effort"] == "max"


@pytest.mark.parametrize("prompt_tokens,requested,extra", [
    (131072, 131072, {}),
    (131073, 131072, {}),
    (100, 0, {}),
    (100, -1, {}),
    (100, 131072, {"max_tokens": 131072}),
    (100, 131072, {"max_completion_tokens": 131072}),
])
async def test_invalid_budget_fails_before_generation(monkeypatch, prompt_tokens, requested, extra):
    monkeypatch.setattr(model_module, "count_prompt_tokens", lambda *args, **kwargs: prompt_tokens)

    def unexpected(request):
        pytest.fail("An invalid token budget reached generation")

    async with AsyncOpenAI(api_key="test", http_client=httpx.AsyncClient(
        transport=httpx.MockTransport(unexpected)
    )) as client:
        with pytest.raises(PromptTokenBudgetError):
            await ChatCompletionsModel(client=client, model="glm", prompt_token_budget=BUDGET).complete(
                [UserMessage(content="hello")], settings=ModelSettings(max_tokens=requested, extra_body=extra)
            )


def test_counter_parses_tool_arguments_and_counts_image_expansion_without_mutating_wire(monkeypatch):
    import sys
    from types import SimpleNamespace

    wire = [
        {"role": "assistant", "reasoning_content": "plan", "tool_calls": [{
            "function": {"name": "bash", "arguments": '{"command":"pwd"}'},
        }]},
        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "image"}}]},
    ]
    observed = []

    class Tokenizer:
        def apply_chat_template(self, conversation, **kwargs):
            observed.append((conversation, kwargs))
            return [1] * 10

    monkeypatch.setattr(token_budget, "_load_processor", lambda *args: (
        Tokenizer(), SimpleNamespace(merge_size=2, get_number_of_image_patches=lambda h, w: 64)
    ))
    monkeypatch.setitem(sys.modules, "transformers.image_utils", SimpleNamespace(
        load_image=lambda *args, **kwargs: SimpleNamespace(height=64, width=64)
    ))
    assert token_budget.count_prompt_tokens(BUDGET, wire, [{"function": {"name": "bash"}}]) == 25
    assert observed[0][0][0]["tool_calls"][0]["function"]["arguments"] == {"command": "pwd"}
    assert observed[0][0][0]["reasoning_content"] == "plan"
    assert observed[0][1]["return_dict"] is False
    assert isinstance(wire[0]["tool_calls"][0]["function"]["arguments"], str)


@pytest.mark.parametrize("overrides", [
    {"tokenize": False}, {"truncation": True, "max_length": 10},
    {"padding": True}, {"chat_template": "hi"},
])
def test_template_kwargs_cannot_bypass_counting(monkeypatch, overrides):
    monkeypatch.setattr(token_budget, "_load_processor", lambda *args: (None, None))
    with pytest.raises(PromptTokenBudgetError):
        token_budget.count_prompt_tokens(BUDGET, [], [], template_kwargs=overrides)


@pytest.mark.parametrize("arguments", [
    '{"command": "ls"', '["ls"]', '"ls"', 'null', '42', 'true',
    '{"value": NaN}', '{"value": Infinity}', '{"value": -Infinity}',
])
def test_counter_preserves_failed_tool_arguments_in_a_renderable_object(monkeypatch, arguments):
    wire = [{"role": "assistant", "tool_calls": [{
        "id": "1", "type": "function",
        "function": {"name": "bash", "arguments": arguments},
    }]}]

    class Tokenizer:
        def apply_chat_template(self, conversation, **kwargs):
            replayed = conversation[0]["tool_calls"][0]["function"]["arguments"]
            assert replayed == {"_invalid_tool_arguments": arguments}
            return [1] * 10

    monkeypatch.setattr(token_budget, "_load_processor", lambda *args: (Tokenizer(), None))
    assert token_budget.count_prompt_tokens(BUDGET, wire, []) == 10
    assert wire[0]["tool_calls"][0]["function"]["arguments"] == arguments


async def test_failed_tool_call_reaches_next_generation_with_same_error_and_budget(monkeypatch):
    malformed = '{"command": "ls"'
    valid = '{ "command": "pwd" }'
    assistant = AssistantMessage(content="checking", reasoning_content="plan", tool_calls=(
        ToolCall(id="bad", name="bash", arguments=malformed),
        ToolCall(id="good", name="bash", arguments=valid),
    ))
    messages = [assistant, ToolMessage(tool_call_id="bad", content="Invalid JSON: retry the tool call"),
                ToolMessage(tool_call_id="good", content="/filesystem")]
    counted = []
    sent = []

    class Tokenizer:
        def apply_chat_template(self, conversation, **kwargs):
            counted.append(conversation)
            assert conversation[0]["tool_calls"][0]["function"]["arguments"] == {
                "_invalid_tool_arguments": malformed,
            }
            assert conversation[0]["tool_calls"][1]["function"]["arguments"] == {"command": "pwd"}
            return [1] * 100

    def respond(request):
        sent.append(json.loads(request.content))
        return httpx.Response(200, json={
            "id": "r", "object": "chat.completion", "created": 0, "model": "glm",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "recovered"},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 1, "total_tokens": 101},
        })

    monkeypatch.setattr(token_budget, "_load_processor", lambda *args: (Tokenizer(), None))
    async with AsyncOpenAI(api_key="test", http_client=httpx.AsyncClient(
        transport=httpx.MockTransport(respond)
    )) as client:
        result = await ChatCompletionsModel(client=client, model="glm", prompt_token_budget=BUDGET).complete(
            messages, settings=ModelSettings(max_tokens=131072)
        )
    assert result.assistant.content == "recovered"
    assert len(sent) == len(counted) == 1
    assert sent[0]["max_tokens"] == 130972
    calls = sent[0]["messages"][0]["tool_calls"]
    assert [call["id"] for call in calls] == ["bad", "good"]
    assert calls[1]["function"]["arguments"] == valid
    assert json.loads(calls[0]["function"]["arguments"]) == {"_invalid_tool_arguments": malformed}
    assert sent[0]["messages"][0]["reasoning_content"] == "plan"
    assert sent[0]["messages"][1:] == [
        {"role": "tool", "tool_call_id": "bad", "content": messages[1].content},
        {"role": "tool", "tool_call_id": "good", "content": messages[2].content},
    ]
    assert assistant.tool_calls[0].arguments == malformed
