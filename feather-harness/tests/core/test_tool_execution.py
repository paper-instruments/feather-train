from __future__ import annotations

import asyncio

import pytest

from feather_harness import (
    AgentRunError,
    AssistantMessage,
    FeatherAgent,
    ToolCall,
    ToolMessage,
    ToolResult,
    UserMessage,
)
from feather_harness.tools.todo import TodoList

from ..helpers import ScriptedModel, request, tool, work_message


async def test_tool_results_match_call_ids_and_execution_order_in_next_request():
    events = []

    async def invoke(raw):
        events.append(("start", raw))
        await asyncio.sleep(0)
        events.append(("finish", raw))
        return ToolResult(content=raw)

    model = ScriptedModel(
        AssistantMessage(
            tool_calls=(
                ToolCall(id="first", name="work", arguments="{bad"),
                ToolCall(id="missing", name="unknown", arguments="{}"),
                ToolCall(id="second", name="work", arguments='{"value":2}'),
            )
        ),
        AssistantMessage(content="done"),
    )
    result = await FeatherAgent(model=model, tools=(tool(invoke),)).run(request())
    assert result.final_answer == "done"
    assert events == [
        ("start", "{bad"),
        ("finish", "{bad"),
        ("start", '{"value":2}'),
        ("finish", '{"value":2}'),
    ]
    replies = [m for m in model.calls[1].request["messages"] if m["role"] == "tool"]
    assert [(m["tool_call_id"], m["content"]) for m in replies] == [
        ("first", "{bad"),
        ("missing", "Tool 'unknown' not found."),
        ("second", '{"value":2}'),
    ]


async def test_empty_arguments_reach_tools_as_an_empty_object():
    model = ScriptedModel(
        AssistantMessage(tool_calls=(ToolCall(id="read", name="todo_write", arguments=""),)),
        AssistantMessage(content="done"),
    )
    result = await FeatherAgent(model=model, tools=(TodoList().tool(),)).run(request())

    assert result.final_answer == "done"
    (reply,) = [m for m in model.calls[1].request["messages"] if m["role"] == "tool"]
    assert reply["content"] == '{"todos":[]}'


async def test_tool_failure_keeps_prior_result_and_stops_remaining_calls():
    failure = OSError("sandbox disconnected")
    invoked = []

    async def invoke(raw):
        invoked.append(raw)
        if raw == "fail":
            raise failure
        return ToolResult(content="saved")

    message = AssistantMessage(
        tool_calls=tuple(
            ToolCall(id=name, name="work", arguments=name) for name in ("save", "fail", "never")
        )
    )
    with pytest.raises(AgentRunError) as raised:
        await FeatherAgent(model=ScriptedModel(message), tools=(tool(invoke),)).run(request())
    result = raised.value.result
    assert raised.value.cause is failure
    assert result.stop_reason == "failed"
    assert result.num_turns == 1
    assert invoked == ["save", "fail"]
    assert result.execution_messages() == (
        UserMessage(content="task"),
        message,
        ToolMessage(tool_call_id="save", content="saved"),
    )


async def test_final_turn_executes_tools_before_exhaustion():
    seen = []

    async def invoke(raw):
        seen.append(raw)
        return ToolResult(content="finished")

    result = await FeatherAgent(model=ScriptedModel(work_message()), tools=(tool(invoke),)).run(
        request(max_turns=1)
    )
    assert seen == ["{}"]
    assert result.stop_reason == "MaxTurnsExceeded"
    assert result.execution_messages()[-1].content == "finished"


async def test_refusal_with_tools_continues():
    async def invoke(raw):
        return ToolResult(content="finished")

    result = await FeatherAgent(
        model=ScriptedModel(work_message(refusal="declined"), AssistantMessage(content="done")),
        tools=(tool(invoke),),
    ).run(request())
    assert result.final_answer == "done"
    assert result.execution_messages()[1].refusal == "declined"


async def test_end_run_hook_ends_after_the_message_tools_run():
    seen = []

    async def invoke(raw):
        return ToolResult(content="presented", metadata={"presented": 1})

    def end_run(message, results):
        seen.append((message.content, [result.metadata for result in results]))
        return True

    message = AssistantMessage(
        content="Here is the deck.",
        tool_calls=(ToolCall(id="k", name="work", arguments="{}"),),
    )
    model = ScriptedModel(message)
    result = await FeatherAgent(
        model=model, tools=(tool(invoke),), end_run_after_tools=end_run
    ).run(request())
    assert result.stop_reason == "completed"
    assert result.final_answer == "Here is the deck."
    assert result.num_turns == 1
    assert len(model.calls) == 1
    assert seen == [("Here is the deck.", [{"presented": 1}])]
    assert result.execution_messages() == (
        UserMessage(content="task"),
        message,
        ToolMessage(tool_call_id="k", content="presented"),
    )


async def test_end_run_hook_returning_false_continues_the_loop():
    async def invoke(raw):
        return ToolResult(content="presented")

    model = ScriptedModel(
        AssistantMessage(
            content="Here is the deck.",
            tool_calls=(
                ToolCall(id="k", name="work", arguments="{}"),
                ToolCall(id="gone", name="unknown", arguments="{}"),
            ),
        ),
        AssistantMessage(content="done"),
    )
    seen = []

    def end_run(message, results):
        seen.append([(result.content, result.error) for result in results])
        return False

    result = await FeatherAgent(
        model=model, tools=(tool(invoke),), end_run_after_tools=end_run
    ).run(request())
    assert result.final_answer == "done"
    assert result.num_turns == 2
    assert seen == [[("presented", None), ("Tool 'unknown' not found.", "tool_error")]]


async def test_without_end_run_hook_tool_messages_always_continue():
    async def invoke(raw):
        return ToolResult(content="presented")

    def run_with(hook):
        model = ScriptedModel(
            AssistantMessage(
                content="Here is the deck.",
                tool_calls=(ToolCall(id="k", name="work", arguments="{}"),),
            ),
            AssistantMessage(content="done"),
        )
        kwargs = {} if hook is None else {"end_run_after_tools": hook}
        return model, FeatherAgent(model=model, tools=(tool(invoke),), **kwargs).run(request())

    baseline_model, baseline = run_with(None)
    declined_model, declined = run_with(lambda message, results: False)
    baseline, declined = await baseline, await declined
    assert baseline.final_answer == declined.final_answer == "done"
    assert [call.request for call in baseline_model.calls] == [
        call.request for call in declined_model.calls
    ]
