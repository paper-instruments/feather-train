from __future__ import annotations

import json

import pytest

from feather_harness import (
    AgentRunError,
    AssistantMessage,
    CompactionConfig,
    FeatherAgent,
    ModelSettings,
    RunRequest,
    SystemMessage,
    ToolCall,
    ToolMessage,
    ToolResult,
    UserMessage,
)
from feather_harness.core.compaction import Accounting, build_compaction_inputs, check_compaction

from ..helpers import ScriptedModel, request, tool, work_message


def test_measured_usage_replaces_estimation_of_the_already_measured_prefix():
    context = [
        UserMessage(content="x" * 200000),
        AssistantMessage(content="done"),
        ToolMessage(tool_call_id="x", content="small"),
    ]
    config = CompactionConfig(enabled=True, context_window_tokens=31000)
    low_usage = Accounting(
        turns=1, turns_since_compaction=1, measured_tokens=10, measured_message_count=2
    )
    high_usage = Accounting(
        turns=1, turns_since_compaction=1, measured_tokens=100000, measured_message_count=2
    )

    assert check_compaction(context, config, low_usage, 10) is None
    tokens = check_compaction(context, config, high_usage, 10)
    assert isinstance(tokens, int)
    assert 100000 < tokens < 100100


@pytest.mark.parametrize(
    "accounting, max_turns, should_compact",
    [
        pytest.param(
            Accounting(turns=1, turns_since_compaction=1), 2, False, id="reserve-final-turn"
        ),
        pytest.param(Accounting(turns=1, compactions=1), 10, False, id="no-consecutive-compaction"),
        pytest.param(
            Accounting(turns=2, compactions=1, turns_since_compaction=1),
            10,
            True,
            id="compact-after-progress",
        ),
    ],
)
def test_compaction_requires_room_and_progress(accounting, max_turns, should_compact):
    context = [UserMessage(content="x" * 200000)]
    config = CompactionConfig(enabled=True, context_window_tokens=31000)

    tokens = check_compaction(context, config, accounting, max_turns)

    assert (tokens is not None) == should_compact


async def test_empty_summary_fails_without_losing_completed_tool_work():
    async def invoke(raw):
        return ToolResult(content="finished")

    model = ScriptedModel(work_message(), *[AssistantMessage(content="")] * 4, tokens=120000)
    with pytest.raises(AgentRunError, match="summary was empty") as raised:
        await FeatherAgent(
            model=model, tools=(tool(invoke),), compaction=CompactionConfig(enabled=True)
        ).run(request())
    result = raised.value.result
    assert result.num_turns == 1
    assert result.compactions == 0
    assert result.execution_messages()[-1] == ToolMessage(tool_call_id="k", content="finished")
    assert [segment.kind for segment in result.segments] == ["execution", *["compaction"] * 4]
    assert all(segment.messages == (AssistantMessage(content=""),) for segment in result.segments[1:])
    assert all(call.request == model.calls[1].request for call in model.calls[1:])


@pytest.mark.parametrize("keep_recent_turns", [0, 1])
async def test_summary_retries_preserve_context_settings_and_completed_tool_work(keep_recent_turns):
    completed = []
    transcripts = []

    async def invoke(raw):
        completed.append(raw)
        return ToolResult(content="finished")

    async def save(content):
        transcripts.append(content)

    settings = ModelSettings(temperature=1, top_p=0.95, reasoning_effort="max")
    model = ScriptedModel(
        work_message(),
        AssistantMessage(content=None, reasoning_content="thinking only"),
        AssistantMessage(content=" "),
        AssistantMessage(content="summary"),
        AssistantMessage(content="done"),
        tokens=120000,
    )
    result = await FeatherAgent(
        model=model,
        tools=(tool(invoke),),
        compaction=CompactionConfig(enabled=True, keep_recent_turns=keep_recent_turns),
        compaction_tool_state=lambda: (f"completed={len(completed)}",),
        save_transcript=save,
    ).run(RunRequest(messages=(UserMessage(content="task"),), model_settings=settings))

    assert result.final_answer == "done"
    assert (result.num_turns, result.compactions, result.recoveries) == (2, 1, 0)
    assert completed == ["{}"]
    assert len(transcripts) == 1
    assert all(call.request == model.calls[1].request for call in model.calls[1:4])
    assert all(
        call.settings == settings.model_copy(update={"max_tokens": 65536})
        for call in model.calls[1:4]
    )
    assert f"completed={1 - keep_recent_turns}" in result.segments[-1].input[1].content
    if keep_recent_turns:
        assert result.segments[-1].input[-1] == ToolMessage(tool_call_id="k", content="finished")
    assert result.execution_messages() == (
        UserMessage(content="task"), work_message(),
        ToolMessage(tool_call_id="k", content="finished"), AssistantMessage(content="done"),
    )
    for segment, call in zip(result.segments, model.calls, strict=True):
        assert [m.model_dump(mode="json") for m in segment.input] == call.request["messages"]


async def test_each_compaction_has_its_own_three_retries():
    async def invoke(raw):
        return ToolResult(content="finished")

    failed = [AssistantMessage(content="")] * 3
    model = ScriptedModel(
        work_message(), *failed, AssistantMessage(content="first summary"),
        work_message(), *failed, AssistantMessage(content="second summary"),
        AssistantMessage(content="done"), tokens=120000,
    )
    result = await FeatherAgent(
        model=model, tools=(tool(invoke),), compaction=CompactionConfig(enabled=True)
    ).run(request())

    assert (result.final_answer, result.num_turns, result.compactions) == ("done", 3, 2)
    assert [call.kind for call in model.calls] == [
        "agent", *["compaction"] * 4, "agent", *["compaction"] * 4, "agent"
    ]


@pytest.mark.parametrize("summary_limit", [None, 131072])
async def test_compaction_preserves_sampling_and_provider_routing(summary_limit):
    async def invoke(raw):
        return ToolResult(content="finished")

    settings = ModelSettings(
        temperature=1.0,
        top_p=0.95,
        max_tokens=131072,
        reasoning_effort="max",
        extra_body={"provider": {"only": ["baseten"]}},
    )
    model = ScriptedModel(
        work_message(), AssistantMessage(content="summary"), AssistantMessage(content="done"), tokens=120000
    )
    compaction = (
        CompactionConfig(enabled=True)
        if summary_limit is None
        else CompactionConfig(enabled=True, max_tokens=summary_limit)
    )
    await FeatherAgent(
        model=model, tools=(tool(invoke),), compaction=compaction
    ).run(RunRequest(messages=(UserMessage(content="task"),), model_settings=settings))

    assert [call.kind for call in model.calls] == ["agent", "compaction", "agent"]
    assert model.calls[0].settings == settings
    assert model.calls[1].settings == settings.model_copy(
        update={"max_tokens": 65536 if summary_limit is None else summary_limit}
    )
    assert model.calls[2].settings == settings


async def test_trace_preserves_the_tool_state_actually_sent_to_the_model():
    async def invoke(raw):
        return ToolResult(content="finished")

    model = ScriptedModel(
        work_message(),
        AssistantMessage(content="summary"),
        AssistantMessage(content="done"),
        tokens=120000,
    )
    snapshot = "## Current Todo List\n\nDo work"
    result = await FeatherAgent(
        model=model,
        tools=(tool(invoke),),
        compaction=CompactionConfig(enabled=True),
        compaction_tool_state=lambda: (snapshot,),
    ).run(request())
    assert result.compactions == 1
    assert snapshot in model.calls[-1].request["messages"][1]["content"]
    assert snapshot in result.segments[-1].input[1].content
    assert [m.model_dump(mode="json") for m in result.segments[-1].input] == model.calls[
        -1
    ].request["messages"]
    assert all(
        "Summary of Previous Work" not in (getattr(m, "content", "") or "")
        for m in result.execution_messages()
    )


async def test_repeated_compaction_records_exact_summary_and_execution_inputs():
    async def invoke(raw):
        return ToolResult(content="finished")

    snapshots = iter(("todo one", "todo two"))
    summary = "Keep this literal text:\n\n---\n\ntodo two\n\n---\n\nContinue the task from this compacted state."
    model = ScriptedModel(
        work_message(),
        AssistantMessage(content="first summary"),
        work_message(),
        AssistantMessage(content=summary),
        AssistantMessage(content="done"),
        tokens=120000,
    )
    result = await FeatherAgent(
        model=model,
        tools=(tool(invoke),),
        compaction=CompactionConfig(enabled=True, keep_recent_turns=0),
        compaction_tool_state=lambda: (next(snapshots),),
    ).run(
        RunRequest(
            messages=(
                SystemMessage(content="original system instructions"),
                UserMessage(content="task"),
            )
        )
    )
    assert result.compactions == 2
    assert result.num_turns == 3
    assert [call.kind for call in model.calls] == [
        "agent",
        "compaction",
        "agent",
        "compaction",
        "agent",
    ]
    expected = (
        "## Summary of Previous Work\n\n"
        + summary
        + "\n\n---\n\ntodo two\n\n---\n\nContinue the task from this compacted state."
    )
    assert result.segments[-1].input[2].content == expected
    assert len(result.segments) == 5
    for segment, call in zip(result.segments, model.calls, strict=True):
        assert [m.model_dump(mode="json") for m in segment.input] == call.request["messages"]
    assert result.segments[3].messages == (AssistantMessage(content=summary),)
    assert all(
        getattr(message, "content", None) != summary for message in result.execution_messages()
    )
    assert result.execution_messages()[0] == SystemMessage(content="original system instructions")
    assert result.segments[-1].input[0] == result.execution_messages()[0]
    for call in model.calls:
        messages = call.request["messages"]
        assert messages[0]["role"] == "system"
        if call.kind == "agent":
            assert messages[0]["content"] == "original system instructions"
        else:
            assert (
                messages[0]["content"]
                == "Summarize the supplied conversation into a compact continuation state."
            )
            assert all(m["role"] != "system" for m in messages[1:])


def test_system_prompt_does_not_enter_unanchored_compaction_estimate():
    context = [SystemMessage(content="s" * 100000), UserMessage(content="task")]
    config = CompactionConfig(enabled=True, context_window_tokens=31000)
    assert check_compaction(context, config, Accounting(), 10) is None


@pytest.mark.parametrize("keep", [0, 1, 2])
def test_compaction_retains_complete_recent_turns_without_losing_tool_results(keep):
    prefix = [UserMessage(content="task")]
    turns = [
        [
            AssistantMessage(tool_calls=(ToolCall(id=str(i), name="work", arguments="{}"),)),
            ToolMessage(tool_call_id=str(i), content=f"result {i}"),
        ]
        for i in range(3)
    ]
    summary_input, recent = build_compaction_inputs(
        [*prefix, *(message for turn in turns for message in turn)],
        prefix=prefix,
        config=CompactionConfig(keep_recent_turns=keep),
    )
    assert recent == [message for turn in turns[3 - keep :] for message in turn]
    assert summary_input[:-1] == [
        *prefix,
        *(message for turn in turns[: 3 - keep] for message in turn),
    ]


@pytest.mark.parametrize("keep", [0, 1, 2, 4])
async def test_summary_uses_state_before_retained_turns_without_rewinding_live_state(keep):
    completed = 0

    async def invoke(raw):
        nonlocal completed
        completed += 1
        return ToolResult(content=f"completed={completed}")

    model = ScriptedModel(
        work_message(),
        work_message(),
        AssistantMessage(content="first summary"),
        work_message(),
        work_message(),
        AssistantMessage(content="second summary"),
        AssistantMessage(content="done"),
        tokens=120000,
    )
    result = await FeatherAgent(
        model=model,
        tools=(tool(invoke),),
        compaction=CompactionConfig(
            enabled=True, keep_recent_turns=keep, min_turns_before_compaction=2
        ),
        compaction_tool_state=lambda: (f"Todo snapshot: completed={completed}",),
    ).run(request())

    assert completed == 4
    assert result.compactions == 2
    for next_call, completed_at_compaction in ((model.calls[3], 2), (model.calls[6], 4)):
        messages = next_call.request["messages"]
        summary = messages[1]["content"]
        historical_count = max(0, completed_at_compaction - keep)
        assert f"Todo snapshot: completed={historical_count}" in summary
        retained_results = [m["content"] for m in messages[2:] if m["role"] == "tool"]
        assert retained_results == [
            f"completed={i}" for i in range(historical_count + 1, completed_at_compaction + 1)
        ]
    for segment, call_index in ((result.segments[2], 3), (result.segments[4], 6)):
        assert [m.model_dump(mode="json") for m in segment.input] == model.calls[
            call_index
        ].request["messages"]


async def test_supplied_history_does_not_get_live_state_labeled_as_historical():
    model = ScriptedModel(AssistantMessage(content="summary"), AssistantMessage(content="done"))
    result = await FeatherAgent(
        model=model,
        compaction=CompactionConfig(enabled=True, keep_recent_turns=2),
        compaction_tool_state=lambda: ("live state that does not describe the old turn",),
    ).run(
        RunRequest(
            messages=(
                UserMessage(content="x" * 400000),
                AssistantMessage(tool_calls=(ToolCall(id="old", name="work", arguments="{}"),)),
                ToolMessage(tool_call_id="old", content="old result"),
            )
        )
    )
    assert result.compactions == 1
    assert "live state" not in json.dumps(model.calls[1].request)
    assert any(m.get("tool_call_id") == "old" for m in model.calls[1].request["messages"])


async def test_transcript_preserves_original_messages_across_compactions():
    snapshots = []

    async def save(content):
        snapshots.append([json.loads(line) for line in content.splitlines()])

    async def invoke(raw):
        return ToolResult(content="tool output")

    model = ScriptedModel(
        work_message(),
        AssistantMessage(content="first summary"),
        work_message(),
        AssistantMessage(content="second summary"),
        AssistantMessage(content="done"),
        tokens=120000,
    )
    result = await FeatherAgent(
        model=model,
        tools=(tool(invoke),),
        compaction=CompactionConfig(enabled=True, keep_recent_turns=1),
        save_transcript=save,
    ).run(request())

    exchange = [work_message(), ToolMessage(tool_call_id="k", content="tool output")]
    original = [UserMessage(content="task"), *exchange, *exchange]
    expected = [message.model_dump(mode="json") for message in original]
    assert result.compactions == 2
    assert snapshots == [expected[:3], expected]


async def test_transcript_save_failure_stops_before_summary_request():
    async def save(content):
        raise OSError("disk full")

    async def invoke(raw):
        return ToolResult(content="finished")

    model = ScriptedModel(work_message(), tokens=120000)
    with pytest.raises(AgentRunError, match="disk full") as raised:
        await FeatherAgent(
            model=model,
            tools=(tool(invoke),),
            compaction=CompactionConfig(enabled=True),
            save_transcript=save,
        ).run(request())

    assert len(model.calls) == 1
    assert raised.value.result.execution_messages()[-1].content == "finished"


def test_explicit_trigger_replaces_the_window_derived_threshold():
    # A 1M window would otherwise compact at 850k; the vendor-aligned trigger waits until 960k.
    context = [UserMessage(content="task"), AssistantMessage(content="done")]
    config = CompactionConfig(enabled=True, context_window_tokens=1_000_000, trigger_tokens=960_000)
    measured = Accounting(
        turns=1, turns_since_compaction=1, measured_tokens=900_000, measured_message_count=2
    )

    assert CompactionConfig(enabled=True, context_window_tokens=1_000_000).threshold_tokens == 850_000
    assert check_compaction(context, config, measured, 10) is None
    with pytest.raises(ValueError, match="below context_window_tokens"):
        CompactionConfig(context_window_tokens=131072, trigger_tokens=131072)
