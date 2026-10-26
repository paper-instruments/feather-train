import json
import subprocess
import sys

from feather_harness import (
    AssistantMessage,
    ProviderUsage,
    RunResult,
    SystemMessage,
    ToolCall,
    ToolMessage,
    TraceSegment,
    UserMessage,
)
from harbor.models.trajectories.agent import Agent
from harbor.models.trajectories.trajectory import Trajectory
from harbor.utils.traces_utils import collect_conversations_from_trial, collect_subagent_traces

from feather_eval.atif import to_atif, write_atif

AGENT = Agent(name="feather", version="test", model_name="test-model")


def test_tool_results_stay_with_their_calls_and_partial_calls_stay_unfinished():
    result = RunResult(
        stop_reason="failed",
        num_turns=1,
        compactions=0,
        segments=(
            TraceSegment(
                kind="execution",
                input=(UserMessage(content="task"),),
                messages=(
                    AssistantMessage(
                        tool_calls=(
                            ToolCall(id="a", name="Read", arguments='{"path":"a"}'),
                            ToolCall(id="b", name="Read", arguments='{"path":"b"}'),
                            ToolCall(id="c", name="Bash", arguments="not JSON"),
                        )
                    ),
                    ToolMessage(tool_call_id="b", content=[{"type": "text", "text": "B contents"}]),
                    ToolMessage(tool_call_id="a", content="A contents"),
                ),
            ),
        ),
    )
    trajectory = to_atif(result, agent=AGENT, session_id="run")["trajectory.json"]
    saved = Trajectory.model_validate_json(trajectory.model_dump_json())
    step = saved.steps[-1]
    assert [call.tool_call_id for call in step.tool_calls] == ["a", "b", "c"]
    assert step.tool_calls[-1].extra == {"raw_arguments": "not JSON"}
    assert [
        (item.source_call_id, item.model_dump(exclude_none=True)["content"])
        for item in step.observation.results
    ] == [
        ("b", [{"type": "text", "text": "B contents"}]),
        ("a", "A contents"),
    ]


def test_compaction_files_preserve_context_without_duplicate_training_targets(tmp_path):
    original = (SystemMessage(content="system"), UserMessage(content="task"))
    first = AssistantMessage(content="First response")
    summary_input = original + (first, UserMessage(content="Summarize"))
    continuation = (
        SystemMessage(content="system"),
        UserMessage(content="summary with old todo"),
        first,
    )
    result = RunResult(
        stop_reason="completed",
        num_turns=2,
        compactions=1,
        segments=(
            TraceSegment(kind="execution", input=original, messages=(first,)),
            TraceSegment(
                kind="compaction",
                input=summary_input,
                messages=(AssistantMessage(content="summary"),),
            ),
            TraceSegment(
                kind="execution",
                input=continuation,
                messages=(AssistantMessage(content="Final answer"),),
            ),
        ),
    )
    write_atif(result, tmp_path / "agent", agent=AGENT, session_id="run")
    trajectories = {
        p.name: Trajectory.model_validate_json(p.read_text())
        for p in (tmp_path / "agent").glob("*.json")
    }
    assert set(trajectories) == {
        "trajectory.json",
        "trajectory.summarization-1.json",
        "trajectory.cont-1.json",
    }
    root = trajectories["trajectory.json"]
    assert all(step.is_copied_context is None for step in root.steps)
    assert root.continued_trajectory_ref == "trajectory.cont-1.json"
    assert (
        root.steps[-1].observation.results[-1].subagent_trajectory_ref[0].trajectory_path
        == "trajectory.summarization-1.json"
    )
    continued = trajectories["trajectory.cont-1.json"]
    assert [step.message for step in continued.steps] == [
        "system",
        "summary with old todo",
        "First response",
        "Final answer",
    ]
    assert [step.is_copied_context for step in continued.steps] == [True, True, True, None]
    summary = trajectories["trajectory.summarization-1.json"]
    assert [step.message for step in summary.steps] == [
        "system",
        "task",
        "First response",
        "Summarize",
        "summary",
    ]
    metadata = dict(
        agent_name="feather",
        model_name="test-model",
        model_provider="test",
        start_time="now",
        run_id="run",
        trial_name="trial",
        task_name="task",
    )
    conversations = collect_conversations_from_trial(tmp_path, metadata)
    assert [row["conversations"][-1]["content"] for row in conversations] == [
        "First response",
        "Final answer",
    ]
    assert [message["content"] for message in conversations[0]["conversations"]] == [
        "system",
        "task",
        "First response",
    ]
    summaries = collect_subagent_traces(tmp_path, metadata)
    assert list(summaries) == ["summarization-1"]
    assert summaries["summarization-1"][0]["conversations"][-1]["content"] == "summary"


def test_supplied_assistant_history_is_context_not_a_training_target(tmp_path):
    result = RunResult(
        stop_reason="completed",
        num_turns=1,
        compactions=0,
        segments=(
            TraceSegment(
                kind="execution",
                input=(
                    SystemMessage(content="system"),
                    UserMessage(content="Earlier question"),
                    AssistantMessage(content="Supplied answer"),
                    UserMessage(content="Current question"),
                ),
                messages=(AssistantMessage(content="New answer"),),
            ),
        ),
    )
    write_atif(result, tmp_path / "agent", agent=AGENT, session_id="run")
    trajectory = Trajectory.model_validate_json((tmp_path / "agent/trajectory.json").read_text())
    assert [step.is_copied_context for step in trajectory.steps] == [None, None, True, None, None]
    conversations = collect_conversations_from_trial(
        tmp_path,
        dict(
            agent_name="feather",
            model_name="test-model",
            model_provider="test",
            start_time="now",
            run_id="run",
            trial_name="trial",
            task_name="task",
        ),
    )
    assert len(conversations) == 1
    assert [message["content"] for message in conversations[0]["conversations"]] == [
        "system",
        "Earlier question",
        "Supplied answer",
        "Current question",
        "New answer",
    ]


def test_failed_summary_is_recorded_but_never_becomes_grader_answer():
    result = RunResult(
        stop_reason="failed",
        num_turns=1,
        compactions=0,
        segments=(
            TraceSegment(
                kind="execution",
                input=(UserMessage(content="task"),),
                messages=(AssistantMessage(content="Actual answer"),),
            ),
            TraceSegment(kind="compaction", input=(UserMessage(content="Summarize"),), messages=()),
        ),
    )
    linked = to_atif(result, agent=AGENT, session_id="run")
    assert linked["trajectory.json"].extra["stop_reason"] == "failed"
    assert linked["trajectory.json"].continued_trajectory_ref is None
    assert len(linked["trajectory.summarization-1.json"].steps) == 1
    flat = to_atif(result, agent=AGENT, session_id="run", grader=True)["trajectory.json"]
    assert [step.message for step in flat.steps] == ["task", "Actual answer"]


def test_grader_trajectory_preserves_execution_and_total_usage():
    result = RunResult(
        stop_reason="completed",
        num_turns=2,
        compactions=1,
        segments=(
            TraceSegment(
                kind="execution",
                input=(UserMessage(content="task"),),
                messages=(AssistantMessage(content="First answer"),),
                calls=(ProviderUsage(kind="agent", usage={"prompt_tokens": 10}),),
            ),
            TraceSegment(
                kind="compaction",
                input=(UserMessage(content="Summarize"),),
                messages=(AssistantMessage(content="summary"),),
                calls=(ProviderUsage(kind="compaction", usage={"prompt_tokens": 20}),),
            ),
            TraceSegment(
                kind="execution",
                input=(UserMessage(content="summary"),),
                messages=(AssistantMessage(content="Final answer"),),
                calls=(ProviderUsage(kind="agent", usage={"prompt_tokens": 30}),),
            ),
        ),
    )

    trajectory = to_atif(result, agent=AGENT, session_id="run", grader=True)["trajectory.json"]

    produced = [step for step in trajectory.steps if step.source == "agent"]
    assert [step.metrics.prompt_tokens for step in produced] == [10, 30]
    assert [step.llm_call_count for step in produced] == [1, 1]
    assert trajectory.final_metrics.total_prompt_tokens == 60
    assert trajectory.final_metrics.extra["model_calls"] == 3


def test_grader_usage_skips_supplied_assistant_history():
    result = RunResult(
        stop_reason="completed",
        num_turns=1,
        compactions=0,
        segments=(
            TraceSegment(
                kind="execution",
                input=(
                    UserMessage(content="earlier"),
                    AssistantMessage(content="Earlier answer"),
                    UserMessage(content="task"),
                ),
                messages=(AssistantMessage(content="Final answer"),),
                calls=(ProviderUsage(kind="agent", usage={"prompt_tokens": 10}),),
            ),
        ),
    )

    trajectory = to_atif(result, agent=AGENT, session_id="run", grader=True)["trajectory.json"]

    supplied, produced = [step for step in trajectory.steps if step.source == "agent"]
    assert supplied.is_copied_context is True
    assert supplied.metrics is None
    assert produced.message == "Final answer"
    assert produced.metrics.prompt_tokens == 10
    assert all(step.is_copied_context is None for step in trajectory.steps if step.source == "user")


def test_cli_converts_train_evidence_into_one_grader_trajectory(tmp_path):
    result = RunResult(
        stop_reason="completed",
        num_turns=2,
        compactions=1,
        segments=(
            TraceSegment(
                kind="execution",
                input=(UserMessage(content="task"),),
                messages=(AssistantMessage(content="Before compaction"),),
            ),
            TraceSegment(
                kind="compaction",
                input=(UserMessage(content="Summarize"),),
                messages=(AssistantMessage(content="Not a final answer"),),
            ),
            TraceSegment(
                kind="execution",
                input=(UserMessage(content="summary"),),
                messages=(AssistantMessage(content="Submitted workbook"),),
            ),
        ),
    )
    source = tmp_path / "agent_trace.json"
    source.write_text(
        json.dumps(
            {
                "schema_version": "tldc.feather-agent-trace.v2",
                "agent": {"name": "feather", "model": "model"},
                "task_id": "task-id",
                "trace": result.model_dump(),
            }
        )
    )
    output = tmp_path / "out"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "feather_eval.atif",
            str(source),
            "--output-dir",
            str(output),
            "--grader",
        ],
        check=True,
    )
    trajectory = Trajectory.model_validate_json((output / "trajectory.json").read_text())
    assert trajectory.session_id == "task-id"
    assert trajectory.agent.model_name == "model"
    assert [step.message for step in trajectory.steps] == [
        "task",
        "Before compaction",
        "Submitted workbook",
    ]
    assert len(list(output.glob("*.json"))) == 1


def test_image_observations_keep_embedded_data_and_call_association():
    url = "data:image/png;base64,iVBORw0KGgo="
    result = RunResult(
        stop_reason="completed", num_turns=2, compactions=0,
        segments=(TraceSegment(kind="execution", input=(UserMessage(content="inspect"),), messages=(
            AssistantMessage(tool_calls=(ToolCall(id="image", name="read", arguments="{}"),)),
            ToolMessage(tool_call_id="image", content=[{"type": "image_url", "image_url": {"url": url}}]),
            AssistantMessage(content="answer"),
        )),),
    )
    trajectory = to_atif(result, agent=AGENT, session_id="images")["trajectory.json"]
    saved = Trajectory.model_validate_json(trajectory.model_dump_json())
    observation = saved.steps[1].observation.results[0]
    assert observation.source_call_id == "image"
    assert observation.content[0].source.path == url
    assert observation.content[0].source.media_type == "image/png"
