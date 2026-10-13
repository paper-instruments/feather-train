"""Grader retries, evidence staging, and reward semantics through public APIs."""

import asyncio
import json

import pytest
from feather_harness import AssistantMessage, RunResult, TraceSegment, UserMessage

pytest.importorskip("verifiers.v1")

from verifiers.v1 import SandboxError

from feather_train.grading.grader import (
    GradingConfig,
    HarborGradingError,
    _packaged_grader_test_command,
    grade_task,
)
from feather_train.grading.report import build_judge_span_output
from feather_train.grading.types import AgentEvidence

from .helpers import GraderSandbox, task


def test_official_fireworks_judge_credential_is_explicit():
    command = _packaged_grader_test_command(
        task_id="task-1",
        world_id="world-1",
        grading_evidence_mode=None,
        judge_api_key="test-fireworks-key",
        judge_api_key_var="FIREWORKS_API_KEY",
        official=True,
    )
    assert "FIREWORKS_AI_API_KEY=test-fireworks-key" in command
    assert "FIREWORKS_API_KEY=test-fireworks-key" in command

    unrelated = _packaged_grader_test_command(
        task_id="task-1",
        world_id="world-1",
        grading_evidence_mode=None,
        judge_api_key="test-other-key",
        judge_api_key_var="OPENAI_VERIFIER_API_KEY",
        official=True,
    )
    assert "test-other-key" not in unrelated


def test_official_judge_explicit_proxy_reaches_benchmark_wrapper(monkeypatch):
    monkeypatch.setenv("BRAINTRUST_API_KEY", "unrelated-secret")
    command = _packaged_grader_test_command(
        task_id="task-1",
        world_id="world-1",
        grading_evidence_mode=None,
        judge_api_key="test-openrouter-key",
        judge_api_key_var="OPENROUTER_API_KEY",
        judge_base_url="https://openrouter.ai/api/v1",
        judge_model="openai/deepseek/deepseek-v4-flash-0731",
        official=True,
    )
    assert "OPENAI_VERIFIER_API_KEY=test-openrouter-key" in command
    assert "OPENAI_VERIFIER_BASE_URL=https://openrouter.ai/api/v1" in command
    assert "OPENAI_VERIFIER_MODEL=openai/deepseek/deepseek-v4-flash-0731" in command
    assert "FIREWORKS_AI_API_KEY" not in command
    assert "unrelated-secret" not in command


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [1, SandboxError("connection dropped")])
async def test_grade_retries_failed_judge(failure):
    sandbox = GraderSandbox(attempts=(failure, 0))
    config = GradingConfig(judge_retry_backoff_seconds=0)
    result = await grade_task(
        task(),
        sandbox,
        instruction="Solve this exactly.",
        config=config,
        final_answer="Answer",
    )
    assert (result.score, result.num_met, result.num_criteria) == (0.75, 3, 4)
    assert result.report_error is None
    assert "report_error" not in build_judge_span_output(result)
    assert sandbox.test_runs == 2
    assert any(
        "rm -f -- /logs/verifier/reward.txt /logs/verifier/reward.json /logs/verifier/reward_detail.json"
        in command
        for command in sandbox.commands
    )
    assert result.failed_attempts[0].exit_code == (1 if failure == 1 else -1)
    assert sandbox.overlays["task/final_response.txt"] == b"Answer"
    assert sandbox.overlays["task/instruction.md"] == b"Solve this exactly."
    assert sandbox.overlays["tests/submission_transport.txt"] == b"text_response\n"


@pytest.mark.asyncio
async def test_benchmark_rows_stage_instruction_and_final_response_only():
    sandbox = GraderSandbox()
    grading_task = task()
    grading_task.source = "benchmark"
    grading_task.submission_transport = None
    result = await grade_task(
        grading_task,
        sandbox,
        instruction="Solve this exactly.",
        config=GradingConfig(judge_retry_backoff_seconds=0),
        final_answer="Answer",
    )
    assert result.score == 0.75
    assert sandbox.overlays["task/final_response.txt"] == b"Answer"
    assert sandbox.overlays["task/instruction.md"] == b"Solve this exactly."
    assert "tests/submission_transport.txt" not in sandbox.overlays
    assert "tests/rubric.yaml" not in sandbox.overlays
    assert "tests/expected_deliverable.json" not in sandbox.overlays


@pytest.mark.asyncio
@pytest.mark.parametrize(("exit_code", "reward"), [(2, b"1"), (0, b"nan"), (0, b"1.1")])
async def test_invalid_grader_result_is_fatal_without_retry(exit_code, reward):
    sandbox = GraderSandbox(attempts=(exit_code,), reward=reward)
    with pytest.raises(HarborGradingError) as raised:
        await grade_task(
            task(),
            sandbox,
            instruction="Solve this exactly.",
            config=GradingConfig(),
            final_answer="",
        )
    assert raised.value.retryable is False
    assert sandbox.test_runs == 1


@pytest.mark.parametrize("exit_code", [0, 2])
async def test_non_ascii_grader_output_preserves_text_and_byte_counts(exit_code):
    stdout = "résultat ✅\n"
    stderr = "échec ❌\n" if exit_code else ""
    sandbox = GraderSandbox(attempts=(exit_code,), stdout=stdout, stderr=stderr)
    config = GradingConfig()

    if exit_code:
        with pytest.raises(HarborGradingError) as raised:
            await grade_task(
                task(),
                sandbox,
                instruction="Solve this exactly.",
                config=config,
                final_answer="",
            )
        assert raised.value.failed_attempts[0].message == stderr
        steps = raised.value.steps
    else:
        result = await grade_task(
            task(),
            sandbox,
            instruction="Solve this exactly.",
            config=config,
            final_answer="",
        )
        assert result.stdout == stdout
        steps = result.steps

    execution = next(step for step in steps if step.name == "run_tests")
    assert execution.metadata["stdout_bytes"] == 14
    assert execution.metadata["stderr_bytes"] == (11 if exit_code else 0)


@pytest.mark.asyncio
async def test_cancellation_does_not_retry_grading():
    sandbox = GraderSandbox(attempts=(asyncio.CancelledError(),))
    with pytest.raises(asyncio.CancelledError):
        await grade_task(
            task(),
            sandbox,
            instruction="Solve this exactly.",
            config=GradingConfig(),
            final_answer="",
        )
    assert sandbox.test_runs == 1


@pytest.mark.asyncio
async def test_optional_malformed_detail_preserves_score_and_stages_full_evidence():
    sandbox = GraderSandbox(detail=b"not json")
    evidence = AgentEvidence(
        model="policy",
        instructions="System",
        trace=RunResult(
            stop_reason="completed",
            num_turns=1,
            compactions=0,
            final_answer="Final",
            segments=(
                TraceSegment(
                    kind="execution",
                    input=(UserMessage(content="task"),),
                    messages=(AssistantMessage(content="Final"),),
                ),
            ),
        ),
    )
    config = GradingConfig(stage_agent_trace=True)
    result = await grade_task(
        task(),
        sandbox,
        instruction="Solve this exactly.",
        config=config,
        final_answer="Final",
        agent_evidence=evidence,
    )
    assert result.score == 0.75
    assert result.criteria == ()
    staged = json.loads(sandbox.overlays["task/agent_trace.json"])
    assert staged["trace"] == evidence.trace.model_dump(mode="json")
    assert staged["final_response"] == "Final"
    assert staged["schema_version"] == "tldc.feather-agent-trace.v2"


async def test_missing_required_trace_evidence_fails_without_running_judge():
    sandbox = GraderSandbox()
    with pytest.raises(HarborGradingError, match="requires agent evidence") as raised:
        await grade_task(
            task(),
            sandbox,
            instruction="Solve",
            final_answer="",
            config=GradingConfig(stage_agent_trace=True),
        )
    assert raised.value.retryable is False
    assert raised.value.steps == ()
    assert sandbox.test_runs == 0


@pytest.mark.parametrize(
    "report,error_type",
    [
        (42, "TypeError"),
        ([{"requirement": "Correct answer", "weight": "bad", "verdict": "MET"}], "ValueError"),
    ],
    ids=["not-iterable", "invalid-weight"],
)
async def test_malformed_report_preserves_reward_without_repeating_grader(report, error_type):
    sandbox = GraderSandbox(
        detail=json.dumps({"num_met": 3, "num_criteria": 4, "report": report}).encode()
    )
    result = await grade_task(
        task(), sandbox, instruction="Solve this.", config=GradingConfig(), final_answer="answer"
    )
    assert result.score == 0.75
    assert result.criteria == ()
    assert result.num_met == 3
    assert result.num_criteria == 4
    assert result.failed_attempts == []
    assert sandbox.test_runs == 1
    assert result.report_error.startswith(f"{error_type}:")
    output = build_judge_span_output(result)
    assert output["report_error"] == result.report_error
    assert output["criteria"] == []
    assert output["errors"] == []


@pytest.mark.asyncio
async def test_official_harbor_mode_grades_from_image_workdir(monkeypatch):
    monkeypatch.setenv("OPENAI_VERIFIER_API_KEY", "should-not-reach-official-test-sh")
    monkeypatch.setenv("OPENAI_VERIFIER_BASE_URL", "https://unrelated.example/v1")
    monkeypatch.setenv("BRAINTRUST_API_KEY", "should-not-reach-official-test-sh")
    monkeypatch.setenv("REDUCTO_API_KEY", "should-not-reach-official-test-sh")
    sandbox = GraderSandbox()
    result = await grade_task(
        task(environment_mode="shared", workdir="/app", verifier_timeout_sec=1800),
        sandbox,
        instruction="Solve this.",
        config=GradingConfig(poll_timeout=300),
        final_answer="answer",
    )
    assert result.score == 0.75
    assert "/opt/venv/bin" in sandbox.test_command
    assert "/root/.bashrc" in sandbox.test_command
    assert "export PATH" in sandbox.test_command
    assert "cd /app" in sandbox.test_command
    assert "--chdir=/app" not in sandbox.test_command
    assert sandbox.test_timeout == 1800
    assert not any("importlib.metadata" in command for command in sandbox.commands)
    assert not any("rm -rf -- /logs &&" in command for command in sandbox.commands)
    assert any("rm -rf -- /logs/verifier" in command for command in sandbox.commands)
    assert "PATH=/usr/local/bin:/usr/bin:/bin" not in sandbox.test_command
    assert "OPENAI_VERIFIER_API_KEY" not in sandbox.test_command
    assert "OPENAI_VERIFIER_BASE_URL" not in sandbox.test_command
    assert "BRAINTRUST_API_KEY" not in sandbox.test_command
    assert "REDUCTO_API_KEY" not in sandbox.test_command


@pytest.mark.asyncio
async def test_official_harbor_mode_reads_reward_json_when_txt_is_absent():
    sandbox = GraderSandbox(reward_json=b'{"reward":0,"apply_failed":1}')
    result = await grade_task(
        task(environment_mode="separate", workdir="/app"),
        sandbox,
        instruction="Solve this.",
        config=GradingConfig(poll_timeout=300, judge_retry_backoff_seconds=0),
        final_answer="answer",
    )
    assert result.score == 0.0
    assert sandbox.test_runs == 1
