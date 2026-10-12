"""Configured judge and evidence settings reach the sandbox command."""

import shlex

import pytest

from feather_train.grading.grader import GradingConfig, grade_task

from .helpers import GraderSandbox, task


@pytest.mark.parametrize(
    ("model", "evidence_mode"),
    [
        (None, None),
        ("judge/model-v2", "direct_text"),
    ],
)
async def test_judge_environment_preserves_config_and_task_identity(
    monkeypatch, model, evidence_mode
):
    monkeypatch.setenv("OPENAI_VERIFIER_API_KEY", "test key with 'quotes'")
    monkeypatch.setenv("OPENAI_VERIFIER_BASE_URL", "https://judge.example/v1")
    grading_task = task()
    grading_task.grading_evidence_mode = evidence_mode
    sandbox = GraderSandbox()
    await grade_task(
        grading_task,
        sandbox,
        instruction="Solve this exactly.",
        config=GradingConfig(judge_model=model),
        final_answer="",
        braintrust_parent="parent span",
        braintrust_project="eval project",
    )

    # Parse the actual env invocation, preserving shell-quoted spaces and apostrophes.
    invocation = shlex.split(sandbox.test_command.split("; ", 1)[1])
    assignments = dict(token.split("=", 1) for token in invocation[1:-2] if "=" in token)
    assert invocation[-2:] == ["/bin/bash", "/tests/test.sh"]
    assert assignments["OPENAI_VERIFIER_API_KEY"] == "test key with 'quotes'"
    assert assignments["OPENAI_VERIFIER_BASE_URL"] == "https://judge.example/v1"
    assert assignments.get("OPENAI_VERIFIER_MODEL") == model
    assert assignments.get("HARBOR_GRADING_EVIDENCE_MODE") == evidence_mode
    assert assignments["HARBOR_TASK_ID"] == "task-1"
    assert assignments["HARBOR_WORLD_ID"] == "world-1"
    assert any("rm -rf -- /logs &&" in command for command in sandbox.commands)
    assert assignments["HARBOR_BRAINTRUST_PARENT"] == "parent span"
    assert assignments["BRAINTRUST_PROJECT"] == "eval project"
