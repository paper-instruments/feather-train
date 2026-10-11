"""Generic guidance shares one final section without altering task text."""

import hashlib

import pytest

from feather_harness.prompt import compose_additional_instructions, prompt_metadata


def test_additional_guidance_expands_workspace_after_literal_task_instructions():
    task = "Keep {{ workspace }} literal in task instructions.\n"
    system = compose_additional_instructions(
        "Base\n", "Use inputs under {{ workspace }}.\n",
        workspace="/task", task_instructions=task,
    )
    assert system == (
        "Base\n\n# Additional task instructions\n\n"
        + task + "\n\nUse inputs under /task.\n"
    )
    assert prompt_metadata(system, profile="train") == {
        "profile": "train",
        "system_prompt_sha256": hashlib.sha256(system.encode()).hexdigest(),
    }


@pytest.mark.parametrize("additional", [None, ""])
def test_omitted_or_empty_guidance_preserves_base_bytes(additional):
    base = "Original system prompt.\n\n"
    assert compose_additional_instructions(base, additional, workspace="/task") == base
