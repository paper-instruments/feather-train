"""Production prose and descriptions; the execution loop is shared with training."""

import json
from dataclasses import replace
from pathlib import Path
from typing import Literal, get_args

from .core.compaction import DEFAULT_COMPACTION_PROMPT, CompactionConfig
from .prompt import RESOURCE_ROOT, build_instructions, compose_additional_instructions
from .scripts.install_skills import WORKSPACE_SKILLS_PATH, render_skills_path
from .tools import Tool

Profile = Literal["train", "prod"]

PRODUCTION_PROMPT = (RESOURCE_ROOT / "production_prompt.md").read_text().strip()
PRODUCTION_COMPACTION_PROMPT = (
    (RESOURCE_ROOT / "production_compaction_prompt.md").read_text().strip()
)
PRODUCTION_TOOL_DESCRIPTIONS = json.loads(
    (RESOURCE_ROOT / "production_tool_descriptions.json").read_text()
)
VISION_ENABLED_FACT = "Image inspection is enabled through read, including source images and rendered pages."
VISION_DISABLED_FACT = "Image inspection is disabled in this run; read handles text files only."
TRAIN_CATALOG_HEADING = "\n\n# Sandbox capability instructions\n\n"
PRODUCTION_CATALOG_HEADING = "\n\n# Tool capability instructions\n\n"


def validate_profile(profile: Profile) -> None:
    if profile not in get_args(Profile):
        raise ValueError(f"Unknown Feather profile: {profile}")


def build_production_instructions(
    *,
    skills_root: Path,
    task_instructions: str | None = None,
    additional_instructions: str | None = None,
    workspace: str = ".",
    skills_path: str = WORKSPACE_SKILLS_PATH,
    enable_image_tools: bool | None = None,
    host_facts: str | None = None,
    vision_fact: str | None = None,
) -> str:
    """Compose production instructions with optional host capabilities and skill paths."""
    prompt = render_skills_path(PRODUCTION_PROMPT, skills_path)
    if vision_fact is None and enable_image_tools is not None:
        vision_fact = VISION_ENABLED_FACT if enable_image_tools else VISION_DISABLED_FACT
    capabilities = " ".join(value for value in (vision_fact, host_facts) if value)
    if capabilities:
        prompt += f"\n\n## Host capabilities\n\n{capabilities}"
    instructions = build_instructions(
        prompt, skills_root=skills_root, skills_path=skills_path
    ).replace(
        TRAIN_CATALOG_HEADING, PRODUCTION_CATALOG_HEADING, 1
    )
    default_instructions = (RESOURCE_ROOT / "system_prompt.md").read_text().strip()
    if task_instructions and task_instructions.strip() == default_instructions:
        task_instructions = None
    return compose_additional_instructions(
        instructions, additional_instructions,
        workspace=workspace, task_instructions=task_instructions,
    )


def production_tools(tools: tuple[Tool, ...], *, image_reads: bool) -> tuple[Tool, ...]:
    """Swap in production descriptions; `read` has one for each image-read mode."""
    result = []
    for tool in tools:
        name = tool.spec.name
        if name == "read" and not image_reads:
            name = "read_without_images"
        description = PRODUCTION_TOOL_DESCRIPTIONS.get(name)
        if description is None:
            result.append(tool)
            continue
        update = {"description": description}
        result.append(replace(tool, spec=tool.spec.model_copy(update=update)))
    return tuple(result)


def production_compaction(config: CompactionConfig) -> CompactionConfig:
    """Use the production continuation prompt unless the caller supplied its own."""
    if config.prompt != DEFAULT_COMPACTION_PROMPT:
        return config
    return config.model_copy(update={"prompt": PRODUCTION_COMPACTION_PROMPT})
