"""Trained prompt and skill discovery text, without SDK capability assembly."""

from __future__ import annotations

import hashlib
from pathlib import Path

RESOURCE_ROOT = Path(__file__).parent
SKILLS_ROOT = RESOURCE_ROOT / "skills"


def build_instructions(
    instructions: str | None = None,
    *,
    skills_root: Path = SKILLS_ROOT,
    skills_path: str = ".agents",
) -> str:
    if instructions is None:
        instructions = (RESOURCE_ROOT / "system_prompt.md").read_text()
    fragments = []
    if instructions:
        fragments.append(f"# Agent instructions\n\n{instructions}")
    skills = []
    for path in sorted(skills_root.glob("*/SKILL.md")):
        content = path.read_text()
        metadata = {}
        lines = content.splitlines()
        if lines and lines[0].strip() == "---":
            for line in lines[1:]:
                if line.strip() == "---":
                    break
                if ":" in line:
                    key, value = line.split(":", 1)
                    metadata[key.strip()] = value.strip()
        name = metadata.get("name", path.parent.name)
        description = metadata.get("description", "No description provided.")
        skills.append((name, f"- {name}: {description} (file: {skills_path}/{name}/SKILL.md)"))
    if skills:
        intro = (
            "Skill directories are local reference packages. Each entry below gives a name, "
            "description, and `SKILL.md` entrypoint. Read a matching `SKILL.md` for package-specific "
            "facts; linked files may provide additional references, scripts, assets, or templates."
        )
        skill_lines = [line for _, line in sorted(skills)]
        text = "\n".join(["## Skills", intro, "### Available skills", *skill_lines])
        fragments.append(f"# Sandbox capability instructions\n\n{text}")
    # The disabled SDK filesystem preamble still contributed this final separator.
    fragments.append("")
    return "\n\n".join(fragments)


def compose_additional_instructions(
    instructions: str, additional_instructions: str | None, *, workspace: str,
    task_instructions: str | None = None,
) -> str:
    additional = [task_instructions] if task_instructions else []
    if additional_instructions:
        additional.append(additional_instructions.replace("{{ workspace }}", workspace))
    if not additional:
        return instructions
    prompt = "\n\n".join(additional)
    return f"{instructions.rstrip()}\n\n# Additional task instructions\n\n{prompt}"


def prompt_metadata(instructions: str, *, profile: str) -> dict:
    return {
        "profile": profile,
        "system_prompt_sha256": hashlib.sha256(instructions.encode()).hexdigest(),
    }
