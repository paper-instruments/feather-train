from __future__ import annotations

from pathlib import Path

import pytest

from feather_harness.local import install_skills
from feather_harness.prompt import SKILLS_ROOT
from feather_harness.scripts.install_skills import (
    BUNDLE_ROOT,
    install_production_skill_tree,
    install_production_skills,
    install_task_skills,
)


@pytest.mark.parametrize("host_owned", [False, True])
def test_production_guidance_remains_utf8_with_a_windows_default_encoding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, host_owned: bool
) -> None:
    original_read_text = Path.read_text

    def windows_read_text(
        self: Path, encoding: str | None = None, errors: str | None = None
    ) -> str:
        return original_read_text(self, encoding=encoding or "cp1252", errors=errors)

    monkeypatch.setattr(Path, "read_text", windows_read_text)
    workspace = tmp_path / "Résumé – 工作"
    root = workspace if host_owned else workspace / ".agents"
    skills_path = root.as_posix() if host_owned else ".agents"
    if host_owned:
        install_production_skill_tree(root)
    else:
        install_skills(workspace)
        install_production_skills(workspace)

    for guide in (BUNDLE_ROOT / "production_skills").glob("*.md"):
        guidance = guide.read_text(encoding="utf-8").replace(".agents/", f"{skills_path}/")
        if guide.name == "design.md":
            installed = root / guide.name
            expected = guidance.encode("utf-8")
        else:
            installed = root / guide.stem / "SKILL.md"
            manual = (SKILLS_ROOT / guide.stem / "SKILL.md").read_bytes()
            expected = manual.rstrip(b"\n") + b"\n\n" + guidance.encode("utf-8")
        assert installed.read_bytes() == expected


def test_install_skills_preserves_complete_tree_and_accepts_repeat_install(tmp_path):
    install_skills(tmp_path)
    install_skills(tmp_path)
    expected = {
        path.relative_to(SKILLS_ROOT): path.read_bytes()
        for path in SKILLS_ROOT.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    }
    actual = {
        path.relative_to(tmp_path / ".agents"): path.read_bytes()
        for path in (tmp_path / ".agents").rglob("*")
        if path.is_file()
    }
    assert actual == expected
    assert expected


@pytest.mark.parametrize("conflict", ["modified", "symlink", "parent_file"])
def test_skill_conflict_is_detected_before_installation_changes_workspace(tmp_path, conflict):
    source = sorted(SKILLS_ROOT.rglob("SKILL.md"))[-1]
    target = tmp_path / ".agents" / source.relative_to(SKILLS_ROOT)
    if conflict == "parent_file":
        (tmp_path / ".agents").write_text("user file")
    else:
        target.parent.mkdir(parents=True)
        if conflict == "modified":
            target.write_text("local edits")
        else:
            outside = tmp_path / "user-file"
            outside.write_text("user file")
            target.symlink_to(outside)
    before = {
        str(path.relative_to(tmp_path)): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    with pytest.raises(FileExistsError):
        install_skills(tmp_path)
    after = {
        str(path.relative_to(tmp_path)): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    assert after == before


def test_install_task_skills_copies_harbor_skills_beside_bundled(tmp_path):
    install_skills(tmp_path)
    extra = tmp_path / "harbor-skills" / "harbor-task"
    extra.mkdir(parents=True)
    (extra / "SKILL.md").write_text("---\nname: harbor-task\ndescription: Extra.\n---\n")
    install_task_skills(tmp_path, extra.parent)
    assert (tmp_path / ".agents" / "harbor-task" / "SKILL.md").read_text() == (
        "---\nname: harbor-task\ndescription: Extra.\n---\n"
    )


def test_install_task_skills_fails_on_bundled_name_collision(tmp_path):
    install_skills(tmp_path)
    bundled = next(path for path in (tmp_path / ".agents").iterdir() if path.is_dir())
    extra = tmp_path / "harbor-skills" / bundled.name
    extra.mkdir(parents=True)
    (extra / "SKILL.md").write_text("collision")
    with pytest.raises(FileExistsError, match="conflicts with installed content"):
        install_task_skills(tmp_path, extra.parent)
