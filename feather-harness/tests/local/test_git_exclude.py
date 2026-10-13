from __future__ import annotations

import subprocess

import pytest

from feather_harness.local import _exclude_harness_paths, install_skills


def _git(cwd, *args):
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout


@pytest.mark.parametrize("subdir", ["", "pkg/app"])
def test_harness_files_stay_out_of_git_add_all(tmp_path, subdir):
    _git(tmp_path, "init", "-q")
    workspace = tmp_path / subdir
    workspace.mkdir(parents=True, exist_ok=True)
    install_skills(workspace)
    (workspace / ".feather").mkdir()
    (workspace / ".feather" / "transcript.jsonl").write_text("{}\n")
    (workspace / "solution.py").write_text("print(1)\n")

    _exclude_harness_paths(workspace)
    _exclude_harness_paths(workspace)
    _git(workspace, "add", "-A")

    staged = _git(tmp_path, "diff", "--cached", "--name-only").split()
    assert staged == [f"{subdir}/solution.py".lstrip("/")]
    exclude = (tmp_path / ".git" / "info" / "exclude").read_text().splitlines()
    assert len([line for line in exclude if line.endswith(".agents/")]) == 1


def test_non_repository_workspace_is_left_untouched(tmp_path):
    _exclude_harness_paths(tmp_path)
    assert list(tmp_path.iterdir()) == []
