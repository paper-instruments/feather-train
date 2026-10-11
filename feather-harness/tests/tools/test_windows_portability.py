"""File-tool behavior that must also hold on native Windows.

Windows-only paths are exercised through the same fallbacks the tools take
there, so these run on every platform.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path, PureWindowsPath
from types import SimpleNamespace

import pytest

from feather_harness import FileTools, LocalFileRuntime
from feather_harness.scripts import _file_worker

sys.path.insert(0, str(Path(_file_worker.__file__).parent))
import _files  # noqa: E402
import _search  # noqa: E402


def test_write_fallback_creates_missing_parents(tmp_path, monkeypatch):
    # Windows has no dir_fd/O_NOFOLLOW, so writes take this fallback.
    monkeypatch.setattr(_files, "_supports_safe_relative_write", lambda: False)
    target = _files.require_write_target(tmp_path, (), "reports/q3/summary.txt", allow_missing=True)

    _files.write_text(target, "done", "utf-8", create_parents=True)

    assert (tmp_path / "reports" / "q3" / "summary.txt").read_text() == "done"


def test_write_fallback_refuses_targets_outside_the_roots(tmp_path, monkeypatch):
    monkeypatch.setattr(_files, "_supports_safe_relative_write", lambda: False)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside" / "new" / "file.txt"

    with pytest.raises(_files.FileToolError):
        target = _files.require_write_target(workspace, (), str(outside), allow_missing=True)
        _files.write_text(target, "x", "utf-8", create_parents=True)

    assert not (tmp_path / "outside").exists()


@pytest.mark.skipif(os.name == "nt", reason="drive letters are native paths on Windows")
def test_drive_letter_paths_stay_rejected_off_windows(tmp_path):
    with pytest.raises(_files.FileToolError):
        _files.require_path(tmp_path, (), "C:/Users/someone/notes.txt")


@pytest.mark.skipif(os.name != "nt", reason="Windows drive-letter paths")
def test_drive_letter_paths_inside_the_workspace_are_accepted(tmp_path):
    (tmp_path / "notes.txt").write_text("hello")

    candidate, _, metadata = _files.require_path(tmp_path, (), str(tmp_path / "notes.txt"))

    assert candidate == tmp_path / "notes.txt"
    assert metadata is not None


def test_git_bash_drive_paths_map_to_windows_drive_paths():
    assert _files._from_msys_path("/c/Users/me/project/a.txt") == "C:/Users/me/project/a.txt"
    assert _files._from_msys_path("/d") == "D:/"
    assert _files._from_msys_path("/cdrive/x") == "/cdrive/x"
    assert _files._from_msys_path("notes/a.txt") == "notes/a.txt"


def test_pattern_validation_needs_no_null_device():
    _search.validate_glob("*.py")
    _search.validate_regex("foo|bar")
    with pytest.raises(_files.FileToolError):
        _search.validate_glob("{unclosed")
    with pytest.raises(_files.FileToolError):
        _search.validate_regex("(")


def test_windows_worker_cleanup_kills_the_process_tree(monkeypatch):
    calls = []
    monkeypatch.setattr(_file_worker.sys, "platform", "win32")
    monkeypatch.setattr(_file_worker.subprocess, "run", lambda args, **kwargs: calls.append(args))

    _file_worker._kill_process_tree(4242)

    assert len(calls) == 1
    assert Path(calls[0][0]).name == "taskkill.exe"
    assert calls[0][1:] == ["/PID", "4242", "/T", "/F"]


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process groups")
def test_posix_worker_cleanup_kills_the_process_group():
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True
    )
    _file_worker._kill_process_tree(process.pid)
    assert process.wait(timeout=5) != 0


def test_windows_workspace_reads_do_not_require_posix_user_ids(tmp_path, monkeypatch):
    target = tmp_path / "notes.txt"
    target.write_text("hello")
    monkeypatch.setattr(
        _files, "os", SimpleNamespace(name="nt", fsencode=os.fsencode, path=os.path)
    )

    candidate, _, metadata = _files.require_path(tmp_path, (), "notes.txt")

    assert candidate == target
    assert metadata is not None


@pytest.mark.skipif(os.name == "nt", reason="POSIX no-follow writes")
@pytest.mark.parametrize("swap_parent", [False, True], ids=["file", "parent"])
def test_safe_write_rejects_symlinks_swapped_in_after_validation(tmp_path, swap_parent):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    parent = workspace / "reports"
    parent.mkdir()
    target = parent / "notes.txt"
    target.write_text("original")
    outside = tmp_path / "outside"
    outside.mkdir()
    protected = outside / "notes.txt"
    protected.write_text("protected")
    validated = _files.require_write_target(workspace, (), str(target))
    target.unlink()
    if swap_parent:
        parent.rmdir()
        parent.symlink_to(outside, target_is_directory=True)
    else:
        target.symlink_to(protected)

    with pytest.raises(_files.FileToolError, match="could not be opened safely"):
        _files.write_text(validated, "replacement", "utf-8")

    assert protected.read_text() == "protected"


@pytest.mark.skipif(os.name == "nt", reason="POSIX symlinks")
def test_alias_repointed_after_validation_still_writes_the_approved_target(tmp_path):
    workspace = tmp_path / "workspace"
    (workspace / "a").mkdir(parents=True)
    (workspace / "b").mkdir()
    alias = workspace / "alias"
    alias.symlink_to(workspace / "a", target_is_directory=True)
    validated = _files.require_write_target(workspace, (), "alias/new.txt", allow_missing=True)
    alias.unlink()
    alias.symlink_to(workspace / "b", target_is_directory=True)

    _files.write_text(validated, "x", "utf-8")

    assert (workspace / "a" / "new.txt").read_text() == "x"
    assert not (workspace / "b" / "new.txt").exists()


@pytest.mark.skipif(os.name == "nt", reason="POSIX symlinks")
def test_write_fallback_follows_allowed_links_but_refuses_links_swapped_in(tmp_path, monkeypatch):
    monkeypatch.setattr(_files, "_supports_safe_relative_write", lambda: False)
    workspace = tmp_path / "workspace"
    (workspace / "real").mkdir(parents=True)
    (workspace / "link").symlink_to(workspace / "real", target_is_directory=True)
    outside = tmp_path / "outside"
    outside.mkdir()

    allowed = _files.require_write_target(workspace, (), "link/new.txt", allow_missing=True)
    _files.write_text(allowed, "hi", "utf-8")
    assert (workspace / "real" / "new.txt").read_text() == "hi"

    swapped = _files.require_write_target(workspace, (), "real/later.txt", allow_missing=True)
    (workspace / "real" / "new.txt").unlink()
    (workspace / "real").rmdir()
    (workspace / "real").symlink_to(outside, target_is_directory=True)
    with pytest.raises(_files.FileToolError, match="could not be opened safely"):
        _files.write_text(swapped, "x", "utf-8")
    assert list(outside.iterdir()) == []


@pytest.mark.skipif(os.name != "nt", reason="Native Windows Git Bash paths")
@pytest.mark.parametrize("operation", ["edit", "write"])
async def test_git_bash_paths_preserve_read_revisions_for_mutations(tmp_path, operation):
    target = tmp_path / "notes.txt"
    target.write_text("before")
    native = target.as_posix()
    msys = "/" + native[0].lower() + native[2:]
    tools = FileTools(LocalFileRuntime(tmp_path))
    read = await tools.read_tool().invoke(json.dumps({"file_path": msys}))
    assert read.error is None
    if operation == "edit":
        result = await tools.edit_tool().invoke(
            json.dumps({"file_path": msys, "old_string": "before", "new_string": "after"})
        )
    else:
        result = await tools.write_tool().invoke(json.dumps({"file_path": msys, "content": "after"}))

    assert result.error is None
    assert target.read_text() == "after"


def test_search_displays_granted_paths_on_another_windows_drive(monkeypatch):
    monkeypatch.setattr(_search, "Path", PureWindowsPath)
    search = _search.BoundedSearch("D:/allowed", "C:/workspace", 10)

    assert search.display_path("D:/allowed/notes.txt") == "D:/allowed/notes.txt"
    assert search.display_path("C:/workspace/notes.txt") == "notes.txt"


def test_file_paths_and_saved_output_without_posix_apis(tmp_path, monkeypatch):
    missing = {"getuid", "O_DIRECTORY", "O_NOFOLLOW", "fchmod"}
    windows_os = SimpleNamespace(**{name: getattr(os, name) for name in dir(os) if name not in missing})
    windows_os.name = "nt"
    monkeypatch.setattr(_files, "os", windows_os)
    monkeypatch.setattr(_files.tempfile, "gettempdir", lambda: str(tmp_path))
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "notes.txt").write_text("hello")

    candidate, _, metadata = _files.require_path(workspace, (), "notes.txt")
    assert candidate == workspace / "notes.txt"
    assert metadata is not None

    content = "x" * (_files.MAX_OUTPUT_CHARS + 100)
    head, notice, tail, saved_path = _files.save_output_preview(content, workspace=workspace)
    assert len(head + notice + tail) <= _files.MAX_OUTPUT_CHARS
    assert saved_path is not None
    assert Path(saved_path).read_text() == content
    assert _files.require_path(workspace, (), saved_path)[0] == Path(saved_path)
    with pytest.raises(_files.FileToolError):
        _files.require_write_target(workspace, (), saved_path)


@pytest.mark.skipif(os.name == "nt", reason="simulate junctions with POSIX symlinks")
def test_output_directory_junction_does_not_grant_its_target(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("private")
    directory = tmp_path / "tool_output"
    directory.symlink_to(outside, target_is_directory=True)
    is_symlink = Path.is_symlink
    monkeypatch.setattr(Path, "is_symlink", lambda path: False if path == directory else is_symlink(path))
    monkeypatch.setattr(_files, "_output_directory", lambda root: directory)

    with pytest.raises(_files.FileToolError):
        _files.require_file(workspace, (), str(directory / "secret.txt"))
