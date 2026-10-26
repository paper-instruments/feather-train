"""Preserve bounded solver outputs without changing benchmark outcomes."""

import hashlib
import json
import sys
import tarfile
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import verifiers.v1 as vf
from feather_harness import CompactionConfig
from feather_harness.core.runner import RunResult
from pydantic import ValidationError

from feather_train import deliverable_capture as capture
from feather_train import harness as module
from feather_train.grading.types import GradeResult
from feather_train.harness import FeatherHarness, FeatherHarnessConfig
from feather_train.sandbox import capture_files as worker
from feather_train.tasks.taskset import FeatherTaskData


def test_archive_preserves_binary_and_app_state_without_following_links(tmp_path):
    root = tmp_path / "filesystem"
    root.mkdir()
    binary = bytes(range(256))
    (root / "answer.xlsx").write_bytes(binary)
    (root / ".apps_data").mkdir()
    (root / ".apps_data/mail.json").write_text('{"sent": true}')
    (root / ".agents").mkdir()
    (root / ".agents/secret").write_text("runtime only")
    private = tmp_path / "private"
    private.write_text("grader private")
    (root / "private-link").symlink_to(private)
    archive = tmp_path / "out.tar.gz"

    inventory = worker.capture_files([str(root)], str(archive))
    with tarfile.open(archive) as packed:
        names = packed.getnames()
        assert not any(Path(name).name == "private-link" or ".agents" in Path(name).parts for name in names)
        answer = next(name for name in names if name.endswith("answer.xlsx"))
        assert packed.extractfile(answer).read() == binary
        assert any(name.endswith(".apps_data/mail.json") for name in names)
    assert inventory == {"files": 2, "file_bytes": 270, "skipped_special_files": 1}


def test_btb_capture_excludes_other_workspace_and_verifier_files(tmp_path):
    root = tmp_path / "filesystem"
    deliverables = root / "banker_workspace/deliverables"
    deliverables.mkdir(parents=True)
    (deliverables / "answer.txt").write_text("final")
    (root / ".banker-verifier-workspace").mkdir()
    (root / ".banker-verifier-workspace/private").write_text("private")
    (root / "banker_workspace/input.txt").write_text("input")
    archive = tmp_path / "out.tar.gz"
    worker.capture_files([str(deliverables)], str(archive))
    with tarfile.open(archive) as packed:
        assert len(packed.getnames()) == 1
        assert packed.getnames()[0].endswith("deliverables/answer.txt")


@pytest.mark.parametrize("limit", ["MAX_FILE_BYTES", "MAX_TOTAL_BYTES", "MAX_FILES"])
def test_capture_fails_instead_of_truncating_large_deliverables(tmp_path, monkeypatch, limit):
    root = tmp_path / "files"
    root.mkdir()
    (root / "a").write_bytes(b"123")
    monkeypatch.setattr(worker, limit, 1 if limit != "MAX_FILES" else 0)
    with pytest.raises(ValueError, match="limit exceeded"):
        worker.capture_files([str(root)], str(tmp_path / "out.tar.gz"))


def test_root_symlink_is_rejected(tmp_path):
    (tmp_path / "files").mkdir()
    (tmp_path / "alias").symlink_to(tmp_path / "files", target_is_directory=True)
    with pytest.raises(OSError):
        worker.capture_files([str(tmp_path / "alias")], str(tmp_path / "out.tar.gz"))


@pytest.mark.parametrize("root", ["/tests", "/filesystem/../tests", "/filesystem/.agents", "/filesystem2", "relative"])
def test_capture_roots_cannot_include_grader_or_runtime(root):
    with pytest.raises(ValidationError):
        FeatherHarnessConfig(capture_deliverables=True, capture_paths=(root,))


@pytest.mark.parametrize(
    "roots",
    [
        ("/filesystem/reports", "/filesystem/reports"),
        ("/filesystem", "/filesystem/reports"),
        ("/filesystem/reports", "/filesystem"),
    ],
)
def test_capture_roots_must_not_overlap(roots):
    with pytest.raises(ValidationError, match="capture_paths must not overlap"):
        FeatherHarnessConfig(capture_deliverables=True, capture_paths=roots)


def test_capture_roots_allow_distinct_siblings():
    config = FeatherHarnessConfig(
        capture_deliverables=True,
        capture_paths=("/filesystem/reports", "/filesystem/images"),
    )

    assert config.capture_paths == ("/filesystem/reports", "/filesystem/images")


@pytest.mark.parametrize("commit_fails", [False, True])
async def test_receipt_is_durable_only_after_volume_commit(tmp_path, monkeypatch, commit_fails):
    data = b"\x00binary archive\xff"
    async def download(world, roots, destination):
        destination.write_bytes(data)
        return {"files": 1, "file_bytes": 20, "skipped_special_files": 0, "archive_bytes": len(data), "archive_sha256": hashlib.sha256(data).hexdigest()}
    monkeypatch.setattr(capture, "_download_archive", download)
    commits = []
    async def commit():
        commits.append(True)
        assert len(list(tmp_path.glob("run/task/trace/files.tar.gz"))) == 1
        if commit_fails:
            raise OSError("commit failed")
    volume = SimpleNamespace(commit=SimpleNamespace(aio=commit))
    monkeypatch.setitem(sys.modules, "modal", SimpleNamespace(Volume=SimpleNamespace(from_name=lambda name: volume)))
    task = SimpleNamespace(
        eval_run_id="run", task_id="task", world_id="world", hf_dataset="org/data",
        hf_revision="a" * 40, e2b_snapshot_id="snapshot",
    )
    config = FeatherHarnessConfig(
        capture_deliverables=True, capture_paths=("/filesystem",),
        capture_output_dir=str(tmp_path), capture_volume_name="volume",
    )
    result = await capture.capture_deliverables(
        None, task, trace_id="trace", final_answer="answer", model="test",
        sampling={"top_p": 0.95}, config=config,
    )
    assert len(commits) == 1
    assert result["status"] == ("failed" if commit_fails else "complete")
    assert result["archive_sha256"] == hashlib.sha256(data).hexdigest()
    assert (tmp_path / "run/task/trace/files.tar.gz").read_bytes() == data
    manifest = json.loads((tmp_path / "run/task/trace/manifest.json").read_text())
    assert manifest["status"] == "captured"  # Publication precedes durability acknowledgement.
    assert manifest["sampling"] == {"top_p": 0.95}
    assert (tmp_path / "run/task/trace/final_response.txt").read_text() == "answer"


async def test_download_failure_never_publishes_partial_capture(tmp_path, monkeypatch):
    async def download(*args):
        raise TimeoutError("download stalled")
    monkeypatch.setattr(capture, "_download_archive", download)
    task = SimpleNamespace(eval_run_id="run", task_id="task", world_id="world", hf_dataset="org/data", hf_revision="a" * 40, e2b_snapshot_id=None)
    result = await capture.capture_deliverables(
        None, task, trace_id="trace", final_answer="answer", model="test", sampling={},
        config=FeatherHarnessConfig(capture_paths=("/filesystem",), capture_output_dir=str(tmp_path)),
    )
    assert result["status"] == "failed"
    assert result["error_type"] == "TimeoutError"
    assert not list(tmp_path.rglob("files.tar.gz"))


async def test_local_capture_succeeds_without_modal_installed(tmp_path, monkeypatch):
    async def download(world, roots, destination):
        destination.write_bytes(b"archive")
        return {"files": 1, "file_bytes": 7, "skipped_special_files": 0}

    monkeypatch.setattr(capture, "_download_archive", download)
    monkeypatch.setitem(sys.modules, "modal", None)
    task = SimpleNamespace(
        eval_run_id="run", task_id="task", world_id="world", hf_dataset="org/data",
        hf_revision="a" * 40, e2b_snapshot_id="snapshot",
    )
    result = await capture.capture_deliverables(
        None, task, trace_id="trace", final_answer="answer", model="test", sampling={},
        config=FeatherHarnessConfig(
            capture_deliverables=True, capture_paths=("/filesystem",),
            capture_output_dir=str(tmp_path),
        ),
    )
    assert result["status"] == "complete"
    assert result["storage"] == "local"
    assert (tmp_path / "run/task/trace/files.tar.gz").read_bytes() == b"archive"


def test_attempts_are_separate_and_existing_capture_cannot_be_overwritten(tmp_path):
    for attempt, answer in (("first", "one"), ("retry", "two")):
        temporary = tmp_path / (".capture-" + attempt)
        temporary.mkdir()
        (temporary / "files.tar.gz").write_bytes(answer.encode())
        capture._publish(tmp_path / attempt, temporary, answer, {"trace_id": attempt})
    with pytest.raises(OSError):
        temporary = tmp_path / "duplicate"
        temporary.mkdir()
        capture._publish(tmp_path / "first", temporary, "wrong", {})
    assert (tmp_path / "first/final_response.txt").read_text() == "one"
    assert (tmp_path / "retry/final_response.txt").read_text() == "two"
    assert not list(tmp_path.glob(".capture-*"))


@pytest.mark.parametrize("enabled,failed", [(False, False), (True, False), (True, True)])
async def test_capture_precedes_isolation_and_failure_preserves_grade(tmp_path, monkeypatch, enabled, failed):
    archive = tmp_path / "runtime.tar.gz"
    archive.write_bytes(b"unused")
    task = FeatherTaskData(
        idx=0, task_id="task", task_name="task", world_id="world", source="benchmark",
        hf_dataset="org/data", hf_revision="a" * 40, eval_run_id="eval-run",
        prompt=[{"role": "user", "content": "Create output"}], system_prompt="Instructions",
        requires_web_search=False, network_mode="public", mcp_servers=[],
    )
    trace = vf.Trace(task=vf.TraceTask(type="FeatherTask", data=task), agent=vf.AgentInfo(model="test", sampling=vf.Sampling()))
    harness = FeatherHarness(FeatherHarnessConfig(
        runtime_archive=archive, enable_braintrust=False, capture_deliverables=enabled,
        capture_paths=("/filesystem",), compaction=CompactionConfig(enabled=False),
    ))
    phases = []
    async def prepare(task):
        phases.append("isolation")
        return object()
    @asynccontextmanager
    async def start(*args, **kwargs):
        yield SimpleNamespace(sandbox=SimpleNamespace(sandbox_id="sandbox"), runtime=SimpleNamespace(mcp_servers=[]), prepare_grading=prepare)
        phases.append("close")
    async def install(*args, **kwargs):
        pass
    async def agent(*args, **kwargs):
        phases.append("agent")
        return RunResult(final_answer="answer", stop_reason="completed", num_turns=1, compactions=0, segments=())
    async def download(world, roots, destination):
        assert phases == ["agent"]
        phases.append("capture")
        if failed:
            raise TimeoutError("download failed")
        destination.write_bytes(b"archive")
        return {"files": 0, "file_bytes": 0, "skipped_special_files": 0}
    async def grade(*args, **kwargs):
        phases.append("grade")
        return GradeResult(score=0.8, num_met=4, num_criteria=5, criteria=(), stdout="", failed_attempts=[])
    monkeypatch.setattr(module.ComposeSandbox, "start", start)
    monkeypatch.setattr(module, "install_runtime", install)
    monkeypatch.setattr(module, "E2BRuntime", lambda *args, **kwargs: object())
    monkeypatch.setattr(module, "run_e2b", agent)
    monkeypatch.setattr(module, "grade_task", grade)
    monkeypatch.setattr(capture, "_download_archive", download)
    harness.config.capture_output_dir = str(tmp_path / "captures")
    outcome = await harness.launch(
        SimpleNamespace(model="test", sampling=vf.Sampling()), trace, None,
        "https://model.invalid/v1", "unused", {},
    )
    assert outcome.exit_code == 0
    assert trace.info["grading_score"] == 0.8
    assert phases == (["agent", "capture", "isolation", "grade", "close"] if enabled else ["agent", "isolation", "grade", "close"])
    if enabled:
        assert trace.info["deliverable_capture"]["status"] == ("failed" if failed else "complete")
    else:
        assert "deliverable_capture" not in trace.info


async def test_remote_worker_isolated_from_task_imports_and_stream_is_closed(tmp_path, monkeypatch):
    import asyncio
    import shutil
    import subprocess

    from feather_train.sandbox.lifecycle import CommandOutput, SandboxIO

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "json.py").write_text('raise RuntimeError("workspace shadow")')
    (workspace / "answer.bin").write_bytes(bytes(range(256)))
    stream_closed = []
    remote_paths = []

    async def execute(session, *args, **kwargs):
        if args[0] == "python3":
            result = await asyncio.to_thread(subprocess.run, (sys.executable, *args[1:]), cwd=workspace, capture_output=True, text=True)
            remote_paths.append(Path(args[-1]))
            return CommandOutput(result.returncode, result.stdout, result.stderr)
        if args[0] == "docker":
            shutil.copyfile(args[-2].split(":", 1)[1], args[-1])
            remote_paths.append(Path(args[-1]))
        elif args[0] == "rm":
            Path(args[-1]).unlink(missing_ok=True)
        return CommandOutput(0, "", "")

    @asynccontextmanager
    async def stream(data):
        async def chunks():
            for i in range(0, len(data), 37):
                yield data[i:i + 37]
        try:
            yield chunks()
        finally:
            stream_closed.append(True)

    async def read(path, **kwargs):
        assert kwargs["format"] == "stream"
        return stream(Path(path).read_bytes())

    monkeypatch.setattr(SandboxIO, "exec", execute)
    world = SimpleNamespace(sandbox=SimpleNamespace(files=SimpleNamespace(read=read)))
    archive = tmp_path / "archive.tar.gz"
    receipt = await capture._download_archive(world, (str(workspace),), archive)
    assert receipt["archive_sha256"] == hashlib.sha256(archive.read_bytes()).hexdigest()
    with tarfile.open(archive) as packed:
        member = next(name for name in packed.getnames() if name.endswith("answer.bin"))
        assert packed.extractfile(member).read() == bytes(range(256))
    assert stream_closed == [True]
    assert all(not path.exists() for path in remote_paths)


@pytest.mark.parametrize("source,run_id", [("factory", "run"), ("benchmark", None), ("benchmark", "../run")])
async def test_capture_rejects_training_and_unsafe_run_identity_before_sandbox(tmp_path, source, run_id):
    from feather_train.errors import NonRetryableHarnessError

    runtime = tmp_path / "runtime.tar.gz"
    runtime.write_bytes(b"unused")
    harness = FeatherHarness(FeatherHarnessConfig(
        runtime_archive=runtime, enable_braintrust=False,
        capture_deliverables=True, capture_paths=("/filesystem",),
    ))
    task = SimpleNamespace(source=source, eval_run_id=run_id, task_id="task")
    with pytest.raises(NonRetryableHarnessError, match="benchmark eval"):
        await harness.execute_task(
            SimpleNamespace(model="test"), SimpleNamespace(id="trace"), task,
            endpoint="unused", secret="unused",
        )
