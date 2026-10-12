import json
import shlex
import subprocess
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from feather_harness import FileTools, ProcessTools
from feather_harness.prompt import SKILLS_ROOT

from feather_train import E2BRuntime
from feather_train.sandbox.runtime import build_runtime, install_runtime


def test_runtime_bundle_has_tools_and_all_skills_but_no_agent_loop(tmp_path):
    archive = tmp_path / "runtime.tar.gz"
    build_runtime(archive)
    with tarfile.open(archive) as tar:
        names = tar.getnames()
    assert {name.removeprefix("skills/") for name in names if name.startswith("skills/")} == {
        str(path.relative_to(SKILLS_ROOT))
        for path in SKILLS_ROOT.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    }
    assert "runtime/read_tool.py" in names
    assert "runtime/call.py" in names
    assert not any(name.startswith("production_skills/") for name in names)
    assert not any(name.endswith(("runner.py", "model.py", "cli.py")) for name in names)


def test_prod_runtime_bundle_materializes_production_skills(tmp_path):
    archive = tmp_path / "prod-runtime.tar.gz"
    build_runtime(archive, profile="prod")
    with tarfile.open(archive) as tar:
        names = set(tar.getnames())
        tar.extractall(tmp_path / "runtime", filter="data")
    assert "production_skills/design.md" in names
    assert "production_skills/xlsx.md" in names
    assert "production_skills/icons/factory.png" in names

    workspace = tmp_path / "workspace"
    subprocess.run(
        [
            sys.executable,
            str(tmp_path / "runtime/runtime/install_skills.py"),
            str(workspace),
            "--overwrite",
            "--profile",
            "prod",
        ],
        check=True,
    )
    assert (workspace / ".agents/design.md").is_file()
    assert (workspace / ".agents/icons/factory.png").is_file()
    assert b"professional" in (workspace / ".agents/xlsx/SKILL.md").read_bytes().lower()


async def test_install_runtime_installs_skills_in_requested_workspace(tmp_path):
    archive = tmp_path / "runtime.tar.gz"
    archive.write_bytes(b"runtime bundle")
    sandbox = SimpleNamespace(
        commands=SimpleNamespace(run=AsyncMock()),
        files=SimpleNamespace(write=AsyncMock(), remove=AsyncMock()),
    )
    await install_runtime(sandbox, archive, workspace="/app")
    command = sandbox.commands.run.await_args.args[0]
    assert "install_skills.py /app --overwrite" in command


@pytest.mark.parametrize("installation_fails", [False, True])
async def test_runtime_cleanup_failure_preserves_installation_result(
    tmp_path, caplog, installation_fails
):
    archive = tmp_path / "runtime.tar.gz"
    archive.write_bytes(b"runtime bundle")
    installation_error = RuntimeError("installation failed")
    sandbox = SimpleNamespace(
        commands=SimpleNamespace(
            run=AsyncMock(side_effect=installation_error if installation_fails else None)
        ),
        files=SimpleNamespace(
            write=AsyncMock(), remove=AsyncMock(side_effect=OSError("cleanup failed"))
        ),
    )

    if installation_fails:
        with pytest.raises(RuntimeError) as raised:
            await install_runtime(sandbox, archive)
        assert raised.value is installation_error
    else:
        await install_runtime(sandbox, archive)

    sandbox.commands.run.assert_awaited_once()
    sandbox.files.remove.assert_awaited_once()
    assert "Could not remove temporary runtime archive" in caplog.text
    assert "cleanup failed" in caplog.text


async def test_private_results_are_on_host_not_in_agent_container(sandbox):
    async with E2BRuntime(
        sandbox, container=sandbox.container, host_runtime=sandbox.host_runtime
    ) as transport:
        await (
            FileTools(transport)
            .write_tool()
            .invoke(json.dumps({"file_path": "answer", "content": "yes"}))
        )
        request = Path(
            shlex.split(next(cmd for cmd in sandbox.commands.started if "remote.py run " in cmd))[3]
        )
        assert not request.exists()
        assert request.with_suffix(".result").exists()
        await sandbox.commands.run(f"docker exec {sandbox.container} test ! -e {request}")
        await sandbox.commands.run(
            f"docker exec {sandbox.container} test ! -e {request.with_suffix('.result')}"
        )
        await sandbox.commands.run(
            f"docker exec {sandbox.container} test ! -e /opt/feather/host/remote.py"
        )
        await sandbox.commands.run(
            f"docker exec {sandbox.container} test ! -S /var/run/docker.sock"
        )
        await sandbox.commands.run(
            f"""docker exec {sandbox.container} python3 -c 'import importlib.util; assert importlib.util.find_spec("feather_harness") is None' """
        )
    assert not request.with_suffix(".result").exists()


async def test_installed_tools_are_readable_but_not_writable_by_unprivileged_agent(sandbox):
    async with E2BRuntime(
        sandbox,
        container=sandbox.container,
        host_runtime=sandbox.host_runtime,
        run_as="65534",
    ) as transport:
        result = (
            await ProcessTools(transport)
            .bash_tool()
            .invoke(
                json.dumps(
                    {
                        "command": "test -r /opt/feather/runtime/read_tool.py && "
                        "test ! -w /opt/feather/runtime/read_tool.py && "
                        "test ! -w /opt/feather/runtime && test ! -w /opt/feather",
                    }
                )
            )
        )
        assert "Exit code: 0" in result.content
        files = FileTools(transport)
        written = await files.write_tool().invoke(
            json.dumps({"file_path": "answer.txt", "content": "yes"})
        )
        assert written.error is None
        read = await files.read_tool().invoke(json.dumps({"file_path": "answer.txt"}))
        assert read.content == "Contents of answer.txt:\n     1\tyes\n"


@pytest.mark.parametrize("sandbox", ["65534"], indirect=True)
async def test_transcript_save_preserves_agent_user_and_background_log_access(sandbox):
    async with E2BRuntime(
        sandbox,
        container=sandbox.container,
        host_runtime=sandbox.host_runtime,
        log_directory="/filesystem/.feather/tasks",
    ) as runtime:
        for text in ("é" * 100000, "updated"):
            content = json.dumps({"role": "user", "content": text}, ensure_ascii=False) + "\n"
            await runtime.save_transcript(content)
            output = await sandbox.commands.run(
                f"docker exec {sandbox.container} cat /filesystem/.feather/transcript.jsonl"
            )
            assert output.stdout == content
        result = (
            await ProcessTools(runtime)
            .bash_tool()
            .invoke(json.dumps({"command": "printf saved", "run_in_background": True}))
        )
        assert result.error is None
        output = await ProcessTools(runtime).task_output_tool().invoke(
            '{"task_id":"task_1","block":true,"timeout":10000}'
        )
        assert "Exit code: 0" in output.content
        assert output.content.endswith("Output:\nsaved")
