import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from feather_harness.production import build_production_instructions
from tests.local.test_cli import cli, model_server, response


@pytest.mark.parametrize("configuration", ["config", "flag"])
def test_cli_selects_task_python_without_changing_runtime_or_system_prompt(tmp_path, configuration):
    project = tmp_path / "project env"
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(project)], check=True)
    chosen = project / "bin" / "custom-python"
    chosen.symlink_to("python")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    program = (
        "import json,sys,importlib.util; "
        "print(json.dumps({'prefix':sys.prefix,'executable':sys.executable,"
        "'feather':importlib.util.find_spec('feather_harness') is not None}))"
    )
    command = " && ".join(
        f"{executable} -c {shlex.quote(program)} > {filename}"
        for executable, filename in (
            ("python", "python.json"), ("python3", "python3.json"),
            ('"$FEATHER_PYTHON"', "feather.json"),
        )
    )
    config = tmp_path / "config.json"
    config.write_text(json.dumps({
        "python": str(chosen if configuration == "config" else tmp_path / "missing"),
    }))
    args = ["--config", str(config), "--workdir", str(workspace), "--model", "test"]
    if configuration == "flag":
        args.extend(("--python", str(chosen.relative_to(tmp_path))))
    with model_server([
        (200, response(tools=[("bash", {"command": command})])),
        (200, response("Verified.")),
    ]) as (url, requests):
        completed = cli(tmp_path, *args, "--base-url", url, "Check the environments.")
    assert completed.returncode == 0, completed.stderr
    for name in ("python.json", "python3.json"):
        result = json.loads((workspace / name).read_text())
        assert Path(result["prefix"]) == project
        assert Path(result["executable"]) == chosen
        assert result["feather"] is False
    feather = json.loads((workspace / "feather.json").read_text())
    assert Path(feather["prefix"]) == Path(sys.prefix)
    assert feather["feather"] is True
    assert requests[0][1]["messages"][0]["content"] == build_production_instructions(
        skills_root=workspace / ".agents", workspace=str(workspace),
    )
    assert requests[0][1]["messages"][1]["content"] == "Check the environments."
    bash = next(tool for tool in requests[0][1]["tools"] if tool["function"]["name"] == "bash")
    assert "$FEATHER_PYTHON" in bash["function"]["description"]


@pytest.mark.parametrize("kind", ["missing", "directory", "not_executable"])
def test_cli_rejects_invalid_task_python_before_model_request(tmp_path, kind):
    chosen = tmp_path / "selected-python"
    if kind == "directory":
        chosen.mkdir()
    elif kind == "not_executable":
        chosen.write_text("not executable")
        chosen.chmod(0o600)
    with model_server([]) as (url, requests):
        completed = cli(
            tmp_path, "--model", "test", "--base-url", url, "--python", str(chosen), "Task",
        )
    assert completed.returncode == 1
    assert "Task Python must be an existing executable file" in completed.stderr
    assert requests == []


@pytest.mark.parametrize("selected", [False, True], ids=["feather_python", "task_python"])
def test_pip_never_falls_through_to_caller_environment(tmp_path, monkeypatch, selected):
    caller = tmp_path / "caller bin"
    caller.mkdir()
    for name in ("pip", "pip3"):
        (caller / name).write_text("#!/bin/sh\necho caller pip\n")
        (caller / name).chmod(0o755)
    monkeypatch.setenv("PATH", f"{caller}{os.pathsep}{os.environ['PATH']}")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    args = ["--workdir", str(workspace), "--model", "test"]
    if selected:
        project = tmp_path / "project env"
        subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(project)], check=True)
        packages = subprocess.check_output(
            [str(project / "bin/python"), "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"],
            text=True,
        ).strip()
        (Path(packages) / "pip.py").write_text("import sys\nprint('pip', sys.prefix)\n")
        args.extend(("--python", str(project / "bin/python")))
    command = 'for tool in pip pip3; do "$tool" install demo >> pip.log 2>&1; echo "$tool=$?" >> pip.log; done'
    with model_server([
        (200, response(tools=[("bash", {"command": command})])),
        (200, response("Done.")),
    ]) as (url, _):
        completed = cli(tmp_path, *args, "--base-url", url, "Install demo.")
    assert completed.returncode == 0, completed.stderr
    log = (workspace / "pip.log").read_text()
    assert "caller pip" not in log
    if selected:
        assert log.splitlines() == [f"pip {project}", "pip=0", f"pip {project}", "pip3=0"]
    else:
        assert log.count("pip is unavailable") == 2
        assert "pip=1" in log and "pip3=1" in log
