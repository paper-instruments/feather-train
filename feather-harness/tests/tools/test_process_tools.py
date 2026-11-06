import json
import re
import shlex
import sys
from pathlib import Path

import pytest


def python(command):
    return f"{shlex.quote(sys.executable)} -c {shlex.quote(command)}"


async def test_foreground_merges_output_and_preserves_exit_code(process_tools):
    result = await process_tools.bash_tool().invoke(
        json.dumps({"command": "printf out; printf err >&2; exit 7"})
    )
    assert result.content == "State: exited\nExit code: 7\nOutput:\nouterr"


async def test_each_command_gets_a_fresh_shell(process_tools):
    await process_tools.bash_tool().invoke(json.dumps({"command": "export VALUE=old"}))
    result = await process_tools.bash_tool().invoke(
        json.dumps({"command": 'printf "${VALUE-unset}"'})
    )
    assert result.content.endswith("Output:\nunset")


async def test_foreground_timeout_drains_partial_output(process_tools):
    result = await process_tools.bash_tool().invoke(
        json.dumps({"command": "printf started; sleep 30", "timeout": 100})
    )
    assert "State: timed_out" in result.content
    assert "Exit code: 143" in result.content
    assert result.content.endswith("Output:\nstarted")


@pytest.mark.parametrize(("signal", "exit_code"), [("TERM", 143), ("KILL", 137)])
async def test_signal_termination_uses_shell_exit_code(process_tools, signal, exit_code):
    result = await process_tools.bash_tool().invoke(json.dumps({"command": f"kill -{signal} $$"}))
    assert f"Exit code: {exit_code}" in result.content


async def test_workdir_changes_start_directory(process_tools, tmp_path):
    (tmp_path / "sub").mkdir()
    result = await process_tools.bash_tool().invoke(
        json.dumps({"command": 'printf "%s" "$PWD"', "workdir": "sub"})
    )
    assert result.content.endswith(str((tmp_path / "sub").resolve()))


async def test_missing_workdir_returns_shell_failure(process_tools):
    result = await process_tools.bash_tool().invoke(
        json.dumps({"command": "pwd", "workdir": "missing"})
    )
    assert "Exit code: 1" in result.content


@pytest.mark.parametrize("separator", ["; ", "\n"])
async def test_missing_workdir_does_not_execute_any_part_of_script(
    process_tools, tmp_path, separator
):
    result = await process_tools.bash_tool().invoke(
        json.dumps({"command": f"printf first{separator}touch marker", "workdir": "missing"})
    )
    assert "Exit code: 1" in result.content
    assert not (tmp_path / "marker").exists()


async def test_workdir_accepts_compound_shell_commands(process_tools, tmp_path):
    (tmp_path / "sub").mkdir()
    result = await process_tools.bash_tool().invoke(
        json.dumps({"command": "for value in one two; do printf '%s ' \"$value\"; done", "workdir": "sub"})
    )
    assert "Exit code: 0" in result.content
    assert result.content.endswith("Output:\none two ")


async def test_workdir_outside_workspace_is_rejected(process_tools):
    result = await process_tools.bash_tool().invoke(
        json.dumps({"command": "pwd", "workdir": "../"})
    )
    assert result.error == "invalid_path"


async def test_bash_can_access_paths_outside_workspace(process_tools):
    result = await process_tools.bash_tool().invoke(json.dumps({"command": "cd /; pwd"}))
    assert "Exit code: 0" in result.content


async def test_unmanaged_background_is_rejected_but_literal_ampersand_is_allowed(process_tools):
    result = await process_tools.bash_tool().invoke(json.dumps({"command": "sleep 30 &"}))
    assert result.content == ("Do not end a command with '&'; set run_in_background=true instead.")
    result = await process_tools.bash_tool().invoke(json.dumps({"command": "printf '&'"}))
    assert result.content.endswith("Output:\n&")


async def test_zero_execution_deadline_is_invalid(process_tools):
    result = await process_tools.bash_tool().invoke(json.dumps({"command": "true", "timeout": 0}))
    assert result.error == "invalid_arguments"


async def test_output_preview_and_saved_log_are_bounded(process_tools):
    result = await process_tools.bash_tool().invoke(
        json.dumps({"command": python("print('a' * 1100000, end='END')")})
    )
    assert "Output truncated: showing a bounded preview of 1100003 bytes." in result.content
    assert len(result.content) < 24000
    assert result.content.endswith("END")
    log = Path(re.search(r"^Log: (.+)$", result.content, re.M)[1])
    data = log.read_bytes()
    assert len(data) <= 1_000_000
    assert b"output bytes omitted" in data
    assert data.endswith(b"END")


async def test_completion_waits_for_child_with_redirected_output(process_tools, tmp_path):
    marker = tmp_path / "finished"
    command = f"(sleep 0.15; touch {shlex.quote(str(marker))}) >/dev/null 2>&1 & true"
    result = await process_tools.bash_tool().invoke(json.dumps({"command": command}))
    assert "State: exited" in result.content
    assert marker.exists()
