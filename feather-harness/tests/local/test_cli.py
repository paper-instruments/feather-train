from __future__ import annotations

import json
import logging
import os
import shlex
import signal
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from openai import AsyncOpenAI

from feather_harness import ChatCompletionsModel, CompactionConfig, run_local
from feather_harness.cli import LocalConfig, _require_writable_output, run
from feather_harness.core.runner import AgentRunError, RunResult
from feather_harness.prompt import SKILLS_ROOT, build_instructions


def response(content=None, *, tools=(), tokens=20):
    calls = [
        {
            "id": f"call-{index}",
            "type": "function",
            "function": {
                "name": name,
                "arguments": json.dumps(arguments),
            },
        }
        for index, (name, arguments) in enumerate(tools)
    ]
    return {
        "id": "test",
        "object": "chat.completion",
        "created": 0,
        "model": "test",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content, "tool_calls": calls},
                "finish_reason": "tool_calls" if calls else "stop",
            }
        ],
        "usage": {"prompt_tokens": tokens, "completion_tokens": 5, "total_tokens": tokens + 5},
    }


@contextmanager
def model_server(replies):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            index = len(requests)
            requests.append((self.path, request))
            status, reply = replies[index]
            if callable(reply):
                reply = reply()
            streamed = request.get("stream") and status == 200
            body = (_as_sse(reply) if streamed else json.dumps(reply)).encode()
            self.send_response(status)
            self.send_header("Content-Type", "text/event-stream" if streamed else "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            try:
                self.wfile.write(body)
            except BrokenPipeError:
                pass  # Cancellation can close the CLI connection before this response.

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def _as_sse(reply):
    choice = reply["choices"][0]
    message = choice["message"]
    delta = {
        "role": "assistant",
        "content": message.get("content"),
        "tool_calls": [{"index": i, **call} for i, call in enumerate(message.get("tool_calls") or [])],
    }
    base = {"id": reply["id"], "object": "chat.completion.chunk", "created": 0, "model": reply["model"]}
    events = [
        {**base, "choices": [{"index": 0, "delta": delta, "finish_reason": choice["finish_reason"]}]},
        {**base, "choices": [], "usage": reply.get("usage")},
    ]
    return "".join(f"data: {json.dumps(event)}\n\n" for event in events) + "data: [DONE]\n\n"


def cli(workspace, *args, input=None):
    return subprocess.run(
        [sys.executable, "-m", "feather_harness", *args],
        cwd=workspace,
        env={**os.environ, "OPENAI_API_KEY": "test-key"},
        input=input,
        capture_output=True,
        text=True,
        timeout=15,
    )


def test_cli_runs_coding_task_through_http_and_real_tools(tmp_path):
    (tmp_path / "answer.txt").write_text("before")
    skill = sorted(SKILLS_ROOT.glob("*/SKILL.md"))[0].relative_to(SKILLS_ROOT)
    todo = {"todos": [{"content": "Update answer", "status": "in_progress"}]}
    replies = [
        (
            200,
            response(
                tools=[
                    ("todo_write", todo),
                    ("read", {"file_path": "answer.txt"}),
                    ("read", {"file_path": f".agents/{skill}"}),
                ]
            ),
        ),
        (
            200,
            response(
                tools=[
                    (
                        "edit",
                        {"file_path": "answer.txt", "old_string": "before", "new_string": "after"},
                    ),
                    ("bash", {"command": 'test "$(cat answer.txt)" = after && printf verified'}),
                ]
            ),
        ),
        (200, response("Updated and verified.")),
    ]
    with model_server(replies) as (url, requests):
        config = tmp_path / "config.json"
        config.write_text(
            json.dumps(
                {"model": "overridden", "base_url": url, "model_settings": {"temperature": 0}}
            )
        )
        completed = cli(
            tmp_path,
            "--config",
            str(config),
            "--model",
            "test-model",
            "--profile",
            "train",
            "-",
            input="Update answer.txt",
        )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result["final_answer"] == "Updated and verified."
    assert result["num_turns"] == 3
    assert (tmp_path / "answer.txt").read_text() == "after"
    assert all(path == "/v1/chat/completions" for path, _ in requests)
    first = requests[0][1]
    assert first["model"] == "test-model"
    assert first["temperature"] == 0
    assert first["messages"][0]["content"] == build_instructions()
    assert [tool["function"]["name"] for tool in first["tools"]] == [
        "bash",
        "task_list",
        "task_output",
        "task_stop",
        "read",
        "edit",
        "write",
        "grep",
        "glob",
        "todo_write",
    ]
    assert any(
        message.get("role") == "tool" and message["content"].endswith("Output:\nverified")
        for message in requests[-1][1]["messages"]
    )


async def test_local_assembly_supplies_todo_state_to_compaction(tmp_path):
    todo = {"todos": [{"content": "Remaining work", "status": "pending"}]}
    with model_server(
        [
            (200, response(tools=[("todo_write", todo)], tokens=2000)),
            (200, response("Remember the task.")),
            (200, response("Done.")),
        ]
    ) as (url, requests):
        async with AsyncOpenAI(api_key="test", base_url=url, max_retries=0) as client:
            result = await run_local(
                "Do the task",
                model=ChatCompletionsModel(client=client, model="test"),
                workspace=tmp_path,
                profile="train",
                compaction=CompactionConfig(
                    enabled=True, context_window_tokens=31000, keep_recent_turns=0
                ),
            )
    assert result.compactions == 1
    transcript = (tmp_path / ".feather/transcript.jsonl").read_text()
    assert [json.loads(line) for line in transcript.splitlines()] == [
        message.model_dump(mode="json") for message in result.execution_messages()[:-1]
    ]
    summary = next(
        message["content"]
        for message in requests[2][1]["messages"]
        if "Remember the task." in message.get("content", "")
    )
    assert "## Current Todo List" in summary
    assert "Remaining work" in summary


async def test_run_local_lists_task_skills_in_the_prompt(tmp_path):
    extra = tmp_path / "harbor-skills" / "harbor-task"
    extra.mkdir(parents=True)
    (extra / "SKILL.md").write_text("---\nname: harbor-task\ndescription: Extra skill.\n---\n")
    with model_server([(200, response("Done."))]) as (url, requests):
        async with AsyncOpenAI(api_key="test", base_url=url, max_retries=0) as client:
            await run_local(
                "Do the task",
                model=ChatCompletionsModel(client=client, model="test"),
                workspace=tmp_path,
                skills_dir=extra.parent,
            )
    prompt = requests[0][1]["messages"][0]["content"]
    assert "- harbor-task: Extra skill. (file: .agents/harbor-task/SKILL.md)" in prompt
    assert (tmp_path / ".agents" / "harbor-task" / "SKILL.md").is_file()


def test_cli_returns_partial_json_after_exhausting_request_retries(tmp_path):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"model_retry": {"max_attempts": 3, "base_backoff_seconds": 0}}))
    with model_server([(500, {"error": {"message": "test failure", "type": "server_error"}})] * 3) as (
        url,
        requests,
    ):
        completed = cli(
            tmp_path, "--config", str(config), "--model", "test", "--base-url", url, "Do the task"
        )
    assert completed.returncode == 1
    result = json.loads(completed.stdout)
    assert result["stop_reason"] == "failed"
    assert result["segments"][-1]["input"][-1]["content"] == "Do the task"
    assert json.loads((tmp_path / ".feather" / "run.json").read_text(encoding="utf-8")) == result
    assert len(requests) == 3
    assert requests[0] == requests[1] == requests[2]


def test_cli_rejects_invalid_config_before_workspace_setup(tmp_path):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"model": "test", "unknown_option": True}))
    completed = cli(tmp_path, "--config", str(config), "Do the task")
    assert completed.returncode == 1
    assert "unknown_option" in completed.stderr
    assert not (tmp_path / ".agents").exists()
    assert not (tmp_path / ".feather" / "run.json").exists()


def test_cli_writes_run_json_under_feather_and_output_path(tmp_path):
    with model_server([(200, response("Done."))]) as (url, _):
        output = tmp_path / "reports" / "result.json"
        completed = cli(
            tmp_path,
            "--model",
            "test",
            "--base-url",
            url,
            "--profile",
            "train",
            "--output",
            str(output),
            "Do the task",
        )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result["final_answer"] == "Done."
    saved = json.loads((tmp_path / ".feather" / "run.json").read_text(encoding="utf-8"))
    copied = json.loads(output.read_text(encoding="utf-8"))
    assert saved == result == copied


def test_cli_resolves_relative_output_from_launch_directory(tmp_path):
    launch = tmp_path / "launch"
    workspace = tmp_path / "workspace"
    launch.mkdir()
    workspace.mkdir()
    with model_server([(200, response("Done."))]) as (url, _):
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "feather_harness",
                "--model",
                "test",
                "--base-url",
                url,
                "--profile",
                "train",
                "--workdir",
                str(workspace),
                "--output",
                "reports/result.json",
                "Do the task",
            ],
            cwd=launch,
            env={**os.environ, "OPENAI_API_KEY": "test-key"},
            capture_output=True,
            text=True,
            timeout=15,
        )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    copied = json.loads((launch / "reports" / "result.json").read_text(encoding="utf-8"))
    saved = json.loads((workspace / ".feather" / "run.json").read_text(encoding="utf-8"))
    assert copied == result == saved
    assert not (workspace / "reports" / "result.json").exists()


@pytest.mark.skipif(
    sys.platform == "win32" or os.geteuid() == 0,
    reason="CLI release target is Linux; root bypasses chmod",
)
def test_cli_rejects_unwritable_output_before_run(tmp_path):
    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(0o555)
    try:
        with model_server([(200, response("Done."))]) as (url, requests):
            completed = cli(
                tmp_path,
                "--model",
                "test",
                "--base-url",
                url,
                "--profile",
                "train",
                "--output",
                str(locked / "result.json"),
                "Do the task",
            )
    finally:
        locked.chmod(0o755)
    assert completed.returncode == 1
    assert "PermissionError" in completed.stderr
    assert not (tmp_path / ".agents").exists()
    assert not (tmp_path / ".feather" / "run.json").exists()
    assert requests == []


@pytest.mark.skipif(sys.platform == "win32", reason="CLI release target is Linux")
def test_output_preflight_leaves_dangling_symlink_alone(tmp_path):
    target = tmp_path / "missing" / "result.json"
    link = tmp_path / "result.json"
    link.symlink_to(target)
    _require_writable_output(link)
    assert link.is_symlink()
    assert not target.exists()
    assert not target.parent.exists()


@pytest.mark.skipif(sys.platform == "win32", reason="CLI release target is Linux")
def test_output_preflight_rejects_dangling_parent_symlink(tmp_path):
    parent = tmp_path / "reports"
    parent.symlink_to(tmp_path / "missing")
    for output in (parent / "result.json", parent / "nested" / "result.json"):
        with pytest.raises(NotADirectoryError):
            _require_writable_output(output)
    assert not (tmp_path / "missing").exists()


def test_output_preflight_does_not_create_missing_directories(tmp_path):
    output = tmp_path / "missing-workdir" / "reports" / "result.json"
    _require_writable_output(output)
    assert not (tmp_path / "missing-workdir").exists()


@pytest.mark.skipif(
    sys.platform == "win32" or os.geteuid() == 0,
    reason="CLI release target is Linux; root bypasses chmod",
)
def test_output_preflight_matches_replace_semantics(tmp_path):
    read_only = tmp_path / "result.json"
    read_only.write_text("old", encoding="utf-8")
    read_only.chmod(0o444)
    _require_writable_output(read_only)

    locked = tmp_path / "locked"
    locked.mkdir()
    writable = locked / "result.json"
    writable.write_text("old", encoding="utf-8")
    locked.chmod(0o555)
    try:
        with pytest.raises(PermissionError):
            _require_writable_output(writable)
    finally:
        locked.chmod(0o755)
    assert sorted(path.name for path in locked.iterdir()) == ["result.json"]


async def test_cli_keeps_failed_run_when_output_write_fails(tmp_path, monkeypatch, caplog, capsys):
    output = tmp_path / "reports" / "result.json"
    result = RunResult(stop_reason="failed", num_turns=1, compactions=0, segments=())

    async def fail_local(*_args, **_kwargs):
        raise AgentRunError("model failed", cause=RuntimeError("boom"), result=result)

    def save(path, payload):
        path = Path(path)
        if path.resolve() == output.resolve():
            raise OSError("read-only filesystem")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload, encoding="utf-8")

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr("feather_harness.cli.run_local", fail_local)
    monkeypatch.setattr("feather_harness.cli.save_transcript", save)
    config = LocalConfig(model="test", workspace=tmp_path, profile="train")
    with caplog.at_level(logging.WARNING):
        code = await run(config, "Do the task", output=output)
    assert code == 1
    assert "model failed" in caplog.text
    assert "Could not write the run result to --output" in caplog.text
    assert json.loads(capsys.readouterr().out)["stop_reason"] == "failed"
    saved = json.loads((tmp_path / ".feather" / "run.json").read_text(encoding="utf-8"))
    assert saved["stop_reason"] == "failed"
    assert not output.exists()


async def test_cli_broken_stdout_still_writes_run_files(tmp_path, monkeypatch):
    result = RunResult(stop_reason="completed", num_turns=1, compactions=0, segments=())

    async def succeed(*_args, **_kwargs):
        return result

    class BrokenStdout:
        def write(self, _payload):
            raise BrokenPipeError()

        def flush(self):
            raise BrokenPipeError()

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr("feather_harness.cli.run_local", succeed)
    monkeypatch.setattr(sys, "stdout", BrokenStdout())
    output = tmp_path / "out.json"
    config = LocalConfig(model="test", workspace=tmp_path, profile="train")
    code = await run(config, "Do the task", output=output)
    assert code == 0
    saved = json.loads((tmp_path / ".feather" / "run.json").read_text(encoding="utf-8"))
    copied = json.loads(output.read_text(encoding="utf-8"))
    assert saved == copied
    assert saved["stop_reason"] == "completed"


def test_cli_web_tools_require_exa_key_before_workspace_setup(tmp_path, monkeypatch):
    monkeypatch.delenv("EXA_API_KEY", raising=False)
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"model": "test", "enable_web_tools": True}))
    completed = cli(tmp_path, "--config", str(config), "Do the task")
    assert completed.returncode == 1
    assert "EXA_API_KEY is required" in completed.stderr
    assert not (tmp_path / ".agents").exists()


@pytest.mark.skipif(sys.platform == "win32", reason="CLI release target is Linux")
def test_cli_interrupt_stops_background_processes(tmp_path):
    waiting = threading.Event()
    release = threading.Event()
    leaked = tmp_path / "leaked"

    def pending_response():
        waiting.set()
        release.wait(5)
        return response("Done.")

    with model_server(
        [
            (
                200,
                response(
                    tools=[
                        (
                            "bash",
                            {
                                "command": f"sleep 1; touch {shlex.quote(str(leaked))}",
                                "run_in_background": True,
                            },
                        )
                    ]
                ),
            ),
            (200, pending_response),
        ]
    ) as (url, _):
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "feather_harness",
                "--model",
                "test",
                "--base-url",
                url,
                "Start a task",
            ],
            cwd=tmp_path,
            env={**os.environ, "OPENAI_API_KEY": "test"},
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            assert waiting.wait(5), "CLI did not reach the second model call"
            process.send_signal(signal.SIGINT)
            _, stderr = process.communicate(timeout=5)
            assert process.returncode == 130, stderr
        finally:
            release.set()
            if process.poll() is None:
                process.kill()
                process.wait()
    time.sleep(1.1)
    assert not leaked.exists()


def test_cli_discovers_and_executes_mcp_tools(tmp_path):
    from ..mcp.server import mcp_server

    with (
        mcp_server() as (mcp_url, server),
        model_server(
            [
                (200, response(tools=[("mcp__mail__echo", {"value": "hello"})])),
                (200, response("Done.")),
            ]
        ) as (url, requests),
    ):
        config = tmp_path / "config.json"
        config.write_text(
            json.dumps(
                {
                    "model": "test",
                    "base_url": url,
                    "mcp_servers": [
                        {"name": "mail", "transport": "http", "url": mcp_url + "/mcp/"}
                    ],
                }
            )
        )
        completed = cli(tmp_path, "--config", str(config), "Use mail")
        assert server.calls == [("echo", {"value": "hello"})]
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["final_answer"] == "Done."
    names = [tool["function"]["name"] for tool in requests[0][1]["tools"]]
    assert names[:4] == ["mcp__mail__echo", "mcp__mail__failure", "mcp__mail__image", "bash"]
    assert requests[1][1]["messages"][-1] == {
        "role": "tool",
        "tool_call_id": "call-0",
        "content": [{"type": "text", "text": "Echo:"}, {"type": "text", "text": "hello"}],
    }


def test_cli_discovers_and_executes_stdio_mcp_tools(tmp_path):
    from ..mcp.stdio_server import __file__ as stdio_server

    with model_server(
        [
            (200, response(tools=[("mcp__mail__echo", {"value": "hello"})])),
            (200, response("Done.")),
        ]
    ) as (url, requests):
        config = tmp_path / "config.json"
        config.write_text(
            json.dumps(
                {
                    "model": "test",
                    "base_url": url,
                    "mcp_servers": [
                        {
                            "name": "mail",
                            "transport": "stdio",
                            "command": sys.executable,
                            "args": [stdio_server],
                        }
                    ],
                }
            )
        )
        completed = cli(tmp_path, "--config", str(config), "Use mail")
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["final_answer"] == "Done."
    assert requests[0][1]["tools"][0]["function"]["name"] == "mcp__mail__echo"
