import hashlib
import json
import logging
import sys
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import pytest
from openai import AsyncOpenAI
from pydantic import ValidationError

from feather_harness import (
    ChatCompletionsModel,
    CompactionConfig,
    McpServerSpec,
    RunResult,
    SubagentConfig,
    run_local,
)
from feather_harness.cli import LocalConfig
from feather_harness.subagents import AGENT_TOOL_NAMES, CHILD_GUIDANCE, PARENT_GUIDANCE
from tests.local.test_cli import response
from tests.local.test_profiles import _WebProvider


async def test_subagents_require_explicit_production_opt_in_before_workspace_setup(
    tmp_path: Path,
) -> None:
    assert LocalConfig(model="test").subagents is None
    assert (
        LocalConfig.model_validate_json('{"model":"test","subagents":{}}').subagents
        == SubagentConfig()
    )
    with pytest.raises(ValidationError, match="only available in the prod profile"):
        LocalConfig(model="test", profile="train", subagents=SubagentConfig())

    def unexpected_request(payload: dict) -> dict:
        pytest.fail("Invalid local configuration must not reach the model")

    async with _model(unexpected_request) as model:
        with pytest.raises(ValueError, match="only available in the prod profile"):
            await run_local(
                "Parent task",
                model=model,
                workspace=tmp_path,
                profile="train",
                subagents=SubagentConfig(),
            )
    assert list(tmp_path.iterdir()) == []


async def test_children_run_shell_commands_concurrently_with_private_prompts_and_logs(
    tmp_path: Path,
) -> None:
    requests: dict[str, list[dict]] = {"Parent private task": [], "first": [], "second": []}

    def respond(payload: dict) -> dict:
        prompt = _first_prompt(payload)
        requests[prompt].append(payload)
        if prompt == "Parent private task":
            if len(requests[prompt]) == 1:
                return response(
                    tools=[("spawn_agent", {"prompt": name}) for name in ("first", "second")]
                )
            return _wait_or_finish(payload, "Both children completed.")
        if len(requests[prompt]) == 1:
            other = "second" if prompt == "first" else "first"
            # Neither command can finish until its sibling's shell is executing too.
            command = (
                f"touch {prompt}.ready; "
                f"while ! test -f {other}.ready; do sleep 0.01; done; "
                f"printf {prompt}-done"
            )
            return response(
                tools=[
                    ("bash", {"command": command, "timeout": 5000}),
                    ("bash", {"command": f"printf {prompt}-background", "run_in_background": True}),
                ]
            )
        if len(requests[prompt]) == 2:
            assert f"{prompt}-done" in _last_outputs(payload)[0]
            return response(tools=[("task_output", {"task_id": "task_1"})])
        assert f"{prompt}-background" in _last_outputs(payload)[0]
        return response(f"{prompt} completed.")

    async with _model(respond) as model:
        result = await run_local(
            "Parent private task",
            model=model,
            workspace=tmp_path,
            subagents=SubagentConfig(max_concurrent=2, max_agents=2, max_turns=5),
        )

    assert result.final_answer == "Both children completed."
    parent_instructions = requests["Parent private task"][0]["messages"][0]["content"]
    assert PARENT_GUIDANCE in parent_instructions
    assert "2 concurrently running children, 2 children total, and 5 turns" in parent_instructions
    base_instructions = parent_instructions.split("\n\n" + PARENT_GUIDANCE)[0]
    for name in ("first", "second"):
        first = requests[name][0]
        assert first["messages"] == [
            {"role": "system", "content": base_instructions + "\n\n" + CHILD_GUIDANCE},
            {"role": "user", "content": name},
        ]
        assert not AGENT_TOOL_NAMES.intersection(
            tool["function"]["name"] for tool in first["tools"]
        )

    manifest = json.loads((tmp_path / ".feather/agents/manifest.json").read_text())
    assert len(manifest["agents"]) == 2
    for status in manifest["agents"]:
        assert status["status"] == "completed"
        directory = tmp_path / ".feather/agents" / status["agent_id"]
        saved = RunResult.model_validate_json((directory / "run-1.json").read_text())
        assert saved.final_answer == status["answer"]
        assert list((directory / "tasks").rglob("*.log"))


async def test_delegation_inherits_additional_guidance_and_web_provider_and_hashes_parent_prompt(
    tmp_path: Path,
) -> None:
    requests: dict[str, list[dict]] = {"parent": [], "child": []}
    provider = _WebProvider()
    metadata_path = tmp_path / "logs/eval-prompt.json"

    def respond(payload: dict) -> dict:
        prompt = _first_prompt(payload)
        requests[prompt].append(payload)
        if prompt == "parent":
            if len(requests[prompt]) == 1:
                return response(tools=[("spawn_agent", {"prompt": "child"})])
            return _wait_or_finish(payload, "Research completed.")
        if len(requests[prompt]) == 1:
            return response(tools=[("web_search", {"query": "annual filing"})])
        assert "https://example.com/filing" in _last_outputs(payload)[0]
        return response("Child research completed.")

    async with _model(respond) as model:
        result = await run_local(
            "parent", model=model, workspace=tmp_path, subagents=SubagentConfig(),
            additional_instructions="Use records in {{ workspace }}.",
            eval_prompt_output=metadata_path, web_provider=provider,
        )

    assert result.final_answer == "Research completed."
    assert provider.queries == ["annual filing"]
    for prompt in ("parent", "child"):
        payload = requests[prompt][0]
        instructions = payload["messages"][0]["content"]
        assert instructions.count(f"Use records in {tmp_path}.") == 1
        assert {"web_search", "web_fetch"} <= {
            tool["function"]["name"] for tool in payload["tools"]
        }
    parent_instructions = requests["parent"][0]["messages"][0]["content"]
    assert json.loads(metadata_path.read_text()) == {
        "profile": "prod",
        "system_prompt_sha256": hashlib.sha256(parent_instructions.encode()).hexdigest(),
    }


async def test_exhausted_child_exposes_its_saved_trace_without_an_extra_turn(tmp_path: Path) -> None:
    counts = {"parent": 0, "child": 0}
    child_id = ""
    report = ""

    def respond(payload: dict) -> dict:
        nonlocal child_id, report
        prompt = _first_prompt(payload)
        counts[prompt] += 1
        if prompt == "child":
            return response(tools=[("write", {"file_path": "draft.txt", "content": "partial work"})])
        if counts[prompt] == 1:
            return response(tools=[("spawn_agent", {"prompt": "child"})])
        if counts[prompt] == 2:
            child_id = json.loads(_last_outputs(payload)[0])["agent_id"]
            return response(tools=[("wait_agent", {"agent_ids": [child_id]})])
        status = json.loads(_last_outputs(payload)[0])[0]
        assert status["status"] == "exhausted"
        report = status["answer"]
        return response("The child left partial work, not a completed deliverable.")

    async with _model(respond) as model:
        await run_local(
            "parent", model=model, workspace=tmp_path, subagents=SubagentConfig(max_turns=1)
        )

    trace_path = tmp_path / ".feather/agents" / child_id / "run-1.json"
    assert str(trace_path) in report
    assert "not a final answer" in report and "Tool result" in report
    assert (tmp_path / "draft.txt").read_text() == "partial work"
    saved = RunResult.model_validate_json(trace_path.read_text())
    assert saved.stop_reason == "MaxTurnsExceeded" and saved.final_answer == ""
    assert counts["child"] == 1


async def test_child_followups_keep_their_own_todos_and_read_revisions(tmp_path: Path) -> None:
    (tmp_path / "shared.txt").write_text("before")
    counts = {"parent": 0, "child": 0}
    child_id = ""

    def respond(payload: dict) -> dict:
        nonlocal child_id
        prompt = _first_prompt(payload)
        turn = counts[prompt]
        counts[prompt] += 1
        if prompt == "parent":
            if turn == 0:
                return response(
                    tools=[
                        ("read", {"file_path": "shared.txt"}),
                        (
                            "todo_write",
                            {"todos": [{"content": "Parent-only todo", "status": "pending"}]},
                        ),
                        ("spawn_agent", {"prompt": "child"}),
                    ]
                )
            if turn == 1:
                child_id = json.loads(_last_outputs(payload)[-1])["agent_id"]
                return response(tools=[("wait_agent", {"agent_ids": [child_id]})])
            if turn == 2:
                assert json.loads(_last_outputs(payload)[0])[0]["status"] == "completed"
                return response(
                    tools=[
                        (
                            "edit",
                            {
                                "file_path": "shared.txt",
                                "old_string": "before",
                                "new_string": "after",
                            },
                        ),
                        (
                            "followup_agent",
                            {"agent_id": child_id, "prompt": "Continue with the same tools."},
                        ),
                    ]
                )
            if turn == 3:
                return response(tools=[("wait_agent", {"agent_ids": [child_id]})])
            return _wait_or_finish(payload, "Verified separate state.")
        if turn == 0:
            return response(
                tools=[
                    (
                        "edit",
                        {"file_path": "shared.txt", "old_string": "before", "new_string": "wrong"},
                    ),
                    ("todo_write", {}),
                    ("read", {"file_path": "shared.txt"}),
                    (
                        "todo_write",
                        {"todos": [{"content": "Child-only todo", "status": "pending"}]},
                    ),
                ]
            )
        if turn == 1:
            outputs = _last_outputs(payload)
            assert "Read shared.txt before modifying" in outputs[0]
            assert json.loads(outputs[1]) == {"todos": []}
            return response("Initial assignment completed.")
        if turn == 2:
            assert payload["messages"][-1]["content"] == "Continue with the same tools."
            return response(
                tools=[
                    (
                        "edit",
                        {"file_path": "shared.txt", "old_string": "after", "new_string": "wrong"},
                    ),
                    ("todo_write", {}),
                ]
            )
        outputs = _last_outputs(payload)
        assert "shared.txt changed since it was read" in outputs[0]
        assert json.loads(outputs[1]) == {
            "todos": [{"content": "Child-only todo", "status": "pending"}]
        }
        return response("Follow-up retained private state.")

    async with _model(respond) as model:
        result = await run_local(
            "parent", model=model, workspace=tmp_path, subagents=SubagentConfig()
        )

    assert result.final_answer == "Verified separate state."
    assert (tmp_path / "shared.txt").read_text() == "after"
    directory = tmp_path / ".feather/agents" / child_id
    saved = RunResult.model_validate_json((directory / "run-2.json").read_text())
    assert saved.final_answer == "Follow-up retained private state."
    assert (directory / "run-1.json").is_file()


async def test_compaction_keeps_parent_agent_state_and_child_transcripts_separate(
    tmp_path: Path,
) -> None:
    execution_counts = {"parent": 0, "child": 0}
    summaries: dict[str, str] = {}

    def respond(payload: dict) -> dict:
        prompt = _first_prompt(payload)
        if not payload.get("tools"):
            return response(f"Summary for {prompt}.")
        turn = execution_counts[prompt]
        execution_counts[prompt] += 1
        if turn == 0:
            tools = [
                ("todo_write", {"todos": [{"content": f"{prompt} todo", "status": "pending"}]})
            ]
            if prompt == "parent":
                tools.append(("spawn_agent", {"prompt": "child"}))
            return response(tools=tools, tokens=2000)
        if turn == 1:
            summaries[prompt] = payload["messages"][-1]["content"]
            if prompt == "parent":
                return response(tools=[("list_agents", {})])
            return response("Child completed after compaction.")
        return _wait_or_finish(payload, "Parent completed after compaction.")

    async with _model(respond) as model:
        result = await run_local(
            "parent",
            model=model,
            workspace=tmp_path,
            subagents=SubagentConfig(),
            compaction=CompactionConfig(
                enabled=True, context_window_tokens=31000, keep_recent_turns=0
            ),
        )

    assert result.compactions == 1
    assert "## Subagents" in summaries["parent"]
    assert "parent todo" in summaries["parent"]
    assert "child todo" not in summaries["parent"]
    assert "child todo" in summaries["child"]
    assert "parent todo" not in summaries["child"]
    assert "## Subagents" not in summaries["child"]
    parent_transcript = (tmp_path / ".feather/transcript.jsonl").read_text()
    child_transcript = next((tmp_path / ".feather/agents").glob("*/transcript.jsonl")).read_text()
    # Transcripts are JSON lines, so multi-paragraph guidance appears with escaped newlines.
    parent_guidance, child_guidance = (json.dumps(text)[1:-1] for text in (PARENT_GUIDANCE, CHILD_GUIDANCE))
    assert parent_guidance in parent_transcript and child_guidance not in parent_transcript
    assert child_guidance in child_transcript and parent_guidance not in child_transcript


async def test_real_mcp_tools_survive_child_followup_and_close_cleanly(tmp_path, caplog) -> None:
    counts = {"parent": 0, "child": 0}
    child_id = ""

    def respond(payload: dict) -> dict:
        nonlocal child_id
        prompt = _first_prompt(payload)
        turn = counts[prompt]
        counts[prompt] += 1
        if prompt == "parent":
            if turn == 0:
                return response(
                    tools=[("mcp__echo__echo", {"value": "parent"}), ("spawn_agent", {"prompt": "child"})]
                )
            if turn == 1:
                outputs = _last_outputs(payload)
                assert outputs[0] == [{"type": "text", "text": "parent"}]
                child_id = json.loads(outputs[1])["agent_id"]
                return response(tools=[("wait_agent", {"agent_ids": [child_id]})])
            if turn == 2:
                assert json.loads(_last_outputs(payload)[0])[0]["status"] == "completed"
                return response(
                    tools=[("followup_agent", {"agent_id": child_id, "prompt": "Continue"})]
                )
            if turn == 3:
                return response(tools=[("wait_agent", {"agent_ids": [child_id]})])
            return _wait_or_finish(payload, "MCP worked on both child assignments.")
        if turn in (0, 2):
            return response(tools=[("mcp__echo__echo", {"value": f"child-{turn}"})])
        assert _last_outputs(payload)[0] == [{"type": "text", "text": f"child-{turn - 1}"}]
        return response("Child MCP call completed.")

    server = McpServerSpec(
        name="echo",
        transport="stdio",
        command=sys.executable,
        args=(str(Path(__file__).parents[1] / "mcp/stdio_server.py"),),
    )
    with caplog.at_level(logging.WARNING, logger="feather_harness.tools.mcp"):
        async with _model(respond) as model:
            result = await run_local(
                "parent",
                model=model,
                workspace=tmp_path,
                subagents=SubagentConfig(),
                mcp_servers=(server,),
            )
    assert result.final_answer == "MCP worked on both child assignments."
    assert counts["child"] == 4
    assert "MCP cleanup failed" not in caplog.text


@asynccontextmanager
async def _model(respond: Callable[[dict], dict]) -> AsyncIterator[ChatCompletionsModel]:
    def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=respond(json.loads(request.content)))

    async with AsyncOpenAI(
        api_key="test",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)),
    ) as client:
        yield ChatCompletionsModel(client=client, model="test")


def _first_prompt(payload: dict) -> str:
    return next(message["content"] for message in payload["messages"] if message["role"] == "user")


def _last_outputs(payload: dict) -> list[str]:
    messages = payload["messages"]
    last_assistant = max(
        index for index, message in enumerate(messages) if message["role"] == "assistant"
    )
    return [
        message["content"]
        for message in messages[last_assistant + 1 :]
        if message["role"] == "tool"
    ]


def _wait_or_finish(payload: dict, answer: str) -> dict:
    statuses = []
    for output in _last_outputs(payload):
        value = json.loads(output)
        statuses.extend(value if isinstance(value, list) else [value])
    running = [status["agent_id"] for status in statuses if status["status"] == "running"]
    if running:
        return response(tools=[("wait_agent", {"agent_ids": running})])
    assert all(status["status"] == "completed" for status in statuses)
    return response(answer)
