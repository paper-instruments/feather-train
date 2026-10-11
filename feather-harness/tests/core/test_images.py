"""Image observations reach provider requests without changing text-only runs."""

import base64
import json
from types import SimpleNamespace

import httpx
import pytest
from mcp.types import CallToolResult, ImageContent, Tool
from openai import AsyncOpenAI

from feather_harness import (
    AssistantMessage,
    ChatCompletionsModel,
    CompactionConfig,
    ToolCall,
    ToolMessage,
)
from feather_harness.cli import LocalConfig
from feather_harness.core.compaction import Accounting, check_compaction
from feather_harness.core.messages import MESSAGES
from feather_harness.core.model import messages_to_wire
from feather_harness.core.runner import RunResult
from feather_harness.local import run_local
from feather_harness.local.filesystem import LocalFileRuntime
from feather_harness.tools.files import FileTools
from feather_harness.tools.mcp import McpServerSpec, _bind

PNG = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aOZkAAAAASUVORK5CYII=")
IMAGE = {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(PNG).decode()}}


@pytest.mark.parametrize("compact", [False, True])
async def test_local_image_reaches_real_chat_request_and_trace(tmp_path, compact):
    (tmp_path / "page.png").write_bytes(PNG)
    requests = []
    replies = iter([
        {"role": "assistant", "tool_calls": [{"id": "image-call", "type": "function", "function": {"name": "read", "arguments": '{"file_path":"page.png"}'}}]},
        *([{"role": "assistant", "content": "Visual findings summarized."}] if compact else []),
        {"role": "assistant", "content": "Finished."},
    ])

    def respond(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={"id": "r", "object": "chat.completion", "created": 0, "model": "vision", "choices": [{"index": 0, "message": next(replies), "finish_reason": "stop"}], "usage": {"prompt_tokens": 120000 if len(requests) == 1 else 10, "completion_tokens": 10, "total_tokens": 120010}})

    async with AsyncOpenAI(api_key="test", http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond))) as client:
        result = await run_local("Inspect page.png", model=ChatCompletionsModel(client=client, model="vision"), workspace=tmp_path, enable_image_tools=True, compaction=CompactionConfig(enabled=compact, keep_recent_turns=0))
    wire = requests[1]["messages"]
    assert wire[-1 if not compact else -2]["content"][-1] == IMAGE
    assert all(part["type"] == "text" for message in wire if message["role"] == "tool" for part in message["content"])
    restored = RunResult.model_validate_json(result.model_dump_json())
    observation = next(m for m in restored.execution_messages() if isinstance(m, ToolMessage))
    assert observation.content[-1] == IMAGE
    assert result.compactions == int(compact)


def test_parallel_tool_results_precede_associated_image_message():
    messages = [AssistantMessage(tool_calls=(ToolCall(id="a", name="image", arguments="{}"), ToolCall(id="b", name="text", arguments="{}"))), ToolMessage(tool_call_id="a", content=[IMAGE]), ToolMessage(tool_call_id="b", content="done")]
    wire = messages_to_wire(messages)
    assert [m["role"] for m in wire] == ["assistant", "tool", "tool", "user"]
    assert "a" in wire[-1]["content"][0]["text"]
    assert wire[-1]["content"][1] == IMAGE
    assert MESSAGES.validate_json(MESSAGES.dump_json(tuple(messages))) == tuple(messages)


async def test_image_worker_enforces_paths_and_format(tmp_path):
    outside = tmp_path.parent / "private.png"
    outside.write_bytes(PNG)
    (tmp_path / "escape.png").symlink_to(outside)
    (tmp_path / "bad.png").write_text("not an image")
    tool = FileTools(LocalFileRuntime(tmp_path), enable_image_reads=True).read_tool()
    assert (await tool.invoke('{"file_path":"escape.png"}')).error == "invalid_path"
    assert (await tool.invoke('{"file_path":"bad.png"}')).error == "tool_error"
    (tmp_path / "large.png").write_bytes(b"x" * (20 * 1024 * 1024 + 1))
    assert "Maximum input size is 20 MB" in (await tool.invoke('{"file_path":"large.png"}')).content


@pytest.mark.parametrize("enabled", [False, True])
async def test_mcp_image_results_require_opt_in(enabled):
    async def call_tool(*args):
        return CallToolResult(content=[ImageContent(type="image", data=base64.b64encode(PNG).decode(), mimeType="image/png")])
    bound = _bind(McpServerSpec(name="m", transport="http", url="http://localhost"), SimpleNamespace(call_tool=call_tool), Tool(name="image", inputSchema={}), "image", enable_image_tools=enabled)
    if enabled:
        assert (await bound.invoke("{}")).content == [IMAGE]
    else:
        with pytest.raises(ValueError, match="image output"):
            await bound.invoke("{}")


def test_base64_size_does_not_drive_compaction_and_images_follow_profile_by_default():
    content = {"type": "image_url", "image_url": {"url": "data:image/png;base64," + "a" * 1000000}}
    assert check_compaction([ToolMessage(tool_call_id="a", content=[content])], CompactionConfig(enabled=True), Accounting(turns=1), 10) is None
    assert LocalConfig(model="text").enable_image_tools is None
