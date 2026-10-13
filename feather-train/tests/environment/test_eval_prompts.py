"""Check the model-visible eval condition without sandbox or model services."""

import hashlib
import json
from contextlib import nullcontext

import httpx
import pytest
from feather_harness import ChatCompletionsModel, CompactionConfig
from feather_harness.production import PRODUCTION_PROMPT
from feather_harness.prompt import RESOURCE_ROOT
from openai import AsyncOpenAI

from feather_train.agent import run_e2b
from feather_train.harness import FeatherHarnessConfig


class Transport:
    workspace = "/home/agent/workspace"
    run_as = "agent"

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return None

    async def save_transcript(self, content):
        return None


class Rollout:
    def __init__(self):
        self.metadata = {}

    def add_metadata(self, values):
        self.metadata.update(values)

    def wrap_tool(self, tool):
        return tool

    def trace_agent(self, model):
        return nullcontext(model)


@pytest.mark.parametrize("profile", ["train", "prod"])
@pytest.mark.parametrize("override", [None, "Keep the workbook format."])
@pytest.mark.parametrize("prompt", [None, "", "Use files in {{ workspace }}.\n"])
async def test_additional_instructions_preserves_task_bytes_and_profile_overrides(
    profile, override, prompt
):
    requests = []
    task = "  Original task.\nReturn the answer exactly as instructed.\n\n"
    rollout = Rollout()
    config = FeatherHarnessConfig(profile=profile, additional_instructions=prompt)

    def respond(request):
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "test",
                "object": "chat.completion",
                "created": 0,
                "model": "test",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "Done."},
                    "finish_reason": "stop",
                }],
            },
        )

    async with AsyncOpenAI(
        api_key="test", http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond))
    ) as client:
        await run_e2b(
            task,
            instructions=override,
            profile=config.profile,
            additional_instructions=config.additional_instructions,
            model=ChatCompletionsModel(client=client, model="test"),
            transport=Transport(),
            tools=("read",),
            compaction=CompactionConfig(enabled=False),
            rollout=rollout,
        )

    messages = requests[0]["messages"]
    assert messages[1] == {"role": "user", "content": task}
    system = messages[0]["content"]
    base = PRODUCTION_PROMPT if profile == "prod" else (
        override or (RESOURCE_ROOT / "system_prompt.md").read_text()
    )
    assert base in system
    if override:
        assert system.count(override) == 1
    expected_sections = int(bool(prompt) or (profile == "prod" and bool(override)))
    assert system.count("# Additional task instructions") == expected_sections
    assert "# Eval adapter instructions" not in system
    if prompt:
        assert "Use files in /home/agent/workspace." in system
        assert system.endswith("Use files in /home/agent/workspace.\n")
        assert "{{ workspace }}" not in system
    if expected_sections:
        catalog = "# Tool capability instructions" if profile == "prod" else "# Sandbox capability instructions"
        assert system.index(catalog) < system.index("# Additional task instructions")
    if profile == "prod" and override and prompt:
        assert system.endswith(f"{override}\n\nUse files in /home/agent/workspace.\n")
    assert rollout.metadata["profile"] == profile
    assert rollout.metadata["system_prompt_sha256"] == hashlib.sha256(system.encode()).hexdigest()
