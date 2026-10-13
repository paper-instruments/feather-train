"""Native E2B eval profile selection without provisioning a sandbox."""

import json

import httpx
import pytest
import verifiers.v1 as vf
from feather_harness import ChatCompletionsModel
from feather_harness.core.compaction import CompactionConfig
from feather_harness.production import (
    PRODUCTION_COMPACTION_PROMPT,
    PRODUCTION_PROMPT,
    PRODUCTION_TOOL_DESCRIPTIONS,
)
from openai import AsyncOpenAI
from pydantic import ValidationError

from feather_train.agent import run_e2b
from feather_train.harness import (
    FeatherHarnessConfig,
    _agent_system_instructions,
    _model_settings_from_sampling,
)


class StubTransport:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return None

    async def save_transcript(self, content):
        return None


@pytest.mark.parametrize("profile", ["train", "prod"])
async def test_e2b_profile_selects_model_visible_prompt_and_tool_descriptions(profile):
    requests = []

    def respond(request):
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "test",
                "object": "chat.completion",
                "created": 0,
                "model": "test",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Done."},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    async with AsyncOpenAI(
        api_key="test", http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond))
    ) as client:
        result = await run_e2b(
            "Do the task",
            instructions="Keep the supplied workbook format.",
            profile=profile,
            model=ChatCompletionsModel(client=client, model="test"),
            transport=StubTransport(),
            tools=("read",),
            compaction=CompactionConfig(enabled=False),
        )

    assert result.final_answer == "Done."
    system = requests[0]["messages"][0]["content"]
    assert _agent_system_instructions(result) == system
    read = requests[0]["tools"][0]["function"]
    if profile == "prod":
        assert PRODUCTION_PROMPT in system
        assert "headless" not in system
        assert "Shell commands run locally with the host's permissions" not in system
        assert "Keep the supplied workbook format." in system
        assert "Tool capability instructions" in system
        assert read["description"] == PRODUCTION_TOOL_DESCRIPTIONS["read_without_images"]
        assert "image" not in read["description"].lower()
    else:
        assert PRODUCTION_PROMPT not in system
        assert system.startswith("# Agent instructions\n\nKeep the supplied workbook format.")
        assert "Tool capability instructions" not in system


async def test_e2b_prod_recovers_one_cut_off_response():
    requests = []

    def respond(request):
        requests.append(json.loads(request.content))
        first = len(requests) == 1
        return httpx.Response(
            200,
            json={
                "id": "test",
                "object": "chat.completion",
                "created": 0,
                "model": "test",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "" if first else "Done."},
                        "finish_reason": "length" if first else "stop",
                    }
                ],
            },
        )

    async with AsyncOpenAI(
        api_key="test", http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond))
    ) as client:
        result = await run_e2b(
            "Do the task",
            profile="prod",
            model=ChatCompletionsModel(client=client, model="test"),
            transport=StubTransport(),
            tools=("read",),
            compaction=CompactionConfig(enabled=False),
        )

    assert result.final_answer == "Done."
    assert result.recoveries == 1
    assert len(requests) == 2


async def test_eval_sampling_reaches_agent_request_with_provider_routing():
    requests = []

    def respond(request):
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "test",
                "object": "chat.completion",
                "created": 0,
                "model": "test",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Done."},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    settings = _model_settings_from_sampling(
        vf.Sampling(
            temperature=1.0,
            top_p=0.95,
            max_tokens=131072,
            top_k=0,
            provider={"only": ["baseten"], "quantizations": ["fp8"]},
        )
    )
    async with AsyncOpenAI(
        api_key="test", http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond))
    ) as client:
        await run_e2b(
            "Do the task",
            model=ChatCompletionsModel(client=client, model="test"),
            model_settings=settings,
            transport=StubTransport(),
            tools=("read",),
            compaction=CompactionConfig(enabled=False),
        )

    assert {key: requests[0][key] for key in ("temperature", "top_p", "max_tokens", "top_k", "provider")} == {
        "temperature": 1.0,
        "top_p": 0.95,
        "max_tokens": 131072,
        "top_k": 0,
        "provider": {"only": ["baseten"], "quantizations": ["fp8"]},
    }


def test_e2b_profile_config_defaults_to_train_and_rejects_unknown_value():
    assert FeatherHarnessConfig().profile == "train"
    assert FeatherHarnessConfig(profile="prod").profile == "prod"
    with pytest.raises(ValidationError):
        FeatherHarnessConfig(profile="unknown")


def test_prod_compaction_keeps_custom_prompt():
    from feather_harness.production import production_compaction

    assert production_compaction(CompactionConfig()).prompt == PRODUCTION_COMPACTION_PROMPT
    assert production_compaction(CompactionConfig(prompt="custom")).prompt == "custom"
