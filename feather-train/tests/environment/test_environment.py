# ruff: noqa: E402
import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

vf = pytest.importorskip("verifiers.v1")

from feather_train.environment import load_environment
from feather_train.harness import FeatherHarness, FeatherHarnessConfig
from feather_train.observability.braintrust import BraintrustRolloutStart
from feather_train.tasks.taskset import FeatherTaskData


def test_model_request_retry_config_keeps_default_and_rejects_negative():
    from pydantic import ValidationError

    assert FeatherHarnessConfig().model_request_retries == 2
    env = load_environment(
        hf_dataset="unused/test", hf_revision="a" * 40,
        enable_braintrust=False, model_request_retries=12,
    )
    assert env.harness.config.model_request_retries == 12
    with pytest.raises(ValidationError, match="model_request_retries"):
        FeatherHarnessConfig(model_request_retries=-1)


@pytest.mark.parametrize("retries,expected_attempts", [(0, 1), (12, 4)])
async def test_configured_retries_reach_actual_harness_client(trace, monkeypatch, retries, expected_attempts):
    import json

    import httpx
    from feather_harness import CompactionConfig, FeatherAgent, RunRequest, UserMessage
    from feather_harness.core.runner import AgentRunError
    from openai import AsyncOpenAI, InternalServerError

    from feather_train import harness as harness_module
    from feather_train.grading.grader import GradingConfig
    from feather_train.grading.types import GradeResult

    attempts = 0
    clients = []

    def respond(request):
        nonlocal attempts
        attempts += 1
        assert json.loads(request.content)["max_tokens"] == 65536
        if attempts <= 3:
            return httpx.Response(503, json={"error": {"message": "temporarily unavailable"}})
        return httpx.Response(200, json={
            "id": "reply", "object": "chat.completion", "created": 0, "model": "test",
            "choices": [{"index": 0, "finish_reason": "stop", "message": {
                "role": "assistant", "content": "done",
            }}],
        })

    def client(**kwargs):
        sdk = AsyncOpenAI(**kwargs, http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)))
        sdk._calculate_retry_timeout = lambda *args, **kwargs: 0
        clients.append(sdk)
        return sdk

    @asynccontextmanager
    async def start(task, **kwargs):
        async def prepare_grading(task):
            return object()

        yield SimpleNamespace(
            sandbox=SimpleNamespace(sandbox_id="test"),
            runtime=SimpleNamespace(mcp_servers=()),
            prepare_grading=prepare_grading,
        )

    async def install(*args, **kwargs):
        pass

    async def run(prompt, *, model, **kwargs):
        return await FeatherAgent(model=model, compaction=CompactionConfig(enabled=False)).run(
            RunRequest(messages=(UserMessage(content=prompt),), model_settings=kwargs["model_settings"])
        )

    async def grade(*args, **kwargs):
        return GradeResult(score=1, num_met=1, num_criteria=1, criteria=(), stdout="", failed_attempts=[])

    monkeypatch.setattr(harness_module, "AsyncOpenAI", client)
    monkeypatch.setattr(harness_module.ComposeSandbox, "start", start)
    monkeypatch.setattr(harness_module, "install_runtime", install)
    monkeypatch.setattr(harness_module, "E2BRuntime", lambda *args, **kwargs: object())
    monkeypatch.setattr(harness_module, "run_e2b", run)
    monkeypatch.setattr(harness_module, "grade_task", grade)
    harness = FeatherHarness(FeatherHarnessConfig(
        model_request_retries=retries,
        enable_braintrust=False,
        grading=GradingConfig(stage_agent_trace=False),
    ))
    launch = harness.launch(
        SimpleNamespace(model="test", sampling=vf.Sampling(max_tokens=65536)), trace, None,
        "https://model.invalid/v1", "test-key", {},
    )
    if retries == 0:
        with pytest.raises(AgentRunError) as failure:
            await launch
        assert isinstance(failure.value.cause, InternalServerError)
    else:
        result = await launch
        assert result.exit_code == 0
        assert trace.info["grading_score"] == 1
    assert clients[0].max_retries == retries
    assert attempts == expected_attempts


def test_new_environment_loads_without_loading_dataset():
    env = load_environment(hf_dataset="unused/test", hf_revision="a" * 40, enable_braintrust=False)
    assert isinstance(env.harness, FeatherHarness)
    assert env.harness.config.compaction.enabled
    assert env.harness.config.id == "feather_train"
    assert env.taskset.config.id == "feather_train"


@pytest.fixture
def trace():
    task = FeatherTaskData.model_validate(
        {
            "idx": 0,
            "task_id": "test",
            "task_name": "test",
            "world_id": "test",
            "hf_dataset": "unused/test",
            "hf_revision": "a" * 40,
            "prompt": [{"role": "user", "content": "solve"}],
            "rubric": [],
            "rubric_doc": {},
            "expected_deliverable": {},
            "submission_transport": "text_response",
            "grading_evidence_mode": "direct_text",
            "requires_web_search": False,
            "network_mode": "public",
            "mcp_servers": [],
        }
    )
    return vf.Trace(
        task=vf.TraceTask(type="FeatherTask", data=task),
        agent=vf.AgentInfo(model="test", sampling=vf.Sampling()),
    )


async def test_completion_is_stamped_before_braintrust_closes(trace):
    class CompletedHarness(FeatherHarness):
        async def launch(self, *args):
            return vf.ProgramResult(exit_code=0, stdout="", stderr="")

    harness = CompletedHarness(FeatherHarnessConfig(enable_braintrust=False))
    rollout = harness.braintrust.open_rollout(
        BraintrustRolloutStart(
            key=trace.id,
            model="test",
            name="test",
            input="",
            expected="",
            metadata={},
        )
    )
    await harness.run(None, trace, None, "endpoint", "secret", {})
    assert rollout.metadata["stop_condition"] == "agent_completed"
    assert harness.braintrust.active_rollout(trace.id) is None


async def test_timeout_closes_with_final_verifiers_stop_condition(trace):
    class CancelledHarness(FeatherHarness):
        async def launch(self, *args):
            raise asyncio.CancelledError

    harness = CancelledHarness(FeatherHarnessConfig(enable_braintrust=False))
    rollout = harness.braintrust.open_rollout(
        BraintrustRolloutStart(
            key=trace.id,
            model="test",
            name="test",
            input="",
            expected="",
            metadata={},
        )
    )
    with pytest.raises(asyncio.CancelledError):
        await harness.run(None, trace, None, "endpoint", "secret", {})
    assert harness.braintrust.active_rollout(trace.id) is rollout
    trace.stop("harness_timeout")
    await harness.score(trace, None)
    assert rollout.metadata["stop_condition"] == "harness_timeout"
    assert harness.braintrust.active_rollout(trace.id) is None


async def test_compaction_does_not_consume_agent_turn_budget(trace):
    from feather_harness import (
        AssistantMessage,
        FeatherAgent,
        RunRequest,
        Tool,
        ToolCall,
        ToolResult,
        ToolSpec,
        UserMessage,
    )
    from feather_harness.core.model import ModelCall
    from verifiers.v1.graph import MessageNode
    from verifiers.v1.session import RolloutLimits

    env = load_environment(
        hf_dataset="unused/test", hf_revision="a" * 40, max_turns=3, enable_braintrust=False
    )
    limits = RolloutLimits(max_turns=env.config.max_turns)
    calls = []
    completed_tools = []

    class Model:
        async def complete(self, context, *, settings, tools=(), kind="agent"):
            assert limits.reached(trace) is None
            calls.append(kind)
            trace.nodes.append(
                MessageNode(message=vf.AssistantMessage(content=kind), sampled=True)
            )
            assistant = (
                AssistantMessage(content="summary")
                if kind == "compaction"
                else AssistantMessage(
                    tool_calls=(ToolCall(id=str(len(calls)), name="work", arguments="{}"),)
                )
            )
            return ModelCall(assistant=assistant, input_tokens=120000, output_tokens=3)

    async def work(raw):
        completed_tools.append(raw)
        return ToolResult(content="finished")

    result = await FeatherAgent(
        model=Model(),
        tools=(Tool(spec=ToolSpec(name="work", description="work", parameters={}), invoke=work),),
        compaction=env.harness.config.compaction,
    ).run(
        RunRequest(messages=(UserMessage(content="work"),), max_turns=env.harness.config.max_turns)
    )

    assert calls == ["agent", "compaction", "agent", "agent"]
    assert len(completed_tools) == result.num_turns == 3
    assert result.compactions == 1
    assert result.stop_reason == "MaxTurnsExceeded"


@pytest.mark.parametrize("configured", [(), ("read",), ("read", "web_fetch", "web_search")])
@pytest.mark.parametrize("required", [False, True])
def test_task_flag_controls_both_web_tools(configured, required):
    config = FeatherHarnessConfig(tools=configured)
    non_web = tuple(name for name in configured if name == "read")
    assert config.tools_for_task(requires_web_search=required) == (
        (*non_web, "web_search", "web_fetch") if required else non_web
    )


async def test_missing_web_credentials_are_non_retryable_before_sandbox_start(trace, monkeypatch):
    from feather_train.errors import NonRetryableHarnessError

    monkeypatch.delenv("EXA_API_KEY", raising=False)
    trace = vf.Trace(
        task=vf.TraceTask(
            type="FeatherTask",
            data=trace.task.data.model_copy(update={"requires_web_search": True}),
        ),
        agent=trace.agent,
    )
    harness = FeatherHarness(FeatherHarnessConfig(enable_braintrust=False))
    with pytest.raises(NonRetryableHarnessError, match="EXA_API_KEY") as error:
        await harness.launch(
            SimpleNamespace(model="test", sampling=vf.Sampling()),
            trace,
            None,
            "endpoint",
            "secret",
            {},
        )
    # Verifiers matches retry exclusions by exact error type name.
    assert type(error.value) is NonRetryableHarnessError


def test_web_content_policy_normalizes_markers_and_rejects_empty_values():
    from pydantic import ValidationError

    config = FeatherHarnessConfig(blocked_web_content_markers=(" ROGOAI ", "rogoai"))
    assert config.blocked_web_content_markers == ("rogoai",)
    assert FeatherHarnessConfig().blocked_web_content_markers == ()
    with pytest.raises(ValidationError, match="nonempty"):
        FeatherHarnessConfig(blocked_web_content_markers=(" ",))


async def test_launch_binds_effective_web_policy_before_sandbox_execution(trace, monkeypatch):
    from feather_train import harness as harness_module
    from feather_train.errors import NonRetryableHarnessError

    monkeypatch.setenv("EXA_API_KEY", "test-key")
    provider_type = harness_module.ExaWebProvider
    providers = []

    def create_provider(**kwargs):
        provider = provider_type(**kwargs)
        providers.append(provider)
        return provider

    def stop_before_sandbox(*args, **kwargs):
        raise NonRetryableHarnessError("offline policy gate")

    monkeypatch.setattr(harness_module, "ExaWebProvider", create_provider)
    monkeypatch.setattr(harness_module.ComposeSandbox, "start", stop_before_sandbox)
    trace = vf.Trace(
        task=vf.TraceTask(
            type="FeatherTask",
            data=trace.task.data.model_copy(update={"requires_web_search": True}),
        ),
        agent=trace.agent,
    )
    harness = FeatherHarness(FeatherHarnessConfig(
        enable_braintrust=False,
        blocked_egress_domains=("GitHub.COM.",),
        blocked_web_content_markers=(" ROGOAI ",),
    ))
    with pytest.raises(NonRetryableHarnessError, match="offline policy gate"):
        await harness.launch(
            SimpleNamespace(model="test", sampling=vf.Sampling()),
            trace, None, "endpoint", "secret", {},
        )
    assert len(providers) == 1
    assert providers[0].blocked_domains == ("github.com",)
    assert providers[0].blocked_content_markers == ("rogoai",)
    rollout = harness.braintrust.active_rollout(trace.id)
    assert rollout.metadata["blocked_egress_domains"] == ["github.com"]
    assert rollout.metadata["web_research_policy"] == {
        "enabled": True,
        "blocked_domains": ["github.com"],
        "blocked_content_markers": ["rogoai"],
    }
