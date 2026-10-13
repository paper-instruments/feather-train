"""Verifiers harness plugin for the native Feather loop and E2B lifecycle."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from typing import Literal, Self

import verifiers.v1 as vf
from feather_harness.core.compaction import CompactionConfig
from feather_harness.core.messages import AssistantMessage, SystemMessage
from feather_harness.core.model import ChatCompletionsModel, ModelSettings
from feather_harness.core.runner import AgentRunError, RunResult
from feather_harness.core.token_budget import PromptTokenBudget
from feather_harness.production import Profile
from feather_harness.tools.web import WEB_TOOL_NAMES
from feather_harness.tools.web.exa import ExaConfigurationError, ExaWebProvider
from openai import AsyncOpenAI
from pydantic import Field, FilePath, JsonValue, field_validator, model_validator

from feather_train.agent import DEFAULT_TOOLS, ToolMetrics, run_e2b, validate_tool_selection
from feather_train.deliverable_capture import capture_deliverables
from feather_train.errors import NON_RETRYABLE_CAUSES, NonRetryableHarnessError
from feather_train.grading.grader import (
    GradingConfig,
    HarborGradingError,
    grade_task,
)
from feather_train.grading.report import build_judge_span_output
from feather_train.grading.rewards import TrajectoryRewardConfig, adjust_trajectory_reward
from feather_train.grading.types import AgentEvidence
from feather_train.grading.unsent_tool_results import (
    count_tool_content_tokens,
    get_unsent_tool_turn,
)
from feather_train.model_retry import EndpointGate, ModelRetryConfig, RetryingModel
from feather_train.observability.braintrust import (
    DEFAULT_BT_PROJECT,
    BraintrustRunLogger,
    emit_grading_step_spans,
)
from feather_train.observability.payloads import (
    braintrust_close_metrics,
    build_braintrust_rollout_start,
)
from feather_train.sandbox.lifecycle import ComposeSandbox
from feather_train.sandbox.paths import WORKSPACE_DIR
from feather_train.sandbox.runtime import build_runtime, install_runtime
from feather_train.sandbox.transport import E2BRuntime
from feather_train.tasks.harbor_spec import ArtifactSpec
from feather_train.tasks.paths import is_safe_task_id

logger = logging.getLogger(__name__)
_DOMAIN = re.compile(r"(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z][a-z0-9-]{0,62}")


class FeatherHarnessConfig(vf.HarnessConfig):
    id: Literal["feather_train"] = "feather_train"
    profile: Profile = "train"
    prompt_token_budget: PromptTokenBudget | None = None
    additional_instructions: str | None = None
    capture_deliverables: bool = False
    capture_paths: tuple[str, ...] = ()
    capture_output_dir: str = "outputs/deliverables"
    capture_volume_name: str | None = None
    enable_image_tools: bool = False
    # Defaults to enable_image_tools; false keeps MCP images while read stays text-only.
    enable_image_reads: bool | None = None
    max_turns: int = Field(default=250, ge=1)
    sandbox_timeout_minutes: int = Field(default=180, gt=0)
    sandbox_start_max_attempts: int = Field(default=3, ge=1)
    generation_timeout_seconds: float = Field(default=600, gt=0)
    model_request_retries: int = Field(default=2, ge=0)
    # Evaluation-only waiting through outages while preserving the active rollout.
    model_retry: ModelRetryConfig | None = None
    runtime_archive: FilePath | None = None
    compaction: CompactionConfig = CompactionConfig(enabled=True)
    tools: tuple[str, ...] = DEFAULT_TOOLS
    # Each domain also blocks its subdomains, for the solver's container only.
    blocked_egress_domains: tuple[str, ...] = ()
    blocked_web_content_markers: tuple[str, ...] = ()
    grading: GradingConfig = GradingConfig()
    trajectory_reward: TrajectoryRewardConfig = TrajectoryRewardConfig()
    enable_braintrust: bool = True
    bt_project: str = DEFAULT_BT_PROJECT
    bt_experiment_name: str | None = None
    bt_experiment_metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("blocked_egress_domains")
    @classmethod
    def valid_domains(cls, domains: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(dict.fromkeys(domain.lower().rstrip(".") for domain in domains))
        invalid = [domain for domain in normalized if not _DOMAIN.fullmatch(domain)]
        if invalid:
            raise ValueError(f"blocked_egress_domains must be DNS names, got {invalid}")
        return normalized

    @field_validator("blocked_web_content_markers")
    @classmethod
    def valid_web_content_markers(cls, markers: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(dict.fromkeys(marker.strip().casefold() for marker in markers))
        if any(not marker for marker in normalized):
            raise ValueError("blocked_web_content_markers must contain nonempty strings")
        return normalized

    @model_validator(mode="after")
    def supported_tools(self) -> Self:
        validate_tool_selection(self.tools)
        if self.enable_image_reads and not self.enable_image_tools:
            raise ValueError("enable_image_reads requires enable_image_tools")
        if self.capture_deliverables and not self.capture_paths:
            raise ValueError("capture_deliverables requires explicit capture_paths")
        for path in self.capture_paths:
            root = PurePosixPath(path)
            if (
                not root.is_absolute()
                or ".." in root.parts
                or root.parts[:2] != ("/", "filesystem")
                or any(part in {".agents", ".feather"} for part in root.parts)
            ):
                raise ValueError("capture_paths must stay under /filesystem outside runtime directories")
        if not self.capture_output_dir.strip():
            raise ValueError("capture_output_dir must not be empty")
        return self

    def tools_for_task(self, *, requires_web_search: bool) -> tuple[str, ...]:
        tools = tuple(name for name in self.tools if name not in WEB_TOOL_NAMES)
        return (*tools, *WEB_TOOL_NAMES) if requires_web_search else tools


class FeatherHarness(vf.Harness[FeatherHarnessConfig]):
    APPENDS_SYSTEM_PROMPT = True
    SUPPORTS_MESSAGE_PROMPT = True

    def __init__(self, config: FeatherHarnessConfig) -> None:
        super().__init__(config)
        self._runtime_directory: TemporaryDirectory | None = None
        if config.runtime_archive is None:
            self._runtime_directory = TemporaryDirectory(prefix="feather-runtime-")
            self._runtime_archive = Path(self._runtime_directory.name) / "runtime.tar.gz"
            build_runtime(self._runtime_archive, profile=config.profile)
        else:
            self._runtime_archive = config.runtime_archive
        self.braintrust = BraintrustRunLogger(
            enable=config.enable_braintrust,
            project=config.bt_project,
            experiment_name=config.bt_experiment_name,
            experiment_metadata=config.bt_experiment_metadata,
        )
        self._endpoint_gate = EndpointGate()

    async def run(
        self,
        ctx: vf.ModelContext,
        trace: vf.Trace,
        runtime: vf.Runtime,
        endpoint: str,
        secret: str,
        mcp_urls: dict[str, str],
    ) -> None:
        # Cancellation propagates so Verifiers can record its timeout before score.
        try:
            await super().run(ctx, trace, runtime, endpoint, secret, mcp_urls)
        except Exception as exc:
            await self.finish_trace(trace, error=exc)
            raise
        else:
            await self.finish_trace(trace)

    async def score(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        await self.finish_trace(trace)
        await super().score(trace, runtime)

    async def finalize_interrupted_trace(self, trace: vf.Trace, error: BaseException) -> None:
        # Shared Verifiers training/eval drivers discover this interruption hook.
        await self.finish_trace(trace, error=error)

    async def finish_trace(self, trace: vf.Trace, *, error: BaseException | None = None) -> None:
        """Close logging once Verifiers has recorded the terminal outcome."""
        rollout = self.braintrust.active_rollout(trace.id)
        if rollout is None:
            return
        trace_error = str(error) if error is not None else None
        if trace.errors:
            trace_error = trace.errors[-1].message
        try:
            self.braintrust.close_rollout(
                trace.id,
                metrics=braintrust_close_metrics(trace, grading_score=rollout.grading_score),
                metadata={
                    "profile": self.config.profile,
                    "compaction_mode": "policy" if self.config.compaction.enabled else None,
                    "stop_condition": trace.stop_condition,
                    # Empty cancellation messages must not erase the recorded failure phase.
                    "error": trace_error or None,
                },
            )
            await self.braintrust.flush_async()
        except Exception:
            logger.warning("Braintrust close failed", exc_info=True)

    async def launch(
        self,
        ctx: vf.ModelContext,
        trace: vf.Trace,
        runtime: vf.Runtime,
        endpoint: str,
        secret: str,
        mcp_urls: dict[str, str],
    ) -> vf.ProgramResult:
        return await self.execute_task(
            ctx, trace, trace.task.data, endpoint=endpoint, secret=secret
        )

    async def count_tool_output_tokens(
        self, ctx: vf.ModelContext, messages: list[dict]
    ) -> int | None:
        """Render locally and count tool content, excluding the assistant call and template."""
        from renderers import DefaultRenderer

        from feather_train.clients.renderer import RendererClient

        if not isinstance(ctx.client, RendererClient):
            return None  # API-only evaluation has no training renderer.
        if issubclass(ctx.client.renderer.renderer_cls, DefaultRenderer):
            return None  # This renderer does not provide body attribution.
        rendered = await asyncio.to_thread(
            ctx.client.renderer.render, messages, add_generation_prompt=False
        )
        return count_tool_content_tokens(rendered)

    async def execute_task(
        self,
        ctx: vf.ModelContext,
        trace: vf.Trace,
        task: vf.TaskData,
        *,
        endpoint: str,
        secret: str,
    ) -> vf.ProgramResult:
        model = ctx.model
        if self.config.capture_deliverables and (
            task.source != "benchmark"
            or not task.eval_run_id
            or not all(is_safe_task_id(value) for value in (task.eval_run_id, task.task_id, trace.id))
        ):
            raise NonRetryableHarnessError("Deliverable capture requires benchmark eval run/task/trace identities")
        tools = self.config.tools_for_task(requires_web_search=task.requires_web_search)
        try:
            web_provider = (
                ExaWebProvider(
                    api_key=os.environ.get("EXA_API_KEY", ""),
                    blocked_domains=self.config.blocked_egress_domains,
                    blocked_content_markers=self.config.blocked_web_content_markers,
                )
                if task.requires_web_search
                else None
            )
        except ExaConfigurationError as exc:
            raise NonRetryableHarnessError(str(exc)) from exc
        prompt = task.prompt
        if (
            not isinstance(prompt, list)
            or len(prompt) != 1
            or not isinstance(prompt[0].content, str)
        ):
            raise NonRetryableHarnessError("Feather tasks require one text prompt")
        if not endpoint or not secret:
            raise NonRetryableHarnessError("Verifiers endpoint and secret are required")
        rollout = self.braintrust.open_rollout(
            build_braintrust_rollout_start(
                task,
                trace,
                model_str=model,
                max_turns=self.config.max_turns,
                tools=list(tools),
            )
        )
        rollout.add_metadata(
            {
                "profile": self.config.profile,
                "sampling_args": ctx.sampling.model_dump(exclude_none=True),
                "blocked_egress_domains": list(self.config.blocked_egress_domains),
                "web_research_policy": {
                    "enabled": web_provider is not None,
                    "blocked_domains": list(web_provider.blocked_domains) if web_provider else [],
                    "blocked_content_markers": (
                        list(web_provider.blocked_content_markers) if web_provider else []
                    ),
                },
            }
        )
        for key, value in [
            ("bt_span_export", rollout.export),
            ("bt_span_id", rollout.span_id),
            ("bt_span_permalink", rollout.permalink),
        ]:
            if value is not None:
                trace.info[key] = value
        if not is_safe_task_id(trace.id):
            raise NonRetryableHarnessError("Invalid trace identifier for run directory")
        phase = "sandbox_start"
        metrics = ToolMetrics()
        try:
            async with ComposeSandbox.start(
                task,
                timeout=self.config.sandbox_timeout_minutes * 60,
                max_attempts=self.config.sandbox_start_max_attempts,
                blocked_egress_domains=self.config.blocked_egress_domains,
            ) as world:
                trace.info["sandbox_id"] = world.sandbox.sandbox_id
                rollout.add_metadata(
                    {
                        "sandbox_id": world.sandbox.sandbox_id,
                        "sandbox_snapshot_id": task.e2b_snapshot_id,
                        "sandbox_provider": "e2b",
                    }
                )
                phase = "runtime_install"
                await install_runtime(
                    world.sandbox,
                    self._runtime_archive,
                    workspace=task.workdir or WORKSPACE_DIR,
                    profile=self.config.profile,
                )
                servers = []
                for server in world.runtime.mcp_servers:
                    if server.name == "submission":
                        if (
                            task.source != "benchmark"
                            and task.submission_transport != "submit_tool"
                        ):
                            continue
                        if task.submission_transport == "submit_tool":
                            server = server.model_copy(update={"raise_on_error_result": True})
                    servers.append(server)
                # The agent sees this path in shell output, so it must not name the benchmark.
                logs = f"/app/.feather/sessions/{trace.id}/tasks"
                workspace = task.workdir or WORKSPACE_DIR
                transport = E2BRuntime(
                    world.sandbox,
                    workspace=workspace,
                    log_directory=logs,
                    write_grants=(workspace, "/task", logs.rsplit("/", 1)[0]),
                )
                phase = "agent"
                agent_model = retrying_model = None
                async with AsyncOpenAI(
                    base_url=endpoint,
                    api_key=secret,
                    max_retries=self.config.model_request_retries,
                    timeout=self.config.generation_timeout_seconds,
                ) as client:
                    agent_model = ChatCompletionsModel(
                        client=rollout.wrap_openai(client), model=model,
                        prompt_token_budget=self.config.prompt_token_budget
                    )
                    if self.config.model_retry is not None:
                        agent_model = retrying_model = RetryingModel(
                            agent_model, self.config.model_retry, self._endpoint_gate
                        )
                    try:
                        result = await run_e2b(
                            prompt[0].content,
                            instructions=task.system_prompt,
                            additional_instructions=self.config.additional_instructions,
                            model=agent_model,
                            transport=transport,
                            max_turns=self.config.max_turns,
                            model_settings=_model_settings_from_sampling(ctx.sampling),
                            enable_image_tools=self.config.enable_image_tools,
                            enable_image_reads=self.config.enable_image_reads,
                            profile=self.config.profile,
                            compaction=self.config.compaction,
                            mcp_servers=tuple(servers),
                            tools=tools,
                            web_provider=web_provider,
                            metrics=metrics,
                            rollout=rollout,
                        )
                    finally:
                        if retrying_model is not None:
                            rollout.add_metadata(
                                {
                                    "endpoint_outage_wait_seconds": round(
                                        retrying_model.waited_seconds, 1
                                    ),
                                    "endpoint_outage_retries": retrying_model.retries,
                                }
                            )
                unsent_tool_turn = get_unsent_tool_turn(result)
                unsent_tool_output_tokens = 0
                if unsent_tool_turn:
                    unsent_tool_output_tokens = await self.count_tool_output_tokens(
                        ctx, unsent_tool_turn
                    )
                trace.info["unsent_tool_output_tokens"] = unsent_tool_output_tokens
                calls = sum(
                    len(message.tool_calls)
                    for message in result.execution_messages()
                    if isinstance(message, AssistantMessage)
                )
                unknown = calls - metrics.known_calls
                tool_metrics = {
                    "tool_call_count": calls,
                    "tool_error_schema_invalid": metrics.invalid_calls,
                    "tool_error_unknown_tool": unknown,
                }
                trace.info.update(tool_metrics)
                trace.record_metrics(tool_metrics)
                rollout.record_agent_result(
                    final_answer=result.final_answer, num_turns=result.num_turns
                )
                usage = result.usage()
                rollout.record_usage(usage)
                # doteval stores trace.info; the proxy's own usage omits BYOK upstream cost.
                trace.info["model_usage"] = usage.model_dump(mode="json", exclude_none=True)
                trace.info["model_usage_calls"] = _usage_calls(result)
                rollout.add_metadata(
                    {
                        "compactions": result.compactions,
                        "agent_stop_reason": None
                        if result.stop_reason == "completed"
                        else result.stop_reason,
                        "agent_stop_message": result.stop_message,
                    }
                )
                if self.config.capture_deliverables:
                    phase = "deliverable_capture"
                    with rollout.child_span(name="deliverable_capture", type="task") as span:
                        capture = await capture_deliverables(
                            world,
                            task,
                            trace_id=trace.id,
                            final_answer=result.final_answer,
                            model=model,
                            sampling=ctx.sampling.model_dump(exclude_none=True),
                            config=self.config,
                        )
                        trace.info["deliverable_capture"] = capture
                        rollout.add_metadata({"deliverable_capture": capture})
                        span.log(output=capture)
                evidence = None
                if self.config.grading.stage_agent_trace:
                    evidence = AgentEvidence(
                        model=model, instructions=_agent_system_instructions(result), trace=result
                    )
                phase = "grading_isolation"
                if evidence is not None and _needs_agent_trajectory_artifact(task):
                    await world.stage_agent_trace(
                        (
                            json.dumps(
                                {
                                    **evidence.as_dict(),
                                    "task_id": task.task_id,
                                    "final_response": result.final_answer,
                                },
                                ensure_ascii=False,
                                sort_keys=True,
                                separators=(",", ":"),
                            )
                            + "\n"
                        ).encode("utf-8")
                    )
                verifier = await world.prepare_grading(task)
                phase = "grading"
                with rollout.child_span(name="judge", type="score", set_current=True) as span:
                    try:
                        grade = await grade_task(
                            task,
                            verifier,
                            instruction=prompt[0].content,
                            config=self.config.grading,
                            final_answer=result.final_answer,
                            agent_evidence=evidence,
                            braintrust_parent=span.export()
                            if rollout.root_span is not None
                            else None,
                            braintrust_project=self.config.bt_project,
                        )
                    except HarborGradingError as exc:
                        emit_grading_step_spans(span, exc.steps)
                        raise
                    report = build_judge_span_output(grade)
                    span.log(output=report)
                    emit_grading_step_spans(span, grade.steps)
                reward = adjust_trajectory_reward(
                    grade.score,
                    tool_call_count=calls,
                    schema_invalid_tool_call_count=metrics.invalid_calls,
                    unknown_tool_call_count=unknown,
                    config=self.config.trajectory_reward,
                )
                values = {
                    "grading_score": grade.score,
                    "training_reward": reward.value,
                    "trajectory_tool_penalty": reward.penalty,
                    "trajectory_zero_tool_calls": float(reward.zero_tool_calls),
                }
                trace.info.update(values, grading_report=report)
                trace.record_metrics(values)
                rollout.record_completed(
                    grading_score=grade.score, judge_attempts=len(grade.failed_attempts) + 1
                )
        except asyncio.CancelledError as exc:
            rollout.record_interrupted(phase=phase, exc=exc, fallback_num_turns=trace.num_turns)
            raise
        except Exception as exc:
            if isinstance(exc, AgentRunError):
                rollout.record_agent_failed(
                    num_turns=exc.result.num_turns,
                    compactions=exc.result.compactions,
                    stop_message=str(exc),
                    tool_metrics={"tool_error_schema_invalid": metrics.invalid_calls},
                )
                usage = exc.result.usage()
                rollout.record_usage(usage)
                # doteval stores trace.info; the proxy's own usage omits BYOK upstream cost.
                trace.info["model_usage"] = usage.model_dump(mode="json", exclude_none=True)
                trace.info["model_usage_calls"] = _usage_calls(exc.result)
                retryable = not isinstance(exc.cause, NON_RETRYABLE_CAUSES)
            elif isinstance(exc, HarborGradingError):
                rollout.record_grading_failed(
                    errors=[failure.as_error() for failure in exc.failed_attempts],
                    judge_attempts=len(exc.failed_attempts),
                )
                retryable = exc.retryable
            else:
                rollout.record_infra_failed(
                    phase=phase, exc=exc, fallback_num_turns=trace.num_turns
                )
                retryable = not isinstance(exc, NON_RETRYABLE_CAUSES)
            if retryable or isinstance(exc, NonRetryableHarnessError):
                raise
            raise NonRetryableHarnessError(str(exc)) from exc
        return vf.ProgramResult(exit_code=0, stdout="", stderr="")


def _model_settings_from_sampling(sampling: vf.Sampling) -> ModelSettings:
    args = sampling.model_dump(exclude_none=True)
    extra_body = args.pop("extra_body", {})
    direct = {
        key: args.pop(key)
        for key in ("temperature", "top_p", "max_tokens", "reasoning_effort")
        if key in args
    }
    return ModelSettings(**direct, extra_body={**extra_body, **args})


def _usage_calls(result: RunResult) -> list[dict]:
    """Per-call billed usage in call order, numbered across compactions."""
    calls = [call for segment in result.segments for call in segment.calls]
    return [{"call": number, **call.accounting_row()} for number, call in enumerate(calls)]


def _agent_system_instructions(result: RunResult) -> str:
    """Preserve the exact system message sent to the policy in grader evidence."""
    for segment in result.segments:
        for message in segment.input:
            if isinstance(message, SystemMessage):
                return message.content
    raise ValueError("Agent trace is missing the model-visible system instructions")


def _needs_agent_trajectory_artifact(task: vf.TaskData) -> bool:
    if task.environment_mode != "separate":
        return False
    return any(
        artifact == "/logs/agent/trajectory.json"
        if isinstance(artifact, str)
        else isinstance(artifact, ArtifactSpec)
        and artifact.source == "/logs/agent/trajectory.json"
        and artifact.service in {"main", "agent"}
        for artifact in task.artifacts
    )
