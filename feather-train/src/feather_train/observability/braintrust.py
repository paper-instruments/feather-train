"""Braintrust logging for Feather training and evaluation."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import time
import traceback
from collections.abc import Iterator, Mapping
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from typing import Any, Literal

import braintrust
from feather_harness.core.messages import Message
from feather_harness.core.model import ChatCompletionsModel, ModelCall, ModelSettings, UsageTotals
from feather_harness.tools import Tool, ToolResult, ToolSpec
from openai import AsyncOpenAI

from feather_train.grading.types import GradeStep

DEFAULT_BT_PROJECT = "feather-agents"
MAX_INLINE_BRAINTRUST_ERROR_CHARS = 1_000_000
ERROR_TRUNCATION_NOTICE = "\n... inline traceback truncated; full traceback attached ...\n"

logger = logging.getLogger(__name__)

RolloutStatus = Literal["incomplete", "completed", "error", "interrupted"]


def emit_grading_step_spans(
    parent: braintrust.Span,
    steps: tuple[GradeStep, ...],
) -> None:
    """Emit externally timed grading work without affecting its outcome."""
    for step in steps:
        try:
            span = parent.start_span(
                name=step.name,
                type="task",
                start_time=step.start,
            )
            span.log(metadata={**step.metadata, "ok": step.ok})
            span.end(step.end)
        except Exception:  # noqa: BLE001
            logger.warning(
                "harness: failed to emit grading step span %r",
                step.name,
                exc_info=True,
            )


@dataclass(frozen=True, slots=True)
class BraintrustRolloutStart:
    key: str
    model: str
    name: str
    input: Any
    expected: Any
    metadata: dict[str, Any]


class BraintrustRollout:
    """Owns live Braintrust spans for one rollout."""

    def __init__(self, root_span: braintrust.Span | None) -> None:
        self.root_span = root_span
        self.export: str | None = None
        self.span_id: str | None = None
        self.permalink: str | None = None
        if root_span is not None:
            try:
                self.export = root_span.export()
                self.span_id = getattr(root_span, "root_span_id")
                self.permalink = root_span.permalink()
            except Exception:  # noqa: BLE001
                logger.debug("braintrust span reference creation failed", exc_info=True)
        self.final_answer = ""
        self.status: RolloutStatus = "incomplete"
        self.num_turns = 0
        self.grading_score: float | None = None
        self.metrics: dict[str, int | float] = {}
        self.metadata: dict[str, Any] = {}
        if self.root_span is not None:
            self.root_span.__enter__()

    def wrap_openai(self, client: AsyncOpenAI) -> AsyncOpenAI:
        if self.root_span is None:
            return client
        return braintrust.wrap_openai(client)

    def wrap_tool(self, tool: Tool) -> Tool:
        if self.root_span is None:
            return tool

        async def invoke(arguments: str) -> ToolResult:
            try:
                inputs = json.loads(arguments)
            except json.JSONDecodeError:
                inputs = arguments
            with braintrust.current_span().start_span(
                name=tool.spec.name,
                type="tool",
                input=inputs,
                set_current=True,
            ) as span:
                result = await tool.invoke(arguments)
                span.log(output=result.content, error=result.error, metadata=result.metadata)
                return result

        return Tool(tool.spec, invoke)

    @contextmanager
    def trace_agent(
        self, model: ChatCompletionsModel
    ) -> Iterator[ChatCompletionsModel | _TracedModel]:
        if self.root_span is None:
            yield model
        else:
            # Enter and exit in the task running the agent, including MCP-owned runs.
            with ExitStack() as windows:
                yield _TracedModel(model, self.root_span, windows)

    @contextmanager
    def child_span(self, **kwargs) -> Iterator[braintrust.Span]:
        parent = self.root_span
        if parent is None:
            yield braintrust.NOOP_SPAN
        else:
            with parent.start_span(**kwargs) as span:
                yield span

    def record_agent_result(self, *, final_answer: str, num_turns: int) -> None:
        self.final_answer = final_answer
        self.num_turns = num_turns

    def record_agent_failed(
        self,
        *,
        num_turns: int,
        compactions: int,
        stop_message: str,
        tool_metrics: Mapping[str, object],
    ) -> None:
        self.status = "error"
        self.num_turns = num_turns
        self.add_metrics(tool_metrics)
        metadata = {
            "compactions": compactions,
            "agent_stop_reason": "AgentRunError",
            "agent_stop_message": stop_message,
        }
        self.add_metadata(metadata)

    def record_usage(self, usage: UsageTotals) -> None:
        """Rollout totals of provider-billed usage; per-call values are on the LLM spans."""
        totals = usage.model_dump(mode="json", exclude_none=True, exclude={"providers"})
        self.add_metrics(
            {name if name.startswith("model_") else f"model_{name}": value
             for name, value in totals.items()}
        )
        self.add_metadata({"model_providers": list(usage.providers)})

    def record_completed(
        self,
        *,
        grading_score: float,
        judge_attempts: int,
    ) -> None:
        self.status = "completed"
        self.grading_score = grading_score
        self.add_metrics({"judge_attempts": judge_attempts})

    def record_grading_failed(
        self, *, errors: list[dict[str, object]], judge_attempts: int
    ) -> None:
        self.status = "error"
        self.add_metrics({"judge_attempts": judge_attempts})
        self.add_metadata(
            {
                "grading_failed": True,
                "grading_errors": errors,
            }
        )

    def record_infra_failed(
        self,
        *,
        phase: str,
        exc: BaseException,
        fallback_num_turns: int,
    ) -> None:
        self.status = "error"
        self.num_turns = self.num_turns or fallback_num_turns
        full_error = "".join(traceback.format_exception(exc))
        error = full_error
        metadata: dict[str, object] = {
            "infra_failed": True,
            "infra_failed_during": phase,
        }
        if len(full_error) > MAX_INLINE_BRAINTRUST_ERROR_CHARS:
            excerpt_chars = MAX_INLINE_BRAINTRUST_ERROR_CHARS - len(ERROR_TRUNCATION_NOTICE)
            head_chars = excerpt_chars // 2
            tail_chars = excerpt_chars - head_chars
            error = full_error[:head_chars] + ERROR_TRUNCATION_NOTICE + full_error[-tail_chars:]
            metadata.update(
                {
                    "error_traceback_attachment": braintrust.Attachment(
                        data=full_error.encode(),
                        filename="error-traceback.txt",
                        content_type="text/plain; charset=utf-8",
                    ),
                    "error_traceback_truncated": True,
                    "error_traceback_chars": len(full_error),
                }
            )
        if self.root_span is not None:
            self.root_span.log(error=error)
        self.add_metadata(metadata)

    def record_interrupted(
        self,
        *,
        phase: str,
        exc: BaseException,
        fallback_num_turns: int,
    ) -> None:
        self.status = "interrupted"
        self.num_turns = self.num_turns or fallback_num_turns
        self.add_metadata(
            {
                "interrupted_during": phase,
                "error": f"{type(exc).__name__} during {phase}",
            }
        )

    def add_metrics(self, metrics: Mapping[str, object]) -> None:
        self.metrics.update(_numeric_metrics(metrics))

    def add_metadata(self, metadata: Mapping[str, object]) -> None:
        clean = {key: value for key, value in metadata.items() if value is not None}
        self.metadata.update(clean)
        # Flush eagerly so post-start fields (sandbox_provider, mcp_servers, ...) are visible
        # on the root span while the rollout is still running, not only at close.
        if self.root_span is not None and clean:
            self.root_span.log(metadata=clean)

    def close(self) -> None:
        if self.root_span is None:
            return
        try:
            self.root_span.log(
                output={
                    "final_answer": self.final_answer,
                    "status": self.status,
                    "num_turns": self.num_turns,
                    "grading_score": self.grading_score,
                },
                metrics=self.metrics or None,
                metadata=self.metadata,
            )
        finally:
            self.root_span.__exit__(None, None, None)
            self.root_span = None


class _TracedModel:
    """Group agent calls and tools by context, with separate summary spans."""

    def __init__(self, model: ChatCompletionsModel, parent: braintrust.Span, windows: ExitStack):
        self.model = model
        self.parent = parent
        self.windows = windows
        self.window_number = 0
        self._open_window()

    def _open_window(self) -> None:
        self.window_number += 1
        self.windows.enter_context(
            self.parent.start_span(
                name=f"context_window_{self.window_number}",
                type="task",
                set_current=True,
            )
        )
        self.window_open = True

    async def complete(
        self,
        messages: list[Message],
        *,
        tools: tuple[ToolSpec, ...] = (),
        settings: ModelSettings,
        kind: Literal["agent", "compaction"] = "agent",
    ) -> ModelCall:
        if kind == "compaction":
            self.windows.close()
            self.window_open = False
            with self.parent.start_span(
                name=f"compaction_{self.window_number}", type="task", set_current=True
            ):
                return await self.model.complete(
                    messages, tools=tools, settings=settings, kind=kind
                )
        if not self.window_open:
            self._open_window()
        return await self.model.complete(messages, tools=tools, settings=settings, kind=kind)


class BraintrustRunLogger:
    """Owns Braintrust experiment, span state, and rollout cleanup logging."""

    def __init__(
        self,
        *,
        enable: bool,
        project: str = DEFAULT_BT_PROJECT,
        experiment_name: str | None = None,
        experiment_metadata: Mapping[str, object] | None = None,
    ) -> None:
        self.enable = enable
        self.project = project
        self.experiment_name = experiment_name
        self.experiment_metadata = experiment_metadata
        self._experiment: braintrust.Experiment | None = None
        self._active_rollouts: dict[str, BraintrustRollout] = {}

    def open_rollout(
        self,
        start: BraintrustRolloutStart,
    ) -> BraintrustRollout:
        exp = self._ensure_experiment(start.model)
        root_span = self._start_root_span(exp, start) if exp is not None else None
        rollout = BraintrustRollout(root_span)
        self._active_rollouts[start.key] = rollout
        return rollout

    def active_rollout(self, key: str) -> BraintrustRollout | None:
        return self._active_rollouts.get(key)

    def close_rollout(
        self,
        key: str,
        *,
        metrics: Mapping[str, object],
        metadata: Mapping[str, object],
    ) -> None:
        rollout = self._active_rollouts.pop(key, None)
        if rollout is None:
            return
        try:
            rollout.add_metrics(metrics)
            rollout.add_metadata(metadata)
        finally:
            rollout.close()

    def flush(self) -> None:
        if self._experiment is not None:
            self._experiment.flush()

    async def flush_async(self) -> None:
        await asyncio.to_thread(self.flush)

    def _start_root_span(
        self,
        experiment: braintrust.Experiment,
        start: BraintrustRolloutStart,
    ) -> braintrust.Span:
        return braintrust.start_span(
            name=start.name,
            set_current=True,
            parent=experiment.export(),
            state=experiment.state,
            input=start.input,
            expected=start.expected,
            metadata=start.metadata,
        )

    def _ensure_experiment(self, model_str: str) -> braintrust.Experiment | None:
        if not self.enable:
            return None
        if self._experiment is None:
            name = self.experiment_name
            if not name:
                model_short = (model_str or "unknown").rsplit("/", 1)[-1]
                ts = time.strftime("%Y%m%d-%H%M%S")
                name = f"{model_short}/coding/{ts}"
            self._experiment = braintrust.init(
                project=self.project,
                experiment=name,
                metadata=self.experiment_metadata,
                update=True,
            )
        return self._experiment


def _numeric_metrics(metrics: Mapping[str, object]) -> dict[str, int | float]:
    out: dict[str, int | float] = {}
    for key, value in metrics.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            out[key] = value
        elif isinstance(value, float) and math.isfinite(value):
            out[key] = value
    return out
