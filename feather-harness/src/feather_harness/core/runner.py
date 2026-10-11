"""A sequential Feather agent loop with explicit execution and compaction segments."""

from __future__ import annotations

import logging
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Literal

from pydantic import Field

from ..tools import Tool, ToolResult
from .compaction import (
    DEFAULT_POLICY_COMPACTION_SYSTEM_PROMPT,
    Accounting,
    CompactionConfig,
    build_compaction_inputs,
    build_summary,
    check_compaction,
)
from .messages import (
    AssistantMessage,
    FrozenModel,
    Message,
    SystemMessage,
    ToolMessage,
    UserMessage,
    transcript_jsonl,
)
from .model import (
    ChatCompletionsModel,
    ModelBehaviorError,
    ModelCall,
    ModelRefusalError,
    ModelSettings,
    ProviderUsage,
    UsageTotals,
    summarize_usage,
)

logger = logging.getLogger(__name__)

StopReason = Literal["completed", "MaxTurnsExceeded", "ModelBehaviorError", "failed"]

# Feedback for the single recovery a production run allows, keyed by failure cause.
RECOVERY_FEEDBACK = {
    "length": (
        "Your previous response reached the output-token limit before it finished. "
        "If the work is complete, give the final answer now. Otherwise, keep your "
        "reasoning brief and continue with a smaller next step."
    ),
    "error": (
        "Your previous response was interrupted by a provider error before it finished. "
        "If the work is complete, give the final answer now. Otherwise, take the next "
        "step with a tool."
    ),
    "empty": (
        "Your previous response contained no answer or tool call. If the work is "
        "complete, give the final answer now. Otherwise, take the next step with a tool."
    ),
}
# Finish reasons that mean the provider stopped the response before it was complete.
_INTERRUPTED_FINISHES = frozenset({"length", "error", "content_filter"})
_COMPACTION_MAX_ATTEMPTS = 4


class RunRequest(FrozenModel):
    messages: tuple[Message, ...] = Field(min_length=1)
    max_turns: int = Field(default=100, ge=1)
    model_settings: ModelSettings = Field(default_factory=ModelSettings)


class TraceSegment(FrozenModel):
    kind: Literal["execution", "compaction"]
    input: tuple[Message, ...]
    messages: tuple[Message, ...]
    # One record per assistant message this segment's model calls produced, in order.
    calls: tuple[ProviderUsage, ...] = ()


class RunResult(FrozenModel):
    schema_version: Literal["feather.run.v1"] = "feather.run.v1"
    final_answer: str = ""
    stop_reason: StopReason
    stop_message: str | None = None
    num_turns: int
    compactions: int
    recoveries: int = 0
    segments: tuple[TraceSegment, ...]

    def execution_messages(self) -> tuple[Message, ...]:
        """Original conversation, without summary calls or repeated context."""
        return _execution_messages(self.segments)

    def usage(self) -> UsageTotals:
        """Provider-billed usage over every model call, summaries included."""
        return summarize_usage([call for segment in self.segments for call in segment.calls])


class AgentRunError(RuntimeError):
    def __init__(self, message: str, *, cause: Exception, result: RunResult):
        super().__init__(message)
        self.cause = cause
        self.result = result


@dataclass
class _RunState:
    context: list[Message]
    input_count: int
    kind: Literal["execution", "compaction"] = "execution"
    segments: list[TraceSegment] = field(default_factory=list)
    calls: list[ProviderUsage] = field(default_factory=list)
    accounting: Accounting = field(default_factory=Accounting)
    recoveries: int = 0

    def current_segment(self) -> TraceSegment:
        return TraceSegment(
            kind=self.kind,
            input=tuple(self.context[: self.input_count]),
            messages=tuple(self.context[self.input_count :]),
            calls=tuple(self.calls),
        )

    def append_call(self, call: ModelCall) -> None:
        self.context.append(call.assistant)
        if call.usage is not None:
            self.calls.append(call.usage)

    def start_segment(
        self, kind: Literal["execution", "compaction"], messages: list[Message]
    ) -> None:
        self.segments.append(self.current_segment())
        self.kind = kind
        self.context = messages
        self.input_count = len(messages)
        self.calls = []

    def result(
        self,
        *,
        final_answer: str = "",
        stop_reason: StopReason,
        stop_message: str | None = None,
    ) -> RunResult:
        return RunResult(
            final_answer=final_answer,
            stop_reason=stop_reason,
            stop_message=stop_message,
            num_turns=self.accounting.turns,
            compactions=self.accounting.compactions,
            recoveries=self.recoveries,
            segments=(*self.segments, self.current_segment()),
        )


class FeatherAgent:
    """Run an agent with its model and bound tools."""

    def __init__(
        self,
        *,
        model: ChatCompletionsModel,
        tools: tuple[Tool, ...] = (),
        compaction: CompactionConfig = CompactionConfig(),
        compaction_tool_state: Callable[[], tuple[str, ...]] | None = None,
        save_transcript: Callable[[str], Awaitable[None]] | None = None,
        response_recovery: bool = False,
        end_run_after_tools: (
            Callable[[AssistantMessage, tuple[ToolResult, ...]], bool] | None
        ) = None,
    ) -> None:
        self.model = model
        self.tools = {tool.spec.name: tool for tool in tools}
        if len(self.tools) != len(tools):
            raise ValueError("Duplicate tool names")
        self.compaction = compaction
        self.compaction_tool_state = compaction_tool_state
        self.save_transcript = save_transcript
        # Production policy: one recovery from a cut-off or empty response per run, and
        # no summary from an interrupted compaction call. Off keeps the original loop.
        self.response_recovery = response_recovery
        self.end_run_after_tools = end_run_after_tools

    async def run(self, request: RunRequest) -> RunResult:
        state = _RunState(context=list(request.messages), input_count=len(request.messages))
        try:
            return await self._run_loop(request, state)
        except AgentRunError:
            raise
        except Exception as exc:
            raise AgentRunError(
                f"agent run failed: {type(exc).__name__}: {exc}",
                cause=exc,
                result=state.result(stop_reason="failed", stop_message=str(exc)),
            ) from exc

    async def _run_loop(self, request: RunRequest, state: _RunState) -> RunResult:
        system_messages = [m for m in request.messages if isinstance(m, SystemMessage)]
        conversation = [m for m in request.messages if not isinstance(m, SystemMessage)]
        # Keep a lone task prompt verbatim after compaction; supplied history is summarized.
        prefix = conversation if len(conversation) == 1 else []
        specs = tuple(tool.spec for tool in self.tools.values())
        turn_tool_states: deque[tuple[str, ...]] = deque(maxlen=self.compaction.keep_recent_turns)
        while state.accounting.turns < request.max_turns:
            compaction_tokens = check_compaction(
                state.context, self.compaction, state.accounting, request.max_turns
            )
            if compaction_tokens is not None:
                try:
                    if self.save_transcript is not None:
                        messages = _execution_messages((*state.segments, state.current_segment()))
                        await self.save_transcript(transcript_jsonl(messages))
                    summary_input, recent = build_compaction_inputs(
                        state.context, prefix=prefix, config=self.compaction
                    )
                    recent_turns = sum(isinstance(message, AssistantMessage) for message in recent)
                    tool_state = ()
                    if self.compaction_tool_state is not None:
                        if recent_turns == 0:
                            tool_state = self.compaction_tool_state()
                        elif recent_turns <= len(turn_tool_states):
                            tool_state = turn_tool_states[-recent_turns]
                        # Supplied history has no pre-turn snapshots; never substitute live state.
                    # Signed and encrypted reasoning belongs to the conversation that produced it,
                    # so it does not cross into the summary request or the compacted context.
                    summary_input = _without_preserved_reasoning(summary_input)
                    recent = _without_preserved_reasoning(recent)
                    compaction_input = [
                        SystemMessage(content=DEFAULT_POLICY_COMPACTION_SYSTEM_PROMPT),
                        *summary_input,
                    ]
                    for attempt in range(1, _COMPACTION_MAX_ATTEMPTS + 1):
                        state.start_segment("compaction", list(compaction_input))
                        summary_call = await self.model.complete(
                            state.context,
                            settings=request.model_settings.model_copy(
                                update={"max_tokens": self.compaction.max_tokens}
                            ),
                            kind="compaction",
                        )
                        state.append_call(summary_call)
                        if self.response_recovery and (
                            summary_call.finish_reason == "content_filter"
                            or summary_call.assistant.refusal
                        ):
                            raise ValueError(
                                "compaction summary is incomplete "
                                f"(finish_reason={summary_call.finish_reason})"
                            )
                        try:
                            if self.response_recovery and summary_call.finish_reason in {
                                "length", "error"
                            }:
                                raise ValueError(
                                    "compaction summary is incomplete "
                                    f"(finish_reason={summary_call.finish_reason})"
                                )
                            summary = build_summary(summary_call.assistant.content, tool_state)
                            break
                        except ValueError as exc:
                            if attempt == _COMPACTION_MAX_ATTEMPTS:
                                raise
                            logger.warning(
                                "Compaction summary failed (attempt %d/%d); retrying: %s",
                                attempt, _COMPACTION_MAX_ATTEMPTS, exc,
                            )
                    context = [*system_messages, *prefix, summary, *recent]
                except Exception as exc:
                    raise AgentRunError(
                        f"agent context compaction failed: {type(exc).__name__}: {exc}",
                        cause=exc,
                        result=state.result(stop_reason="failed", stop_message=str(exc)),
                    ) from exc
                state.start_segment("execution", context)
                state.accounting.record_compaction()
            if (
                self.compaction.enabled
                and self.compaction.keep_recent_turns
                and self.compaction_tool_state
            ):
                turn_tool_states.append(self.compaction_tool_state())
            try:
                call = await self.model.complete(
                    state.context,
                    tools=specs,
                    settings=request.model_settings,
                )
            except ModelBehaviorError as exc:
                return state.result(stop_reason="ModelBehaviorError", stop_message=str(exc))
            state.append_call(call)
            context_tokens = call.input_tokens + call.output_tokens
            if self.compaction.exclude_unreplayed_reasoning:
                context_tokens -= call.unreplayed_reasoning_tokens
            state.accounting.record_turn(context_tokens, len(state.context))
            if self.response_recovery:
                cause = _failure_cause(call)
                if cause is not None:
                    reason = f"{cause} response (finish_reason={call.finish_reason})"
                    if call.assistant.tool_calls:
                        return state.result(
                            stop_reason="ModelBehaviorError",
                            stop_message=f"{reason}; none of its tool calls were executed",
                        )
                    if cause == "content_filter":
                        return state.result(stop_reason="ModelBehaviorError", stop_message=reason)
                    if state.recoveries:
                        return state.result(
                            stop_reason="ModelBehaviorError",
                            stop_message=f"{reason} after the run's one recovery",
                        )
                    if state.accounting.turns >= request.max_turns:
                        break
                    state.recoveries += 1
                    logger.warning("Recovering once from %s", reason)
                    state.context.append(UserMessage(content=RECOVERY_FEEDBACK[cause]))
                    continue
            results: list[ToolResult] = []
            for tool_call in call.assistant.tool_calls:
                tool = self.tools.get(tool_call.name)
                if tool is None:
                    content = f"Tool '{tool_call.name}' not found."
                    results.append(ToolResult(content=content, error="tool_error"))
                else:
                    output = await tool.invoke(tool_call.arguments)
                    content = output.content
                    results.append(output)
                state.context.append(ToolMessage(tool_call_id=tool_call.id, content=content))
            if (
                call.assistant.tool_calls
                and self.end_run_after_tools is not None
                and self.end_run_after_tools(call.assistant, tuple(results))
            ):
                return state.result(
                    stop_reason="completed", final_answer=call.assistant.content or ""
                )
            if not call.assistant.tool_calls:
                if call.assistant.refusal:
                    raise ModelRefusalError(
                        f"Model refused to produce output: {call.assistant.refusal}"
                    )
                return state.result(
                    stop_reason="completed", final_answer=call.assistant.content or ""
                )
        return state.result(
            stop_reason="MaxTurnsExceeded",
            stop_message=f"global max_turns ({request.max_turns}) exhausted",
        )


def _failure_cause(call: ModelCall) -> str | None:
    """Name a response the provider cut off, or one with nothing to act on; None otherwise."""
    tool_calls = call.assistant.tool_calls
    if call.assistant.refusal and not tool_calls:
        return None  # Refusals keep their existing handling and are never recovered.
    if call.finish_reason in _INTERRUPTED_FINISHES:
        return call.finish_reason
    content = call.assistant.content
    if not tool_calls and not (content and content.strip()):
        return "empty"
    return None


def _without_preserved_reasoning(messages: list[Message]) -> list[Message]:
    return [
        message.without_preserved_reasoning() if isinstance(message, AssistantMessage) else message
        for message in messages
    ]


def _execution_messages(segments: tuple[TraceSegment, ...]) -> tuple[Message, ...]:
    messages = list(segments[0].input) if segments else []
    for segment in segments:
        if segment.kind == "execution":
            messages.extend(segment.messages)
    return tuple(messages)
