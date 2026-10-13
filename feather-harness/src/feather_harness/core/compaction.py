"""Policy compaction over complete Feather messages; no SDK hooks or items."""

from __future__ import annotations

import json
from dataclasses import dataclass

from pydantic import Field, model_validator

from .messages import (
    AssistantMessage,
    FrozenModel,
    Message,
    SystemMessage,
    ToolMessage,
    UserMessage,
)

DEFAULT_COMPACTION_PROMPT = """Summarize the conversation history into a compact continuation state that preserves everything needed to continue the work.

Preserve the task's explicit requirements and constraints, the current plan, and what is already done versus
still open. Keep the documents, files, and artifacts inspected or changed, with the exact paths, identifiers,
commands, and values that matter. Keep the decisions made and the reasoning behind them, the working
hypotheses still open, and any constraint or gotcha learned the hard way — including an approach that failed
and why it failed. State which conclusions are verified and which are still assumptions. Omit repetitive logs
and abandoned dead ends, but keep the lesson from a dead end when it would otherwise be repeated."""
DEFAULT_POLICY_COMPACTION_SYSTEM_PROMPT = (
    "Summarize the supplied conversation into a compact continuation state."
)
COMPACTION_CONTINUE_SUFFIX = "Continue the task from this compacted state."
SUMMARY_RETURN_INSTRUCTION = "Return only the compact continuation summary."


class CompactionConfig(FrozenModel):
    enabled: bool = False
    max_tokens: int = Field(default=65536, gt=0)
    context_window_tokens: int = Field(default=131072, gt=30000)
    min_turns_before_compaction: int = Field(default=0, ge=0)
    keep_recent_turns: int = Field(default=2, ge=0)
    # Off keeps the original estimate, which counts hidden reasoning as context and so
    # compacts early for models that think at length without returning their reasoning.
    exclude_unreplayed_reasoning: bool = False
    # Explicit trigger for models whose vendor harness compacts at a known point (e.g. Codex's
    # 90% of 272k for GPT Sol); None keeps the window-derived default.
    trigger_tokens: int | None = Field(default=None, gt=0)
    prompt: str = DEFAULT_COMPACTION_PROMPT

    @model_validator(mode="after")
    def _trigger_fits_window(self) -> CompactionConfig:
        if self.trigger_tokens is not None and self.trigger_tokens >= self.context_window_tokens:
            raise ValueError("trigger_tokens must be below context_window_tokens")
        return self

    @property
    def threshold_tokens(self) -> int:
        if self.trigger_tokens is not None:
            return self.trigger_tokens
        return min(self.context_window_tokens * 85 // 100, self.context_window_tokens - 30000)


@dataclass
class Accounting:
    turns: int = 0
    compactions: int = 0
    turns_since_compaction: int = 0
    measured_tokens: int = 0
    measured_message_count: int = 0

    def record_turn(self, tokens: int, context_message_count: int) -> None:
        self.turns += 1
        self.turns_since_compaction += 1
        if tokens:
            self.measured_tokens = tokens
            self.measured_message_count = context_message_count

    def record_compaction(self) -> None:
        self.compactions += 1
        self.turns_since_compaction = 0
        self.measured_tokens = 0
        self.measured_message_count = 0


def check_compaction(
    context: list[Message],
    config: CompactionConfig,
    accounting: Accounting,
    max_turns: int,
) -> int | None:
    if not config.enabled or max_turns - accounting.turns <= 1:
        return None
    if accounting.compactions and accounting.turns_since_compaction == 0:
        return None
    if accounting.turns_since_compaction < config.min_turns_before_compaction:
        return None

    measurement_covers_prefix = (
        accounting.measured_tokens > 0 and 0 < accounting.measured_message_count <= len(context)
    )
    if measurement_covers_prefix:
        unmeasured_messages = context[accounting.measured_message_count :]
        measured_tokens = accounting.measured_tokens
    else:
        unmeasured_messages = context
        measured_tokens = 0

    characters = 0
    image_count = 0
    for message in unmeasured_messages:
        if isinstance(message, SystemMessage):
            continue
        value = message.model_dump(mode="json")
        if isinstance(message, ToolMessage) and isinstance(message.content, list):
            image_count += sum(part["type"] == "image_url" for part in message.content)
            value["content"] = [part for part in message.content if part["type"] == "text"]
        characters += len(json.dumps(value, ensure_ascii=False))
    # Provider usage replaces this provisional estimate on the next completed request.
    # Base64 length is not a visual token count and would force repeated compactions.
    estimated_tokens = measured_tokens + (characters + 2) // 3 + image_count * 8192
    if estimated_tokens >= config.threshold_tokens:
        return estimated_tokens
    return None


def build_compaction_inputs(
    context: list[Message],
    *,
    prefix: list[Message],
    config: CompactionConfig,
) -> tuple[list[Message], list[Message]]:
    conversation = [message for message in context if not isinstance(message, SystemMessage)]
    remaining_messages = conversation[len(prefix) :]
    assistant_turn_starts = [
        index
        for index, message in enumerate(remaining_messages)
        if isinstance(message, AssistantMessage)
    ]
    recent_start = len(remaining_messages)
    if config.keep_recent_turns and assistant_turn_starts:
        turns_to_keep = min(config.keep_recent_turns, len(assistant_turn_starts))
        recent_start = assistant_turn_starts[-turns_to_keep]

    summary_input = [
        *prefix,
        *remaining_messages[:recent_start],
        UserMessage(content=f"{config.prompt}\n\n{SUMMARY_RETURN_INSTRUCTION}"),
    ]
    recent_messages = remaining_messages[recent_start:]
    return summary_input, recent_messages


def build_summary(text: str | None, tool_state: tuple[str, ...]) -> UserMessage:
    """Build the continuation summary with tool state at the compaction boundary."""
    summary = (text or "").strip()
    if not summary:
        raise ValueError("compaction summary was empty")
    heading = "## Summary of Previous Work\n\n" + summary
    parts = [text.strip() for text in tool_state if text.strip()]
    return UserMessage(content="\n\n---\n\n".join([heading, *parts, COMPACTION_CONTINUE_SUFFIX]))
