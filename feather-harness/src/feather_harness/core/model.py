"""Chat Completions using the caller's client, optionally streamed and retried."""

from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import Sequence
from typing import Literal

import httpx
from openai import APIConnectionError, APIError, APIStatusError
from openai.types.chat import ChatCompletion
from pydantic import Field, JsonValue

from ..tools import ToolSpec
from .messages import (
    AssistantMessage,
    FrozenModel,
    Message,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
    tool_arguments_for_replay,
)
from .token_budget import PromptTokenBudget, PromptTokenBudgetError, count_prompt_tokens

logger = logging.getLogger(__name__)


class ModelBehaviorError(ValueError):
    pass


class ModelRefusalError(RuntimeError):
    pass


class ModelSettings(FrozenModel):
    temperature: float | None = None
    top_p: float | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    max_tokens: int | None = Field(default=None, gt=0)
    parallel_tool_calls: bool | None = None
    reasoning_effort: str | None = None
    top_logprobs: int | None = None
    extra_body: dict[str, JsonValue] = Field(default_factory=dict)


class ProviderUsage(FrozenModel):
    """Provider-reported accounting for one completed model call, kept verbatim.

    `usage` is the response's raw usage object (empty when the provider sent none), so totals
    can be re-derived if a provider's field names change. Prompt tokens include cache reads and
    completion tokens include reasoning, as billed.
    """

    kind: Literal["agent", "compaction"]
    response_id: str | None = None
    provider: str | None = None
    usage: dict[str, JsonValue] = Field(default_factory=dict)

    @property
    def prompt_tokens(self) -> int | None:
        return _count(self.usage, "prompt_tokens")

    @property
    def completion_tokens(self) -> int | None:
        return _count(self.usage, "completion_tokens")

    @property
    def cached_tokens(self) -> int | None:
        return _count(self.usage, "prompt_tokens_details", "cached_tokens")

    @property
    def cache_write_tokens(self) -> int | None:
        return _count(self.usage, "prompt_tokens_details", "cache_write_tokens")

    @property
    def reasoning_tokens(self) -> int | None:
        return _count(self.usage, "completion_tokens_details", "reasoning_tokens")

    @property
    def is_byok(self) -> bool | None:
        value = self.usage.get("is_byok")
        return value if isinstance(value, bool) else None

    @property
    def openrouter_cost_usd(self) -> float | None:
        return _amount(self.usage, "cost")

    @property
    def upstream_cost_usd(self) -> float | None:
        return _amount(self.usage, "cost_details", "upstream_inference_cost")

    def accounting_row(self) -> dict[str, JsonValue]:
        """One call's identity and billed totals, without the raw usage payload."""
        row: dict[str, JsonValue] = {
            "kind": self.kind,
            "response_id": self.response_id,
            "provider": self.provider,
            "prompt_tokens": self.prompt_tokens,
            "cached_tokens": self.cached_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "completion_tokens": self.completion_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "is_byok": self.is_byok,
            "openrouter_cost_usd": self.openrouter_cost_usd,
            "upstream_cost_usd": self.upstream_cost_usd if self.is_byok else None,
            "cost_usd": self.cost_usd,
        }
        return {key: value for key, value in row.items() if value is not None}

    @property
    def cost_usd(self) -> float | None:
        """Total spend: OpenRouter's charge plus, for BYOK calls, the upstream provider's."""
        charged = self.openrouter_cost_usd
        upstream = self.upstream_cost_usd if self.is_byok else None
        if charged is None and upstream is None:
            return None
        return (charged or 0.0) + (upstream or 0.0)


class UsageTotals(FrozenModel):
    """Sums over model calls; a total is None when no call reported that field."""

    model_calls: int
    calls_without_usage: int
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cached_tokens: int | None = None
    cache_write_tokens: int | None = None
    reasoning_tokens: int | None = None
    cost_usd: float | None = None
    openrouter_cost_usd: float | None = None
    upstream_cost_usd: float | None = None
    byok_calls: int = 0
    providers: tuple[str, ...] = ()


def summarize_usage(calls: Sequence[ProviderUsage]) -> UsageTotals:
    def total(values):
        reported = [value for value in values if value is not None]
        return sum(reported) if reported else None

    return UsageTotals(
        model_calls=len(calls),
        calls_without_usage=sum(not call.usage for call in calls),
        prompt_tokens=total(call.prompt_tokens for call in calls),
        completion_tokens=total(call.completion_tokens for call in calls),
        cached_tokens=total(call.cached_tokens for call in calls),
        cache_write_tokens=total(call.cache_write_tokens for call in calls),
        reasoning_tokens=total(call.reasoning_tokens for call in calls),
        cost_usd=total(call.cost_usd for call in calls),
        openrouter_cost_usd=total(call.openrouter_cost_usd for call in calls),
        upstream_cost_usd=total(
            call.upstream_cost_usd for call in calls if call.is_byok
        ),
        byok_calls=sum(call.is_byok is True for call in calls),
        providers=tuple(sorted({call.provider for call in calls if call.provider})),
    )


class ModelCall(FrozenModel):
    assistant: AssistantMessage
    input_tokens: int
    output_tokens: int
    finish_reason: str | None = None
    usage: ProviderUsage | None = None
    # Reasoning billed as output but absent from the next request's context.
    unreplayed_reasoning_tokens: int = 0


class RetryPolicy(FrozenModel):
    """Retries for transient request failures; defaults match the Fireworks training sampler."""

    max_attempts: int = Field(default=7, ge=1)
    base_backoff_seconds: float = Field(default=2.0, ge=0)
    max_backoff_seconds: float = Field(default=30.0, ge=0)


class TruncatedStreamError(RuntimeError):
    """The provider closed a stream before sending a finish reason."""


_RETRYABLE_STATUS_CODES = frozenset({408, 429, 500, 502, 503, 504})
_NON_RETRYABLE_STREAM_ERRORS = frozenset({
    "invalid_request_error", "authentication_error", "permission_error", "permission_denied",
    "not_found_error", "model_not_found",
    "insufficient_quota", "context_length_exceeded", "content_filter", "content_policy_violation",
})


def _is_retryable_error(
    exc: APIError | httpx.TransportError | TruncatedStreamError, *, stream: bool,
) -> bool:
    if isinstance(exc, APIStatusError):
        return exc.status_code in _RETRYABLE_STATUS_CODES
    if isinstance(exc, (APIConnectionError, httpx.TransportError, TruncatedStreamError)):
        return True
    if not stream or type(exc) is not APIError:
        return False
    # The SDK raises a bare APIError for SSE error events after HTTP 200. These
    # incomplete responses are safe to discard: no partial tool calls were executed.
    code, error_type = str(exc.code), str(exc.type)
    if {code, error_type} & _NON_RETRYABLE_STREAM_ERRORS:
        return False
    if code.isdigit():
        return int(code) in _RETRYABLE_STATUS_CODES
    return True


class ChatCompletionsModel:
    def __init__(
        self, *, client, model: str, replay_reasoning: bool = True,
        prompt_token_budget: PromptTokenBudget | None = None,
        stream: bool = False, retry: RetryPolicy | None = None,
    ):
        self.client = client
        self.model = model
        self.replay_reasoning = replay_reasoning
        self.prompt_token_budget = prompt_token_budget
        # Streaming turns the client's request timeout into an idle timeout, so long
        # generations that keep producing tokens are not cut off and resent unchanged.
        self.stream = stream
        self.retry = retry

    async def complete(
        self,
        messages: list[Message],
        *,
        tools: tuple[ToolSpec, ...] = (),
        settings: ModelSettings,
        kind: Literal["agent", "compaction"] = "agent",
    ) -> ModelCall:
        wire = messages_to_wire(messages, replay_reasoning=self.replay_reasoning)
        options = settings.model_dump(exclude_none=True)
        if options.get("parallel_tool_calls") is True and not tools:
            options.pop("parallel_tool_calls")
        if "top_logprobs" in options:
            options["logprobs"] = True
        extra = options["extra_body"]
        if set(extra) & {"messages", "model", "tools", "stream", "tool_choice"}:
            raise ValueError("extra_body cannot override the conversation or execution contract")
        request = {"model": self.model, "messages": wire, **options}
        if tools:
            request["tools"] = [tool.to_wire() for tool in tools]
        prompt_tokens = None
        if self.prompt_token_budget is not None:
            if set(extra) & {"max_tokens", "max_completion_tokens", "chat_template"}:
                raise PromptTokenBudgetError("extra_body cannot override the prompt token budget")
            reasoning = extra.get("reasoning")
            effort = extra.get("reasoning_effort", settings.reasoning_effort)
            if effort is None and isinstance(reasoning, dict):
                effort = reasoning.get("effort")
            prompt_tokens = await asyncio.to_thread(
                count_prompt_tokens,
                self.prompt_token_budget,
                wire,
                request.get("tools", []),
                reasoning_effort=effort,
                template_kwargs=extra.get("chat_template_kwargs"),
            )
            remaining = self.prompt_token_budget.context_window_tokens - prompt_tokens
            if remaining <= 0:
                raise PromptTokenBudgetError(
                    f"Prompt uses {prompt_tokens} tokens; no output room remains in "
                    f"the {self.prompt_token_budget.context_window_tokens}-token context"
                )
            requested = settings.max_tokens if settings.max_tokens is not None else remaining
            request["max_tokens"] = min(requested, remaining)
        response = await self._request(request)
        if not response.choices:
            provider_error = getattr(response, "error", None)
            details = f": {provider_error}" if provider_error is not None else ""
            raise RuntimeError(
                f"ChatCompletion response has no choices (possible provider error payload){details}"
            )
        choice = response.choices[0]
        message = choice.message
        raw = message.model_dump(mode="json", exclude_unset=True)
        if raw.get("thinking_blocks"):
            raise ValueError("Signed thinking-block replay is not currently supported")
        if raw.get("audio"):
            raise ValueError("Audio is not currently supported")
        calls = []
        for call in message.tool_calls or ():
            if call.type != "function":
                raise ModelBehaviorError(f"Unsupported tool call type: {call.type}")
            calls.append(
                ToolCall(
                    id=call.id,
                    name=call.function.name,
                    arguments=call.function.arguments,
                )
            )
        assistant = AssistantMessage(
            content=message.content,
            reasoning_content=raw.get("reasoning_content") or raw.get("reasoning"),
            reasoning_details=_preserved_reasoning(raw.get("reasoning_details")),
            refusal=message.refusal,
            tool_calls=tuple(calls),
        )
        input_tokens = response.usage.prompt_tokens if response.usage else 0
        if prompt_tokens is not None and response.usage and input_tokens != prompt_tokens:
            logger.warning(
                "Prompt token count differs for %s (%s): local=%d provider=%d",
                self.model, kind, prompt_tokens, input_tokens,
            )
        output_tokens = response.usage.completion_tokens if response.usage else 0
        usage = ProviderUsage(
            kind=kind,
            response_id=response.id or None,
            provider=(response.model_extra or {}).get("provider"),
            usage=response.usage.model_dump(mode="json", exclude_none=True) if response.usage else {},
        )
        return ModelCall(
            assistant=assistant,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            finish_reason=getattr(choice, "finish_reason", None),
            usage=usage,
            unreplayed_reasoning_tokens=self._unreplayed_reasoning_tokens(assistant, usage),
        )

    def _unreplayed_reasoning_tokens(
        self, assistant: AssistantMessage, usage: ProviderUsage
    ) -> int:
        if self.replay_reasoning and assistant.reasoning_details:
            return 0
        reasoning = usage.reasoning_tokens or 0
        if self.replay_reasoning and assistant.reasoning_content:
            # Same characters-per-token estimate compaction uses for unmeasured context.
            reasoning -= (len(assistant.reasoning_content) + 2) // 3
        return max(reasoning, 0)

    async def _request(self, request: dict) -> ChatCompletion:
        if self.retry is None:
            return await self._send(request)
        attempt, backoff = 1, self.retry.base_backoff_seconds
        while True:
            try:
                return await self._send(request)
            except (APIError, httpx.TransportError, TruncatedStreamError) as exc:
                if attempt == self.retry.max_attempts or not _is_retryable_error(
                    exc, stream=self.stream,
                ):
                    raise
                delay = backoff * (0.5 + random.random())
                retry_after = _retry_after_seconds(exc)
                if retry_after is not None:
                    delay = max(delay, min(retry_after, self.retry.max_backoff_seconds))
                logger.warning(
                    "Transient model request failure (attempt %d/%d); retrying in %.1fs: %s",
                    attempt, self.retry.max_attempts, delay, exc,
                )
                await asyncio.sleep(delay)
                attempt, backoff = attempt + 1, min(backoff * 2, self.retry.max_backoff_seconds)

    async def _send(self, request: dict) -> ChatCompletion:
        if not self.stream:
            return await self.client.chat.completions.create(**request)
        stream = await self.client.chat.completions.create(
            **request, stream=True, stream_options={"include_usage": True}
        )
        return await _collect_stream(stream)


async def _collect_stream(stream) -> ChatCompletion:
    """Assemble streamed chunks into the nonstreaming response shape."""
    content, reasoning, refusal = [], [], []
    reasoning_details: dict[int, dict] = {}
    calls: dict[int, dict] = {}
    finish_reason = usage = provider = None
    response_id, created, model = "", 0, ""
    async for chunk in stream:
        response_id, created, model = chunk.id, chunk.created, chunk.model
        usage = chunk.usage or usage
        provider = (chunk.model_extra or {}).get("provider") or provider
        if not chunk.choices:
            continue
        choice = chunk.choices[0]
        delta = choice.delta.model_dump(mode="json", exclude_none=True)
        content.append(delta.get("content") or "")
        reasoning.append(delta.get("reasoning_content") or delta.get("reasoning") or "")
        refusal.append(delta.get("refusal") or "")
        for part in delta.get("reasoning_details") or ():
            _merge_reasoning_fragment(reasoning_details, part)
        for part in delta.get("tool_calls") or ():
            call = calls.setdefault(
                part["index"], {"id": "", "type": "function", "function": {"name": "", "arguments": ""}}
            )
            call["id"] = part.get("id") or call["id"]
            call["type"] = part.get("type") or call["type"]
            function = part.get("function") or {}
            call["function"]["name"] += function.get("name") or ""
            call["function"]["arguments"] += function.get("arguments") or ""
        finish_reason = choice.finish_reason or finish_reason
    if finish_reason is None:
        raise TruncatedStreamError("model stream ended without a finish reason")
    # OpenRouter ends a stream with finish_reason "error" when the upstream provider fails
    # mid-response. The partial response is unusable and the SDK rejects the value, so retry.
    if finish_reason == "error":
        raise TruncatedStreamError("model stream ended with a provider error")
    message = {
        "role": "assistant",
        "content": "".join(content) or None,
        "refusal": "".join(refusal) or None,
        "tool_calls": [calls[index] for index in sorted(calls)] or None,
    }
    if any(reasoning):
        message["reasoning_content"] = "".join(reasoning)
    if reasoning_details:
        message["reasoning_details"] = [reasoning_details[i] for i in sorted(reasoning_details)]
    return ChatCompletion.model_validate(
        {
            "id": response_id,
            "object": "chat.completion",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
            "usage": usage.model_dump() if usage else None,
            **({"provider": provider} if provider else {}),
        }
    )


def _preserved_reasoning(details: JsonValue) -> tuple[dict[str, JsonValue], ...]:
    """Keep reasoning blocks only when plaintext replay would lose them (signed or encrypted)."""
    if not isinstance(details, list):
        return ()
    blocks = tuple(block for block in details if isinstance(block, dict))
    if any(block.get("signature") or block.get("type") == "reasoning.encrypted" for block in blocks):
        return blocks
    return ()


def _merge_reasoning_fragment(blocks: dict[int, dict], part: dict) -> None:
    """Rebuild streamed reasoning blocks: text and summary arrive in pieces, the rest whole."""
    block = blocks.setdefault(part.get("index", len(blocks)), {})
    for key, value in part.items():
        if key in ("text", "summary") and isinstance(value, str):
            block[key] = block.get(key, "") + value
        elif value is not None:
            block[key] = value


def _count(usage: dict[str, JsonValue], *path: str) -> int | None:
    value = _lookup(usage, path)
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _amount(usage: dict[str, JsonValue], *path: str) -> float | None:
    value = _lookup(usage, path)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _lookup(usage: dict[str, JsonValue], path: tuple[str, ...]) -> JsonValue:
    value: JsonValue = usage
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _retry_after_seconds(exc: Exception) -> float | None:
    if not isinstance(exc, APIStatusError):
        return None
    try:
        return float(exc.response.headers["retry-after"])
    except (KeyError, ValueError):
        return None


def messages_to_wire(messages: Sequence[Message], *, replay_reasoning: bool = True) -> list[dict]:
    output = []
    images = []
    for message in messages:
        # All parallel results must precede the image-bearing user message.
        if images and not isinstance(message, ToolMessage):
            output.append({"role": "user", "content": images})
            images = []
        if isinstance(message, SystemMessage):
            output.append({"role": "system", "content": message.content})
        elif isinstance(message, UserMessage):
            output.append({"role": "user", "content": message.content})
        elif isinstance(message, ToolMessage):
            content = message.content
            if isinstance(content, list) and any(part["type"] == "image_url" for part in content):
                images.append({"type": "text", "text": f"Images returned by tool call {message.tool_call_id}:"})
                images.extend(part for part in content if part["type"] == "image_url")
                content = [part for part in content if part["type"] == "text"]
                content.append({"type": "text", "text": "Tool images are attached in the following user message."})
            output.append(
                {
                    "role": "tool",
                    "tool_call_id": message.tool_call_id,
                    "content": content,
                }
            )
        elif isinstance(message, AssistantMessage):
            if not (message.content or message.tool_calls or message.refusal):
                continue  # Reference converter does not replay standalone reasoning/empty messages.
            wire = {"role": "assistant"}
            if message.content:
                wire["content"] = message.content
            if message.tool_calls:
                if not message.refusal:
                    wire.setdefault("content", None)
                wire["tool_calls"] = [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.name,
                            "arguments": tool_arguments_for_replay(call.arguments),
                        },
                    }
                    for call in message.tool_calls
                ]
            if message.refusal:
                wire["refusal"] = message.refusal
            if replay_reasoning and message.reasoning_details:
                wire["reasoning_details"] = list(message.reasoning_details)
            elif replay_reasoning and message.reasoning_content:
                wire["reasoning_content"] = message.reasoning_content
            output.append(wire)
        else:
            raise TypeError(f"Unsupported message: {type(message).__name__}")
    if images:
        output.append({"role": "user", "content": images})
    return output
