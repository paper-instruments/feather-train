"""Feather's cumulative conversation format, independent of any agent SDK."""

from __future__ import annotations

import json
from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ToolCall(FrozenModel):
    id: str
    name: str
    arguments: str


class SystemMessage(FrozenModel):
    role: Literal["system"] = "system"
    content: str


class UserMessage(FrozenModel):
    role: Literal["user"] = "user"
    content: str


class AssistantMessage(FrozenModel):
    role: Literal["assistant"] = "assistant"
    content: str | None = None
    reasoning_content: str | None = None
    # OpenRouter reasoning blocks kept verbatim when signed or encrypted: the only form in which
    # Claude, GPT and Muse accept their own prior reasoning back into context.
    reasoning_details: tuple[dict[str, JsonValue], ...] = ()
    refusal: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()

    def without_preserved_reasoning(self) -> AssistantMessage:
        return self.model_copy(update={"reasoning_details": ()}) if self.reasoning_details else self


class TextBlock(TypedDict):
    type: Literal["text"]
    text: str


class ImageURL(TypedDict):
    url: Annotated[str, Field(pattern=r"^data:image/(png|jpeg|gif|webp);base64,[A-Za-z0-9+/=]+$")]


class ImageBlock(TypedDict):
    type: Literal["image_url"]
    image_url: ImageURL


ToolContent = str | list[TextBlock | ImageBlock]


class ToolMessage(FrozenModel):
    role: Literal["tool"] = "tool"
    tool_call_id: str
    content: ToolContent


Message = Annotated[
    SystemMessage | UserMessage | AssistantMessage | ToolMessage, Field(discriminator="role")
]
MESSAGES = TypeAdapter(tuple[Message, ...])


def transcript_jsonl(messages: tuple[Message, ...] | list[Message]) -> str:
    return "".join(message.model_dump_json() + "\n" for message in messages)


def tool_arguments_for_replay(arguments: str) -> str:
    """Keep failed calls visible while satisfying object-only serving templates."""
    arguments = arguments or "{}"
    try:
        if isinstance(json.loads(arguments, parse_constant=_reject_json_constant), dict):
            return arguments
    except ValueError:
        pass
    return json.dumps({"_invalid_tool_arguments": arguments})


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Invalid JSON constant: {value}")
