"""GLM prompt counting with the upstream template, tokenizer and image geometry."""

import json
from copy import deepcopy
from functools import lru_cache
from threading import Lock
from typing import Literal

from pydantic import Field

from .messages import FrozenModel, tool_arguments_for_replay

_PROCESSOR_LOCK = Lock()


class PromptTokenBudget(FrozenModel):
    processor_name: Literal["zai-org/GLM-5.3-Flash"]
    processor_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    context_window_tokens: int = Field(gt=0)


class PromptTokenBudgetError(ValueError):
    """A request cannot fit or overrides its configured output budget."""


def count_prompt_tokens(
    budget: PromptTokenBudget,
    messages: list[dict],
    tools: list[dict],
    *,
    reasoning_effort: str | None = None,
    template_kwargs: dict | None = None,
) -> int:
    kwargs = {"reasoning_effort": reasoning_effort, **(template_kwargs or {})}
    if set(kwargs) - {"reasoning_effort", "clear_thinking"}:
        raise PromptTokenBudgetError("Unsupported GLM chat_template_kwargs for prompt counting")
    with _PROCESSOR_LOCK:
        tokenizer, image_processor = _load_processor(budget.processor_name, budget.processor_revision)
    conversation = deepcopy(messages)
    image_tokens = 0
    for message in conversation:
        for call in message.get("tool_calls", ()):
            arguments = call["function"]["arguments"]
            if isinstance(arguments, str):
                call["function"]["arguments"] = json.loads(tool_arguments_for_replay(arguments))
        content = message.get("content")
        if isinstance(content, list):
            for part in content:
                if part["type"] == "image_url":
                    from transformers.image_utils import load_image

                    image = load_image(part["image_url"]["url"], timeout=30)
                    patches = image_processor.get_number_of_image_patches(image.height, image.width)
                    # The template already contains one special image token per image.
                    image_tokens += patches // image_processor.merge_size**2 - 1
    tokens = tokenizer.apply_chat_template(
        conversation, tools=tools or None, tokenize=True, return_dict=False,
        add_generation_prompt=True, **kwargs
    )
    return len(tokens) + image_tokens


@lru_cache(maxsize=4)
def _load_processor(name: str, revision: str):
    from transformers import AutoTokenizer
    from transformers.models.glm5_next.image_processing_pil_glm5_next import (
        Glm5NextImageProcessorPil,
    )

    return (
        AutoTokenizer.from_pretrained(name, revision=revision),
        Glm5NextImageProcessorPil.from_pretrained(name, revision=revision),
    )
