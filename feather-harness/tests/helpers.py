"""Scripted model and fixed work-tool setup used by loop tests."""

from types import SimpleNamespace

from feather_harness import AssistantMessage, RunRequest, Tool, ToolCall, ToolSpec, UserMessage
from feather_harness.core.model import ModelCall


class ScriptedModel:
    prompt_token_budget = None

    def __init__(self, *messages, tokens=10):
        self.messages = iter(messages)
        self.tokens = tokens
        self.calls = []

    async def complete(self, context, *, settings, tools=(), kind="agent"):
        self.calls.append(
            SimpleNamespace(
                kind=kind,
                settings=settings,
                request={"messages": [message.model_dump(mode="json") for message in context]},
            )
        )
        return ModelCall(
            assistant=next(self.messages),
            input_tokens=self.tokens,
            output_tokens=3,
        )


def tool(invoke):
    return Tool(spec=ToolSpec(name="work", description="work", parameters={}), invoke=invoke)


def request(*, max_turns=100):
    return RunRequest(messages=(UserMessage(content="task"),), max_turns=max_turns)


def work_message(*, refusal=None):
    return AssistantMessage(
        tool_calls=(ToolCall(id="k", name="work", arguments="{}"),),
        refusal=refusal,
    )
