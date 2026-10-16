"""Convert Feather's native trace to Harbor ATIF, without changing the conversation."""

from __future__ import annotations

import argparse
import json
from importlib.metadata import version
from pathlib import Path

from feather_harness import ProviderUsage, RunResult, UsageTotals
from feather_harness.core.messages import AssistantMessage, Message, ToolMessage
from harbor.models.trajectories.agent import Agent
from harbor.models.trajectories.final_metrics import FinalMetrics
from harbor.models.trajectories.metrics import Metrics
from harbor.models.trajectories.observation import Observation
from harbor.models.trajectories.observation_result import ObservationResult
from harbor.models.trajectories.step import Step
from harbor.models.trajectories.subagent_trajectory_ref import SubagentTrajectoryRef
from harbor.models.trajectories.tool_call import ToolCall
from harbor.models.trajectories.trajectory import Trajectory


def to_atif(
    result: RunResult,
    *,
    agent: Agent,
    session_id: str,
    grader: bool = False,
) -> dict[str, Trajectory]:
    """Return ATIF files: context segments for Harbor, or one execution transcript for grading.

    Graders such as Gandalf read one file and select its last agent answer. Their view
    deliberately excludes summaries and copied context; it is not a model-input export.
    """
    if grader:
        return {
            "trajectory.json": Trajectory(
                session_id=session_id,
                agent=agent,
                steps=_steps(result.execution_messages()),
            )
        }
    trajectories: dict[str, Trajectory] = {}
    previous_execution: Trajectory | None = None
    execution_count = 0
    compaction_count = 0
    for index, segment in enumerate(result.segments):
        if segment.kind == "execution":
            name = (
                "trajectory.json"
                if execution_count == 0
                else f"trajectory.cont-{execution_count}.json"
            )
            execution_count += 1
        else:
            compaction_count += 1
            name = f"trajectory.summarization-{compaction_count}.json"
        steps = _steps(segment.input + segment.messages, copied_inputs=len(segment.input))
        _attach_usage(steps, segment.calls)
        if index == 0:
            # Supplied assistant history remains context, not new training targets.
            for step in steps:
                if step.source != "agent":
                    step.is_copied_context = None
        trajectory = Trajectory(
            session_id=session_id,
            agent=agent,
            steps=steps,
            extra={"segment_kind": segment.kind},
        )
        trajectories[name] = trajectory
        if segment.kind == "execution":
            if previous_execution is not None:
                previous_execution.continued_trajectory_ref = name
            previous_execution = trajectory
        elif previous_execution is not None:
            # Reference the actual summary request, including an unfinished/failed request.
            # No content is invented for the reference itself.
            step = previous_execution.steps[-1]
            if step.observation is None:
                step.observation = Observation(results=[])
            step.observation.results.append(
                ObservationResult(
                    subagent_trajectory_ref=[
                        SubagentTrajectoryRef(trajectory_path=name),
                    ]
                )
            )
    if trajectories:
        root = trajectories["trajectory.json"]
        root.extra.update(stop_reason=result.stop_reason, stop_message=result.stop_message)
        root.final_metrics = _final_metrics(result.usage())
    return trajectories


def write_atif(
    result: RunResult,
    directory: Path,
    *,
    agent: Agent,
    session_id: str,
    grader: bool = False,
) -> None:
    """Write validated ATIF documents with relative continuation and summary references."""
    trajectories = to_atif(result, agent=agent, session_id=session_id, grader=grader)
    directory.mkdir(parents=True, exist_ok=True)
    for name, trajectory in trajectories.items():
        (directory / name).write_text(trajectory.model_dump_json(exclude_none=True, indent=2))


def _steps(messages: tuple[Message, ...], *, copied_inputs: int = 0) -> list[Step]:
    steps: list[Step] = []
    call_steps: dict[str, Step] = {}
    for index, message in enumerate(messages):
        if isinstance(message, ToolMessage):
            # Results belong to their call's step, even when calls finish out of order.
            step = call_steps[message.tool_call_id]
            if step.observation is None:
                step.observation = Observation(results=[])
            step.observation.results.append(
                ObservationResult(
                    source_call_id=message.tool_call_id,
                    content=_observation_content(message.content),
                )
            )
            continue
        copied = True if index < copied_inputs else None
        if isinstance(message, AssistantMessage):
            calls = []
            for call in message.tool_calls:
                try:
                    arguments = json.loads(call.arguments)
                except json.JSONDecodeError:
                    arguments = None
                # ATIF requires an object; preserve malformed/non-object requests losslessly.
                extra = None
                if not isinstance(arguments, dict):
                    arguments = {}
                    extra = {"raw_arguments": call.arguments}
                calls.append(
                    ToolCall(
                        tool_call_id=call.id,
                        function_name=call.name,
                        arguments=arguments,
                        extra=extra,
                    )
                )
            step = Step(
                step_id=len(steps) + 1,
                source="agent",
                message=message.content or "",
                reasoning_content=message.reasoning_content,
                tool_calls=calls or None,
                is_copied_context=copied,
                extra={"refusal": message.refusal} if message.refusal is not None else None,
            )
            for call in message.tool_calls:
                call_steps[call.id] = step
        else:
            step = Step(
                step_id=len(steps) + 1,
                source=message.role,
                message=message.content,
                is_copied_context=copied,
            )
        steps.append(step)
    return steps


def _attach_usage(steps: list[Step], calls: tuple[ProviderUsage, ...]) -> None:
    """Give each model-produced agent step its own call's provider-billed usage."""
    produced = [step for step in steps if step.source == "agent" and not step.is_copied_context]
    if not calls:
        return
    if len(calls) != len(produced):
        raise ValueError(
            f"segment has {len(calls)} model calls for {len(produced)} produced agent steps"
        )
    for step, call in zip(produced, calls, strict=True):
        step.llm_call_count = 1
        step.metrics = Metrics(
            prompt_tokens=call.prompt_tokens,
            completion_tokens=call.completion_tokens,
            cached_tokens=call.cached_tokens,
            cost_usd=call.cost_usd,
            extra=_present(
                reasoning_tokens=call.reasoning_tokens,
                cache_write_tokens=call.cache_write_tokens,
                is_byok=call.is_byok,
                openrouter_cost_usd=call.openrouter_cost_usd,
                # Non-BYOK `cost` already includes the upstream charge.
                upstream_cost_usd=call.upstream_cost_usd if call.is_byok else None,
                provider=call.provider,
                response_id=call.response_id,
                kind=call.kind,
            ),
        )


def _final_metrics(totals: UsageTotals) -> FinalMetrics | None:
    if not totals.model_calls:
        return None
    return FinalMetrics(
        total_prompt_tokens=totals.prompt_tokens,
        total_completion_tokens=totals.completion_tokens,
        total_cached_tokens=totals.cached_tokens,
        total_cost_usd=totals.cost_usd,
        extra=totals.model_dump(
            mode="json",
            exclude_none=True,
            exclude={"prompt_tokens", "completion_tokens", "cached_tokens", "cost_usd"},
        ),
    )


def _present(**values):
    return {key: value for key, value in values.items() if value is not None} or None


def _observation_content(content):
    if isinstance(content, str):
        return content
    return [
        part if part["type"] == "text" else {
            "type": "image",
            "source": {
                "media_type": part["image_url"]["url"].split(";", 1)[0].removeprefix("data:"),
                "path": part["image_url"]["url"],
            },
        }
        for part in content
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--version", action="version", version=f"feather-atif {version('feather-eval')}"
    )
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model")
    parser.add_argument("--session-id")
    parser.add_argument(
        "--grader", action="store_true", help="Write one chronological execution transcript"
    )
    args = parser.parse_args()
    document = json.loads(args.trace.read_text())
    agent_name = "feather"
    model = args.model
    session_id = args.session_id
    if document.get("schema_version") == "tldc.feather-agent-trace.v2":
        agent_name = document["agent"]["name"]
        model = model or document["agent"]["model"]
        session_id = session_id or document["task_id"]
        document = document["trace"]
    if document.get("schema_version") != "feather.run.v1":
        parser.error("Expected feather.run.v1 or tldc.feather-agent-trace.v2")
    if session_id is None:
        parser.error("--session-id is required for a raw Feather run")
    write_atif(
        RunResult.model_validate(document),
        args.output_dir,
        agent=Agent(name=agent_name, version=version("feather-harness"), model_name=model),
        session_id=session_id,
        grader=args.grader,
    )


if __name__ == "__main__":
    main()
