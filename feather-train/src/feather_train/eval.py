"""Evaluate Feather on baked PI tasks: python -m feather_train.eval config.json."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import uuid
from pathlib import Path
from statistics import mean

import httpx
import verifiers.v1 as vf
from feather_harness.core.compaction import CompactionConfig
from feather_harness.production import Profile
from pydantic import BaseModel, ConfigDict, Field, FilePath
from verifiers.v1.clients import EvalClient
from verifiers.v1.errors import model_error

from feather_train.environment import load_environment
from feather_train.errors import NonRetryableHarnessError
from feather_train.grading.grader import GradingConfig
from feather_train.tasks.taskset import FeatherTask, ImmutableRevision

logger = logging.getLogger(__name__)


class EvalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hf_dataset: str
    hf_revision: ImmutableRevision
    task_ids: list[str] | None = None
    max_tasks: int | None = Field(1, ge=1)
    model: str
    base_url: str
    api_key_env: str = "OPENAI_API_KEY"
    sampling: vf.Sampling = Field(default_factory=vf.Sampling)
    concurrency: int = Field(1, ge=1)
    model_request_timeout_seconds: float = Field(1200, gt=0)
    rollouts_per_example: int = Field(1, ge=1)
    runtime_archive: FilePath | None = None
    output_dir: Path
    braintrust_project: str
    braintrust_experiment: str | None = None
    enable_image_tools: bool = False
    capture_deliverables: bool = False
    capture_paths: tuple[str, ...] = ()
    profile: Profile = "train"
    additional_instructions: str | None = None
    max_turns: int = Field(250, ge=1)
    sandbox_timeout_minutes: int = Field(180, gt=0)
    rollout_max_retries: int = Field(2, ge=0)
    compaction: CompactionConfig = Field(default_factory=lambda: CompactionConfig(enabled=True))
    grading: GradingConfig = Field(default_factory=GradingConfig)



class EvalModelClient(EvalClient):
    """Size the provider connection pool and bound inference request duration."""

    def __init__(self, config: EvalConfig, api_key: str) -> None:
        self.base_url = config.base_url.rstrip("/")
        self.api_key = api_key
        self.headers = {}
        self.request_timeout = config.model_request_timeout_seconds
        self.http = httpx.AsyncClient(
            timeout=httpx.Timeout(self.request_timeout, connect=60, pool=60, write=120),
            limits=httpx.Limits(
                max_connections=config.concurrency,
                max_keepalive_connections=config.concurrency,
            ),
        )

    async def _request(
        self, url: str, body: dict, headers: httpx.Headers, *, stream: bool = False
    ) -> httpx.Response:
        # A rollout timeout need not cancel the shared interception handler.
        try:
            async with asyncio.timeout(self.request_timeout):
                return await super()._request(url, body, headers, stream=stream)
        except TimeoutError as exc:
            raise model_error(
                f"Upstream total request timeout: {self.request_timeout:g}s", status_code=504
            ) from exc


async def evaluate(config: EvalConfig) -> dict:
    """Save each finished trace; fail the run on a nonretryable harness error."""
    api_key = os.environ[config.api_key_env]
    run_id = uuid.uuid4().hex
    experiment = config.braintrust_experiment or f"feather-eval-{run_id}"
    manifest = config.model_dump(mode="json") | {
        "run_id": run_id,
        "braintrust_experiment": experiment,
    }
    env = load_environment(
        hf_dataset=config.hf_dataset,
        hf_revision=config.hf_revision,
        task_ids=config.task_ids,
        max_tasks=config.max_tasks,
        max_turns=config.max_turns,
        enable_image_tools=config.enable_image_tools,
        capture_deliverables=config.capture_deliverables,
        capture_paths=config.capture_paths,
        capture_output_dir=str(config.output_dir / "deliverables"),
        profile=config.profile,
        additional_instructions=config.additional_instructions,
        sandbox_timeout_minutes=config.sandbox_timeout_minutes,
        generation_timeout_seconds=config.model_request_timeout_seconds,
        rollout_max_retries=config.rollout_max_retries,
        compaction=config.compaction,
        grading=config.grading,
        runtime_archive=config.runtime_archive,
        bt_project=config.braintrust_project,
        bt_experiment_name=experiment,
        bt_experiment_metadata={"run_manifest": manifest},
    )
    tasks = await asyncio.to_thread(env.taskset.select)
    if not tasks:
        raise ValueError("No tasks selected")
    config.output_dir.mkdir(parents=True, exist_ok=False)
    traces_dir = config.output_dir / "traces"
    traces_dir.mkdir()
    _write_json(config.output_dir / "config.json", json.dumps(manifest, indent=2))
    total = len(tasks) * config.rollouts_per_example
    results: list[dict] = []
    status = "interrupted"
    client = EvalModelClient(config, api_key)
    ctx = vf.ModelContext(model=config.model, client=client, sampling=config.sampling)

    async def save(trace: vf.Trace) -> None:
        await asyncio.to_thread(
            _write_json, traces_dir / f"{trace.id}.json", trace.model_dump_json()
        )
        result = {
            "trace_id": trace.id,
            "task_id": trace.task.data.task_id,
            "rollout_group_id": trace.task.data.rollout_group_id,
            "score": trace.info.get("grading_score") if trace.error is None else None,
            "error": trace.error.model_dump() if trace.error else None,
            "stop_condition": trace.stop_condition,
            "braintrust_url": trace.info.get("bt_span_permalink"),
        }
        results.append(result)
        logger.info(
            "[%d/%d] %s score=%s error=%s %s",
            len(results),
            total,
            result["task_id"],
            result["score"],
            result["error"],
            result["braintrust_url"] or "",
        )
        if trace.error and trace.error.type == "NonRetryableHarnessError":
            raise NonRetryableHarnessError(trace.error.message)

    for idx, task in enumerate(tasks):
        group_id = task.data.rollout_group_id
        if group_id is None:
            group_id = uuid.uuid4().hex
        # Eval scores each rollout independently; group length shaping is training-only.
        tasks[idx] = FeatherTask(
            task.data.model_copy(
                update={
                    "eval_run_id": run_id,
                    "rollout_group_id": group_id,
                }
            ),
            task.config,
        )
    pending = (task for task in tasks for _ in range(config.rollouts_per_example))

    async def worker() -> None:
        for task in pending:
            episode = env.episode(task, ctx)
            try:
                await episode.run(on_complete=save)
            except BaseException as exc:
                for rollout in episode.rollouts:
                    if rollout.trace is not None:
                        await env.harness.finish_trace(rollout.trace, error=exc)
                raise

    try:
        async with env.serving():
            async with asyncio.TaskGroup() as group:
                for _ in range(min(config.concurrency, total)):
                    group.create_task(worker())
        status = "completed"
    finally:
        await client.close()
        scores = [result["score"] for result in results if result["score"] is not None]
        summary = {
            "run_id": run_id,
            "status": status,
            "total": total,
            "completed": len(results),
            "failed": len(results) - len(scores),
            "scored": len(scores),
            "mean_score": mean(scores) if scores else None,
            "braintrust_project": config.braintrust_project,
            "braintrust_experiment": experiment,
            "results": results,
        }
        _write_json(config.output_dir / "summary.json", json.dumps(summary, indent=2))
        logger.info(
            "Eval %s: %d/%d completed, %d failed, mean score=%s; results: %s",
            status,
            len(results),
            total,
            summary["failed"],
            summary["mean_score"],
            config.output_dir,
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    values = json.loads(args.config.read_text())
    config = EvalConfig.model_validate(values)
    summary = asyncio.run(evaluate(config))
    raise SystemExit(1 if summary["failed"] else 0)


def _write_json(path: Path, data: str) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(data + "\n")
    temporary.replace(path)


if __name__ == "__main__":
    main()
