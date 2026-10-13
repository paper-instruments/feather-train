"""Standalone evaluation with released Verifiers v1."""

from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager

import verifiers.v1 as vf

from feather_train.harness import FeatherHarnessConfig
from feather_train.tasks.taskset import FeatherTasksetConfig

logger = logging.getLogger(__name__)


class FeatherEnv(vf.Environment):
    def episode(self, task: vf.Task, ctx: vf.ModelContext, n: int = 1) -> vf.Episode:
        if task.data.rollout_group_id is None:
            task = type(task)(
                task.data.model_copy(update={"rollout_group_id": uuid.uuid4().hex}), task.config
            )
        return super().episode(task, ctx, n=n)

    @asynccontextmanager
    async def serving(self):
        try:
            async with super().serving():
                yield
        finally:
            try:
                await self.harness.braintrust.flush_async()
            except Exception:
                logger.warning("Braintrust flush failed", exc_info=True)


def load_environment(
    *,
    hf_dataset: str,
    hf_revision: str,
    task_ids: list[str] | None = None,
    expected_world_ids: list[str] | None = None,
    max_tasks: int | None = None,
    system_prompt: str | None = None,
    rollout_max_retries: int = 2,
    **harness_options,
) -> FeatherEnv:
    config = FeatherHarnessConfig(**harness_options)
    return FeatherEnv(
        vf.EnvConfig(
            taskset=FeatherTasksetConfig(
                hf_dataset=hf_dataset,
                hf_revision=hf_revision,
                task_ids=task_ids,
                expected_world_ids=expected_world_ids,
                max_tasks=max_tasks,
                system_prompt=system_prompt,
            ),
            harness=config,
            retries=vf.RetryConfig(
                rollout=vf.RolloutRetryConfig(
                    max_retries=rollout_max_retries,
                    exclude=["NonRetryableHarnessError"],
                )
            ),
        )
    )
