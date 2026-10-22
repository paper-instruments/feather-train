import asyncio
import json

import pytest
import verifiers.v1 as vf

from feather_train import eval as feather_eval
from feather_train.environment import load_environment
from feather_train.errors import NonRetryableHarnessError
from feather_train.harness import FeatherHarness, FeatherHarnessConfig
from feather_train.observability.braintrust import BraintrustRolloutStart
from feather_train.tasks.taskset import FeatherTask, FeatherTaskData, FeatherTaskset


@pytest.fixture
def evaluation(tmp_path, monkeypatch):
    config = feather_eval.EvalConfig(
        hf_dataset="unused/test",
        hf_revision="a" * 40,
        model="test",
        base_url="http://unused.invalid/v1",
        output_dir=tmp_path / "eval",
        braintrust_project="test",
        concurrency=2,
        max_tasks=None,
        rollout_max_retries=0,
    )
    env = load_environment(
        hf_dataset=config.hf_dataset,
        hf_revision=config.hf_revision,
        rollout_max_retries=0,
        enable_braintrust=False,
    )
    tasks = []
    for idx in range(4):
        data = FeatherTaskData(
            idx=idx,
            task_id=f"task-{idx}",
            task_name="test",
            world_id="test",
            rollout_group_id="existing-group" if idx == 0 else None,
            hf_dataset=config.hf_dataset,
            hf_revision=config.hf_revision,
            prompt=[{"role": "user", "content": "solve"}],
            rubric=[],
            rubric_doc={},
            expected_deliverable={},
            submission_transport="text_response",
            grading_evidence_mode="direct_text",
            requires_web_search=False,
            network_mode="public",
            mcp_servers=[],
        )
        tasks.append(FeatherTask(data, env.taskset.config.task))

    class Taskset(FeatherTaskset):
        def load(self):
            return iter(tasks)

    env.taskset = Taskset(env.taskset.config)
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setattr(feather_eval, "load_environment", lambda **kwargs: env)
    return config, env


async def test_bounds_concurrency_and_saves_results_before_run_finishes(evaluation):
    config, env = evaluation
    release = asyncio.Event()
    second_started = asyncio.Event()

    class Harness(FeatherHarness):
        active = 0
        peak = 0
        launches = 0

        async def launch(self, ctx, trace, *args):
            self.active += 1
            self.launches += 1
            self.peak = max(self.peak, self.active)
            try:
                if self.launches == 1:
                    await second_started.wait()
                else:
                    second_started.set()
                    await release.wait()
                if trace.task.data.idx == 2:
                    raise vf.HarnessError("provider unavailable")
                trace.info.update(
                    grading_score=0.75, bt_span_permalink="https://braintrust.dev/trace"
                )
                return vf.ProgramResult(exit_code=0, stdout="", stderr="")
            finally:
                self.active -= 1

    env.harness = Harness(FeatherHarnessConfig(enable_braintrust=False))
    async with asyncio.TaskGroup() as group:
        run = group.create_task(feather_eval.evaluate(config))
        async with asyncio.timeout(10):
            while not list((config.output_dir / "traces").glob("*.json")):
                await asyncio.sleep(0.01)
        assert not run.done()
        release.set()
    summary = run.result()
    assert env.harness.peak == 2
    assert summary["completed"] == summary["total"] == 4
    assert summary["failed"] == 1
    assert summary["mean_score"] == 0.75
    assert summary["scored"] == 3
    traces = [
        json.loads(path.read_text()) for path in (config.output_dir / "traces").glob("*.json")
    ]
    assert {trace["task"]["data"]["task_id"] for trace in traces} == {f"task-{i}" for i in range(4)}
    assert all(trace["task"]["data"]["eval_run_id"] == summary["run_id"] for trace in traces)
    assert sum(result["braintrust_url"] is not None for result in summary["results"]) == 3
    assert json.loads((config.output_dir / "summary.json").read_text()) == summary


async def test_existing_output_directory_fails_before_loading_tasks(evaluation, monkeypatch):
    config, _env = evaluation
    config.output_dir.mkdir()
    (config.output_dir / "summary.json").write_text("{}")

    def load_environment(**kwargs):
        raise AssertionError("tasks were loaded for an unusable output directory")

    monkeypatch.setattr(feather_eval, "load_environment", load_environment)
    with pytest.raises(FileExistsError, match="already exists"):
        await feather_eval.evaluate(config)
    assert (config.output_dir / "summary.json").read_text() == "{}"


async def test_repeated_rollouts_share_one_group_per_task(evaluation):
    config, env = evaluation
    config.rollouts_per_example = 3

    class Harness(FeatherHarness):
        async def launch(self, ctx, trace, *args):
            trace.info["grading_score"] = 0.75
            return vf.ProgramResult(exit_code=0, stdout="", stderr="")

    env.harness = Harness(FeatherHarnessConfig(enable_braintrust=False))
    summary = await feather_eval.evaluate(config)
    traces = [
        json.loads(path.read_text()) for path in (config.output_dir / "traces").glob("*.json")
    ]
    groups = {}
    for trace in traces:
        task_id = trace["task"]["data"]["task_id"]
        group_id = trace["task"]["data"]["rollout_group_id"]
        assert trace["info"]["rollout_group_id"] == group_id
        groups.setdefault(task_id, []).append(group_id)
    assert all(len(ids) == 3 and len(set(ids)) == 1 for ids in groups.values())
    assert len({ids[0] for ids in groups.values()}) == 4
    assert groups["task-0"] == ["existing-group"] * 3
    assert all(
        result["rollout_group_id"] == groups[result["task_id"]][0] for result in summary["results"]
    )


async def test_nonretryable_error_stops_other_rollouts_and_preserves_failure(evaluation):
    config, env = evaluation
    config.rollouts_per_example = 3
    started = asyncio.Event()
    cancelled = asyncio.Event()
    logged_rollouts = {}

    class Harness(FeatherHarness):
        launches = 0

        async def launch(self, ctx, trace, *args):
            logged_rollouts[trace.id] = self.braintrust.open_rollout(
                BraintrustRolloutStart(
                    key=trace.id, model=ctx.model, name="test", input="", expected=None, metadata={}
                )
            )
            self.launches += 1
            if self.launches == 1:
                await started.wait()
                raise NonRetryableHarnessError("missing baked runtime")
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError as exc:
                logged_rollouts[trace.id].record_interrupted(
                    phase="agent", exc=exc, fallback_num_turns=0
                )
                raise
            finally:
                cancelled.set()

    env.harness = Harness(FeatherHarnessConfig(enable_braintrust=False))
    with pytest.raises(ExceptionGroup) as error:
        await asyncio.wait_for(feather_eval.evaluate(config), timeout=10)
    assert error.value.subgroup(NonRetryableHarnessError) is not None
    assert cancelled.is_set()
    assert len(logged_rollouts) == 2
    assert any(
        item.metadata.get("error") == "CancelledError during agent"
        for item in logged_rollouts.values()
    )
    assert all(env.harness.braintrust.active_rollout(key) is None for key in logged_rollouts)
    assert any(
        "missing baked runtime" in item.metadata["error"] for item in logged_rollouts.values()
    )
    summary = json.loads((config.output_dir / "summary.json").read_text())
    assert summary["status"] == "interrupted"
    assert summary["completed"] == summary["failed"] == 1
    assert summary["mean_score"] is None
    (trace_path,) = (config.output_dir / "traces").glob("*.json")
    assert json.loads(trace_path.read_text())["errors"][-1]["type"] == "NonRetryableHarnessError"


@pytest.mark.parametrize("score, failed", [(None, 4), (0.0, 0)])
async def test_unscored_timeout_is_a_failure_but_zero_is_a_valid_score(evaluation, score, failed):
    config, env = evaluation

    class Harness(FeatherHarness):
        async def run(self, ctx, trace, *args):
            if score is None:
                trace.stop("harness_timeout")
            else:
                trace.info["grading_score"] = score
                trace.stop("agent_completed")

    env.harness = Harness(FeatherHarnessConfig(enable_braintrust=False))
    summary = await feather_eval.evaluate(config)
    assert summary["failed"] == failed
    assert summary["mean_score"] == score
    if score is None:
        assert all(result["stop_condition"] == "harness_timeout" for result in summary["results"])
