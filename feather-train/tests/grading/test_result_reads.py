"""Reading a completed grade must not unnecessarily execute the judge again."""

from functools import partial

import pytest
from e2b.exceptions import AuthenticationException
from tenacity import AsyncRetrying
from verifiers.v1 import SandboxError

from feather_train.grading.grader import GradingConfig, HarborGradingError, grade_task
from feather_train.sandbox import io
from feather_train.sandbox.lifecycle import CommandOutput, SandboxIO

from .helpers import GraderSandbox, _is_harbor_test_command, task


@pytest.mark.parametrize("filename", ["reward.txt", "reward.json", "reward_detail.json"])
async def test_transient_result_read_recovers_without_repeating_grader(filename):
    class InterruptedRead(GraderSandbox):
        reads = 0

        async def exec(self, *args, **kwargs):
            if args[-1].startswith("base64 -w0 ") and args[-1].endswith(filename):
                self.reads += 1
                if self.reads == 1:
                    raise SandboxError("connection lost")
            return await super().exec(*args, **kwargs)

    sandbox = InterruptedRead(reward_json=b'{"reward":0.75}' if filename == "reward.json" else None)
    result = await grade_task(
        task(),
        sandbox,
        instruction="Solve this exactly.",
        config=GradingConfig(),
        final_answer="",
    )
    assert result.score == 0.75
    assert result.num_met == 3
    assert sandbox.reads == 2
    assert sandbox.test_runs == 1
    assert not result.failed_attempts


async def test_reward_read_exhausts_local_retries_before_failing_grade(monkeypatch):
    waits = []

    async def sleep(delay):
        waits.append(delay)

    # Skip only the backoff clock; keep the real read and grading policies.
    monkeypatch.setattr(io, "AsyncRetrying", partial(AsyncRetrying, sleep=sleep))

    class UnavailableRead(GraderSandbox):
        reads = 0

        async def exec(self, *args, **kwargs):
            if args[-1].startswith("base64 -w0 "):
                self.reads += 1
                raise SandboxError("connection lost")
            return await super().exec(*args, **kwargs)

    sandbox = UnavailableRead()
    with pytest.raises(HarborGradingError) as raised:
        await grade_task(
            task(),
            sandbox,
            instruction="Solve this exactly.",
            config=GradingConfig(judge_max_attempts=1),
            final_answer="",
        )
    assert raised.value.retryable
    assert sandbox.reads == 10
    assert sandbox.test_runs == 1
    assert waits == [1, 2, 4, 8, 16, 32, 64, 128, 256]


@pytest.mark.parametrize("filename", ["reward.txt", "reward.json", "reward_detail.json"])
async def test_authentication_failure_is_not_retried_or_hidden_as_optional_detail(filename):
    failure = AuthenticationException("credentials rejected")

    class RejectedRead(GraderSandbox):
        reads = 0

        async def exec(self, *args, **kwargs):
            if args[-1].startswith("base64 -w0 ") and args[-1].endswith(filename):
                self.reads += 1
                raise failure
            return await super().exec(*args, **kwargs)

    sandbox = RejectedRead()
    with pytest.raises(AuthenticationException) as raised:
        await grade_task(
            task(),
            sandbox,
            instruction="Solve this exactly.",
            config=GradingConfig(),
            final_answer="",
        )
    assert raised.value is failure
    assert sandbox.reads == 1
    assert sandbox.test_runs == 1


@pytest.mark.parametrize(
    "write_fresh_reward", [False, True], ids=["missing-reward", "missing-detail"]
)
async def test_failed_judge_cannot_leave_rewards_for_the_next_attempt(sandbox, write_fresh_reward):
    session = SandboxIO(sandbox, container=sandbox.container)

    class JudgeWithPartialOutputs(GraderSandbox):
        async def exec(self, *args, timeout=None):
            command = args[-1]
            if _is_harbor_test_command(command):
                self.test_runs += 1
                if self.test_runs == 1:
                    written = await session.exec(
                        "mkdir -p /logs/verifier && "
                        "echo 1 > /logs/verifier/reward.txt && "
                        'echo \'{"num_met":4,"num_criteria":4}\' > /logs/verifier/reward_detail.json',
                        shell=True,
                    )
                    assert written.ok(), written.stderr
                    return CommandOutput(1, "", "judge failed after writing results")
                if write_fresh_reward:
                    written = await session.exec(
                        "echo 0.25 > /logs/verifier/reward.txt", shell=True
                    )
                    assert written.ok(), written.stderr
                return CommandOutput(0, "", "")
            # Run the grader's emitted reset and reads against actual container files.
            if command.startswith(("rm -f -- ", "base64 -w0 ")):
                return await session.exec(*args, timeout=timeout)
            return await super().exec(*args, timeout=timeout)

    judge = JudgeWithPartialOutputs()
    config = GradingConfig(judge_max_attempts=2, judge_retry_backoff_seconds=0)
    if write_fresh_reward:
        result = await grade_task(
            task(),
            judge,
            instruction="Solve this exactly.",
            config=config,
            final_answer="",
        )
        assert result.score == 0.25
        assert result.num_met is None
        assert result.num_criteria is None
    else:
        with pytest.raises(HarborGradingError, match="reward read failed"):
            await grade_task(
                task(),
                judge,
                instruction="Solve this exactly.",
                config=config,
                final_answer="",
            )
    assert judge.test_runs == 2
