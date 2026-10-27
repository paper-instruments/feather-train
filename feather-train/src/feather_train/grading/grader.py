"""Run task-owned Harbor graders in the isolated verifier."""

from __future__ import annotations

import asyncio
import io
import json
import logging
import math
import os
import shlex
import tarfile
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from huggingface_hub import snapshot_download
from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError, RevisionNotFoundError
from pydantic import BaseModel, ConfigDict, Field
from verifiers.v1 import SandboxError

from feather_train.errors import NON_RETRYABLE_CAUSES, NonRetryableHarnessError
from feather_train.grading.dependencies import GRADER_PIP_DEPS as GRADER_PIP_DEPS
from feather_train.grading.dependencies import (
    GRADER_SETUP_CONFIG_ERROR,
    grader_dependency_check_script,
)
from feather_train.grading.report import report_to_criteria
from feather_train.grading.types import (
    AgentEvidence,
    GradeResult,
    GradeStep,
    JudgeAttemptFailure,
)
from feather_train.sandbox.io import exec_bash, read_remote_bytes
from feather_train.sandbox.paths import (
    BAKED_TESTS_DIR,
    SNAPSHOTS_DIR,
    STATE_EXPORT_TOKEN_ENV,
    STATE_EXPORT_TOKEN_PATH,
    TASK_DIR,
    TESTS_DIR,
    VERIFIER_LOGS_DIR,
)
from feather_train.tasks.paths import is_safe_task_id

if TYPE_CHECKING:
    from feather_train.sandbox.lifecycle import SandboxIO
    from feather_train.tasks.taskset import FeatherTaskData

logger = logging.getLogger(__name__)

_HF_CONFIGURATION_ERRORS = (
    GatedRepoError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)

# Task-owned grader input and output files.
HARBOR_REWARD_PATH = f"{VERIFIER_LOGS_DIR}/reward.txt"
HARBOR_REWARD_JSON_PATH = f"{VERIFIER_LOGS_DIR}/reward.json"
HARBOR_REWARD_DETAIL_PATH = f"{VERIFIER_LOGS_DIR}/reward_detail.json"
HARBOR_SUBMISSION_TRANSPORT_PATH = f"{TESTS_DIR}/submission_transport.txt"
HARBOR_AGENT_TRACE_PATH = f"{TASK_DIR}/agent_trace.json"

TRUSTED_GRADER_ENV = "PATH=/usr/local/bin:/usr/bin:/bin PYTHONSAFEPATH=1 PYTHONNOUSERSITE=1"
NON_RETRYABLE_GRADER_EXIT_CODE = 2


class GradingConfig(BaseModel):
    """Per-rollout grading config for the harbor grader.

    ``poll_timeout`` bounds each in-sandbox ``test.sh`` attempt. Failed judge
    executions are retried up to ``judge_max_attempts``; verifier exit 2 and a
    judge run that exits 0 but writes an invalid ``reward.txt`` fail fast instead.
    API keys are read from the environment by the harness.
    """

    model_config = ConfigDict(extra="forbid")

    # A healthy judge run (Reducto parse + rubric judge call) finishes
    # in minutes; an hour-long attempt is a hang, and a timed-out exec yields
    # no stdout/stderr to diagnose. Fail the attempt fast and let the retry
    # loop (and rollout retry above it) own recovery.
    poll_timeout: float = Field(300.0)
    judge_max_attempts: int = Field(3, ge=1)
    judge_retry_backoff_seconds: float = Field(2.0, ge=0.0)
    judge_retry_max_backoff_seconds: float = Field(30.0, ge=0.0)
    # Trajectory conversion and staging scale with rollout length; graders opt in explicitly.
    stage_agent_trace: bool = False
    # Evaluation runs set this explicitly so the immutable run receipt names
    # the exact model invoked by the portable grader. Legacy benchmark runs may
    # omit it and retain the grader archive's default.
    judge_model: str | None = Field(default=None, min_length=1, max_length=500)
    judge_api_key_var: str | None = Field(default=None, min_length=1, max_length=100)
    judge_base_url: str | None = Field(default=None, min_length=1, max_length=500)


@dataclass(frozen=True, slots=True)
class HarborGradeResult:
    score: float
    detail: dict[str, object]
    stdout: str


class HarborGradingError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        failed_attempts: list[JudgeAttemptFailure] | None = None,
        steps: tuple[GradeStep, ...] = (),
        retryable: bool = True,
    ) -> None:
        super().__init__(message)
        self.failed_attempts = failed_attempts or []
        self.steps = steps
        self.retryable = retryable


async def grade_task(
    task: FeatherTaskData,
    session: SandboxIO,
    *,
    instruction: str,
    final_answer: str,
    config: GradingConfig,
    agent_evidence: AgentEvidence | None = None,
    braintrust_parent: str | None = None,
    braintrust_project: str | None = None,
) -> GradeResult:
    """Restore baked grader tests, write per-task overlays, run test.sh, read the result.

    Pure I/O against a live sandbox session.
    Returns a valid parsed score or raises ``HarborGradingError`` with
    structured attempt errors.
    """
    grading_run_id = "gr_" + uuid.uuid4().hex[:12]
    overlays = {
        f"{TASK_DIR}/instruction.md": instruction.encode("utf-8"),
        f"{TASK_DIR}/final_response.txt": final_answer.encode("utf-8"),
    }
    if task.source != "benchmark":
        overlays.update(
            {
                f"{TESTS_DIR}/rubric.yaml": json.dumps(
                    task.rubric_doc, indent=2, ensure_ascii=False
                ).encode("utf-8"),
                f"{TESTS_DIR}/expected_deliverable.json": json.dumps(
                    task.expected_deliverable or {}
                ).encode("utf-8"),
                HARBOR_SUBMISSION_TRANSPORT_PATH: f"{task.submission_transport}\n".encode(),
            }
        )
    steps: list[GradeStep] = []
    try:
        if config.stage_agent_trace:
            if agent_evidence is None:
                raise NonRetryableHarnessError(
                    "stage_agent_trace requires agent evidence from Feather"
                )
            overlays[HARBOR_AGENT_TRACE_PATH] = (
                json.dumps(
                    {
                        **agent_evidence.as_dict(),
                        "task_id": task.task_id,
                        "final_response": final_answer,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            ).encode("utf-8")

        await _prepare_harbor_directories(
            session,
            steps=steps,
            timeout=config.poll_timeout,
            preserve_logs=task.environment_mode is not None,
        )
        await _stage_packaged_grader(
            session,
            steps=steps,
            hf_dataset=task.hf_dataset,
            hf_revision=task.hf_revision,
            task_id=task.task_id,
            has_task_tests=task.task_artifacts.tests,
            overlays=overlays,
            timeout=config.poll_timeout,
        )
        if task.environment_mode is None:
            await _assert_harbor_grader_dependencies(
                session,
                steps=steps,
                timeout=config.poll_timeout,
            )
    except Exception as exc:
        raise HarborGradingError(
            f"grading[{grading_run_id}] {type(exc).__name__}: {exc}",
            steps=tuple(steps),
            retryable=not isinstance(exc, NON_RETRYABLE_CAUSES),
        ) from exc

    workdir = task.workdir if task.environment_mode is not None else TESTS_DIR
    timeout = task.verifier_timeout_sec or config.poll_timeout
    cmd = _packaged_grader_test_command(
        task_id=task.task_id,
        world_id=task.world_id,
        grading_evidence_mode=task.grading_evidence_mode,
        judge_model=config.judge_model,
        judge_api_key=(
            _judge_api_key(config)
            if task.environment_mode is None or config.judge_api_key_var
            else ""
        ),
        judge_api_key_var=config.judge_api_key_var,
        judge_base_url=(
            _judge_base_url(config)
            if task.environment_mode is None
            else config.judge_base_url or ""
        ),
        braintrust_parent=braintrust_parent,
        braintrust_project=braintrust_project,
        workdir=workdir or TESTS_DIR,
        official=task.environment_mode is not None,
    )
    failed_attempts: list[JudgeAttemptFailure] = []
    for attempt in range(1, config.judge_max_attempts + 1):
        result = await _run_harbor_grader_attempt(
            session,
            cmd,
            attempt=attempt,
            timeout=timeout,
            steps=steps,
        )

        if isinstance(result, HarborGradeResult):
            detail = result.detail
            report_error = None
            try:
                criteria = tuple(report_to_criteria(detail.get("report") or [], rubric=task.rubric))
            except (TypeError, ValueError) as exc:
                report_error = f"{type(exc).__name__}: {exc}"
                logger.warning(
                    "Could not convert optional grader report for task %s; keeping reward",
                    task.task_id,
                    exc_info=True,
                )
                criteria = ()
            grade = GradeResult(
                score=result.score,
                num_met=detail.get("num_met"),
                num_criteria=detail.get("num_criteria"),
                criteria=criteria,
                report_error=report_error,
                stdout=result.stdout,
                failed_attempts=failed_attempts,
                steps=tuple(steps),
                judge_usage=(
                    dict(detail.get("usage") or {}) if isinstance(detail.get("usage"), dict) else {}
                ),
            )
            logger.info(
                "feather_train: harbor grader score=%.3f met=%s/%s failed_attempts=%d",
                grade.score,
                grade.num_met,
                grade.num_criteria,
                len(grade.failed_attempts),
            )
            return grade

        failed_attempts.append(result)
        if not result.retriable:
            raise HarborGradingError(
                f"grading[{grading_run_id}] harbor grader failed fatally on "
                f"attempt {attempt}: {result.message}",
                failed_attempts=failed_attempts,
                steps=tuple(steps),
                retryable=False,
            )
        if attempt >= config.judge_max_attempts:
            raise HarborGradingError(
                f"grading[{grading_run_id}] harbor grader failed after "
                f"{config.judge_max_attempts} attempts: {result.message}",
                failed_attempts=failed_attempts,
                steps=tuple(steps),
            )

        logger.warning(
            "feather_train: harbor grader attempt %d/%d failed (%s); retrying",
            attempt,
            config.judge_max_attempts,
            result.message,
        )
        if config.judge_retry_backoff_seconds:
            await asyncio.sleep(_judge_retry_sleep_seconds(config, attempt))
        try:
            reset = await exec_bash(
                session,
                f"rm -f -- {shlex.quote(HARBOR_REWARD_PATH)} "
                f"{shlex.quote(HARBOR_REWARD_JSON_PATH)} "
                f"{shlex.quote(HARBOR_REWARD_DETAIL_PATH)}",
                timeout=config.poll_timeout,
            )
            if not reset.ok():
                raise SandboxError(f"reward file reset failed: exit {reset.exit_code}")
        except SandboxError as exc:
            raise HarborGradingError(
                f"grading[{grading_run_id}] {type(exc).__name__}: {exc}",
                failed_attempts=failed_attempts,
                steps=tuple(steps),
            ) from exc


async def _prepare_harbor_directories(
    session: SandboxIO,
    *,
    steps: list[GradeStep],
    timeout: float,
    preserve_logs: bool = False,
) -> None:
    with _timed_step(steps, "prepare_grader_directories") as step:
        # Empty ``/tests`` for a clean restore from baked tests. Do not unlink the
        # mountpoint: the peer verifier bind-mounts this host path for the snapshot life.
        # Official separate collect lands under ``/logs/artifacts``; reset only
        # ``/logs/verifier`` so a planted reward cannot survive into the read.
        logs_reset = (
            f"rm -rf -- {VERIFIER_LOGS_DIR}"
            if preserve_logs
            else "rm -rf -- /logs"
        )
        command = (
            f"{logs_reset} && "
            f"rm -rf -- {SNAPSHOTS_DIR} && "
            f"rm -rf -- {shlex.quote(HARBOR_AGENT_TRACE_PATH)} && "
            f"mkdir -p -- {TESTS_DIR} && "
            f"find {TESTS_DIR} -mindepth 1 -delete && "
            f"chown 0:0 {TESTS_DIR} && "
            f"chmod 700 {TESTS_DIR} && "
            f"install -d -o 0 -g 0 -m 700 "
            f"/logs {VERIFIER_LOGS_DIR} {SNAPSHOTS_DIR}"
        )
        result = await exec_bash(session, command, timeout=timeout)
        step.metadata["exit_code"] = result.exit_code
        if result.ok():
            return
        detail = result.stderr or result.stdout or "<no output>"
        raise SandboxError(f"grader directory setup failed: {detail[:1500]}")


async def _stage_packaged_grader(
    session: SandboxIO,
    *,
    steps: list[GradeStep],
    hf_dataset: str,
    hf_revision: str,
    task_id: str,
    overlays: dict[str, bytes],
    timeout: float,
    has_task_tests: bool = False,
) -> None:
    with _timed_step(steps, "restore_baked_grader_data") as step:
        restore = (
            f"if ! test -f {shlex.quote(BAKED_TESTS_DIR)}/test.sh; then "
            f"echo 'baked test.sh is missing' >&2; exit {GRADER_SETUP_CONFIG_ERROR}; fi; "
            f"cp -a {shlex.quote(BAKED_TESTS_DIR)}/. {shlex.quote(TESTS_DIR)}/ && "
            f"test -f {shlex.quote(TESTS_DIR)}/test.sh"
        )
        result = await exec_bash(session, restore, timeout=timeout)
        step.metadata["exit_code"] = result.exit_code
        step.metadata["source"] = BAKED_TESTS_DIR
        if not result.ok():
            detail = result.stderr or result.stdout or "<no output>"
            error = (
                NonRetryableHarnessError
                if result.exit_code == GRADER_SETUP_CONFIG_ERROR
                else SandboxError
            )
            raise error(
                f"restore baked grader tests from {BAKED_TESTS_DIR} into "
                f"{TESTS_DIR} failed: {detail[:1500]}"
            )

    staged_overlays: dict[str, bytes] = {}
    if has_task_tests:
        with _timed_step(steps, "download_task_grader_data") as step:
            staged_overlays = await _download_harbor_task_test_files(
                hf_dataset,
                task_id,
                token=os.environ.get("HF_TOKEN"),
                hf_revision=hf_revision,
            )
            step.metadata["files"] = len(staged_overlays)
            step.metadata["bytes"] = sum(map(len, staged_overlays.values()))

    staged_overlays.update(overlays)

    with _timed_step(steps, "write_grader_overlays") as step:
        archive = _build_overlay_archive(staged_overlays)
        result = await session.extract_archive(archive, "/", timeout=timeout)
        step.metadata["exit_code"] = result.exit_code
        if not result.ok():
            detail = result.stderr or result.stdout or "<no output>"
            raise SandboxError(f"grader overlay extraction failed: {detail[:1500]}")
        step.metadata["files"] = len(staged_overlays)
        step.metadata["bytes"] = sum(map(len, staged_overlays.values()))


def _build_overlay_archive(overlays: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for path, data in sorted(overlays.items()):
            info = tarfile.TarInfo(path.removeprefix("/"))
            info.size = len(data)
            info.mode = 0o600
            info.mtime = 0
            archive.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


def _files_under(source: Path, *, destination: str) -> dict[str, bytes]:
    return {
        f"{destination}/{path.relative_to(source).as_posix()}": path.read_bytes()
        for path in sorted(source.rglob("*"))
        if path.is_file()
    }


async def _download_harbor_task_test_files(
    hf_dataset: str,
    task_id: str,
    *,
    token: str | None,
    hf_revision: str,
) -> dict[str, bytes]:
    def download() -> dict[str, bytes]:
        if not is_safe_task_id(task_id):
            raise NonRetryableHarnessError("task_id must be one safe dataset path component")
        prefix = f"tasks/{task_id}/tests"
        try:
            snapshot = snapshot_download(
                hf_dataset,
                repo_type="dataset",
                token=token,
                revision=hf_revision,
                allow_patterns=[f"{prefix}/**"],
            )
        except _HF_CONFIGURATION_ERRORS as exc:
            raise NonRetryableHarnessError(
                f"cannot load task grader data from {hf_dataset}@{hf_revision}: {exc}"
            ) from exc
        source_dir = Path(snapshot, prefix)
        if not source_dir.is_dir():
            raise NonRetryableHarnessError(
                f"task {task_id!r} declares tests but {prefix}/ was not found in "
                f"{hf_dataset}@{hf_revision}"
            )
        files = _files_under(source_dir, destination=f"{TESTS_DIR}/task")
        if not files:
            raise NonRetryableHarnessError(
                f"task {task_id!r} declares tests but {prefix}/ contains no files"
            )
        return files

    return await asyncio.to_thread(download)


async def _assert_harbor_grader_dependencies(
    session: SandboxIO,
    *,
    steps: list[GradeStep],
    timeout: float,
) -> None:
    """Fail closed when the verifier image lacks the pinned grader packages."""
    with _timed_step(steps, "assert_grader_dependencies") as step:
        cmd = f"/usr/bin/env {TRUSTED_GRADER_ENV} /usr/local/bin/python -c " + shlex.quote(
            grader_dependency_check_script()
        )
        result = await exec_bash(session, cmd, timeout=timeout)
        step.metadata["exit_code"] = result.exit_code
        if result.ok():
            return
        detail = result.stderr or result.stdout or "<no output>"
        error = (
            NonRetryableHarnessError
            if result.exit_code == GRADER_SETUP_CONFIG_ERROR
            else SandboxError
        )
        raise error(f"grader dependency check failed in verifier image: {detail[:1500]}")


def _judge_api_key(config: GradingConfig) -> str:
    return os.environ.get(config.judge_api_key_var or "OPENAI_VERIFIER_API_KEY", "")


def _judge_base_url(config: GradingConfig) -> str:
    return config.judge_base_url or os.environ.get("OPENAI_VERIFIER_BASE_URL", "")


def _packaged_grader_test_command(
    *,
    task_id: str,
    world_id: str,
    grading_evidence_mode: str | None,
    judge_model: str | None = None,
    judge_api_key: str = "",
    judge_api_key_var: str | None = None,
    judge_base_url: str = "",
    braintrust_parent: str | None = None,
    braintrust_project: str | None = None,
    workdir: str = TESTS_DIR,
    official: bool = False,
) -> str:
    if official:
        env_vars = {
            "HARBOR_TASK_ID": task_id,
            "HARBOR_WORLD_ID": world_id,
        }
        if judge_api_key_var in {"FIREWORKS_AI_API_KEY", "FIREWORKS_API_KEY"}:
            env_vars["FIREWORKS_AI_API_KEY"] = judge_api_key
            env_vars["FIREWORKS_API_KEY"] = judge_api_key
        if judge_model or judge_base_url:
            env_vars.update({
                "OPENAI_VERIFIER_API_KEY": judge_api_key,
                "OPENAI_VERIFIER_BASE_URL": judge_base_url,
                "OPENAI_VERIFIER_MODEL": judge_model or "",
            })
        trusted = ""
    else:
        env_vars = {
            "OPENAI_VERIFIER_API_KEY": judge_api_key,
            "OPENAI_VERIFIER_BASE_URL": judge_base_url,
            "OPENAI_VERIFIER_MODEL": judge_model or "",
            "REDUCTO_API_KEY": os.environ.get("REDUCTO_API_KEY", ""),
            "GRADER_INPUT_CHAR_CEILING": os.environ.get("GRADER_INPUT_CHAR_CEILING", ""),
            "BRAINTRUST_API_KEY": os.environ.get("BRAINTRUST_API_KEY", ""),
            "BRAINTRUST_PROJECT": braintrust_project or "",
            "BRAINTRUST_ORG": os.environ.get("BRAINTRUST_ORG", ""),
            "BRAINTRUST_API_URL": os.environ.get("BRAINTRUST_API_URL", ""),
            "BRAINTRUST_ENVIRONMENT": os.environ.get("BRAINTRUST_ENVIRONMENT", ""),
            "HARBOR_BRAINTRUST_PARENT": braintrust_parent or "",
            "HARBOR_BRAINTRUST_CONTENT": os.environ.get("HARBOR_BRAINTRUST_CONTENT", ""),
            "HARBOR_TASK_ID": task_id,
            "HARBOR_WORLD_ID": world_id,
            "HARBOR_GRADING_EVIDENCE_MODE": grading_evidence_mode or "",
        }
        trusted = TRUSTED_GRADER_ENV
    env_assigns = " ".join(
        f"{key}={shlex.quote(value)}" for key, value in env_vars.items() if value
    )
    env_prefix = " ".join(part for part in (trusted, env_assigns) if part)
    env_prefix = f"{env_prefix} " if env_prefix else ""
    token = (
        f'token="$(cat {shlex.quote(STATE_EXPORT_TOKEN_PATH)} 2>/dev/null || true)"; '
        f'export {STATE_EXPORT_TOKEN_ENV}="$token"; '
    )
    # Official pins hide pytest in /opt/venv and only add it from ~/.bashrc.
    # grader_process already execs `bash -lc`; a nested non-interactive bash -c
    # hits Ubuntu's "return if not interactive" guard, so activate here.
    if official:
        suffix = " ".join(
            part for part in (env_assigns, f"/bin/bash {TESTS_DIR}/test.sh") if part
        )
        return (
            f"{token}"
            f"cd {shlex.quote(workdir)} && "
            f"{{ [ -d /opt/venv/bin ] && PATH=/opt/venv/bin:$PATH; "
            f"[ -f /root/.bashrc ] && . /root/.bashrc; "
            f"export PATH; }} && "
            f"{suffix}"
        )
    # Peer verifier materialises the JSON-state token at prepare time (not container create).
    return (
        f"{token}"
        f"/usr/bin/env --chdir={shlex.quote(workdir)} "
        f"{env_prefix}"
        f'{STATE_EXPORT_TOKEN_ENV}="$token" '
        f"/bin/bash {TESTS_DIR}/test.sh"
    )


async def _run_harbor_grader_attempt(
    session: SandboxIO,
    cmd: str,
    *,
    attempt: int,
    timeout: float,
    steps: list[GradeStep],
) -> HarborGradeResult | JudgeAttemptFailure:
    with _timed_step(steps, "run_tests", attempt=attempt) as step:
        try:
            process = await session.exec_grader(cmd, timeout=timeout)
        except SandboxError as exc:
            step.ok = False
            step.metadata["error"] = str(exc)
            return JudgeAttemptFailure(
                attempt=attempt,
                message=f"grader exec failed: {exc}",
                exit_code=-1,
            )
        step.ok = process.ok()
        step.metadata.update(
            {
                "exit_code": process.exit_code,
                "stdout_bytes": len(process.stdout.encode("utf-8")),
                "stderr_bytes": len(process.stderr.encode("utf-8")),
            }
        )

    if not process.ok():
        return JudgeAttemptFailure(
            attempt=attempt,
            message=process.stderr[-1500:] or process.stdout[-1500:] or "<no output>",
            retriable=process.exit_code != NON_RETRYABLE_GRADER_EXIT_CODE,
            exit_code=process.exit_code,
        )

    with _timed_step(steps, "read_reward", attempt=attempt) as step:
        try:
            score, source, payload = await _read_harbor_reward(session)
        except SandboxError as exc:
            step.ok = False
            step.metadata["error"] = str(exc)
            return JudgeAttemptFailure(
                attempt=attempt,
                message=f"reward read failed: {exc}",
                exit_code=process.exit_code,
            )
        step.metadata["source"] = source
        if score is None or not math.isfinite(score) or not 0.0 <= score <= 1.0:
            step.ok = False
            return JudgeAttemptFailure(
                attempt=attempt,
                message=f"invalid {source}: {payload[:500]!r}",
                retriable=False,
                exit_code=process.exit_code,
            )

    with _timed_step(steps, "read_grader_detail", attempt=attempt) as step:
        detail, detail_ok = await _read_reward_detail(session)
        step.ok = detail_ok
        step.metadata["keys"] = len(detail)

    return HarborGradeResult(
        score=score,
        detail=detail,
        stdout=process.stdout,
    )


async def _read_harbor_reward(
    session: SandboxIO,
) -> tuple[float | None, str, str]:
    """Official Harbor suites write reward.json; Factory HarborGrader writes reward.txt."""
    reward_json = await read_remote_bytes(session, HARBOR_REWARD_JSON_PATH, check=False)
    if reward_json is not None:
        payload = reward_json.decode("utf-8", errors="replace")
        try:
            parsed = json.loads(payload)
            score = float(parsed["reward"])
        except (ValueError, TypeError, KeyError, json.JSONDecodeError):
            return None, HARBOR_REWARD_JSON_PATH, payload
        return score, HARBOR_REWARD_JSON_PATH, payload
    reward_raw = await read_remote_bytes(session, HARBOR_REWARD_PATH)
    payload = reward_raw.decode("utf-8", errors="replace").strip()
    try:
        return float(payload), HARBOR_REWARD_PATH, payload
    except ValueError:
        return None, HARBOR_REWARD_PATH, payload


async def _read_reward_detail(
    session: SandboxIO,
) -> tuple[dict[str, object], bool]:
    try:
        detail_raw = await read_remote_bytes(session, HARBOR_REWARD_DETAIL_PATH)
        detail = json.loads(detail_raw.decode("utf-8", errors="replace"))
    except NON_RETRYABLE_CAUSES:
        raise
    except Exception:  # noqa: BLE001
        logger.warning(
            "feather_train: failed to read or parse %s; continuing with empty grading detail",
            HARBOR_REWARD_DETAIL_PATH,
            exc_info=True,
        )
        return {}, False
    if not isinstance(detail, dict):
        logger.warning(
            "feather_train: %s did not contain a JSON object; continuing with empty grading detail",
            HARBOR_REWARD_DETAIL_PATH,
        )
        return {}, False
    return detail, True


@contextmanager
def _timed_step(
    steps: list[GradeStep],
    name: str,
    **metadata: object,
) -> Iterator[GradeStep]:
    start = time.time()
    step = GradeStep(name=name, start=start, end=start, metadata=metadata)
    steps.append(step)
    try:
        yield step
    except BaseException:
        step.ok = False
        raise
    finally:
        step.end = time.time()


def _judge_retry_sleep_seconds(config: GradingConfig, attempt: int) -> float:
    return min(
        config.judge_retry_backoff_seconds * (2 ** (attempt - 1)),
        config.judge_retry_max_backoff_seconds,
    )
