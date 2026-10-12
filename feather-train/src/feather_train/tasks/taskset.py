"""Factory and benchmark rows plus baked E2B snapshots consumed by Verifiers."""

from __future__ import annotations

import json
import logging
import os
import tomllib
from collections.abc import Iterator
from pathlib import Path
from typing import Annotated, Any, Literal, Self
from urllib.parse import urlsplit

import verifiers.v1 as vf
import yaml
from feather_harness.prompt import RESOURCE_ROOT
from feather_harness.tools.mcp import McpServerSpec
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import EntryNotFoundError
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from feather_train.grading.rewards import harbor_grader_reward
from feather_train.sandbox.paths import WORKSPACE_DIR
from feather_train.tasks.harbor_spec import (
    CollectHook,
    HarborArtifact,
    HarborSpec,
    NetworkMode,
    parse_harbor_spec,
)
from feather_train.tasks.paths import is_safe_task_id

logger = logging.getLogger(__name__)

CODING_SYSTEM_PROMPT = (RESOURCE_ROOT / "system_prompt.md").read_text()

SUBMISSION_MCP_NAME = "submission"
NonEmptyString = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1),
]
SubmissionTransport = Literal[
    "path_manifest",
    "submit_tool",
    "snapshot_diff",
    "text_response",
]
GradingEvidenceMode = Literal[
    "direct_text",
    "submit_tool",
    "deliverables_folder",
    "in_place_edit",
    "world_state_diff",
]
_SUPPORTED_GRADING_EVIDENCE_CONTRACTS = frozenset(
    {
        ("path_manifest", "new_file", "deliverables_folder"),
        ("path_manifest", "edit", "deliverables_folder"),
        ("path_manifest", "edit", "in_place_edit"),
        ("text_response", "text_response", "direct_text"),
        ("submit_tool", "new_file", "submit_tool"),
        ("submit_tool", "text_response", "submit_tool"),
        ("submit_tool", "edit", "in_place_edit"),
        ("snapshot_diff", "new_file", "world_state_diff"),
        ("snapshot_diff", "edit", "world_state_diff"),
    }
)
ImmutableRevision = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        pattern=r"^[0-9a-f]{40}$",
    ),
]


class TaskArtifacts(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    inputs: bool = False
    tests: bool = False


class TaskSourceMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_job_id: str | None = None
    query_id: str | None = None
    content_hash: str | None = None
    dataset_id: str | None = None
    dataset_revision: str | None = None


class SnapshotEntry(BaseModel):
    model_config = ConfigDict(strict=True)

    id: str = Field(min_length=1)
    kind: Literal["compose"]
    dataset_env_oid: str | None = None


class SourceMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    resources: Any
    source_job_id: str | None = None
    source_query_id: str | None = None
    task_snapshot_hash: str | None = None
    dataset_id: str | None = None
    dataset_revision: str | None = None
    task_artifacts: TaskArtifacts = Field(default_factory=TaskArtifacts)

    @model_validator(mode="after")
    def complete_identity(self) -> Self:
        identity = {"source_job_id", "source_query_id", "task_snapshot_hash"}
        present = self.model_fields_set & identity
        if present and present != identity:
            raise ValueError("source provenance must be one complete atomic identity")
        return self


BENCHMARK_SOURCE_KEYS = frozenset(
    {"task_id", "world_id", "instruction", "requires_web_search", "metadata"}
)


class BenchmarkMetadata(BaseModel):
    """Runtime flags for five-key Harbor rows. Extra producer keys are retained."""

    model_config = ConfigDict(extra="allow", strict=True)

    task_artifacts: TaskArtifacts
    resources: Any = None
    source_job_id: str | None = None
    source_query_id: str | None = None
    task_snapshot_hash: str | None = None
    dataset_id: str | None = None
    dataset_revision: str | None = None

    @field_validator("task_artifacts", mode="before")
    @classmethod
    def explicit_artifact_flags(cls, value: object) -> object:
        if not isinstance(value, dict) or set(value) != {"inputs", "tests"}:
            raise ValueError(
                "must explicitly declare exactly metadata.task_artifacts.inputs/tests"
            )
        return value


class BenchmarkSourceRow(BaseModel):
    """Harbor five-key rows. Packaged ``tests/`` own the grader; Train invents no Factory fields."""

    model_config = ConfigDict(extra="forbid", strict=True)

    task_id: str
    world_id: NonEmptyString
    instruction: NonEmptyString
    requires_web_search: bool
    metadata: BenchmarkMetadata

    @field_validator("task_id")
    @classmethod
    def safe_task_id(cls, value: str) -> str:
        if not is_safe_task_id(value):
            raise ValueError("task_id must be a safe path component")
        return value


class SourceRow(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    task_id: str
    world_id: NonEmptyString
    instruction: str
    rubric: dict
    expected_deliverable: dict
    submission_transport: SubmissionTransport
    grading_evidence_mode: GradingEvidenceMode
    requires_web_search: bool
    metadata: SourceMetadata

    @field_validator("task_id")
    @classmethod
    def safe_task_id(cls, value: str) -> str:
        if not is_safe_task_id(value):
            raise ValueError("task_id must be a safe path component")
        return value

    @model_validator(mode="after")
    def grading_contract(self) -> Self:
        contract = (
            self.submission_transport,
            self.expected_deliverable.get("kind"),
            self.grading_evidence_mode,
        )
        if contract not in _SUPPORTED_GRADING_EVIDENCE_CONTRACTS:
            raise ValueError(f"incompatible grading evidence contract {contract!r}")
        return self


class FeatherTaskData(vf.TaskData):
    task_id: str
    task_name: str
    world_id: NonEmptyString
    domain: str | None = None
    source: Literal["factory", "benchmark"] = "factory"
    rubric: list[dict] = Field(default_factory=list)
    rubric_doc: dict = Field(default_factory=dict)
    expected_deliverable: dict = Field(default_factory=dict)
    submission_transport: SubmissionTransport | None = None
    grading_evidence_mode: GradingEvidenceMode | None = None
    requires_web_search: bool
    network_mode: NetworkMode
    environment_mode: Literal["shared", "separate"] | None = None
    workdir: str | None = None
    collect_hooks: tuple[CollectHook, ...] = ()
    artifacts: tuple[HarborArtifact, ...] = ()
    verifier_timeout_sec: float | None = None
    hf_dataset: NonEmptyString
    hf_revision: ImmutableRevision
    task_artifacts: TaskArtifacts = Field(default_factory=TaskArtifacts)
    source_metadata: TaskSourceMetadata = Field(default_factory=TaskSourceMetadata)
    eval_run_id: str | None = None
    e2b_snapshot_id: str | None = None
    # Producer stamp retained as snapshot provenance; no runtime freshness lookup.
    dataset_env_oid: str | None = None
    mcp_servers: list[McpServerSpec]
    rollout_group_id: str | None = None
    sampled_policy_version: int | str | None = None

    @model_validator(mode="after")
    def factory_uses_peer_verifier(self) -> Self:
        if self.source == "factory" and self.environment_mode is not None:
            raise ValueError(
                f"{self.hf_dataset}: Factory task {self.task_id} in world {self.world_id} "
                "requires the baked peer verifier; environment_mode must be unset"
            )
        return self


class FeatherTask(vf.Task[FeatherTaskData]):
    async def setup(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        del runtime
        if self.data.eval_run_id is not None:
            trace.info["eval_run_id"] = self.data.eval_run_id
        if self.data.rollout_group_id is not None:
            trace.info["rollout_group_id"] = self.data.rollout_group_id
        if self.data.sampled_policy_version is not None:
            trace.info["sampled_policy_version"] = self.data.sampled_policy_version
        trace.info["hf_revision"] = self.data.hf_revision

    @vf.reward
    async def harbor_grader_reward(self, trace: vf.Trace) -> float:
        return await harbor_grader_reward(self.data, trace)


class FeatherTasksetConfig(vf.TasksetConfig):
    id: Literal["feather_train"] = "feather_train"
    hf_dataset: NonEmptyString
    hf_revision: ImmutableRevision
    task_ids: list[str] | None = None
    expected_world_ids: list[NonEmptyString] | None = Field(default=None, min_length=1)
    max_tasks: int | None = None
    system_prompt: str | None = None


class FeatherTaskset(vf.Taskset[FeatherTask, FeatherTasksetConfig]):
    """Harbor-style world data with its staged ``/tests`` grader."""

    def load(self) -> Iterator[FeatherTask]:
        tasks = _load_tasks(
            self.config.hf_dataset,
            self.config.task_ids,
            self.config.max_tasks,
            hf_revision=self.config.hf_revision,
            system_prompt=self.config.system_prompt or CODING_SYSTEM_PROMPT,
            expected_world_ids=self.config.expected_world_ids,
        )
        for task in tasks:
            yield FeatherTask(task, self.config.task)


def _load_tasks(
    hf_dataset: str,
    task_ids: list[str] | None,
    max_tasks: int | None,
    *,
    hf_revision: str,
    system_prompt: str,
    expected_world_ids: list[str] | None = None,
) -> list[FeatherTaskData]:
    tasks = _load_selected_tasks(
        hf_dataset,
        task_ids,
        max_tasks,
        hf_revision=hf_revision,
    )
    if expected_world_ids is not None:
        unexpected = sorted({task.world_id for task in tasks} - set(expected_world_ids))
        if unexpected:
            raise ValueError(
                f"{hf_dataset}: selected tasks have unexpected world ID(s): {unexpected[:10]}; "
                f"expected one of {expected_world_ids}"
            )
    factory_rubrics = {
        task.task_id: _normalize_rubric(task.rubric)
        for task in tasks
        if isinstance(task, SourceRow)
    }
    bad_rubric = [task_id for task_id, rubric in factory_rubrics.items() if not rubric]
    if bad_rubric:
        raise ValueError(f"{hf_dataset}: task(s) have no usable rubric: {bad_rubric[:10]}")

    # Snapshot correctness belongs to the offline bake. Loading tasks does not
    # revalidate remote world hashes or make per-row freshness requests.
    e2b_snapshot_manifest = _load_snapshot_manifest(hf_dataset, hf_revision=hf_revision)
    harbor_spec_by_world: dict[str, HarborSpec] = {}

    loaded_tasks: list[FeatherTaskData] = []
    for idx, task in enumerate(tasks):
        if task.world_id in harbor_spec_by_world:
            harbor_spec = harbor_spec_by_world[task.world_id]
        else:
            harbor_spec = _load_world_harbor_spec(
                hf_dataset, task.world_id, hf_revision=hf_revision
            )
        declared_mcp = list(harbor_spec.mcp_servers)
        if isinstance(task, SourceRow):
            _validate_submission_mcp(
                declared_mcp,
                submission_transport=task.submission_transport,
                hf_dataset=hf_dataset,
                task_id=task.task_id,
            )
            deliverable = task.expected_deliverable
            target = _deliverable_target_path(deliverable)
            if target:
                deliverable = {**deliverable, "path": target}
            rubric = factory_rubrics[task.task_id]
            rubric_doc = task.rubric
            submission_transport = task.submission_transport
            grading_evidence_mode = task.grading_evidence_mode
            source: Literal["factory", "benchmark"] = "factory"
        else:
            deliverable = {}
            rubric = []
            rubric_doc = {}
            submission_transport = None
            grading_evidence_mode = None
            source = "benchmark"
        if task.world_id not in harbor_spec_by_world:
            _assert_world_compose(hf_dataset, task.world_id, declared_mcp, hf_revision=hf_revision)
            harbor_spec_by_world[task.world_id] = harbor_spec
        try:
            snapshot = SnapshotEntry.model_validate(e2b_snapshot_manifest.get(task.world_id))
        except ValueError as exc:
            raise ValueError(
                f"{hf_dataset}: world {task.world_id} has an invalid snapshot: {exc}"
            ) from exc
        loaded_tasks.append(
            FeatherTaskData(
                idx=idx,
                name=task.task_id,
                prompt=[{"role": "user", "content": task.instruction}],
                system_prompt=system_prompt,
                task_id=task.task_id,
                task_name=task.task_id,
                world_id=task.world_id,
                source=source,
                rubric=rubric,
                rubric_doc=rubric_doc,
                expected_deliverable=deliverable,
                submission_transport=submission_transport,
                grading_evidence_mode=grading_evidence_mode,
                requires_web_search=task.requires_web_search,
                network_mode=harbor_spec.network_mode,
                environment_mode=harbor_spec.environment_mode,
                workdir=harbor_spec.workdir,
                collect_hooks=harbor_spec.collect_hooks,
                artifacts=harbor_spec.artifacts,
                verifier_timeout_sec=harbor_spec.verifier_timeout_sec,
                hf_dataset=hf_dataset,
                hf_revision=hf_revision,
                task_artifacts=task.metadata.task_artifacts,
                source_metadata=_task_source_metadata(
                    task.metadata,
                    hf_dataset=hf_dataset,
                    hf_revision=hf_revision,
                ),
                e2b_snapshot_id=snapshot.id,
                dataset_env_oid=snapshot.dataset_env_oid,
                mcp_servers=declared_mcp,
            )
        )
    return loaded_tasks


def _load_selected_tasks(
    hf_dataset: str,
    task_ids: list[str] | None,
    max_tasks: int | None,
    *,
    hf_revision: str,
) -> list[SourceRow | BenchmarkSourceRow]:
    if max_tasks is not None and max_tasks < 0:
        raise ValueError(f"{hf_dataset}: max_tasks must be non-negative, got {max_tasks}")

    tasks_path = _download(hf_dataset, "tasks.jsonl", hf_revision=hf_revision)
    with open(tasks_path) as task_file:
        all_tasks = []
        for index, line in enumerate(task_file, start=1):
            if line.strip():
                try:
                    all_tasks.append(_parse_exported_row(line))
                except ValueError as exc:
                    raise ValueError(f"{hf_dataset}: tasks.jsonl row {index}: {exc}") from exc
    has_factory = any(isinstance(task, SourceRow) for task in all_tasks)
    has_benchmark = any(isinstance(task, BenchmarkSourceRow) for task in all_tasks)
    if has_factory and has_benchmark:
        raise ValueError(
            f"{hf_dataset}: tasks.jsonl mixes Factory rows with five-key Harbor rows"
        )

    tasks = all_tasks
    if task_ids is not None:
        wanted = set(task_ids)
        available = {task.task_id for task in all_tasks}
        missing = sorted(wanted - available)
        if missing:
            raise ValueError(
                f"{hf_dataset}: {len(missing)} requested task ID(s) not found: {missing[:10]}"
            )
        tasks = [t for t in tasks if t.task_id in wanted]
    if max_tasks is not None:
        tasks = tasks[:max_tasks]
    if not tasks:
        raise ValueError(f"{hf_dataset}: task selection returned no tasks")
    logger.info("feather_train: %d/%d tasks", len(tasks), len(all_tasks))
    return tasks


def _load_world_harbor_spec(
    hf_dataset: str,
    world_id: str,
    *,
    hf_revision: str,
) -> HarborSpec:
    """Download and parse a world’s MCP/network declarations."""
    try:
        toml_path = _download(
            hf_dataset,
            f"worlds/{world_id}/task.harbor.template.toml",
            hf_revision=hf_revision,
        )
    except EntryNotFoundError as exc:
        raise ValueError(
            f"{hf_dataset}: world {world_id} ships no task.harbor.template.toml; "
            "every Compose world must declare network_mode and MCP servers there"
        ) from exc

    with Path(toml_path).open("rb") as handle:
        template = tomllib.load(handle)
    verifier = template.get("verifier", {})
    environment = template.get("environment", {})
    dockerfile_path = None
    if (
        isinstance(verifier, dict)
        and verifier.get("environment_mode") in {"shared", "separate"}
        and isinstance(environment, dict)
        and environment.get("workdir") is None
    ):
        try:
            dockerfile_path = _download(
                hf_dataset, f"worlds/{world_id}/environment/Dockerfile", hf_revision=hf_revision
            )
        except EntryNotFoundError as exc:
            raise ValueError(
                f"{hf_dataset}: world {world_id} environment_mode requires workdir "
                "or an environment/Dockerfile WORKDIR"
            ) from exc
    spec = parse_harbor_spec(
        toml_path, where=f"{hf_dataset}: world {world_id}", dockerfile_path=dockerfile_path
    )
    logger.info(
        "feather_train: world %s declares %d MCP server(s), network_mode=%s",
        world_id,
        len(spec.mcp_servers),
        spec.network_mode,
    )
    return spec


def _validate_submission_mcp(
    mcp_servers: list[McpServerSpec],
    *,
    submission_transport: SubmissionTransport,
    hf_dataset: str,
    task_id: str,
) -> None:
    submission_servers = [server for server in mcp_servers if server.name == SUBMISSION_MCP_NAME]
    if len(submission_servers) > 1:
        raise ValueError(
            f"{hf_dataset}: task {task_id} declares multiple reserved "
            f"{SUBMISSION_MCP_NAME!r} MCP servers"
        )
    if submission_transport == "submit_tool" and not submission_servers:
        raise ValueError(
            f"{hf_dataset}: task {task_id} uses submit_tool but its world does not "
            f"declare the reserved {SUBMISSION_MCP_NAME!r} MCP server"
        )
    if submission_servers and not _is_valid_submission_mcp(submission_servers[0]):
        raise ValueError(
            f"{hf_dataset}: task {task_id} has an invalid reserved "
            f"{SUBMISSION_MCP_NAME!r} MCP descriptor"
        )


def _is_valid_submission_mcp(server: McpServerSpec) -> bool:
    endpoint = urlsplit(server.url)
    try:
        port = endpoint.port
    except ValueError:
        return False
    return (
        server.transport in {"streamable-http", "streamable_http"}
        and endpoint.scheme == "http"
        and endpoint.hostname == "submit-mcp"
        and port == 5000
        and endpoint.path == "/mcp"
        and not endpoint.username
        and not endpoint.password
        and not endpoint.query
        and not endpoint.fragment
    )


def _assert_world_compose(
    hf_dataset: str,
    world_id: str,
    declared_mcp: list[McpServerSpec],
    *,
    hf_revision: str,
) -> None:
    """Fail unless the world is a Compose world serving every MCP server it declares.

    Sidecars are not required in themselves: only ``submit_tool`` needs the submission server,
    and ``_validate_submission_mcp`` already enforces that. A ``path_manifest`` or
    ``text_response`` world legitimately ships ``main`` alone.

    What must hold is that the Harbor template and the Compose file agree — every declared MCP
    server needs a service to run in. Provisioning checks this too, but only once a sandbox is
    already up, so a mismatched export would otherwise cost a full spawn before failing.
    """
    try:
        compose_path = _download(
            hf_dataset,
            f"worlds/{world_id}/environment/docker-compose.yaml",
            hf_revision=hf_revision,
        )
    except EntryNotFoundError as exc:
        raise ValueError(
            f"{hf_dataset}: world {world_id} ships no environment/docker-compose.yaml; "
            "Train runs every world as a Compose stack"
        ) from exc

    with open(compose_path) as handle:
        compose = yaml.safe_load(handle)
    services = compose.get("services") if isinstance(compose, dict) else None
    if not isinstance(services, dict) or "main" not in services:
        raise ValueError(
            f"{hf_dataset}: world {world_id} docker-compose.yaml must define service 'main'"
        )
    # An MCP server's internal URL hostname identifies its Compose service.
    missing = sorted(
        {
            host
            for server in declared_mcp
            if (host := urlsplit(server.url).hostname) and host not in services
        }
    )
    if missing:
        raise ValueError(
            f"{hf_dataset}: world {world_id} declares MCP servers with no Compose service to "
            f"run them: {missing}"
        )


def _deliverable_target_path(deliverable: dict) -> str:
    """Absolute path under WORKSPACE_DIR where the agent must produce the deliverable.

    Edit-task paths are already under /filesystem; new_file / text_response tasks store
    a root-relative path (e.g. ``/proxy-x.md``), but the agent works in /filesystem — so
    map those into it. Returns "" when no path/filename is available.
    """
    path = deliverable.get("path") or ""
    if path == WORKSPACE_DIR or path.startswith(WORKSPACE_DIR + "/"):
        return path
    rel = path.lstrip("/") or deliverable.get("filename") or ""
    return f"{WORKSPACE_DIR}/{rel}" if rel else ""


def _parse_exported_row(line: str) -> SourceRow | BenchmarkSourceRow:
    payload = json.loads(line)
    if not isinstance(payload, dict):
        raise ValueError("row must be a JSON object")
    if set(payload) == BENCHMARK_SOURCE_KEYS:
        return BenchmarkSourceRow.model_validate(payload)
    return SourceRow.model_validate(payload)


def _normalize_rubric(rubric: dict) -> list[dict]:
    """Flatten the harbor rubric ``{id, sections[].criteria[]}`` to ``[{verifier_id, criteria}]``.

    The only shape the dataset ships; anything else yields ``[]`` and fails
    the load-time validation.
    """
    return [
        _criterion_to_verifier(criterion, section)
        for section in rubric.get("sections") or []
        for criterion in section.get("criteria") or []
    ]


def _criterion_to_verifier(criterion: dict, section: dict) -> dict:
    identity = {
        "criterion_id": criterion.get("id") or criterion.get("criterion_id"),
        "section_id": section.get("id") or section.get("section_id"),
        "section_name": section.get("name") or section.get("title"),
        "section_category": section.get("category") or section.get("rubric_category"),
        "rubric_category": criterion.get("rubric_category") or criterion.get("category"),
    }
    projection = {
        "verifier_id": identity["criterion_id"] or "",
        "criteria": criterion.get("requirement") or "",
        **identity,
    }
    return {key: value for key, value in projection.items() if value is not None}


def _task_source_metadata(
    metadata: SourceMetadata | BenchmarkMetadata,
    *,
    hf_dataset: str,
    hf_revision: str,
) -> TaskSourceMetadata:
    return TaskSourceMetadata(
        source_job_id=metadata.source_job_id,
        query_id=metadata.source_query_id,
        content_hash=metadata.task_snapshot_hash,
        dataset_id=metadata.dataset_id or hf_dataset,
        dataset_revision=metadata.dataset_revision or hf_revision,
    )


def _load_snapshot_manifest(hf_dataset: str, *, hf_revision: str) -> dict[str, object]:
    path = _download(hf_dataset, "snapshot_manifest.json", hf_revision=hf_revision)
    with open(path) as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"snapshot_manifest.json for {hf_dataset} must be a JSON object")
    return data


def _download(hf_dataset: str, filename: str, *, hf_revision: str) -> str:
    return hf_hub_download(
        hf_dataset,
        filename,
        repo_type="dataset",
        token=os.environ.get("HF_TOKEN"),
        revision=hf_revision,
    )
