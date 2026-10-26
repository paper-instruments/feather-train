"""Loading exported tasks preserves producer text and uses one manifest per dataset."""

from __future__ import annotations

import json
from collections import Counter

import pytest
from pydantic import ValidationError

pytest.importorskip("verifiers.v1")

from feather_train.tasks import taskset
from feather_train.tasks.taskset import FeatherTaskData, FeatherTaskset, FeatherTasksetConfig

REVISION = "a" * 40


@pytest.mark.parametrize("mode", ["shared", "separate"])
def test_factory_task_boundary_rejects_official_verifier_modes(mode):
    data = dict(
        idx=0, task_id="task", task_name="task", world_id="world", source="factory",
        requires_web_search=False, network_mode="public", environment_mode=mode,
        hf_dataset="test/tasks", hf_revision=REVISION, mcp_servers=[], prompt="solve",
    )
    with pytest.raises(ValidationError, match="Factory task task in world world.*peer verifier"):
        FeatherTaskData(**data)
    benchmark = FeatherTaskData(**{**data, "source": "benchmark"})
    assert benchmark.environment_mode == mode
    factory = FeatherTaskData(**{**data, "environment_mode": None})
    assert factory.environment_mode is None


@pytest.mark.parametrize("dataset", [{}, {"hf_dataset": ""}, {"hf_dataset": "   "}])
def test_requires_explicit_nonempty_dataset(dataset):
    with pytest.raises(ValidationError) as exc_info:
        FeatherTasksetConfig(hf_revision=REVISION, **dataset)
    assert exc_info.value.errors()[0]["loc"] == ("hf_dataset",)


@pytest.fixture
def exported_dataset(tmp_path, monkeypatch):
    instruction = "  Follow the exported instruction exactly.\n\n"
    rows = [
        {
            "task_id": name,
            "world_id": "world",
            "instruction": instruction,
            "rubric": {
                "sections": [
                    {
                        "id": "s",
                        "criteria": [
                            {"id": "c", "requirement": "Correct answer"},
                        ],
                    }
                ]
            },
            "expected_deliverable": {"kind": "text_response"},
            "submission_transport": "text_response",
            "grading_evidence_mode": "direct_text",
            "requires_web_search": False,
            "metadata": {"resources": {}},
        }
        for name in ("one", "two")
    ]
    files = {
        "tasks.jsonl": "\n".join(json.dumps(row) for row in rows),
        "snapshot_manifest.json": json.dumps(
            {
                "world": {"id": "snapshot", "kind": "compose", "dataset_env_oid": "trusted-stamp"},
            }
        ),
        "worlds/world/task.harbor.template.toml": '[environment]\nnetwork_mode="no-network"\n',
        "worlds/world/environment/docker-compose.yaml": "services:\n  main:\n    image: test\n",
    }
    requests = []

    def download(repo_id, filename, **kwargs):
        assert repo_id == "test/tasks"
        assert kwargs["revision"] == REVISION
        requests.append(filename)
        path = tmp_path / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(files[filename])
        return str(path)

    monkeypatch.setattr(taskset, "hf_hub_download", download)
    return rows, files, requests


def load(**kwargs):
    return list(
        FeatherTaskset(
            FeatherTasksetConfig(
                hf_dataset="test/tasks",
                hf_revision=REVISION,
                **kwargs,
            )
        ).load()
    )


def test_preserves_instructions_and_loads_shared_world_once(exported_dataset):
    rows, files, requests = exported_dataset
    tasks = load()
    assert [task.data.task_id for task in tasks] == ["one", "two"]
    for task in tasks:
        assert task.data.system_prompt == taskset.CODING_SYSTEM_PROMPT
        assert len(task.data.prompt) == 1
        assert task.data.prompt[0].role == "user"
        assert task.data.prompt[0].content == rows[0]["instruction"]
        assert task.data.e2b_snapshot_id == "snapshot"
        assert task.data.dataset_env_oid == "trusted-stamp"
        assert not task.data.task_artifacts.inputs
    assert Counter(requests) == Counter(files.keys())


@pytest.mark.parametrize("mode", ["shared", "separate"])
def test_factory_world_cannot_override_peer_verifier_in_task_loading(exported_dataset, mode):
    _, files, _ = exported_dataset
    files["worlds/world/task.harbor.template.toml"] = (
        ('artifacts=["/logs/artifacts/answer.txt"]\n' if mode == "separate" else "")
        + '[environment]\nnetwork_mode="no-network"\nworkdir="/app"\n'
        + f'[verifier]\nenvironment_mode="{mode}"\n'
    )
    with pytest.raises(ValueError, match="test/tasks: Factory task one in world world.*peer verifier"):
        load()


@pytest.mark.parametrize("explicit_workdir", [False, True])
def test_official_world_downloads_dockerfile_only_for_workdir_fallback(
    tmp_path, monkeypatch, explicit_workdir
):
    requested = []
    template = '[environment]\nnetwork_mode="public"\n'
    if explicit_workdir:
        template += 'workdir="/explicit"\n'
    template += '[verifier]\nenvironment_mode="shared"\n'
    contents = {
        "worlds/world/task.harbor.template.toml": template,
        "worlds/world/environment/Dockerfile": "FROM debian\nWORKDIR /app\n",
    }

    def download(repo_id, filename, **kwargs):
        requested.append(filename)
        blob = tmp_path / ("template-blob" if filename.endswith("toml") else "dockerfile-blob")
        blob.write_text(contents[filename])
        path = tmp_path / "snapshot" / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.symlink_to(blob)
        return str(path)

    monkeypatch.setattr(taskset, "hf_hub_download", download)
    spec = taskset._load_world_harbor_spec("test/tasks", "world", hf_revision=REVISION)
    assert spec.workdir == ("/explicit" if explicit_workdir else "/app")
    assert requested == (
        ["worlds/world/task.harbor.template.toml"]
        if explicit_workdir
        else list(contents)
    )


def test_expected_world_guard_rejects_historical_artifact_before_snapshot_or_world_reads(exported_dataset):
    rows, files, requests = exported_dataset
    files["tasks.jsonl"] = json.dumps({
        "task_id": "one", "world_id": "bigfinance-public-feather",
        "instruction": rows[0]["instruction"], "requires_web_search": True,
        "metadata": {"task_artifacts": {"inputs": False, "tests": True}},
    })
    with pytest.raises(ValueError, match="unexpected world ID.*bigfinance-public-feather"):
        load(expected_world_ids=["bigfinance-public-feather-protected-v2"])
    assert requests == ["tasks.jsonl"]


def test_expected_world_guard_accepts_protected_rows_and_preserves_snapshot_mapping(exported_dataset):
    rows, files, requests = exported_dataset
    world = "bigfinance-public-feather-protected-v2"
    files["tasks.jsonl"] = json.dumps({
        "task_id": "one", "world_id": world, "instruction": rows[0]["instruction"],
        "requires_web_search": True,
        "metadata": {"task_artifacts": {"inputs": False, "tests": True}},
    })
    files["snapshot_manifest.json"] = json.dumps({world: {"id": "protected-snapshot", "kind": "compose"}})
    files[f"worlds/{world}/task.harbor.template.toml"] = files["worlds/world/task.harbor.template.toml"]
    files[f"worlds/{world}/environment/docker-compose.yaml"] = files["worlds/world/environment/docker-compose.yaml"]
    [task] = load(expected_world_ids=[world])
    assert task.data.world_id == world and task.data.e2b_snapshot_id == "protected-snapshot"
    assert task.data.prompt[0].content == rows[0]["instruction"].strip()
    assert requests == ["tasks.jsonl", "snapshot_manifest.json", f"worlds/{world}/task.harbor.template.toml",
                        f"worlds/{world}/environment/docker-compose.yaml"]


def test_expected_world_guard_checks_all_selected_rows_but_respects_task_selection(exported_dataset):
    rows, files, requests = exported_dataset
    rows[1]["world_id"] = "historical"
    files["tasks.jsonl"] = "\n".join(json.dumps(row) for row in rows)
    with pytest.raises(ValueError, match="unexpected world ID.*historical"):
        load(expected_world_ids=["world"])
    assert requests == ["tasks.jsonl"]
    requests.clear()
    [task] = load(task_ids=["one"], expected_world_ids=["world", "another-protected-world"])
    assert task.data.task_id == "one" and task.data.e2b_snapshot_id == "snapshot"


@pytest.mark.parametrize("worlds", [[], [""], ["   "], [False]])
def test_expected_world_guard_requires_explicit_nonempty_world_ids(worlds):
    with pytest.raises(ValidationError):
        FeatherTasksetConfig(hf_dataset="test/tasks", hf_revision=REVISION, expected_world_ids=worlds)


def test_environment_routes_expected_world_ids_to_taskset_before_snapshot_loading(exported_dataset):
    from feather_train.environment import load_environment

    _, _, requests = exported_dataset
    env = load_environment(hf_dataset="test/tasks", hf_revision=REVISION,
                           expected_world_ids=["protected"], enable_braintrust=False)
    assert env.taskset.config.expected_world_ids == ["protected"]
    with pytest.raises(ValueError, match="unexpected world ID"):
        list(env.taskset.load())
    assert requests == ["tasks.jsonl"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("submission_transport", "unknown"),
        ("requires_web_search", "false"),
        ("grading_evidence_mode", "submit_tool"),
        ("task_id", "../private"),
    ],
)
def test_rejects_bad_producer_contract_before_loading_manifest(exported_dataset, field, value):
    rows, files, requests = exported_dataset
    rows[0][field] = value
    files["tasks.jsonl"] = "\n".join(json.dumps(row) for row in rows)
    with pytest.raises(ValueError):
        load()
    assert requests == ["tasks.jsonl"]


def test_rejects_duplicate_task_ids_before_loading_manifest(exported_dataset):
    rows, files, requests = exported_dataset
    rows[1]["task_id"] = rows[0]["task_id"]
    files["tasks.jsonl"] = "\n".join(json.dumps(row) for row in rows)

    with pytest.raises(ValueError, match=r"duplicate task ID.*one"):
        load()

    assert requests == ["tasks.jsonl"]


def test_checks_requested_ids_before_limiting_tasks(exported_dataset):
    with pytest.raises(ValueError, match="requested task ID"):
        load(task_ids=["one", "missing"], max_tasks=1)


@pytest.mark.parametrize("entry", [None, "legacy-template", {"id": "snapshot", "kind": "other"}])
def test_requires_baked_compose_snapshot(exported_dataset, entry):
    _, files, _ = exported_dataset
    files["snapshot_manifest.json"] = json.dumps({"world": entry})
    with pytest.raises(ValueError, match="snapshot"):
        load()


def test_keeps_reserved_submission_descriptor_and_validates_service(exported_dataset):
    rows, files, _ = exported_dataset
    rows[0]["submission_transport"] = "submit_tool"
    rows[0]["grading_evidence_mode"] = "submit_tool"
    files["tasks.jsonl"] = json.dumps(rows[0])
    files["worlds/world/task.harbor.template.toml"] += (
        '[[environment.mcp_servers]]\nname="submission"\n'
        'transport="streamable-http"\nurl="http://submit-mcp:5000/mcp"\n'
    )
    with pytest.raises(ValueError, match="no Compose service"):
        load()
    files["worlds/world/environment/docker-compose.yaml"] += "  submit-mcp:\n    image: submit\n"
    task = load()[0]
    assert task.data.mcp_servers[0].name == "submission"
    restored = type(task.data).model_validate_json(task.data.model_dump_json())
    assert restored.mcp_servers == task.data.mcp_servers


@pytest.mark.parametrize(
    "metadata",
    [
        {},
        {"resources": {}, "source_job_id": "partial"},
        {"resources": {}, "task_artifacts": {"inputs": "false"}},
        {"resources": {}, "query_id": "unsupported-alias"},
    ],
)
def test_rejects_invalid_metadata_before_manifest(exported_dataset, metadata):
    rows, files, requests = exported_dataset
    rows[0]["metadata"] = metadata
    files["tasks.jsonl"] = json.dumps(rows[0])
    with pytest.raises(ValueError):
        load()
    assert requests == ["tasks.jsonl"]


def test_preserves_artifacts_provenance_and_rubric(exported_dataset):
    rows, files, _ = exported_dataset
    rows[0]["metadata"] = {
        "resources": {},
        "source_job_id": "job",
        "source_query_id": "query",
        "task_snapshot_hash": "hash",
        "task_artifacts": {"inputs": True},
    }
    files["tasks.jsonl"] = json.dumps(rows[0])
    task = load()[0].data
    assert task.task_artifacts.inputs
    assert not task.task_artifacts.tests
    assert task.source_metadata.model_dump() == {
        "source_job_id": "job",
        "query_id": "query",
        "content_hash": "hash",
        "dataset_id": "test/tasks",
        "dataset_revision": REVISION,
    }
    assert task.rubric == [
        {
            "verifier_id": "c",
            "criteria": "Correct answer",
            "criterion_id": "c",
            "section_id": "s",
        }
    ]


@pytest.mark.parametrize(
    "transport,kind,mode",
    [
        ("path_manifest", "new_file", "deliverables_folder"),
        ("path_manifest", "edit", "deliverables_folder"),
        ("path_manifest", "edit", "in_place_edit"),
        ("text_response", "text_response", "direct_text"),
        ("submit_tool", "new_file", "submit_tool"),
        ("submit_tool", "text_response", "submit_tool"),
        ("submit_tool", "edit", "in_place_edit"),
        ("snapshot_diff", "new_file", "world_state_diff"),
        ("snapshot_diff", "edit", "world_state_diff"),
    ],
)
def test_accepts_producer_grading_contracts(exported_dataset, transport, kind, mode):
    rows, files, _ = exported_dataset
    rows[0].update(
        submission_transport=transport,
        expected_deliverable={"kind": kind},
        grading_evidence_mode=mode,
    )
    files["tasks.jsonl"] = json.dumps(rows[0])
    if transport == "submit_tool":
        files["worlds/world/task.harbor.template.toml"] += (
            '[[environment.mcp_servers]]\nname="submission"\n'
            'transport="streamable-http"\nurl="http://submit-mcp:5000/mcp"\n'
        )
        files["worlds/world/environment/docker-compose.yaml"] += (
            "  submit-mcp:\n    image: submit\n"
        )
    task = load()[0].data
    assert task.submission_transport == transport
    assert task.expected_deliverable["kind"] == kind
    assert task.grading_evidence_mode == mode


@pytest.mark.parametrize("aliases", [False, True], ids=["canonical", "legacy-aliases"])
def test_display_and_grading_agree_on_rubric_identity(exported_dataset, aliases):
    from feather_train.grading.report import report_to_criteria

    rows, files, _ = exported_dataset
    section = {
        "section_id": "s",
        "title": "Accuracy",
        "rubric_category": "quality",
        "criteria": [
            {"criterion_id": "c", "category": "correctness", "requirement": "Correct answer"}
        ],
    }
    if not aliases:
        # Canonical keys win when both spellings are present.
        section.update(
            id="s",
            name="Accuracy",
            category="quality",
            section_id="ignored",
            title="ignored",
            rubric_category="ignored",
        )
        section["criteria"][0].update(
            id="c", rubric_category="correctness", criterion_id="ignored", category="ignored"
        )
    rows[0]["rubric"] = {"sections": [section]}
    files["tasks.jsonl"] = json.dumps(rows[0])
    data = load()[0].data
    identity = {
        "criterion_id": "c",
        "section_id": "s",
        "section_name": "Accuracy",
        "section_category": "quality",
        "rubric_category": "correctness",
    }
    assert data.rubric == [{"verifier_id": "c", "criteria": "Correct answer", **identity}]
    (grade,) = report_to_criteria([{"criterion_id": "c", "verdict": "MET"}], rubric=data.rubric)
    assert {key: grade[key] for key in identity} == identity
    assert grade["criterion_resolution_state"] == "resolved"


def test_cached_world_still_validates_each_task_submission_contract(exported_dataset):
    rows, files, requests = exported_dataset
    rows[1].update(submission_transport="submit_tool", grading_evidence_mode="submit_tool")
    files["tasks.jsonl"] = "\n".join(json.dumps(row) for row in rows)
    with pytest.raises(ValueError, match="task two uses submit_tool"):
        load()
    assert Counter(requests) == Counter(files.keys())


def test_invalid_later_world_fails_before_yielding_any_task(exported_dataset):
    rows, files, _ = exported_dataset
    rows[1]["world_id"] = "invalid-world"
    files["tasks.jsonl"] = "\n".join(json.dumps(row) for row in rows)
    files["worlds/invalid-world/task.harbor.template.toml"] = (
        '[environment]\nnetwork_mode="allowlist"\n'
    )
    tasks = FeatherTaskset(
        FeatherTasksetConfig(hf_dataset="test/tasks", hf_revision=REVISION)
    ).load()
    with pytest.raises(ValueError, match="allowlist"):
        next(tasks)


def test_five_key_rows_load_without_factory_fields(exported_dataset):
    _, files, requests = exported_dataset
    files["tasks.jsonl"] = json.dumps(
        {
            "task_id": "one",
            "world_id": "world",
            "instruction": "Solve the packaged world.",
            "requires_web_search": False,
            "metadata": {
                "task_artifacts": {"inputs": False, "tests": False},
                "domain": "finance",
            },
        }
    )
    files["tasks/one/tests/source_task.json"] = json.dumps({"unused": True})
    [task] = load()
    assert task.data.source == "benchmark"
    assert task.data.submission_transport is None
    assert task.data.grading_evidence_mode is None
    assert task.data.expected_deliverable == {}
    assert task.data.rubric == []
    assert task.data.rubric_doc == {}
    assert task.data.task_artifacts.tests is False
    assert "tasks/one/tests/source_task.json" not in requests


def test_five_key_rows_require_explicit_artifact_flags(exported_dataset):
    _, files, requests = exported_dataset
    files["tasks.jsonl"] = json.dumps(
        {
            "task_id": "one",
            "world_id": "world",
            "instruction": "Solve the packaged world.",
            "requires_web_search": False,
            "metadata": {"resources": []},
        }
    )
    with pytest.raises(ValueError):
        load()
    assert requests == ["tasks.jsonl"]


def test_rejects_mixed_factory_and_five_key_rows(exported_dataset):
    rows, files, requests = exported_dataset
    files["tasks.jsonl"] = "\n".join(
        [
            json.dumps(rows[0]),
            json.dumps(
                {
                    "task_id": "two",
                    "world_id": "world",
                    "instruction": "Solve the packaged world.",
                    "requires_web_search": False,
                    "metadata": {"task_artifacts": {"inputs": False, "tests": True}},
                }
            ),
        ]
    )
    with pytest.raises(ValueError, match="mixes Factory rows"):
        load()
    assert requests == ["tasks.jsonl"]
