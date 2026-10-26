from pathlib import Path

import pytest

from feather_train.tasks.harbor_spec import ArtifactSpec, parse_harbor_spec, parse_image_workdir


def _write_world(tmp_path: Path, template: str, dockerfile: str | None = None) -> Path:
    world = tmp_path / "world"
    environment = world / "environment"
    environment.mkdir(parents=True)
    (world / "task.harbor.template.toml").write_text(template, encoding="utf-8")
    if dockerfile is not None:
        (environment / "Dockerfile").write_text(dockerfile, encoding="utf-8")
    return world / "task.harbor.template.toml"


def test_default_template_stays_peer_verifier(tmp_path):
    path = _write_world(
        tmp_path,
        '[environment]\nnetwork_mode = "no-network"\n',
    )
    spec = parse_harbor_spec(path, where="peer")
    assert spec.environment_mode is None
    assert spec.workdir is None
    assert spec.collect_hooks == ()
    assert spec.artifacts == ()


def test_mcp_descriptor_preserves_headers_deadline_and_error_policy(tmp_path):
    path = _write_world(
        tmp_path,
        '[environment]\nnetwork_mode = "public"\n'
        '[[environment.mcp_servers]]\nname = "research"\ntransport = "http"\n'
        'url = "http://research:5000/mcp/"\nrequest_timeout_seconds = 900\n'
        'raise_on_error_result = true\nheaders = {Authorization = "Bearer test"}\n',
    )
    server = parse_harbor_spec(path, where="research").mcp_servers[0]
    assert server.headers == {"Authorization": "Bearer test"}
    assert server.request_timeout_seconds == 900
    assert server.raise_on_error_result is True


def test_baked_world_rejects_stdio_mcp_descriptor(tmp_path):
    path = _write_world(
        tmp_path,
        '[environment]\nnetwork_mode = "public"\n'
        '[[environment.mcp_servers]]\nname = "local"\ntransport = "stdio"\n'
        'command = "python"\n',
    )
    with pytest.raises(ValueError, match="baked worlds require HTTP or SSE"):
        parse_harbor_spec(path, where="local")


@pytest.mark.parametrize("deadline", [None, 0, 120.5])
def test_mcp_response_timeout_is_loaded_from_world(tmp_path, deadline):
    timeout_line = "" if deadline is None else f"request_timeout_seconds = {deadline}\n"
    path = _write_world(
        tmp_path,
        '[environment]\nnetwork_mode = "public"\n'
        '[[environment.mcp_servers]]\nname = "research"\ntransport = "http"\n'
        'url = "http://research:5000/mcp/"\n' + timeout_line,
    )
    spec = parse_harbor_spec(path, where="research")
    assert spec.mcp_servers[0].request_timeout_seconds == (60 if deadline is None else deadline)


@pytest.mark.parametrize("deadline", ["-1", "nan", "inf", "true", '"60"'])
def test_invalid_mcp_response_timeout_names_the_world(tmp_path, deadline):
    path = _write_world(
        tmp_path,
        '[environment]\nnetwork_mode = "public"\n'
        '[[environment.mcp_servers]]\nname = "research"\ntransport = "http"\n'
        'url = "http://research:5000/mcp/"\n'
        f"request_timeout_seconds = {deadline}\n",
    )
    with pytest.raises(ValueError, match="research has an invalid MCP server"):
        parse_harbor_spec(path, where="research")


def test_shared_mode_reads_dockerfile_workdir(tmp_path):
    path = _write_world(
        tmp_path,
        '[environment]\nnetwork_mode = "public"\n\n[verifier]\nenvironment_mode = "shared"\n',
        "FROM debian\nWORKDIR /tmp\nWORKDIR /app\n",
    )
    spec = parse_harbor_spec(path, where="tb")
    assert spec.environment_mode == "shared"
    assert spec.workdir == "/app"


def test_shared_mode_reads_snapshot_sibling_of_symlinked_template(tmp_path):
    template = _write_world(
        tmp_path,
        '[environment]\nnetwork_mode = "public"\n[verifier]\nenvironment_mode = "shared"\n',
        "FROM debian\nWORKDIR /app\n",
    )
    blob = tmp_path / "blobs" / "template-content"
    blob.parent.mkdir()
    template.rename(blob)
    template.symlink_to(blob)
    assert parse_harbor_spec(template, where="snapshot").workdir == "/app"


def test_separate_requires_artifacts(tmp_path):
    path = _write_world(
        tmp_path,
        '[environment]\nnetwork_mode = "no-network"\nworkdir = "/app"\n\n'
        '[verifier]\nenvironment_mode = "separate"\n',
    )
    with pytest.raises(ValueError, match="artifacts"):
        parse_harbor_spec(path, where="swe")


def test_separate_accepts_artifact_without_collect_hook(tmp_path):
    path = _write_world(
        tmp_path,
        'artifacts = ["/logs/agent/trajectory.json"]\n\n'
        '[environment]\nnetwork_mode = "public"\nworkdir = "/app"\n\n'
        '[verifier]\nenvironment_mode = "separate"\n',
    )
    spec = parse_harbor_spec(path, where="apex")
    assert spec.collect_hooks == ()
    assert spec.artifacts == ("/logs/agent/trajectory.json",)


def test_separate_accepts_world_service_collect_and_artifact(tmp_path):
    path = _write_world(
        tmp_path,
        'artifacts = ["/logs/agent/trajectory.json", '
        '{source = "/logs/world/final_snapshot.tar.gz", service = "world"}]\n\n'
        '[environment]\nnetwork_mode = "public"\nworkdir = "/app"\n\n'
        '[verifier]\nenvironment_mode = "separate"\n\n'
        '[[verifier.collect]]\ncommand = "python /app/tools/collect_world_snapshot.py"\n'
        'service = "world"\ntimeout_sec = 660\n',
    )
    spec = parse_harbor_spec(path, where="apex")
    assert spec.collect_hooks[0].service == "world"
    assert spec.artifacts[1] == ArtifactSpec(
        source="/logs/world/final_snapshot.tar.gz", service="world"
    )


def test_artifact_basename_collision_is_rejected(tmp_path):
    path = _write_world(
        tmp_path,
        'artifacts = ["/logs/agent/state.json", '
        '{source = "/logs/world/state.json", service = "world"}]\n\n'
        '[environment]\nnetwork_mode = "public"\nworkdir = "/app"\n\n'
        '[verifier]\nenvironment_mode = "separate"\n',
    )
    with pytest.raises(ValueError, match="duplicate path or basename"):
        parse_harbor_spec(path, where="apex")


def test_unknown_artifact_service_is_rejected(tmp_path):
    path = _write_world(
        tmp_path,
        'artifacts = [{source = "/logs/state.json", service = "sidecar"}]\n\n'
        '[environment]\nnetwork_mode = "public"\nworkdir = "/app"\n\n'
        '[verifier]\nenvironment_mode = "separate"\n',
    )
    with pytest.raises(ValueError, match="artifacts\\[0\\] is invalid"):
        parse_harbor_spec(path, where="apex")


@pytest.mark.parametrize(
    "artifact",
    [
        '"/logs/output/"',
        '{source = "/logs/output/", service = "world"}',
    ],
)
def test_artifact_source_must_name_a_file(tmp_path, artifact):
    path = _write_world(
        tmp_path,
        f"artifacts = [{artifact}]\n\n"
        '[environment]\nnetwork_mode = "public"\nworkdir = "/app"\n\n'
        '[verifier]\nenvironment_mode = "separate"\n',
    )

    with pytest.raises(ValueError, match=r"artifacts\[0\]\.source must be an absolute file path"):
        parse_harbor_spec(path, where="apex")


def test_separate_parses_collect_and_artifacts(tmp_path):
    path = _write_world(
        tmp_path,
        "artifacts = [\"/logs/artifacts/model.patch\"]\n\n"
        '[environment]\nnetwork_mode = "no-network"\nworkdir = "/app"\n\n'
        "[verifier]\nenvironment_mode = \"separate\"\n"
        "timeout_sec = 1800\n\n"
        "[[verifier.collect]]\n"
        'command = "git diff --binary HEAD"\n'
        "timeout_sec = 300\n",
    )
    spec = parse_harbor_spec(path, where="swe")
    assert spec.environment_mode == "separate"
    assert spec.artifacts == ("/logs/artifacts/model.patch",)
    assert spec.collect_hooks[0].command == "git diff --binary HEAD"
    assert spec.collect_hooks[0].timeout_sec == 300
    assert spec.verifier_timeout_sec == 1800


def test_collect_without_separate_is_rejected(tmp_path):
    path = _write_world(
        tmp_path,
        "artifacts = [\"/logs/artifacts/model.patch\"]\n\n"
        '[environment]\nnetwork_mode = "public"\n\n'
        "[[verifier.collect]]\n"
        'command = "true"\n',
    )
    with pytest.raises(ValueError, match="environment_mode=separate"):
        parse_harbor_spec(path, where="bad")


def test_collect_service_must_be_declared_service(tmp_path):
    path = _write_world(
        tmp_path,
        "artifacts = [\"/logs/artifacts/model.patch\"]\n\n"
        '[environment]\nnetwork_mode = "no-network"\nworkdir = "/app"\n\n'
        "[verifier]\nenvironment_mode = \"separate\"\n\n"
        "[[verifier.collect]]\n"
        'command = "true"\n'
        'service = "sidecar"\n',
    )
    with pytest.raises(ValueError, match="main, agent, or world"):
        parse_harbor_spec(path, where="bad")


def test_unknown_environment_mode_is_rejected(tmp_path):
    path = _write_world(
        tmp_path,
        '[environment]\nnetwork_mode = "public"\n\n[verifier]\nenvironment_mode = "peer"\n',
    )
    with pytest.raises(ValueError, match="shared or separate"):
        parse_harbor_spec(path, where="bad")


def test_parse_image_workdir_uses_last_instruction():
    assert (
        parse_image_workdir("FROM debian\nWORKDIR /tmp\nWORKDIR /root\n", where="img")
        == "/root"
    )
