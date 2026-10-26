"""World network policy, MCP descriptors, and official Harbor verifier modes."""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal, TypeAlias

from feather_harness.tools.mcp import McpServerSpec
from pydantic import BaseModel, ConfigDict

NetworkMode = Literal["public", "no-network"]
EnvironmentMode = Literal["shared", "separate"]

_WORKDIR_LINE = re.compile(r"^\s*WORKDIR\s+(\S+)\s*$", re.IGNORECASE)


class CollectHook(BaseModel):
    """One ``[[verifier.collect]]`` command, run in the agent container after the solver."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    command: str
    timeout_sec: float
    service: str | None = None


class ArtifactSpec(BaseModel):
    """A Harbor artifact collected from a named Compose service."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source: str
    service: Literal["main", "agent", "world"]


HarborArtifact: TypeAlias = str | ArtifactSpec


@dataclass(frozen=True, slots=True)
class HarborSpec:
    """Everything Train reads out of one world's Harbor template."""

    mcp_servers: tuple[McpServerSpec, ...]
    network_mode: NetworkMode
    environment_mode: EnvironmentMode | None = None
    workdir: str | None = None
    collect_hooks: tuple[CollectHook, ...] = ()
    artifacts: tuple[HarborArtifact, ...] = ()
    verifier_timeout_sec: float | None = None


def parse_harbor_spec(
    template_path: str | Path, *, where: str, dockerfile_path: str | Path | None = None
) -> HarborSpec:
    """Parse a world's Harbor template, raising on anything Train cannot honour.

    ``where`` identifies the world in error messages, so a bad export names itself.
    """
    template_path = Path(template_path)
    with template_path.open("rb") as handle:
        harbor = tomllib.load(handle)
    environment = harbor.get("environment")
    if not isinstance(environment, dict):
        raise ValueError(f"{where} Harbor template lacks [environment]")
    verifier = harbor.get("verifier")
    if verifier is None:
        verifier = {}
    if not isinstance(verifier, dict):
        raise ValueError(f"{where} [verifier] must be a table")
    environment_mode = _environment_mode(verifier, where=where)
    collect_hooks = _collect_hooks(verifier, where=where)
    artifacts = _artifacts(harbor, where=where)
    if environment_mode == "separate" and not artifacts:
        raise ValueError(f"{where} separate verifier requires artifacts")
    if environment_mode != "separate" and (collect_hooks or artifacts):
        raise ValueError(
            f"{where} [[verifier.collect]] and artifacts require environment_mode=separate"
        )
    spec = HarborSpec(
        mcp_servers=_mcp_servers(environment, where=where),
        network_mode=_network_mode(environment, where=where),
        environment_mode=environment_mode,
        workdir=_workdir(environment, where=where),
        collect_hooks=collect_hooks,
        artifacts=artifacts,
        verifier_timeout_sec=_timeout_sec(verifier, where=f"{where} [verifier]"),
    )
    if spec.environment_mode is not None and spec.workdir is None:
        # HF snapshot paths symlink into blobs; keep the snapshot's sibling layout.
        dockerfile = (
            Path(dockerfile_path)
            if dockerfile_path is not None
            else template_path.parent / "environment" / "Dockerfile"
        )
        if dockerfile.is_file():
            spec = replace(
                spec,
                workdir=parse_image_workdir(dockerfile.read_text(encoding="utf-8"), where=where),
            )
        else:
            raise ValueError(f"{where}: environment_mode requires workdir or a Dockerfile WORKDIR")
    return spec


def parse_image_workdir(dockerfile: str, *, where: str) -> str:
    """Last Dockerfile WORKDIR, which is the official Harbor agent cwd."""
    found = None
    for raw in dockerfile.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        match = _WORKDIR_LINE.match(line)
        if match is None:
            continue
        found = match.group(1).strip().strip("\"'")
    if found is None:
        raise ValueError(f"{where}: Dockerfile has no WORKDIR")
    return _absolute_workdir(found, where=where)


def _mcp_servers(environment: dict, *, where: str) -> tuple[McpServerSpec, ...]:
    try:
        servers = tuple(
            McpServerSpec.model_validate(entry)
            for entry in environment.get("mcp_servers") or []
        )
        if any(server.transport == "stdio" for server in servers):
            raise ValueError("baked worlds require HTTP or SSE MCP services")
        return servers
    except ValueError as exc:
        raise ValueError(f"{where} has an invalid MCP server: {exc}") from exc


def _network_mode(environment: dict, *, where: str) -> NetworkMode:
    if "allow_internet" in environment:
        raise ValueError(
            f"{where} uses legacy allow_internet; declare network_mode "
            f"('public' or 'no-network') instead"
        )
    if "network_mode" not in environment:
        raise ValueError(f"{where} Harbor template must declare network_mode")
    mode = environment["network_mode"]
    if mode == "allowlist":
        raise ValueError(
            f"{where} requests Harbor network_mode='allowlist', which Train cannot enforce"
        )
    if mode not in ("public", "no-network"):
        raise ValueError(f"{where} has unsupported network_mode {mode!r}")
    return mode


def _environment_mode(verifier: dict, *, where: str) -> EnvironmentMode | None:
    if "environment_mode" not in verifier:
        return None
    mode = verifier["environment_mode"]
    if mode not in ("shared", "separate"):
        raise ValueError(f"{where} [verifier].environment_mode must be shared or separate")
    return mode


def _collect_hooks(verifier: dict, *, where: str) -> tuple[CollectHook, ...]:
    raw = verifier.get("collect")
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise ValueError(f"{where} [[verifier.collect]] must be an array of tables")
    hooks: list[CollectHook] = []
    for index, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise ValueError(f"{where} [[verifier.collect]][{index}] must be a table")
        command = entry.get("command")
        if not isinstance(command, str) or not command.strip():
            raise ValueError(f"{where} [[verifier.collect]][{index}].command is required")
        timeout = entry.get("timeout_sec", 300)
        if not isinstance(timeout, int | float) or isinstance(timeout, bool) or timeout <= 0:
            raise ValueError(f"{where} [[verifier.collect]][{index}].timeout_sec must be positive")
        service = entry.get("service")
        if service is not None and service not in {"main", "agent", "world"}:
            raise ValueError(
                f"{where} [[verifier.collect]][{index}].service must be main, agent, or world, "
                f"got {service!r}"
            )
        extra = set(entry) - {"command", "timeout_sec", "service"}
        if extra:
            raise ValueError(f"{where} [[verifier.collect]][{index}] has unknown keys {sorted(extra)}")
        hooks.append(
            CollectHook(command=command, timeout_sec=float(timeout), service=service)
        )
    return tuple(hooks)


def _artifacts(harbor: dict, *, where: str) -> tuple[HarborArtifact, ...]:
    raw = harbor.get("artifacts")
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise ValueError(f"{where} artifacts must be a list of absolute paths or service sources")
    artifacts: list[HarborArtifact] = []
    paths: set[str] = set()
    basenames: set[str] = set()
    for index, item in enumerate(raw):
        if isinstance(item, str):
            artifact: HarborArtifact = item
            source = item
        elif isinstance(item, dict):
            try:
                artifact = ArtifactSpec.model_validate(item)
            except ValueError as exc:
                raise ValueError(f"{where} artifacts[{index}] is invalid: {exc}") from exc
            source = artifact.source
        else:
            raise ValueError(f"{where} artifacts[{index}] must be a path or service source")
        path = Path(source)
        if (
            not path.is_absolute()
            or source == "/"
            or source.endswith("/")
            or any(part in {".", ".."} for part in source.split("/"))
        ):
            raise ValueError(f"{where} artifacts[{index}].source must be an absolute file path")
        if source in paths or path.name in basenames:
            raise ValueError(f"{where} artifacts contain a duplicate path or basename: {source}")
        paths.add(source)
        basenames.add(path.name)
        artifacts.append(artifact)
    return tuple(artifacts)


def _workdir(environment: dict, *, where: str) -> str | None:
    value = environment.get("workdir")
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{where} [environment].workdir must be a string")
    return _absolute_workdir(value, where=where)


def _timeout_sec(table: dict, *, where: str) -> float | None:
    value = table.get("timeout_sec")
    if value is None:
        return None
    if not isinstance(value, int | float) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{where}.timeout_sec must be a positive number")
    return float(value)


def _absolute_workdir(value: str, *, where: str) -> str:
    workdir = value.strip()
    if not workdir.startswith("/") or workdir == "/":
        raise ValueError(f"{where}: workdir must be an absolute directory, got {value!r}")
    return workdir
