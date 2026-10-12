"""Resume baked E2B Compose worlds and isolate grading from agent execution."""

from __future__ import annotations

import logging
import shlex
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import TYPE_CHECKING
from urllib.parse import urlsplit, urlunsplit

import yaml
from feather_harness.tools.mcp import McpServerSpec
from verifiers.v1 import SandboxError

from feather_train.errors import NonRetryableHarnessError
from feather_train.sandbox.paths import (
    BAKED_TESTS_DIR,
    STATE_EXPORT_TOKEN_ENV,
    STATE_EXPORT_TOKEN_PATH,
    WORKSPACE_DIR,
)
from feather_train.tasks.harbor_spec import (
    ArtifactSpec,
    CollectHook,
    EnvironmentMode,
    HarborArtifact,
)

if TYPE_CHECKING:
    from feather_train.sandbox.lifecycle import CommandOutput, SandboxIO

logger = logging.getLogger(__name__)

ENV_DIR = "/opt/apex_world"
WORLD_COMPOSE_FILE = f"{ENV_DIR}/docker-compose.yaml"
COMPOSE_FILE = f"{ENV_DIR}/compose.train.yaml"
COMPOSE_PROJECT = "apex"
AGENT_CONTAINER = "agent"
VERIFIER_CONTAINER = "verifier"
SEPARATE_VERIFIER_IMAGE = "feather-separate-verifier"
MCP_BASE_PORT = 5000
WORKSPACE_MOUNT_ENV = "HARBOR_WORKSPACE_MOUNT"
WORKSPACE_WRITE_ACCESS_ENV = "HARBOR_WORKSPACE_WRITE_ACCESS"
VERIFIER_NETWORK_ACCESS_ENV = "HARBOR_VERIFIER_NETWORK_ACCESS"
STATE_INITIALIZE_PATH_ENV = "APP_STATE_INITIALIZE_PATH"
STATE_FINALIZE_PATH_ENV = "APP_STATE_FINALIZE_PATH"
STATE_HOOK_TIMEOUT_SECONDS = 3600
SUBMISSION_VOLUME = "submission-state"
SUBMISSION_VOLUME_MOUNT = "/submission-state"
SUBMISSION_SERVICE = "submit-mcp"
IPV6_LINK_LOCAL_CIDR = "fe80::/10"


@dataclass(frozen=True, slots=True)
class StateExportSpec:
    service: str
    port: int
    path: str = "/state-export"
    initialize_path: str | None = None
    finalize_path: str | None = None


@dataclass(frozen=True, slots=True)
class VerifierNetworkSpec:
    service: str
    network: str


@dataclass(frozen=True, slots=True)
class ComposeRuntime:
    mcp_servers: list[McpServerSpec]
    state_export: StateExportSpec | None
    verifier_network: VerifierNetworkSpec | None = None


async def resume_compose_stack(
    session: SandboxIO,
    *,
    mcp_servers: list[McpServerSpec],
    require_verifier: bool = True,
) -> ComposeRuntime:
    """Require the baked stack to be alive and rewrite MCP URLs for this sandbox.

    Networking and host-side MCP proxies are installed before snapshotting. A warm
    restore checks them without rebuilding or silently changing the task world.
    """
    world_text = await _sh(
        session, f"cat {WORLD_COMPOSE_FILE}", operation="read world Compose config", check=True
    )
    override_text = await _sh(
        session, f"cat {COMPOSE_FILE}", operation="read Compose override", check=True
    )
    override = yaml.safe_load(override_text)
    port_by_service = _ports_from_override(override)
    public_host_ports = _public_host_ports_from_override(override)
    endpoint_by_service = _mcp_endpoints(mcp_servers)
    world = yaml.safe_load(world_text)
    services = _services_from_world(world)
    _validate_mcp_internal_ports(services, endpoint_by_service)
    _validate_submission_volume(world, services, mcp_servers)
    state_export = _state_export_spec(services)
    verifier_network = _verifier_network_spec(services)
    _validate_workspace_mounts(services, endpoint_by_service)
    _validate_workspace_write_access(services, endpoint_by_service)
    if state_export is not None and verifier_network is not None:
        raise NonRetryableHarnessError(
            "Harbor Compose must not combine structured-state export with verifier network access"
        )
    _validate_agent_networks(services, endpoint_by_service)

    await _assert_live_stack(session)
    await _block_agent_host_ports(session, public_host_ports)

    await _wait_for_ports(session, port_by_service, endpoint_by_service)
    if require_verifier:
        await assert_verifier_container_running(session)
    return ComposeRuntime(
        _mcp_urls(session, port_by_service, mcp_servers),
        state_export,
        verifier_network,
    )


async def assert_agent_container_survived(session: SandboxIO) -> None:
    """Fail the attempt if the agent's container did not survive the rollout.

    The agent's shell is a ``docker exec`` into this container, and that transport fails with exit
    1 — the same code ``grep`` returns on no match — so a dead container is indistinguishable from
    an ordinary failed command and reaches the model as a normal observation. Raising here keeps
    the attempt out of training.

    ``RestartCount`` matters as much as ``Running``: a restarted container is up again with
    everything the agent did inside it gone.
    """
    result = await _inspect_container(session, AGENT_CONTAINER)
    fields = result.stdout.split()
    if result.exit_code != 0 or len(fields) != 3:
        detail = result.stderr or result.stdout
        raise SandboxError(
            f"feather: could not inspect the agent container after the rollout; the sandbox or "
            f"its docker daemon did not survive: {detail[-300:]!r}"
        )
    running, exit_code, restarts = fields
    if running != "true" or restarts != "0":
        raise SandboxError(
            f"feather: the agent container did not survive the rollout "
            f"(running={running} exit_code={exit_code} restarts={restarts}); "
            "the agent's shell transport failed, so this attempt cannot be trained on"
        )


async def assert_verifier_container_running(session: SandboxIO) -> None:
    """Fail closed when the baked peer verifier is missing from a warm snapshot."""
    result = await _inspect_container(session, VERIFIER_CONTAINER)
    fields = result.stdout.split()
    detail = result.stderr or result.stdout
    if result.exit_code != 0:
        # A successful listing distinguishes a missing baked container from a daemon failure.
        listed = await session.exec(
            "docker container ls -a --format '{{.Names}}'",
            shell=True,
            timeout=200,
        )
        if listed.exit_code == 0 and VERIFIER_CONTAINER not in listed.stdout.splitlines():
            raise NonRetryableHarnessError(
                f"feather: baked verifier container {VERIFIER_CONTAINER!r} is missing; re-bake the world snapshot"
            )
        raise SandboxError(f"feather: could not inspect verifier container: {detail[-300:]!r}")
    if len(fields) != 3:
        raise SandboxError(f"feather: could not inspect verifier container: {detail[-300:]!r}")
    running, exit_code, restarts = fields
    if running != "true" or restarts != "0":
        raise SandboxError(
            f"feather: baked verifier container {VERIFIER_CONTAINER!r} did not survive "
            f"(running={running} exit_code={exit_code} restarts={restarts}); "
            "the verifier is not healthy"
        )


async def _inspect_container(session: SandboxIO, container: str) -> CommandOutput:
    return await session.exec(
        "docker inspect --format "
        "'{{.State.Running}} {{.State.ExitCode}} {{.RestartCount}}' " + shlex.quote(container),
        shell=True,
        timeout=200,
    )


async def prepare_compose_grading(
    session: SandboxIO,
    state_export: StateExportSpec | None,
    *,
    verifier_network: VerifierNetworkSpec | None = None,
    mount_submission_volume: bool = False,
    environment_mode: EnvironmentMode | None = None,
    collect_hooks: tuple[CollectHook, ...] = (),
    artifacts: tuple[HarborArtifact, ...] = (),
) -> None:
    """Isolate grading. Peer-verifier worlds remove ``main``; shared worlds keep it."""
    if environment_mode == "shared":
        await _copy_baked_tests_into_container(session, AGENT_CONTAINER)
        return
    if environment_mode == "separate":
        await _prepare_separate_verifier(session, collect_hooks, artifacts)
        return
    command = _compose_command()
    await _sh(
        session,
        f"{command} stop main 2>&1 && {command} rm -f main 2>&1",
        300,
        operation="stop agent container",
        check=True,
    )
    if state_export is not None and state_export.finalize_path is not None:
        await invoke_state_hook(session, state_export, state_export.finalize_path)
    await assert_verifier_container_running(session)

    if mount_submission_volume:
        await _sh(
            session,
            f"docker exec {VERIFIER_CONTAINER} test -d {shlex.quote(SUBMISSION_VOLUME_MOUNT)}",
            operation="check submission volume",
            check=True,
        )

    if state_export is not None:
        service = shlex.quote(state_export.service)
        inspect_env = "{{range .Config.Env}}{{println .}}{{end}}"
        inspect_networks = (
            "{{range $name, $_ := .NetworkSettings.Networks}}{{println $name}}{{end}}"
        )
        await _sh(
            session,
            f'state_container="$({command} ps -q {service})" && '
            'test -n "$state_container" && '
            f'state_token="$(docker inspect --format '
            f"'{inspect_env}' \"$state_container\" | "
            f"sed -n 's/^{STATE_EXPORT_TOKEN_ENV}=//p')\" && "
            'test -n "$state_token" && '
            f'state_network="$(docker inspect --format '
            f'\'{inspect_networks}\' "$state_container" | head -n 1)" && '
            'test -n "$state_network" && '
            f'docker network disconnect "$state_network" {VERIFIER_CONTAINER} '
            f">/dev/null 2>&1 || true; "
            f'docker network connect "$state_network" {VERIFIER_CONTAINER} && '
            f'printf %s "$state_token" | docker exec -i {VERIFIER_CONTAINER} '
            f"sh -c 'cat > {STATE_EXPORT_TOKEN_PATH}'",
            120,
            operation="configure verifier state export",
            check=True,
        )

    if verifier_network is not None:
        runtime_network = shlex.quote(f"{COMPOSE_PROJECT}_{verifier_network.network}")
        await _sh(
            session,
            f"docker network disconnect {runtime_network} {VERIFIER_CONTAINER} "
            f">/dev/null 2>&1 || true; "
            f"docker network connect {runtime_network} {VERIFIER_CONTAINER}",
            120,
            operation="connect verifier network",
            check=True,
        )


def _validate_submission_volume(world, services: dict[str, dict], mcp_servers) -> None:
    violations = _submission_volume_violations(world, services, mcp_servers)
    if violations:
        raise NonRetryableHarnessError(
            f"Harbor Compose declares invalid authored volumes: {violations}"
        )


def _submission_volume_violations(world, services: dict[str, dict], mcp_servers) -> list[str]:
    volumes = world.get("volumes")
    has_descriptor = any(server.name == "submission" for server in mcp_servers)
    has_service = SUBMISSION_SERVICE in services
    has_volume = isinstance(volumes, dict) and SUBMISSION_VOLUME in volumes
    contract_present = has_descriptor or has_service or has_volume

    if not contract_present:
        violations = ["volumes"] if volumes else []
        violations.extend(
            f"{service_name}.volumes"
            for service_name, service in services.items()
            if service.get("volumes")
        )
        return violations

    violations = []
    if not has_descriptor:
        violations.append("submission MCP descriptor")
    if not has_service:
        violations.append(SUBMISSION_SERVICE)
    if not isinstance(volumes, dict) or set(volumes) != {SUBMISSION_VOLUME}:
        violations.append("volumes")
    elif volumes[SUBMISSION_VOLUME] != {}:
        violations.append(f"volumes.{SUBMISSION_VOLUME}")

    for service_name, service in services.items():
        mounts = service.get("volumes") or []
        if service_name == SUBMISSION_SERVICE:
            if len(mounts) != 1 or not _is_submission_volume_mount(mounts[0]):
                violations.append(f"{service_name}.volumes")
        elif mounts:
            violations.append(f"{service_name}.volumes")
    return violations


def _is_submission_volume_mount(mount) -> bool:
    if isinstance(mount, str):
        parts = mount.split(":")
        return (
            len(parts) in {2, 3}
            and parts[:2] == [SUBMISSION_VOLUME, SUBMISSION_VOLUME_MOUNT]
            and (len(parts) == 2 or parts[2] == "rw")
        )
    if not isinstance(mount, dict):
        return False
    return (
        set(mount) <= {"type", "source", "target", "read_only"}
        and mount.get("type") == "volume"
        and mount.get("source") == SUBMISSION_VOLUME
        and mount.get("target") == SUBMISSION_VOLUME_MOUNT
        and mount.get("read_only", False) is False
    )


def _service_networks(service: dict) -> set[str]:
    networks = service.get("networks")
    if isinstance(networks, dict):
        return {str(network) for network in networks}
    if isinstance(networks, list):
        return {str(network) for network in networks}
    return set()


def _mcp_endpoints(mcp_servers) -> dict[str, str]:
    endpoints: dict[str, str] = {}
    for spec in mcp_servers:
        url = spec.url
        endpoint = urlsplit(url)
        try:
            endpoint.port
        except ValueError as exc:
            raise NonRetryableHarnessError(f"invalid internal MCP URL: {url!r}") from exc
        service_name = endpoint.hostname
        if (
            endpoint.scheme != "http"
            or not service_name
            or endpoint.username
            or endpoint.password
            or endpoint.fragment
        ):
            raise NonRetryableHarnessError(f"invalid internal MCP URL: {url!r}")
        if service_name in endpoints and endpoints[service_name] != url:
            raise NonRetryableHarnessError(
                f"multiple MCP endpoints declared for Compose service {service_name!r}"
            )
        endpoints[service_name] = url
    return endpoints


def _validate_mcp_internal_ports(
    services: dict[str, dict],
    endpoint_by_service: dict[str, str],
) -> None:
    missing_services = sorted(set(endpoint_by_service) - set(services))
    if missing_services:
        raise NonRetryableHarnessError(
            f"Harbor Compose is missing MCP services: {missing_services}"
        )

    for service_name, endpoint_url in endpoint_by_service.items():
        environment = services[service_name].get("environment") or {}
        if not isinstance(environment, dict):
            raise NonRetryableHarnessError(
                f"Harbor Compose service {service_name!r} environment must be an object"
            )
        endpoint = urlsplit(endpoint_url)
        value = endpoint.port or environment.get("MCP_PORT", MCP_BASE_PORT)
        try:
            port = int(value)
        except (TypeError, ValueError) as exc:
            raise NonRetryableHarnessError(
                f"Harbor Compose service {service_name!r} has an invalid MCP port"
            ) from exc
        if not 1 <= port <= 65535:
            raise NonRetryableHarnessError(
                f"Harbor Compose service {service_name!r} has an invalid MCP port"
            )


def _state_export_spec(services: dict[str, dict]) -> StateExportSpec | None:
    candidates = []
    for service_name, service in services.items():
        environment = service.get("environment") or {}
        if not isinstance(environment, dict):
            raise NonRetryableHarnessError(
                f"Harbor Compose service {service_name!r} environment must be an object"
            )
        if STATE_EXPORT_TOKEN_ENV not in environment:
            continue
        if service_name == "main":
            raise NonRetryableHarnessError(
                f"Harbor Compose must not expose {STATE_EXPORT_TOKEN_ENV} to main"
            )
        port = environment.get("APP_BACKEND_PORT") or environment.get("MCP_PORT")
        try:
            candidates.append(
                StateExportSpec(
                    service=service_name,
                    port=int(port or MCP_BASE_PORT),
                    initialize_path=_state_management_path(
                        environment, STATE_INITIALIZE_PATH_ENV
                    ),
                    finalize_path=_state_management_path(
                        environment, STATE_FINALIZE_PATH_ENV
                    ),
                )
            )
        except (TypeError, ValueError) as exc:
            raise NonRetryableHarnessError(
                f"Harbor Compose service {service_name!r} has an invalid state-export port"
            ) from exc
    if len(candidates) > 1:
        raise NonRetryableHarnessError(
            "Harbor Compose declares multiple structured-state export services"
        )
    return candidates[0] if candidates else None


def _verifier_network_spec(
    services: dict[str, dict],
) -> VerifierNetworkSpec | None:
    candidates = []
    for service_name, service in services.items():
        environment = service.get("environment") or {}
        if VERIFIER_NETWORK_ACCESS_ENV not in environment:
            continue
        if service_name in {"main", SUBMISSION_SERVICE}:
            raise NonRetryableHarnessError(
                f"Harbor Compose service {service_name!r} must not declare "
                f"{VERIFIER_NETWORK_ACCESS_ENV}"
            )
        marker = environment[VERIFIER_NETWORK_ACCESS_ENV]
        if marker != "1":
            raise NonRetryableHarnessError(
                f"Harbor Compose service {service_name!r} must set "
                f"{VERIFIER_NETWORK_ACCESS_ENV}='1', got {marker!r}"
            )
        networks = _service_networks(service)
        if len(networks) != 1:
            raise NonRetryableHarnessError(
                f"Harbor Compose service {service_name!r} must use exactly one network "
                f"when declaring {VERIFIER_NETWORK_ACCESS_ENV}"
            )
        network = next(iter(networks))
        network_services = sorted(
            name
            for name, candidate_service in services.items()
            if network in _service_networks(candidate_service)
        )
        if network_services != [service_name]:
            raise NonRetryableHarnessError(
                f"Harbor Compose service {service_name!r} must use a dedicated network "
                f"when declaring {VERIFIER_NETWORK_ACCESS_ENV}; found {network_services}"
            )
        candidates.append(VerifierNetworkSpec(service_name, network))
    if len(candidates) > 1:
        raise NonRetryableHarnessError(
            "Harbor Compose declares multiple verifier-access network services"
        )
    return candidates[0] if candidates else None


def _validate_workspace_mounts(
    services: dict[str, dict],
    endpoint_by_service: dict[str, str],
) -> None:
    for service_name, service in services.items():
        environment = service.get("environment") or {}
        if WORKSPACE_MOUNT_ENV not in environment:
            continue
        mount_path = environment[WORKSPACE_MOUNT_ENV]
        if service_name in {"main", SUBMISSION_SERVICE}:
            raise NonRetryableHarnessError(
                f"Harbor Compose service {service_name!r} must not declare {WORKSPACE_MOUNT_ENV}"
            )
        if (
            environment.get(VERIFIER_NETWORK_ACCESS_ENV) != "1"
            and service_name not in endpoint_by_service
        ):
            raise NonRetryableHarnessError(
                f"Harbor Compose service {service_name!r} must be a declared MCP endpoint "
                f"or set {VERIFIER_NETWORK_ACCESS_ENV}='1' before declaring "
                f"{WORKSPACE_MOUNT_ENV}"
            )
        if mount_path != WORKSPACE_DIR:
            raise NonRetryableHarnessError(
                f"Harbor Compose service {service_name!r} must set "
                f"{WORKSPACE_MOUNT_ENV}={WORKSPACE_DIR!r}, got {mount_path!r}"
            )


def _validate_workspace_write_access(
    services: dict[str, dict],
    endpoint_by_service: dict[str, str],
) -> None:
    for service_name, service in services.items():
        environment = service.get("environment") or {}
        if WORKSPACE_WRITE_ACCESS_ENV not in environment:
            continue
        marker = environment[WORKSPACE_WRITE_ACCESS_ENV]
        if marker != "1":
            raise NonRetryableHarnessError(
                f"Harbor Compose service {service_name!r} must set "
                f"{WORKSPACE_WRITE_ACCESS_ENV}='1', got {marker!r}"
            )
        if environment.get(WORKSPACE_MOUNT_ENV) != WORKSPACE_DIR:
            raise NonRetryableHarnessError(
                f"Harbor Compose service {service_name!r} must set "
                f"{WORKSPACE_MOUNT_ENV}={WORKSPACE_DIR!r} before declaring "
                f"{WORKSPACE_WRITE_ACCESS_ENV}='1'"
            )
        if service_name not in endpoint_by_service:
            raise NonRetryableHarnessError(
                f"Harbor Compose service {service_name!r} must be declared as an MCP "
                f"endpoint before declaring {WORKSPACE_WRITE_ACCESS_ENV}='1'"
            )


def _validate_agent_networks(
    services: dict[str, dict],
    endpoint_by_service: dict[str, str],
) -> None:
    main_networks = _service_networks(services["main"])
    hidden_networks = set().union(
        *(
            main_networks & _service_networks(services[service_name])
            for service_name in endpoint_by_service
        )
    )
    if endpoint_by_service and not main_networks - hidden_networks:
        raise NonRetryableHarnessError(
            "Harbor Compose leaves main without an agent network after MCP isolation"
        )


async def _block_agent_host_ports(
    session: SandboxIO,
    ports: Iterable[int],
) -> None:
    ports = sorted(set(ports))
    if not ports:
        return
    port_rules = " ".join(str(port) for port in ports)
    await _sh(
        session,
        f"""
        agent_ipv4s="$(docker inspect --format '{{{{range .NetworkSettings.Networks}}}}{{{{.IPAddress}}}} {{{{end}}}}' {AGENT_CONTAINER})"
        set -- $agent_ipv4s
        test "$#" -gt 0 || {{ echo "agent has no IPv4 address" >&2; exit 1; }}
        for agent_ip in "$@"; do
            for port in {port_rules}; do
                iptables -C INPUT -s "$agent_ip/32" -p tcp --dport "$port" -j REJECT --reject-with tcp-reset 2>/dev/null ||
                iptables -I INPUT 1 -s "$agent_ip/32" -p tcp --dport "$port" -j REJECT --reject-with tcp-reset || exit 1
            done
        done
        agent_ipv6s="$(docker inspect --format '{{{{range .NetworkSettings.Networks}}}}{{{{.GlobalIPv6Address}}}} {{{{end}}}}' {AGENT_CONTAINER})"
        set -- $agent_ipv6s
        test "$#" -gt 0 || {{ echo "agent has no IPv6 address" >&2; exit 1; }}
        for agent_ip in "$@"; do
            for port in {port_rules}; do
                ip6tables -C INPUT -s "$agent_ip/128" -p tcp --dport "$port" -j REJECT --reject-with tcp-reset 2>/dev/null ||
                ip6tables -I INPUT 1 -s "$agent_ip/128" -p tcp --dport "$port" -j REJECT --reject-with tcp-reset || exit 1
            done
        done
        for port in {port_rules}; do
            ip6tables -C INPUT -s {IPV6_LINK_LOCAL_CIDR} -p tcp --dport "$port" -j REJECT --reject-with tcp-reset 2>/dev/null ||
            ip6tables -I INPUT 1 -s {IPV6_LINK_LOCAL_CIDR} -p tcp --dport "$port" -j REJECT --reject-with tcp-reset || exit 1
        done
        """,
        operation="block agent access to host ports",
        check=True,
    )


async def _assert_live_stack(session: SandboxIO) -> None:
    """Fail closed when a warm provider snapshot did not restore a running Compose stack."""
    command = _compose_command()
    out = await _sh(
        session,
        f"docker info >/dev/null 2>&1 || {{ echo STACK_DEAD; echo DOCKER_DOWN; exit 0; }}; "
        f'expected="$({command} config --services | sort)"; '
        f'running="$({command} ps --services --filter status=running | sort)"; '
        f'ids="$({command} ps -q)"; '
        f'states="$(docker inspect --format '
        f"'{{{{if .State.Health}}}}{{{{.State.Health.Status}}}}"
        f"{{{{else}}}}{{{{.State.Status}}}}{{{{end}}}}' $ids 2>/dev/null)\"; "
        f"bad=\"$(printf '%s\\n' \"$states\" | grep -Ev '^(healthy|running)$' || true)\"; "
        f'[ "$expected" = "$running" ] && [ -z "$bad" ] '
        f"&& echo STACK_LIVE || (echo STACK_DEAD; {command} ps; printf '%s\\n' \"$states\")",
        60,
        operation="check Compose stack health",
    )
    if "STACK_LIVE" not in out:
        raise RuntimeError(
            "feather: warm snapshot did not restore a live Compose stack "
            "(stop-before-snapshot and filesystem-only bakes are incompatible; re-bake)"
        )


async def _wait_for_ports(
    session: SandboxIO, port_by_service: dict[str, int], endpoint_by_service: dict[str, str]
) -> None:
    # On resume the ports come from the baked snapshot and the endpoints from the current task, so
    # they can disagree; say so rather than raising a KeyError the retry layer treats as transient.
    _validate_mcp_port_contract(port_by_service, endpoint_by_service)
    for service_name, port in sorted(port_by_service.items()):
        endpoint = urlsplit(endpoint_by_service[service_name])
        path = endpoint.path or "/"
        if endpoint.query:
            path = f"{path}?{endpoint.query}"
        local_url = shlex.quote(f"http://127.0.0.1:{port}{path}")
        # Ready means the server answered, whatever it answered: a streamable-http endpoint may
        # legitimately 404 or 307 a bare GET. curl exits non-zero only when nothing responded.
        out = await _sh(
            session,
            f"for i in $(seq 1 40); do "
            f"curl -sS -o /dev/null --max-time 5 {local_url} "
            f"&& echo PORT_{port}_READY && break; "
            f"sleep 2; done",
            120,
            operation=f"check MCP service {service_name!r} on port {port}",
        )
        if f"PORT_{port}_READY" not in out:
            raise RuntimeError(
                f"feather: MCP service {service_name!r} on port {port} never became ready"
            )


def _state_management_path(environment: dict, name: str) -> str | None:
    value = environment.get(name)
    if value is None:
        return None
    if not isinstance(value, str):
        raise NonRetryableHarnessError(
            f"Harbor Compose {name} must be an absolute HTTP path"
        )
    path = PurePosixPath(value)
    if (
        not value.startswith("/")
        or value.startswith("//")
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts[1:])
    ):
        raise NonRetryableHarnessError(
            f"Harbor Compose {name} must be a canonical absolute HTTP path"
        )
    return value


async def invoke_state_hook(
    session: SandboxIO,
    state_export: StateExportSpec,
    path: str,
) -> None:
    """POST the Compose state hook from inside the state-export service."""
    service = shlex.quote(state_export.service)
    inspect_env = "{{range .Config.Env}}{{println .}}{{end}}"
    hook_url = shlex.quote(f"http://127.0.0.1:{state_export.port}{path}")
    hook_script = shlex.quote(
        "set -eu; umask 077; header_file=$(mktemp); "
        "trap 'rm -f \"$header_file\"' EXIT; "
        "printf '%s' \"Authorization: Bearer ${APP_STATE_EXPORT_TOKEN:?}\" "
        '> "$header_file"; '
        "curl --fail --silent --show-error "
        '-H "@$header_file" '
        f"-X POST {hook_url}"
    )
    command = _compose_command()
    await _sh(
        session,
        f'state_container="$({command} ps -q {service})" && '
        'test -n "$state_container" && '
        f'state_token="$(docker inspect --format '
        f"'{inspect_env}' \"$state_container\" | "
        f"sed -n 's/^{STATE_EXPORT_TOKEN_ENV}=//p')\" && "
        'test -n "$state_token" && '
        f"docker exec --env {STATE_EXPORT_TOKEN_ENV}=\"$state_token\" "
        f'"$state_container" sh -lc {hook_script}',
        STATE_HOOK_TIMEOUT_SECONDS,
        check=True,
        operation=f"invoke state hook {path}",
    )


def _mcp_urls(
    session: SandboxIO, port_by_service: dict[str, int], mcp_servers: list[McpServerSpec]
) -> list[McpServerSpec]:
    updated = []
    for spec in mcp_servers:
        declared = urlsplit(spec.url)
        port = port_by_service[declared.hostname]
        updated.append(
            spec.model_copy(update={
                "url": urlunsplit(
                    (
                        "https",
                        session.sandbox.get_host(port),
                        declared.path,
                        declared.query,
                        declared.fragment,
                    )
                ),
            })
        )
    logger.info("feather: MCP services: %s", [server.name for server in updated])
    return updated


def _ports_from_override(override: dict) -> dict[str, int]:
    ports: dict[str, int] = {}
    for service_name, service in (override.get("services") or {}).items():
        published = service.get("ports") or []
        if published:
            ports[service_name] = int(str(published[0]).rsplit(":", 2)[-2])
    return ports


def _public_host_ports_from_override(override: dict) -> list[int]:
    ports = []
    for service in (override.get("services") or {}).values():
        for published in service.get("ports") or []:
            host, port, _ = str(published).rsplit(":", 2)
            if host == "0.0.0.0":
                ports.append(int(port))
    return ports


def _services_from_world(world) -> dict[str, dict]:
    services = world.get("services") if isinstance(world, dict) else None
    if not isinstance(services, dict):
        raise NonRetryableHarnessError("invalid Harbor Compose services in baked snapshot")
    if not isinstance(services.get("main"), dict):
        raise NonRetryableHarnessError("invalid Harbor Compose main service in baked snapshot")
    invalid_services = sorted(
        service_name for service_name, service in services.items() if not isinstance(service, dict)
    )
    if invalid_services:
        raise NonRetryableHarnessError(
            f"invalid Harbor Compose service objects in baked snapshot: {invalid_services}"
        )
    return services


def _validate_mcp_port_contract(
    port_by_service: dict[str, int], endpoint_by_service: dict[str, str]
) -> None:
    if set(port_by_service) != set(endpoint_by_service):
        raise NonRetryableHarnessError(
            f"snapshot publishes MCP ports for {sorted(port_by_service)} but the world declares "
            f"{sorted(endpoint_by_service)}; re-bake the world"
        )


def _compose_command() -> str:
    return (
        f"cd {ENV_DIR} && docker-compose -p {COMPOSE_PROJECT} "
        f"-f {WORLD_COMPOSE_FILE} -f {COMPOSE_FILE}"
    )


async def _prepare_separate_verifier(
    session: SandboxIO,
    collect_hooks: tuple[CollectHook, ...],
    artifacts: tuple[HarborArtifact, ...],
) -> None:
    """Collect declared evidence, then grade in a fresh tests-image container."""
    if not artifacts:
        raise NonRetryableHarnessError("separate verifier requires artifact paths")
    world_container = None
    if any(hook.service == "world" for hook in collect_hooks) or any(
        isinstance(artifact, ArtifactSpec) and artifact.service == "world"
        for artifact in artifacts
    ):
        world_container = await _sh(
            session,
            f"{_compose_command()} ps -q world",
            operation="resolve Harbor world service",
            check=True,
        )
        if not world_container or len(world_container.splitlines()) != 1:
            raise NonRetryableHarnessError("Harbor world service is absent or ambiguous")
    for hook in collect_hooks:
        container = world_container if hook.service == "world" else AGENT_CONTAINER
        await _sh(
            session,
            "docker exec -u root "
            f"{shlex.quote(container)} /bin/bash -lc {shlex.quote(hook.command)}",
            hook.timeout_sec,
            operation="run Harbor collect hook",
            check=True,
        )
    host_root = "/tmp/feather-separate-artifacts"
    await _sh(
        session,
        f"rm -rf {host_root} && mkdir -p {host_root}",
        operation="stage collect artifact host dir",
        check=True,
    )
    for artifact in artifacts:
        path = artifact.source if isinstance(artifact, ArtifactSpec) else artifact
        container = world_container if isinstance(artifact, ArtifactSpec) and artifact.service == "world" else AGENT_CONTAINER
        await _sh(
            session,
            f"docker cp {shlex.quote(container)}:{shlex.quote(path)} "
            f"{host_root}/{shlex.quote(PurePosixPath(path).name)}",
            operation=f"copy collect artifact {path}",
            check=True,
        )
    await _sh(
        session,
        f"test -f {shlex.quote(BAKED_TESTS_DIR)}/Dockerfile",
        operation="require baked tests Dockerfile",
        check=True,
    )
    await _sh(
        session,
        f"docker image inspect {SEPARATE_VERIFIER_IMAGE}",
        operation="require baked tests-image",
        check=True,
    )
    command = _compose_command()
    await _sh(
        session,
        f"{command} stop main 2>&1 && {command} rm -f main 2>&1",
        300,
        operation="stop agent container",
        check=True,
    )
    await _sh(
        session,
        f"docker rm -f {VERIFIER_CONTAINER} >/dev/null 2>&1 || true; "
        f"docker run -d --name {VERIFIER_CONTAINER} --entrypoint sleep "
        f"{SEPARATE_VERIFIER_IMAGE} infinity",
        300,
        operation="start fresh tests-image verifier",
        check=True,
    )
    await _copy_baked_tests_into_container(session, VERIFIER_CONTAINER)
    for artifact in artifacts:
        path = artifact.source if isinstance(artifact, ArtifactSpec) else artifact
        dest_dir = str(PurePosixPath(path).parent)
        await _sh(
            session,
            f"docker exec {VERIFIER_CONTAINER} mkdir -p {shlex.quote(dest_dir)} && "
            f"docker cp {host_root}/{shlex.quote(PurePosixPath(path).name)} "
            f"{VERIFIER_CONTAINER}:{shlex.quote(path)}",
            operation=f"restore collect artifact {path}",
            check=True,
        )


async def _copy_baked_tests_into_container(session: SandboxIO, container: str) -> None:
    """Official modes grade inside agent/tests images that do not mount host ``BAKED_TESTS_DIR``."""
    await _sh(
        session,
        f"docker exec -u root {shlex.quote(container)} mkdir -p {shlex.quote(BAKED_TESTS_DIR)} && "
        f"docker cp {shlex.quote(BAKED_TESTS_DIR)}/. "
        f"{shlex.quote(container)}:{shlex.quote(BAKED_TESTS_DIR)}/",
        operation=f"copy baked tests into {container}",
        check=True,
    )


async def _sh(
    session: SandboxIO,
    cmd: str,
    timeout: float = 200,
    check: bool = False,
    *,
    operation: str,
) -> str:
    result = await session.exec(cmd, shell=True, timeout=timeout)
    # Compose config and command output can contain credentials and private endpoints.
    logger.info("feather[compose] %s -> exit %s", operation, result.exit_code)
    if check and result.exit_code != 0:
        raise RuntimeError(f"feather: {operation} failed (exit {result.exit_code})")
    return (result.stdout + result.stderr).strip()
