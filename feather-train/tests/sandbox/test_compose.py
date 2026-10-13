"""Warm worlds retain MCP routing and separate agent and grader privileges."""

import subprocess
from types import SimpleNamespace

import pytest
import yaml
from feather_harness.tools.mcp import McpServerSpec

pytest.importorskip("verifiers.v1")
from verifiers.v1 import SandboxError

from feather_train.errors import NonRetryableHarnessError
from feather_train.sandbox.compose import (
    ComposeRuntime,
    StateExportSpec,
    VerifierNetworkSpec,
    assert_agent_container_survived,
    prepare_compose_grading,
    resume_compose_stack,
)
from feather_train.tasks.harbor_spec import ArtifactSpec, CollectHook


class BakedWorld:
    def __init__(
        self,
        world=None,
        ports=None,
        *,
        live=True,
        agent="true 0 0",
        verifier="true 0 0",
        bind_host="127.0.0.1",
    ):
        self.world = world or {"services": {"main": {"networks": ["default"]}}}
        self.ports = ports or {}
        self.bind_host = bind_host
        self.live = live
        self.agent = agent
        self.verifier = verifier
        self.commands = []
        self.sandbox = SimpleNamespace(get_host=lambda port: f"sandbox-{port}.example.test")

    async def exec(self, command, *, shell, timeout):
        self.commands.append(command)
        if command == "cat /opt/apex_world/docker-compose.yaml":
            output = yaml.safe_dump(self.world)
        elif command == "cat /opt/apex_world/compose.train.yaml":
            output = yaml.safe_dump(
                {
                    "services": {
                        service: {"ports": [f"{self.bind_host}:{port}:5000"]}
                        for service, port in self.ports.items()
                    }
                }
            )
        elif "docker info" in command:
            output = "STACK_LIVE" if self.live else "STACK_DEAD"
        elif command.startswith("for i in $(seq 1 40)"):
            output = "\n".join(f"PORT_{port}_READY" for port in self.ports.values())
        elif command.startswith("docker container ls"):
            output = "verifier\n" if self.verifier else ""
        elif "ps -q world" in command:
            output = "world-container-id"
        elif "Config.Image" in command:
            output = "task-image:v1"
        elif command.startswith("docker inspect"):
            state = self.agent if command.endswith(" agent") else self.verifier
            return SimpleNamespace(
                stdout=state,
                stderr="" if state else "No such container",
                exit_code=0 if state else 1,
            )
        else:
            output = ""
        return SimpleNamespace(stdout=output, stderr="", exit_code=0)


async def resume(world, servers=()):
    return await resume_compose_stack(
        world,
        mcp_servers=[McpServerSpec.model_validate(server) for server in servers],
    )


@pytest.mark.parametrize("request_timeout_seconds", [0, 60, 120.5])
async def test_resume_rewrites_mcp_origin_preserving_endpoint_and_query(request_timeout_seconds):
    world = BakedWorld(
        {
            "services": {
                "main": {"networks": ["default", "mcp"]},
                "mail": {"networks": ["mcp"], "environment": {"MCP_PORT": "5000"}},
            },
        },
        {"mail": 5003},
    )
    runtime = await resume(
        world,
        [
            {
                "name": "mail",
                "transport": "streamable-http",
                "url": "http://mail:5000/api/mcp?tenant=test",
                "request_timeout_seconds": request_timeout_seconds,
                "headers": {"Authorization": "Bearer test"},
                "raise_on_error_result": True,
            }
        ],
    )
    assert runtime == ComposeRuntime(
        [
            McpServerSpec(
                name="mail",
                transport="streamable-http",
                url="https://sandbox-5003.example.test/api/mcp?tenant=test",
                request_timeout_seconds=request_timeout_seconds,
                headers={"Authorization": "Bearer test"},
                raise_on_error_result=True,
            )
        ],
        None,
    )
    assert all(
        " up " not in command and "network connect" not in command for command in world.commands
    )


@pytest.mark.parametrize(
    ("options", "error", "message"),
    [
        ({"live": False}, RuntimeError, "live Compose stack"),
        ({"verifier": ""}, NonRetryableHarnessError, "baked verifier container"),
        ({"verifier": "true 0 1"}, SandboxError, "did not survive"),
    ],
)
async def test_resume_refuses_dead_or_restarted_baked_containers(options, error, message):
    with pytest.raises(error, match=message):
        await resume(BakedWorld(**options))


async def test_resume_rejects_ports_from_a_different_world():
    with pytest.raises(NonRetryableHarnessError, match="snapshot publishes MCP ports"):
        await resume(BakedWorld(ports={"old-service": 5000}))


async def test_submission_volume_must_not_be_mounted_into_agent():
    world = BakedWorld(
        {
            "services": {
                "main": {
                    "networks": ["default"],
                    "volumes": ["submission-state:/submission-state"],
                },
                "submit-mcp": {
                    "networks": ["submission"],
                    "volumes": ["submission-state:/submission-state"],
                },
            },
            "volumes": {"submission-state": {}},
        },
        {"submit-mcp": 5000},
    )
    with pytest.raises(NonRetryableHarnessError, match="main.volumes"):
        await resume(
            world,
            [
                {
                    "name": "submission",
                    "transport": "streamable-http",
                    "url": "http://submit-mcp:5000/mcp",
                }
            ],
        )
    assert len(world.commands) == 2  # Rejected before any Docker operation.


@pytest.mark.parametrize("state", ["", "false 137 0", "true 0 1"])
async def test_dead_or_restarted_agent_is_an_infrastructure_failure(state):
    with pytest.raises(SandboxError):
        await assert_agent_container_survived(BakedWorld(agent=state))


async def test_grading_removes_agent_before_attaching_private_json_state():
    world = BakedWorld()
    await prepare_compose_grading(
        world,
        StateExportSpec("private-store", 5000),
        mount_submission_volume=True,
    )
    stop, inspect, submission_check, private_state = world.commands
    assert "stop main" in stop and "rm -f main" in stop
    assert "verifier" in inspect
    assert "docker exec verifier test -d /submission-state" == submission_check
    assert "ps -q private-store" in private_state
    assert 'docker network connect "$state_network" verifier' in private_state
    assert "docker exec -i verifier" in private_state
    assert "network connect" not in stop


async def test_shared_grading_keeps_the_agent_container():
    world = BakedWorld()
    await prepare_compose_grading(world, None, environment_mode="shared")
    joined = "\n".join(world.commands)
    assert not any("stop main" in command for command in world.commands)
    assert "docker exec -u root agent mkdir -p /opt/apex_grader_tests" in joined
    assert "docker cp /opt/apex_grader_tests/. agent:/opt/apex_grader_tests/" in joined


async def test_resume_skips_verifier_when_not_required():
    world = BakedWorld(verifier="")
    await resume_compose_stack(world, mcp_servers=[], require_verifier=False)
    assert not any("verifier" in command and "inspect" in command for command in world.commands)


async def test_separate_grading_collects_then_refreshes_verifier_from_tests_image():
    world = BakedWorld()
    await prepare_compose_grading(
        world,
        None,
        environment_mode="separate",
        collect_hooks=(CollectHook(command="git diff --binary HEAD", timeout_sec=300),),
        artifacts=("/logs/artifacts/model.patch",),
    )
    joined = "\n".join(world.commands)
    assert "docker exec -u root agent /bin/bash -lc" in joined
    assert "git diff --binary HEAD" in joined
    assert "docker cp agent:/logs/artifacts/model.patch" in joined
    assert "test -f /opt/apex_grader_tests/Dockerfile" in joined
    assert "docker image inspect feather-separate-verifier" in joined
    assert "stop main" in joined and "rm -f main" in joined
    assert "docker build -t feather-separate-verifier" not in joined
    assert "docker rm -f verifier" in joined
    assert "docker run -d --name verifier --entrypoint sleep feather-separate-verifier infinity" in joined
    assert "docker cp /opt/apex_grader_tests/. verifier:/opt/apex_grader_tests/" in joined
    assert "docker cp /tmp/feather-separate-artifacts/model.patch verifier:/logs/artifacts/model.patch" in joined


async def test_separate_grading_without_collect_hook_copies_agent_artifact():
    world = BakedWorld()
    await prepare_compose_grading(
        world,
        None,
        environment_mode="separate",
        artifacts=("/logs/agent/trajectory.json",),
    )
    joined = "\n".join(world.commands)
    assert "docker cp agent:/logs/agent/trajectory.json" in joined
    assert "docker exec -u root agent /bin/bash -lc" not in joined
    assert "docker cp /tmp/feather-separate-artifacts/trajectory.json verifier:/logs/agent/trajectory.json" in joined


async def test_separate_grading_collects_world_artifact_from_world_service():
    world = BakedWorld()
    await prepare_compose_grading(
        world,
        None,
        environment_mode="separate",
        collect_hooks=(
            CollectHook(command="python /app/tools/collect_world_snapshot.py", timeout_sec=660, service="world"),
        ),
        artifacts=(
            "/logs/agent/trajectory.json",
            ArtifactSpec(source="/logs/world/final_snapshot.tar.gz", service="world"),
        ),
    )
    joined = "\n".join(world.commands)
    assert "ps -q world" in joined
    assert "docker exec -u root world-container-id /bin/bash -lc" in joined
    assert "docker cp world-container-id:/logs/world/final_snapshot.tar.gz" in joined
    assert "docker cp /tmp/feather-separate-artifacts/final_snapshot.tar.gz verifier:/logs/world/final_snapshot.tar.gz" in joined


async def test_grading_without_private_state_does_not_attach_networks():
    world = BakedWorld()
    await prepare_compose_grading(world, None)
    assert len(world.commands) == 2
    assert all("network connect" not in command for command in world.commands)


async def test_grading_connects_only_verifier_to_declared_private_network():
    world = BakedWorld()
    await prepare_compose_grading(
        world,
        None,
        verifier_network=VerifierNetworkSpec("judge-api", "judge-private"),
    )
    assert "rm -f main" in world.commands[0]
    assert "docker network connect apex_judge-private verifier" in world.commands[-1]


@pytest.mark.parametrize("bind_host", ["127.0.0.1", "0.0.0.0"])
async def test_resume_blocks_agent_access_to_public_host_bindings(bind_host):
    world = BakedWorld(
        {
            "services": {
                "main": {"networks": ["default", "mcp"]},
                "mail": {"networks": ["mcp"]},
            },
        },
        {"mail": 5003},
        bind_host=bind_host,
    )
    await resume(
        world,
        [
            {
                "name": "mail",
                "transport": "streamable-http",
                "url": "http://mail:5000/mcp",
            }
        ],
    )
    firewall_commands = [command for command in world.commands if "iptables" in command]
    if bind_host == "127.0.0.1":
        assert not firewall_commands
        return
    [firewall] = firewall_commands
    assert "for port in 5003;" in firewall
    assert 'iptables -I INPUT 1 -s "$agent_ip/32"' in firewall
    assert 'ip6tables -I INPUT 1 -s "$agent_ip/128"' in firewall
    assert "ip6tables -I INPUT 1 -s fe80::/10" in firewall
    assert world.commands.index(firewall) < next(
        i for i, command in enumerate(world.commands) if "curl " in command
    )


@pytest.mark.parametrize(
    ("service", "message"),
    [
        (["not-an-object"], "invalid Harbor Compose service objects"),
        ({"environment": ["HARBOR_VERIFIER_NETWORK_ACCESS=1"]}, "environment must be an object"),
    ],
)
async def test_resume_rejects_malformed_services_before_topology_checks(service, message):
    world = BakedWorld({"services": {"main": {"networks": ["default"]}, "private": service}})
    with pytest.raises(NonRetryableHarnessError, match=message):
        await resume(world)
    assert len(world.commands) == 2


async def test_resume_accepts_submission_volume_only_on_submission_service():
    world = BakedWorld(
        {
            "services": {
                "main": {"networks": ["default"]},
                "submit-mcp": {
                    "networks": ["submission"],
                    "volumes": ["submission-state:/submission-state"],
                },
            },
            "volumes": {"submission-state": {}},
        },
        {"submit-mcp": 5000},
    )
    runtime = await resume(
        world,
        [
            {
                "name": "submission",
                "transport": "streamable-http",
                "url": "http://submit-mcp:5000/mcp",
            }
        ],
    )
    assert [server.model_dump(exclude_defaults=True) for server in runtime.mcp_servers] == [
        {
            "name": "submission",
            "transport": "streamable-http",
            "url": "https://sandbox-5000.example.test/mcp",
        }
    ]


async def test_failed_verifier_inspection_is_not_proof_of_bad_bake():
    diagnostic = "Docker indisponible — réessayer"

    class UnreachableDocker(BakedWorld):
        async def exec(self, command, *, shell, timeout):
            if command.startswith(("docker inspect", "docker container ls")):
                return SimpleNamespace(stdout="", stderr=diagnostic, exit_code=1)
            return await super().exec(command, shell=shell, timeout=timeout)

    with pytest.raises(SandboxError, match="could not inspect verifier") as raised:
        await resume(UnreachableDocker())
    assert diagnostic in str(raised.value)


@pytest.mark.parametrize("failure", ["ipv4", "ipv6", "link-local", "", "already-present"])
async def test_public_port_rules_propagate_each_insertion_failure(failure):
    # Execute the generated shell, replacing only external programs. An early failed
    # insertion must not be hidden by a later successful rule or address lookup.
    prelude = r"""
    docker() {
        case "$*" in
            *GlobalIPv6Address*) echo fd00::2 ;;
            *) echo 172.18.0.2 ;;
        esac
    }
    rule() {
        family="$1"; shift
        if [ "$1" = -C ]; then
            [ "$FAILURE" = already-present ]; return
        fi
        if [ "$FAILURE" = already-present ]; then
            echo 'unexpected insertion' >&2; return 1
        fi
        if [ "$family" = "$FAILURE" ]; then
            echo "failed $family insertion" >&2; return 1
        fi
        return 0
    }
    iptables() { rule ipv4 "$@"; }
    ip6tables() {
        case "$*" in
            *fe80::/10*) rule link-local "$@" ;;
            *) rule ipv6 "$@" ;;
        esac
    }
    """

    class ShellWorld(BakedWorld):
        async def exec(self, command, *, shell, timeout):
            if "iptables" not in command:
                return await super().exec(command, shell=shell, timeout=timeout)
            self.commands.append(command)
            result = subprocess.run(
                ["bash", "-c", f"FAILURE={failure}\n" + prelude + command],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return SimpleNamespace(
                stdout=result.stdout, stderr=result.stderr, exit_code=result.returncode
            )

    world = ShellWorld(
        {
            "services": {
                "main": {"networks": ["default", "mcp"]},
                "mail": {"networks": ["mcp"]},
            },
        },
        {"mail": 5003},
        bind_host="0.0.0.0",
    )
    servers = [{"name": "mail", "transport": "streamable-http", "url": "http://mail:5000/mcp"}]
    if failure in {"ipv4", "ipv6", "link-local"}:
        with pytest.raises(RuntimeError, match=r"block agent access to host ports failed \(exit 1\)"):
            await resume(world, servers)
        assert not any("curl " in command for command in world.commands)
    else:
        assert (await resume(world, servers)).mcp_servers[
            0
        ].url == "https://sandbox-5003.example.test/mcp"


async def test_container_inspection_ignores_stderr_notices():
    class NoisyDocker(BakedWorld):
        async def exec(self, command, *, shell, timeout):
            result = await super().exec(command, shell=shell, timeout=timeout)
            if command.startswith("docker inspect"):
                result.stderr = "Docker CLI notice\n"
            return result

    world = NoisyDocker()
    await assert_agent_container_survived(world)
    await resume(world)


async def test_resume_detects_private_state_export_service():
    world = BakedWorld(
        {
            "services": {
                "main": {"networks": ["default"]},
                "store": {
                    "networks": ["private"],
                    "environment": {
                        "APP_STATE_EXPORT_TOKEN": "placeholder",
                        "APP_BACKEND_PORT": "8000",
                    },
                },
            }
        }
    )
    runtime = await resume(world)
    assert runtime.state_export == StateExportSpec("store", 8000)
    assert runtime.verifier_network is None


async def test_resume_detects_dedicated_verifier_network():
    world = BakedWorld(
        {
            "services": {
                "main": {"networks": ["default"]},
                "judge": {
                    "networks": ["judge-private"],
                    "environment": {"HARBOR_VERIFIER_NETWORK_ACCESS": "1"},
                },
            }
        }
    )
    runtime = await resume(world)
    assert runtime.verifier_network == VerifierNetworkSpec("judge", "judge-private")
    assert runtime.state_export is None


@pytest.mark.parametrize(
    "services,message",
    [
        (
            {
                "main": {
                    "networks": ["default"],
                    "environment": {"APP_STATE_EXPORT_TOKEN": "secret"},
                }
            },
            "must not expose APP_STATE_EXPORT_TOKEN to main",
        ),
        (
            {
                "main": {"networks": ["default"]},
                "judge": {
                    "networks": ["default"],
                    "environment": {"HARBOR_VERIFIER_NETWORK_ACCESS": "1"},
                },
            },
            "must use a dedicated network",
        ),
        (
            {
                "main": {"networks": ["default"]},
                "store": {
                    "networks": ["state"],
                    "environment": {"APP_STATE_EXPORT_TOKEN": "placeholder"},
                },
                "judge": {
                    "networks": ["judge-private"],
                    "environment": {"HARBOR_VERIFIER_NETWORK_ACCESS": "1"},
                },
            },
            "must not combine structured-state export",
        ),
    ],
    ids=["token-exposed-to-agent", "shared-verifier-network", "conflicting-grading-contracts"],
)
async def test_resume_rejects_unsafe_grading_contract_before_docker(services, message):
    world = BakedWorld({"services": services})
    with pytest.raises(NonRetryableHarnessError, match=message):
        await resume(world)
    assert all(command.startswith("cat ") for command in world.commands)


async def test_resume_rejects_isolation_that_would_remove_all_agent_networks():
    world = BakedWorld(
        {
            "services": {
                "main": {"networks": ["mcp"]},
                "mail": {"networks": ["mcp"]},
            }
        },
        {"mail": 5003},
    )
    with pytest.raises(NonRetryableHarnessError, match="without an agent network"):
        await resume(
            world, [{"name": "mail", "transport": "streamable-http", "url": "http://mail:5000/mcp"}]
        )
    assert all(command.startswith("cat ") for command in world.commands)


async def test_resume_reads_compose_state_hooks():
    world = BakedWorld(
        {
            "services": {
                "main": {"networks": ["default"]},
                "archipelago": {
                    "networks": ["mcp-access"],
                    "environment": {
                        "APP_STATE_EXPORT_TOKEN": "placeholder",
                        "APP_BACKEND_PORT": "8080",
                        "APP_STATE_INITIALIZE_PATH": "/state-initialize",
                        "APP_STATE_FINALIZE_PATH": "/state-finalize",
                    },
                },
            }
        }
    )
    runtime = await resume(world)
    assert runtime.state_export == StateExportSpec(
        "archipelago",
        8080,
        initialize_path="/state-initialize",
        finalize_path="/state-finalize",
    )


async def test_grading_finalizes_compose_state_after_removing_main():
    world = BakedWorld()
    await prepare_compose_grading(
        world,
        StateExportSpec("archipelago", 8080, finalize_path="/state-finalize"),
    )
    stop, finalize, inspect, private_state = world.commands
    assert "stop main" in stop and "rm -f main" in stop
    assert "ps -q archipelago" in finalize
    assert "POST" in finalize and "/state-finalize" in finalize
    assert "verifier" in inspect
    assert "docker exec -i verifier" in private_state
