"""The agent egress denylist admits connections by requested name and destination."""

import asyncio
import builtins
import io
import json
import socket
import ssl
import subprocess
from contextlib import redirect_stdout
from functools import partial
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from feather_train.harness import FeatherHarnessConfig
from feather_train.sandbox import egress_proxy
from feather_train.sandbox.egress import _install_script, _interface_discovery_script
from feather_train.sandbox.egress_proxy import (
    NEED_MORE,
    EgressFilter,
    is_blocked_host,
    requested_host,
)

DOMAINS = ("huggingface.co", "hf.co")


def _client_hello(server_name: str) -> bytes:
    context = ssl.create_default_context()
    incoming, outgoing = ssl.MemoryBIO(), ssl.MemoryBIO()
    connection = context.wrap_bio(incoming, outgoing, server_hostname=server_name)
    with pytest.raises(ssl.SSLWantReadError):
        connection.do_handshake()
    return outgoing.read()


def test_tls_server_name_is_read_only_once_the_client_hello_is_complete():
    hello = _client_hello("cdn-lfs.huggingface.co")

    assert requested_host(hello[:3]) is NEED_MORE
    assert requested_host(hello[:-1]) is NEED_MORE
    assert requested_host(hello) == "cdn-lfs.huggingface.co"


def test_http_host_strips_port_and_waits_for_complete_headers():
    request = b"GET /api/models HTTP/1.1\r\nUser-Agent: x\r\nHOST: hf.co:80\r\n\r\n"

    assert requested_host(request[:-2]) is NEED_MORE
    assert requested_host(request) == "hf.co"
    assert requested_host(b"GET / HTTP/1.0\r\n\r\n") is None


@pytest.mark.parametrize(
    ("host", "blocked"),
    [
        ("huggingface.co", True),
        ("HuggingFace.CO.", True),
        ("cas-bridge.xethub.hf.co", True),
        ("nothuggingface.co", False),
        ("huggingface.co.evil.test", False),
        ("pypi.org", False),
        ("%2568uggingface.co", True),
        ("ｈｕｇｇｉｎｇｆａｃｅ.co", True),
        ("h&#117;ggingface.co", True),
    ],
)
def test_domains_block_themselves_and_subdomains_only(host, blocked):
    assert is_blocked_host(host, DOMAINS) is blocked


async def test_blocked_names_and_addresses_never_reach_upstream_while_others_are_spliced(
    monkeypatch,
):
    upstream_requests = []

    async def upstream(reader, writer):
        upstream_requests.append(await reader.readuntil(b"\r\n\r\n"))
        writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok")
        await writer.drain()
        writer.close()

    server = await asyncio.start_server(upstream, "127.0.0.1", 0)
    upstream_address = server.sockets[0].getsockname()[:2]
    # SO_ORIGINAL_DST needs Linux NAT; every redirected connection targets the fake upstream.
    monkeypatch.setattr(egress_proxy, "_original_destination", lambda sock: upstream_address)
    egress = EgressFilter(DOMAINS)
    proxy = await asyncio.start_server(egress.handle, "127.0.0.1", 0)
    proxy_port = proxy.sockets[0].getsockname()[1]

    async def request(host: str) -> bytes:
        reader, writer = await asyncio.open_connection("127.0.0.1", proxy_port)
        writer.write(f"GET / HTTP/1.1\r\nHost: {host}\r\n\r\n".encode())
        response = await reader.read()
        writer.close()
        return response

    async with server, proxy:
        assert (await request("huggingface.co")).startswith(b"HTTP/1.1 403")
        assert (await request("pypi.org")).endswith(b"ok")
        egress.blocked_addresses.add(upstream_address[0])
        assert (await request("pypi.org")).startswith(b"HTTP/1.1 403")

    assert upstream_requests == [b"GET / HTTP/1.1\r\nHost: pypi.org\r\nConnection: close\r\n\r\n"]


@pytest.mark.parametrize("mode", ["buffered", "later", "fixed_length_body"])
async def test_plaintext_second_request_never_reaches_admitted_upstream(monkeypatch, mode):
    received = []
    observed = asyncio.Event()
    respond = asyncio.Event()

    async def upstream(reader, writer):
        received.append(await reader.read())
        observed.set()
        await respond.wait()
        writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok")
        await writer.drain()
        writer.close()

    server = await asyncio.start_server(upstream, "127.0.0.1", 0)
    address = server.sockets[0].getsockname()[:2]
    monkeypatch.setattr(egress_proxy, "_original_destination", lambda sock: address)
    proxy = await asyncio.start_server(EgressFilter(DOMAINS).handle, "127.0.0.1", 0)
    first = b"GET / HTTP/1.1\r\nHost: pypi.org\r\nConnection: keep-alive\r\n\r\n"
    expected = b"GET / HTTP/1.1\r\nHost: pypi.org\r\nConnection: close\r\n\r\n"
    second = b"GET /secret HTTP/1.1\r\nHost: hf.co\r\n\r\n"
    if mode == "fixed_length_body":
        first = b"POST / HTTP/1.1\r\nHost: pypi.org\r\nContent-Length: 4\r\n\r\nbody"
        expected = b"POST / HTTP/1.1\r\nHost: pypi.org\r\nContent-Length: 4\r\nConnection: close\r\n\r\nbody"
    async with server, proxy:
        reader, writer = await asyncio.open_connection("127.0.0.1", proxy.sockets[0].getsockname()[1])
        if mode == "fixed_length_body":
            writer.write(first[:-2])
            await writer.drain()
            await asyncio.sleep(0.01)
            writer.write(first[-2:] + second)
        else:
            writer.write(first + (second if mode == "buffered" else b""))
        await writer.drain()
        await asyncio.wait_for(observed.wait(), 1)
        if mode == "later":
            writer.write(second)
            await writer.drain()
        respond.set()
        assert (await asyncio.wait_for(reader.read(), 1)).endswith(b"ok")
        writer.close()
    assert received == [expected]


@pytest.mark.parametrize("header", [
    b"Content-Length: 0\r\nContent-Length: 0",
    b"Transfer-Encoding: chunked",
    b"Content-Length: -1",
    b"Content-Length: 1, 1",
    b"Content-Length : 0",
    b"Upgrade: websocket",
    b" folded-header: value",
])
async def test_ambiguous_or_tunneled_plaintext_framing_never_opens_upstream(monkeypatch, header):
    monkeypatch.setattr(egress_proxy, "_original_destination", lambda sock: ("203.0.113.7", 8080))
    proxy = await asyncio.start_server(EgressFilter(DOMAINS).handle, "127.0.0.1", 0)
    connect = asyncio.open_connection

    async def reject_upstream(*args, **kwargs):
        pytest.fail("ambiguous request reached upstream")

    async with proxy:
        reader, writer = await connect("127.0.0.1", proxy.sockets[0].getsockname()[1])
        monkeypatch.setattr(asyncio, "open_connection", reject_upstream)
        writer.write(b"POST / HTTP/1.1\r\nHost: pypi.org\r\n" + header + b"\r\n\r\n")
        await writer.drain()
        assert (await asyncio.wait_for(reader.read(), 1)).startswith(b"HTTP/1.1 403")
        writer.close()


def test_install_script_is_valid_bash():
    script = _install_script(DOMAINS)

    subprocess.run(["bash", "-n"], input=script, text=True, check=True)
    assert "python3 /tmp/feather-egress-proxy.py 15080 huggingface.co hf.co" in script


def test_all_public_tcp_ports_are_filtered_and_other_public_routes_are_closed():
    script = _install_script(DOMAINS)

    assert "iptables -t nat -A FEATHER_EGRESS -p tcp -j REDIRECT --to-ports 15080" in script
    assert "iptables -A FEATHER_EGRESS -p udp -j REJECT" in script
    assert "iptables -A FEATHER_EGRESS -j REJECT" in script
    assert "ip6tables -A FEATHER_EGRESS -j REJECT" in script
    assert "--dport" not in script
    for cidr in ("127.0.0.0/8", "10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16"):
        assert script.index(f"iptables -t nat -A FEATHER_EGRESS -d {cidr} -j RETURN") < script.index("-p tcp -j REDIRECT")
        assert script.index(f"iptables -A FEATHER_EGRESS -d {cidr} -j RETURN") < script.index("-p udp -j REJECT")
        assert script.index(f"iptables -A FEATHER_EGRESS -d {cidr} -j RETURN") < script.index("iptables -A FEATHER_EGRESS -j REJECT")
    assert script.index("ip6tables -A FEATHER_EGRESS -d fc00::/7 -j RETURN") < script.index("ip6tables -A FEATHER_EGRESS -j REJECT")


def test_rules_bind_solver_bridge_ingress_even_with_spoofed_or_unassigned_source_addresses():
    script = _install_script(DOMAINS)

    for command in ("iptables -t nat -I PREROUTING 1", "iptables -I FORWARD 1", "ip6tables -I FORWARD 1"):
        assert command + ' -m physdev --physdev-in "$agent_interface" -j FEATHER_EGRESS' in script
    assert 'for agent_interface in $agent_interfaces' in script
    assert " -s " not in script
    assert "agent_ipv4s" not in script and "agent_ipv6s" not in script
    assert "bridge-nf-call-iptables bridge-nf-call-ip6tables bridge-nf-filter-vlan-tagged" in script
    assert script.index("agent_interfaces=") < script.index("setsid nohup")
    assert script.index('test "$(cat "/proc/sys/net/bridge/$setting")" = 1') < script.index("setsid nohup")


@pytest.fixture
def interface_discovery_inputs():
    container = {"State": {"Running": True, "Pid": 123}, "NetworkSettings": {"Networks": {"primary": {}, "mcp": {}}}}
    solver = [{"ifname": "lo", "ifindex": 1}]
    host = []
    for index, name in enumerate(("vethprimary", "vethmcp"), 2):
        solver.append({"ifname": f"eth{index - 2}", "ifindex": index, "link_index": index + 10,
                       "linkinfo": {"info_kind": "veth"}})
        host.append({"ifname": name, "ifindex": index + 10, "link_index": index,
                     "master": f"br-{index}", "linkinfo": {"info_kind": "veth"}})
        host.append({"ifname": f"br-{index}", "ifindex": index + 20, "linkinfo": {"info_kind": "bridge"}})
    return container, solver, host


def _discover_interfaces(monkeypatch, inputs, capability_status=None):
    container, solver, host = inputs
    responses = {
        ("docker", "inspect", "agent"): [container],
        ("nsenter", "-t", "123", "-n", "ip", "-d", "-j", "link", "show"): solver,
        ("ip", "-d", "-j", "link", "show"): host,
    }
    calls = []

    def read(command, **kwargs):
        calls.append(command)
        return json.dumps(responses[command])

    monkeypatch.setattr(subprocess, "check_output", read)

    def read_status(path):
        assert path == "/proc/123/status"
        return io.StringIO(capability_status or "CapEff:\t00002000\nCapPrm:\t00002000\nCapBnd:\t00002000\n")

    monkeypatch.setattr(builtins, "open", read_status)
    output = io.StringIO()
    with redirect_stdout(output):
        exec(_interface_discovery_script(), {})
    assert calls == list(responses)
    return output.getvalue().strip().split()


def test_both_solver_networks_are_discovered_from_host_and_netns_peer_identity(monkeypatch, interface_discovery_inputs):
    assert _discover_interfaces(monkeypatch, interface_discovery_inputs) == ["vethprimary", "vethmcp"]


def test_kernel_default_down_tunnels_do_not_hide_verified_solver_veths(monkeypatch, interface_discovery_inputs):
    _, solver, _ = interface_discovery_inputs
    defaults = {"tunl0": "ipip", "gre0": "gre", "gretap0": "gretap", "erspan0": "erspan",
                "ip_vti0": "vti", "ip6_vti0": "vti6", "sit0": "sit", "ip6tnl0": "ip6tnl", "ip6gre0": "ip6gre"}
    solver.extend({"ifname": name, "ifindex": index, "flags": ["NOARP"], "operstate": "DOWN",
                   "linkinfo": {"info_kind": kind}} for index, (name, kind) in enumerate(defaults.items(), 30))

    assert _discover_interfaces(monkeypatch, interface_discovery_inputs) == ["vethprimary", "vethmcp"]


@pytest.mark.parametrize("interface", [
    {"ifname": "tunl0", "flags": ["UP", "NOARP"], "operstate": "DOWN", "linkinfo": {"info_kind": "ipip"}},
    {"ifname": "custom0", "flags": [], "linkinfo": {"info_kind": "ipip"}},
    {"ifname": "tunl0", "flags": [], "linkinfo": {"info_kind": "dummy"}},
    {"ifname": "tunl0", "linkinfo": {"info_kind": "ipip"}},
])
def test_active_unknown_or_unverifiable_non_veth_interfaces_fail_closed(monkeypatch, interface_discovery_inputs, interface):
    _, solver, _ = interface_discovery_inputs
    solver.append(interface)
    with pytest.raises(ValueError, match="not a veth"):
        _discover_interfaces(monkeypatch, interface_discovery_inputs)


@pytest.mark.parametrize("field", ["CapEff", "CapPrm", "CapBnd"])
def test_solver_net_admin_is_rejected_even_if_only_permitted_or_bounding(monkeypatch, interface_discovery_inputs, field):
    status = "".join(f"{name}:\t{0x3000 if name == field else 0x2000:08x}\n" for name in ("CapEff", "CapPrm", "CapBnd"))
    with pytest.raises(ValueError, match="NET_ADMIN"):
        _discover_interfaces(monkeypatch, interface_discovery_inputs, status)


@pytest.mark.parametrize("mismatch", ["peer", "reverse_peer", "bridge", "network_count", "non_veth", "unsafe_name", "stopped"])
def test_solver_interface_discovery_fails_closed_on_identity_or_topology_mismatch(monkeypatch, interface_discovery_inputs, mismatch):
    container, solver, host = interface_discovery_inputs
    if mismatch == "peer":
        solver[1]["link_index"] = 999
    elif mismatch == "reverse_peer":
        host[0]["link_index"] = 999
    elif mismatch == "bridge":
        host[1]["linkinfo"]["info_kind"] = "dummy"
    elif mismatch == "network_count":
        container["NetworkSettings"]["Networks"]["extra"] = {}
    elif mismatch == "non_veth":
        solver[1]["linkinfo"]["info_kind"] = "dummy"
    elif mismatch == "unsafe_name":
        host[0]["ifname"] = "veth+"
    else:
        container["State"]["Running"] = False
    with pytest.raises(ValueError):
        _discover_interfaces(monkeypatch, interface_discovery_inputs)


@pytest.mark.parametrize("data", [
    b"GET / HTTP/1.1\r\nHost: pypi.org\r\nHost: hf.co\r\n\r\n",
    b"GET / HTTP/1.1\r\nHost: hf.co:\r\n\r\n",
    b"GET / HTTP/1.1\r\nHost: hf.co:unknown\r\n\r\n",
    b"GET / HTTP/1.1\r\nHost: hf.co allowed.test\r\n\r\n",
    b"GET / HTTP/1.1\r\nHost: hf.co@allowed.test\r\n\r\n",
    b"CONNECT hf.co:443 HTTP/1.1\r\nHost: pypi.org\r\n\r\n",
    b"GET https://hf.co/file HTTP/1.1\r\nHost: pypi.org\r\n\r\n",
    b"GET / HTTP/1.1\r\n\r\n",
    b"SSH-2.0-client\r\n\r\n",
])
def test_unknown_or_conflicting_stream_authorities_are_not_admitted(data):
    assert requested_host(data) is None


async def test_unknown_protocol_and_nameless_tls_never_open_upstream(monkeypatch):
    upstream_connections = []

    async def upstream(reader, writer):
        upstream_connections.append(True)
        writer.close()

    server = await asyncio.start_server(upstream, "127.0.0.1", 0)
    address = server.sockets[0].getsockname()[:2]
    monkeypatch.setattr(egress_proxy, "_original_destination", lambda sock: address)
    egress = EgressFilter(DOMAINS)
    proxy = await asyncio.start_server(egress.handle, "127.0.0.1", 0)
    port = proxy.sockets[0].getsockname()[1]
    # A complete non-ClientHello TLS handshake names no server.
    payloads = (b"SSH-2.0-client\r\n\r\n", b"\x16\x03\x03\x00\x04\x02\x00\x00\x00")
    async with server, proxy:
        for payload in payloads:
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            writer.write(payload)
            await writer.drain()
            await asyncio.wait_for(reader.read(), 1)
            writer.close()
    assert upstream_connections == []


async def test_resolved_blocked_destination_cannot_be_disguised_as_allowed_host(monkeypatch):
    loop = asyncio.get_running_loop()
    original_resolve = loop.getaddrinfo

    async def resolve(domain, port, **kwargs):
        assert domain in DOMAINS
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("203.0.113.7", 0))]

    monkeypatch.setattr(loop, "getaddrinfo", resolve)
    egress = EgressFilter(DOMAINS)
    await egress.resolve()
    assert egress.blocked_addresses == {"203.0.113.7"}
    monkeypatch.setattr(loop, "getaddrinfo", original_resolve)
    monkeypatch.setattr(egress_proxy, "_original_destination", lambda sock: ("203.0.113.7", 8443))
    proxy = await asyncio.start_server(egress.handle, "127.0.0.1", 0)

    async def reject_upstream(*args, **kwargs):
        pytest.fail("blocked destination reached upstream")

    connect = asyncio.open_connection
    async with proxy:
        reader, writer = await connect("127.0.0.1", proxy.sockets[0].getsockname()[1])
        monkeypatch.setattr(asyncio, "open_connection", reject_upstream)
        writer.write(b"GET / HTTP/1.1\r\nHost: pypi.org\r\n\r\n")
        await writer.drain()
        assert (await asyncio.wait_for(reader.read(), 1)).startswith(b"HTTP/1.1 403")
        writer.close()


async def test_tls_sni_blocks_the_named_domain_and_preserves_allowed_stream(monkeypatch):
    received = []

    async def upstream(reader, writer):
        received.append(await reader.read(65536))
        writer.write(b"accepted")
        await writer.drain()
        writer.close()

    server = await asyncio.start_server(upstream, "127.0.0.1", 0)
    address = server.sockets[0].getsockname()[:2]
    monkeypatch.setattr(egress_proxy, "_original_destination", lambda sock: address)
    proxy = await asyncio.start_server(EgressFilter(DOMAINS).handle, "127.0.0.1", 0)
    allowed = _client_hello("pypi.org")
    async with server, proxy:
        for hello, expected in ((_client_hello("cdn.huggingface.co"), b""), (allowed, b"accepted")):
            reader, writer = await asyncio.open_connection("127.0.0.1", proxy.sockets[0].getsockname()[1])
            writer.write(hello)
            await writer.drain()
            assert await asyncio.wait_for(reader.read(), 1) == expected
            writer.close()
    assert received == [allowed]


def test_config_normalizes_domains_and_rejects_urls_and_wildcards():
    config = FeatherHarnessConfig(blocked_egress_domains=("HuggingFace.co.", "hf.co", "hf.co"))
    assert config.blocked_egress_domains == ("huggingface.co", "hf.co")

    for invalid in ("https://huggingface.co", "*.hf.co", "1.2.3.4", "huggingface.co; reboot"):
        with pytest.raises(ValidationError, match="blocked_egress_domains"):
            FeatherHarnessConfig(blocked_egress_domains=(invalid,))


async def test_solver_is_restricted_from_start_until_grading_begins(monkeypatch):
    from e2b import AsyncSandbox

    from feather_train.sandbox import compose, egress
    from feather_train.sandbox.lifecycle import ComposeSandbox

    events = []

    async def create(**kwargs):
        async def kill(**kwargs):
            events.append("kill")

        return SimpleNamespace(kill=kill, sandbox_id="sandbox")

    async def resume(*args, **kwargs):
        events.append("resume")
        return SimpleNamespace(state_export=None, verifier_network=None)

    async def record(name, *args, **kwargs):
        events.append(name)

    monkeypatch.setattr(AsyncSandbox, "create", create)
    monkeypatch.setattr(compose, "resume_compose_stack", resume)
    monkeypatch.setattr(compose, "assert_agent_container_survived", partial(record, "survived"))
    monkeypatch.setattr(compose, "prepare_compose_grading", partial(record, "grading"))
    monkeypatch.setattr(egress, "install_agent_egress_denylist", partial(record, "install"))
    monkeypatch.setattr(egress, "lift_agent_egress_denylist", partial(record, "lift"))
    task = SimpleNamespace(
        e2b_snapshot_id="baked-world",
        network_mode="public",
        mcp_servers=[],
        task_artifacts=SimpleNamespace(inputs=False),
        environment_mode="shared",
        submission_transport="text_response",
        collect_hooks=(),
        artifacts=(),
    )

    async with ComposeSandbox.start(task, blocked_egress_domains=DOMAINS) as world:
        events.append("agent")
        await world.prepare_grading(task)

    assert events == ["resume", "install", "agent", "survived", "lift", "grading", "kill"]
