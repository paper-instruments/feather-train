"""Block the agent container from reaching configured domains, enforced on the E2B host.

Host rules bind the solver's bridge ingress interfaces rather than its spoofable source IP.
The agent lacks NET_ADMIN and cannot replace those host-owned interfaces. IPv4 TCP traffic to public
addresses is redirected into ``egress_proxy.py``, which admits named HTTP/TLS streams and
rejects denylisted TLS SNI and HTTP Host names. Other public IPv4 protocols and IPv6 are rejected so web clients
fall back to that path. Private-network DNS and service traffic remain available. A dead proxy
refuses redirected connections, so the control fails closed.
"""

from __future__ import annotations

import shlex
from pathlib import Path
from typing import TYPE_CHECKING

from feather_train.sandbox.compose import AGENT_CONTAINER, IPV6_LINK_LOCAL_CIDR, _sh
from feather_train.sandbox.retries import write_file

if TYPE_CHECKING:
    from feather_train.sandbox.lifecycle import SandboxIO

EGRESS_PROXY_PORT = 15080
EGRESS_PROXY_PATH = "/tmp/feather-egress-proxy.py"
EGRESS_PROXY_LOG = "/var/log/feather-egress-proxy.log"
EGRESS_CHAIN = "FEATHER_EGRESS"
_LOCAL_IPV4_CIDRS = (
    "0.0.0.0/8",
    "10.0.0.0/8",
    "100.64.0.0/10",
    "127.0.0.0/8",
    "169.254.0.0/16",
    "172.16.0.0/12",
    "192.168.0.0/16",
    "224.0.0.0/3",
)
_LOCAL_IPV6_CIDRS = ("::1/128", "fc00::/7", IPV6_LINK_LOCAL_CIDR, "ff00::/8")


async def install_agent_egress_denylist(session: SandboxIO, domains: tuple[str, ...]) -> None:
    """Start the host proxy, then route the agent's public web traffic through it."""
    await write_file(
        session.sandbox,
        EGRESS_PROXY_PATH,
        Path(__file__).with_name("egress_proxy.py").read_bytes(),
    )
    await _sh(
        session,
        _install_script(domains),
        timeout=120,
        check=True,
        operation="install agent egress denylist",
    )


async def lift_agent_egress_denylist(session: SandboxIO) -> None:
    """Restore unrestricted egress once the solver is done; graders are trusted."""
    await _sh(
        session,
        f"iptables -t nat -F {EGRESS_CHAIN} && iptables -F {EGRESS_CHAIN} && "
        f"{{ ! ip6tables -S {EGRESS_CHAIN} >/dev/null 2>&1 || ip6tables -F {EGRESS_CHAIN}; }}",
        check=True,
        operation="lift agent egress denylist",
    )


def _install_script(domains: tuple[str, ...]) -> str:
    chain = EGRESS_CHAIN
    ipv4_local = "\n".join(
        f"iptables -t nat -A {chain} -d {cidr} -j RETURN\niptables -A {chain} -d {cidr} -j RETURN"
        for cidr in _LOCAL_IPV4_CIDRS
    )
    ipv6_local = "\n".join(
        f"ip6tables -A {chain} -d {cidr} -j RETURN" for cidr in _LOCAL_IPV6_CIDRS
    )
    proxy = shlex.join(["python3", EGRESS_PROXY_PATH, str(EGRESS_PROXY_PORT), *domains])
    return f"""
set -e
agent_interfaces="$(python3 - <<'PY'
{_interface_discovery_script()}
PY
)"
test -n "$agent_interfaces" || {{ echo "agent has no verified bridge interfaces" >&2; exit 1; }}
test -e /proc/sys/net/bridge/bridge-nf-call-iptables || modprobe br_netfilter
for setting in bridge-nf-call-iptables bridge-nf-call-ip6tables bridge-nf-filter-vlan-tagged; do
    test -f "/proc/sys/net/bridge/$setting"
    echo 1 >"/proc/sys/net/bridge/$setting"
    test "$(cat "/proc/sys/net/bridge/$setting")" = 1
done
setsid nohup {proxy} >{EGRESS_PROXY_LOG} 2>&1 </dev/null &
for _ in $(seq 1 300); do grep -q READY {EGRESS_PROXY_LOG} && break; sleep 0.1; done
grep -q READY {EGRESS_PROXY_LOG} || {{ cat {EGRESS_PROXY_LOG} >&2; exit 1; }}
iptables -t nat -N {chain}
iptables -N {chain}
ip6tables -N {chain}
{ipv4_local}
iptables -t nat -A {chain} -p tcp -j REDIRECT --to-ports {EGRESS_PROXY_PORT}
iptables -A {chain} -p udp -j REJECT
iptables -A {chain} -j REJECT
{ipv6_local}
ip6tables -A {chain} -j REJECT
for agent_interface in $agent_interfaces; do
    iptables -t nat -I PREROUTING 1 -m physdev --physdev-in "$agent_interface" -j {chain}
    iptables -I FORWARD 1 -m physdev --physdev-in "$agent_interface" -j {chain}
    ip6tables -I FORWARD 1 -m physdev --physdev-in "$agent_interface" -j {chain}
done
"""


def _interface_discovery_script() -> str:
    return f"""import json
import re
import subprocess

def read(*command):
    return json.loads(subprocess.check_output(command, text=True))

container, = read('docker', 'inspect', '{AGENT_CONTAINER}')
pid = container['State']['Pid']
if not container['State']['Running'] or type(pid) is not int or pid <= 0:
    raise ValueError('solver is not running')
with open(f'/proc/{{pid}}/status') as status_file:
    status = dict(line.split(':', 1) for line in status_file if ':' in line)
if any(int(status[field].strip(), 16) & (1 << 12) for field in ('CapEff', 'CapPrm', 'CapBnd')):
    raise ValueError('solver can acquire NET_ADMIN')
solver_links = read('nsenter', '-t', str(pid), '-n', 'ip', '-d', '-j', 'link', 'show')
host_links = read('ip', '-d', '-j', 'link', 'show')
host_by_index = {{link['ifindex']: link for link in host_links}}
host_by_name = {{link['ifname']: link for link in host_links}}
interfaces = []
kernel_tunnels = {{'tunl0': 'ipip', 'gre0': 'gre', 'gretap0': 'gretap', 'erspan0': 'erspan',
                  'ip_vti0': 'vti', 'ip6_vti0': 'vti6', 'sit0': 'sit', 'ip6tnl0': 'ip6tnl',
                  'ip6gre0': 'ip6gre', 'ip6gretap0': 'ip6gretap', 'ip6erspan0': 'ip6erspan'}}
for link in solver_links:
    if link['ifname'] == 'lo':
        continue
    if link.get('linkinfo', {{}}).get('info_kind') != 'veth':
        flags = link.get('flags')
        if (link['ifname'] in kernel_tunnels
                and link.get('linkinfo', {{}}).get('info_kind') == kernel_tunnels[link['ifname']]
                and isinstance(flags, list) and 'UP' not in flags):
            continue
        raise ValueError('solver interface is not a veth')
    peer = host_by_index.get(link.get('link_index'), {{}})
    bridge = host_by_name.get(peer.get('master'), {{}})
    name = peer.get('ifname', '')
    if (peer.get('linkinfo', {{}}).get('info_kind') != 'veth'
            or peer.get('link_index') != link['ifindex']
            or bridge.get('linkinfo', {{}}).get('info_kind') != 'bridge'
            or not re.fullmatch(r'[a-zA-Z0-9_.-]{{1,15}}', name)):
        raise ValueError('solver veth peer/bridge identity mismatch')
    interfaces.append(name)
if (not interfaces or len(set(interfaces)) != len(interfaces)
        or len(interfaces) != len(container['NetworkSettings']['Networks'])):
    raise ValueError('solver network/interface count mismatch')
print(' '.join(interfaces))"""
