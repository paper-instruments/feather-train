"""Stdlib-only transparent egress filter, run on the E2B host for the agent container.

iptables redirects all agent public IPv4 TCP here. Only named HTTP/TLS connections are admitted
by the requested TLS SNI or HTTP Host and original destination address. TLS is spliced unchanged;
plaintext HTTP forwards only one unambiguously framed request and forces connection close.
Nameless streams and explicit HTTP proxy tunnels fail closed. Encrypted HTTP Host names cannot
be inspected, so TLS domain fronting through an allowed SNI remains outside this filter.
"""

import asyncio
import html
import logging
import re
import socket
import struct
import sys
import unicodedata
from urllib.parse import unquote, urlsplit

SO_ORIGINAL_DST = 80
HEAD_LIMIT = 64 * 1024
HEAD_TIMEOUT_SECONDS = 10
CONNECT_TIMEOUT_SECONDS = 30
RESOLVE_INTERVAL_SECONDS = 60
NEED_MORE = object()
FORBIDDEN = (
    b"HTTP/1.1 403 Forbidden\r\nContent-Type: text/plain\r\nConnection: close\r\n\r\n"
    b"Blocked by the sandbox egress policy.\n"
)

logger = logging.getLogger("feather.egress")


def is_blocked_host(host: str, domains: tuple[str, ...]) -> bool:
    host = _normalized_host(host)
    return any(host == domain or host.endswith("." + domain) for domain in domains)


def requested_host(data: bytes):
    """Return the hostname a stream requests, None if it names none, or NEED_MORE."""
    if not data:
        return NEED_MORE
    if data[0] == 0x16:
        return _tls_server_name(data)
    return _http_host(data)


class EgressFilter:
    def __init__(self, domains: tuple[str, ...]) -> None:
        self.domains = domains
        # Addresses only accumulate: a name that once resolved somewhere stays blocked there.
        self.blocked_addresses: set[str] = set()

    async def resolve(self) -> None:
        loop = asyncio.get_running_loop()
        results = await asyncio.gather(
            *(loop.getaddrinfo(domain, None, family=socket.AF_INET) for domain in self.domains),
            return_exceptions=True,
        )
        for infos in results:
            if not isinstance(infos, BaseException):
                self.blocked_addresses.update(info[4][0] for info in infos)

    async def keep_resolving(self) -> None:
        while True:
            await asyncio.sleep(RESOLVE_INTERVAL_SECONDS)
            await self.resolve()

    async def handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        upstream = None
        try:
            destination = _original_destination(writer.get_extra_info("socket"))
            # A direct connection to the proxy has no NAT entry to splice to.
            if destination == writer.get_extra_info("sockname")[:2]:
                return
            head, host = await _read_head(reader)
            tls = head[:1] == b"\x16"
            http_request = None if tls else _http_request(head)
            if (not host or (not tls and http_request is None)
                    or destination[0] in self.blocked_addresses or is_blocked_host(host, self.domains)):
                logger.warning("blocked host=%s destination=%s:%s", host, *destination)
                if head[:1] != b"\x16":
                    writer.write(FORBIDDEN)
                    await writer.drain()
                return
            upstream_reader, upstream = await asyncio.wait_for(
                asyncio.open_connection(*destination), CONNECT_TIMEOUT_SECONDS
            )
            if tls:
                upstream.write(head)
                await asyncio.gather(_pipe(reader, upstream), _pipe(upstream_reader, writer))
            else:
                await _forward_http(reader, writer, upstream_reader, upstream, http_request)
        except (OSError, TimeoutError) as exc:
            logger.info("connection ended: %s", exc)
        finally:
            for stream in (upstream, writer):
                if stream is not None:
                    stream.close()


async def main(port: int, domains: tuple[str, ...]) -> None:
    egress = EgressFilter(domains)
    await egress.resolve()
    server = await asyncio.start_server(egress.handle, "0.0.0.0", port, reuse_address=True)
    logger.info("READY port=%s domains=%s", port, ",".join(domains))
    async with server:
        await asyncio.gather(server.serve_forever(), egress.keep_resolving())


def _tls_server_name(data: bytes):
    handshake = b""
    offset = 0
    while True:
        if len(data) < offset + 5:
            return NEED_MORE
        if data[offset] != 0x16:
            return None
        end = offset + 5 + int.from_bytes(data[offset + 3 : offset + 5])
        if len(data) < end:
            return NEED_MORE
        handshake += data[offset + 5 : end]
        offset = end
        if len(handshake) < 4:
            continue
        if handshake[0] != 1:
            return None
        length = int.from_bytes(handshake[1:4])
        if len(handshake) >= 4 + length:
            return _client_hello_server_name(handshake[4 : 4 + length])


def _client_hello_server_name(body: bytes) -> str | None:
    try:
        position = 2 + 32
        position += 1 + body[position]
        position += 2 + int.from_bytes(body[position : position + 2])
        position += 1 + body[position]
        end = position + 2 + int.from_bytes(body[position : position + 2])
        position += 2
        while position + 4 <= end:
            extension = int.from_bytes(body[position : position + 2])
            length = int.from_bytes(body[position + 2 : position + 4])
            position += 4
            if extension == 0:
                # server_name_list: u16 list length, u8 name type (0 = host_name), u16 length.
                if body[position + 2] != 0:
                    return None
                name_length = int.from_bytes(body[position + 3 : position + 5])
                return body[position + 5 : position + 5 + name_length].decode("ascii")
            position += length
    except (IndexError, UnicodeDecodeError):
        return None
    return None


def _http_host(data: bytes):
    end = data.find(b"\r\n\r\n")
    if end < 0:
        return NEED_MORE
    lines = data[:end].split(b"\r\n")
    request = lines[0].split()
    if len(request) != 3 or not request[2].startswith(b"HTTP/") or request[0].upper() == b"CONNECT":
        return None
    hosts = []
    for line in lines[1:]:
        name, separator, value = line.partition(b":")
        if separator and name.strip().lower() == b"host":
            host = value.strip().decode("latin-1")
            if ":" in host:
                host, port = host.split(":", 1)
                if len(port) > 5 or not port.isascii() or not port.isdigit() or not 0 < int(port) <= 65535:
                    return None
            normalized = _normalized_host(host)
            if not all(re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", label)
                       for label in normalized.split(".")):
                return None
            hosts.append(host)
    if len(hosts) != 1 or not hosts[0]:
        return None
    # Absolute-form requests can ask a forward proxy for a different destination.
    try:
        authority = urlsplit(request[1].decode("ascii")).hostname
    except (ValueError, UnicodeDecodeError):
        return None
    if authority and _normalized_host(authority) != _normalized_host(hosts[0]):
        return None
    return hosts[0]


def _normalized_host(host: str) -> str:
    normalized = unicodedata.normalize("NFKC", host).casefold()
    while True:
        decoded = html.unescape(unquote(normalized))
        if decoded == normalized:
            break
        normalized = unicodedata.normalize("NFKC", decoded).casefold()
    try:
        return normalized.rstrip(".").encode("idna").decode("ascii")
    except UnicodeError:
        return normalized.rstrip(".")


def _http_request(data: bytes) -> tuple[bytes, bytes, int] | None:
    end = data.find(b"\r\n\r\n")
    if end < 0:
        return None
    lines = data[:end].split(b"\r\n")
    request = lines[0].split(b" ")
    if (len(request) != 3 or not re.fullmatch(rb"[A-Z]+", request[0])
            or request[2] not in (b"HTTP/1.0", b"HTTP/1.1")
            or any(byte <= 32 or byte == 127 for byte in request[1])):
        return None
    headers = [lines[0]]
    content_lengths = []
    for line in lines[1:]:
        name, separator, value = line.partition(b":")
        if (not separator or not re.fullmatch(rb"[!#$%&'*+.^_`|~0-9A-Za-z-]+", name)
                or any(byte < 32 and byte != 9 or byte == 127 for byte in value)):
            return None
        if name.lower() in (b"transfer-encoding", b"upgrade"):
            return None
        if name.lower() == b"content-length":
            length = value.strip()
            if not re.fullmatch(rb"[0-9]+", length) or len(length) > 20:
                return None
            content_lengths.append(int(length))
        if name.lower() not in (b"connection", b"proxy-connection"):
            headers.append(line)
    if len(content_lengths) > 1:
        return None
    length = content_lengths[0] if content_lengths else 0
    body = data[end + 4 : end + 4 + length]
    return b"\r\n".join(headers) + b"\r\nConnection: close\r\n\r\n", body, length - len(body)


async def _forward_http(reader, writer, upstream_reader, upstream, request) -> None:
    headers, body, remaining = request
    upstream.write(headers + body)
    await upstream.drain()
    sending = asyncio.create_task(_send_http_body(reader, upstream, remaining))
    receiving = asyncio.create_task(_pipe(upstream_reader, writer))
    try:
        done, _ = await asyncio.wait((sending, receiving), return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            task.result()
        if receiving not in done:
            await receiving
    finally:
        for task in (sending, receiving):
            if not task.done():
                task.cancel()
        await asyncio.gather(sending, receiving, return_exceptions=True)


async def _send_http_body(reader, upstream, remaining: int) -> None:
    while remaining:
        chunk = await reader.read(min(remaining, 65536))
        if not chunk:
            raise OSError("incomplete HTTP request body")
        upstream.write(chunk)
        await upstream.drain()
        remaining -= len(chunk)
    if upstream.can_write_eof():
        upstream.write_eof()


async def _read_head(reader: asyncio.StreamReader) -> tuple[bytes, str | None]:
    data = b""
    deadline = asyncio.get_running_loop().time() + HEAD_TIMEOUT_SECONDS
    while (host := requested_host(data)) is NEED_MORE and len(data) < HEAD_LIMIT:
        try:
            async with asyncio.timeout_at(deadline):
                chunk = await reader.read(HEAD_LIMIT - len(data))
        except TimeoutError:
            break
        if not chunk:
            break
        data += chunk
    return data, None if host is NEED_MORE else host


async def _pipe(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        while chunk := await reader.read(65536):
            writer.write(chunk)
            await writer.drain()
        if writer.can_write_eof():
            writer.write_eof()
    except OSError:
        pass


def _original_destination(sock) -> tuple[str, int]:
    raw = sock.getsockopt(socket.IPPROTO_IP, SO_ORIGINAL_DST, 16)
    port, address = struct.unpack_from("!2xH4s", raw)
    return socket.inet_ntoa(address), port


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s %(message)s")
    asyncio.run(main(int(sys.argv[1]), tuple(sys.argv[2:])))
