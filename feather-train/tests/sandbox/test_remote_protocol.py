"""Exercise image reads through the actual host result protocol."""

import base64
import json
import struct
import sys
import zlib
from pathlib import Path

import pytest
from feather_harness.local.filesystem import SCRIPTS

from feather_train.sandbox import remote


# The largest image read passes through unchanged just under 3.75 MB, about 5 MB of base64.
@pytest.mark.parametrize("image_bytes", [1_100_000, 3_700_000])
async def test_image_read_result_survives_host_publication_and_recovery(
    tmp_path, monkeypatch, capsys, image_bytes
):
    image = _png(image_bytes)
    (tmp_path / "page.png").write_bytes(image)
    monkeypatch.setenv("PYTHONPATH", str(SCRIPTS))
    path = tmp_path / "call.json"
    request = {
        "name": "read",
        "arguments": {"file_path": "page.png", "offset": 1, "limit": 2000, "image_reads": True},
        "config": {
            "workspace": str(tmp_path),
            "read_grants": [],
            "write_grants": [],
        },
        "background": False,
        "docker": [sys.executable, str(Path(remote.__file__).with_name("call.py"))],
    }
    await remote.run(path, json.dumps(request))
    saved = await remote.inspect(path, wait=True)
    assert "failure" not in saved
    assert json.loads(capsys.readouterr().out) == saved
    encoded = saved["result"]["content"][1]["image_url"]["url"].split(",", 1)[1]
    assert base64.b64decode(encoded) == image


async def test_nonimage_protocol_output_retains_existing_bound(tmp_path, capsys):
    path = tmp_path / "call.json"
    request = {
        "name": "bash",
        "background": False,
        "docker": [
            sys.executable,
            "-c",
            "import json,sys; json.load(sys.stdin); print(json.dumps({'done':True,'result':{'content':'x'*1100000}}))",
        ],
    }
    await remote.run(path, json.dumps(request))
    result = await remote.inspect(path, wait=True)
    assert "exceed the limit" in result["failure"]
    assert json.loads(capsys.readouterr().out) == result


def _png(size):
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    prefix = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
    prefix += chunk(b"IDAT", zlib.compress(b"\0\xff\0\0"))
    suffix = chunk(b"IEND", b"")
    return prefix + chunk(b"tEXt", b"Comment\0" + b"x" * (size - len(prefix) - len(suffix) - 20)) + suffix
