import base64
import io
import json
import os
import sys
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from feather_harness import FileTools, LocalFileRuntime

SCRIPTS = Path(__file__).parents[2] / "src/feather_harness/scripts"
BASE64_LIMIT = 5 * 1024 * 1024


def _decode(block: dict) -> tuple[str, Image.Image, int]:
    header, encoded = block["image_url"]["url"].split(",", 1)
    data = base64.b64decode(encoded)
    return header.removeprefix("data:").removesuffix(";base64"), Image.open(io.BytesIO(data)), len(encoded)


async def _read(workspace: Path, name: str, *, image_reads: bool = True):
    files = FileTools(LocalFileRuntime(workspace), enable_image_reads=image_reads)
    return await files.read_tool().invoke(json.dumps({"file_path": name}))


async def test_oversized_screenshot_is_downscaled_losslessly_and_labeled(tmp_path: Path) -> None:
    image = Image.new("RGB", (4000, 2500), "white")
    ImageDraw.Draw(image).rectangle((400, 400, 3600, 2100), outline="black", width=12)
    image.save(tmp_path / "chart.png")

    result = await _read(tmp_path, "chart.png")

    assert result.error is None
    assert result.content[0]["text"] == "Image from chart.png (original 4000x2500, shown at 2000x1250):"
    mime, shown, encoded_size = _decode(result.content[1])
    assert mime == "image/png"
    assert shown.size == (2000, 1250)
    assert encoded_size <= BASE64_LIMIT


@pytest.mark.parametrize("name, fmt", [("scan.png", "PNG"), ("photo.jpg", "JPEG")])
async def test_incompressible_images_are_reencoded_within_the_provider_limit(
    tmp_path: Path, name: str, fmt: str
) -> None:
    noise = Image.frombytes("RGB", (2400, 2400), os.urandom(2400 * 2400 * 3))
    noise.save(tmp_path / name, format=fmt, quality=100)

    result = await _read(tmp_path, name)

    mime, shown, encoded_size = _decode(result.content[1])
    assert mime == ("image/png" if fmt == "PNG" else "image/jpeg")
    assert shown.size == (2000, 2000)
    assert encoded_size <= BASE64_LIMIT


async def test_tiff_scans_are_converted_to_png(tmp_path: Path) -> None:
    Image.new("RGB", (300, 200), "white").save(tmp_path / "scan.tiff")

    result = await _read(tmp_path, "scan.tiff")

    mime, shown, _ = _decode(result.content[1])
    assert (mime, shown.size) == ("image/png", (300, 200))
    assert result.content[0]["text"] == "Image from scan.tiff:"


def test_without_pillow_small_images_pass_and_large_ones_ask_for_a_smaller_render(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    small = tmp_path / "small.png"
    Image.new("RGB", (10, 10), "white").save(small)
    large = tmp_path / "large.png"
    large.write_bytes(small.read_bytes() + b"\0" * (4 * 1024 * 1024))
    monkeypatch.syspath_prepend(str(SCRIPTS))
    monkeypatch.setitem(sys.modules, "PIL", None)
    from _files import FileToolError
    from _images import load_image

    assert load_image(small, "small.png", small.stat())[1]["image_url"]["url"].startswith(
        "data:image/png;base64,"
    )
    with pytest.raises(FileToolError, match="Save it at a lower resolution"):
        load_image(large, "large.png", large.stat())


@pytest.mark.parametrize(
    "image_reads, accepted, pointer",
    [(True, "text and image files only", "Read its contents through the pdf skill."), (False, "text files only", None)],
)
async def test_read_errors_state_what_read_accepts(
    tmp_path: Path, image_reads: bool, accepted: str, pointer: str | None
) -> None:
    (tmp_path / "memo.pdf").write_bytes(b"%PDF-1.7\n")
    (tmp_path / "blob.bin").write_bytes(b"\0\1\2" * 100)

    document = await _read(tmp_path, "memo.pdf", image_reads=image_reads)
    binary = await _read(tmp_path, "blob.bin", image_reads=image_reads)

    assert f"This tool reads {accepted}; the requested path has a binary document format." in document.content
    assert (pointer in document.content) if pointer else ("skill" not in document.content)
    assert f"This tool reads {accepted}; the file appears to be binary." in binary.content
