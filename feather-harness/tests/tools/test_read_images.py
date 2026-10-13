import base64
import json
from pathlib import Path

import pytest

from feather_harness import FileTools, LocalFileRuntime

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aOZkAAAAASUVORK5CYII="
)


@pytest.mark.parametrize("enabled", [False, True])
async def test_image_reads_require_capability_and_never_record_text_revisions(
    tmp_path: Path, enabled: bool
) -> None:
    (tmp_path / "page.png").write_bytes(PNG)
    files = FileTools(LocalFileRuntime(tmp_path), enable_image_reads=enabled)

    result = await files.read_tool().invoke(
        '{"file_path":"page.png","offset":100,"limit":1}'
    )

    if enabled:
        assert result.content == [
            {"type": "text", "text": "Image from page.png:"},
            {
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64," + base64.b64encode(PNG).decode()},
            },
        ]
    else:
        assert result.content == "Vision is not available in this environment."
    assert result.error is None
    assert result.metadata == {}
    assert files.revisions == {}


@pytest.mark.parametrize(
    "case, error, message",
    [
        ("invalid_format", "tool_error", "Expected a PNG, JPEG, GIF, or WebP image"),
        ("oversized", "tool_error", "Maximum input size is 20 MB"),
        ("outside", "invalid_path", "Provide a path inside the workspace"),
        ("symlink", "invalid_path", "Provide a path inside the workspace"),
    ],
)
async def test_image_reads_enforce_loader_boundaries(
    tmp_path: Path, case: str, error: str, message: str
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    image = workspace / "page.png"
    if case == "invalid_format":
        image.write_bytes(b"not an image")
    elif case == "oversized":
        with image.open("wb") as stream:
            stream.truncate(20 * 1024 * 1024 + 1)
    else:
        outside = tmp_path / "private.png"
        outside.write_bytes(PNG)
        if case == "symlink":
            image.symlink_to(outside)
        else:
            image = outside
    files = FileTools(LocalFileRuntime(workspace), enable_image_reads=True)

    result = await files.read_tool().invoke(json.dumps({"file_path": str(image)}))

    assert result.error == error
    assert message in result.content
    assert files.revisions == {}


@pytest.mark.parametrize("symlink", [False, True])
async def test_image_reads_respect_explicit_read_grants(tmp_path: Path, symlink: bool) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "source.png"
    outside.write_bytes(PNG)
    image = outside
    if symlink:
        image = workspace / "linked.png"
        image.symlink_to(outside)
    files = FileTools(
        LocalFileRuntime(workspace, read_grants=(outside,)), enable_image_reads=True
    )

    result = await files.read_tool().invoke(json.dumps({"file_path": str(image)}))

    assert result.error is None
    assert result.content[-1]["type"] == "image_url"
    assert files.revisions == {}
