"""Prepare a path-confined image for the model, downscaling it like Claude Code and ZCode do."""

import base64
import io
import os
from pathlib import Path

from _files import FileToolError

MAX_IMAGE_INPUT_BYTES = 20 * 1024 * 1024
# Providers reject larger base64 payloads; raw bytes expand by 4/3 when encoded.
MAX_IMAGE_BASE64_BYTES = 5 * 1024 * 1024
IMAGE_TARGET_BYTES = MAX_IMAGE_BASE64_BYTES * 3 // 4
MAX_IMAGE_DIMENSION = 2000
FALLBACK_IMAGE_DIMENSION = 1000
JPEG_QUALITIES = (80, 60, 40, 20)


def load_image(path: Path, display: str, metadata: os.stat_result) -> list[dict]:
    if metadata.st_size > MAX_IMAGE_INPUT_BYTES:
        raise FileToolError(
            f"File validation failed for {display}: Image is too large ({_megabytes(metadata.st_size)}). Maximum input size is 20 MB."
        )
    data = path.read_bytes()
    mime = _sniff(data)
    try:
        from PIL import Image
    except ImportError:
        # Hosts without Pillow can still send images that need no resizing.
        if mime is None:
            raise FileToolError(
                f"File validation failed for {display}: Expected a PNG, JPEG, GIF, or WebP image."
            ) from None
        if len(data) > IMAGE_TARGET_BYTES:
            raise FileToolError(
                f"File validation failed for {display}: Image is {_megabytes(len(data))}, and images above 3.75 MB must be resized, which is unavailable here. Save it at a lower resolution, then read it again."
            ) from None
        return _blocks(display, mime, data, None)

    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except (OSError, Image.DecompressionBombError) as exc:
        raise FileToolError(
            f"File validation failed for {display}: Expected a PNG, JPEG, GIF, or WebP image."
        ) from exc
    width, height = image.size
    if mime is not None and len(data) <= IMAGE_TARGET_BYTES and max(width, height) <= MAX_IMAGE_DIMENSION:
        return _blocks(display, mime, data, None)
    return _encode(image, display, mime)


def _encode(image, display: str, mime: str | None) -> list[dict]:
    from PIL import ImageOps

    original = image.size
    image = ImageOps.exif_transpose(image)
    image = _fit(image, MAX_IMAGE_DIMENSION)
    for encoded_mime, data in _candidates(image, mime):
        if len(data) <= IMAGE_TARGET_BYTES:
            return _blocks(display, encoded_mime, data, (original, image.size))
    image = _fit(image, FALLBACK_IMAGE_DIMENSION)
    data = _jpeg(image, JPEG_QUALITIES[-1])
    if len(data) > IMAGE_TARGET_BYTES:
        raise FileToolError(
            f"File validation failed for {display}: Image could not be reduced below 3.75 MB. Save a smaller crop, then read it again."
        )
    return _blocks(display, "image/jpeg", data, (original, image.size))


def _candidates(image, mime: str | None):
    if mime == "image/jpeg":
        for quality in JPEG_QUALITIES:
            yield "image/jpeg", _jpeg(image, quality)
        return
    # Lossless first keeps screenshots, charts, and transparency exact.
    yield "image/png", _png(image)
    yield "image/png", _png(image.convert("RGBA").quantize(256))
    for quality in JPEG_QUALITIES:
        yield "image/jpeg", _jpeg(image, quality)


def _fit(image, limit: int):
    from PIL import Image

    width, height = image.size
    if max(width, height) <= limit:
        return image
    scale = limit / max(width, height)
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return image.resize(size, Image.Resampling.LANCZOS)


def _png(image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def _jpeg(image, quality: int) -> bytes:
    from PIL import Image

    if image.mode in {"RGBA", "LA", "P"}:
        rgba = image.convert("RGBA")
        background = Image.new("RGB", rgba.size, "white")
        background.paste(rgba, mask=rgba.getchannel("A"))
        image = background
    elif image.mode != "RGB":
        image = image.convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality, optimize=True)
    return buffer.getvalue()


def _sniff(data: bytes) -> str | None:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def _blocks(display: str, mime: str, data: bytes, sizes) -> list[dict]:
    label = f"Image from {display}:"
    if sizes is not None and sizes[0] != sizes[1]:
        (width, height), (shown_width, shown_height) = sizes
        label = f"Image from {display} (original {width}x{height}, shown at {shown_width}x{shown_height}):"
    encoded = base64.b64encode(data).decode("ascii")
    return [
        {"type": "text", "text": label},
        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}},
    ]


def _megabytes(size: int) -> str:
    return f"{size / 1024 / 1024:.1f} MB"
