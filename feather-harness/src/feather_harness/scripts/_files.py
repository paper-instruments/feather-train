"""Filesystem checks and text handling shared by the file workers."""

import errno
import hashlib
import json
import os
import re
import stat
import sys
import tempfile
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn

from charset_normalizer import from_bytes

MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024
MAX_OUTPUT_CHARS = 24000
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tif", ".tiff"}
DOCUMENT_EXTENSIONS = {
    ".doc",
    ".docm",
    ".docx",
    ".pdf",
    ".ppt",
    ".pptm",
    ".pptx",
    ".xls",
    ".xlsb",
    ".xlsm",
    ".xlsx",
}

# Byte groups used by the binaryornot 0.4.4 heuristic below. ASCII text includes
# ordinary whitespace; bytes above ASCII may belong to an encoded text character.
_PRINTABLE_ASCII = b"\n\r\t\f\b" + bytes(range(32, 127))
_PRINTABLE_HIGH = bytes(range(127, 256))
# Recognizable file headers reject binary formats even when the filename has a text extension.
_BINARY_MAGIC_PREFIXES = (
    b"%PDF-",  # PDF
    b"\x89PNG\r\n\x1a\n",  # PNG
    b"\xff\xd8\xff",  # JPEG
    b"GIF87a",
    b"GIF89a",
    b"PK\x03\x04",  # ZIP, including modern Office documents
    b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1",  # OLE container, including older Office documents
    b"\x7fELF",  # ELF executable or library
    b"\x1f\x8b",  # gzip
)


class FileToolError(Exception):
    def __init__(self, content: str, category: str = "tool_error") -> None:
        super().__init__(content)
        self.category = category


def invalid_path(display: str, parameter: str = "file_path") -> NoReturn:
    raise FileToolError(
        f"Invalid `{parameter}` parameter: {display}. Provide a path inside the workspace; "
        "relative paths resolve from its root and absolute in-workspace paths are also accepted "
        "(for example, notes/report.md).",
        "invalid_path",
    )


@dataclass(frozen=True)
class WriteTarget:
    """A file write that path validation approved.

    ``display`` is the path the model asked for; revisions and messages use it.
    ``path`` is the resolved target that validation checked, and writes reach it
    through ``boundary`` (a resolved write root, or an exact file grant's parent).
    Carrying the target means an alias re-pointed after validation cannot
    redirect the write, and no second resolve can pick an unchecked target.
    """

    display: str
    path: Path
    boundary: Path
    relative: Path
    metadata: os.stat_result | None
    exact_grant: bool


def require_file(
    root: str | Path,
    read_grants: Sequence[str],
    raw: str,
    *,
    write_grants: Sequence[str] = (),
    for_write: bool = False,
    allow_missing: bool = False,
) -> tuple[Path, str, os.stat_result | None]:
    candidate, _target, display, metadata = _require_file(
        root,
        read_grants,
        raw,
        write_grants=write_grants,
        for_write=for_write,
        allow_missing=allow_missing,
    )
    return candidate, display, metadata


def require_write_target(
    root: str | Path,
    read_grants: Sequence[str],
    raw: str,
    *,
    write_grants: Sequence[str] = (),
    allow_missing: bool = False,
) -> WriteTarget:
    _candidate, target, display, metadata = _require_file(
        root,
        read_grants,
        raw,
        write_grants=write_grants,
        for_write=True,
        allow_missing=allow_missing,
    )
    boundary, relative, exact_grant = _write_boundary(
        target, Path(root).resolve(), [Path(grant).resolve() for grant in write_grants]
    )
    return WriteTarget(display, target, boundary, relative, metadata, exact_grant)


def _require_file(
    root: str | Path,
    read_grants: Sequence[str],
    raw: str,
    *,
    write_grants: Sequence[str],
    for_write: bool,
    allow_missing: bool,
) -> tuple[Path, Path, str, os.stat_result | None]:
    candidate, target, display, metadata = _validate_path(
        root,
        read_grants,
        raw,
        write_grants=write_grants,
        for_write=for_write,
        allow_missing=allow_missing,
    )
    if metadata is None:
        if allow_missing:
            return candidate, target, display, None
        raise FileToolError(f"Invalid `file_path` parameter: {raw}. The file does not exist.")
    if stat.S_ISDIR(metadata.st_mode):
        if allow_missing:
            raise FileToolError(
                f"Invalid `file_path` parameter: {raw}. The path is a directory; provide a file path."
            )
        raise FileToolError(
            f"Invalid `file_path` parameter: {raw}. The path is a directory; a regular file is required."
        )
    if not stat.S_ISREG(metadata.st_mode):
        raise FileToolError(
            f"Invalid `file_path` parameter: {raw}. The path is not a regular file."
        )
    return candidate, target, display, metadata


def _from_msys_path(raw: str) -> str:
    # Git Bash prints /c/Users/me/project for C:\\Users\\me\\project. Models copy
    # those paths from `pwd` or `ls` output into file tools.
    match = re.match(r"^/([A-Za-z])(?=/|$)(.*)$", raw)
    return f"{match.group(1).upper()}:{match.group(2) or '/'}" if match else raw


def require_path(
    root: str | Path,
    read_grants: Sequence[str],
    raw: str,
    *,
    write_grants: Sequence[str] = (),
    for_write: bool = False,
    allow_missing: bool = False,
    parameter: str = "file_path",
) -> tuple[Path, str, os.stat_result | None]:
    candidate, _target, display, metadata = _validate_path(
        root,
        read_grants,
        raw,
        write_grants=write_grants,
        for_write=for_write,
        allow_missing=allow_missing,
        parameter=parameter,
    )
    return candidate, display, metadata


def _validate_path(
    root: str | Path,
    read_grants: Sequence[str],
    raw: str,
    *,
    write_grants: Sequence[str] = (),
    for_write: bool = False,
    allow_missing: bool = False,
    parameter: str = "file_path",
) -> tuple[Path, Path, str, os.stat_result | None]:
    """Return the requested path, its resolved target, display text, and metadata."""
    output_directory = _output_directory(root)
    if not output_directory.is_symlink() and output_directory.resolve() == output_directory:
        read_grants = (*read_grants, str(output_directory))
    grants = [*read_grants, *write_grants]
    # Drive-letter paths are native on Windows (and still confined below);
    # elsewhere they would silently join under the workspace root.
    if os.name != "nt" and re.match(r"^[A-Za-z]:[/\\]", raw):
        invalid_path(raw, parameter)
    if os.name == "nt":
        raw = _from_msys_path(raw)
    requested = Path(raw.replace("\\", "/"))
    candidate = Path(os.path.normpath(os.path.join(root, requested)))
    if not requested.is_absolute() and not candidate.is_relative_to(root):
        invalid_path(raw, parameter)
    if candidate.is_relative_to(root):
        display = str(candidate.relative_to(root))
    else:
        display = str(candidate)

    # Check the requested path first, then its symlink target. Both must be allowed.
    roots = [Path(root), *(Path(grant) for grant in grants)]
    containing_roots = [boundary for boundary in roots if candidate.is_relative_to(boundary)]
    if not containing_roots or "\x00" in raw:
        invalid_path(raw, parameter)
    if for_write and not candidate.is_relative_to(root):
        # A narrower read-only grant takes precedence over a writable parent.
        grant = max(containing_roots, key=lambda path: len(path.parts))
        if str(grant) in read_grants:
            invalid_path(display, parameter)

    resolved_roots = [boundary.resolve() for boundary in roots]
    for grant, resolved_grant in zip(grants, resolved_roots[1:]):
        if resolved_grant == resolved_grant.parent:
            raise ValueError(f"extra path grant must not resolve to filesystem root: {grant}")
    resolved = candidate.resolve()
    if not any(resolved.is_relative_to(boundary) for boundary in resolved_roots):
        invalid_path(display, parameter)
    if for_write and not resolved.is_relative_to(resolved_roots[0]):
        resolved_containing_roots = [
            boundary for boundary in resolved_roots[1:] if resolved.is_relative_to(boundary)
        ]
        grant = max(resolved_containing_roots, key=lambda path: len(path.parts))
        readonly_roots = {Path(path).resolve() for path in read_grants}
        if grant in readonly_roots:
            invalid_path(display, parameter)

    boundary = max(containing_roots, key=lambda path: len(path.parts))
    for parent in reversed(candidate.parents):
        if not parent.is_relative_to(boundary):
            continue
        try:
            metadata = parent.stat()
        except NotADirectoryError:
            raise FileToolError(
                f"Invalid `{parameter}` parameter: {display}. The parent component {parent.parent.name!r} is not a directory."
            ) from None
        except FileNotFoundError:
            if allow_missing:
                return candidate, resolved, display, None
            raise FileToolError(
                f"Invalid `{parameter}` parameter: {display}. A parent directory does not exist."
            ) from None
        if not stat.S_ISDIR(metadata.st_mode):
            raise FileToolError(
                f"Invalid `{parameter}` parameter: {display}. The parent component {parent.name!r} is not a directory."
            )
    try:
        metadata = candidate.stat()
    except NotADirectoryError:
        raise FileToolError(
            f"Invalid `{parameter}` parameter: {display}. The parent component {candidate.parent.name!r} is not a directory."
        ) from None
    except FileNotFoundError:
        return candidate, resolved, display, None
    return candidate, resolved, display, metadata


def _is_binary(data: bytes) -> bool:
    if not data:
        return False
    if data.startswith(_BINARY_MAGIC_PREFIXES):
        return True
    match = from_bytes(data).best()
    encoding = "ascii"
    if match is not None:
        encoding = (match.encoding or "ascii").lower()
    # NUL bytes usually indicate binary data, but UTF-16/32 text commonly contains them.
    # A byte-order mark (BOM) identifies those encodings and their byte order.
    if b"\x00" in data:
        wide_text_bom = data.startswith(
            (
                b"\xff\xfe",
                b"\xfe\xff",
                b"\xff\xfe\x00\x00",
                b"\x00\x00\xfe\xff",
            )
        )
        if not (wide_text_bom and encoding.startswith(("utf_16", "utf_32"))):
            return True
        try:
            data.decode(encoding)
        except UnicodeDecodeError:
            return True
    # translate(None, group) deletes that group: these ratios count bytes outside it.
    # Keep the inherited thresholds so unusual text is classified as before; an
    # encoding detected by charset-normalizer can override the byte heuristic.
    outside_ascii_text_ratio = len(data.translate(None, _PRINTABLE_ASCII)) / len(data)
    below_127_ratio = len(data.translate(None, _PRINTABLE_HIGH)) / len(data)
    likely_binary = (outside_ascii_text_ratio > 0.3 and below_127_ratio < 0.05) or (
        outside_ascii_text_ratio > 0.8 and below_127_ratio > 0.8
    )
    detected_text_encoding = match is not None and encoding != "ascii"
    if likely_binary:
        return not detected_text_encoding
    if detected_text_encoding:
        return False
    return b"\x00" in data or b"\xff" in data


def save_and_truncate_output(
    content: str, tool_name: str = "read", *, workspace: str | Path
) -> str:
    """Save oversized output where this worker runs, then return its bounded preview."""
    head, notice, tail, _ = save_output_preview(content, tool_name, workspace=workspace)
    return head + notice + tail


def save_output_preview(
    content: str, tool_name: str = "read", *, workspace: str | Path
) -> tuple[str, str, str, str | None]:
    """Return preview boundaries and the path only after the complete output is saved."""
    if len(content) <= MAX_OUTPUT_CHARS:
        return content, "", "", None
    saved_path = None
    try:
        directory = _output_directory(workspace)
        directory.mkdir(mode=0o700, exist_ok=True)
        if _supports_safe_relative_write():
            directory_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            try:
                os.fchmod(directory_fd, 0o700)
                filename = f"{tool_name}_{uuid.uuid4().hex[:8]}.txt"
                descriptor = os.open(
                    filename, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600, dir_fd=directory_fd
                )
                with os.fdopen(descriptor, "wb") as output:
                    output.write(content.encode("utf-8"))
                saved_path = str(directory / filename)
            finally:
                os.close(directory_fd)
        else:
            # Windows lacks directory-relative opens; reject redirected output roots.
            if directory.is_symlink() or directory.resolve() != directory:
                raise OSError("tool-output directory must not be redirected")
            path = directory / f"{tool_name}_{uuid.uuid4().hex[:8]}.txt"
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, "wb") as output:
                output.write(content.encode("utf-8"))
            saved_path = str(path)
    except OSError:
        pass  # Output persistence is best effort, as in the pinned tool.

    return *truncate_output_parts(content, saved_path), saved_path


def truncate_output(content: str, saved_path: str | None = None) -> str:
    """Format the legacy head/tail preview, including its notice within the cap."""
    return "".join(truncate_output_parts(content, saved_path))


def truncate_output_parts(
    content: str, saved_path: str | None = None
) -> tuple[str, str, str]:
    if len(content) <= MAX_OUTPUT_CHARS:
        return content, "", ""

    def notice(head: int, tail: int) -> str:
        text = f"[Output truncated: kept {head} leading and {tail} trailing characters from {len(content)} total. "
        if saved_path:
            text += "Use read offsets for lines; for long single-line text/JSON, use a shell command to slice the saved file. "
            text += f"Complete returned output saved at {saved_path}. "
        return text.rstrip() + "]"

    reserve = len("\n\n..." + notice(MAX_OUTPUT_CHARS, MAX_OUTPUT_CHARS) + "...\n\n")
    budget = MAX_OUTPUT_CHARS - reserve
    head = budget // 2
    tail = budget - head
    return content[:head], "\n\n..." + notice(head, tail) + "...\n\n", content[-tail:]


def load_text(
    path: Path, display: str, metadata: os.stat_result, *, scope: str = "reads text files"
) -> tuple[str, str, str]:
    if metadata.st_size > MAX_FILE_SIZE_BYTES:
        raise FileToolError(
            f"File validation failed for {display}: File is too large ({metadata.st_size / 1024 / 1024:.1f}MB). Maximum allowed size is 10MB."
        )
    # Read raw bytes so decoding does not silently normalize CRLF line endings.
    data = path.read_bytes()
    if _is_binary(data[:1024]):
        raise FileToolError(
            f"File validation failed for {display}: This tool {scope} only; the file appears to be binary."
        )
    # Short UTF-8 text is easy to misidentify. Guess only if strict decoding fails.
    encoding = "utf-8-sig" if data.startswith(b"\xef\xbb\xbf") else "utf-8"
    try:
        text = data.decode(encoding)
    except UnicodeDecodeError:
        match = from_bytes(data).best()
        encoding = match.encoding if match is not None and match.encoding else "utf-8"
        try:
            text = data.decode(encoding)
        except (LookupError, UnicodeDecodeError):
            encoding = "utf-8"
            text = data.decode(encoding, errors="replace")
    # Hash original bytes: even an encoding or newline-only change makes a read stale.
    return text, encoding, hashlib.sha256(data).hexdigest()


def validate_write_extension(path: Path, display: str) -> None:
    if path.suffix.lower() in IMAGE_EXTENSIONS | DOCUMENT_EXTENSIONS:
        raise FileToolError(
            f"File validation failed for {display}: This tool writes text files only; the requested extension belongs to a binary document or media format."
        )


def require_current_read(revisions: Mapping[str, str], display: str) -> str:
    previous = revisions.get(display)
    if previous is None:
        raise FileToolError(
            f"Read {display} before modifying it so its current contents are known."
        )
    return previous


def check_revision(display: str, expected: str, actual: str) -> None:
    if expected != actual:
        raise FileToolError(
            f"{display} changed since it was read. Read it again before modifying it."
        )


def write_text(
    target: WriteTarget,
    content: str,
    encoding: str,
    *,
    create_parents: bool = False,
) -> str:
    # Encode before opening the file for writing, so an unrepresentable character
    # cannot truncate the existing file. Preserve its detected encoding.
    try:
        data = content.encode(encoding)
    except UnicodeEncodeError:
        raise FileToolError(
            f"The updated content cannot be represented in {target.display}'s current {encoding} encoding; no changes were made."
        ) from None
    if os.name == "nt" or not _supports_safe_relative_write():
        # No dir_fd/O_NOFOLLOW here (Windows), so this path cannot rule out a
        # concurrent junction swap. The validated target is already resolved, so
        # a link or junction added on it since makes it resolve elsewhere;
        # re-check that right before each mutation.
        def require_unchanged() -> None:
            if target.path.resolve() != target.path:
                _unsafe_write_path(target.display)

        require_unchanged()
        if create_parents:
            target.path.parent.mkdir(parents=True, exist_ok=True)
            require_unchanged()
        target.path.write_bytes(data)
    else:
        _write_bytes_relative(target, data, create_parents=create_parents)
    return hashlib.sha256(data).hexdigest()


def _unsafe_write_path(display: str) -> NoReturn:
    raise FileToolError(
        f"File validation failed for {display}: the write path could not be opened safely."
    )


def _supports_safe_relative_write() -> bool:
    return (
        hasattr(os, "O_DIRECTORY")
        and hasattr(os, "O_NOFOLLOW")
        and os.open in os.supports_dir_fd
        and os.mkdir in os.supports_dir_fd
    )


def _write_bytes_relative(target: WriteTarget, data: bytes, *, create_parents: bool) -> None:
    # Walk the validated target from its resolved boundary with O_NOFOLLOW on
    # every component, so a link swapped onto that path fails the walk. Only
    # components below the boundary are protected; the boundary is opened by
    # its absolute path.
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    try:
        directory_fd = os.open(target.boundary, directory_flags)
    except OSError as exc:
        _raise_path_error(exc, target.display)
    try:
        for component in target.relative.parts[:-1]:
            try:
                next_fd = os.open(component, directory_flags, dir_fd=directory_fd)
            except FileNotFoundError:
                if not create_parents:
                    raise
                os.mkdir(component, mode=0o777, dir_fd=directory_fd)
                next_fd = os.open(component, directory_flags, dir_fd=directory_fd)
            os.close(directory_fd)
            directory_fd = next_fd
        file_flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW
        file_fd = os.open(target.relative.name, file_flags, 0o666, dir_fd=directory_fd)
        try:
            view = memoryview(data)
            while view:
                written = os.write(file_fd, view)
                view = view[written:]
        finally:
            os.close(file_fd)
    except OSError as exc:
        _raise_path_error(exc, target.display)
    finally:
        os.close(directory_fd)


def _raise_path_error(exc: OSError, display: str) -> NoReturn:
    # A link swapped onto the path (ELOOP), or a component that is missing or
    # is not a folder (for example an exact file grant whose parent is gone).
    if exc.errno in {errno.ELOOP, errno.ENOENT, errno.ENOTDIR, getattr(errno, "ESTALE", -1)}:
        _unsafe_write_path(display)
    raise exc


def _write_boundary(
    target: Path,
    root: Path,
    write_grants: Sequence[Path],
) -> tuple[Path, Path, bool]:
    """Pick the deepest resolved write root containing ``target``.

    Returns the boundary, the target relative to it, and whether the target is
    an exact file grant (whose boundary is the grant's parent).
    """
    candidates: list[tuple[Path, Path, bool]] = []
    for boundary in (root, *write_grants):
        if target == boundary and boundary != root:
            candidates.append((boundary.parent, Path(boundary.name), True))
        elif target.is_relative_to(boundary):
            candidates.append((boundary, target.relative_to(boundary), False))
    if not candidates:
        invalid_path(str(target))
    boundary, relative, exact_grant = max(candidates, key=lambda item: len(item[0].parts))
    if not relative.parts or relative.name in {"", ".", ".."}:
        invalid_path(str(target))
    return boundary, relative, exact_grant


def main(invoke):
    payload = json.load(sys.stdin)
    try:
        result = invoke(payload)
    except FileToolError as exc:
        result = {"content": str(exc), "error": exc.category}
    if isinstance(result["content"], str):
        result["content"] = save_and_truncate_output(
            result["content"], invoke.__name__, workspace=payload["workspace"]
        )
    print(json.dumps(result, ensure_ascii=False))


def _output_directory(root: str | Path) -> Path:
    workspace = hashlib.sha256(os.fsencode(Path(root).resolve())).hexdigest()[:32]
    owner = (
        hashlib.sha256(os.fsencode(Path.home())).hexdigest()[:16]
        if os.name == "nt"
        else str(os.getuid())
    )
    return Path(tempfile.gettempdir()).resolve() / f"feather_tool_output_{owner}_{workspace}"
