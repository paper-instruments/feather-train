"""Bounded regular-file archiving, also executed inside the agent container."""

from __future__ import annotations

import json
import os
import stat
import sys
import tarfile
from pathlib import Path

MAX_FILE_BYTES = 64 * 1024 * 1024
MAX_TOTAL_BYTES = 256 * 1024 * 1024
MAX_FILES = 10000
MAX_ARCHIVE_BYTES = MAX_TOTAL_BYTES + 16 * 1024 * 1024
EXCLUDED_DIRECTORIES = {".agents", ".feather"}


def capture_files(roots: list[str], destination: str) -> dict:
    """Archive only regular files; directory descriptors fence symlink swaps."""
    count = total = skipped = 0
    with tarfile.open(destination, "w:gz", dereference=False) as archive:
        for root in roots:
            root_fd = _open_directory(root)
            try:
                for directory, directories, files, dir_fd in os.fwalk(
                    ".", dir_fd=root_fd, follow_symlinks=False
                ):
                    directories[:] = sorted(
                        name for name in directories if name not in EXCLUDED_DIRECTORIES
                    )
                    for name in directories:
                        if stat.S_ISLNK(os.stat(name, dir_fd=dir_fd, follow_symlinks=False).st_mode):
                            skipped += 1
                    for name in sorted(files):
                        entry = os.stat(name, dir_fd=dir_fd, follow_symlinks=False)
                        if not stat.S_ISREG(entry.st_mode):
                            skipped += 1
                            continue
                        fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=dir_fd)
                        with os.fdopen(fd, "rb") as source:
                            entry = os.fstat(source.fileno())
                            if not stat.S_ISREG(entry.st_mode):
                                raise ValueError("File changed type during capture")
                            count += 1
                            total += entry.st_size
                            if (
                                entry.st_size > MAX_FILE_BYTES
                                or total > MAX_TOTAL_BYTES
                                or count > MAX_FILES
                            ):
                                raise ValueError("Deliverable capture limit exceeded")
                            relative = Path(directory) / name
                            info = tarfile.TarInfo(str(Path(root.lstrip("/")) / relative))
                            info.size = entry.st_size
                            info.mode = stat.S_IMODE(entry.st_mode)
                            info.mtime = entry.st_mtime
                            archive.addfile(info, source)
            finally:
                os.close(root_fd)
    if os.path.getsize(destination) > MAX_ARCHIVE_BYTES:
        raise ValueError("Deliverable archive limit exceeded")
    return {"files": count, "file_bytes": total, "skipped_special_files": skipped}


def _open_directory(path: str) -> int:
    fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        for component in Path(path).parts[1:]:
            child = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = child
        return fd
    except BaseException:
        os.close(fd)
        raise


if __name__ == "__main__":
    print(json.dumps(capture_files(json.loads(sys.argv[1]), sys.argv[2])))
