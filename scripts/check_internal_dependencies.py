"""Resolve built wheels against an inert hostile index, using only local Git objects.

Run with pip and hatchling installed. No candidate package is installed or executed;
only the controlled harness source's build backend prepares its dependency metadata.
"""

from __future__ import annotations

import argparse
import contextlib
import html
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import zipfile
from collections.abc import Iterator
from email.message import Message
from email.parser import BytesParser
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parents[1]
SOURCE_PREFIX = "git+https://github.com/paper-instruments/feather-train.git@"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel_directory", type=Path)
    args = parser.parse_args()
    wheels = {}
    for path in args.wheel_directory.glob("*.whl"):
        metadata = wheel_metadata(path)
        wheels[metadata["Name"]] = (path.resolve(), metadata)
    for package in ("feather-train", "feather-eval"):
        _path, metadata = wheels[package]
        internal = [
            Requirement(value) for value in metadata.get_all("Requires-Dist", [])
            if Requirement(value).name == "feather-harness"
        ]
        assert len(internal) == 1
        url = internal[0].url or ""
        assert re.fullmatch(
            re.escape(SOURCE_PREFIX) + r"[0-9a-f]{40}#subdirectory=feather-harness", url
        ), f"{package} must pin its harness source in standard wheel metadata"

    with tempfile.TemporaryDirectory(prefix="feather-dependency-check-") as raw:
        root = Path(raw)
        index = root / "index"
        wheelhouse = index / "wheels"
        wheelhouse.mkdir(parents=True)
        for name, version in (
            ("feather-harness", "0.1.0"), ("e2b", "2.42.0"), ("harbor", "0.22.0"),
            ("openai", "2.45.0"), ("pydantic", "2.13.4"),
            ("charset-normalizer", "3.4.7"), ("mcp", "1.27.1"),
            ("httpx", "0.28.1"), ("tenacity", "9.1.4"),
        ):
            wheel = inert_wheel(wheelhouse, name, version)
            simple = index / "simple" / name
            simple.mkdir(parents=True)
            (simple / "index.html").write_text(
                f'<a href="/wheels/{html.escape(wheel.name)}">{html.escape(wheel.name)}</a>',
                encoding="utf-8",
            )
        baseline = inert_wheel(root, "uncontrolled-consumer", "0.0.0", "feather-harness==0.1.0")
        with hostile_index(index) as (url, requests):
            environment = {
                **os.environ,
                "PIP_CONFIG_FILE": os.devnull,
                "PIP_INDEX_URL": url,
                "PIP_EXTRA_INDEX_URL": "",
                "PIP_DISABLE_PIP_VERSION_CHECK": "1",
                "PYTHONNOUSERSITE": "1",
                "GIT_CONFIG_NOSYSTEM": "1",
                "GIT_CONFIG_GLOBAL": os.devnull,
                "GIT_CONFIG_COUNT": "2",
                "GIT_CONFIG_KEY_0": f"url.{ROOT.as_uri()}.insteadOf",
                "GIT_CONFIG_VALUE_0": SOURCE_PREFIX.removeprefix("git+").removesuffix("@"),
                "GIT_CONFIG_KEY_1": "protocol.file.allow",
                "GIT_CONFIG_VALUE_1": "always",
            }
            environment.pop("PYTHONPATH", None)
            environment.pop("PYTHONHOME", None)
            baseline_report = resolve(baseline, root / "baseline.json", environment)
            assert any(
                item["metadata"]["name"] == "feather-harness"
                and "/wheels/feather_harness-" in item["download_info"]["url"]
                for item in baseline_report["install"]
            ), "The control must prove an unbound requirement can select the hostile wheel"
            for package in ("feather-train", "feather-eval"):
                requests.clear()
                wheel, metadata = wheels[package]
                report = resolve(wheel, root / f"{package}.json", environment)
                harness = next(
                    item for item in report["install"]
                    if item["metadata"]["name"] == "feather-harness"
                )
                expected = next(
                    Requirement(value).url for value in metadata.get_all("Requires-Dist", [])
                    if Requirement(value).name == "feather-harness"
                )
                vcs = harness["download_info"]["vcs_info"]
                assert vcs["commit_id"] == expected.split("@", 1)[1].split("#", 1)[0]
                assert not any("/simple/feather-harness" in path for path in requests)
                assert not any("/wheels/feather_harness-" in path for path in requests)
    print("Train and Eval resolve the pinned harness; hostile namespace control rejected")


def wheel_metadata(path: Path) -> Message:
    with zipfile.ZipFile(path) as archive:
        name = next(name for name in archive.namelist() if name.endswith(".dist-info/METADATA"))
        return BytesParser().parsebytes(archive.read(name))


def inert_wheel(directory: Path, name: str, version: str, dependency: str = "") -> Path:
    stem = f"{name.replace('-', '_')}-{version}"
    dist = f"{stem}.dist-info"
    path = directory / f"{stem}-py3-none-any.whl"
    metadata = f"Metadata-Version: 2.3\nName: {name}\nVersion: {version}\n"
    if dependency:
        metadata += f"Requires-Dist: {dependency}\n"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(f"{dist}/METADATA", metadata + "\n")
        archive.writestr(f"{dist}/WHEEL", "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n")
        archive.writestr(f"{dist}/RECORD", "")
    return path


@contextlib.contextmanager
def hostile_index(directory: Path) -> Iterator[tuple[str, list[str]]]:
    requests: list[str] = []

    class Handler(SimpleHTTPRequestHandler):
        def do_GET(self) -> None:
            requests.append(self.path)
            super().do_GET()

        def log_message(self, format: str, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(Handler, directory=str(directory)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/simple", requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def resolve(wheel: Path, report: Path, environment: dict[str, str]) -> dict[str, Any]:
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--dry-run", "--ignore-installed",
         "--no-build-isolation", "--no-cache-dir", "--report", str(report), str(wheel)],
        cwd=report.parent, env=environment, text=True, capture_output=True, timeout=90, check=False,
    )
    assert result.returncode == 0, result.stderr[-3000:]
    return json.loads(report.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
