"""Verifier images must contain the chosen release and immutable source provenance."""

import json
import subprocess
import sys

import pytest

from feather_train.grading.dependencies import (
    GRADER_PACKAGE_VERSIONS,
    GRADER_SOURCE_URLS,
    grader_dependency_check_script,
)


@pytest.mark.parametrize("mismatch", [None, "version", "missing_source", "wrong_source", "invalid_json"])
def test_verifier_refuses_same_version_from_uncontrolled_source(tmp_path, mismatch):
    for name, version in GRADER_PACKAGE_VERSIONS.items():
        if name == "rubric" and mismatch == "version":
            version = "2.2.0"
        dist = tmp_path / f"{name}-{version}.dist-info"
        dist.mkdir()
        (dist / "METADATA").write_text(
            f"Metadata-Version: 2.3\nName: {name}\nVersion: {version}\n", encoding="utf-8"
        )
        if name not in GRADER_SOURCE_URLS or mismatch == "missing_source":
            continue
        source = {"url": GRADER_SOURCE_URLS[name], "archive_info": {}}
        if mismatch == "wrong_source":
            source["url"] = "https://uncontrolled.example.test/rubric.tar.gz"
        (dist / "direct_url.json").write_text(
            "[]" if mismatch == "invalid_json" else json.dumps(source), encoding="utf-8"
        )
    result = subprocess.run(
        [sys.executable, "-I", "-c",
         f"import sys; sys.path.insert(0, {str(tmp_path)!r})\n" + grader_dependency_check_script()],
        check=False, text=True, capture_output=True,
    )
    assert result.returncode == (78 if mismatch else 0)
    assert result.stdout == ""
    assert ("rubric" in result.stderr) == bool(mismatch)
