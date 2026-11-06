import json
from pathlib import Path

import pytest

from feather_harness import FileTools, LocalFileRuntime


@pytest.mark.parametrize("encoding", ["cp1252", "ascii"])
async def test_file_tools_use_utf8_protocol_with_a_legacy_stdio_encoding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, encoding: str
) -> None:
    monkeypatch.setenv("PYTHONIOENCODING", encoding)
    monkeypatch.setenv("PYTHONUTF8", "0")
    path = tmp_path / "notes.txt"
    original = "Résumé — analyst’s €10"
    path.write_text(original, encoding="utf-8")
    files = FileTools(LocalFileRuntime(tmp_path))

    read = await files.read_tool().invoke(json.dumps({"file_path": path.name}))
    assert read.error is None
    assert original in read.content

    updated = "東京 — revised 🚀"
    edit = await files.edit_tool().invoke(
        json.dumps({"file_path": path.name, "old_string": original, "new_string": updated})
    )
    assert edit.error is None
    assert path.read_text(encoding="utf-8") == updated

    name = "Résumé – 東京.txt"
    write = await files.write_tool().invoke(json.dumps({"file_path": name, "content": updated}))
    assert write.error is None
    reread = await files.read_tool().invoke(json.dumps({"file_path": name}))
    assert reread.error is None
    assert updated in reread.content
