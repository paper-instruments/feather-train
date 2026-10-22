from __future__ import annotations

import json

import pytest

from feather_harness import FileTools, LocalFileRuntime


async def invoke(files, name, path, **arguments):
    tool = getattr(files, f"{name}_tool")()
    return await tool.invoke(json.dumps({"file_path": str(path), **arguments}))


@pytest.mark.parametrize(
    "name, arguments",
    [
        ("edit", {"old_string": "before", "new_string": "after"}),
        ("write", {"content": "after"}),
    ],
)
async def test_existing_files_require_a_current_read(tmp_path, name, arguments):
    path = tmp_path / "notes.txt"
    path.write_text("before")
    files = FileTools(LocalFileRuntime(tmp_path))
    result = await invoke(files, name, "notes.txt", **arguments)
    assert "Read notes.txt before modifying it" in result.content
    assert path.read_text() == "before"

    await invoke(files, "read", "notes.txt")
    path.write_text("someone else's edit")
    result = await invoke(files, name, "notes.txt", **arguments)
    assert "changed since it was read" in result.content
    assert path.read_text() == "someone else's edit"


async def test_write_creates_parents_and_tracks_its_new_revision(tmp_path):
    files = FileTools(LocalFileRuntime(tmp_path))
    first = await invoke(files, "write", "dir/new.txt", content="first")
    second = await invoke(files, "write", "dir/new.txt", content="second")
    assert first.content == "Created dir/new.txt."
    assert second.content == "Overwrote dir/new.txt."
    assert (tmp_path / "dir/new.txt").read_bytes() == b"second"


async def test_edit_rejects_ambiguity_and_preserves_replacement_whitespace(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("before\nbefore\n")
    files = FileTools(LocalFileRuntime(tmp_path))
    await invoke(files, "read", "notes.txt", limit=1)
    ambiguous = await invoke(files, "edit", "notes.txt", old_string="before", new_string="after")
    assert "matched 2 locations" in ambiguous.content
    assert path.read_text() == "before\nbefore\n"
    result = await invoke(
        files, "edit", "notes.txt", old_string=" before ", new_string=" after ", replace_all=True
    )
    assert result.content.startswith("Replaced 2 occurrences in notes.txt.")
    assert path.read_bytes() == b" after \n after \n"
    again = await invoke(
        files, "edit", "notes.txt", old_string="after", new_string="done", replace_all=True
    )
    assert again.error is None
    assert path.read_bytes() == b" done \n done \n"


async def test_multiline_edit_matches_crlf_without_normalizing_unedited_text(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_bytes(b"start\r\none\r\ntwo\r\nend\r\n")
    files = FileTools(LocalFileRuntime(tmp_path))
    await invoke(files, "read", "notes.txt")
    result = await invoke(files, "edit", "notes.txt", old_string="one\ntwo", new_string="new\ntext")
    assert result.error is None
    assert path.read_bytes() == b"start\r\nnew\ntext\r\nend\r\n"


async def test_edit_preserves_detected_utf16_encoding(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_bytes("résumé".encode("utf-16"))
    files = FileTools(LocalFileRuntime(tmp_path))
    await invoke(files, "read", "notes.txt")
    await invoke(files, "edit", "notes.txt", old_string="résumé", new_string="café")
    assert path.read_bytes() == "café".encode("utf-16")


async def test_edit_refuses_text_that_only_decodes_with_replacement(tmp_path):
    path = tmp_path / "notes.txt"
    # Text first, so the binary sniff passes; no encoding decodes the trailing bytes.
    original = b"setting = 1\n" * 100 + bytes(range(128, 256)) * 4 + b"\nend\n"
    path.write_bytes(original)
    files = FileTools(LocalFileRuntime(tmp_path))
    await invoke(files, "read", "notes.txt", limit=1)

    result = await invoke(files, "edit", "notes.txt", old_string="end", new_string="done")

    assert result.error == "tool_error"
    assert "cannot be decoded as text" in result.content
    assert path.read_bytes() == original


@pytest.mark.parametrize("encoding", ["utf-8", "utf-8-sig"], ids=["plain-utf8", "utf8-with-bom"])
async def test_short_utf8_round_trip_preserves_text_and_encoding(tmp_path, encoding):
    path = tmp_path / "notes.txt"
    original = "alpha\nβeta\nrepeat\nrepeat\n"
    path.write_bytes(original.encode(encoding))
    files = FileTools(LocalFileRuntime(tmp_path))

    result = await invoke(files, "read", "notes.txt", offset=2, limit=2)
    assert "     2\tβeta\n     3\trepeat\n" in result.content
    result = await invoke(files, "edit", "notes.txt", old_string="βeta", new_string="café")
    assert result.error is None
    assert path.read_bytes() == original.replace("βeta", "café").encode(encoding)


@pytest.mark.parametrize("alias", [False, True])
async def test_readonly_grant_cannot_be_written_directly_or_through_workspace_alias(
    tmp_path, alias
):
    root = tmp_path / "workspace"
    root.mkdir()
    target = tmp_path / "outside.txt"
    target.write_text("before")
    (root / "alias").symlink_to(target)
    files = FileTools(LocalFileRuntime(root, read_grants=(target,)))
    path = "alias" if alias else target
    await invoke(files, "read", path)
    result = await invoke(files, "write", path, content="after")
    assert result.error == "invalid_path"
    assert target.read_text() == "before"


async def test_most_specific_grant_controls_external_writes(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    external = tmp_path / "external"
    external.mkdir()
    protected = external / "protected.txt"
    protected.write_text("protected")
    files = FileTools(LocalFileRuntime(root, read_grants=(protected,), write_grants=(external,)))
    await invoke(files, "read", protected)
    assert (await invoke(files, "write", protected, content="changed")).error == "invalid_path"
    created = await invoke(files, "write", external / "new.txt", content="created")
    assert created.error is None
    assert (external / "new.txt").read_text() == "created"
    assert protected.read_text() == "protected"


async def test_writes_and_edits_follow_symlinks_the_path_check_allows(tmp_path):
    root = tmp_path / "workspace"
    (root / "real").mkdir(parents=True)
    (root / "real" / "f.txt").write_text("old")
    (root / "link").symlink_to(root / "real", target_is_directory=True)
    (root / "flink").symlink_to(root / "real" / "f.txt")
    files = FileTools(LocalFileRuntime(root))

    assert (await invoke(files, "write", "link/new.txt", content="hi")).error is None
    await invoke(files, "read", "flink")
    assert (await invoke(files, "write", "flink", content="new")).error is None
    edited = await invoke(files, "edit", "link/new.txt", old_string="hi", new_string="bye")
    assert edited.error is None

    assert (root / "real" / "new.txt").read_text() == "bye"
    assert (root / "real" / "f.txt").read_text() == "new"
    assert (root / "flink").is_symlink()
    # Revisions stay keyed by the path the model used.
    assert set(files.revisions) == {"link/new.txt", "flink"}


async def test_symlink_into_an_external_write_grant_is_writable(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    external = tmp_path / "external"
    external.mkdir()
    (root / "out").symlink_to(external, target_is_directory=True)
    files = FileTools(LocalFileRuntime(root, write_grants=(external,)))

    assert (await invoke(files, "write", "out/report.md", content="done")).error is None
    assert (external / "report.md").read_text() == "done"


async def test_symlink_to_an_ungranted_location_is_refused(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "out").symlink_to(outside, target_is_directory=True)
    files = FileTools(LocalFileRuntime(root))

    result = await invoke(files, "write", "out/report.md", content="done")

    assert result.error == "invalid_path"
    assert list(outside.iterdir()) == []


async def test_exact_file_grant_allows_the_file_but_not_new_parents(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    external = tmp_path / "external"
    external.mkdir()
    granted = external / "notes.txt"
    granted.write_text("before")
    missing = external / "missing" / "new.txt"
    files = FileTools(LocalFileRuntime(root, write_grants=(granted, missing)))

    await invoke(files, "read", granted)
    assert (await invoke(files, "write", granted, content="after")).error is None
    assert granted.read_text() == "after"
    assert (await invoke(files, "write", missing, content="x")).error is not None
    assert not missing.parent.exists()


async def test_unrepresentable_replacement_does_not_change_file_or_revision(tmp_path):
    path = tmp_path / "notes.txt"
    original = b"Caf\xe9 " * 40
    path.write_bytes(original)
    files = FileTools(LocalFileRuntime(tmp_path))
    await invoke(files, "read", "notes.txt")
    revision = files.revisions["notes.txt"]
    result = await invoke(files, "write", "notes.txt", content="😀")
    assert "cannot be represented" in result.content
    assert path.read_bytes() == original
    assert files.revisions["notes.txt"] == revision


async def test_repeated_edits_count_mixed_newlines_and_same_line_matches(tmp_path):
    path = tmp_path / "notes.txt"
    original = b"header\r\nneedle needle\rskip\nneedle\r\n" * 1000
    path.write_bytes(original)
    files = FileTools(LocalFileRuntime(tmp_path))
    await invoke(files, "read", "notes.txt", limit=1)

    ambiguous = await invoke(files, "edit", "notes.txt", old_string="needle", new_string="done")
    expected_lines = ", ".join(str(line) for line in range(2, 41, 2))
    assert ambiguous.content == (
        "old_string matched 3000 locations in notes.txt on "
        f"lines [{expected_lines}, ... (1980 more)]; no replacement was made. "
        "Read the relevant range and retry with more context, or set replace_all=true."
    )
    assert path.read_bytes() == original

    result = await invoke(
        files, "edit", "notes.txt", old_string="needle", new_string="done", replace_all=True
    )
    assert result.content.startswith("Replaced 3000 occurrences")
    assert path.read_bytes() == original.replace(b"needle", b"done")


async def test_dense_same_line_matches_report_exact_ambiguity_without_changes(tmp_path):
    path = tmp_path / "notes.txt"
    original = b"a" * 200_000
    files = FileTools(LocalFileRuntime(tmp_path))
    await invoke(files, "write", "notes.txt", content=original.decode())
    revision = files.revisions["notes.txt"]

    result = await invoke(files, "edit", "notes.txt", old_string="a", new_string="b")

    assert result.content == (
        "old_string matched 200000 locations in notes.txt on lines [1]; "
        "no replacement was made. Read the relevant range and retry with more context, "
        "or set replace_all=true."
    )
    assert path.read_bytes() == original
    assert files.revisions["notes.txt"] == revision


async def test_dense_replace_all_preserves_literal_replacement_and_nonoverlapping_matches(tmp_path):
    path = tmp_path / "notes.txt"
    files = FileTools(LocalFileRuntime(tmp_path))
    await invoke(files, "write", "notes.txt", content="a" * 200_001)
    replacement = "\\1é\n"

    result = await invoke(
        files,
        "edit",
        "notes.txt",
        old_string="aa",
        new_string=replacement,
        replace_all=True,
    )

    assert result.content.startswith("Replaced 100000 occurrences in notes.txt.")
    assert path.read_text() == replacement * 100_000 + "a"


async def test_multiline_replace_all_preserves_unmatched_mixed_newlines(tmp_path):
    path = tmp_path / "notes.txt"
    original = b"head\r\none\r\ntwo\r\nmiddle\none\ntwo\rtail\rone\rtwo\r\n"
    path.write_bytes(original)
    files = FileTools(LocalFileRuntime(tmp_path))
    await invoke(files, "read", "notes.txt")

    result = await invoke(
        files,
        "edit",
        "notes.txt",
        old_string="one\ntwo",
        new_string="new\ntext",
        replace_all=True,
    )

    assert result.content.startswith("Replaced 3 occurrences in notes.txt.")
    assert path.read_bytes() == b"head\r\nnew\ntext\r\nmiddle\nnew\ntext\rtail\rnew\ntext\r\n"


@pytest.mark.parametrize("old_string", ["absent", " absent ", " \t "])
async def test_unmatched_edit_keeps_file_and_revision(tmp_path, old_string):
    path = tmp_path / "notes.txt"
    path.write_text("unchanged")
    files = FileTools(LocalFileRuntime(tmp_path))
    await invoke(files, "read", "notes.txt")
    revision = files.revisions["notes.txt"]

    result = await invoke(
        files, "edit", "notes.txt", old_string=old_string, new_string="replacement"
    )

    assert result.content == (
        "old_string was not found in notes.txt; no replacement was made. "
        "Read the relevant range and retry with current text."
    )
    assert path.read_text() == "unchanged"
    assert files.revisions["notes.txt"] == revision


@pytest.mark.parametrize("newline", [b"\n", b"\r\n", b"\r"])
async def test_edit_preview_shows_only_surrounding_lines(tmp_path, newline):
    path = tmp_path / "notes.txt"
    original = newline.join(f"line {i}".encode() for i in range(1, 101))
    path.write_bytes(original)
    files = FileTools(LocalFileRuntime(tmp_path))
    await invoke(files, "read", "notes.txt")
    result = await invoke(
        files, "edit", "notes.txt", old_string="line 8", new_string="changed", replace_all=True
    )
    assert "     4\tline 4" in result.content
    assert "     8\tchanged" in result.content
    assert "    12\tline 12" in result.content
    assert "line 13" not in result.content
    assert path.read_bytes() == original.replace(b"line 8", b"changed")


async def test_revision_lookup_accepts_equivalent_path_spellings(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("before")
    files = FileTools(LocalFileRuntime(tmp_path))
    await invoke(files, "read", "./notes.txt")
    result = await invoke(files, "edit", path, old_string="before", new_string="after")
    assert result.error is None
    result = await invoke(files, "write", ".\\notes.txt", content="done")
    assert result.error is None
    assert path.read_text() == "done"
