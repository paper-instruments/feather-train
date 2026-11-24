"""Replace text only after the current revision has been read."""

import re
from array import array
from bisect import bisect_left
from collections.abc import Iterable, Iterator
from io import StringIO
from itertools import chain
from pathlib import Path

from _files import (
    FileToolError,
    check_revision,
    load_text,
    main,
    require_current_read,
    require_write_target,
    validate_write_extension,
    write_text,
)


def edit(payload):
    args = payload["arguments"]
    old_string = args["old_string"]
    new_string = args["new_string"]
    replace_all = args["replace_all"]
    target = require_write_target(
        payload["workspace"],
        payload["read_grants"],
        args["file_path"],
        write_grants=payload["write_grants"],
    )
    display = target.display
    validate_write_extension(Path(display), display)
    expected = require_current_read(args["revisions"], display)
    content, encoding, revision = load_text(
        target.path, display, target.metadata, scope="edits text files", lossless=True
    )
    check_revision(display, expected, revision)
    occurrences = _find_occurrences(content, old_string)
    first = next(occurrences, None)
    if first is None:
        stripped = old_string.strip()
        if stripped and stripped != old_string:
            occurrences = _find_occurrences(content, stripped)
            first = next(occurrences, None)
    if first is None:
        raise FileToolError(
            f"old_string was not found in {display}; no replacement was made. "
            "Read the relevant range and retry with current text."
        )

    first_line, first_start, first_end = first
    if replace_all:
        updated, replacement_count = _replace_all(content, new_string, chain((first,), occurrences))
    else:
        _require_unique_match(first_line, occurrences, display)
        updated = content[:first_start] + new_string + content[first_end:]
        replacement_count = 1
    new_revision = write_text(target, updated, encoding)

    start_line = max(1, first_line - 4)
    end_line = first_line + 4 + new_string.count("\n")
    snippet = _slice(updated, start_line, end_line).rstrip("\r\n")
    noun = "occurrence" if replacement_count == 1 else "occurrences"
    output = f"Replaced {replacement_count} {noun} in {display}.\n" + _numbered_output(
        snippet, f"a snippet of {display}", start_line
    )

    return {
        "content": output,
        "metadata": {"path": display, "revision": new_revision, "encoding": encoding},
    }


def _replace_all(
    content: str,
    replacement: str,
    matches: Iterable[tuple[int, int, int]],
) -> tuple[str, int]:
    # Stream matches into the output instead of allocating a list per occurrence.
    with StringIO() as buffer:
        cursor = 0
        for replacement_count, (_line, start, end) in enumerate(matches, 1):
            buffer.write(content[cursor:start])
            buffer.write(replacement)
            cursor = end
        buffer.write(content[cursor:])
        return buffer.getvalue(), replacement_count


def _require_unique_match(
    first_line: int,
    remaining_matches: Iterator[tuple[int, int, int]],
    display: str,
) -> None:
    match_count = 1
    line_numbers = [first_line]
    distinct_line_count = 1
    previous_line = first_line
    for line, _start, _end in remaining_matches:
        match_count += 1
        if line != previous_line:
            distinct_line_count += 1
            if len(line_numbers) < 20:
                line_numbers.append(line)
            previous_line = line
    if match_count == 1:
        return

    shown_lines = ", ".join(str(line) for line in line_numbers)
    if distinct_line_count > 20:
        shown_lines += f", ... ({distinct_line_count - 20} more)"
    raise FileToolError(
        f"old_string matched {match_count} locations in {display} on "
        f"lines [{shown_lines}]; no replacement was made. Read the relevant "
        "range and retry with more context, or set replace_all=true."
    )


def _numbered_output(content: str, description: str, start_line: int = 1) -> str:
    lines = _split_text_lines(content)
    if not lines:
        lines = [""]
    numbered = "\n".join(f"{number:6}\t{line}" for number, line in enumerate(lines, start_line))
    return f"Contents of {description}:\n{numbered}\n"


def _find_occurrences(content: str, target: str) -> Iterator[tuple[int, int, int]]:
    # Each match carries its line number and start/end offsets in the original text.
    if "\r" in target or "\n" in target:
        # Match LF (Unix), CRLF (Windows), and CR line endings interchangeably.
        # Apply replacements to the original text to preserve everything outside the match.
        normalized_content, removed_cr_positions = _normalize_newlines(content)
        normalized_target, _ = _normalize_newlines(target)
        line_number = 1
        line_cursor = 0
        for match in re.finditer(re.escape(normalized_target), normalized_content):
            normalized_start = match.start()
            normalized_end = match.end()
            line_number += normalized_content.count("\n", line_cursor, normalized_start)
            line_cursor = normalized_start
            # Count removed CRs before each boundary to map normalized offsets back
            # to the original text. A boundary at a newline stays before its CRLF pair.
            start = normalized_start + bisect_left(removed_cr_positions, normalized_start)
            end = normalized_end + bisect_left(removed_cr_positions, normalized_end)
            yield line_number, start, end
        return
    line_number = 1
    cursor = 0
    for match in re.finditer(re.escape(target), content):
        # Count only the new interval, treating CRLF as one newline.
        start = match.start()
        lf_count = content.count("\n", cursor, start)
        cr_count = content.count("\r", cursor, start)
        crlf_count = content.count("\r\n", cursor, start)
        line_number += lf_count + cr_count - crlf_count
        yield line_number, start, match.end()
        cursor = start


def _normalize_newlines(content: str) -> tuple[str, array]:
    if "\r" not in content:
        return content, array("I")
    # Store normalized offsets of removed CRs as compact unsigned integers.
    # For example, "a\r\nb" becomes "a\nb", with a removed CR at offset 1.
    removed_cr_positions = array("I")
    search_from = 0
    while (index := content.find("\r\n", search_from)) >= 0:
        removed_cr_positions.append(index - len(removed_cr_positions))
        search_from = index + 2
    normalized = content.replace("\r\n", "\n").replace("\r", "\n")
    return normalized, removed_cr_positions


def _slice(content: str, start_line: int, end_line: int) -> str:
    start = 0
    for line, match in enumerate(re.finditer(r"\r\n|\r|\n", content), 1):
        if line < start_line:
            start = match.end()
        if line == end_line:
            return content[start : match.end()]
    return content[start:]


def _split_text_lines(content: str) -> list[str]:
    # str.splitlines() also splits on Unicode separators; the tools count only CR/LF.
    lines = re.split(r"\r\n|\r|\n", content)
    if lines[-1] == "":
        lines.pop()
    return lines


if __name__ == "__main__":
    main(edit)
