"""Text-file tool bindings and the revisions observed by this agent."""

from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import replace
from typing import Protocol

from pydantic import Field, JsonValue, ValidationError

from ..core.messages import FrozenModel
from ..scripts._files import MAX_OUTPUT_CHARS, truncate_output
from . import Tool, ToolResult, ToolSpec, invalid_arguments_result

logger = logging.getLogger(__name__)

READ_DESCRIPTION = (
    "Read a text file with line numbers. `offset` and `limit` bound the returned range. A "
    "successful read records the file revision required to edit or overwrite its current contents."
)
READ_IMAGE_DESCRIPTION = (
    "Read a text file or an image. Text is returned with line numbers, `offset` and `limit` bound "
    "the returned range, and a successful text read records the file revision required to edit or "
    "overwrite its current contents. PNG, JPEG, GIF, and WebP images are returned for visual "
    "inspection, downscaled to at most 2000 pixels on the longest side. To inspect a PDF or Office "
    "file visually, render the needed pages to PNG at a legible resolution and read only those, "
    "since each image adds substantial context."
)
READ_IMAGE_PATH_HELP = (
    "Path to a text file or image inside the workspace. Relative paths resolve from the workspace "
    "root; absolute in-workspace or explicitly granted paths are also accepted."
)


EDIT_DESCRIPTION = (
    "Make a text replacement in an existing file. Requires a prior read of the file's current "
    "contents; the call is rejected if the file changed since that read. `old_string` must match "
    "the current content. If no exact match exists, only leading or trailing whitespace around "
    "old_string is ignored; new_string is always preserved exactly. `replace_all` permits every "
    "matched occurrence to change instead of requiring one match."
)

WRITE_DESCRIPTION = (
    "Write the complete contents of a text file, creating parent directories when needed. This "
    "creates a new file or overwrites an existing one. Overwriting requires a prior read of the "
    "current contents and is rejected if the file changed since that read. The tool accepts text "
    "files only."
)


class ReadArgs(FrozenModel):
    file_path: str = Field(
        min_length=1,
        description=(
            "Path to a text file inside the workspace. Relative paths resolve "
            "from the workspace root; absolute in-workspace or explicitly granted paths "
            "are also accepted."
        ),
    )
    offset: int = Field(
        default=1,
        ge=1,
        description="One-based line number at which to begin reading.",
    )
    limit: int = Field(
        default=2_000,
        ge=1,
        le=2_000,
        description="Maximum number of lines to return.",
    )


class EditArgs(FrozenModel):
    file_path: str = Field(
        min_length=1,
        description=(
            "Path to an existing text file inside the workspace. Relative paths resolve "
            "from the workspace root; absolute in-workspace or explicitly granted paths "
            "are also accepted."
        ),
    )
    old_string: str = Field(
        min_length=1,
        description=(
            "Text to replace, copied from the file with its indentation and enough surrounding "
            "context to identify the intended occurrence."
        ),
    )
    new_string: str = Field(
        description="Exact replacement text. Use an empty string to delete old_string."
    )
    replace_all: bool = Field(
        default=False,
        description="Replace every matched occurrence instead of requiring a unique match.",
    )


class WriteArgs(FrozenModel):
    file_path: str = Field(
        min_length=1,
        description=(
            "Path to the text file inside the workspace. Relative paths resolve from the "
            "workspace root; absolute in-workspace or explicitly granted paths are also accepted."
        ),
    )
    content: str = Field(description="Complete content to write to the file.")


GREP_DESCRIPTION = (
    "Search text files with a ripgrep-compatible regular expression. Results are bounded and "
    "include paths and line numbers. `include` optionally restricts filenames with a glob pattern. "
    "Hidden and ignored files are excluded."
)

GLOB_DESCRIPTION = (
    "Find files by glob pattern beneath a directory. Results are bounded, sorted paths; hidden "
    "and ignored files are excluded."
)


class GrepArgs(FrozenModel):
    pattern: str = Field(
        min_length=1,
        description="Ripgrep-compatible regular expression to search for.",
    )
    path: str = Field(
        default=".",
        min_length=1,
        description=(
            "File or directory to search inside the workspace. Relative paths resolve from the "
            "workspace root; absolute in-workspace or explicitly granted paths are also accepted. "
            "When this names a file, only that file is searched."
        ),
    )
    include: str | None = Field(
        default=None,
        min_length=1,
        description=(
            "Optional glob pattern restricting files in a directory search, for example "
            "'*.py' or '*.{ts,tsx}'."
        ),
    )
    limit: int = Field(
        default=100,
        ge=1,
        le=1_000,
        description="Maximum number of matching lines to return.",
    )


class GlobArgs(FrozenModel):
    pattern: str = Field(
        min_length=1,
        description=(
            "Glob pattern to match. Supports *, **, ?, character ranges, and brace "
            "alternatives; for example '**/*.xlsx' or '*.{ts,tsx}'."
        ),
    )
    path: str = Field(
        default=".",
        min_length=1,
        description=(
            "Directory to search inside the workspace. Relative paths resolve from the "
            "workspace root; absolute in-workspace or explicitly granted paths are also accepted."
        ),
    )
    limit: int = Field(
        default=100,
        ge=1,
        le=1_000,
        description="Maximum number of matching paths to return.",
    )


class FileToolTransport(Protocol):
    async def run(self, name: str, arguments: dict[str, JsonValue]) -> ToolResult: ...


class FileTools:
    def __init__(
        self, transport: FileToolTransport, *, enable_image_reads: bool = False
    ) -> None:
        self.transport = transport
        self.enable_image_reads = enable_image_reads
        self.revisions: dict[str, str] = {}

    def read_tool(self) -> Tool:
        if not self.enable_image_reads:
            return self._tool("read", READ_DESCRIPTION, ReadArgs)
        tool = self._tool("read", READ_IMAGE_DESCRIPTION, ReadArgs)
        parameters = deepcopy(tool.spec.parameters)
        parameters["properties"]["file_path"]["description"] = READ_IMAGE_PATH_HELP
        return replace(tool, spec=tool.spec.model_copy(update={"parameters": parameters}))

    def edit_tool(self) -> Tool:
        return self._tool("edit", EDIT_DESCRIPTION, EditArgs)

    def write_tool(self) -> Tool:
        return self._tool("write", WRITE_DESCRIPTION, WriteArgs)

    def grep_tool(self) -> Tool:
        return self._tool("grep", GREP_DESCRIPTION, GrepArgs)

    def glob_tool(self) -> Tool:
        return self._tool("glob", GLOB_DESCRIPTION, GlobArgs)

    def _tool(self, name: str, description: str, args_model: type[FrozenModel]) -> Tool:
        async def invoke(raw_arguments: str) -> ToolResult:
            try:
                args = args_model.model_validate_json(raw_arguments)
            except ValidationError as exc:
                return invalid_arguments_result(exc)
            if isinstance(args, EditArgs) and args.old_string == args.new_string:
                return ToolResult(
                    content=f"Invalid `new_string` parameter: {args.new_string}. old_string and new_string must be different.",
                    error="tool_error",
                )
            arguments = args.model_dump(mode="json")
            if name == "read" and self.enable_image_reads:
                arguments["image_reads"] = True
            if name in {"edit", "write"}:
                arguments["revisions"] = self.revisions
            result = await self.transport.run(name, arguments)
            if "revision" in result.metadata:
                self.revisions[result.metadata["path"]] = result.metadata["revision"]
            return result

        async def invoke_bounded(raw_arguments: str) -> ToolResult:
            result = await invoke(raw_arguments)
            # File operations already save/truncate their output in the worker. This host-side
            # check covers errors invoke() creates before dispatch, such as argument validation.
            if not isinstance(result.content, str) or len(result.content) <= MAX_OUTPUT_CHARS:
                return result
            try:
                bounded = await self.transport.run(
                    "_save_and_truncate_output", {"content": result.content, "tool_name": name}
                )
                content = bounded.content
            except Exception:
                # Saving is best effort; cancellation still propagates.
                logger.warning("Could not save full %s output", name, exc_info=True)
                content = truncate_output(result.content)
            return result.model_copy(update={"content": content})

        return Tool(
            spec=ToolSpec(
                name=name, description=description, parameters=args_model.model_json_schema()
            ),
            invoke=invoke_bounded,
        )
