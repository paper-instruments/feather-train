# Tool reference

Arguments below are the model-facing JSON fields. Defaults are shown where an
argument can be omitted. The local CLI includes file, search, Bash, todo, and
configured MCP tools; web tools are available to library callers.

## Files

| Tool | Arguments | Behavior |
| --- | --- | --- |
| `read` | `file_path`; `offset=1`; `limit=2000` | Read text with line numbers, starting at a one-based offset. Limit: 1–2000 lines. With images enabled, also returns images. |
| `write` | `file_path`; `content` | Create a text file or replace its full contents, creating parent directories as needed. |
| `edit` | `file_path`; `old_string`; `new_string`; `replace_all=false` | Replace matching text in an existing file. An empty `new_string` deletes it. |
| `grep` | `pattern`; `path="."`; optional `include`; `limit=100` | Search with a ripgrep-compatible regex; return paths and line numbers. `include` filters filenames in directory searches. |
| `glob` | `pattern`; `path="."`; `limit=100` | Find files by glob pattern and return sorted paths. |

Paths are relative to the workspace root. Absolute paths are accepted within the
workspace or explicitly configured grants. File tools check resolved symlink
targets too; read-only grants allow inspection but not writes.

The worker saves large responses in a private, per-workspace temporary directory.
File tools automatically receive a read-only grant to that directory, not other
workspaces' saved output or the rest of the temporary directory. Reusing a workspace
reuses its output directory. This is not isolation from Bash or other processes with
the same OS permissions; host isolation and retention remain caller concerns.

Search excludes hidden and ignored files and bounds its output. `grep` and `glob`
accept limits from 1 to 1000. Reading and editing are for text files; edits preserve
encoding and line endings outside the replacement. New files use UTF-8.

When images are enabled, `read` also returns PNG/JPEG/GIF/WebP images (and converts
BMP/TIFF), downscaled to at most 2000 px on the longest side and 5 MB of base64. Image
reads ignore text line ranges and do not establish an editable text revision. With
images disabled, `read` returns a disabled-vision response for images. See
[profiles](profiles.md) for the image capability override.

### Read before modifying

A successful `read` records the revision of the entire file, even when only a range
of lines is returned. `edit` and overwriting `write` require that revision to match
the current file. If another tool or process changes it, read it again before
modifying it. Creating a new file does not require a read.

Successful edits and writes update the remembered revision, so consecutive changes
do not require repeated reads. Search does not establish a revision, and Bash
modifications do not update the file tools' remembered revision.

`edit` requires a unique match unless `replace_all=true`. It first tries exact
matching, then permits leading/trailing whitespace around `old_string` to differ.
The replacement text is preserved as supplied. An ambiguous or missing match
returns feedback without changing the file.

Schemas: [file tools](../src/feather_harness/tools/files.py).

## Bash

| Argument | Meaning |
| --- | --- |
| `command` | Required Bash command. |
| `workdir` | Optional working directory; defaults to the workspace. Relative paths resolve from the workspace root. |
| `timeout` | Process deadline in milliseconds, from 1 to 600000. Defaults to 120000 for foreground commands; no deadline for background commands when omitted. |
| `run_in_background=false` | Return a managed task immediately instead of waiting for completion. |
| `description` | Optional background-task label, up to 160 characters. |

Every call starts a fresh shell. Shell variables, `cd`, and other shell state do
not carry between calls; filesystem changes do. Commands use the environment's
installed programs and OS permissions. File-tool grants do not restrict what a
shell command can access.

Foreground execution waits for the shell and its tracked descendants to finish,
or for the deadline. Timeout stops the owned command processes. This means a shell
that exits while leaving tracked children running does not immediately complete
the tool call. Output combines stdout and stderr and is bounded, including saved
command logs. There is no interactive stdin or PTY interface.

Use background mode for servers, watchers, and other long-lived work. A trailing
bare `&` is rejected with feedback to use `run_in_background=true` instead.

## Background tasks

| Tool | Arguments | Behavior |
| --- | --- | --- |
| `task_list` | `active_only=true`; `limit=20` | List this session's tasks. Set `active_only=false` to include recent completed tasks. Limit: 1–100. |
| `task_output` | `task_id`; `block=true`; `timeout=30000` | Read task state and output, optionally waiting for completion. Wait timeout: 0–600000 milliseconds. |
| `task_stop` | `task_id` | Stop the task and its tracked descendants. Already-finished tasks return their terminal state. |

`bash` returns the `task_id` used by these tools. The `task_output` timeout limits
only that inspection call: it does not extend or shorten the command's deadline.
Active listings also include failed tasks whose processes still need cleanup.
Closing the run stops its owned background commands.

Schemas: [Bash and task tools](../src/feather_harness/tools/process.py).

## Todo list

`todo_write` reads or replaces the session's list:

- `{}` returns the current list without changing it.
- `{"todos": [...]}` replaces the entire list, rather than merging items.
- `{"todos": []}` clears it.

Each item has `content` (1–160 characters) and `status` (`pending`, `in_progress`,
or `completed`). A list can contain up to 20 items.

Todo state survives compaction. The summary context includes the list as it stood
before the retained recent turns, so later todo updates remain in chronological
order. The live list still reflects the latest update.

Schema: [todo tool](../src/feather_harness/tools/todo.py).

## MCP

Configured MCP servers supply their own tool names, argument schemas, and semantics.
Feather exposes them as `mcp__server__tool`. The CLI accepts Streamable HTTP and SSE
servers, with optional authentication/routing headers:

```json
{
  "model": "your-model",
  "mcp_servers": [
    {"name": "mail", "transport": "http", "url": "http://localhost:9000/mcp/"}
  ]
}
```

Argument errors return corrective tool feedback. Server `isError` responses remain
tool content by default; `raise_on_error_result=true` propagates non-validation
server failures instead. Tool invocations are not automatically replayed. Text
results within 24,000 aggregate characters retain their exact content-block structure.

After a successful full-text save, larger results receive a head/tail preview with a
24,000-character total budget including the notice. The worker saves text in original
order, and the notice supplies its readable path. Use line ranges or search for ordinary
text; a long single-line payload may require slicing with a shell command. Images retain
their existing order and are not counted as text. Retrieval does not re-call the MCP tool.

`run_local` supplies its file transport. Custom `run_with_mcp` callers can pass
`file_transport=` using the same transport as their `FileTools`; in a remote sandbox,
this must execute on the agent filesystem, not the coordinator. Without a transport,
or when saving fails, return the original full MCP result: do not discard evidence that
the agent cannot retrieve. Native training callers without a transport therefore keep
their historical observations. Successful spilling intentionally changes oversized
observations for transport-equipped callers in both profiles, including local `train`.

Configuration: [MCP integration](../src/feather_harness/tools/mcp.py).

## Web

`web_search` takes a `query` and returns up to five sources. `web_fetch` takes a
`url` and returns bounded page text. Both restrict access to public URLs.

Library callers can supply a provider such as `ExaWebProvider`. The local CLI adds
both tools only when its config sets `enable_web_tools: true`, and then requires
`EXA_API_KEY` in the environment before the run starts.

Schemas: [search](../src/feather_harness/tools/web/search.py),
[fetch](../src/feather_harness/tools/web/fetch.py).
