# Usage reference

Details for installing, configuring, and embedding Feather. The
[README](../README.md) covers the architecture and a quick start.

## Installer

The repository's [installer](../../install.sh) supports macOS and Linux. It requires
Bash, curl, and Git, and downloads its own uv and managed Python 3.12. Install with:

```bash
curl -fsSL https://raw.githubusercontent.com/paper-instruments/feather-train/main/install.sh | bash
```

To install from a local checkout, run from the repository root:

```bash
FEATHER_SOURCE_DIR="$PWD" bash install.sh
```

The installer uses the repository lockfile to install Harness with the `office`
extra into `~/.local/share/feather/versions/<installation>/venv`. This includes
`paper-docx`, `paper-pptx`, `paper-xlsx`, the PDF libraries, and the data libraries
listed under [recommended packages](../README.md#recommended-packages). Python packages stay
in this Feather environment, regardless of the directory or active virtualenv
from which you install. It does not use system pip, install Train or Eval, require
sudo, or modify shell profiles.

The launcher at `~/.local/bin/feather` sets `VIRTUAL_ENV` and puts Feather's virtualenv
first on `PATH`. It clears inherited `PYTHONHOME` and `PYTHONPATH` and disables
user-site packages. By default, both the CLI and the model's `python` and `python3`
shell commands use Feather's packages without activating an environment.
If `~/.local/bin` is not on `PATH`, use the full launcher path below or add that
directory yourself.

```bash
mkdir -p workspace
export OPENAI_API_KEY=YOUR_API_KEY

~/.local/bin/feather \
  --model YOUR_MODEL --base-url http://localhost:8000/v1 \
  --workdir "$PWD/workspace" "Summarize the documents in this workspace into a briefing.md file." > run.json
```

Use the model identifier and endpoint supplied by your provider. Omit `--base-url`
for the OpenAI client's default endpoint. For an unauthenticated local server,
use a placeholder API key. The workspace must already exist; running the CLI
installs skills and writes logs there. Installation itself does not write into
your working directory.

Rerun the installer to update. It builds a fresh version and switches the `current`
symlink only after installation succeeds, retaining the previous version. Configure
the installer with environment variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `FEATHER_INSTALL_DIR` | `~/.local/share/feather` | Installation root; overrides must be absolute paths. |
| `FEATHER_BIN_DIR` | `~/.local/bin` | Launcher directory; overrides must be absolute paths. |
| `FEATHER_REF` | `main` | Repository branch, tag, or commit to install. |
| `FEATHER_REPO` | `https://github.com/paper-instruments/feather-train.git` | Git URL used for remote installations. |
| `FEATHER_SOURCE_DIR` | Unset | Absolute path to a local checkout to install instead. |

For example, install a pinned version from a local copy of the installer with
`FEATHER_REF=<full-commit-sha> bash install.sh`. To uninstall the default installation:

```bash
rm -f "$HOME/.local/bin/feather"
rm -rf "$HOME/.local/share/feather"
```

If you customized the directories, remove the launcher and installation root at
those paths instead. Workspace files, skills, and run logs remain in their original
locations.

## Source development


```bash
uv sync --package feather-harness --extra office --locked
uv run --package feather-harness --extra office --locked feather \
  --model YOUR_MODEL --workdir ./workspace "Summarize this workspace." > run.json
```

This source workflow uses the repository's `.venv`. Use `uv run` or activate it so
model shell commands see its packages. Harness is not currently published on PyPI;
library consumers should use the [pinned Git source configuration](../../README.md#use-the-packages-from-another-repository).

## Task Python

Use `--python` to select an existing interpreter for the model's task commands:

```bash
~/.local/bin/feather --model YOUR_MODEL \
  --python /absolute/path/to/project/.venv/bin/python \
  --workdir /absolute/path/to/project "Analyze this project's data." > run.json
```

The optional JSON configuration field `python` accepts the same path. Relative
paths resolve from the directory where you launch Feather, not `--workdir`.
Without an override, the installed CLI uses Feather's environment for task Python.

With an override, bare `python` and `python3` both invoke exactly the selected
interpreter, and its directory is added to the task's `PATH` for project console
tools. Feather's CLI and internal workers continue using its own runtime. The
override does not install packages into your selected environment or combine its
site-packages with Feather's.

The task shell exposes `FEATHER_PYTHON`, which points to Feather's interpreter.
For example, model shell commands can use each environment explicitly:

```bash
python analyze_project.py
"$FEATHER_PYTHON" create_deck.py
```

The first command uses your selected project's dependencies; the second uses the
bundled Paper packages. A script that needs dependencies from both environments
must have them installed together in one environment or split the work into
separate commands. A short Bash tool description explains the interpreter choice
to the model; selecting task Python does not change the system prompt or skills.
## Configuration

JSON config supports model settings, context compaction, and MCP servers. For example:

```json
{
  "model": "YOUR_MODEL",
  "base_url": "http://localhost:8000/v1",
  "workspace": "./workspace",
  "profile": "prod",
  "max_turns": 100,
  "model_settings": {"max_tokens": 8192},
  "compaction": {"enabled": true, "context_window_tokens": 131072}
}
```

```bash
cat task.txt | ~/.local/bin/feather --config config.json - > run.json
```

CLI flags override matching config fields. The workspace must already exist. Choose
an actual model-supported context window; compaction is disabled unless enabled.
See [LocalConfig](../src/feather_harness/cli.py)
and [tool configuration](tools.md) for the full interfaces.

The default `prod` profile adds professional knowledge-work instructions, professional
guidance inside the bundled format skills, a shared design guide, and ordinary image
tools. It requires an image-capable endpoint unless `enable_image_tools` is explicitly
`false`. Use `--profile train` (or `"profile": "train"` in JSON) for the minimal
training prompt, original manuals and descriptions, and disabled images. Profiles are
fixed at startup. See [profiles](profiles.md) for compatibility details.

Both CLI profiles stream model responses and retry transient inference failures with
the Fireworks training sampler's policy: up to seven attempts per request with 2-30
second jittered exponential backoff, honoring `Retry-After` up to the cap. Connection
errors, timeouts, streams that end without a finish reason, and HTTP 408, 429, 500,
502, 503 and 504 are retried; other client errors are not. Streaming makes the client's
request timeout an idle timeout, so a long generation that keeps producing tokens is not
cut off and resent unchanged. `model_retry` in the JSON config overrides the policy.
This does not replay a tool or restart a trajectory. Library callers retain ownership
of their supplied client's retry policy and stay nonstreaming unless they opt in. Editable workspace dependencies use the checked-out
source; unchanged package versions do not isolate new builds from this change.

Each compaction allows three retries after its initial summary request (four attempts
total) when the model returns an empty summary. With production response validation
enabled, summaries interrupted by `length` or `error` are also retried; refusals and
content filtering remain fatal. Each attempt receives the same input and settings,
and its response is retained in a separate compaction trace segment. Completed tools
are not replayed, only successful compactions are counted, and exhaustion preserves
the original execution evidence in `AgentRunError`. Transport failures keep the model
or supplied client's existing retry policy; compaction does not multiply that budget.

`additional_instructions` appends generic guidance to the system prompt, expanding
`{{ workspace }}` to the resolved workspace path while preserving task text. Set
`eval_prompt_output` to save the selected profile and SHA-256 of the actual system
prompt before model execution. `enable_web_tools: true` adds Exa-backed `web_search`
and `web_fetch`, using `EXA_API_KEY`; library callers pass a `web_provider` to `run_local`.
An optional `prompt_token_budget` configures local GLM tokenizer accounting; install
`feather-harness[token-count]` for the pinned tokenizer dependencies.

Feather installs bundled skills under `.agents` and saves shell logs under
`.feather/tasks` in the workspace. `skills_dir` copies task-provided skills after
bundled installation. Both `.agents/` and `.feather/` are added to the checkout's local
Git exclude file to keep harness state out of patch submissions. MCP connections support HTTP/SSE and stdio transports.

Parallel subagents are opt-in for `prod`. Add `"subagents": {"max_concurrent": 3,
"max_agents": 8, "max_turns": 50}` to JSON config, or pass a `SubagentConfig` to
`run_local`. Children share the workspace but have separate context, tools, and logs.
The parent can start children, wait for results, send follow-ups, and stop them.
Existing configurations remain unchanged. See [subagents](subagents.md) for
lifecycle limits and application integration.

## Execution traces

The CLI writes `RunResult` JSON to stdout. It records execution and compaction
segments with their starting contexts and new messages, so you can inspect how the
agent reached its result even across compactions. `execution_messages()` provides
a chronological transcript without duplicated context or summary calls.

The CLI currently runs one task at a time; interactive chat, resume, and image attachments
on the initial command are not supported. Enabled image tools can inspect files and
MCP image results during the run. See the [tool reference](tools.md).

## Library use

The caller owns the model client and supplies it to the loop:

```python
import asyncio
import os

from openai import AsyncOpenAI
from feather_harness import ChatCompletionsModel, FeatherAgent, RunRequest, UserMessage

async def main():
    async with AsyncOpenAI() as client:
        agent = FeatherAgent(model=ChatCompletionsModel(client=client, model=os.environ["MODEL"]))
        result = await agent.run(RunRequest(messages=(UserMessage(content="What is 2 + 2?"),)))
        print(result.model_dump_json())

asyncio.run(main())
```

This minimal example has no tools. `run_local` assembles the CLI's local tools and
selected profile; custom callers can bind `Tool` objects themselves. See
the [tool reference](tools.md) for file revisions, Bash tasks, and todo semantics.

`TraceSegment.calls` records provider usage for execution and compaction calls.
`RunResult.usage()` totals those records, including cached and reasoning tokens and
provider-reported costs. Missing provider fields remain unknown rather than being
reported as zero. Native signed/encrypted reasoning is replayed within an execution
window and removed when compaction replaces that window's context.

Custom callers can pass `end_run_after_tools(message, results)` to `FeatherAgent`.
This synchronous hook receives the assistant message and a tuple of `ToolResult`
objects in tool-call order after all of that message's tools have run. Returning
`True` completes the run with the message's text as `final_answer` (or an empty
string when it has no text); the trace retains the tool results. Unknown tools
produce a result with `error="tool_error"`. When the hook is omitted or returns
`False`, tool-bearing messages continue to the next model turn as usual. The hook
is configured by the caller, independently of the `train` and `prod` profiles.
Production response recovery rejects interrupted tool calls before executing
tools or invoking this hook.

## Image inspection

`prod` enables image reads through `read` and MCP images by default. Set
`enable_image_tools: false` for a text-only endpoint; model names are not used to infer
capabilities. `read` accepts PNG/JPEG/GIF/WebP, and converts BMP/TIFF, up to 20 MB of
input. Like Claude Code and ZCode, it downscales images to at most 2000 px on the longest
side and re-encodes them to at most 5 MB of base64, telling the model the original and
displayed sizes. Resizing uses Pillow when the execution environment has it; without
Pillow, images that need no resizing still pass through. Audio remains unsupported.
Render document pages to PNG, as the pdf skill describes, before inspecting them.

Native traces retain image data. API requests place images in a labeled user message
after all parallel tool results, because Chat Completions tool messages accept text
only. Compaction receives the actual images; unmeasured images provisionally count as
8192 tokens each until provider usage is available. This estimate is model-dependent,
not an exact visual-token budget. An explicit `enable_image_tools: true` also
enables image reads and MCP images in `train`; without it, training `read(image)` keeps
its disabled-vision response.
