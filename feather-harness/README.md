# Feather Harness

Feather is a lightweight agent harness for knowledge work. It combines file and
shell tools with skills for working on documents, spreadsheets, presentations,
and PDFs using specialized Paper packages.

- **Knowledge-work skills:** bundled workflows and API references for working with
  Word documents, Excel spreadsheets, PowerPoint presentations, and PDFs.
- **Tools:** file editing, search, shell execution, background tasks, todo lists,
  connections to MCP servers, and optional Exa web search and fetch.
- **Long-running tasks:** context compaction summarizes earlier work while retaining
  recent turns and todo state.
- **CLI and library:** run against an OpenAI-compatible Chat Completions endpoint
  or embed the loop with your own model client and tools.
- **Execution traces:** structured results preserve tool interactions and context
  changes for inspection.

## Prompting

Ultramarine was trained with large-scale non-verifiable rewards, so it has stronger
aesthetic and prose-writing sensibilities than traditional models. Tune your prompts
carefully to get the most out of it, and test changes on your own tasks.

This harness contains the prompts we've tuned for internal testing. Start with:

- **System prompts:** [`system_prompt.md`](src/feather_harness/system_prompt.md)
  (`train`) and [`production_prompt.md`](src/feather_harness/production_prompt.md)
  (`prod`), and how the [two profiles](#two-profiles-train-and-prod) differ.
- **Tool descriptions:** the originals in [`tools/`](src/feather_harness/tools/) and
  the production rewrites in
  [`production_tool_descriptions.json`](src/feather_harness/production_tool_descriptions.json).
- **Skills:** the format manuals in [`skills/`](src/feather_harness/skills/) and the
  production guidance and design guide in
  [`production_skills/`](src/feather_harness/production_skills/).
- **Compaction prompt:**
  [`production_compaction_prompt.md`](src/feather_harness/production_compaction_prompt.md).

See [Prompting](https://docs.paperinstruments.com/docs/ultramarine/prompting) in the
Ultramarine docs for request settings such as reasoning effort and sampling.

## How it works

The whole harness is a few small pieces. If you want to rebuild it yourself, these
are the files to read.

| Piece | Where |
| --- | --- |
| Agent loop | [`core/runner.py`](src/feather_harness/core/runner.py) (`FeatherAgent`) |
| Model adapter (OpenAI Chat Completions) | [`core/model.py`](src/feather_harness/core/model.py) |
| Local assembly: prompt, tools, skills, profile | [`local/__init__.py`](src/feather_harness/local/__init__.py) (`run_local`) |
| CLI | [`cli.py`](src/feather_harness/cli.py) |

**The loop** is a plain sequential cycle: call the model with the context and tool
schemas, run each tool call in order, append the results, and repeat. The run ends
when the model replies without tool calls or `max_turns` is reached. Before each
model call the loop checks whether to compact the context (see below).

### Two profiles: `train` and `prod`

There is one loop and one set of tools. A profile only changes the text the model
sees and a few defaults. `prod` is the default; `train` reproduces the conditions
the model was trained under.

| | `train` | `prod` |
| --- | --- | --- |
| System prompt | [`system_prompt.md`](src/feather_harness/system_prompt.md) | [`production_prompt.md`](src/feather_harness/production_prompt.md) |
| Tool descriptions | Originals in [`tools/`](src/feather_harness/tools/) | Overrides in [`production_tool_descriptions.json`](src/feather_harness/production_tool_descriptions.json) |
| Skills | Technical manuals in [`skills/`](src/feather_harness/skills/) | Same manuals + guidance in [`production_skills/`](src/feather_harness/production_skills/), a shared design guide, and an icon set |
| Compaction prompt | `DEFAULT_COMPACTION_PROMPT` in [`core/compaction.py`](src/feather_harness/core/compaction.py) | [`production_compaction_prompt.md`](src/feather_harness/production_compaction_prompt.md) |
| Image reads | Off | On |
| Cut-off or empty model response | No special handling | One recovery per run |
| Parallel subagents | Not available | Opt-in |

Profile wiring lives in [`production.py`](src/feather_harness/production.py). See
[profiles](docs/profiles.md) for the full comparison.

### System prompts

- **Train:** [`system_prompt.md`](src/feather_harness/system_prompt.md) is a
  deliberately minimal six-line prompt. [`prompt.py`](src/feather_harness/prompt.py)
  (`build_instructions`) appends the skills catalog.
- **Prod:** [`production_prompt.md`](src/feather_harness/production_prompt.md) is a
  professional knowledge-work prompt (scoping the request, evidence and sources,
  numbers and models, deliverables). `build_production_instructions` in
  [`production.py`](src/feather_harness/production.py) adds host capabilities, the
  same skills catalog, and any `additional_instructions`.

### Tools

Each tool is a JSON schema plus an async function that takes raw arguments and
returns a result ([`tools/__init__.py`](src/feather_harness/tools/__init__.py)).

| Tools | What they do | Source |
| --- | --- | --- |
| `read`, `write`, `edit`, `grep`, `glob` | Workspace-scoped file access. `edit` and overwriting `write` require a prior `read` of the current file. In `prod`, `read` also returns images. | [`tools/files.py`](src/feather_harness/tools/files.py) |
| `bash`, `task_list`, `task_output`, `task_stop` | Shell commands, with background tasks for long-running work | [`tools/process.py`](src/feather_harness/tools/process.py) |
| `todo_write` | Session todo list, carried across compaction | [`tools/todo.py`](src/feather_harness/tools/todo.py) |
| `mcp__<server>__<tool>` | Tools from configured MCP servers | [`tools/mcp.py`](src/feather_harness/tools/mcp.py) |
| `spawn_agent`, `wait_agent`, `followup_agent`, `stop_agent`, `list_agents` | Parallel subagents (`prod`, opt-in) | [`tools/agents.py`](src/feather_harness/tools/agents.py), [`subagents.py`](src/feather_harness/subagents.py) |
| `web_search`, `web_fetch` | Exa-backed web access (opt-in with `enable_web_tools`) | [`tools/web/`](src/feather_harness/tools/web/) |

See the [tool reference](docs/tools.md) for arguments and behavior.

### Skills

[`skills/`](src/feather_harness/skills/) holds one skill per format (`docx`, `pptx`,
`xlsx`, `pdf`): a `SKILL.md` entry point plus API references and scripts. At
startup they are copied into the workspace's `.agents/` directory
([`scripts/install_skills.py`](src/feather_harness/scripts/install_skills.py)), and
the system prompt lists each skill's name and description. The model reads the
`SKILL.md` itself when a task needs it.

### Context compaction

Compaction is off by default in the CLI and `run_local`; enable it with
`"compaction": {"enabled": true, "context_window_tokens": ...}` set to your model's
real window. Feather Train's training and eval configs turn it on. It lives in
[`core/compaction.py`](src/feather_harness/core/compaction.py), called from the loop
in [`core/runner.py`](src/feather_harness/core/runner.py).

1. **Trigger.** Before each model call, estimate the context size: the token count
   the provider reported for the last call, plus roughly one token per three
   characters of newer messages (8,192 per image). Compact when the estimate reaches
   `min(85% of the context window, window − 30,000)`, or an explicit
   `trigger_tokens`.
2. **Split.** Keep the system prompt, the original task prompt (when the run started
   from a single prompt), and the last `keep_recent_turns` assistant turns (default
   2). Everything in between gets summarized.
3. **Summarize.** The same model writes a continuation summary using the compaction
   prompt (up to 65,536 output tokens by default). An empty summary is retried, up
   to four attempts.
4. **Rebuild.** The new context is: system prompt, task prompt, a
   `## Summary of Previous Work` message (with the todo list as it stood at the
   boundary appended), then the recent turns. Replayed model reasoning is dropped
   at the boundary.

Each compaction is recorded as its own segment in the `RunResult` trace, so the
full history stays inspectable.

## Install and run

There are four ways to use Feather:

- **[Run the CLI](#run-the-cli):** install a standalone `feather` command and run
  tasks against a folder of files. No checkout needed.
- **[Build from source](#build-from-source):** clone this repository to modify the
  harness or run it from a checkout.
- **[Use it in your project](#use-it-in-your-project):** add the harness as a Python
  dependency and call the agent from your own code.
- **[Evaluate on Harbor tasks](#evaluate-on-harbor-tasks):** run Feather inside
  [Harbor](https://pypi.org/project/harbor/) task sandboxes with
  [`feather-eval`](../feather-eval/README.md).

The CLI and the library both run shell commands locally with your OS permissions,
and both need the [system packages](#recommended-packages).

### Run the CLI

The repository's [installer](../install.sh) works on macOS and Linux. It needs Bash,
curl, and Git, and brings its own uv and Python 3.12:

```bash
curl -fsSL https://raw.githubusercontent.com/paper-instruments/feather-train/main/install.sh | bash
```

It installs Feather and the [recommended Python packages](#recommended-packages)
into their own environment and puts a launcher at `~/.local/bin/feather`. It does
not require sudo or modify shell profiles. Then run a task:

```bash
mkdir -p workspace
export OPENAI_API_KEY=YOUR_API_KEY

~/.local/bin/feather \
  --model YOUR_MODEL --base-url http://localhost:8000/v1 \
  --workdir "$PWD/workspace" "Summarize the documents in this workspace into a briefing.md file." > run.json
```

Use the model identifier and endpoint supplied by your provider. Omit `--base-url`
for the OpenAI client's default endpoint. The default `prod` profile expects an
image-capable model; pass `--profile train` for the minimal prompt. The CLI writes
a `RunResult` JSON trace to stdout.

See the [usage reference](docs/usage.md) for installer options, task Python, JSON
config, compaction, MCP, and subagents.

### Build from source

Clone the repository and run the CLI from its uv workspace (Python 3.12+):

```bash
git clone https://github.com/paper-instruments/feather-train.git
cd feather-train
uv sync --package feather-harness --extra office --locked
mkdir -p workspace

uv run --package feather-harness --extra office --locked feather \
  --model YOUR_MODEL --workdir ./workspace "Summarize this workspace." > run.json
```

This uses the repository's `.venv`, so model shell commands see its packages when you
launch through `uv run`. See [build from source](docs/usage.md#build-from-source) and
[development](#development) for tests and linting.

### Use it in your project

Harness is not on PyPI yet. Add it from Git, pinned to a commit, with the `office`
extra:

```bash
uv add "feather-harness[office] @ git+https://github.com/paper-instruments/feather-train.git@<full-commit-sha>#subdirectory=feather-harness"
```

`run_local` runs the same agent as the CLI, with its prompt, tools, skills, and
profile. You supply the model client:

```python
import asyncio
from pathlib import Path

from openai import AsyncOpenAI
from feather_harness import ChatCompletionsModel, run_local

async def main() -> None:
    workspace = Path("workspace")
    workspace.mkdir(exist_ok=True)
    async with AsyncOpenAI(base_url="http://localhost:8000/v1") as client:
        model = ChatCompletionsModel(client=client, model="YOUR_MODEL")
        result = await run_local(
            "Summarize the documents in this workspace into a briefing.md file.",
            model=model,
            workspace=workspace,
        )
        print(result.final_answer)

asyncio.run(main())
```

The agent's shell commands use the Python environment your code runs in, which is
why the `office` extra goes into your project. To bring your own tools or loop,
build `FeatherAgent` directly; see [library use](docs/usage.md#library-use).

### Evaluate on Harbor tasks

[`feather-eval`](../feather-eval/README.md) is a Harbor agent adapter. It installs
Feather into each task's sandbox, runs it against an OpenAI-compatible endpoint, and
exports the trajectory as ATIF. Harbor manages the sandbox and grading. See its
[first run](../feather-eval/README.md#first-run) to evaluate a bundled smoke task.

## Recommended packages

The model works best with these available in the agent's environment: the bundled
skills use some of them, and the model's own task scripts expect the rest. The
installer (or the `office` extra) provides the Python packages; system packages
are separate.

**Python** (`office` extra):

| Package | Used for |
| --- | --- |
| [`paper-docx`](https://pypi.org/project/paper-docx/) | Word documents (imports as `docx`) |
| [`paper-pptx`](https://pypi.org/project/paper-pptx/) | PowerPoint presentations (imports as `pptx`) |
| [`paper-xlsx`](https://pypi.org/project/paper-xlsx/) | Excel workbooks (imports as `openpyxl`) |
| `reportlab`, `pypdf`, `pdfplumber`, `pypdfium2` | PDF authoring, extraction, and rendering |
| `pdf2image`, `Pillow` | Rendering pages to images and resizing images for `read` |
| `pandas` (2.x), `requests`, `beautifulsoup4`, `lxml` | Data analysis, HTTP, and HTML/XML parsing in task scripts |

The Paper packages are drop-in upgrades of `python-docx`, `python-pptx`, and
`openpyxl` that add safer edits and inspection. Do not install the upstream
distributions alongside them.

[`PyMuPDF`](https://pypi.org/project/PyMuPDF/) is also worth installing for PDF
work, but it is not in the extra because it is AGPL-licensed. Add it yourself if
its license suits you.

**System** (Debian/Ubuntu shown; on macOS use your package manager):

```bash
sudo apt-get update
sudo apt-get install -y bash ripgrep poppler-utils libreoffice-calc tesseract-ocr \
  git jq curl zip unzip fonts-dejavu-core fonts-liberation
```

| Package | Used for |
| --- | --- |
| `bash`, `ripgrep` | Required: the `bash`, `grep`, and `glob` tools |
| `poppler-utils` | PDF rendering |
| `libreoffice-calc` | Formula recalculation (`openpyxl.oracle`) |
| `tesseract-ocr` | OCR for scanned documents (add the language packs you need) |
| `fonts-dejavu-core`, `fonts-liberation` | Fonts for rendering documents through LibreOffice |
| `git`, `jq`, `curl`, `zip`, `unzip` | Common command-line tools in task scripts |

Only `bash` and `ripgrep` are strictly required; without the rest, the related
work fails or renders differently.

## Development

From the workspace root:

```bash
uv run --package feather-harness --extra dev pytest -c feather-harness/pyproject.toml feather-harness/tests -q
uv run --package feather-harness --extra dev ruff check feather-harness/src feather-harness/tests
```

Tests require Bash and ripgrep, with local test servers for model and MCP calls.
Linux-specific process tests skip on macOS.

## License

[MIT](LICENSE).
