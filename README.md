# Feather

Feather is a lightweight agent harness for knowledge work, with tools and skills
for documents, spreadsheets, presentations, and PDFs. Use it locally with an
OpenAI-compatible model, evaluate it on Harbor tasks, or integrate its rollouts
with model training.

## Packages

| Package | Purpose |
| --- | --- |
| [Feather Harness](feather-harness/README.md) | Agent loop, file and shell tools, Paper document skills, context compaction, and a local CLI. Includes two built-in harness profiles: **`prod`** (default), with professional guidance and image support; and **`train`**, with minimal prompts and original skill guidance for training. |
| [Feather Train](feather-train/README.md) | Rollouts and grading on snapshotted E2B worlds, Braintrust logging, and a reference Tinker training integration through Verifiers. |
| [Feather Eval](feather-eval/README.md) | Run Feather on Harbor tasks and export trajectories in Harbor's ATIF format. |

Each package is independently installable and publishable. A shared `uv` workspace
manages their development dependencies and lockfile.

## Run Feather locally

The [installer](install.sh) supports macOS and Linux and requires Bash, curl, and
Git. Install with:

```bash
curl -fsSL https://raw.githubusercontent.com/paper-instruments/feather-train/main/install.sh | bash
```

To install from a local checkout, run from its root:

```bash
FEATHER_SOURCE_DIR="$PWD" bash install.sh
```

The installer provisions its own uv and Python 3.12, then installs the CLI, Paper
document packages, PDF libraries, and data libraries into a dedicated Feather virtual environment
under `~/.local/share/feather/`. It does not install Python packages into your
current directory, active environment, or system Python. The launcher at
`~/.local/bin/feather` sets the execution environment so the model's `python` and
`python3` shell commands use those installed packages by default. No activation is needed.
If `~/.local/bin` is not on your `PATH`, invoke `~/.local/bin/feather` directly or
add that directory to `PATH` yourself; the installer does not edit shell profiles.

Install ripgrep separately; PDF rendering needs Poppler, and spreadsheet
recalculation needs LibreOffice. See [Harness setup](feather-harness/README.md#recommended-packages)
for system packages and installation options. Then run:

```bash
mkdir -p workspace
export OPENAI_API_KEY=YOUR_API_KEY

~/.local/bin/feather \
  --model YOUR_MODEL --base-url https://YOUR_PROVIDER/v1 \
  --workdir "$PWD/workspace" \
  "Summarize the documents in this workspace into briefing.md." > run.json
```

Put your input files in `workspace/`. `run.json` contains the final answer and the
agent's execution trace. The model identifier and endpoint come from your provider;
omit `--base-url` for the OpenAI client's default endpoint. Shell tools run with
your OS permissions. Running Feather creates skill and log directories inside the
selected workspace.

To use a project's Python for task commands, pass
`--python /absolute/path/to/project/.venv/bin/python` or set `python` in the JSON
configuration. Feather keeps its own runtime environment. Task shell commands
can use `"$FEATHER_PYTHON"` to access the bundled Paper packages. See
[task Python selection](feather-harness/docs/usage.md#task-python) for examples.

For evaluation, follow the [Harbor example](feather-eval/README.md#first-run).
For training, start with [Feather Train](feather-train/README.md): it takes task
datasets and prepared E2B snapshots as inputs.

## Use the packages from another repository

Pin the packages you use to the same full Git commit. For a uv-managed consumer,
keep package dependencies in the `dependencies` array under `[project]` and set
their source:

```toml
[tool.uv.sources]
feather-harness = { git = "https://github.com/paper-instruments/feather-train.git", rev = "<full-commit-sha>", subdirectory = "feather-harness" }
feather-train = { git = "https://github.com/paper-instruments/feather-train.git", rev = "<full-commit-sha>", subdirectory = "feather-train" }
feather-eval = { git = "https://github.com/paper-instruments/feather-train.git", rev = "<full-commit-sha>", subdirectory = "feather-eval" }
```

Replace `<full-commit-sha>` with the commit you want to consume and run `uv lock`.
Include the `feather-harness` source when using Train or Eval so their shared
harness dependency resolves to this repository. Omit packages you do not use.
Choose Train's `verifiers`, `tinker`, or `training` extra for the integration you
need. Commit the consumer's lockfile to preserve the resolved revisions.

## Development

From this workspace directory:

```bash
uv sync --package feather-harness --extra office --locked
uv run --package feather-harness --extra office --locked feather \
  --model YOUR_MODEL --workdir ./workspace "Summarize this workspace."

uv sync --all-packages --all-extras --no-extra token-count --locked
uv build --all-packages
```

The optional Harness `token-count` extra uses the GLM image processor's pinned
Transformers version and cannot share an environment with Train's `training`
extra. Install it separately with `uv sync --package feather-harness --extra
token-count --extra dev --locked` when using exact GLM prompt budgets.

The package READMEs contain their test and lint commands. Run their pytest suites
separately because each has its own configuration. See [AGENTS.md](AGENTS.md) for
code quality and contribution guidance.

CI runs Linux lint and package tests, Docker integration tests, and clean
installation checks for the built distributions. Windows CI also checks file-tool portability. These checks need no cloud credentials.

## License

[MIT](LICENSE).
