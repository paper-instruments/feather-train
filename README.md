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

Use Linux with Python 3.12+, uv, Bash, and ripgrep. PDF rendering also needs Poppler.
See [Harness setup](feather-harness/README.md#install-and-run) for system packages.
From this workspace directory:

```bash
uv sync --package feather-harness --extra office --locked
mkdir -p workspace
export OPENAI_API_KEY=YOUR_API_KEY

uv run --package feather-harness --extra office feather \
  --model YOUR_MODEL --base-url https://YOUR_PROVIDER/v1 \
  --workdir "$PWD/workspace" \
  "Summarize the documents in this workspace into briefing.md." > run.json
```

Put your input files in `workspace/`. The `office` extra supplies the Paper packages
and PDF libraries. `run.json` records the agent's execution trace.

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

Source and verification records for the package sync are in
[Harness parity](docs/plans/harness-train-stack-parity/parity-audit.md) and
[Train and Eval parity](docs/plans/integration-train-stack-parity/spec.md).

## Development

From this workspace directory:

```bash
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
