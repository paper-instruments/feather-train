# Contributing to Feather

Thanks for contributing. Feather is a Python 3.12 `uv` workspace with three
independently packaged projects:

| Package | Scope |
| --- | --- |
| `feather-harness` | Agent loop, local tools, skills, and the `feather` CLI |
| `feather-train` | Remote rollouts, grading, observability, and training integrations |
| `feather-eval` | Harbor adapter and ATIF export |

Read [AGENTS.md](AGENTS.md) before changing code. It defines the repository's
design, testing, and review conventions.

## Set up a fork

Install [uv](https://docs.astral.sh/uv/), Git, Bash, and ripgrep. Then fork the
repository and clone your fork:

```bash
git clone https://github.com/YOUR-USER/feather-train.git
cd feather-train
git remote add upstream https://github.com/paper-instruments/feather-train.git
```

Keep each branch focused on one logical change. Start every branch from the
current upstream `main`, so it contains only the commits intended for its pull
request:

```bash
git fetch upstream
git switch -c your-change upstream/main
```

## Install and run focused checks

Use the smallest environment and test suite that covers the package you changed.
The lockfile is authoritative, so include `--locked` when syncing.

### Harness

```bash
uv sync --package feather-harness --extra office --extra dev --locked
uv run --no-sync ruff check feather-harness/src feather-harness/tests
uv run --no-sync pytest -c feather-harness/pyproject.toml feather-harness/tests -q
```

The image tests import Pillow, which the `office` extra provides. The optional
`token-count` extra must be installed separately from Train's `training` extra.

### Train

Train's reference recipe and tests use its training integrations:

```bash
uv sync --package feather-train --extra training --extra dev --locked
uv run --no-sync ruff check feather-train/src feather-train/tests
uv run --no-sync pytest -c feather-train/pyproject.toml feather-train/tests -m 'not docker' -q
```

Docker-marked tests additionally require the test image:

```bash
docker build -t feather-e2b-test -f feather-train/tests/Dockerfile feather-train/tests
uv run --no-sync pytest -c feather-train/pyproject.toml feather-train/tests -m docker -q
```

### Eval

```bash
uv sync --package feather-eval --extra dev --locked
uv run --no-sync ruff check feather-eval/src feather-eval/tests
uv run --no-sync pytest -c feather-eval/pyproject.toml feather-eval/tests -q
```

## Run the workspace checks

Before submitting a change that crosses packages, run the same non-Docker scope
as the main CI job:

```bash
uv sync --all-packages --all-extras --no-extra token-count --locked
uv run --no-sync ruff check feather-harness/src feather-harness/tests \
  feather-train/src feather-train/tests feather-eval/src feather-eval/tests scripts
uv run --no-sync pytest -c feather-harness/pyproject.toml feather-harness/tests -q
uv run --no-sync pytest -c feather-train/pyproject.toml feather-train/tests -m 'not docker' -q
uv run --no-sync pytest -c feather-eval/pyproject.toml feather-eval/tests -q
```

Packaging changes also need a distribution build. CI installs the built wheels
into a fresh environment outside the checkout and runs the entrypoints; to repeat
that locally:

```bash
uv build --all-packages --out-dir dist
uv venv /tmp/feather-wheels
uv pip install --python /tmp/feather-wheels/bin/python dist/*.whl
(cd /tmp && feather-wheels/bin/feather --help && feather-wheels/bin/feather-atif --help)
```

## Submit the pull request

- Explain the problem and the observable behavior of the solution.
- Link the relevant issue when one exists.
- Include meaningful tests with behavior changes and list the checks you ran.
- Keep generated files, credentials, local workspaces, and unrelated cleanup out
  of the commit.
- Call out intentional behavior changes and any remaining uncertainty.

CI runs lint, package tests, Docker integration tests, Windows file-tool tests,
and distribution installation checks. A pull request is ready for review when its
scope is clear, its relevant local checks pass, and the CI results are green.
