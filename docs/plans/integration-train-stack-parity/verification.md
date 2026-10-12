# Verification

Source: Train PR 636 at `80c1deb8fe9eaceba5a4bc797d798c5434e63184`.
PR extension base: `0d5267145c2e5621a9f83f3a2243c2b63090cced`.
Checks below ran on the completed extension on October 12, 2026.

## Package suites

- `uv sync --all-packages --all-extras --no-extra token-count --locked` — passed.
- `uv run --no-sync pytest -c feather-harness/pyproject.toml feather-harness/tests -q`
  — **509 passed, 5 platform skips** on macOS.
- `uv run --no-sync pytest -c feather-eval/pyproject.toml feather-eval/tests -q`
  — **34 passed**.
- Train non-Docker on macOS, excluding only
  `test_connection_pool_admits_more_than_verifiers_default_128_requests` —
  **443 passed, 69 deselected**. The unmodified full macOS run reproduced its
  existing listener-limit failure with 443 other passes.
- `docker build -t feather-e2b-test -f feather-train/tests/Dockerfile feather-train/tests`
  and `uv run --no-sync pytest -c feather-train/pyproject.toml feather-train/tests -m docker -q`
  — **68 passed, 444 deselected** on the final code.

The complete unmodified Train non-Docker suite also ran in Linux:

```bash
docker run --rm -v "$PWD:/workspace:ro" -w /workspace \
  -e UV_PROJECT_ENVIRONMENT=/tmp/feather-linux-sync \
  feather-e2b-test bash -c \
  'pip install -q uv==0.11.14 && uv sync --package feather-train --extra training --extra dev --locked && uv run --no-sync pytest -p no:cacheprovider -c feather-train/pyproject.toml feather-train/tests -m "not docker" -q'
```

Result: **444 passed, 68 deselected**. No test weakening or production connection-
pool change was used to handle the macOS limit.

## Lint and distributions

- Workspace Ruff over all three packages' source/tests — passed.
- `uv lock --check` and `git diff --check` — passed.
- `uv build --all-packages` — all three wheels and source distributions built;
  wheels were built from their source distributions.
- Clean Python 3.12 wheel environment outside the checkout, installing all wheels
  plus Harness `[office]` and Train `[verifiers]` — passed; dependency check found
  **163 compatible packages**.
- Installed `feather --help`, `feather-atif --help`, and runtime generation — passed.
  Runtime includes read/process/call/remote workers and all four skill manuals.
- Installed Harbor construction plus serialized `LocalConfig` round trip retains
  max output, enabled compaction and its summary limit, web enablement and generic
  additional instructions. Installed native Train config accepts research controls.
- New capture/egress scripts are present in the Train wheel. All package wheels
  retain their license metadata. Ordinary local capture imports require no Modal.
- A bounded credential-pattern scan over the changed package code/tests found no
  representative private-key, GitHub-token or Slack-token patterns.

## Review and scope

Independent review approved after correcting Factory mode isolation, HF-cache
WORKDIR fallback and JSON reward read recovery. Independent checks passed
**34 Eval** and **130 focused Train** tests. Manual simplification retained one
Factory validation boundary and reused existing byte-read recovery without
introducing a private dependency or training abstraction.

Source inventory has zero missing files across all three packages. Intentional
differences are recorded in [parity-audit.md](parity-audit.md).
Native Windows checks remain in the existing CI job. No real model inference,
E2B bake/resume, customer-data task, package publishing or cloud qualification
was performed. This verification qualifies the package sync, not an end-to-end
customer deployment or model performance.
