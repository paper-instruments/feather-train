# Verification

## Package suites

- Harness: `uv run --no-sync pytest -c feather-harness/pyproject.toml feather-harness/tests -q`
  — 509 passed, 5 platform skips.
- Train on macOS: `uv run --no-sync pytest -c feather-train/pyproject.toml feather-train/tests -m 'not docker' -k 'not test_connection_pool_admits_more_than_verifiers_default_128_requests' -q`
  — 291 passed, 68 deselected.
- Docker integration: `uv run --no-sync pytest -c feather-train/pyproject.toml feather-train/tests -m docker -q`
  — 67 passed, 292 deselected. Built `feather-e2b-test` from the checked-in Dockerfile.
- Eval: `uv run --no-sync pytest -c feather-eval/pyproject.toml feather-eval/tests -q`
  — 17 passed.
- Isolated token-count environment: `UV_PROJECT_ENVIRONMENT=/tmp/feather-parity-token-count uv sync --package feather-harness --extra token-count --extra dev --locked`, followed by
  `UV_PROJECT_ENVIRONMENT=/tmp/feather-parity-token-count uv run --no-sync pytest -c feather-harness/pyproject.toml feather-harness/tests/core/test_token_budget.py -q`
  — 33 passed. Imported `Glm5NextImageProcessorPil` from the installed Transformers 5.17.0.

The initial full macOS Train suite found the obsolete single-attempt compaction
span assertion; it now checks all four retry spans and passes. The unchanged
129-connection pool test fails on macOS. A standalone aiohttp/httpx probe without
Feather reproduced the cause: `kern.ipc.somaxconn=128`, with 128 simultaneous
requests succeeding and the 129th reset; staggering requests succeeds. Its code
and locked aiohttp/HTTPX/HTTPcore/AnyIO/Verifiers versions match the base branch.
The full suite is therefore also verified in Linux:

```bash
docker run --rm \
  -v "$PWD:/workspace:ro" -w /workspace \
  -e UV_PROJECT_ENVIRONMENT=/tmp/feather-linux-venv \
  feather-e2b-test bash -c 'pip install -q uv==0.11.14 && uv sync --package feather-train --extra training --extra dev --locked && uv run --no-sync pytest -p no:cacheprovider -c feather-train/pyproject.toml feather-train/tests -m "not docker" -q'
```

Result: **292 passed, 67 deselected**. Together with the Harness, Docker and Eval
suites this verifies 885 distinct passing tests. An initial Linux setup without
the `training` extra could not collect cookbook tests; the final command installs
the required extra and runs the entire non-Docker suite.

## Lint, distributions and review

- `uv run --no-sync ruff check feather-harness/src feather-harness/tests feather-train/src feather-train/tests feather-eval/src feather-eval/tests` — passed.
- `uv lock --check` and `git diff --check` — passed.
- `uv build --all-packages --out-dir /tmp/feather-parity-dist` — all three wheels and sdists built.
- Installed all wheels into `/tmp/feather-parity-wheels` and ran from `/tmp`, outside
  the checkout. `feather --help`, `feather-atif --help`, installed exports, and
  `python -m feather_train.sandbox.runtime /tmp/feather-parity-runtime.tar.gz` pass.
  Runtime includes `read_tool.py`, `_images.py`, process/call/remote workers and all
  four skill manuals, with no obsolete `view_image_tool.py`.
- Source file audit: all 2,650 files present; 2,633 byte-identical. Remaining
  differences are documented in [parity-audit.md](parity-audit.md).
- Independent review approved after restoring cross-drive Windows search behavior.
  Manual simplification pass complete. No model, paid eval, or cloud sandbox run.

Native Windows validation remains in the existing Windows CI job; its platform
tests skip on macOS. Linux-specific Harness process cases also skip on macOS;
the Train Docker suite exercises the actual Linux runtime/process lifecycle.
