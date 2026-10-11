# Phase 1 — Merge and integrate

Goal: source harness behavior plus existing standalone features.
Work: three-way merge against the initial standalone extraction; resolve runtime,
prompt, file-tool and local/subagent differences. Bring source regression tests.
Files: `feather-harness/`, `uv.lock`, and necessary dependent package callers/tests.
Constraints: preserve target-only capabilities, licenses and release metadata;
keep scripts provider-neutral. No cloud calls or benchmark ports.
Verification: Harness, Train non-Docker, Eval tests and workspace Ruff.
