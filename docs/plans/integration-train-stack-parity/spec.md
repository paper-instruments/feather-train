---
title: Standalone package parity with the current Train stack
created: 2026-10-12
status: executed
---

# Standalone package parity with the current Train stack

Extend PR 11 beyond its completed agent-harness sync to the reusable Train and
Eval packages at Train PR 636, commit
`80c1deb8fe9eaceba5a4bc797d798c5434e63184`, `feather-workspace/`.
The user authorized updating the existing PR. Preserve its additive history and
standalone-only features; do not replace it with a sibling branch.

## Requirements

- Preserve existing agent-harness parity, subagents, host-owned skills, safe file
  mutation/platform behavior and the caller completion hook.
- Port generic Harbor configuration forwarding, toolkit installation, prompt and
  usage receipts, ATIF updates, and their regression tests.
- Port reusable native Train contracts: sampling/profile/compaction forwarding,
  MCP descriptor preservation, Harbor verifier/workdir/timeout modes, task
  validation, egress restrictions, local deliverable capture, and observability.
- Keep packages independently installable without the private Train checkout.
  Preserve meaningful destination-only fixes and document any intentional delta.
- Update public package docs and provenance, verify the whole workspace and clean
  distributions, review/simplify, and push only this extension to PR 11.

## Out of scope

Private Train infrastructure, benchmark datasets/presets, Datagen/Observatory,
live model/cloud runs, package publishing and visibility changes are excluded.
New customer product features from the audit (chat/resume, skill ownership,
retry-spend accounting, snapshot baking) and the separate PR 637 terminal-tool
feature are not part of a sync to the PR 636 source. Generic local capture may be
ported without requiring Modal for its ordinary local path.

## Verification

Run Harness, Train non-Docker, Train Docker and Eval package suites, workspace
Ruff and lock consistency. Build all wheels/sdists, install outside the checkout,
check installed entrypoints and generated runtime, and exercise installed Harbor
configuration forwarding. Run independent code review and manual simplification.
Record source comparisons and report any remaining platform/live-call limits.
