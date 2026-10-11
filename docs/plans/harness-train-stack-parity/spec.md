---
title: Harness parity with Train stack PR 636
created: 2026-10-11
status: executed
---

# Harness parity with Train stack PR 636

## Motivation

Port the reusable harness changes accumulated in Train's stack containing PR 636
into the standalone Feather repository. Source: `paper-instruments/train` commit
`80c1deb8fe9eaceba5a4bc797d798c5434e63184`, package
`feather-workspace/feather-harness`; destination starts at `e4a1b14`.

## Requirements

- Bring over streaming/retries, provider usage, native reasoning replay, context
  budgets, compaction recovery, MCP bounds/timeouts, web guards, image reads,
  prompt composition, local runtime housekeeping, and their regression tests.
- Retain destination subagents, host-owned skill installation, symlink mutation
  fixes, Windows path handling, and the caller-owned end-after-tools hook.
- Keep distribution metadata/licenses/assets and existing downstream packages
  usable; adapt dependent call sites where the source changed harness APIs.
- Record intentional merge differences and verify all package suites and wheels.

## Scope

Harness code, tests, package metadata, lockfile, documentation, and necessary
integration adaptations are in scope. Train stack benchmark datasets, private
training infrastructure, eval presets, paid runs, and publishing packages are out
of scope. Deliver a draft PR against `feather-train/main`.

## Verification

Run Harness tests, Train tests without Docker, Eval tests, workspace Ruff, build
all distributions, and clean wheel installation/runtime generation. Review the
source-to-target diff and perform a simplification and independent code review.
Docker/platform checks unavailable locally are reported explicitly.
