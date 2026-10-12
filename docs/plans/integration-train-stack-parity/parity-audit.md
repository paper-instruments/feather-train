# Package source parity

Source: [Train PR 636](https://github.com/paper-instruments/train/pull/636),
`80c1deb8fe9eaceba5a4bc797d798c5434e63184`, `feather-workspace/`.
Extension base: standalone PR 11 at
`0d5267145c2e5621a9f83f3a2243c2b63090cced`.

| Package | Source files | Byte identical | Intentional differences | Missing |
| --- | ---: | ---: | ---: | ---: |
| Harness | 2650 | 2633 | 17 | 0 |
| Train | 122 | 106 | 16 | 0 |
| Eval | 31 | 30 | 1 | 0 |

Counts compare source-tracked package files, including docs/tests/assets, against
the destination. Caches and destination-only files are excluded from that source
inventory. Harness differences and destination subagents remain as documented in
[the original audit](../harness-train-stack-parity/parity-audit.md). Train retains
its destination-only `model_retry.py`. Eval executable code, tests, task example,
metadata and license match the pinned source; only its README adds standalone
configuration/evidence guidance and removes a stale unavailable-web claim.

## Imported behavior

Harbor now forwards model settings, compaction, prompt token budgets, web tools,
and additional system instructions into the validated CLI config. Its installer
supplies the current document toolkit and conditional image/tokenizer
dependencies. Prompt receipts, Harbor context usage and segmented ATIF metrics
follow the recorded native calls, including compaction. Failed inference requests
whose usage never reaches the trace remain outside those totals.

Train now carries source sampling/profile/budget settings, prompt and per-call
usage receipts, expected-world validation, image-read selection and research
controls. Native benchmark tasks support explicit shared/separate verifier modes,
workdirs, collection hooks, artifact sources, declared grading deadlines and JSON
rewards. The package also includes source egress enforcement and bounded local
benchmark deliverable capture. Modal volume capture stays an explicit optional
integration; ordinary imports and local capture require no Modal.

## Retained and corrected differences

- Keep the optional destination outage retry gate. With it unset, SDK retries use
  the source `model_request_retries` configuration. Keep redacted Compose errors
  rather than embedding arbitrary remote output in exception text.
- Validate full HTTP/SSE MCP descriptors and preserve headers, timeout and error
  policy through endpoint rewriting. Reject unsupported baked-world stdio rather
  than silently expanding the supported transport contract.
- Retain the destination assertion that four failed compaction-summary attempts
  close their spans; the pinned source's corresponding test expected one despite
  its newer Harness retry behavior. Update old lifecycle test fixtures to include
  the newly required mode field.
- Use the existing local byte-read recovery for optional reward.json reads. A
  dropped transport reply must not cause an already completed judge to run again;
  permanent authentication errors still propagate.
- Preserve Hugging Face snapshot sibling paths when finding a Dockerfile WORKDIR,
  and download that Dockerfile conditionally when official modes omit explicit
  workdir. Resolving the TOML cache symlink into blobs broke the source fallback.
- Reject explicit shared/separate modes for Factory tasks at the authoritative
  `FeatherTaskData` boundary. Those modes are available to benchmarks. Factory
  tasks must keep their baked peer verifier, solver-removal, private-state and
  submission-volume contract; accepting the source mode widening would bypass it.

Regression tests cover the descriptor, read-recovery, HF-cache and Factory-mode
differences. Public README/architecture/grading docs describe the standalone
interfaces and capability boundaries instead of private Train entrypoints.

## Scope boundary

No private Train infrastructure, benchmark presets/datasets or Datagen persistence
was imported. This does not implement snapshot baking, interactive/resumable CLI
sessions, custom-skill ownership, full failed-attempt spend accounting or the
separate PR 637 terminal-tool contract. Package publishing, repository visibility,
paid inference and live cloud qualification remain outside this update.

## Review and simplification

Independent review approved after correcting Factory isolation, HF-cache WORKDIR
fallback and reward.json read recovery. Independent checks passed 34 Eval and 130
focused Train tests. Manual simplification reused existing transport recovery and
one Factory validation boundary, preserved ownership, and introduced no cross-
experiment abstraction. Full verification is recorded in [verification.md](verification.md).
