# Build sequence

```yaml
phases:
  - id: 1
    name: Merge harness and integrate callers
    depends_on: []
  - id: 2
    name: Verify and review
    depends_on: [1]
  - id: 3
    name: Documentation and draft PR
    depends_on: [2]
```

## Tasks

- [x] Pin source/destination and audit both histories.
- [x] Merge reusable runtime, model, tools and regression tests.
- [x] Integrate local/subagent assembly and dependent callers.
- [x] Run package tests, lint, builds and clean installation.
- [x] Independent review and manual `/simplify` equivalent.
- [x] Update docs/provenance.
- [ ] Create the draft PR.

Retain destination-only supported features. Do not copy private Train code or
change benchmark/training behavior outside the reusable package boundary.
