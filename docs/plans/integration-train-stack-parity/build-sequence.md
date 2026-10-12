# Build sequence

```yaml
phases:
  - id: 1
    name: Port Eval integration
    depends_on: []
  - id: 2
    name: Port Train integration
    depends_on: []
  - id: 3
    name: Verify review and simplify
    depends_on: [1, 2]
  - id: 4
    name: Document provenance and update PR 11
    depends_on: [3]
```

## Tasks

- [x] Pin current source and destination; compare package boundaries.
- [x] Port Eval integration and its regression tests.
- [x] Port Train integration and its regression tests.
- [x] Run workspace suites, Docker, lint and clean package checks.
- [x] Run independent review and manual simplification.
- [x] Document intentional deltas and verified evidence.
- [x] Push additive commit and update PR 11's title/body.

The two implementation phases own disjoint package directories. The primary
agent owns workspace metadata, plan/provenance docs, cross-package checks, review
fixes outside worker ownership, and publication.
