# Source parity audit

Source: [Train PR 636](https://github.com/paper-instruments/train/pull/636),
`80c1deb8fe9eaceba5a4bc797d798c5434e63184`,
`feather-workspace/feather-harness/`. Destination base: `e4a1b14`.

All 2,650 source package files exist in the destination. 2,633 match byte for byte.
The other 17 retain standalone capabilities or describe their merged behavior:

| Paths | Reason for difference |
| --- | --- |
| `README.md`, `docs/profiles.md` | Document combined features, retained hook, and actual compaction retry behavior. |
| `src/feather_harness/__init__.py`, `cli.py`, `local/__init__.py` | Keep subagent exports/configuration and lifecycle; add source features to parent and child runs. |
| `src/feather_harness/core/runner.py`, `tests/core/test_tool_execution.py` | Keep caller-owned `end_run_after_tools` hook and its regression tests. |
| `src/feather_harness/production.py`, `scripts/install_skills.py`, `production_skills/pptx.md` | Keep host-owned skill tree/path APIs, explicit host/vision overrides, and quoting for skill paths containing spaces. Default production wording matches source. |
| `src/feather_harness/local/filesystem.py`, `scripts/_files.py`, `scripts/_search.py`, `tests/tools/test_windows_portability.py` | Keep MSYS revision normalization, per-user spill ownership and cross-drive search display. Adopt source portable spill behavior and tests. |
| `src/feather_harness/scripts/process_worker.py`, `tests/local/test_process_cleanup.py` | Keep standalone wall-clock termination grace period including process-discovery time. |
| `tests/local/test_profiles.py` | Combine source profile expectations with retained host-owned skill tests. |

Destination-only subagent modules, tests, and documentation remain. All other
source code, tests, metadata, skills, icons and licenses are identical. No source
file is missing. The source's separate `view_image` worker/tool removal is carried
over: image inspection uses `read` in both profiles when explicitly enabled.

Minimal integration adaptations remove that tool from Train's selection, adopt
the source prompt/tool signatures, allow images in training reads, and raise the
remote JSON response budget for `read`. The Braintrust compaction failure test now
checks that all four empty-summary attempts close their spans without opening an
execution window. Eval documents the new image tool contract.

The new `token-count` extra pins Transformers 5.17.0. As in Train, the workspace
declares its conflict with the training cookbook's older Transformers requirement.
CI and normal development exclude `token-count` when installing all training
extras; token-count is installed and tested in its own environment.

An independent review found and resolved the Windows cross-drive search regression.
The manual simplification pass reviewed the 17 intentional differences, model-call
and trace ownership, parent/child assembly, remote image budgets, and packaging.
No additional abstraction or unrelated private Train code was introduced.
