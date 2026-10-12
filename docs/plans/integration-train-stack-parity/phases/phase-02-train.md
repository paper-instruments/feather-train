# Train integration

Goal: match reusable runtime/task/grading behavior of the pinned Train source.

Files: `feather-train/**` only.

Work: compare code/tests, preserve destination-only fixes, and port current
runtime/MCP descriptors, Harbor verifier modes/workdir/timeouts, config/sampling,
task validation, egress, local capture and evidence. Keep optional cloud-volume
capture explicit and ordinary imports/local capture usable without Modal.
Update docs to reflect the standalone APIs rather than private eval.py commands.

Done when: Train non-Docker tests and scoped Ruff pass; root will run full Docker
integration and clean installed distributions. No cloud calls or new baking
product workflow are part of this sync.
