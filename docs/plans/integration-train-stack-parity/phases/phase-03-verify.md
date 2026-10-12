# Verification and review

Files: all changed package code may be reviewed, workspace metadata and CI only
when needed for the port; this plan's verification/provenance files.

Run all package suites and Docker integration, workspace Ruff, lock check,
wheel/sdist builds and clean installed runtime/configuration probes. Review the
complete diff with a code-reviewer and address concrete findings. Reread for
unnecessary indirection/dead code without refactoring inherited research code.

Done when: fresh verification output supports the result and source deltas are
documented. Unavailable live model/sandbox qualification remains explicit.
