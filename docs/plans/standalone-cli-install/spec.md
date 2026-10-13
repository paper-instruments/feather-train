---
status: executed
---

# Standalone CLI installation

Provide a one-command macOS/Linux installer for the Feather CLI and its pinned Office extra, without installing Python packages into the caller's project, active virtualenv, or system Python. Use a per-user installation under `~/.local/share/feather` and a launcher at `~/.local/bin/feather`. The launcher must expose that installation's Python to shell tools, preserving the caller's working directory and arguments.

Install the locked Harness/Office dependencies with private uv and managed Python. Fetch a requested Git revision, or install an explicitly provided local checkout. Build a fresh environment before switching the launcher; a failed update preserves the previous installation. Record the installed source revision. Do not edit shell profiles or require sudo. Document PATH setup, local-checkout installation, system tools, updates, and removal.

Out of scope: PyPI publication, public hosting/repository visibility changes, interactive CLI features, installing global OS packages, changing training or Harbor execution environments.

## Task Python override

Add CLI `--python` and optional JSON `python` to select the interpreter used by local Bash commands, including subagents. Keep Feather and its internal workers on their own interpreter. Route both `python` and `python3` to the selected executable, preserving virtualenv symlink paths. Expose the runtime interpreter through `FEATHER_PYTHON` for Paper tasks and describe it only in the local Bash tool metadata when an override is selected; do not change system prompts or skills. Do not install into or combine packages with the selected environment. Update README installation wording to describe public installation directly.
