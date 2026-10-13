---
status: executed
---

# Standalone CLI installation

Provide a one-command macOS/Linux installer for the Feather CLI and its pinned Office extra, without installing Python packages into the caller's project, active virtualenv, or system Python. Use a per-user installation under `~/.local/share/feather` and a launcher at `~/.local/bin/feather`. The launcher must expose that installation's Python to shell tools, preserving the caller's working directory and arguments.

Install the locked Harness/Office dependencies with private uv and managed Python. Fetch a requested Git revision, or install an explicitly provided local checkout. Build a fresh environment before switching the launcher; a failed update preserves the previous installation. Record the installed source revision. Do not edit shell profiles or require sudo. Document PATH setup, private-repository use, system tools, updates, and removal.

Out of scope: PyPI publication, public hosting/repository visibility changes, interactive CLI features, installing global OS packages, changing training or Harbor execution environments.
