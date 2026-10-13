# Verification

- `bash -n install.sh` and `shellcheck install.sh`: passed.
- `uv run --no-project --python 3.12 python scripts/check_install.py`: passed on macOS arm64.
- The same integration check passed in a Linux arm64 Python 3.12 container with the source checkout mounted read-only.
- The installer bootstrapped private uv and managed Python, installed the locked Harness/Office packages, and ran from an unrelated project with an active virtualenv and conflicting Python variables. That project and environment were unchanged.
- Local-checkout and Git-fetch source paths passed. Successful updates retained the prior version; a failed update preserved the selected environment. During an active fake-model task, an update switched `current`; both subsequent `python` and `python3` shell commands still used that run's original environment and generated valid Word, PowerPoint, and Excel files.
- Full Harness suite: **509 passed, 5 platform skips**, using `uv run --no-sync pytest -c feather-harness/pyproject.toml feather-harness/tests -q`.
- Workspace Ruff including the installer integration check: passed. `uv lock --check`: passed. Wheels and source distributions for all three packages built successfully.
- Independent code review approved the final installer after its environment-pinning correction. A manual simplification pass retained the lock, failure rollback, and concrete environment path because they protect real installation/update behavior.

CI now runs the isolated installation check in the distribution job. It needs network access for public dependencies and runtime downloads; model requests use a synthetic local endpoint, with no paid inference or customer data. OS binaries (ripgrep, Poppler, LibreOffice/OCR) remain separately documented prerequisites. Public curl installation becomes available once the script is merged and the repository is publicly accessible; authenticated local checkout installation works now.
