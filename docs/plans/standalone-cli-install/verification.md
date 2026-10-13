# Verification

- `bash -n install.sh` and `shellcheck install.sh`: passed.
- `uv run --no-project --python 3.12 python scripts/check_install.py`: passed on macOS arm64.
- The same integration check passed in a Linux arm64 Python 3.12 container with the source checkout mounted read-only.
- The installer bootstrapped private uv and managed Python, installed the locked Harness/Office packages, and ran from an unrelated project with an active virtualenv and conflicting Python variables. That project and environment were unchanged.
- Local-checkout and Git-fetch source paths passed. Successful updates retained the prior version; a failed update preserved the selected environment. During an active fake-model task, an update switched `current`; both subsequent `python` and `python3` shell commands still used that run's original environment and generated valid Word, PowerPoint, and Excel files.
- Full Harness suite: **509 passed, 5 platform skips**, using `uv run --no-sync pytest -c feather-harness/pyproject.toml feather-harness/tests -q`.
- Workspace Ruff including the installer integration check: passed. `uv lock --check`: passed. Wheels and source distributions for all three packages built successfully.
- Independent code review approved the final installer after its environment-pinning correction. A manual simplification pass retained the lock, failure rollback, and concrete environment path because they protect real installation/update behavior.

CI now runs the isolated installation check in the distribution job. It needs network access for public dependencies and runtime downloads; model requests use a synthetic local endpoint, with no paid inference or customer data. OS binaries (ripgrep, Poppler, LibreOffice/OCR) remain separately documented prerequisites.

## Task Python override verification

- Full Harness suite: **514 passed, 5 platform skips** after the override changes.
- The five new CLI cases passed: JSON configuration, flag precedence with a relative symlink interpreter, and missing/directory/non-executable rejection before model requests. Both `python` and `python3` used the exact selected executable while `FEATHER_PYTHON` retained Feather's interpreter. The model's system prompt and user instruction were unchanged; the local Bash description exposed Paper routing.
- Real installed-package integration passed again on macOS arm64 and Linux arm64 with read-only source mounting. An isolated project-only dependency was available to task Python and unavailable to Feather Python; Paper was available through `FEATHER_PYTHON` and unavailable to project Python. Office files were generated successfully, and the unrelated project/environment was unchanged.
- Workspace Ruff, installer Bash/ShellCheck, lock consistency, all three wheel/sdist builds and diff checks passed. Independent review approved the routing and cleanup. Manual simplification retained the two temporary command aliases because merely prepending an executable's directory cannot guarantee both `python` and `python3` select that exact interpreter.
- No system-prompt or skill files changed; no packages are installed into the selected task environment.
