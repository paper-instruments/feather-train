# Installer, verification and documentation

Files: `install.sh`, `scripts/check_install.py`, `.github/workflows/feather.yml`, `README.md`, `feather-harness/README.md`, this plan.

Build a fresh managed environment; make the launcher select its Python regardless of the caller's active environment. Keep cloud integrations unchanged. Verify with `bash -n install.sh`, `shellcheck install.sh`, `python scripts/check_install.py`, Harness pytest/Ruff and wheel builds. The integration check must install real dependencies outside the checkout and execute an actual shell tool selected by a fake model, then verify Paper imports and caller-environment isolation. Document prerequisite OS tools separately; do not install system packages.
