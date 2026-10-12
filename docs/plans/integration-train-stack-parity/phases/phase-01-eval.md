# Eval integration

Goal: match the generic Eval behavior of the pinned Train source while keeping
the standalone installation and example usable.

Files: `feather-eval/**` only.

Work: compare source/destination and port Harbor settings, toolkit dependencies,
prompt/usage evidence, ATIF, corresponding tests and README/example changes.
Remove stale claims that web/profile settings are unavailable. Keep paths and
instructions standalone. Do not copy private benchmark adapters or presets.

Done when: `uv run --no-sync pytest -c feather-eval/pyproject.toml
feather-eval/tests -q` and scoped Ruff pass, and settings are proven to survive
construction and serialized CLI config.
