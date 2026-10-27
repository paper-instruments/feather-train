"""Pinned verifier packages and installed-source checks for trusted grader images."""

GRADER_SETUP_CONFIG_ERROR = 78

GRADER_PACKAGE_VERSIONS = {
    "braintrust": "0.31.0",
    "rubric": "2.2.1",
    "openai": "2.45.0",
    "httpx": "0.28.1",
    "pydantic": "2.13.4",
    "tenacity": "9.1.4",
}

# A prepared security backport preserves the released criterion_number schema.
# This is an immutable source dependency, not a claim of a published PyPI release.
GRADER_SOURCE_URLS = {
    "rubric": "https://github.com/paper-instruments/rubric/archive/"
    "9dd4a93b8fddeeeac01d5da468b3ed6de1b76013.tar.gz",
}
GRADER_PIP_DEPS = tuple(
    f"{name} @ {GRADER_SOURCE_URLS[name]}" if name in GRADER_SOURCE_URLS else f"{name}=={version}"
    for name, version in GRADER_PACKAGE_VERSIONS.items()
)


def grader_dependency_check_script() -> str:
    """Return a stdlib-only check for the separately installed verifier Python."""
    return (
        "import importlib.metadata as m, json, sys\n"
        f"expected={GRADER_PACKAGE_VERSIONS!r}\n"
        f"sources={GRADER_SOURCE_URLS!r}\n"
        "missing=[]\n"
        "for name, version in expected.items():\n"
        "    try: dist=m.distribution(name)\n"
        "    except m.PackageNotFoundError:\n"
        "        missing.append(name); continue\n"
        "    if dist.version != version:\n"
        "        missing.append(name); continue\n"
        "    if name in sources:\n"
        "        try: source=json.loads(dist.read_text('direct_url.json') or '{}')\n"
        "        except (ValueError, TypeError, OSError):\n"
        "            missing.append(name); continue\n"
        "        if not isinstance(source, dict) or source.get('url') != sources[name]:\n"
        "            missing.append(name)\n"
        "if missing: print(f'missing_or_mismatch={missing}', file=sys.stderr)\n"
        f"raise SystemExit({GRADER_SETUP_CONFIG_ERROR} if missing else 0)\n"
    )
