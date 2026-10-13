#!/usr/bin/env bash
# Install Feather in a private, per-user Python environment.
set -euo pipefail
umask 077

fail() { printf 'Feather install: %s\n' "$*" >&2; exit 1; }

if [[ $# -gt 0 ]]; then
    [[ $# -eq 1 && "$1" == "--help" ]] || fail 'Only --help is supported; configure installation with environment variables.'
    cat <<'HELP'
Install Feather and its Paper/Office packages without changing the caller's Python environment.

Variables:
  FEATHER_INSTALL_DIR  Installation root (default: ~/.local/share/feather; absolute path)
  FEATHER_BIN_DIR      Launcher directory (default: ~/.local/bin; absolute path)
  FEATHER_REF          Git branch, tag or commit (default: main)
  FEATHER_REPO         Git URL (default: https://github.com/paper-instruments/feather-train.git)
  FEATHER_SOURCE_DIR   Install from an existing checkout instead (absolute path)

Requires macOS/Linux, Bash, curl and git. Install ripgrep and PDF/Office system tools separately.
HELP
    exit 0
fi
case "$(uname -s)" in Darwin|Linux) ;; *) fail 'This installer supports macOS and Linux.' ;; esac
for tool in curl git; do command -v "$tool" >/dev/null || fail "Install $tool first."; done

install_root=${FEATHER_INSTALL_DIR:-"$HOME/.local/share/feather"}
bin_dir=${FEATHER_BIN_DIR:-"$HOME/.local/bin"}
for directory in "$install_root" "$bin_dir"; do
    [[ "$directory" == /* && "$directory" != / ]] || fail 'Installation directories must be absolute paths other than /.'
done
launcher="$bin_dir/feather"
if [[ -e "$launcher" || -L "$launcher" ]]; then
    if [[ ! -f "$launcher" ]] || ! grep -Fqx '# Feather managed launcher' "$launcher"; then
        fail "Refusing to replace an unrelated $launcher."
    fi
fi
[[ ! -e "$install_root/current" || -L "$install_root/current" ]] || fail "$install_root/current must be a symlink."
mkdir -p "$install_root/versions" "$bin_dir"
mkdir "$install_root/.install-lock" 2>/dev/null || fail "Another install is running; if it was interrupted, remove $install_root/.install-lock."
version_dir=
launcher_temporary="$bin_dir/.feather.new.$$"
completed=false
cleanup() {
    if [[ "$completed" != true && -n "$version_dir" && ! "$install_root/current" -ef "$version_dir" ]]; then
        rm -rf -- "$version_dir"
    fi
    rm -f -- "$install_root/uv-installer.sh" "$install_root/current.new" "$launcher_temporary"
    rmdir "$install_root/.install-lock"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# Ignore an activated environment and Python module injection from the caller.
unset VIRTUAL_ENV PYTHONHOME PYTHONPATH UV_CONFIG_FILE
export PYTHONNOUSERSITE=1
export UV_PYTHON_INSTALL_DIR="$install_root/python"
export UV_PYTHON_BIN_DIR="$install_root/tools"
export UV_CACHE_DIR="$install_root/cache"
export UV_PYTHON_DOWNLOADS=automatic
uv_bin="$install_root/tools/uv"
if [[ ! -x "$uv_bin" ]]; then
    curl -fsSL https://astral.sh/uv/0.11.14/install.sh -o "$install_root/uv-installer.sh"
    UV_UNMANAGED_INSTALL="$install_root/tools" sh "$install_root/uv-installer.sh"
fi
version_dir=$(mktemp -d "$install_root/versions/install.XXXXXX")
source_dir=${FEATHER_SOURCE_DIR:-}
if [[ -n "$source_dir" ]]; then
    [[ "$source_dir" == /* ]] || fail 'FEATHER_SOURCE_DIR must be absolute.'
else
    source_dir="$version_dir/source"
    git init -q "$source_dir"
    git -C "$source_dir" remote add origin "${FEATHER_REPO:-https://github.com/paper-instruments/feather-train.git}"
    GIT_TERMINAL_PROMPT=0 git -C "$source_dir" fetch --depth=1 origin "${FEATHER_REF:-main}"
    git -C "$source_dir" checkout -q --detach FETCH_HEAD
fi
[[ -f "$source_dir/uv.lock" && -f "$source_dir/feather-harness/pyproject.toml" ]] || fail 'Source must be a Feather workspace checkout with uv.lock.'
export UV_PROJECT_ENVIRONMENT="$version_dir/venv"
"$uv_bin" sync --project "$source_dir" --no-config --locked --no-editable \
    --no-dev --no-default-groups --package feather-harness --extra office \
    --python 3.12 --managed-python
"$version_dir/venv/bin/python" -I - <<'PY'
from importlib.metadata import version
import docx, pptx, openpyxl
for name in ("feather-harness", "paper-docx", "paper-pptx", "paper-xlsx"):
    print(f"Installed {name} {version(name)}")
PY
"$version_dir/venv/bin/python" -I -m feather_harness --help >/dev/null
revision=$(git -C "$source_dir" rev-parse HEAD)
printf '%s\n' "$revision" > "$version_dir/source-revision"

{
    printf '#!/usr/bin/env bash\n# Feather managed launcher\nset -euo pipefail\n'
    # Keep command substitution in the generated launcher, so each run pins its version.
    # shellcheck disable=SC2016
    printf 'VIRTUAL_ENV=$(cd -P -- %q && pwd -P)\nexport VIRTUAL_ENV\n' "$install_root/current/venv"
    cat <<'LAUNCHER'
export PATH="$VIRTUAL_ENV/bin:$PATH"
unset PYTHONHOME PYTHONPATH BASH_ENV ENV
export PYTHONNOUSERSITE=1
exec "$VIRTUAL_ENV/bin/python" -I -m feather_harness "$@"
LAUNCHER
} > "$launcher_temporary"
chmod 755 "$launcher_temporary"
"$version_dir/venv/bin/python" -I - "$install_root" "$version_dir" "$launcher_temporary" "$launcher" <<'PY'
import os
import sys
from pathlib import Path
root, installed, temporary_launcher, launcher = map(Path, sys.argv[1:])
os.symlink(installed, root / "current.new")
previous = os.readlink(root / "current") if (root / "current").is_symlink() else None
os.replace(root / "current.new", root / "current")
try:
    os.replace(temporary_launcher, launcher)
except OSError:
    if previous is None:
        (root / "current").unlink()
    else:
        os.symlink(previous, root / "current.new")
        os.replace(root / "current.new", root / "current")
    raise
PY
completed=true
printf '\nInstalled Feather at %s (source %s).\n' "$launcher" "$revision"
case ":$PATH:" in
    *":$bin_dir:"*) ;;
    *) printf 'Add the launcher to your PATH: export PATH=%q:%s\n' "$bin_dir" "\"\$PATH\"" ;;
esac
if ! command -v rg >/dev/null; then printf 'Install ripgrep (rg) before using file-search tools.\n'; fi
printf 'Try: feather --help\n'
