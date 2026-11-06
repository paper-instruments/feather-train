"""Materialize bundled skills, replacing harness-owned copies when requested."""

import argparse
import shutil
from pathlib import Path

BUNDLE_ROOT = Path(__file__).resolve().parents[1]
DESIGN_GUIDE = "design.md"
ICONS = "icons"
WORKSPACE_SKILLS_PATH = ".agents"


def install_skills(workspace: Path, *, overwrite: bool = False) -> None:
    """Install bundled references; reject local conflicts unless replacement is requested."""
    source_root = BUNDLE_ROOT / "skills"
    files = [
        (workspace / ".agents" / source.relative_to(source_root), source.read_bytes())
        for source in sorted(source_root.rglob("*"))
        # Wheel installation may compile skill scripts; caches are not skill resources.
        if source.is_file() and "__pycache__" not in source.parts
    ]
    _write_files(files, workspace, overwrite=overwrite)


def install_production_skills(workspace: Path) -> None:
    """Append production guidance to the bundled format skills and add the shared design assets.

    Each skill keeps the name, description, and technical body the model was trained with, so
    the catalog is unchanged and the guidance is read with the manual. Run after
    `install_skills`; the harness-owned files are always replaced.
    """
    files = _production_files(workspace / ".agents", WORKSPACE_SKILLS_PATH)
    _write_files(files, workspace, overwrite=True)


def install_production_skill_tree(root: Path) -> None:
    """Install the bundled manuals with production guidance into a host-owned directory.

    This serves hosts that keep skills outside the task workspace. The guidance names skill
    files by path, so its references are rewritten to `root`, and the host builds its prompt
    with the same path as `skills_path`. The directory is harness-owned and replaced in full.
    """
    skills_path = root.as_posix()
    if not root.is_absolute():
        raise ValueError(f"Production skill tree root must be absolute: {root}")
    source_root = BUNDLE_ROOT / "skills"
    files = {
        root / source.relative_to(source_root): source.read_bytes()
        for source in sorted(source_root.rglob("*"))
        if source.is_file() and "__pycache__" not in source.parts
    }
    files.update(_production_files(root, skills_path))
    _write_files(list(files.items()), root, overwrite=True)


def render_skills_path(text: str, skills_path: str) -> str:
    """Point the production guidance's `.agents/` references at the host's skill directory."""
    if not skills_path:
        raise ValueError("Skills path must be non-empty")
    return text.replace(f"{WORKSPACE_SKILLS_PATH}/", f"{skills_path.rstrip('/')}/")


def install_task_skills(workspace: Path, skills_dir: Path) -> None:
    source_root = skills_dir.resolve()
    if not source_root.is_dir():
        raise ValueError(f"Task skills_dir is not a directory: {source_root}")
    dest_root = workspace / ".agents"
    pending = []
    for source in sorted(source_root.iterdir()):
        if source.name == "__pycache__":
            continue
        if source.is_symlink():
            raise FileExistsError(f"Task skill path is a symlink: {source}")
        destination = dest_root / source.name
        if destination.exists():
            raise FileExistsError(f"Task skill conflicts with installed content: {destination}")
        pending.append((source, destination))
    dest_root.mkdir(parents=True, exist_ok=True)
    for source, destination in pending:
        if source.is_dir():
            shutil.copytree(source, destination, symlinks=False)
        elif source.is_file():
            shutil.copy2(source, destination)
        else:
            raise ValueError(f"Task skill is not a file or directory: {source}")


def _production_files(root: Path, skills_path: str) -> list[tuple[Path, bytes]]:
    guidance_root = BUNDLE_ROOT / "production_skills"

    def guidance(path: Path) -> bytes:
        return render_skills_path(path.read_text(encoding="utf-8"), skills_path).encode("utf-8")

    files = [(root / DESIGN_GUIDE, guidance(guidance_root / DESIGN_GUIDE))]
    for guide in sorted(guidance_root.glob("*.md")):
        if guide.name == DESIGN_GUIDE:
            continue
        manual = BUNDLE_ROOT / "skills" / guide.stem / "SKILL.md"
        if not manual.is_file():
            raise FileNotFoundError(f"Production guidance has no matching bundled skill: {guide}")
        content = manual.read_bytes().rstrip(b"\n") + b"\n\n" + guidance(guide)
        files.append((root / guide.stem / "SKILL.md", content))
    icons = guidance_root / ICONS
    files += [
        (root / ICONS / source.relative_to(icons), source.read_bytes())
        for source in sorted(icons.rglob("*"))
        if source.is_file() and "__pycache__" not in source.parts
    ]
    return files


def _write_files(files: list[tuple[Path, bytes]], workspace: Path, *, overwrite: bool) -> None:
    pending = []
    for destination, content in files:
        for path in (destination, *destination.parents):
            if path == workspace:
                break
            if path.is_symlink():
                raise FileExistsError(f"Skill installation path is a symlink: {path}")
            if path != destination and path.exists() and not path.is_dir():
                raise FileExistsError(f"Skill installation parent is not a directory: {path}")
        if destination.exists():
            if not destination.is_file():
                raise FileExistsError(
                    f"Skill installation path is not a regular file: {destination}"
                )
            if destination.read_bytes() == content:
                continue
            if not overwrite:
                raise FileExistsError(f"Bundled skill conflicts with local content: {destination}")
            # Replacing in place would write through to a linked file outside the workspace.
            if destination.stat().st_nlink > 1:
                raise FileExistsError(f"Skill installation path is a hard link: {destination}")
        pending.append((destination, content))
    # Check every destination before writing, so a known conflict leaves the workspace untouched.
    for destination, content in pending:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Install bundled Feather skills.")
    parser.add_argument("workspace", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--profile", choices=("train", "prod"), default="train")
    args = parser.parse_args()
    install_skills(args.workspace, overwrite=args.overwrite)
    if args.profile == "prod":
        install_production_skills(args.workspace)
