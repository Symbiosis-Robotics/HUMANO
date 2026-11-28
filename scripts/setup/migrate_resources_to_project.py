#!/usr/bin/env python3
"""Copy shared HUMANO digital resources into the project-local data root.

The source is read-only for this migration.  The destination contains regular
files, so a checkout can be moved without relying on machine-specific links.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from humano.utils.paths import RESOURCE_DIRS, canonical_resource_path  # noqa: E402
from humano.utils.resource_layers import relocate_resource_layer  # noqa: E402

_LEGACY_REQUIRED = {"assets", "robots", "scenes", "vMaterials_2", "assets_simplified_x10"}
_REQUIRED = {"objects", "robots", "scenes", "materials"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True, help="Existing shared resources root")
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path("data/resources"),
        help="Project-local destination (default: data/resources)",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Replace an existing destination directory after validation",
    )
    return parser.parse_args()


def _assert_no_symlinks(root: Path) -> None:
    symlinks = [path for path in root.rglob("*") if path.is_symlink()]
    if symlinks:
        sample = ", ".join(str(path) for path in symlinks[:5])
        raise RuntimeError(f"resource tree contains symlinks ({sample})")


def main() -> None:
    args = _parse_args()
    source = args.source.expanduser().resolve()
    destination = args.destination.expanduser()
    if not source.is_dir():
        raise FileNotFoundError(f"resource source does not exist: {source}")

    present = {name for name in RESOURCE_DIRS if (source / name).is_dir()}
    required = _REQUIRED if "objects" in present else _LEGACY_REQUIRED
    missing = sorted(required - present)
    if missing:
        raise FileNotFoundError(f"resource source is missing directories: {', '.join(missing)}")
    _assert_no_symlinks(source)

    if destination.exists():
        if not args.replace:
            raise FileExistsError(f"destination already exists: {destination}; use --replace")
        backup = destination.with_name(destination.name + ".backup")
        if backup.exists():
            raise FileExistsError(f"backup path already exists: {backup}")
        destination.rename(backup)

    destination.mkdir(parents=True, exist_ok=False)
    try:
        for name in sorted(present):
            root = source / name
            if name in {"objects", "materials", "robots", "dependencies"}:
                shutil.copytree(root, destination / name, dirs_exist_ok=True)
                continue
            if name == "scenes":
                (destination / name).mkdir(exist_ok=True)
            for path in root.rglob("*"):
                relative = path.relative_to(source)
                target = destination / canonical_resource_path(relative)
                if path.is_dir():
                    if relative.parts != ("scenes", "objects"):
                        target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, target)
                    relocate_resource_layer(target)
        _assert_no_symlinks(destination)
    except Exception:
        shutil.rmtree(destination, ignore_errors=True)
        backup = destination.with_name(destination.name + ".backup")
        if backup.exists():
            backup.rename(destination)
        raise

    print(f"copied {', '.join(sorted(present))} from {source} to {destination}")


if __name__ == "__main__":
    main()
