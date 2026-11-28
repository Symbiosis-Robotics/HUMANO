#!/usr/bin/env python3
"""Rewrite public-dataset resource paths for the project-local resource root."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from humano.utils.paths import RESOURCE_DIRS, canonical_resource_path  # noqa: E402

_ROOT_PATTERN = "|".join(sorted(RESOURCE_DIRS))
_RESOURCE_PATH = re.compile(
    rf"(?:/home/[^/]+(?:/bsh)?/HUMANO(?:-release|[-_]work)?/data/|/home/[^/]+/bsh/data/HUMANO/public/|(?<![\w/])data/)"
    rf'(?:resources/)?(?P<relative>(?:{_ROOT_PATTERN})/[^"\\\s]+)'
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--backup-root", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    return parser.parse_args()


def rewrite(text: str) -> str:
    return _RESOURCE_PATH.sub(
        lambda m: f"data/resources/{canonical_resource_path(Path(m['relative']))}", text
    )


def validate_targets(project_root: Path, text: str) -> None:
    for match in re.finditer(rf"data/resources/({_ROOT_PATTERN})/[^\"\\ ]+", text):
        target = project_root / match.group(0)
        if not target.exists():
            raise FileNotFoundError(f"rewritten metadata refers to a missing resource: {target}")


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().absolute()
    project_root = Path(__file__).resolve().parents[2]
    files = sorted(
        Path(root) / name
        for root, _, names in os.walk(dataset_root, followlinks=True)
        for name in names
        if name.endswith((".json", ".jsonl"))
    )
    changed: list[tuple[Path, str]] = []

    for path in files:
        text = path.read_text()
        updated = rewrite(text)
        if updated != text:
            validate_targets(project_root, updated)
            changed.append((path, updated))

    print(f"{len(changed)} of {len(files)} metadata files need resource-path updates")
    if not args.apply:
        return

    for path, updated in changed:
        backup = args.backup_root / path.relative_to(dataset_root)
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, backup)
        path.write_text(updated)

    print(f"updated {len(changed)} files; backups are in {args.backup_root}")


if __name__ == "__main__":
    main()
