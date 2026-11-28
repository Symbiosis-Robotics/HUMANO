#!/usr/bin/env python3
"""Copy source textures required by object USDs into their relative locations."""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

TEXTURE_SUFFIXES = (".png", ".jpg", ".jpeg", ".tga", ".exr")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resources", type=Path, default=Path("data/resources"))
    parser.add_argument(
        "--source", type=Path, required=True, help="Psi0 extracted objects/thor directory"
    )
    return parser.parse_args()


def _missing_refs(resources: Path) -> list[tuple[Path, str]]:
    missing = []
    for usd in resources.joinpath("objects", "rigid").rglob("*.usd"):
        text = usd.read_text(errors="ignore")
        for ref in re.findall(r"@([^@]+)@", text):
            if not ref.lower().endswith(TEXTURE_SUFFIXES):
                continue
            target = (usd.parent / ref).resolve()
            if not target.is_file():
                missing.append((usd, ref))
    return missing


def main() -> None:
    args = _parse_args()
    resources = args.resources.expanduser().resolve()
    source = args.source.expanduser().resolve()
    missing = _missing_refs(resources)
    copied = 0
    for usd, ref in missing:
        category, model_id, *_ = usd.relative_to(resources / "objects/rigid").parts
        category = category.capitalize()
        object_id = model_id.rsplit("_", 1)[-1]
        source_texture = source / f"{category}_{object_id}_mesh" / "Textures" / Path(ref).name
        if not source_texture.is_file() or source_texture.is_symlink():
            raise FileNotFoundError(f"no regular source texture for {usd}: {source_texture}")
        target = (usd.parent / ref).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_texture, target)
        copied += 1

    remaining = _missing_refs(resources)
    if remaining:
        raise RuntimeError(f"{len(remaining)} USD texture references remain unresolved")
    print(f"copied {copied} missing USD textures into {resources}")


if __name__ == "__main__":
    main()
