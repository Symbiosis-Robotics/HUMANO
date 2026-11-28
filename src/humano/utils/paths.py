from __future__ import annotations

import importlib.resources as res
import os
import re
from importlib.resources import as_file
from pathlib import Path

RESOURCE_DIRS = frozenset(
    {
        "objects",
        "materials",
        "scenes",
        "robots",
        "dependencies",
        "assets",
        "assets_simplified_x10",
        "source_assets",
        "vMaterials_2",
    }
)
_RIGID_COLLECTIONS = frozenset(
    {
        "models",
        "collision_models_mujoco",
        "ruled_models",
        "stable",
    }
)


def canonical_resource_path(relative: Path) -> Path:
    """Translate paths recorded before resources were grouped by category."""
    parts = relative.parts
    if parts[:2] in {("assets", "objects"), ("assets_simplified_x10", "objects")}:
        tail = parts[2:]
        if len(tail) >= 2 and tail[0] not in _RIGID_COLLECTIONS:
            variant = "original" if parts[0] == "assets" else "simplified_x10"
            tail = (*tail[:2], variant, *tail[2:])
        return Path("objects", "rigid", *tail)
    if parts[:2] in {("assets", "articulated"), ("assets", "graspnet")}:
        return Path("objects", *parts[1:])
    if parts[:3] == ("scenes", "objects", "thor"):
        return Path("objects", "thor", *parts[3:])
    if parts[:1] == ("source_assets",):
        return Path("objects", "source", *parts[1:])
    if parts[:1] == ("vMaterials_2",):
        return Path("materials", *parts)
    return relative


def resolve_res_path(rel_path=None) -> str:
    res_dir = get_res_dir()
    if not rel_path:
        return res_dir
    with as_file(res_dir / rel_path) as res_path:
        if not res_path.exists():
            raise FileNotFoundError(res_path)
    return str(res_path)


def get_res_dir():
    return res.files("humano") / "resources"


def get_data_dir():
    return Path(str(res.files("humano").parent.parent)).resolve() / "data"


def get_resources_dir() -> Path:
    """Return the project-local digital-resource root."""
    return get_data_dir() / "resources"


def _normalise_data_path(rel_path: str) -> tuple[Path, bool]:
    path = Path(rel_path)
    if path.is_absolute():
        project_root = get_data_dir().parent
        try:
            path = path.relative_to(project_root)
        except ValueError:
            # Episodes can retain the collecting host's absolute asset paths.
            match = re.search(
                r"/HUMANO(?:-release|[-_]work)?/(?:data(?:/resources)?|public/resources)/(.+)$",
                path.as_posix(),
            )
            if match and not path.exists():
                relative = Path(match[1])
                if relative.parts[0] in RESOURCE_DIRS:
                    local = get_resources_dir() / canonical_resource_path(relative)
                    if local.exists():
                        return local, True
            return path, False

    parts = list(path.parts)
    if parts[:1] == ["data"]:
        parts = parts[1:]
    if parts[:1] == ["resources"]:
        parts = parts[1:]

    use_resources = bool(parts) and parts[0] in RESOURCE_DIRS
    if use_resources:
        return get_resources_dir() / canonical_resource_path(Path(*parts)), True
    return get_data_dir().joinpath(*parts), False


def _parse_zip_file_from_rel_path(rel_path: str) -> str:
    rel_path = rel_path.replace("\\", "/").strip("/")

    for prefix in ("data/resources/", "resources/", "data/"):
        if rel_path.startswith(prefix):
            rel_path = rel_path.removeprefix(prefix)
            break
    if rel_path.startswith("materials/"):
        rel_path = rel_path.removeprefix("materials/")
    if rel_path.startswith("objects/rigid/"):
        return "assets_objects.zip"
    if rel_path.startswith(("objects/graspnet/", "objects/articulated/")):
        rel_path = "assets/" + rel_path.removeprefix("objects/")

    pattern = r"^vMaterials_2/(?:[^/]+/)*[^/]+\.mdl$"
    match = re.match(pattern, rel_path)
    if match:
        return "vMaterials_2.zip"

    pattern = r"^robots/([^/]+)/.+\.(?:yml|xml)$"
    match = re.match(pattern, rel_path)
    if match:
        robot_name = match.group(1)
        return f"robots_{robot_name}.zip"

    pattern = r"^scenes/([^/]+)/([^/]+)$"
    match = re.match(pattern, rel_path)
    if match:
        scene_category = match.group(1)
        scene_name = match.group(2)
        return f"scenes_{scene_category}_{scene_name}.zip"

    pattern = r"^assets/graspnet/(.*)$"
    match = re.match(pattern, rel_path)
    if match:
        return "assets_graspnet.zip"

    pattern = r"^assets/([^/]+)/([^/]+)/.*\.(xml|usd)$"
    match = re.search(pattern, rel_path)
    if match:
        asset_category = match.group(1)
        return f"assets_{asset_category}.zip"

    pattern = r"^assets/([^/]+)/([^/]+)/([^/]+)/(.+)$"
    match = re.match(pattern, rel_path)
    if match:
        asset_category = match.group(1)
        dex_category = match.group(2)
        hand_uid = match.group(3)
        asset_label = match.group(4)
        return f"assets_{asset_category}_{dex_category}_{hand_uid}_{asset_label}.zip"

    pattern = r"^assets/([^/]+)/([^/]+)$"
    match = re.match(pattern, rel_path)
    if match:
        asset_category = match.group(1)
        asset_name = match.group(2)
        return f"assets_{asset_category}_{asset_name}.zip"

    pattern = r"^assets/([^/]+)/(.+)$"
    match = re.match(pattern, rel_path)
    if match:
        asset_category = match.group(1)
        asset_name = match.group(2).replace("/", "_")
        return f"assets_{asset_category}.zip"

    raise FileNotFoundError(f"{rel_path}")


def resolve_data_path(rel_path=None, create_if_not_exist=False, auto_download=False) -> str:
    if not rel_path:
        return str(get_data_dir())

    res_path, _ = _normalise_data_path(os.fspath(rel_path))
    if not res_path.exists():
        if not auto_download:
            raise FileNotFoundError(res_path)

        if create_if_not_exist:
            os.makedirs(res_path, exist_ok=True)
        else:
            zip_file = _parse_zip_file_from_rel_path(os.fspath(rel_path))
            raise FileNotFoundError(
                f"{res_path} is missing. Download the required HUMANO assets "
                f"and place {zip_file} under {get_resources_dir()}."
            )

    return str(res_path)
