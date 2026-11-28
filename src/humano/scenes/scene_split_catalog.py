"""Validate the combined scene catalog used by Basic and Advanced data flows."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from humano.scenes.floorplan_split import load_global_floorplan_split, validate_split_compatibility

SCENE_SPLIT_CATALOG_SCHEMA = "humano.scene-split-catalog.v1"


def _resolve(catalog_path: Path, relative_path: str) -> Path:
    return (catalog_path.parent / relative_path).resolve()


def _floor_plan_id(scene_uid: str) -> str:
    match = re.search(r"FloorPlan\d+", scene_uid)
    if match is None:
        raise ValueError(f"Scene uid has no physical FloorPlan id: {scene_uid}")
    return match.group(0)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_scene_split_catalog(
    path: Path, *, active_split: Path | None = None
) -> dict[str, Any]:
    catalog_path = path.expanduser().resolve()
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    if catalog.get("schema") != SCENE_SPLIT_CATALOG_SCHEMA:
        raise ValueError(f"Unsupported scene split catalog schema: {catalog.get('schema')!r}")

    scene_spec = catalog["scene_index"]
    scene_index_path = _resolve(catalog_path, scene_spec["path"])
    scene_index = json.loads(scene_index_path.read_text(encoding="utf-8"))
    if scene_index.get("schema") != scene_spec["schema"]:
        raise ValueError("Scene split catalog scene index schema does not match")
    if _sha256(scene_index_path) != scene_spec["sha256"]:
        raise ValueError("Scene split catalog scene index hash does not match")
    scene_uids = frozenset(
        str(scene["uid"]) for scene in scene_index.get("scenes", []) if scene.get("uid")
    )
    scene_floor_plans = frozenset(_floor_plan_id(uid) for uid in scene_uids)
    if len(scene_uids) != scene_spec["scene_uid_count"]:
        raise ValueError("Scene split catalog scene UID count does not match")
    if len(scene_floor_plans) != scene_spec["floor_plan_count"]:
        raise ValueError("Scene split catalog FloorPlan count does not match")

    loaded_splits = {}
    for name, spec in catalog["splits"].items():
        split_path = _resolve(catalog_path, spec["path"])
        split = load_global_floorplan_split(split_path)
        if (
            split.get("schema") != spec["schema"]
            or split.get("content_hash") != spec["content_hash"]
        ):
            raise ValueError(f"Scene split catalog split metadata does not match: {name}")
        loaded_splits[name] = split

    reference_name = catalog["compatibility"]["reference"]
    candidate_name = catalog["compatibility"]["candidate"]
    validate_split_compatibility(loaded_splits[reference_name], loaded_splits[candidate_name])
    reference = loaded_splits[reference_name]
    candidate = loaded_splits[candidate_name]
    reference_selected = set(reference["train_floor_plans"]) | set(reference["eval_floor_plans"])
    candidate_selected = set(candidate["train_floor_plans"]) | set(candidate["eval_floor_plans"])
    merged = catalog["merged"]
    merged_selected = set(merged["floor_plans"])
    if merged_selected != reference_selected | candidate_selected:
        raise ValueError("Scene split catalog merged FloorPlan set does not match its splits")
    if merged_selected != scene_floor_plans:
        raise ValueError("Scene split catalog contains a scene outside the formal scene index")
    if set(merged["scene_uids"]) != scene_uids:
        raise ValueError("Scene split catalog scene UID set does not match the formal scene index")
    if active_split is not None:
        active_path = active_split.expanduser().resolve()
        allowed_paths = {
            _resolve(catalog_path, spec["path"]) for spec in catalog["splits"].values()
        }
        if active_path not in allowed_paths:
            raise ValueError(
                "Active split is not registered in scene split catalog: "
                f"{active_path}"
            )
    return catalog
