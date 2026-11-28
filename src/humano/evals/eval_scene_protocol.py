"""Build and validate portable eval-scene metadata for merged FloorPlan splits."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from humano.scenes.floorplan_split import (
    GLOBAL_FLOORPLAN_SPLIT_SCHEMA,
    MERGED_GLOBAL_FLOORPLAN_SPLIT_SCHEMA,
    floor_plan_id,
    load_global_floorplan_split,
    split_floor_plan_sets,
)

EVAL_SCENE_SELECTION_SCHEMA = "humano.eval-scene-selection.v3"


def _resolve_repository_path(repository_root: Path, value: str) -> Path:
    relative = Path(value)
    if relative.is_absolute():
        raise ValueError("Eval scene v3 paths must be repository-relative")
    return (repository_root / relative).resolve()


def _validate_eval_cache_paths(selection: dict[str, Any], *, repository_root: Path) -> None:
    def require_file(path: Path, kind: str) -> None:
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing required eval scene {kind} cache: {path}. "
                "Generate the corresponding test scene cache before evaluation."
            )

    asset_hiding = selection.get("asset_hiding")
    asset_hiding = asset_hiding if isinstance(asset_hiding, dict) else {}
    bev_roots = []
    bev_cache = selection.get("bev_cache")
    if isinstance(bev_cache, dict) and bev_cache.get("root"):
        bev_roots.append(str(bev_cache["root"]))
    if asset_hiding.get("cache_root"):
        bev_roots.append(str(asset_hiding["cache_root"]))
    for value in dict.fromkeys(bev_roots):
        cache_root = _resolve_repository_path(repository_root, value)
        manifest_path = cache_root / "manifest.json"
        require_file(manifest_path, "BEV manifest")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key in ("scene_manifest", "trajectory_manifest", "scene_support_manifest"):
            manifest_file = manifest.get(key)
            if manifest_file:
                require_file(cache_root / str(manifest_file), f"BEV {key}")

    if not asset_hiding.get("binding_root"):
        return
    binding_root = _resolve_repository_path(repository_root, str(asset_hiding["binding_root"]))
    for filename, kind in (
        ("manifest.json", "binding manifest"),
        ("placements.json", "binding placements"),
        ("episodes.jsonl", "binding episodes"),
    ):
        require_file(binding_root / filename, kind)


def _distribution_floor_plans(
    split: dict[str, Any], view: str | None, distribution: str
) -> frozenset[str]:
    train, eval_floor_plans = split_floor_plan_sets(split, view=view)
    if distribution == "id":
        return train
    if distribution == "scene_ood":
        return eval_floor_plans
    raise ValueError(f"Unsupported eval scene distribution: {distribution!r}")


def build_eval_scene_selection(
    candidate: dict[str, Any],
    *,
    split: dict[str, Any],
    split_manifest_path: Path,
    view: str | None,
    distribution: str,
    candidate_manifest_hash: str,
    source_selection: dict[str, Any] | None = None,
    layout_group_key: str | None = None,
) -> dict[str, Any]:
    if split.get("schema") not in (
        GLOBAL_FLOORPLAN_SPLIT_SCHEMA,
        MERGED_GLOBAL_FLOORPLAN_SPLIT_SCHEMA,
    ):
        raise ValueError("Eval scene v3 requires a supported FloorPlan split")
    if split_manifest_path.is_absolute():
        raise ValueError("Eval scene v3 split manifest path must be repository-relative")
    scene_uid = str(candidate["scene_uid"])
    floor_plan = floor_plan_id(scene_uid)
    allowed = _distribution_floor_plans(split, view, distribution)
    if floor_plan not in allowed:
        raise ValueError(
            f"Eval scene FloorPlan {floor_plan} does not belong to {view} {distribution}"
        )

    source = source_selection or {}
    provenance = {
        "candidate_manifest_hash": candidate_manifest_hash,
        **({"source_schema": str(source["schema"])} if source.get("schema") is not None else {}),
        **(
            {"source_scene_index_sha256": str(source["scene_index_hash"])}
            if source.get("scene_index_hash") is not None
            else {}
        ),
        **(
            {"source_floorplan_split_content_hash": str(source["global_floorplan_split_hash"])}
            if source.get("global_floorplan_split_hash") is not None
            else {}
        ),
    }
    return {
        "schema": EVAL_SCENE_SELECTION_SCHEMA,
        "scene": {
            "uid": scene_uid,
            "revision": str(candidate["scene_revision"]),
        },
        "split": {
            "manifest_path": split_manifest_path.as_posix(),
            "manifest_content_hash": str(split["content_hash"]),
            "view": view,
            "distribution": distribution,
            "floor_plan": floor_plan,
        },
        "placement": dict(candidate["placement"]),
        "hidden_prim_paths": list(candidate.get("hidden_prim_paths", [])),
        "immutable_obstacle_collision_free": candidate.get("immutable_obstacle_collision_free"),
        "immutable_obstacle_count": candidate.get("immutable_obstacle_count"),
        "provenance": provenance,
        **({"layout_group_key": layout_group_key} if layout_group_key is not None else {}),
    }


def resolve_eval_scene_selection(
    selection: dict[str, Any], *, repository_root: Path
) -> dict[str, Any]:
    if selection.get("schema") != EVAL_SCENE_SELECTION_SCHEMA:
        raise ValueError(f"Unsupported eval scene schema: {selection.get('schema')!r}")
    scene = selection.get("scene")
    split_binding = selection.get("split")
    if not isinstance(scene, dict) or not isinstance(split_binding, dict):
        raise ValueError("Eval scene v3 requires scene and split mappings")

    split_path = _resolve_repository_path(
        repository_root,
        str(split_binding["manifest_path"]),
    )
    split = load_global_floorplan_split(split_path)
    if split.get("schema") not in (
        GLOBAL_FLOORPLAN_SPLIT_SCHEMA,
        MERGED_GLOBAL_FLOORPLAN_SPLIT_SCHEMA,
    ):
        raise ValueError("Eval scene v3 split manifest is not a supported schema")
    if split["content_hash"] != split_binding.get("manifest_content_hash"):
        raise ValueError("Eval scene split manifest hash does not match")

    scene_uid = str(scene["uid"])
    floor_plan = floor_plan_id(scene_uid)
    if floor_plan != split_binding.get("floor_plan"):
        raise ValueError("Eval scene FloorPlan does not match its scene UID")
    view = split_binding["view"]
    distribution = str(split_binding["distribution"])
    if floor_plan not in _distribution_floor_plans(split, view, distribution):
        raise ValueError(
            f"Eval scene FloorPlan {floor_plan} does not belong to {view} {distribution}"
        )

    scene_index_spec = split["scene_index"]
    scene_index_path = (split_path.parent / scene_index_spec["path"]).resolve()
    actual_scene_index_hash = hashlib.sha256(scene_index_path.read_bytes()).hexdigest()
    if actual_scene_index_hash != scene_index_spec["sha256"]:
        raise ValueError(f"Eval scene index content changed: {scene_index_path}")
    scene_index = json.loads(scene_index_path.read_text(encoding="utf-8"))
    if scene_index.get("schema") != scene_index_spec["schema"]:
        raise ValueError("Eval scene index schema does not match the merged split")
    indexed_scene = next(
        (value for value in scene_index.get("scenes", ()) if value.get("uid") == scene_uid),
        None,
    )
    if indexed_scene is None:
        raise ValueError(f"Eval scene UID is missing from the scene index: {scene_uid}")
    scene_revision = str(scene["revision"])
    if str(indexed_scene.get("revision")) != scene_revision:
        raise ValueError(
            f"Scene revision changed for {scene_uid}: expected {scene_revision}, "
            f"found {indexed_scene.get('revision')}"
        )
    if not isinstance(selection.get("provenance"), dict):
        raise ValueError("Eval scene v3 provenance must be a mapping")
    _validate_eval_cache_paths(selection, repository_root=repository_root)
    return {
        "scene_index": str(scene_index_path),
        "scene_index_hash": actual_scene_index_hash,
        "scene_uid": scene_uid,
        "scene_revision": scene_revision,
        "placement": dict(selection["placement"]),
        "hidden_prim_paths": list(selection.get("hidden_prim_paths", [])),
        "immutable_obstacle_collision_free": selection.get("immutable_obstacle_collision_free"),
        "immutable_obstacle_count": selection.get("immutable_obstacle_count"),
    }
