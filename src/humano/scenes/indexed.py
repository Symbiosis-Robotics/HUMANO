"""Lightweight indexed room scenes for offline Isaac replay."""

from __future__ import annotations

import copy
import json
import math
import os
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from humano.core.scene import TabletopScene


@dataclass
class IndexedSceneTransform:
    translation: list[float]
    rotation_wxyz: list[float]
    scale: list[float]

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> IndexedSceneTransform:
        return cls(
            translation=[float(v) for v in value.get("translation", [0.0, 0.0, 0.0])],
            rotation_wxyz=[float(v) for v in value.get("rotation_wxyz", [1.0, 0.0, 0.0, 0.0])],
            scale=[float(v) for v in value.get("scale", [1.0, 1.0, 1.0])],
        )


@dataclass
class IndexedSceneBounds:
    min_xy: list[float]
    max_xy: list[float]
    floor_z: float
    margin_m: float
    policy: str
    source: str
    validated: bool

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> IndexedSceneBounds:
        return cls(
            min_xy=[float(v) for v in value["min_xy"]],
            max_xy=[float(v) for v in value["max_xy"]],
            floor_z=float(value.get("floor_z", 0.0)),
            margin_m=float(value.get("margin_m", 0.05)),
            policy=str(value.get("policy", "strict")),
            source=str(value.get("source", "unknown")),
            validated=bool(value.get("validated", False)),
        )


@dataclass
class IndexedVisualItem:
    prim_path: str
    category: str
    min_xy: list[float]
    max_xy: list[float]
    min_z: float | None = None
    max_z: float | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> IndexedVisualItem:
        return cls(
            prim_path=str(value["prim_path"]),
            category=str(value["category"]),
            min_xy=[float(v) for v in value["min_xy"]],
            max_xy=[float(v) for v in value["max_xy"]],
            min_z=float(value["min_z"]) if value.get("min_z") is not None else None,
            max_z=float(value["max_z"]) if value.get("max_z") is not None else None,
        )


@dataclass(frozen=True)
class IndexedNavigationObstacle:
    prim_path: str
    category: str
    min_xyz: tuple[float, float, float]
    max_xyz: tuple[float, float, float]

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> IndexedNavigationObstacle:
        min_xyz = tuple(float(v) for v in value["min_xyz"])
        max_xyz = tuple(float(v) for v in value["max_xyz"])
        if len(min_xyz) != 3 or len(max_xyz) != 3:
            raise ValueError("Immutable obstacle bounds must contain three coordinates")
        return cls(
            prim_path=str(value["prim_path"]),
            category=str(value["category"]),
            min_xyz=min_xyz,
            max_xyz=max_xyz,
        )


@dataclass
class IndexedVisualProfile:
    profile: str = "structure"
    visible_prim_paths: list[str] = field(default_factory=list)
    hidden_prim_paths: list[str] = field(default_factory=list)
    background_items: list[IndexedVisualItem] = field(default_factory=list)
    included_prim_paths: list[str] = field(default_factory=list)
    excluded_prim_paths: list[str] = field(default_factory=list)
    hide_ceiling: bool = True
    disable_authored_physics: bool = True

    @classmethod
    def from_dict(cls, value: dict[str, Any] | None) -> IndexedVisualProfile:
        value = value or {}
        return cls(
            profile=str(value.get("profile", "structure")),
            visible_prim_paths=[str(v) for v in value.get("visible_prim_paths", [])],
            hidden_prim_paths=[str(v) for v in value.get("hidden_prim_paths", [])],
            background_items=[
                IndexedVisualItem.from_dict(item) for item in value.get("background_items", [])
            ],
            included_prim_paths=[str(v) for v in value.get("included_prim_paths", [])],
            excluded_prim_paths=[str(v) for v in value.get("excluded_prim_paths", [])],
            hide_ceiling=bool(value.get("hide_ceiling", True)),
            disable_authored_physics=bool(value.get("disable_authored_physics", True)),
        )


class IndexedBackgroundScene(TabletopScene):
    def __init__(
        self,
        *,
        uid: str,
        provider: str,
        index_path: Path,
        usd_path: Path,
        transform: IndexedSceneTransform,
        bounds: IndexedSceneBounds,
        visual_profile: IndexedVisualProfile,
        immutable_obstacles: tuple[IndexedNavigationObstacle, ...],
        revision: str,
        metadata: dict[str, Any],
        occupancy_path: Path | None = None,
    ) -> None:
        self.uid = uid
        self.provider = provider
        self.index_path = str(index_path)
        self.usd_path = str(usd_path)
        self.data_dir = str(usd_path.parent)
        self.name = usd_path.stem
        self.transform = transform
        self.bounds = bounds
        self.visual_profile = visual_profile
        self.immutable_obstacles = immutable_obstacles
        self.revision = revision
        self.metadata = metadata
        self.occupancy_path = occupancy_path
        self.table2 = None

    def make_contact_actors(self):
        from humano.core.actor import ArticulatedObjectActor
        from humano.core.asset import ArticulatedAsset
        from humano.utils import resolve_data_path

        actors = {}
        for key, item in self.metadata.get("contact_assets", {}).items():
            asset = ArticulatedAsset(
                item["name"],
                resolve_data_path(item["usd_path"], auto_download=False),
                resolve_data_path(item["mjcf_path"], auto_download=False),
            )
            asset.name = asset.label = item["name"]
            asset.articulate_init_joint_qpos = None
            if item.get("usd_body_mapping"):
                asset.usd_body_mapping = resolve_data_path(
                    item["usd_body_mapping"], auto_download=False
                )
            actors[key] = ArticulatedObjectActor(asset)
        return actors

    @property
    def cache_key(self) -> str:
        return f"{self.uid}@{self.revision}"

    def copy(self) -> IndexedBackgroundScene:
        return copy.deepcopy(self)


def load_configured_background_scene(config_path: str, entry: str) -> IndexedBackgroundScene:
    from humano.utils import resolve_data_path

    path = Path(config_path)
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[3] / path
    entries = json.loads(path.read_text())
    def resolve_entry(name):
        config = entries[name]
        if "base_entry" in config:
            config = {**resolve_entry(config["base_entry"]), **config}
        for actor, source_name in config.get("include_actors", {}).items():
            source = entries[source_name]
            config = {
                **config,
                "contact_assets": {**config["contact_assets"], actor: source["contact_assets"][actor]},
                "placements": {**config["placements"], actor: source["placements"][actor]},
            }
        return config

    config = resolve_entry(entry)
    return IndexedBackgroundScene(
        uid=config["uid"],
        provider="configured",
        index_path=path,
        usd_path=Path(resolve_data_path(config["usd_path"], auto_download=False)),
        transform=IndexedSceneTransform.from_dict(config.get("transform", {})),
        bounds=IndexedSceneBounds.from_dict(
            {
                **config["bounds"],
                "policy": "task",
                "source": "asset_bounds",
                "validated": False,
            }
        ),
        visual_profile=IndexedVisualProfile.from_dict(
            {"hide_ceiling": False, **config.get("visual_profile", {})}
        ),
        immutable_obstacles=(),
        revision="visual-v1",
        metadata=config,
    )


def load_indexed_background_scene(index_path: str | Path, scene_uid: str) -> IndexedBackgroundScene:
    resolved_index = Path(index_path).expanduser().absolute()
    data = json.loads(resolved_index.read_text(encoding="utf-8"))
    if data.get("schema") != "simple.scene-index.v2":
        raise ValueError(f"Unsupported scene index schema: {data.get('schema')}")

    entries = {str(entry["uid"]): entry for entry in data.get("scenes", [])}
    try:
        entry = entries[scene_uid]
    except KeyError as exc:
        raise KeyError(f"Scene {scene_uid!r} is not present in {resolved_index}") from exc
    if entry.get("metadata", {}).get("floor_alignment_status") == "mixed_levels":
        raise ValueError(f"Indexed scene has unresolved floor levels: {scene_uid}")

    usd_path = Path(entry["usd_path"])
    if not usd_path.is_absolute():
        usd_path = Path(os.path.abspath(resolved_index.parent / usd_path))
    usd_path = usd_path.resolve()
    if not usd_path.is_file():
        raise FileNotFoundError(usd_path)

    bounds = IndexedSceneBounds.from_dict(entry["bounds"])
    if not bounds.validated or bounds.policy != "strict":
        raise ValueError(f"Indexed scene must have validated strict bounds: {scene_uid}")
    navigation = entry.get("navigation") or {}
    if navigation.get("schema") != "simple.scene-navigation.v1" or not navigation.get("validated"):
        raise ValueError(f"Indexed scene has no validated navigation obstacles: {scene_uid}")
    return IndexedBackgroundScene(
        uid=scene_uid,
        provider=str(data.get("provider", "unknown")),
        index_path=resolved_index,
        usd_path=usd_path,
        transform=IndexedSceneTransform.from_dict(entry.get("transform", {})),
        bounds=bounds,
        visual_profile=IndexedVisualProfile.from_dict(entry.get("visual_profile")),
        immutable_obstacles=tuple(
            IndexedNavigationObstacle.from_dict(item)
            for item in navigation.get("immutable_obstacles", [])
        ),
        revision=str(entry.get("revision", "unknown")),
        metadata=dict(entry.get("metadata", {})),
        occupancy_path=None,
    )


def restore_indexed_hidden_assets(scene: IndexedBackgroundScene) -> list[str]:
    """Restore static-index assets so runtime collision checks decide visibility."""
    profile = scene.visual_profile
    restored = sorted(set(profile.hidden_prim_paths))
    restored_set = set(restored)
    profile.visible_prim_paths = sorted(set(profile.visible_prim_paths).union(restored_set))
    profile.hidden_prim_paths = []
    profile.included_prim_paths = sorted(set(profile.included_prim_paths).union(restored_set))
    profile.excluded_prim_paths = [
        path for path in profile.excluded_prim_paths if path not in restored_set
    ]
    scene.metadata["restored_index_hidden_prim_paths"] = restored
    return restored


def apply_indexed_asset_visibility(
    scene: IndexedBackgroundScene,
    *,
    show_all_indexed_assets: bool = False,
) -> list[str]:
    """Keep index-hidden assets hidden unless diagnostics explicitly restore them."""
    if show_all_indexed_assets:
        return restore_indexed_hidden_assets(scene)
    return []


def _transform_xy_aabb(
    min_xy: list[float],
    max_xy: list[float],
    *,
    translation_xy: tuple[float, float],
    yaw_rad: float,
) -> tuple[list[float], list[float]]:
    cosine = math.cos(yaw_rad)
    sine = math.sin(yaw_rad)
    corners = (
        (min_xy[0], min_xy[1]),
        (min_xy[0], max_xy[1]),
        (max_xy[0], min_xy[1]),
        (max_xy[0], max_xy[1]),
    )
    transformed = tuple(
        (
            cosine * x_value - sine * y_value + translation_xy[0],
            sine * x_value + cosine * y_value + translation_xy[1],
        )
        for x_value, y_value in corners
    )
    return (
        [min(point[0] for point in transformed), min(point[1] for point in transformed)],
        [max(point[0] for point in transformed), max(point[1] for point in transformed)],
    )


def _quaternion_product(left: list[float], right: list[float]) -> list[float]:
    left_w, left_x, left_y, left_z = left
    right_w, right_x, right_y, right_z = right
    return [
        left_w * right_w - left_x * right_x - left_y * right_y - left_z * right_z,
        left_w * right_x + left_x * right_w + left_y * right_z - left_z * right_y,
        left_w * right_y - left_x * right_z + left_y * right_w + left_z * right_x,
        left_w * right_z + left_x * right_y - left_y * right_x + left_z * right_w,
    ]


def align_indexed_background_scene(
    scene: IndexedBackgroundScene,
    *,
    translation_xy: tuple[float, float],
    yaw_rad: float,
    hidden_anchor_prim_path: str | None,
    hide_all_background_assets: bool = False,
    additional_hidden_prim_paths: tuple[str, ...] = (),
) -> IndexedBackgroundScene:
    """Return a copy with a world-space XY/yaw delta applied to only the background."""
    aligned = scene.copy()
    cosine = math.cos(yaw_rad)
    sine = math.sin(yaw_rad)
    base_translation = scene.transform.translation
    aligned.transform.translation = [
        cosine * base_translation[0] - sine * base_translation[1] + translation_xy[0],
        sine * base_translation[0] + cosine * base_translation[1] + translation_xy[1],
        base_translation[2],
    ]
    delta_quaternion = [math.cos(0.5 * yaw_rad), 0.0, 0.0, math.sin(0.5 * yaw_rad)]
    aligned.transform.rotation_wxyz = _quaternion_product(
        delta_quaternion, scene.transform.rotation_wxyz
    )

    aligned.bounds.min_xy, aligned.bounds.max_xy = _transform_xy_aabb(
        scene.bounds.min_xy,
        scene.bounds.max_xy,
        translation_xy=translation_xy,
        yaw_rad=yaw_rad,
    )
    for aligned_item, source_item in zip(
        aligned.visual_profile.background_items,
        scene.visual_profile.background_items,
        strict=True,
    ):
        aligned_item.min_xy, aligned_item.max_xy = _transform_xy_aabb(
            source_item.min_xy,
            source_item.max_xy,
            translation_xy=translation_xy,
            yaw_rad=yaw_rad,
        )

    transformed_obstacles = []
    for obstacle in scene.immutable_obstacles:
        min_xy, max_xy = _transform_xy_aabb(
            obstacle.min_xyz[:2],
            obstacle.max_xyz[:2],
            translation_xy=translation_xy,
            yaw_rad=yaw_rad,
        )
        transformed_obstacles.append(
            replace(
                obstacle,
                min_xyz=(*min_xy, obstacle.min_xyz[2]),
                max_xyz=(*max_xy, obstacle.max_xyz[2]),
            )
        )
    aligned.immutable_obstacles = tuple(transformed_obstacles)

    if (
        hidden_anchor_prim_path is not None
        or hide_all_background_assets
        or additional_hidden_prim_paths
    ):
        visible = set(aligned.visual_profile.visible_prim_paths)
        hidden = set(aligned.visual_profile.hidden_prim_paths)
        included = set(aligned.visual_profile.included_prim_paths)
        excluded = set(aligned.visual_profile.excluded_prim_paths)

        if hide_all_background_assets:
            asset_paths = {item.prim_path for item in scene.visual_profile.background_items}
            asset_families = {_indexed_prim_family(path) for path in asset_paths}
            linked_paths = asset_paths | {
                path for path in visible | included if _indexed_prim_family(path) in asset_families
            }
        elif hidden_anchor_prim_path is not None:
            linked_paths = expand_linked_furniture_paths(
                scene.visual_profile.background_items,
                {hidden_anchor_prim_path},
            )
        else:
            linked_paths = set()
        linked_paths.update(additional_hidden_prim_paths)
        linked_paths.update(detached_cabinet_fronts(scene.visual_profile.background_items, linked_paths))
        visible.difference_update(linked_paths)
        hidden.update(linked_paths)
        included.difference_update(linked_paths)
        excluded.update(linked_paths)
        aligned.visual_profile.visible_prim_paths = sorted(visible)
        aligned.visual_profile.hidden_prim_paths = sorted(hidden)
        aligned.visual_profile.included_prim_paths = sorted(included)
        aligned.visual_profile.excluded_prim_paths = sorted(excluded)

    aligned.metadata["task_alignment"] = {
        "mode": "xy_yaw",
        "translation_xy": list(translation_xy),
        "yaw_rad": float(yaw_rad),
        "yaw_deg": math.degrees(yaw_rad),
        "hidden_anchor_prim_path": hidden_anchor_prim_path,
        "hide_all_background_assets": hide_all_background_assets,
        "additional_hidden_prim_paths": list(additional_hidden_prim_paths),
        "preserves_task_layout": True,
        "scale_applied": False,
    }
    return aligned


def table_workspace_bounds(
    *,
    size: list[float],
    position: list[float],
    quaternion: list[float],
    clearance_m: float,
) -> tuple[list[float], list[float]]:
    quat = np.asarray(quaternion, dtype=np.float64)
    norm = float(np.linalg.norm(quat))
    if norm <= 0.0:
        raise ValueError("Recorded table has an invalid zero quaternion")
    w, x, y, z = quat / norm
    r00 = 1.0 - 2.0 * (y * y + z * z)
    r01 = 2.0 * (x * y - w * z)
    r10 = 2.0 * (x * y + w * z)
    r11 = 1.0 - 2.0 * (x * x + z * z)
    extent_x = abs(r00) * 0.5 * size[0] + abs(r01) * 0.5 * size[1] + clearance_m
    extent_y = abs(r10) * 0.5 * size[0] + abs(r11) * 0.5 * size[1] + clearance_m
    return (
        [position[0] - extent_x, position[1] - extent_y],
        [position[0] + extent_x, position[1] + extent_y],
    )


def _indexed_prim_family(path: str) -> str:
    parts = path.rsplit("_", 3)
    if len(parts) == 4 and all(part.isdigit() for part in parts[1:]):
        return parts[0]
    return path


_ATTACHMENT_CATEGORIES = {
    "armchair": frozenset({"pillow"}),
    "sofa": frozenset({"pillow"}),
    "cabinet": frozenset({"fixture", "sink"}),
    "counter": frozenset({"cabinet", "fixture", "sink"}),
    "countertop": frozenset({"cabinet", "fixture", "sink"}),
    "kitchen_counter": frozenset({"cabinet", "fixture", "sink"}),
    "sink": frozenset({"fixture"}),
}

_INDEPENDENT_MOUNTED_CATEGORIES = frozenset(
    {
        "painting",
        "poster",
        "mirror",
        "curtain",
        "blinds",
        "window",
        "door",
        "wall",
        "ceiling",
        "lightfixture",
        "ceilinglamp",
        "walllamp",
        "sconce",
    }
)


_SUPPORTING_CATEGORIES = frozenset(
    {
        "basket",
        "box",
        "cabinet",
        "coffeetable",
        "counter",
        "countertop",
        "crate",
        "desk",
        "diningtable",
        "kitchen_counter",
        "nightstand",
        "plate",
        "shelf",
        "sidetable",
        "stand",
        "table",
        "tray",
        "workbench",
    }
)


def _xy_overlap_area(first: IndexedVisualItem, second: IndexedVisualItem) -> float:
    return max(
        0.0, min(first.max_xy[0], second.max_xy[0]) - max(first.min_xy[0], second.min_xy[0])
    ) * max(0.0, min(first.max_xy[1], second.max_xy[1]) - max(first.min_xy[1], second.min_xy[1]))


def detached_cabinet_fronts(items: Iterable[IndexedVisualItem], hidden: set[str]) -> set[str]:
    """Link shallow drawer fronts to the removed cabinet face immediately below them."""
    items = tuple(items)
    counters = [
        item
        for item in items
        if item.prim_path in hidden
        and item.category in {"counter", "countertop", "kitchen_counter"}
    ]
    cabinets = [
        item
        for item in items
        if item.prim_path in hidden
        and item.category == "cabinet"
        and any(_xy_overlap_area(item, counter) > 0 for counter in counters)
    ]
    if not cabinets:
        return set()
    families = set()
    for front in items:
        if front.category not in {"drawer", "door"} or front.min_z is None or front.max_z is None:
            continue
        if not 0.025 <= front.max_z - front.min_z <= 0.30:
            continue
        for cabinet in cabinets:
            if cabinet.max_z is None or not -0.03 <= front.min_z - cabinet.max_z <= 0.10:
                continue
            # Match the front's width along the cabinet face, including inset drawers.
            axis = max(range(2), key=lambda i: cabinet.max_xy[i] - cabinet.min_xy[i])
            if (
                max(
                    abs(front.min_xy[axis] - cabinet.min_xy[axis]),
                    abs(front.max_xy[axis] - cabinet.max_xy[axis]),
                )
                > 0.05
            ):
                continue
            if _xy_overlap_area(cabinet, front) > 0:
                families.add(_indexed_prim_family(front.prim_path))
                break
    return {item.prim_path for item in items if _indexed_prim_family(item.prim_path) in families}


def _linked_by_geometry(
    parent: IndexedVisualItem,
    child: IndexedVisualItem,
    *,
    supports_children: bool,
    support_contact: bool = True,
) -> bool:
    if None in (parent.min_z, parent.max_z, child.min_z, child.max_z):
        return False
    assert parent.min_z is not None and parent.max_z is not None
    assert child.min_z is not None and child.max_z is not None
    parent_area = (parent.max_xy[0] - parent.min_xy[0]) * (parent.max_xy[1] - parent.min_xy[1])
    child_area = (child.max_xy[0] - child.min_xy[0]) * (child.max_xy[1] - child.min_xy[1])
    if min(parent_area, child_area) <= 0.0:
        return False
    overlap = _xy_overlap_area(parent, child)
    if overlap / child_area < 0.7:
        return False
    same_geometry = (
        overlap / parent_area >= 0.9
        and abs(parent.min_z - child.min_z) <= 0.1
        and abs(parent.max_z - child.max_z) <= 0.1
    )
    surface_mesh = (
        supports_children
        and child.category.strip().lower() == "mesh"
        and child_area <= parent_area
        and child.min_z >= parent.min_z - 0.1
        and abs(child.max_z - parent.max_z) <= 0.1
    )
    supported = child_area <= parent_area and -0.15 <= child.min_z - parent.max_z <= 0.18
    return same_geometry or surface_mesh or (supports_children and support_contact and supported)


def _support_mask_contains(
    mask: np.ndarray,
    origin: tuple[float, float],
    cell_m: float,
    child: IndexedVisualItem,
) -> bool:
    center_x = 0.5 * (child.min_xy[0] + child.max_xy[0])
    center_y = 0.5 * (child.min_xy[1] + child.max_xy[1])
    column = int(round((center_x - origin[0]) / cell_m))
    row = int(round((center_y - origin[1]) / cell_m))
    return 0 <= row < mask.shape[0] and 0 <= column < mask.shape[1] and bool(mask[row, column])


def expand_linked_furniture_paths(
    background_items: Iterable[IndexedVisualItem],
    seed_paths: Iterable[str],
    *,
    occupancy: tuple[np.ndarray, tuple[str, ...], tuple[float, float], float] | None = None,
    support_relations: tuple[tuple[str, str], ...] | None = None,
) -> set[str]:
    """Hide USD descendants, co-located parts, and objects supported by removed assets."""
    items = tuple(background_items)
    mask_by_path = (
        {path: occupancy[0][number] for number, path in enumerate(occupancy[1])}
        if occupancy is not None
        else {}
    )
    support_parent_families = frozenset(
        (_indexed_prim_family(parent), child) for parent, child in (support_relations or ())
    )
    supported_children = frozenset(child for _, child in (support_relations or ()))
    expanded = set(seed_paths)
    while True:
        previous_count = len(expanded)
        hidden_families = {_indexed_prim_family(path) for path in expanded}
        expanded.update(
            item.prim_path
            for item in items
            if _indexed_prim_family(item.prim_path) in hidden_families
            or any(item.prim_path.startswith(path + "/") for path in expanded)
        )
        if support_relations is not None:
            expanded.update(child for parent, child in support_relations if parent in expanded)
        for hidden_item in items:
            if hidden_item.prim_path not in expanded:
                continue
            hidden_category = hidden_item.category.strip().lower()
            for candidate in items:
                if candidate.prim_path in expanded:
                    continue
                candidate_category = candidate.category.strip().lower()
                if (
                    candidate.prim_path in supported_children
                    and (_indexed_prim_family(hidden_item.prim_path), candidate.prim_path)
                    not in support_parent_families
                ):
                    continue
                if candidate_category in _INDEPENDENT_MOUNTED_CATEGORIES or any(
                    token in candidate.prim_path.lower()
                    for token in ("lightfixture", "ceilinglamp", "walllamp")
                ):
                    continue
                if (
                    "uppercabinet" in candidate.prim_path.lower()
                    and "uppercabinet" not in hidden_item.prim_path.lower()
                ):
                    continue
                support_mask = mask_by_path.get(hidden_item.prim_path)
                support_contact = (
                    occupancy is None
                    or support_mask is None
                    or _support_mask_contains(support_mask, occupancy[2], occupancy[3], candidate)
                )
                if _linked_by_geometry(
                    hidden_item,
                    candidate,
                    supports_children=hidden_category in _SUPPORTING_CATEGORIES,
                    support_contact=support_contact,
                ):
                    expanded.add(candidate.prim_path)
                    continue
                if "uppercabinet" in hidden_item.prim_path.lower():
                    continue
                if candidate_category not in _ATTACHMENT_CATEGORIES.get(hidden_category, ()):
                    continue
                if not _xy_overlap_area(hidden_item, candidate):
                    continue
                lower_cabinet = candidate_category == "cabinet" and hidden_category in {
                    "counter",
                    "countertop",
                    "kitchen_counter",
                }
                if not lower_cabinet and None not in (
                    hidden_item.min_z,
                    hidden_item.max_z,
                    candidate.min_z,
                    candidate.max_z,
                ):
                    assert hidden_item.min_z is not None and hidden_item.max_z is not None
                    assert candidate.min_z is not None and candidate.max_z is not None
                    if (
                        candidate.min_z > hidden_item.max_z + 0.18
                        or hidden_item.min_z > candidate.max_z + 0.18
                    ):
                        continue
                expanded.add(candidate.prim_path)
        expanded.update(detached_cabinet_fronts(items, expanded))
        if len(expanded) == previous_count:
            return expanded


def configure_indexed_furniture(
    scene: IndexedBackgroundScene,
    table_cfg: dict[str, Any],
    clearance_m: float = 0.5,
    additional_table_cfgs: list[dict[str, Any]] | None = None,
    table2_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    profile = scene.visual_profile
    table_cfgs = [
        table_cfg,
        *([table2_cfg] if table2_cfg is not None else []),
        *(additional_table_cfgs or []),
    ]
    workspaces = [
        table_workspace_bounds(
            size=[float(v) for v in cfg["size"]],
            position=[float(v) for v in cfg["pose"]["position"]],
            quaternion=[float(v) for v in cfg["pose"]["quaternion"]],
            clearance_m=clearance_m,
        )
        for cfg in table_cfgs
    ]
    visible = set(profile.visible_prim_paths)
    hidden = set(profile.hidden_prim_paths)
    included = set(profile.included_prim_paths)
    excluded = set(profile.excluded_prim_paths)
    workspace_hidden_seed = set(excluded)
    workspace_overlapping: list[dict[str, Any]] = []
    records: dict[str, dict[str, Any]] = {}

    for item in profile.background_items:
        record = {
            "prim_path": item.prim_path,
            "category": item.category,
            "min_xy": item.min_xy,
            "max_xy": item.max_xy,
        }
        records[item.prim_path] = record
        intersects = any(
            not (
                item.max_xy[0] < workspace_min[0]
                or workspace_max[0] < item.min_xy[0]
                or item.max_xy[1] < workspace_min[1]
                or workspace_max[1] < item.min_xy[1]
            )
            for workspace_min, workspace_max in workspaces
        )
        if intersects:
            workspace_overlapping.append(record)
            if item.prim_path not in included and "task_alignment" not in scene.metadata:
                workspace_hidden_seed.add(item.prim_path)

    workspace_hidden_paths = (
        workspace_hidden_seed
        if "task_alignment" in scene.metadata
        else expand_linked_furniture_paths(profile.background_items, workspace_hidden_seed)
    )
    workspace_hidden: list[dict[str, Any]] = []
    retained: list[dict[str, Any]] = []
    for item in profile.background_items:
        record = records[item.prim_path]
        if item.prim_path in workspace_hidden_paths:
            visible.discard(item.prim_path)
            hidden.add(item.prim_path)
            workspace_hidden.append(record)
        else:
            visible.add(item.prim_path)
            hidden.discard(item.prim_path)
            retained.append(record)

    profile.visible_prim_paths = sorted(visible)
    profile.hidden_prim_paths = sorted(hidden)
    result = {
        "clearance_m": float(clearance_m),
        "workspace_min_xy": workspaces[0][0],
        "workspace_max_xy": workspaces[0][1],
        "workspace_bounds": [
            {"min_xy": workspace_min, "max_xy": workspace_max}
            for workspace_min, workspace_max in workspaces
        ],
        "workspaces": [
            {
                "table": "table" if index == 1 else f"table{index}",
                "workspace_min_xy": workspace_min,
                "workspace_max_xy": workspace_max,
            }
            for index, (workspace_min, workspace_max) in enumerate(workspaces, start=1)
        ],
        "retained_items": retained,
        "workspace_hidden_items": workspace_hidden,
        "workspace_overlapping_items": workspace_overlapping,
    }
    scene.metadata["furniture_filter"] = result
    return result
