"""Plan task-aware indexed backgrounds around immutable task layouts.

The scenes package uses lazy exports so the batch planner can load these helpers
without importing simulator dependencies before launching Isaac Sim.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

TABLE_CATEGORIES = frozenset(
    {
        "coffee_table",
        "coffeetable",
        "counter",
        "countertop",
        "desk",
        "dining_table",
        "diningtable",
        "kitchen_counter",
        "side_table",
        "sidetable",
        "table",
        "workbench",
    }
)
CANDIDATE_MANIFEST_SCHEMA = "humano.eval-scene-candidates.v2"
EVAL_SCENE_SELECTION_SCHEMA = "humano.eval-scene-selection.v2"
SCENE_INDEX_SCHEMA = "simple.scene-index.v2"
OCCUPANCY_CELL_M = 0.05


@dataclass(frozen=True)
class AxisAlignedBounds:
    min_xy: tuple[float, float]
    max_xy: tuple[float, float]

    @property
    def center_xy(self) -> tuple[float, float]:
        return (
            0.5 * (self.min_xy[0] + self.max_xy[0]),
            0.5 * (self.min_xy[1] + self.max_xy[1]),
        )

    @property
    def size_xy(self) -> tuple[float, float]:
        return (
            self.max_xy[0] - self.min_xy[0],
            self.max_xy[1] - self.min_xy[1],
        )

    def contains_point(self, point_xy: tuple[float, float]) -> bool:
        return (
            self.min_xy[0] <= point_xy[0] <= self.max_xy[0]
            and self.min_xy[1] <= point_xy[1] <= self.max_xy[1]
        )

    def contains_bounds(self, other: AxisAlignedBounds) -> bool:
        return self.contains_point(other.min_xy) and self.contains_point(other.max_xy)

    def intersection_area(self, other: AxisAlignedBounds) -> float:
        width = min(self.max_xy[0], other.max_xy[0]) - max(self.min_xy[0], other.min_xy[0])
        height = min(self.max_xy[1], other.max_xy[1]) - max(self.min_xy[1], other.min_xy[1])
        return max(0.0, width) * max(0.0, height)


@dataclass(frozen=True)
class TaskAnchor:
    center_xy: tuple[float, float]
    size_xy: tuple[float, float]
    yaw_rad: float


@dataclass(frozen=True)
class TaskLayout:
    episode_index: int
    instruction: str
    protected_bounds: AxisAlignedBounds
    asset_categories: tuple[str, ...] = ()
    anchor_center_xy: tuple[float, float] | None = None
    anchor_size_xy: tuple[float, float] | None = None
    anchor_yaw_rad: float | None = None
    static_bounds: tuple[AxisAlignedBounds, ...] = ()
    occupied_xy: np.ndarray | None = None
    hide_occupied_xy: np.ndarray | None = None
    secondary_anchor: TaskAnchor | None = None

    @property
    def anchor(self) -> TaskAnchor | None:
        if (
            self.anchor_center_xy is None
            or self.anchor_size_xy is None
            or self.anchor_yaw_rad is None
        ):
            return None
        return TaskAnchor(self.anchor_center_xy, self.anchor_size_xy, self.anchor_yaw_rad)


@dataclass(frozen=True)
class SceneRecord:
    uid: str
    bounds: AxisAlignedBounds
    items: tuple[SceneItem, ...]
    immutable_obstacles: tuple[SceneObstacle, ...] = ()
    semantic_tags: tuple[str, ...] = ()
    revision: str = "unknown"
    occupancy_path: Path | None = None
    support_relations: tuple[tuple[str, str], ...] | None = None


@dataclass(frozen=True)
class SceneItem:
    prim_path: str
    category: str
    bounds: AxisAlignedBounds
    min_z: float | None = None
    max_z: float | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> SceneItem:
        return cls(
            prim_path=str(value["prim_path"]),
            category=str(value["category"]),
            bounds=AxisAlignedBounds(
                tuple(float(item) for item in value["min_xy"]),
                tuple(float(item) for item in value["max_xy"]),
            ),
            min_z=float(value["min_z"]) if value.get("min_z") is not None else None,
            max_z=float(value["max_z"]) if value.get("max_z") is not None else None,
        )


@dataclass(frozen=True)
class SceneObstacle:
    prim_path: str
    category: str
    bounds: AxisAlignedBounds
    min_z: float
    max_z: float

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> SceneObstacle:
        min_xyz = tuple(float(item) for item in value["min_xyz"])
        max_xyz = tuple(float(item) for item in value["max_xyz"])
        if len(min_xyz) != 3 or len(max_xyz) != 3:
            raise ValueError("Immutable obstacle bounds must contain three coordinates")
        return cls(
            prim_path=str(value["prim_path"]),
            category=str(value["category"]),
            bounds=AxisAlignedBounds(min_xyz[:2], max_xyz[:2]),
            min_z=min_xyz[2],
            max_z=max_xyz[2],
        )


@dataclass(frozen=True)
class ScenePlacement:
    scene_uid: str
    strategy: str
    translation_xy: tuple[float, float]
    yaw_rad: float
    anchor_prim_path: str | None
    task_anchor_center_xy: tuple[float, float] | None
    hidden_item_count: int
    hidden_overlap_area_m2: float
    score: float
    semantic_penalty: float = 0.0
    search_stage: str = "coarse"
    auto_hidden_prim_paths: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "scene_uid": self.scene_uid,
            "strategy": self.strategy,
            "translation_xy": list(self.translation_xy),
            "yaw_rad": self.yaw_rad,
            "yaw_deg": math.degrees(self.yaw_rad),
            "anchor_prim_path": self.anchor_prim_path,
            "task_anchor_center_xy": (
                list(self.task_anchor_center_xy) if self.task_anchor_center_xy is not None else None
            ),
            "hidden_item_count": self.hidden_item_count,
            "hidden_overlap_area_m2": self.hidden_overlap_area_m2,
            "semantic_penalty": self.semantic_penalty,
            "score": self.score,
            "search_stage": self.search_stage,
            "auto_hidden_prim_paths": list(self.auto_hidden_prim_paths),
            "fallback_reason": {
                "fine_translation": "coarse_search_exhausted",
                "local_asset_hide": "fine_search_exhausted",
                "all_hide_fallback": "fine_search_exhausted",
            }.get(self.search_stage),
        }


def _normalized_category(category: str) -> str:
    value = category.strip().lower().replace("-", "_").replace(" ", "_")
    return "_".join(part for part in value.split("_") if part)


def _yaw_from_quaternion(quaternion: Iterable[float]) -> float:
    values = tuple(float(value) for value in quaternion)
    if len(values) != 4:
        raise ValueError("Quaternion must contain four values in wxyz order")
    norm = math.sqrt(sum(value * value for value in values))
    if norm <= 0.0:
        raise ValueError("Quaternion must be non-zero")
    w, x, y, z = (value / norm for value in values)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _position_xy(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, dict):
        return None
    position = value.get("position")
    if position is None and isinstance(value.get("pose"), dict):
        position = value["pose"].get("position")
    if not isinstance(position, (list, tuple)) or len(position) < 2:
        return None
    return float(position[0]), float(position[1])


def _table_bounds(anchor: TaskAnchor, clearance_m: float) -> AxisAlignedBounds:
    cosine = abs(math.cos(anchor.yaw_rad))
    sine = abs(math.sin(anchor.yaw_rad))
    extent_x = 0.5 * (cosine * anchor.size_xy[0] + sine * anchor.size_xy[1]) + clearance_m
    extent_y = 0.5 * (sine * anchor.size_xy[0] + cosine * anchor.size_xy[1]) + clearance_m
    return AxisAlignedBounds(
        (anchor.center_xy[0] - extent_x, anchor.center_xy[1] - extent_y),
        (anchor.center_xy[0] + extent_x, anchor.center_xy[1] + extent_y),
    )


def _bounds_around_points(
    points: Iterable[tuple[float, float]], clearance_m: float
) -> AxisAlignedBounds:
    values = tuple(points)
    if not values:
        values = ((0.0, 0.0),)
    return AxisAlignedBounds(
        (
            min(point[0] for point in values) - clearance_m,
            min(point[1] for point in values) - clearance_m,
        ),
        (
            max(point[0] for point in values) + clearance_m,
            max(point[1] for point in values) + clearance_m,
        ),
    )


def _union_bounds(first: AxisAlignedBounds, second: AxisAlignedBounds) -> AxisAlignedBounds:
    return AxisAlignedBounds(
        (min(first.min_xy[0], second.min_xy[0]), min(first.min_xy[1], second.min_xy[1])),
        (max(first.max_xy[0], second.max_xy[0]), max(first.max_xy[1], second.max_xy[1])),
    )


def parse_task_layout(
    environment_config: dict[str, Any] | str,
    *,
    episode_index: int,
    clearance_m: float,
) -> TaskLayout:
    """Extract the immutable task footprint and optional table anchor from one episode."""
    if clearance_m < 0.0:
        raise ValueError("Task clearance must be non-negative")
    config = (
        json.loads(environment_config)
        if isinstance(environment_config, str)
        else environment_config
    )
    dr_state = config.get("dr_state_dict", {})
    if not isinstance(dr_state, dict):
        raise ValueError("environment_config.dr_state_dict must be a mapping")

    instruction = str(dr_state.get("language", ""))
    spatial = dr_state.get("spatial")
    points = (
        tuple(
            point
            for point in (_position_xy(value) for value in spatial.values())
            if point is not None
        )
        if isinstance(spatial, dict)
        else ()
    )
    protected = _bounds_around_points(points, clearance_m)

    anchor = None
    secondary_anchor = None
    scene_state = dr_state.get("scene")
    tables = (
        tuple(
            (name, scene_state.get(name))
            for name in ("table", "table2")
            if isinstance(scene_state.get(name), dict)
        )
        if isinstance(scene_state, dict)
        else ()
    )
    static_bounds = [_bounds_around_points((point,), clearance_m) for point in points]
    for table_name, table in tables:
        assert isinstance(table, dict)
        pose = table.get("pose", {})
        position = pose.get("position", []) if isinstance(pose, dict) else []
        size = table.get("size", [])
        quaternion = pose.get("quaternion", []) if isinstance(pose, dict) else []
        if len(position) < 2 or len(size) < 2 or len(quaternion) != 4:
            raise ValueError(f"Recorded task {table_name} must define size and pose")
        table_anchor = TaskAnchor(
            center_xy=(float(position[0]), float(position[1])),
            size_xy=(float(size[0]), float(size[1])),
            yaw_rad=_yaw_from_quaternion(quaternion),
        )
        table_bounds = _table_bounds(table_anchor, clearance_m)
        protected = _union_bounds(protected, table_bounds)
        static_bounds.append(table_bounds)
        if table_name == "table":
            anchor = table_anchor
        else:
            secondary_anchor = table_anchor

    return TaskLayout(
        episode_index=episode_index,
        instruction=instruction,
        protected_bounds=protected,
        asset_categories=(
            tuple(
                sorted(
                    {
                        _normalized_category(str(actor_name).split("/", maxsplit=1)[0])
                        for actor_name in spatial
                        if not str(actor_name).startswith("g1_")
                    }
                )
            )
            if isinstance(spatial, dict)
            else ()
        ),
        anchor_center_xy=anchor.center_xy if anchor is not None else None,
        anchor_size_xy=anchor.size_xy if anchor is not None else None,
        anchor_yaw_rad=anchor.yaw_rad if anchor is not None else None,
        static_bounds=tuple(static_bounds),
        secondary_anchor=secondary_anchor,
    )


def project_trajectory_occupancy(
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    static_bounds: tuple[AxisAlignedBounds, ...],
    *,
    margin_m: float,
) -> np.ndarray:
    """Union trajectory AABBs over time on the same XY grid as room sidecars."""
    lower = np.asarray(bounds_min, dtype=np.float64)[:, :2] - margin_m
    upper = np.asarray(bounds_max, dtype=np.float64)[:, :2] + margin_m
    if static_bounds:
        lower = np.concatenate((lower, np.asarray([box.min_xy for box in static_bounds])))
        upper = np.concatenate((upper, np.asarray([box.max_xy for box in static_bounds])))
    if len(lower) == 0:
        raise ValueError("Cannot project an empty task occupancy")
    start = np.floor(lower / OCCUPANCY_CELL_M).astype(np.int32)
    stop = np.ceil(upper / OCCUPANCY_CELL_M).astype(np.int32) + 1
    origin = start.min(axis=0)
    start -= origin
    stop -= origin
    shape = tuple(stop.max(axis=0)[::-1] + 1)
    difference = np.zeros(shape, dtype=np.int32)
    np.add.at(difference, (start[:, 1], start[:, 0]), 1)
    np.add.at(difference, (stop[:, 1], start[:, 0]), -1)
    np.add.at(difference, (start[:, 1], stop[:, 0]), -1)
    np.add.at(difference, (stop[:, 1], stop[:, 0]), 1)
    rows, columns = np.nonzero(difference.cumsum(axis=0).cumsum(axis=1))
    return (np.column_stack((columns, rows)) + origin) * OCCUPANCY_CELL_M


def load_scene_records(
    scene_index: str | Path, *, bev_cache_root: str | Path | None = None
) -> tuple[SceneRecord, ...]:
    path = Path(scene_index).expanduser().absolute()
    content = json.loads(path.read_text(encoding="utf-8"))
    if content.get("schema") != SCENE_INDEX_SCHEMA:
        raise ValueError(f"Unsupported scene index schema: {path}")
    records = []
    cached_scenes = None
    support_cache = {}
    if bev_cache_root is not None:
        from humano.scenes.bev_cache import scene_sidecars, scene_support_relations

        cached_scenes = scene_sidecars(bev_cache_root, path)
        support_cache = scene_support_relations(bev_cache_root)
    for entry in content.get("scenes", []):
        bounds = entry.get("bounds", {})
        if not entry.get("uid") or not bounds.get("validated") or bounds.get("policy") != "strict":
            continue
        if entry.get("metadata", {}).get("floor_alignment_status") == "mixed_levels":
            continue
        if cached_scenes is not None and entry["uid"] not in cached_scenes:
            continue
        profile = entry.get("visual_profile") or {}
        navigation = entry.get("navigation") or {}
        if navigation.get("schema") != "simple.scene-navigation.v1" or not navigation.get(
            "validated"
        ):
            raise ValueError(f"Scene has no validated navigation obstacles: {entry['uid']}")
        occupancy_path = None
        if cached_scenes is not None:
            revision, occupancy_path = cached_scenes[entry["uid"]]
            if revision != str(entry.get("revision", "unknown")):
                raise ValueError(f"Public BEV scene revision changed: {entry['uid']}")
        records.append(
            SceneRecord(
                uid=str(entry["uid"]),
                bounds=AxisAlignedBounds(
                    tuple(float(value) for value in bounds["min_xy"]),
                    tuple(float(value) for value in bounds["max_xy"]),
                ),
                items=tuple(
                    SceneItem.from_dict(item) for item in profile.get("background_items", [])
                ),
                immutable_obstacles=tuple(
                    SceneObstacle.from_dict(item)
                    for item in navigation.get("immutable_obstacles", [])
                ),
                semantic_tags=_scene_semantic_tags(entry),
                revision=str(entry.get("revision", "unknown")),
                occupancy_path=occupancy_path,
                support_relations=support_cache.get(str(entry["uid"])),
            )
        )
    if not records:
        raise ValueError(f"Scene index has no validated strict scene: {path}")
    return tuple(records)


def freeze_placement_hidden_paths(scene: SceneRecord, placement: ScenePlacement) -> tuple[str, ...]:
    """Expand the selected BEV conflict seeds before render or eval replay."""
    from humano.scenes.indexed import IndexedVisualItem, expand_linked_furniture_paths

    if scene.occupancy_path is None:
        raise ValueError(f"Scene has no asset BEV occupancy: {scene.uid}")
    seeds = set(placement.auto_hidden_prim_paths)
    if placement.strategy == "all_hide_fallback":
        seeds.update(item.prim_path for item in scene.items)
    elif placement.anchor_prim_path is not None:
        seeds.add(placement.anchor_prim_path)
    if not seeds:
        return ()
    items = tuple(
        IndexedVisualItem(
            item.prim_path,
            item.category,
            list(item.bounds.min_xy),
            list(item.bounds.max_xy),
            item.min_z,
            item.max_z,
        )
        for item in scene.items
    )
    occupancy = _load_occupancy(scene.occupancy_path, scene.uid, scene.revision)
    return tuple(
        sorted(
            expand_linked_furniture_paths(
                items, seeds, occupancy=occupancy, support_relations=scene.support_relations
            )
        )
    )


def _transform_point(
    point_xy: tuple[float, float], yaw_rad: float, translation_xy: tuple[float, float]
) -> tuple[float, float]:
    cosine = math.cos(yaw_rad)
    sine = math.sin(yaw_rad)
    return (
        cosine * point_xy[0] - sine * point_xy[1] + translation_xy[0],
        sine * point_xy[0] + cosine * point_xy[1] + translation_xy[1],
    )


def transform_bounds(
    bounds: AxisAlignedBounds, yaw_rad: float, translation_xy: tuple[float, float]
) -> AxisAlignedBounds:
    corners = (
        bounds.min_xy,
        (bounds.min_xy[0], bounds.max_xy[1]),
        bounds.max_xy,
        (bounds.max_xy[0], bounds.min_xy[1]),
    )
    transformed = tuple(_transform_point(point, yaw_rad, translation_xy) for point in corners)
    return AxisAlignedBounds(
        (min(point[0] for point in transformed), min(point[1] for point in transformed)),
        (max(point[0] for point in transformed), max(point[1] for point in transformed)),
    )


def _scene_semantic_tags(entry: dict[str, Any]) -> tuple[str, ...]:
    metadata = entry.get("metadata") if isinstance(entry.get("metadata"), dict) else {}
    values = {
        _normalized_category(str(value))
        for key in ("room_type", "scene_type")
        if (value := metadata.get(key))
    }
    values.update(
        _normalized_category(str(value))
        for key in ("room_types", "semantic_tags", "tags")
        if isinstance((items := metadata.get(key)), (list, tuple))
        for value in items
    )
    match = re.search(r"FloorPlan(\d+)", str(entry.get("uid", "")), re.IGNORECASE)
    if match is not None:
        floor_plan = int(match.group(1))
        if 1 <= floor_plan <= 30:
            values.add("kitchen")
        elif 201 <= floor_plan <= 230:
            values.add("living_room")
        elif 301 <= floor_plan <= 330:
            values.add("bedroom")
        elif 401 <= floor_plan <= 430:
            values.add("bathroom")
    return tuple(sorted(values))


def _preferred_task_room_types(task: TaskLayout) -> frozenset[str]:
    lowered = f"{task.instruction} {' '.join(task.asset_categories)}".lower()
    keyword_groups = {
        "bathroom": ("bath", "shower", "toilet", "towel"),
        "bedroom": ("bed", "blanket", "pillow"),
        "kitchen": (
            "apple",
            "banana",
            "bowl",
            "cabinet",
            "cup",
            "dish",
            "faucet",
            "food",
            "fruit",
            "fridge",
            "microwave",
            "oven",
            "pan",
            "refrigerator",
            "sink",
        ),
        "living_room": ("ball", "book", "chair", "couch", "remote", "sofa", "television"),
    }
    return frozenset(
        room_type
        for room_type, keywords in keyword_groups.items()
        if any(keyword in lowered for keyword in keywords)
    )


def _semantic_penalty(scene: SceneRecord, task: TaskLayout) -> float:
    preferred = _preferred_task_room_types(task)
    if not preferred or not scene.semantic_tags:
        return 0.0
    return 0.0 if preferred.intersection(scene.semantic_tags) else 10000.0


@lru_cache(maxsize=128)
def _load_occupancy(path: Path, uid: str, revision: str):
    with np.load(path) as data:
        if str(data["uid"]) != uid or str(data["revision"]) != revision:
            raise ValueError(f"Stale occupancy sidecar: {path}")
        masks = np.unpackbits(data["masks"], axis=2, count=int(data["width"]))
        paths = tuple(str(value) for value in data["paths"])
        origin = tuple(float(value) for value in data["origin"])
        cell_m = float(data["cell_m"])
    masks.setflags(write=False)
    return masks, paths, origin, cell_m


@lru_cache(maxsize=512)
def _occupancy_prefix(path: Path, uid: str, revision: str, hidden_paths: frozenset[str]):
    masks, paths, origin, cell_m = _load_occupancy(path, uid, revision)
    keep = [index for index, item_path in enumerate(paths) if item_path not in hidden_paths]
    occupied = np.any(masks[keep], axis=0) if keep else np.zeros(masks.shape[1:], dtype=bool)
    prefix = np.pad(occupied.cumsum(axis=0).cumsum(axis=1), ((1, 0), (1, 0)))
    return prefix, occupied, origin, cell_m


def _occupancy_hits(
    scene: SceneRecord,
    task: TaskLayout,
    yaw_rad: float,
    translation_xy: tuple[float, float],
    hidden_paths: frozenset[str],
) -> bool:
    assert scene.occupancy_path is not None
    prefix, occupied, origin, cell_m = _occupancy_prefix(
        scene.occupancy_path, scene.uid, scene.revision, hidden_paths
    )
    inverse_translation = _transform_point(
        (-translation_xy[0], -translation_xy[1]), -yaw_rad, (0.0, 0.0)
    )
    local = transform_bounds(task.protected_bounds, -yaw_rad, inverse_translation)
    x0 = max(0, int(math.floor((local.min_xy[0] - origin[0]) / cell_m)))
    y0 = max(0, int(math.floor((local.min_xy[1] - origin[1]) / cell_m)))
    x1 = min(prefix.shape[1] - 1, int(math.ceil((local.max_xy[0] - origin[0]) / cell_m)) + 1)
    y1 = min(prefix.shape[0] - 1, int(math.ceil((local.max_xy[1] - origin[1]) / cell_m)) + 1)
    if not (prefix[y1, x1] - prefix[y0, x1] - prefix[y1, x0] + prefix[y0, x0]):
        return False
    if task.occupied_xy is None:
        return True
    cosine, sine = math.cos(yaw_rad), math.sin(yaw_rad)
    offset = task.occupied_xy - np.asarray(translation_xy)
    local_x = cosine * offset[:, 0] + sine * offset[:, 1]
    local_y = -sine * offset[:, 0] + cosine * offset[:, 1]
    columns = np.rint((local_x - origin[0]) / cell_m).astype(np.int32)
    rows = np.rint((local_y - origin[1]) / cell_m).astype(np.int32)
    inside = (
        (columns >= 0) & (columns < occupied.shape[1]) & (rows >= 0) & (rows < occupied.shape[0])
    )
    return bool(np.any(occupied[rows[inside], columns[inside]]))


def _colliding_occupancy_paths(
    scene: SceneRecord,
    task: TaskLayout,
    yaw_rad: float,
    translation_xy: tuple[float, float],
    hidden_paths: frozenset[str],
) -> set[str]:
    assert scene.occupancy_path is not None
    masks, paths, origin, cell_m = _load_occupancy(scene.occupancy_path, scene.uid, scene.revision)
    points = task.hide_occupied_xy if task.hide_occupied_xy is not None else task.occupied_xy
    if points is None:
        inverse_translation = _transform_point(
            (-translation_xy[0], -translation_xy[1]), -yaw_rad, (0.0, 0.0)
        )
        local = transform_bounds(task.protected_bounds, -yaw_rad, inverse_translation)
        x0 = max(0, int(math.floor((local.min_xy[0] - origin[0]) / cell_m)))
        y0 = max(0, int(math.floor((local.min_xy[1] - origin[1]) / cell_m)))
        x1 = min(masks.shape[2], int(math.ceil((local.max_xy[0] - origin[0]) / cell_m)) + 1)
        y1 = min(masks.shape[1], int(math.ceil((local.max_xy[1] - origin[1]) / cell_m)) + 1)
        hits = np.any(masks[:, y0:y1, x0:x1], axis=(1, 2))
    else:
        cosine, sine = math.cos(yaw_rad), math.sin(yaw_rad)
        offset = points - np.asarray(translation_xy)
        columns = np.rint(
            (cosine * offset[:, 0] + sine * offset[:, 1] - origin[0]) / cell_m
        ).astype(np.int32)
        rows = np.rint((-sine * offset[:, 0] + cosine * offset[:, 1] - origin[1]) / cell_m).astype(
            np.int32
        )
        inside = (columns >= 0) & (columns < masks.shape[2]) & (rows >= 0) & (rows < masks.shape[1])
        hits = np.any(masks[:, rows[inside], columns[inside]], axis=1)
    return {path for path, hit in zip(paths, hits, strict=True) if hit and path not in hidden_paths}


def _task_intersects_bounds(task: TaskLayout, obstacle: AxisAlignedBounds) -> bool:
    if obstacle.intersection_area(task.protected_bounds) <= 0.0:
        return False
    if task.occupied_xy is None:
        return True
    points = task.occupied_xy
    margin = 0.5 * OCCUPANCY_CELL_M
    return bool(
        np.any(
            (points[:, 0] >= obstacle.min_xy[0] - margin)
            & (points[:, 0] <= obstacle.max_xy[0] + margin)
            & (points[:, 1] >= obstacle.min_xy[1] - margin)
            & (points[:, 1] <= obstacle.max_xy[1] + margin)
        )
    )


def _placement_cost(
    scene: SceneRecord,
    task: TaskLayout,
    *,
    strategy: str,
    translation_xy: tuple[float, float],
    yaw_rad: float,
    anchor_item: SceneItem | None,
    anchor_size_error: float,
    seed: int,
    hidden_paths: frozenset[str] = frozenset(),
    allow_asset_hiding: bool = False,
) -> ScenePlacement | None:
    transformed_room = transform_bounds(scene.bounds, yaw_rad, translation_xy)
    if not transformed_room.contains_bounds(task.protected_bounds):
        return None

    for obstacle in scene.immutable_obstacles:
        transformed_obstacle = transform_bounds(obstacle.bounds, yaw_rad, translation_xy)
        if _task_intersects_bounds(task, transformed_obstacle):
            return None

    auto_hidden: frozenset[str] = frozenset()
    if not allow_asset_hiding:
        if scene.occupancy_path is not None:
            if _occupancy_hits(scene, task, yaw_rad, translation_xy, hidden_paths):
                raised = frozenset(
                    item.prim_path
                    for item in scene.items
                    if item.category == "raisedfloor" and item.prim_path not in hidden_paths
                )
                if not raised or _occupancy_hits(
                    scene, task, yaw_rad, translation_xy, hidden_paths | raised
                ):
                    return None
                auto_hidden = raised
        elif any(
            item.prim_path not in hidden_paths
            and _task_intersects_bounds(
                task, transform_bounds(item.bounds, yaw_rad, translation_xy)
            )
            for item in scene.items
        ):
            return None
    if scene.occupancy_path is not None:
        colliding = (
            _colliding_occupancy_paths(scene, task, yaw_rad, translation_xy, hidden_paths)
            if allow_asset_hiding or task.hide_occupied_xy is not None
            else set()
        )
    else:
        colliding = (
            {
                item.prim_path
                for item in scene.items
                if item.prim_path not in hidden_paths
                and _task_intersects_bounds(
                    task, transform_bounds(item.bounds, yaw_rad, translation_xy)
                )
            }
            if allow_asset_hiding
            else set()
        )
    if colliding:
        from humano.scenes.indexed import IndexedVisualItem, expand_linked_furniture_paths

        indexed_items = tuple(
            IndexedVisualItem(
                item.prim_path,
                item.category,
                list(item.bounds.min_xy),
                list(item.bounds.max_xy),
                item.min_z,
                item.max_z,
            )
            for item in scene.items
        )
        occupancy = (
            _load_occupancy(scene.occupancy_path, scene.uid, scene.revision)
            if scene.occupancy_path is not None
            else None
        )
        auto_hidden = (
            frozenset(
                expand_linked_furniture_paths(
                    indexed_items,
                    colliding,
                    occupancy=occupancy,
                    support_relations=scene.support_relations,
                )
            )
            - hidden_paths
        )
        if scene.occupancy_path is not None and _occupancy_hits(
            scene, task, yaw_rad, translation_xy, hidden_paths | auto_hidden
        ):
            return None
    translation_distance = math.hypot(*translation_xy)
    digest = hashlib.sha256(
        f"{seed}:{task.episode_index}:{scene.uid}:{strategy}:{yaw_rad:.9f}".encode()
    ).digest()
    tie_break = int.from_bytes(digest[:4], "big") / (2**32)
    semantic_penalty = _semantic_penalty(scene, task)
    score = (
        semantic_penalty + 10.0 * anchor_size_error + 0.01 * translation_distance + 1e-6 * tie_break
    )
    return ScenePlacement(
        scene_uid=scene.uid,
        strategy=strategy,
        translation_xy=translation_xy,
        yaw_rad=yaw_rad,
        anchor_prim_path=anchor_item.prim_path if anchor_item is not None else None,
        task_anchor_center_xy=task.anchor_center_xy,
        hidden_item_count=len(hidden_paths | auto_hidden),
        hidden_overlap_area_m2=0.0,
        score=score,
        semantic_penalty=semantic_penalty,
        auto_hidden_prim_paths=tuple(sorted(auto_hidden)),
    )


def _table_placements(
    scene: SceneRecord,
    task: TaskLayout,
    seed: int,
    free_space_grid_m: float,
    *,
    max_results: int | None = None,
    allow_asset_hiding: bool = False,
) -> tuple[ScenePlacement, ...]:
    from humano.scenes.indexed import IndexedVisualItem, expand_linked_furniture_paths

    assert task.anchor is not None
    indexed_items = tuple(
        IndexedVisualItem(
            item.prim_path,
            item.category,
            list(item.bounds.min_xy),
            list(item.bounds.max_xy),
            item.min_z,
            item.max_z,
        )
        for item in scene.items
    )
    occupancy = (
        _load_occupancy(scene.occupancy_path, scene.uid, scene.revision)
        if scene.occupancy_path is not None
        else None
    )
    candidates = []
    seen_parts: set[str] = set()
    target_size = sorted(task.anchor.size_xy)
    by_area = sorted(
        scene.items,
        key=lambda item: item.bounds.size_xy[0] * item.bounds.size_xy[1],
        reverse=True,
    )
    for item in by_area:
        if item.prim_path in seen_parts:
            continue
        category = _normalized_category(item.category)
        if category not in TABLE_CATEGORIES:
            continue
        hidden_paths = frozenset(
            expand_linked_furniture_paths(
                indexed_items,
                {item.prim_path},
                occupancy=occupancy,
                support_relations=scene.support_relations,
            )
        )
        seen_parts.update(hidden_paths)
        item_bounds = item.bounds
        item_size = item_bounds.size_xy
        item_yaw = 0.0 if item_size[0] >= item_size[1] else math.pi / 2.0
        yaw = task.anchor.yaw_rad - item_yaw
        rotated_center = _transform_point(item_bounds.center_xy, yaw, (0.0, 0.0))
        translation = (
            task.anchor.center_xy[0] - rotated_center[0],
            task.anchor.center_xy[1] - rotated_center[1],
        )
        size_error = sum(
            abs(scene_value - task_value) / max(task_value, 1e-6)
            for scene_value, task_value in zip(sorted(item_size), target_size, strict=True)
        )
        exact = _placement_cost(
            scene,
            task,
            strategy="table_anchor",
            translation_xy=translation,
            yaw_rad=yaw,
            anchor_item=item,
            anchor_size_error=size_error,
            seed=seed,
            hidden_paths=hidden_paths,
            allow_asset_hiding=allow_asset_hiding,
        )
        if exact is not None:
            candidates.append(exact)
            continue
        nearby = _free_floor_placements(
            scene,
            task,
            seed,
            free_space_grid_m,
            anchor_item=item,
            hidden_paths=hidden_paths,
            allow_asset_hiding=allow_asset_hiding,
            yaw_angles=(yaw,),
            max_results=max_results,
            translation_center_xy=translation,
            translation_limit_m=max(task.protected_bounds.size_xy),
        )
        nearby_limit_m = max(task.protected_bounds.size_xy)
        candidates.extend(
            replace(
                placement,
                strategy="table_nearby",
                score=placement.score
                + 10.0 * size_error
                + math.dist(placement.translation_xy, translation),
            )
            for placement in nearby
            if math.dist(placement.translation_xy, translation) <= nearby_limit_m
        )
    return tuple(candidates)


def _grid_values(lower: float, upper: float, step: float) -> tuple[float, ...]:
    if lower > upper:
        return ()
    count = int(math.floor((upper - lower) / step))
    values = [lower + index * step for index in range(count + 1)]
    if not values or values[-1] < upper - 1e-9:
        values.append(upper)
    midpoint = 0.5 * (lower + upper)
    return tuple(sorted(set((*values, midpoint))))


def _free_floor_placements(
    scene: SceneRecord,
    task: TaskLayout,
    seed: int,
    free_space_grid_m: float,
    *,
    anchor_item: SceneItem | None = None,
    hidden_paths: frozenset[str] = frozenset(),
    yaw_angles: tuple[float, ...] = (0.0, math.pi / 2.0, math.pi, -math.pi / 2.0),
    max_results: int | None = None,
    translation_center_xy: tuple[float, float] | None = None,
    translation_limit_m: float | None = None,
    allow_asset_hiding: bool = False,
) -> tuple[ScenePlacement, ...]:
    task_center = task.protected_bounds.center_xy
    task_half_size = tuple(0.5 * value for value in task.protected_bounds.size_xy)
    candidates = []
    for yaw in yaw_angles:
        rotated_room = transform_bounds(scene.bounds, yaw, (0.0, 0.0))
        sample_x = _grid_values(
            rotated_room.min_xy[0] + task_half_size[0],
            rotated_room.max_xy[0] - task_half_size[0],
            free_space_grid_m,
        )
        sample_y = _grid_values(
            rotated_room.min_xy[1] + task_half_size[1],
            rotated_room.max_xy[1] - task_half_size[1],
            free_space_grid_m,
        )
        if max_results is not None:
            sample_x = tuple(sorted(sample_x, key=lambda value: abs(value - task_center[0])))
            sample_y = tuple(sorted(sample_y, key=lambda value: abs(value - task_center[1])))
        for x_value in sample_x:
            for y_value in sample_y:
                translation_xy = (task_center[0] - x_value, task_center[1] - y_value)
                if (
                    translation_center_xy is not None
                    and translation_limit_m is not None
                    and math.dist(translation_xy, translation_center_xy) > translation_limit_m
                ):
                    continue
                placement = _placement_cost(
                    scene,
                    task,
                    strategy="free_floor",
                    translation_xy=translation_xy,
                    yaw_rad=yaw,
                    anchor_item=anchor_item,
                    anchor_size_error=0.0,
                    seed=seed,
                    hidden_paths=hidden_paths,
                    allow_asset_hiding=allow_asset_hiding,
                )
                if placement is not None:
                    candidates.append(placement)
                    if max_results is not None and len(candidates) >= max_results:
                        return tuple(candidates)
    return tuple(candidates)


def rank_scene_placements(
    scenes: Iterable[SceneRecord],
    task: TaskLayout,
    *,
    max_candidates: int,
    seed: int,
    free_space_grid_m: float = 0.5,
    placement_variant: int = 0,
    max_fallback_scenes: int = 10,
    allow_all_hide_fallback: bool = True,
) -> tuple[ScenePlacement, ...]:
    """Rank one of the top four placements per scene for a fixed task layout."""
    if max_candidates < 1:
        raise ValueError("max_candidates must be positive")
    if free_space_grid_m <= 0.0:
        raise ValueError("free_space_grid_m must be positive")
    if placement_variant < 0:
        raise ValueError("placement_variant must be non-negative")
    if max_fallback_scenes < 1:
        raise ValueError("max_fallback_scenes must be positive")

    scene_pool = tuple(scenes)
    selected_by_scene: dict[str, ScenePlacement] = {}
    for grid_m, stage in (
        (free_space_grid_m, "coarse"),
        (0.05, "fine_translation"),
        (free_space_grid_m, "local_asset_hide"),
    ):
        active_scenes = scene_pool if stage == "coarse" else scene_pool[:max_fallback_scenes]
        for scene in active_scenes:
            if scene.uid in selected_by_scene:
                continue
            max_results = 4 if stage != "coarse" else None
            table_placements = (
                _table_placements(
                    scene,
                    task,
                    seed,
                    grid_m,
                    max_results=max_results,
                    allow_asset_hiding=stage == "local_asset_hide",
                )
                if task.anchor is not None
                else ()
            )
            if table_placements:
                placements = table_placements
            else:
                free_floor = _free_floor_placements(
                    scene,
                    task,
                    seed,
                    grid_m,
                    max_results=max_results,
                    allow_asset_hiding=stage == "local_asset_hide",
                )
                placements = (
                    tuple(
                        replace(placement, strategy="free_floor_insert_task_table")
                        for placement in free_floor
                    )
                    if task.anchor is not None
                    else free_floor
                )
            if placements:
                ranked = sorted(placements, key=lambda value: value.score)
                variant_count = min(4, len(ranked))
                selected_by_scene[scene.uid] = replace(
                    ranked[placement_variant % variant_count], search_stage=stage
                )
        if len(selected_by_scene) >= max_candidates:
            break

    cleared = []
    for scene in (
        scene_pool[:max_fallback_scenes]
        if allow_all_hide_fallback and len(selected_by_scene) < max_candidates
        else ()
    ):
        if scene.uid in selected_by_scene:
            continue
        hidden_paths = frozenset(item.prim_path for item in scene.items)
        placements = _free_floor_placements(
            scene,
            task,
            seed,
            free_space_grid_m,
            hidden_paths=hidden_paths,
            max_results=4,
        )
        if placements:
            ranked = sorted(placements, key=lambda value: value.score)
            selected = ranked[placement_variant % len(ranked)]
            cleared.append(
                replace(selected, strategy="all_hide_fallback", search_stage="all_hide_fallback")
            )
    if not selected_by_scene and not cleared:
        raise ValueError("No indexed scene can contain the protected task footprint")
    stage_rank = {"coarse": 0, "fine_translation": 1, "local_asset_hide": 2, "all_hide_fallback": 3}
    return tuple(
        sorted(
            (*selected_by_scene.values(), *cleared),
            key=lambda value: (
                value.semantic_penalty,
                stage_rank[value.search_stage],
                value.strategy not in {"table_anchor", "table_nearby"},
                value.score,
            ),
        )[:max_candidates]
    )
