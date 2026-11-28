from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from humano.dr import TabletopSceneDRCfg
from humano.dr.types import Box

MOVEMENT_MIN_ROBOT_TARGET_DISTANCE_M = 1.0
MOVEMENT_EXTRA_ROBOT_TARGET_DISTANCE_M = 0.5
MOVEMENT_GROUND_PICK_EXTRA_ROBOT_TARGET_DISTANCE_M = 0.3


def apply_movement_layout(
    dr_cfgs: dict[str, Any],
    *,
    target_region_name: str = "target_region",
    flip_articulated_yaw: bool = False,
    extra_robot_target_distance_m: float = 0.0,
) -> None:
    """Apply the movement-suite layout to a copied Basic task DR config."""
    spatial_cfg = dr_cfgs["spatial"]
    target_region = getattr(spatial_cfg, target_region_name)
    target_position = list(target_region.middle())
    scene_cfg = dr_cfgs.get("scene")

    if isinstance(scene_cfg, TabletopSceneDRCfg):
        table_position = scene_cfg.table_position
        if table_position is None:
            raise ValueError("movement tabletop tasks require a fixed table position")
        table_x = float(table_position.middle()[0])
        target_position[0] = 2.0 * table_x - float(target_position[0])
        setattr(
            spatial_cfg,
            target_region_name,
            Box(low=target_position, high=list(target_position)),
        )
        if flip_articulated_yaw:
            yaw = float(spatial_cfg.articulated_rotate_z.middle()) + math.pi
            spatial_cfg.articulated_rotate_z = Box(low=yaw, high=yaw)
        return

    robot_position = list(spatial_cfg.robot_region.middle())
    offset_x = float(robot_position[0]) - float(target_position[0])
    offset_y = float(robot_position[1]) - float(target_position[1])
    distance = math.hypot(offset_x, offset_y)
    minimum_distance = MOVEMENT_MIN_ROBOT_TARGET_DISTANCE_M + extra_robot_target_distance_m
    if distance >= minimum_distance:
        return
    if distance == 0.0:
        offset_x, offset_y, distance = -1.0, 0.0, 1.0
    # Add a tiny margin so floating-point roundoff cannot put the result below
    # the configured minimum distance.
    scale = (minimum_distance + 1e-6) / distance
    robot_position[0] = float(target_position[0]) + offset_x * scale
    robot_position[1] = float(target_position[1]) + offset_y * scale
    spatial_cfg.robot_region = Box(low=robot_position, high=list(robot_position))


TABLE_INITIAL_POSITION_REGIONS: dict[str, tuple[float, float]] = {
    "near_left_corner": (-0.275, 0.30),
    "near_right_corner": (-0.275, -0.30),
    "near_edge_center": (-0.275, 0.0),
    "side_edge_center": (-0.05, -0.32),
    "inward_center": (-0.05, 0.0),
}


def normalize_initial_position(value: str, param_name: str = "initial_position") -> str:
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    if normalized not in TABLE_INITIAL_POSITION_REGIONS:
        choices = ", ".join(TABLE_INITIAL_POSITION_REGIONS)
        raise ValueError(f"Unknown {param_name}={value!r}. Expected one of: {choices}")
    return normalized


def resolve_initial_position_region(
    *,
    initial_position: str | None,
    legacy_initial_position: str | None = None,
    legacy_name: str = "legacy_initial_position",
) -> Box | None:
    if initial_position is None and legacy_initial_position is None:
        return None

    if initial_position is not None:
        position_name = normalize_initial_position(initial_position)
    else:
        position_name = normalize_initial_position(legacy_initial_position or "", legacy_name)

    if legacy_initial_position is not None:
        legacy_position_name = normalize_initial_position(legacy_initial_position, legacy_name)
        if legacy_position_name != position_name:
            raise ValueError(
                f"Conflicting initial positions: initial_position={initial_position!r}, "
                f"{legacy_name}={legacy_initial_position!r}"
            )

    position = list(TABLE_INITIAL_POSITION_REGIONS[position_name])
    return Box(low=position, high=list(position))


def resolve_table_articulated_region(
    *,
    initial_position: str | None,
    z: float,
) -> Box | None:
    if initial_position is None:
        return None

    position_name = normalize_initial_position(initial_position)
    xy = TABLE_INITIAL_POSITION_REGIONS[position_name]
    position = [xy[0], xy[1], z]
    return Box(low=position, high=list(position))


@dataclass(frozen=True)
class AdvancedTablePairLayout:
    name: str
    target_xy: tuple[float, float]
    container_xy: tuple[float, float]


@dataclass(frozen=True)
class AdvancedGroundToTableLayout:
    name: str
    target_xy: tuple[float, float]
    table_xy: tuple[float, float]
    place_xy: tuple[float, float]


@dataclass(frozen=True)
class AdvancedPushToTableLayout:
    name: str
    robot_xyz: tuple[float, float, float]
    object_xy: tuple[float, float]
    table_xy: tuple[float, float]


ADVANCED_TABLE_PAIR_LAYOUTS: dict[str, tuple[tuple[float, float], tuple[float, float]]] = {
    "near_left_corner": ((-0.275, 0.24), (-0.05, -0.18)),
    "near_right_corner": ((-0.275, -0.24), (-0.05, 0.18)),
    "near_edge_center": ((-0.275, 0.0), (-0.05, 0.22)),
    "side_edge_center": ((-0.05, -0.30), (-0.275, 0.06)),
    "inward_center": ((-0.05, 0.0), (-0.275, -0.22)),
}


ADVANCED_MILK_TO_TRASH_CAN_LAYOUTS: dict[str, tuple[tuple[float, float], tuple[float, float]]] = {
    "near_left_corner": ((-0.275, 0.24), (-0.52, 0.28)),
    "near_right_corner": ((-0.275, -0.24), (-0.52, -0.28)),
    "near_edge_center": ((-0.275, 0.0), (-0.52, 0.18)),
    "side_edge_center": ((-0.05, -0.30), (-0.52, -0.18)),
    "inward_center": ((-0.05, 0.0), (-0.52, 0.0)),
}


ADVANCED_GROUND_TO_TABLE_LAYOUTS: dict[
    str, tuple[tuple[float, float], tuple[float, float], tuple[float, float]]
] = {
    "near_left_corner": ((-0.62, 0.24), (0.32, 0.16), (0.05, 0.16)),
    "near_right_corner": ((-0.62, -0.24), (0.32, -0.16), (0.05, -0.16)),
    "near_edge_center": ((-0.62, 0.0), (0.32, 0.02), (0.05, 0.02)),
    "side_edge_center": ((-0.52, -0.34), (0.32, -0.10), (0.05, -0.10)),
    "inward_center": ((-0.52, 0.12), (0.42, 0.0), (0.16, 0.0)),
}


PUSH_TO_TABLE_INITIAL_POSITIONS = ("front", "left", "right")


def _push_to_table_layouts(
    *,
    robot_x: float,
    object_x: float,
    table_x: float,
) -> dict[str, tuple[tuple[float, float, float], tuple[float, float], tuple[float, float]]]:
    distance = object_x - robot_x
    object_xy = (object_x, 0.0)
    table_xy = (table_x, 0.0)
    return {
        "front": ((robot_x, 0.0, 0.0), object_xy, table_xy),
        "left": ((object_x, distance, 0.0), object_xy, table_xy),
        "right": ((object_x, -distance, 0.0), object_xy, table_xy),
    }


ADVANCED_CHAIR_PUSH_TO_TABLE_LAYOUTS = _push_to_table_layouts(
    robot_x=-0.95,
    object_x=-0.28,
    table_x=1.0,
)


ADVANCED_DOLLY_PUSH_TO_TABLE_LAYOUTS = _push_to_table_layouts(
    robot_x=-1.15,
    object_x=0.25,
    table_x=1.35,
)


BOOK_SHELF_LAYER_INITIAL_POSITIONS = ("layer1", "layer2", "layer3")
_BOOK_SHELF_LAYER_ALIASES = {
    "layer1": "layer1",
    "lower": "layer1",
    "bottom": "layer1",
    "下层": "layer1",
    "第一层": "layer1",
    "layer2": "layer2",
    "middle": "layer2",
    "中层": "layer2",
    "第二层": "layer2",
    "layer3": "layer3",
    "upper": "layer3",
    "top": "layer3",
    "上层": "layer3",
    "第三层": "layer3",
}
BOOK_SHELF_LAYER_PLACE_OFFSETS: dict[str, tuple[float, float, float]] = {
    "layer1": (-0.04, -0.11, 0.30),
    "layer2": (-0.04, -0.11, 0.52),
    "layer3": (-0.04, -0.11, 0.93),
}


def _box_xy(xy: tuple[float, float]) -> Box:
    position = [float(xy[0]), float(xy[1])]
    return Box(low=position, high=list(position))


def _box_xyz(xyz: tuple[float, float, float]) -> Box:
    position = [float(xyz[0]), float(xyz[1]), float(xyz[2])]
    return Box(low=position, high=list(position))


def _normalize_table_layout_name(value: str, param_name: str = "initial_position") -> str:
    return normalize_initial_position(value, param_name=param_name)


def normalize_push_to_table_initial_position(
    value: str, param_name: str = "initial_position"
) -> str:
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    if normalized not in PUSH_TO_TABLE_INITIAL_POSITIONS:
        choices = ", ".join(PUSH_TO_TABLE_INITIAL_POSITIONS)
        raise ValueError(f"Unknown {param_name}={value!r}. Expected one of: {choices}")
    return normalized


def resolve_advanced_table_pair_layout(
    initial_position: str | None,
) -> tuple[AdvancedTablePairLayout, Box, Box] | None:
    if initial_position is None:
        return None
    name = _normalize_table_layout_name(initial_position)
    target_xy, container_xy = ADVANCED_TABLE_PAIR_LAYOUTS[name]
    layout = AdvancedTablePairLayout(name=name, target_xy=target_xy, container_xy=container_xy)
    return layout, _box_xy(target_xy), _box_xy(container_xy)


def resolve_advanced_milk_to_trash_can_layout(
    initial_position: str | None,
) -> tuple[AdvancedTablePairLayout, Box, Box] | None:
    if initial_position is None:
        return None
    name = _normalize_table_layout_name(initial_position)
    target_xy, container_xy = ADVANCED_MILK_TO_TRASH_CAN_LAYOUTS[name]
    layout = AdvancedTablePairLayout(name=name, target_xy=target_xy, container_xy=container_xy)
    return layout, _box_xy(target_xy), _box_xy(container_xy)


def resolve_advanced_ground_to_table_layout(
    initial_position: str | None,
) -> tuple[AdvancedGroundToTableLayout, Box, Box] | None:
    if initial_position is None:
        return None
    name = _normalize_table_layout_name(initial_position)
    target_xy, table_xy, place_xy = ADVANCED_GROUND_TO_TABLE_LAYOUTS[name]
    layout = AdvancedGroundToTableLayout(
        name=name,
        target_xy=target_xy,
        table_xy=table_xy,
        place_xy=place_xy,
    )
    return layout, _box_xy(target_xy), _box_xy(table_xy)


def resolve_advanced_chair_push_to_table_layout(
    initial_position: str | None,
) -> tuple[AdvancedPushToTableLayout, Box, Box, Box] | None:
    if initial_position is None:
        return None
    name = normalize_push_to_table_initial_position(initial_position)
    robot_xyz, object_xy, table_xy = ADVANCED_CHAIR_PUSH_TO_TABLE_LAYOUTS[name]
    layout = AdvancedPushToTableLayout(
        name=name,
        robot_xyz=robot_xyz,
        object_xy=object_xy,
        table_xy=table_xy,
    )
    return layout, _box_xyz(robot_xyz), _box_xy(object_xy), _box_xy(table_xy)


def resolve_advanced_dolly_push_to_table_layout(
    initial_position: str | None,
) -> tuple[AdvancedPushToTableLayout, Box, Box, Box] | None:
    if initial_position is None:
        return None
    name = normalize_push_to_table_initial_position(initial_position)
    robot_xyz, object_xy, table_xy = ADVANCED_DOLLY_PUSH_TO_TABLE_LAYOUTS[name]
    layout = AdvancedPushToTableLayout(
        name=name,
        robot_xyz=robot_xyz,
        object_xy=object_xy,
        table_xy=table_xy,
    )
    articulated_xyz = (object_xy[0], object_xy[1], 0.0)
    return layout, _box_xyz(robot_xyz), _box_xyz(articulated_xyz), _box_xy(table_xy)


def normalize_book_shelf_layer_initial_position(value: str) -> str:
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    if normalized not in _BOOK_SHELF_LAYER_ALIASES:
        choices = ", ".join(BOOK_SHELF_LAYER_INITIAL_POSITIONS)
        raise ValueError(f"Unknown initial_position={value!r}. Expected one of: {choices}")
    return _BOOK_SHELF_LAYER_ALIASES[normalized]


def resolve_book_shelf_layer_place_offset(
    initial_position: str | None,
) -> tuple[float, float, float]:
    layer_name = normalize_book_shelf_layer_initial_position(initial_position or "layer1")
    return BOOK_SHELF_LAYER_PLACE_OFFSETS[layer_name]


DRAWER_INITIAL_POSITIONS = (
    "low_table_edge",
    "low_table_center",
    "high_table_edge",
    "high_table_center",
)
_DRAWER_INITIAL_POSITION_ALIASES = {
    "low_table_edge": "low_table_edge",
    "low_edge": "low_table_edge",
    "低桌桌边": "low_table_edge",
    "低桌边": "low_table_edge",
    "low_table_center": "low_table_center",
    "low_center": "low_table_center",
    "低桌桌中": "low_table_center",
    "低桌中": "low_table_center",
    "high_table_edge": "high_table_edge",
    "high_edge": "high_table_edge",
    "高桌桌边": "high_table_edge",
    "高桌边": "high_table_edge",
    "high_table_center": "high_table_center",
    "high_center": "high_table_center",
    "高桌桌中": "high_table_center",
    "高桌中": "high_table_center",
}


def normalize_drawer_initial_position(value: str) -> str:
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    if normalized not in _DRAWER_INITIAL_POSITION_ALIASES:
        choices = ", ".join(DRAWER_INITIAL_POSITIONS)
        raise ValueError(f"Unknown initial_position={value!r}. Expected one of: {choices}")
    return _DRAWER_INITIAL_POSITION_ALIASES[normalized]


def resolve_drawer_initial_position(
    initial_position: str | None,
) -> tuple[str, float, Box] | None:
    if initial_position is None:
        return None

    position_name = normalize_drawer_initial_position(initial_position)
    table_height = 0.35 if position_name.startswith("low_table") else 0.65
    x = -0.25 if position_name.endswith("_edge") else -0.05
    position = [x, 0.0, table_height]
    return position_name, table_height, Box(low=position, high=list(position))


BOOK_INITIAL_POSITIONS = ("fallen_long", "fallen_cross", "upright_tall", "upright_long_edge")
_BOOK_INITIAL_POSITION_ALIASES = {
    "fallen_long": "fallen_long",
    "fallen": "fallen_long",
    "lying": "fallen_long",
    "laying": "fallen_long",
    "倒下": "fallen_long",
    "倒着": "fallen_long",
    "fallen_cross": "fallen_cross",
    "horizontal": "fallen_cross",
    "sideways": "fallen_cross",
    "横着": "fallen_cross",
    "横向": "fallen_cross",
    "upright_tall": "upright_tall",
    "vertical": "upright_tall",
    "upright": "upright_tall",
    "standing": "upright_tall",
    "竖着": "upright_tall",
    "竖立": "upright_tall",
    "upright_long_edge": "upright_long_edge",
    "long_edge": "upright_long_edge",
    "edge": "upright_long_edge",
    "立在长边": "upright_long_edge",
}


def normalize_book_initial_position(value: str) -> str:
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    if normalized not in _BOOK_INITIAL_POSITION_ALIASES:
        choices = ", ".join(BOOK_INITIAL_POSITIONS)
        raise ValueError(f"Unknown initial_position={value!r}. Expected one of: {choices}")
    return _BOOK_INITIAL_POSITION_ALIASES[normalized]


def resolve_book_initial_position(
    *,
    initial_position: str | None,
    legacy_initial_position: str | None = None,
    legacy_name: str = "book_initial_position",
) -> str:
    if initial_position is None and legacy_initial_position is None:
        return "fallen_long"

    if initial_position is not None:
        position_name = normalize_book_initial_position(initial_position)
    else:
        position_name = normalize_book_initial_position(legacy_initial_position or "")

    if legacy_initial_position is not None:
        legacy_position_name = normalize_book_initial_position(legacy_initial_position)
        if legacy_position_name != position_name:
            raise ValueError(
                f"Conflicting initial positions: initial_position={initial_position!r}, "
                f"{legacy_name}={legacy_initial_position!r}"
            )

    return position_name


SURFACE_INITIAL_POSITIONS = ("ground", "low_table", "high_table")
_SURFACE_INITIAL_POSITION_ALIASES = {
    "floor": "ground",
    "ground": "ground",
    "on_ground": "ground",
    "地上": "ground",
    "低桌": "low_table",
    "低桌子": "low_table",
    "low": "low_table",
    "low_table": "low_table",
    "高桌": "high_table",
    "高桌子": "high_table",
    "high": "high_table",
    "high_table": "high_table",
    "table": "high_table",
}


def normalize_surface_initial_position(value: str) -> str:
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    if normalized not in _SURFACE_INITIAL_POSITION_ALIASES:
        choices = ", ".join(SURFACE_INITIAL_POSITIONS)
        raise ValueError(f"Unknown initial_position={value!r}. Expected one of: {choices}")
    return _SURFACE_INITIAL_POSITION_ALIASES[normalized]


def resolve_surface_scene_cfg(
    initial_position: str | None,
) -> tuple[str, TabletopSceneDRCfg | None]:
    surface = normalize_surface_initial_position(initial_position or "ground")
    if surface == "ground":
        return surface, None

    table_height = 0.35 if surface == "low_table" else 0.65
    return surface, TabletopSceneDRCfg(
        scene_mode="fixed",
        table_position=Box(low=[0.3, 0.0], high=[0.3, 0.0]),
        table_height=Box(low=table_height, high=table_height),
        room_choices=["hssd:scene0"],
        scene_manager="hssd",
    )


ROBOT_AROUND_OBJECT_INITIAL_POSITIONS = ("front", "back", "left", "right")
_ROBOT_AROUND_OBJECT_ALIASES = {
    "front": "front",
    "forward": "front",
    "前": "front",
    "前方": "front",
    "back": "back",
    "behind": "back",
    "后": "back",
    "后方": "back",
    "left": "left",
    "左": "left",
    "左方": "left",
    "right": "right",
    "右": "right",
    "右方": "right",
}


def normalize_robot_around_object_initial_position(value: str) -> str:
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    if normalized not in _ROBOT_AROUND_OBJECT_ALIASES:
        choices = ", ".join(ROBOT_AROUND_OBJECT_INITIAL_POSITIONS)
        raise ValueError(f"Unknown initial_position={value!r}. Expected one of: {choices}")
    return _ROBOT_AROUND_OBJECT_ALIASES[normalized]


def yaw_quaternion(yaw: float) -> list[float]:
    half_yaw = 0.5 * yaw
    return [math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw)]


def resolve_robot_around_object_spawn(
    *,
    initial_position: str | None,
    object_xy: tuple[float, float],
    distance: float,
    z: float = 0.0,
    face_target: bool = False,
) -> tuple[str, Box, Box | None]:
    position_name = normalize_robot_around_object_initial_position(initial_position or "front")
    object_x, object_y = object_xy
    if position_name == "front":
        robot_xy = (object_x - distance, object_y)
        yaw = 0.0
    elif position_name == "back":
        robot_xy = (object_x + distance, object_y)
        yaw = math.pi
    elif position_name == "left":
        robot_xy = (object_x, object_y + distance)
        yaw = -0.5 * math.pi
    else:
        robot_xy = (object_x, object_y - distance)
        yaw = 0.5 * math.pi

    robot_position = [robot_xy[0], robot_xy[1], z]
    robot_orientation = yaw_quaternion(yaw) if face_target else None
    return (
        position_name,
        Box(low=robot_position, high=list(robot_position)),
        Box(low=robot_orientation, high=list(robot_orientation))
        if robot_orientation is not None
        else None,
    )
