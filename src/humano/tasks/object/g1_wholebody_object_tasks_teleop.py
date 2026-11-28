from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np
import transforms3d as t3d

from humano.assets import AssetManager
from humano.core.actor import Actor, ObjectActor
from humano.dr import SpatialDRCfg
from humano.dr.types import Box
from humano.tasks.advanced.g1_wholebody_goal_placement_teleop import (
    _G1WholebodyTableObjectToReceptacleTaskTeleop,
)
from humano.tasks.basic.g1_wholebody_ground_apple_pick_teleop import (
    G1WholebodyGroundApplePickTaskTeleop,
)
from humano.tasks.basic.g1_wholebody_table_apple_pick_teleop import (
    G1WholebodyTableApplePickTaskTeleop,
)
from humano.tasks.goal._goal_base import (
    CABINET_INTERIOR_HIGH,
    CABINET_INTERIOR_LOW,
    CABINET_XY,
    actor_local_position,
    actor_points_in_body,
    configure_fixed_cabinet,
    contact_between,
    hand_contact,
)
from humano.tasks.registry import TaskRegistry


@dataclass(frozen=True)
class ObjectVariant:
    target_asset_id: str
    instruction: str
    container_asset_id: str | None = None


class _ObjectVariantMixin:
    """Select the operated asset for an episode through ``initial_condition``."""

    variants: dict[str, ObjectVariant]
    target_masses_kg: dict[str, float] = {}
    center_target_on_robot_y: frozenset[str] = frozenset()

    def __init__(self, *args, initial_condition: str = "random", **kwargs) -> None:
        choices = ("random", *self.variants)
        if initial_condition not in choices:
            raise ValueError(
                f"Invalid initial_condition {initial_condition!r}; expected one of {choices}."
            )
        self.initial_condition_mode = initial_condition
        self.selected_initial_condition: str | None = None
        self._variant_rng = np.random.default_rng()
        super().__init__(*args, **kwargs)

    @property
    def initial_conditions(self) -> tuple[str, ...]:
        return tuple(self.variants)

    def _select_variant(self, seed: int | None, options: dict[str, Any] | None) -> None:
        saved = None
        if options is not None and isinstance(options.get("state_dict"), dict):
            saved = options["state_dict"].get("initial_condition")
        if saved is not None:
            if saved not in self.variants:
                raise ValueError(f"Saved object variant {saved!r} is invalid for {self.uid}.")
            self.selected_initial_condition = str(saved)
        elif self.initial_condition_mode != "random":
            self.selected_initial_condition = self.initial_condition_mode
        else:
            if seed is not None:
                self._variant_rng = np.random.default_rng(seed)
            index = int(self._variant_rng.integers(0, len(self.variants)))
            self.selected_initial_condition = tuple(self.variants)[index]

    def _set_asset_randomizer(self, name: str, asset_id: str | None) -> None:
        if asset_id is None:
            return
        randomizer = self.dr.get_randomizer(name)
        if randomizer is None:
            raise RuntimeError(f"{self.uid} has no {name!r} randomizer for {asset_id!r}.")
        randomizer.cfg.asset_id = asset_id
        randomizer.res_id, randomizer.obj_id = asset_id.split(":", 1)

    def _apply_variant_after_reset(self, variant: ObjectVariant) -> None:
        del variant

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> None:
        self._select_variant(seed, options)
        assert self.selected_initial_condition is not None
        variant = self.variants[self.selected_initial_condition]
        self._set_asset_randomizer("target", variant.target_asset_id)
        self._set_asset_randomizer("container", variant.container_asset_id)
        language = self.dr.get_randomizer("language")
        if language is not None:
            language.cfg.instructions = [variant.instruction]
        super().reset(seed, options)
        if isinstance(getattr(type(self), "instruction", None), str):
            self.instruction = variant.instruction
        self._apply_variant_after_reset(variant)
        target = self.layout.actors.get("target")
        if self.selected_initial_condition in self.center_target_on_robot_y:
            robot = self.layout.actors["robot"]
            if isinstance(target, ObjectActor):
                target.pose.position[1] = float(robot.pose.position[1])
        target_mass = self.target_masses_kg.get(self.selected_initial_condition)
        if target_mass is not None and isinstance(target, ObjectActor):
            target.mujoco_mass = target_mass

    def state_dict(self) -> dict[str, Any]:
        state = super().state_dict()
        state["initial_condition"] = self.selected_initial_condition
        return state

    def preload_objects(self) -> list[Actor]:
        objects = AssetManager.get("objects")
        load_ids: list[str] = []
        for variant in self.variants.values():
            for asset_id in (variant.target_asset_id, variant.container_asset_id):
                if asset_id is not None and asset_id.startswith("objects:"):
                    load_id = asset_id.split(":", 1)[1]
                    if load_id not in load_ids:
                        load_ids.append(load_id)
        return [ObjectActor(asset=objects.load(load_id)) for load_id in load_ids]


APPLE = "objects:apple/apple_001"
BANANA = "objects:banana/banana_020"
CAKE = "objects:cake/cake_001"
PLATE = "objects:plate/plate_3"
TRAY = "objects:tray/tray_001"
BOTTLE = "objects:bottle/bottle_048"
BASKET = "objects:picnic_basket/picnic_basket_009"
BOX = "objects:box/box_007"
OBJECT_PICK_STORAGE_BOX = "objects:storage_box/storage_box_003_height2"
MILK = "objects:Carton_Toy_3C923930_Milk/Carton_Toy_3C923930_Milk_1"
BREAD = "objects:bread/bread_006"
CABINET = "objects:white_cabinet/white_cabinet_1"
TRASH_CAN = "objects:dustbin/dustbin_011"


_TABLE_SMALL_VARIANTS = {
    "apple": ObjectVariant(APPLE, "pick up the apple from the table with one hand"),
    "banana": ObjectVariant(BANANA, "pick up the banana from the table with one hand"),
    "cake": ObjectVariant(CAKE, "pick up the cake from the table with one hand"),
}

_TABLE_LARGE_VARIANTS = {
    "plate": ObjectVariant(PLATE, "grasp the plate with both hands and lift it from the table"),
    "tray": ObjectVariant(TRAY, "grasp the tray with both hands and lift it from the table"),
}

_TABLE_MIXED_VARIANTS = {
    "apple": ObjectVariant(APPLE, "grasp the apple from above with one hand and lift it"),
    "banana": ObjectVariant(BANANA, "grasp the banana around its side with one hand and lift it"),
    "plate": ObjectVariant(PLATE, "grasp opposite sides of the plate with both hands and lift it"),
    "tray": ObjectVariant(TRAY, "grasp opposite sides of the tray with both hands and lift it"),
}

_GROUND_SMALL_VARIANTS = {
    "bottle": ObjectVariant(BOTTLE, "pick up the bottle from the ground with one hand"),
    "basket": ObjectVariant(BASKET, "pick up the basket from the ground with one hand"),
}

_GROUND_LARGE_VARIANTS = {
    "box": ObjectVariant(BOX, "grasp the box with both hands and lift it from the ground"),
    "storage_box": ObjectVariant(
        OBJECT_PICK_STORAGE_BOX,
        "grasp the storage box with both hands and lift it from the ground",
    ),
}

_GROUND_MIXED_VARIANTS = {
    "bottle": ObjectVariant(BOTTLE, "grasp the bottle around its side with one hand and lift it"),
    "basket": ObjectVariant(BASKET, "grasp the basket handle with one hand and lift it"),
    "box": _GROUND_LARGE_VARIANTS["box"],
    "storage_box": _GROUND_LARGE_VARIANTS["storage_box"],
}

_TABLE_LARGE_MASSES_KG = {"plate": 0.8, "tray": 1.0}
_GROUND_SMALL_MASSES_KG = {"basket": 0.5}
_GROUND_LARGE_MASSES_KG = {"box": 2.5, "storage_box": 1.0}
_PLATE_CABINET_MIN_INSIDE_FRACTION = 0.70
_TRAY_CABINET_MIN_INSIDE_FRACTION = 0.60
_TRAY_CABINET_INTERIOR_LOW = CABINET_INTERIOR_LOW - np.array([0.0, 0.0, 0.02], dtype=np.float64)


@TaskRegistry.register("g1_wholebody_table_small_object_pick_teleop")
class G1WholebodyTableSmallObjectPickTaskTeleop(
    _ObjectVariantMixin, G1WholebodyTableApplePickTaskTeleop
):
    uid = "g1_wholebody_table_small_object_pick_teleop"
    label = "G1 TELEOP Table Small Object Pick"
    description = (
        "A task where the G1 robot must pick up a selected small table object with one hand."
    )
    variants = _TABLE_SMALL_VARIANTS
    dr_cfgs = deepcopy(G1WholebodyTableApplePickTaskTeleop.dr_cfgs)


@TaskRegistry.register("g1_wholebody_table_large_object_pick_teleop")
class G1WholebodyTableLargeObjectPickTaskTeleop(
    _ObjectVariantMixin, G1WholebodyTableApplePickTaskTeleop
):
    uid = "g1_wholebody_table_large_object_pick_teleop"
    label = "G1 TELEOP Table Large Object Pick"
    description = (
        "A task where the G1 robot must pick up a selected large table object with both hands."
    )
    variants = _TABLE_LARGE_VARIANTS
    target_masses_kg = _TABLE_LARGE_MASSES_KG
    dr_cfgs = deepcopy(G1WholebodyTableApplePickTaskTeleop.dr_cfgs)


@TaskRegistry.register("g1_wholebody_table_object_pick_teleop")
class G1WholebodyTableObjectPickTaskTeleop(
    _ObjectVariantMixin, G1WholebodyTableApplePickTaskTeleop
):
    uid = "g1_wholebody_table_object_pick_teleop"
    label = "G1 TELEOP Table Object Pick"
    description = (
        "A task where the G1 robot must pick up a selected table object using its instructed grasp."
    )
    variants = _TABLE_MIXED_VARIANTS
    target_masses_kg = _TABLE_LARGE_MASSES_KG
    dr_cfgs = deepcopy(G1WholebodyTableApplePickTaskTeleop.dr_cfgs)


@TaskRegistry.register("g1_wholebody_ground_small_object_pick_teleop")
class G1WholebodyGroundSmallObjectPickTaskTeleop(
    _ObjectVariantMixin, G1WholebodyGroundApplePickTaskTeleop
):
    uid = "g1_wholebody_ground_small_object_pick_teleop"
    label = "G1 TELEOP Ground Small Object Pick"
    description = "A task where the G1 robot must pick up a selected ground object with one hand."
    variants = _GROUND_SMALL_VARIANTS
    target_masses_kg = _GROUND_SMALL_MASSES_KG
    dr_cfgs = deepcopy(G1WholebodyGroundApplePickTaskTeleop.dr_cfgs)


class _StorageBoxPickMixin:
    storage_box_hold_s = 0.5
    storage_box_clearance_m = 0.02

    @property
    def requires_success_at_token_end(self) -> bool:
        return self.selected_initial_condition == "storage_box"

    def reset(self, seed=None, options=None) -> None:
        self._storage_box_hold_started_at = None
        self._storage_box_next_diagnostic_at = 0.0
        super().reset(seed, options)

    def _storage_box_both_hands_contact(self, env) -> bool:
        model = env.mjModel
        target_id = model.body(self.target.simulation_name).id
        spec = self.robot.hand_spec
        # Dex3 palm geoms belong to wrist_yaw bodies in the SONIC MJCF.
        hand_links = (
            ("left_wrist_yaw_link", "right_wrist_yaw_link")
            if spec.name == "dex3"
            else (spec.left_ee_link, spec.right_ee_link)
        )
        hand_ids = {model.body(link).id for link in hand_links}
        touching = set()
        for contact in env.mjData.contact:
            if contact.dist > 0:
                continue
            bodies = [int(model.geom_bodyid[g]) for g in (contact.geom1, contact.geom2)]
            if target_id not in bodies:
                continue
            body = bodies[1] if bodies[0] == target_id else bodies[0]
            while body:
                if body in hand_ids:
                    touching.add(body)
                    break
                body = int(model.body_parentid[body])
        return touching == hand_ids

    def check_success(self, info, *, mujoco_env, **kwargs) -> bool:
        lifted = super().check_success(info, mujoco_env=mujoco_env, **kwargs)
        if not self.requires_success_at_token_end:
            return lifted
        points = actor_points_in_body(mujoco_env, self.target, "world")
        clearance = None if points is None else float(points[:, 2].min())
        both_hands = self._storage_box_both_hands_contact(mujoco_env)
        now = float(mujoco_env.mjData.time)
        if now >= self._storage_box_next_diagnostic_at:
            held = (
                0.0 if self._storage_box_hold_started_at is None
                else now - self._storage_box_hold_started_at
            )
            print(
                f"[StorageBox] lifted={lifted} clearance_m={clearance} "
                f"both_hands={both_hands} held_s={held:.3f}",
                flush=True,
            )
            self._storage_box_next_diagnostic_at = now + 1.0
        if (
            not lifted
            or clearance is None
            or clearance < self.storage_box_clearance_m
            or not both_hands
        ):
            self._storage_box_hold_started_at = None
            return False
        now = float(mujoco_env.mjData.time)
        if self._storage_box_hold_started_at is None:
            self._storage_box_hold_started_at = now
        return now - self._storage_box_hold_started_at + 1e-9 >= self.storage_box_hold_s


@TaskRegistry.register("g1_wholebody_ground_large_object_pick_teleop")
class G1WholebodyGroundLargeObjectPickTaskTeleop(
    _StorageBoxPickMixin, _ObjectVariantMixin, G1WholebodyGroundApplePickTaskTeleop
):
    uid = "g1_wholebody_ground_large_object_pick_teleop"
    label = "G1 TELEOP Ground Large Object Pick"
    description = (
        "A task where the G1 robot must pick up a selected large ground object with both hands."
    )
    variants = _GROUND_LARGE_VARIANTS
    target_masses_kg = _GROUND_LARGE_MASSES_KG
    center_target_on_robot_y = frozenset({"storage_box"})
    dr_cfgs = deepcopy(G1WholebodyGroundApplePickTaskTeleop.dr_cfgs)


@TaskRegistry.register("g1_wholebody_ground_object_pick_teleop")
class G1WholebodyGroundObjectPickTaskTeleop(
    _StorageBoxPickMixin, _ObjectVariantMixin, G1WholebodyGroundApplePickTaskTeleop
):
    uid = "g1_wholebody_ground_object_pick_teleop"
    label = "G1 TELEOP Ground Object Pick"
    description = "A task where the G1 robot must pick up a selected ground object using its instructed grasp."
    variants = _GROUND_MIXED_VARIANTS
    target_masses_kg = {**_GROUND_SMALL_MASSES_KG, **_GROUND_LARGE_MASSES_KG}
    center_target_on_robot_y = frozenset({"storage_box"})
    dr_cfgs = deepcopy(G1WholebodyGroundApplePickTaskTeleop.dr_cfgs)


class _ObjectPlacementTask(_ObjectVariantMixin, _G1WholebodyTableObjectToReceptacleTaskTeleop):
    pass


@TaskRegistry.register("g1_wholebody_table_small_object_to_basket_teleop")
class G1WholebodyTableSmallObjectToBasketTaskTeleop(_ObjectPlacementTask):
    uid = "g1_wholebody_table_small_object_to_basket_teleop"
    label = "G1 TELEOP Table Small Object To Basket"
    description = "A task where the G1 robot must put a selected small table object into a basket with one hand."
    target_asset_id = APPLE
    target_load_id = "apple/apple_001"
    target_label = "small_object"
    container_asset_id = BASKET
    container_load_id = "picnic_basket/picnic_basket_009"
    container_label = "basket"
    receptacle_kind = "basket"
    instruction = "put the apple into the basket with one hand"
    variants = {
        "apple": ObjectVariant(APPLE, instruction, BASKET),
        "banana": ObjectVariant(BANANA, "put the banana into the basket with one hand", BASKET),
        "cake": ObjectVariant(CAKE, "put the cake into the basket with one hand", BASKET),
    }


@TaskRegistry.register("g1_wholebody_table_large_object_to_cabinet_teleop")
class G1WholebodyTableLargeObjectToCabinetTaskTeleop(_ObjectPlacementTask):
    uid = "g1_wholebody_table_large_object_to_cabinet_teleop"
    label = "G1 TELEOP Table Large Object To Cabinet"
    description = "A task where the G1 robot must put a selected large table object fully inside a cabinet with both hands."
    target_asset_id = PLATE
    target_load_id = "plate/plate_3"
    target_label = "large_object"
    container_asset_id = CABINET
    container_load_id = "white_cabinet/white_cabinet_1"
    container_label = "cabinet"
    receptacle_kind = "tray"
    instruction = "put the plate upright and fully inside the cabinet with both hands"
    cabinet_xy = CABINET_XY
    target_xy = (-0.275, -0.25)
    variants = {
        "plate": ObjectVariant(PLATE, instruction, CABINET),
        "tray": ObjectVariant(
            TRAY,
            "put the tray upright and fully inside the cabinet with both hands",
            CABINET,
        ),
    }
    target_masses_kg = _TABLE_LARGE_MASSES_KG

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        spatial = self.dr.get_randomizer("spatial")
        if spatial is None:
            raise RuntimeError(f"{self.uid} has no spatial randomizer.")
        assert isinstance(spatial.cfg, SpatialDRCfg)
        spatial.cfg.target_region = Box(low=self.target_xy, high=self.target_xy)
        spatial.cfg.container_region = Box(low=self.cabinet_xy, high=self.cabinet_xy)
        self._container_xy = list(self.cabinet_xy)

    def _apply_variant_after_reset(self, variant: ObjectVariant) -> None:
        del variant
        configure_fixed_cabinet(self.container, xy=self.cabinet_xy)

    def check_object_in_container(
        self, info: dict[str, Any] | None = None, *args, **kwargs
    ) -> bool:
        del info, args
        mujoco_env = kwargs.get("mujoco_env")
        if mujoco_env is None or not isinstance(self.target, ObjectActor):
            return False
        points = actor_points_in_body(mujoco_env, self.target, self.container.simulation_name)
        geometry_inside = False
        if points is not None and len(points):
            if self.selected_initial_condition == "plate":
                coordinate_inside = (points >= CABINET_INTERIOR_LOW) & (
                    points <= CABINET_INTERIOR_HIGH
                )
                point_inside = np.all(coordinate_inside, axis=1)
                center = actor_local_position(
                    mujoco_env,
                    self.target,
                    self.container.simulation_name,
                )
                geometry_inside = bool(
                    center is not None
                    and np.all(center >= CABINET_INTERIOR_LOW)
                    and np.all(center <= CABINET_INTERIOR_HIGH)
                    and np.mean(point_inside) >= _PLATE_CABINET_MIN_INSIDE_FRACTION
                )
            else:
                # Convex decomposition gives the tray a highly nonuniform vertex
                # distribution. Measure its projected overlap instead so the
                # percentage represents physical extent on each cabinet axis.
                point_low = np.min(points, axis=0)
                point_high = np.max(points, axis=0)
                extent = point_high - point_low
                overlap = np.maximum(
                    0.0,
                    np.minimum(point_high, CABINET_INTERIOR_HIGH)
                    - np.maximum(point_low, _TRAY_CABINET_INTERIOR_LOW),
                )
                axis_inside_fraction = np.ones(3, dtype=np.float64)
                nonzero_extent = extent > np.finfo(np.float64).eps
                axis_inside_fraction[nonzero_extent] = (
                    overlap[nonzero_extent] / extent[nonzero_extent]
                )
                axis_inside_fraction[~nonzero_extent] = (
                    point_low[~nonzero_extent] >= _TRAY_CABINET_INTERIOR_LOW[~nonzero_extent]
                ) & (point_high[~nonzero_extent] <= CABINET_INTERIOR_HIGH[~nonzero_extent])
                geometry_inside = bool(
                    np.all(axis_inside_fraction >= _TRAY_CABINET_MIN_INSIDE_FRACTION)
                )
        return bool(
            geometry_inside
            and contact_between(
                mujoco_env,
                self.target.simulation_name,
                self.container.simulation_name,
            )
            and not hand_contact(mujoco_env, self.target)
        )

    def decompose(self):
        steps = super().decompose()
        cabinet_rotation = t3d.quaternions.quat2mat(self.container.pose.quaternion)
        branch = self.selected_initial_condition or "plate"
        local_position = np.array(
            [0.0, -0.04, 0.167 if branch == "plate" else 0.153],
            dtype=np.float64,
        )
        relative_rotation = (
            np.eye(3) if branch == "plate" else t3d.euler.euler2mat(0.5 * np.pi, 0.0, 0.0)
        )
        for step in steps:
            if step.phase == "move_to_cabinet":
                step.meta["position"] = (
                    np.asarray(self.container.pose.position, dtype=np.float64)
                    + cabinet_rotation @ local_position
                )
                step.meta["orientation"] = t3d.quaternions.mat2quat(
                    cabinet_rotation @ relative_rotation
                )
        return steps


@TaskRegistry.register("g1_wholebody_table_object_to_trash_can_teleop")
class G1WholebodyTableObjectToTrashCanTaskTeleop(_ObjectPlacementTask):
    uid = "g1_wholebody_table_object_to_trash_can_teleop"
    label = "G1 TELEOP Table Object To Trash Can"
    description = "A task where the G1 robot must throw a selected table object into a trash can."
    target_asset_id = MILK
    target_load_id = "Carton_Toy_3C923930_Milk/Carton_Toy_3C923930_Milk_1"
    target_label = "object"
    container_asset_id = TRASH_CAN
    container_load_id = "dustbin/dustbin_011"
    container_label = "trash_can"
    receptacle_kind = "trash_can"
    instruction = "throw the milk carton from the table into the trash can"
    container_on_ground = True
    variants = {
        "milk": ObjectVariant(MILK, instruction, TRASH_CAN),
        "bread": ObjectVariant(
            BREAD, "throw the bread from the table into the trash can", TRASH_CAN
        ),
        "cake": ObjectVariant(CAKE, "throw the cake from the table into the trash can", TRASH_CAN),
    }


@TaskRegistry.register("g1_wholebody_food_to_tray_teleop")
class G1WholebodyFoodToTrayTaskTeleop(_ObjectPlacementTask):
    uid = "g1_wholebody_food_to_tray_teleop"
    label = "G1 TELEOP Food To Tray"
    description = "A task where the G1 robot must put a selected food item into a tray."
    target_asset_id = BREAD
    target_load_id = "bread/bread_006"
    target_label = "food"
    container_asset_id = TRAY
    container_load_id = "tray/tray_001"
    container_label = "tray"
    receptacle_kind = "tray"
    instruction = "put the bread into the tray"
    variants = {
        "bread": ObjectVariant(BREAD, instruction, TRAY),
        "cake": ObjectVariant(CAKE, "put the cake into the tray", TRAY),
    }


OBJECT_TASK_TYPES = (
    G1WholebodyTableSmallObjectPickTaskTeleop,
    G1WholebodyTableLargeObjectPickTaskTeleop,
    G1WholebodyTableObjectPickTaskTeleop,
    G1WholebodyGroundSmallObjectPickTaskTeleop,
    G1WholebodyGroundLargeObjectPickTaskTeleop,
    G1WholebodyGroundObjectPickTaskTeleop,
    G1WholebodyTableSmallObjectToBasketTaskTeleop,
    G1WholebodyTableLargeObjectToCabinetTaskTeleop,
    G1WholebodyTableObjectToTrashCanTaskTeleop,
    G1WholebodyFoodToTrayTaskTeleop,
)
