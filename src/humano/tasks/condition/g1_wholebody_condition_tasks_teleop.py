from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import transforms3d as t3d
import trimesh

from humano.assets import AssetManager
from humano.core.actor import Actor, ObjectActor
from humano.dr import (
    ArticulatedObjectDrCfg,
    DistractorDRCfg,
    LanguageDRCfg,
    SpatialDRCfg,
    TabletopSceneDRCfg,
    TargetDRCfg,
)
from humano.dr.types import Box
from humano.tasks.basic.g1_wholebody_chair_push_teleop import (
    _PUSH_CHAIR_MUJOCO_MASS_KG,
)
from humano.tasks.simple.g1_wholebody_xmove_pick_teleop import G1WholebodyXMovePickTaskTeleop
from humano.tasks.long.g1_wholebody_long_tasks_teleop import (
    G1WholebodyLongGroundBasketToTableAppleToBasketTaskTeleop,
)
from humano.tasks.registry import TaskRegistry

_APPLE = "objects:apple/apple_001"
_APPLE_LOAD = "apple/apple_001"
_BANANA_LOAD = "banana/banana_020"
_BASKET = "objects:picnic_basket/picnic_basket_009"
_BASKET_LOAD = "picnic_basket/picnic_basket_009"
_BOTTLE = "objects:bottle/bottle_048"
_BOTTLE_LOAD = "bottle/bottle_048"
_STORAGE_BOX = "objects:storage_box/storage_box_003"
_STORAGE_BOX_LOAD = "storage_box/storage_box_003"
_BREAD = "objects:bread/bread_006"
_BREAD_LOAD = "bread/bread_006"
_CAKE_LOAD = "cake/cake_001"
_CHAIR = "objects:chair/chair_319_0p9"
_CHAIR_LOAD = "chair/chair_319_0p9"
_ERASER = "objects:eraser/eraser_002"
_ERASER_LOAD = "eraser/eraser_002"
_PLATE = "objects:plate/plate_3"
_PLATE_LOAD = "plate/plate_3"
_TRAY_LOAD = "tray/tray_001"

_DRAWER = "articulated:8"
_DRAWER_JOINT = "articulate_joint_0"
_DRAWER_BODY = "E_drawer_59"
_DRAWER_CLOSED = 0.0
_DRAWER_OPEN = -0.20
_DRAWER_OPEN_THRESHOLD = -0.12
_DRAWER_LOW = np.array([-0.17, 0.025, -0.13], dtype=np.float64)
_DRAWER_HIGH = np.array([0.17, 0.335, 0.06], dtype=np.float64)
_DRAWER_BOTTOM_Z = (-0.13, -0.085)

_PENCIL_CASE = "articulated:9"
_PENCIL_CASE_JOINT = "articulate_joint_0"
_PENCIL_CASE_BASE = "009_base"
_PENCIL_CASE_CLOSED = 0.0
_PENCIL_CASE_OPEN = -np.deg2rad(100.0)
_PENCIL_CASE_OPEN_THRESHOLD = -np.deg2rad(70.0)
_PENCIL_CASE_LOW = np.array([-0.096, -0.023, -0.004], dtype=np.float64)
_PENCIL_CASE_HIGH = np.array([0.086, 0.023, 0.021], dtype=np.float64)

_DOLLY = "articulated:7"
_C9_TABLE_XY = np.array([1.15, 0.0], dtype=np.float64)
_C9_CHAIR_XY = np.array([-0.68, 0.0], dtype=np.float64)
# Keep the dolly in the chair-to-table lane for ``block`` while leaving a
# visible clearance from the chair's initial footprint.
_C9_DOLLY_BLOCK_XY = np.array([0.15, 0.0], dtype=np.float64)
_C9_DOLLY_CLEAR_XY = np.array([0.15, 1.10], dtype=np.float64)
_C9_CHAIR_YAW = 0.5 * np.pi
_C9_DOLLY_YAW = 0.0

_HIGH_TABLE = 0.65
_LOW_TABLE = 0.35
_HOLD_STEPS = 10
_SIT_HOLD_STEPS = 30

_STORAGE_BOX_HALF_XY = np.array([0.215, 0.16], dtype=np.float64)
# The bottle is taller than the storage box, so its center remains slightly
# above the rim even when its base is resting correctly inside the box.
_STORAGE_BOX_BOTTLE_CENTER_Z = (-0.005, 0.22)
_BASKET_HALF_XY = np.array([0.14, 0.11], dtype=np.float64)
_BASKET_Z = (-0.08, 0.18)
_TRAY_HALF_XY = np.array([0.15, 0.11], dtype=np.float64)
_TRAY_Z = (0.0, 0.10)
_PLATE_RADIUS = 0.18
_PLATE_Z = (-0.01, 0.08)

_C8_BREAD_XY = np.array([-0.28, -0.20], dtype=np.float64)
_C8_PLATE_XY = np.array([-0.04, 0.20], dtype=np.float64)
_C8_TRAY_XY = np.array([0.10, -0.20], dtype=np.float64)
_C8_CAKE_IN_TRAY_LOCAL = np.array([0.0, 0.0, 0.03], dtype=np.float64)


def _fixed_scene(height: float, table_xy: tuple[float, float] = (0.3, 0.0)):
    return TabletopSceneDRCfg(
        scene_mode="fixed",
        table_position=Box(low=list(table_xy), high=list(table_xy)),
        table_height=Box(low=height, high=height),
        room_choices=["hssd:scene0"],
        scene_manager="hssd",
    )


def _body_pose(mujoco_env, body_name: str) -> tuple[np.ndarray, np.ndarray] | None:
    if mujoco_env is None:
        return None
    try:
        body = mujoco_env.mjData.body(body_name)
    except KeyError:
        return None
    return (
        np.asarray(body.xpos, dtype=np.float64),
        np.asarray(body.xmat, dtype=np.float64).reshape(3, 3),
    )


def _body_name(model, geom_id: int) -> str:
    geom = model.geom(int(geom_id))
    return model.body(int(np.asarray(geom.bodyid).item())).name


def _contact_between(mujoco_env, name_a: str, name_b: str) -> bool:
    if mujoco_env is None:
        return False
    model = mujoco_env.mjModel
    data = mujoco_env.mjData
    for contact_id in range(data.ncon):
        contact = data.contact[contact_id]
        body1 = _body_name(model, contact.geom1)
        body2 = _body_name(model, contact.geom2)
        if (name_a in body1 and name_b in body2) or (name_a in body2 and name_b in body1):
            return True
    return False


def _hand_contact(mujoco_env, actor: ObjectActor) -> bool:
    return _contact_between(mujoco_env, actor.simulation_name, "hand")


@lru_cache(maxsize=32)
def _collision_points(asset_uid: str) -> np.ndarray:
    asset = AssetManager.get("objects").load(asset_uid)
    points: list[np.ndarray] = []
    for mesh_path in asset.collision_meshes_mujoco:
        mesh = trimesh.load_mesh(Path(mesh_path), force="mesh", process=False)
        if not mesh.is_empty and len(mesh.vertices):
            points.append(np.asarray(mesh.vertices, dtype=np.float64))
    if not points:
        raise RuntimeError(f"No collision vertices found for objects:{asset_uid}.")
    return np.vstack(points)


def _points_in_body_local(
    mujoco_env,
    actor: ObjectActor,
    container_body: str,
) -> np.ndarray | None:
    actor_pose = _body_pose(mujoco_env, actor.simulation_name)
    container_pose = _body_pose(mujoco_env, container_body)
    if actor_pose is None or container_pose is None:
        return None
    actor_position, actor_rotation = actor_pose
    container_position, container_rotation = container_pose
    local_points = _collision_points(str(actor.asset.uid))
    world_points = local_points @ actor_rotation.T + actor_position
    return (world_points - container_position) @ container_rotation


def _center_in_actor_local(
    mujoco_env,
    actor: ObjectActor,
    container: ObjectActor,
) -> np.ndarray | None:
    actor_pose = _body_pose(mujoco_env, actor.simulation_name)
    container_pose = _body_pose(mujoco_env, container.simulation_name)
    if actor_pose is None or container_pose is None:
        return None
    actor_position, _ = actor_pose
    container_position, container_rotation = container_pose
    return container_rotation.T @ (actor_position - container_position)


def _stable_pose(actor: ObjectActor) -> tuple[float, list[float]]:
    pose = np.asarray(actor.asset.stable_poses[0], dtype=np.float64)
    return float(pose[2]), [float(v) for v in pose[3:7]]


def _place_on_surface(actor: ObjectActor, xy: tuple[float, float], surface_height: float) -> None:
    stable_z, quaternion = _stable_pose(actor)
    actor.pose.position = [float(xy[0]), float(xy[1]), surface_height + stable_z]
    actor.pose.quaternion = quaternion


def _move_on_surface_preserve_orientation(
    actor: ObjectActor,
    xy: tuple[float, float],
    surface_height: float,
) -> None:
    stable_z, _ = _stable_pose(actor)
    actor.pose.position = [float(xy[0]), float(xy[1]), surface_height + stable_z]


class _InitialConditionMixin:
    initial_conditions: tuple[str, str]

    def __init__(self, *args, initial_condition: str = "random", **kwargs) -> None:
        allowed = ("random", *self.initial_conditions)
        if initial_condition not in allowed:
            raise ValueError(
                f"Invalid initial_condition {initial_condition!r}; expected one of {allowed}."
            )
        self.initial_condition_mode = initial_condition
        self.selected_initial_condition: str | None = None
        self._condition_rng = np.random.default_rng()
        super().__init__(*args, **kwargs)

    def _select_initial_condition(
        self,
        seed: int | None,
        options: dict[str, Any] | None,
    ) -> None:
        saved = None
        if options is not None and isinstance(options.get("state_dict"), dict):
            saved = options["state_dict"].get("initial_condition")
        if saved is not None:
            if saved not in self.initial_conditions:
                raise ValueError(f"Saved initial condition {saved!r} is invalid for {self.uid}.")
            self.selected_initial_condition = str(saved)
            return
        if self.initial_condition_mode != "random":
            self.selected_initial_condition = self.initial_condition_mode
            return
        if seed is not None:
            self._condition_rng = np.random.default_rng(seed)
        index = int(self._condition_rng.integers(0, len(self.initial_conditions)))
        self.selected_initial_condition = self.initial_conditions[index]

    def _configure_condition_before_reset(self) -> None:
        return None

    def _configure_condition_after_reset(self) -> None:
        return None

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> None:
        self._select_initial_condition(seed, options)
        self._configure_condition_before_reset()
        super().reset(seed, options)
        self._configure_condition_after_reset()

    def state_dict(self) -> dict[str, Any]:
        state = super().state_dict()
        state["initial_condition"] = self.selected_initial_condition
        return state


class _ConditionObjectToDrawerTask(_InitialConditionMixin, G1WholebodyXMovePickTaskTeleop):
    initial_conditions = ("closed", "open")
    target_asset_id: str
    target_load_id: str

    def __init__(self, *args, target: str | None = None, hold_steps: int = _HOLD_STEPS, **kwargs):
        self.dr_cfgs = deepcopy(G1WholebodyXMovePickTaskTeleop.dr_cfgs)
        self.dr_cfgs["language"] = LanguageDRCfg(instructions=[self.default_instruction])
        self.dr_cfgs["target"] = TargetDRCfg(asset_id=self.target_asset_id)
        self.dr_cfgs["articulated"] = ArticulatedObjectDrCfg(
            asset_id=_DRAWER,
            articulate_init_joint_qpos={_DRAWER_JOINT: _DRAWER_CLOSED},
        )
        self.dr_cfgs["distractors"] = DistractorDRCfg(
            res_id="objects", number_of_distractors=0, allow_duplicates=False
        )
        self.dr_cfgs["spatial"] = SpatialDRCfg(
            spatial_mode="fixed",
            robot_region=Box(low=[-0.95, 0.0, 0.0], high=[-0.95, 0.0, 0.0]),
            target_region=Box(low=[-0.275, -0.18], high=[-0.275, -0.18]),
            distractors_region=Box(low=[0.0, 0.0], high=[0.0, 0.0]),
            articulated_region=Box(low=[-0.08, 0.18, 0.651], high=[-0.08, 0.18, 0.651]),
            target_stable_indices=[0],
            target_rotate_z=Box(low=0.0, high=0.0),
            articulated_rotate_z=Box(low=-0.5 * np.pi, high=-0.5 * np.pi),
            fixed_stable_pose_idx=0,
            obj_surface_map={"target": "table"},
        )
        self.dr_cfgs["scene"] = _fixed_scene(_HIGH_TABLE)
        self.hold_steps = int(hold_steps)
        self._success_hold_count = 0
        super().__init__(*args, target=target, **kwargs)

    def _configure_condition_before_reset(self) -> None:
        qpos = _DRAWER_CLOSED if self.selected_initial_condition == "closed" else _DRAWER_OPEN
        cfg = self.dr.get_randomizer("articulated").cfg
        cfg.articulate_init_joint_qpos = {_DRAWER_JOINT: qpos}
        self.dr_cfgs["articulated"].articulate_init_joint_qpos = {_DRAWER_JOINT: qpos}

    def _configure_condition_after_reset(self) -> None:
        self._success_hold_count = 0
        self.reward = 0.0

    def _contacts_drawer_bottom(self, mujoco_env) -> bool:
        drawer_pose = _body_pose(mujoco_env, _DRAWER_BODY)
        if drawer_pose is None:
            return False
        drawer_position, drawer_rotation = drawer_pose
        model = mujoco_env.mjModel
        data = mujoco_env.mjData
        for contact_id in range(data.ncon):
            contact = data.contact[contact_id]
            body1 = _body_name(model, contact.geom1)
            body2 = _body_name(model, contact.geom2)
            if not (
                self.target.simulation_name in body1
                and _DRAWER_BODY in body2
                or self.target.simulation_name in body2
                and _DRAWER_BODY in body1
            ):
                continue
            local = (np.asarray(contact.pos, dtype=np.float64) - drawer_position) @ drawer_rotation
            if (
                _DRAWER_LOW[0] <= local[0] <= _DRAWER_HIGH[0]
                and _DRAWER_LOW[1] <= local[1] <= _DRAWER_HIGH[1]
                and _DRAWER_BOTTOM_Z[0] <= local[2] <= _DRAWER_BOTTOM_Z[1]
            ):
                return True
        return False

    def check_object_in_container(self, info=None, *args, **kwargs) -> bool:
        del info, args
        mujoco_env = kwargs.get("mujoco_env")
        if mujoco_env is None:
            return False
        points = _points_in_body_local(mujoco_env, self.target, _DRAWER_BODY)
        drawer_qpos = float(mujoco_env.mjData.joint(_DRAWER_JOINT).qpos[0])
        return bool(
            drawer_qpos <= _DRAWER_OPEN_THRESHOLD
            and points is not None
            and np.all(points >= _DRAWER_LOW)
            and np.all(points <= _DRAWER_HIGH)
            and self._contacts_drawer_bottom(mujoco_env)
            and not _hand_contact(mujoco_env, self.target)
        )

    def compute_reward(self, info: dict[str, Any], *args, **kwargs) -> float:
        if self.check_object_in_container(info, *args, **kwargs):
            self._success_hold_count += 1
        else:
            self._success_hold_count = 0
        self.reward = 0.1 * self._success_hold_count
        return float(self.reward)

    def check_success(self, info: dict[str, Any], *args, **kwargs) -> bool:
        del info, args, kwargs
        return self._success_hold_count >= self.hold_steps

    def preload_objects(self) -> list[Actor]:
        return [ObjectActor(asset=AssetManager.get("objects").load(self.target_load_id))]


@TaskRegistry.register("g1_wholebody_condition_plate_to_drawer_teleop")
class G1WholebodyConditionPlateToDrawerTaskTeleop(_ConditionObjectToDrawerTask):
    uid = "g1_wholebody_condition_plate_to_drawer_teleop"
    label = "G1 TELEOP Condition Plate To Drawer"
    description = "A task where the G1 robot must put the plate into the drawer, opening it first if it is closed."
    default_instruction = "if the drawer is closed, open it and put the plate inside; if it is open, put the plate inside directly"
    target_asset_id = _PLATE
    target_load_id = _PLATE_LOAD


@TaskRegistry.register("g1_wholebody_condition_apple_to_drawer_teleop")
class G1WholebodyConditionAppleToDrawerTaskTeleop(_ConditionObjectToDrawerTask):
    uid = "g1_wholebody_condition_apple_to_drawer_teleop"
    label = "G1 TELEOP Condition Apple To Drawer"
    description = "A task where the G1 robot must put the apple into the drawer, opening it first if it is closed."
    default_instruction = "if the drawer is closed, open it and put the apple inside; if it is open, put the apple inside directly"
    target_asset_id = _APPLE
    target_load_id = _APPLE_LOAD


class _ConditionChooseReceptacleTask(_InitialConditionMixin, G1WholebodyXMovePickTaskTeleop):
    target_asset_id: str
    target_load_id: str
    receptacle_asset_id: str
    receptacle_load_id: str
    occupant_load_id: str
    left_instance_id: str
    right_instance_id: str
    inside_half_xy: np.ndarray
    inside_z: tuple[float, float]
    occupant_local_z: float
    full_containment: bool = False

    def __init__(self, *args, target: str | None = None, hold_steps: int = _HOLD_STEPS, **kwargs):
        self.dr_cfgs = deepcopy(G1WholebodyXMovePickTaskTeleop.dr_cfgs)
        self.dr_cfgs["language"] = LanguageDRCfg(instructions=[self.default_instruction])
        self.dr_cfgs["target"] = TargetDRCfg(asset_id=self.target_asset_id)
        self.dr_cfgs["container"] = TargetDRCfg(asset_id=self.receptacle_asset_id)
        self.dr_cfgs["distractors"] = DistractorDRCfg(
            res_id="objects",
            number_of_distractors=2,
            allow_duplicates=False,
            include=[self.receptacle_load_id, self.occupant_load_id],
        )
        self.dr_cfgs["spatial"] = SpatialDRCfg(
            spatial_mode="fixed",
            robot_region=Box(low=[-0.95, 0.0, 0.0], high=[-0.95, 0.0, 0.0]),
            target_region=Box(low=[-0.28, 0.0], high=[-0.28, 0.0]),
            container_region=Box(low=[-0.02, 0.21], high=[-0.02, 0.21]),
            distractors_region=getattr(
                self,
                "initial_distractors_region",
                [
                    Box(low=[-0.02, -0.21], high=[-0.02, -0.21]),
                    Box(low=[0.18, 0.0], high=[0.18, 0.0]),
                ],
            ),
            target_stable_indices=[0],
            target_rotate_z=Box(low=0.0, high=0.0),
            container_rotate_z=Box(low=0.0, high=0.0),
            distractor_rotate_z=getattr(self, "initial_distractor_rotate_z", None),
            fixed_stable_pose_idx=0,
            obj_surface_map={"target": "table", "container": "table", "distractor": "table"},
        )
        self.dr_cfgs["scene"] = _fixed_scene(_LOW_TABLE)
        self.hold_steps = int(hold_steps)
        self._success_hold_count = 0
        self._selected_receptacle_key = "container"
        super().__init__(*args, target=target, **kwargs)

    @property
    def left_receptacle(self) -> ObjectActor:
        actor = self.layout.actors["container"]
        assert isinstance(actor, ObjectActor)
        return actor

    @property
    def right_receptacle(self) -> ObjectActor:
        actor = self.layout.actors["distractor_0"]
        assert isinstance(actor, ObjectActor)
        return actor

    @property
    def occupant(self) -> ObjectActor:
        actor = self.layout.actors["distractor_1"]
        assert isinstance(actor, ObjectActor)
        return actor

    @property
    def selected_receptacle(self) -> ObjectActor:
        return (
            self.left_receptacle
            if self._selected_receptacle_key == "container"
            else self.right_receptacle
        )

    def _configure_condition_after_reset(self) -> None:
        self.left_receptacle.instance_id = self.left_instance_id
        self.right_receptacle.instance_id = self.right_instance_id
        left_selected = self.selected_initial_condition in ("left_empty", "left_clear")
        self._selected_receptacle_key = "container" if left_selected else "distractor_0"
        occupied = self.right_receptacle if left_selected else self.left_receptacle
        occupied_position = np.asarray(occupied.pose.position, dtype=np.float64)
        self.occupant.pose.position = (
            occupied_position + np.array([0.0, 0.0, self.occupant_local_z], dtype=np.float64)
        ).tolist()
        self.occupant.pose.quaternion = _stable_pose(self.occupant)[1]
        # Existing occupants are normally static scene contents.  The banana
        # in the fruit-free-basket task is intentionally placed above the
        # basket so it can settle under gravity before the task starts.
        self.occupant.mujoco_freejoint = self.uid == (
            "g1_wholebody_condition_apple_to_fruit_free_basket_teleop"
        )
        self._success_hold_count = 0
        self.reward = 0.0

    def _target_in_selected_receptacle(self, mujoco_env) -> bool:
        selected = self.selected_receptacle
        if self.full_containment:
            points = _points_in_body_local(mujoco_env, self.target, selected.simulation_name)
            if points is None:
                return False
            inside = bool(
                np.all(np.abs(points[:, :2]) <= self.inside_half_xy)
                and np.all(points[:, 2] >= self.inside_z[0])
                and np.all(points[:, 2] <= self.inside_z[1])
            )
        else:
            local = _center_in_actor_local(mujoco_env, self.target, selected)
            if local is None:
                return False
            inside = bool(
                np.all(np.abs(local[:2]) <= self.inside_half_xy)
                and self.inside_z[0] <= float(local[2]) <= self.inside_z[1]
            )
        return bool(
            inside
            and _contact_between(mujoco_env, self.target.simulation_name, selected.simulation_name)
            and not _hand_contact(mujoco_env, self.target)
        )

    def compute_reward(self, info: dict[str, Any], *args, **kwargs) -> float:
        del info, args
        if self._target_in_selected_receptacle(kwargs.get("mujoco_env")):
            self._success_hold_count += 1
        else:
            self._success_hold_count = 0
        self.reward = 0.1 * self._success_hold_count
        return float(self.reward)

    def check_success(self, info: dict[str, Any], *args, **kwargs) -> bool:
        del info, args, kwargs
        return self._success_hold_count >= self.hold_steps

    def preload_objects(self) -> list[Actor]:
        objects = AssetManager.get("objects")
        return [
            ObjectActor(asset=objects.load(self.target_load_id)),
            ObjectActor(
                asset=objects.load(self.receptacle_load_id), instance_id=self.left_instance_id
            ),
            ObjectActor(
                asset=objects.load(self.receptacle_load_id), instance_id=self.right_instance_id
            ),
            ObjectActor(asset=objects.load(self.occupant_load_id)),
        ]


@TaskRegistry.register("g1_wholebody_condition_bottle_to_empty_box_teleop")
class G1WholebodyConditionBottleToEmptyBoxTaskTeleop(_ConditionChooseReceptacleTask):
    uid = "g1_wholebody_condition_bottle_to_empty_box_teleop"
    label = "G1 TELEOP Condition Bottle To Empty Storage Box"
    description = (
        "A task where the G1 robot must put the bottle into the storage box that is empty."
    )
    default_instruction = "put the bottle into the storage box that is empty"
    initial_conditions = ("left_empty", "right_empty")
    target_initial_xy = (-0.52, 0.0)
    target_asset_id = _BOTTLE
    target_load_id = _BOTTLE_LOAD
    receptacle_asset_id = _STORAGE_BOX
    receptacle_load_id = _STORAGE_BOX_LOAD
    occupant_load_id = _APPLE_LOAD
    left_instance_id = "storage_box_left"
    right_instance_id = "storage_box_right"
    inside_half_xy = _STORAGE_BOX_HALF_XY
    inside_z = _STORAGE_BOX_BOTTLE_CENTER_Z
    occupant_local_z = 0.04
    initial_distractors_region = [
        Box(low=[-0.02, -0.21], high=[-0.02, -0.21]),
        Box(low=[0.34, 0.0], high=[0.34, 0.0]),
    ]
    initial_distractor_rotate_z = Box(low=0.0, high=0.0)

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> None:
        super().reset(seed, options)
        if isinstance(self._target, ObjectActor):
            _place_on_surface(self._target, type(self).target_initial_xy, 0.0)


@TaskRegistry.register("g1_wholebody_condition_apple_to_fruit_free_basket_teleop")
class G1WholebodyConditionAppleToFruitFreeBasketTaskTeleop(_ConditionChooseReceptacleTask):
    uid = "g1_wholebody_condition_apple_to_fruit_free_basket_teleop"
    label = "G1 TELEOP Condition Apple To Fruit-Free Basket"
    description = (
        "A task where the G1 robot must put the apple into the basket that has no other fruit."
    )
    default_instruction = "put the apple into the basket that has no other fruit"
    initial_conditions = ("left_clear", "right_clear")
    target_asset_id = _APPLE
    target_load_id = _APPLE_LOAD
    receptacle_asset_id = _BASKET
    receptacle_load_id = _BASKET_LOAD
    occupant_load_id = _BANANA_LOAD
    left_instance_id = "basket_left"
    right_instance_id = "basket_right"
    inside_half_xy = _BASKET_HALF_XY
    inside_z = _BASKET_Z
    occupant_local_z = 0.10


@TaskRegistry.register("g1_wholebody_condition_chair_to_table_and_sit_teleop")
class G1WholebodyConditionChairToTableAndSitTaskTeleop(
    _InitialConditionMixin, G1WholebodyXMovePickTaskTeleop
):
    uid = "g1_wholebody_condition_chair_to_table_and_sit_teleop"
    label = "G1 TELEOP Condition Chair To Table And Sit"
    description = "A task where the G1 robot must sit on the chair in front of the table, pushing the chair there first if necessary."
    default_instruction = "if the chair is not in front of the table, push it there and sit down; if it is already there, sit down directly"
    initial_conditions = ("away", "at_table")

    def __init__(
        self, *args, target: str | None = None, hold_steps: int = _SIT_HOLD_STEPS, **kwargs
    ):
        self.dr_cfgs = deepcopy(G1WholebodyXMovePickTaskTeleop.dr_cfgs)
        self.dr_cfgs["language"] = LanguageDRCfg(instructions=[self.default_instruction])
        self.dr_cfgs["target"] = TargetDRCfg(asset_id=_CHAIR)
        self.dr_cfgs["distractors"] = DistractorDRCfg(
            res_id="objects", number_of_distractors=0, allow_duplicates=False
        )
        self.dr_cfgs["spatial"] = SpatialDRCfg(
            spatial_mode="fixed",
            robot_region=Box(low=[-1.2, 0.0, 0.0], high=[-1.2, 0.0, 0.0]),
            target_region=Box(low=[-0.62, 0.0], high=[-0.62, 0.0]),
            distractors_region=Box(low=[0.0, 0.0], high=[0.0, 0.0]),
            target_stable_indices=[0],
            target_rotate_z=Box(low=-0.5 * np.pi, high=-0.5 * np.pi),
            fixed_stable_pose_idx=0,
            obj_surface_map={"target": "default"},
        )
        self.dr_cfgs["scene"] = _fixed_scene(_HIGH_TABLE, (0.55, 0.0))
        self.hold_steps = int(hold_steps)
        self._chair_at_table_phase = False
        self._sit_hold_count = 0
        self._chair_init_z = 0.0
        self._chair_init_up = np.array([0.0, 0.0, 1.0])
        super().__init__(*args, target=target, **kwargs)

    def _configure_condition_after_reset(self) -> None:
        assert isinstance(self.target, ObjectActor)
        chair_xy = (-0.23, 0.0) if self.selected_initial_condition == "at_table" else (-0.62, 0.0)
        _move_on_surface_preserve_orientation(self.target, chair_xy, 0.0)
        self.target.mujoco_freejoint = True
        self.target.mujoco_mass = _PUSH_CHAIR_MUJOCO_MASS_KG
        self.target.mujoco_friction = [1.8, 0.08, 0.01]
        self._chair_at_table_phase = self.selected_initial_condition == "at_table"
        self._sit_hold_count = 0
        self._chair_init_z = float(self.target.pose.position[2])
        self._chair_init_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    def _chair_metrics(self, mujoco_env) -> dict[str, Any]:
        pose = _body_pose(mujoco_env, self.target.simulation_name)
        if pose is None:
            return {"at_table": False, "stable": False, "position": np.zeros(3)}
        position, rotation = pose
        up = rotation[:, 2]
        at_table = float(np.linalg.norm(position[:2] - np.array([-0.23, 0.0]))) <= 0.22
        stable = bool(
            float(np.dot(up, self._chair_init_up)) >= np.cos(np.deg2rad(25.0))
            and float(position[2] - self._chair_init_z) <= 0.08
        )
        return {"at_table": at_table, "stable": stable, "position": position}

    def _has_sit_contact(self, mujoco_env) -> bool:
        model = mujoco_env.mjModel
        data = mujoco_env.mjData
        for contact_id in range(data.ncon):
            contact = data.contact[contact_id]
            body1 = _body_name(model, contact.geom1)
            body2 = _body_name(model, contact.geom2)
            chair_first = self.target.simulation_name in body1
            chair_second = self.target.simulation_name in body2
            if chair_first == chair_second:
                continue
            other = body2 if chair_first else body1
            if any(key in other.lower() for key in ("pelvis", "hip", "waist", "torso")):
                return True
        return False

    def _sitting_now(self, metrics: dict[str, Any], mujoco_env) -> bool:
        pelvis = np.asarray(mujoco_env.mjData.body("pelvis").xpos, dtype=np.float64)
        qvel = np.asarray(mujoco_env.mjData.qvel, dtype=np.float64)
        return bool(
            metrics["at_table"]
            and metrics["stable"]
            and np.linalg.norm(pelvis[:2] - metrics["position"][:2]) <= 0.32
            and 0.38 <= float(pelvis[2]) <= 0.72
            and np.linalg.norm(qvel[:3]) <= 0.15
            and np.linalg.norm(qvel[3:6]) <= 0.8
            and self._has_sit_contact(mujoco_env)
        )

    def compute_reward(self, info: dict[str, Any], *args, **kwargs) -> float:
        del info, args
        mujoco_env = kwargs.get("mujoco_env")
        if mujoco_env is None:
            return 0.0
        metrics = self._chair_metrics(mujoco_env)
        if metrics["at_table"] and metrics["stable"]:
            self._chair_at_table_phase = True
        if self._chair_at_table_phase and self._sitting_now(metrics, mujoco_env):
            self._sit_hold_count += 1
        else:
            self._sit_hold_count = 0
        return float(np.clip(self._sit_hold_count / self.hold_steps, 0.0, 1.0))

    def check_success(self, info: dict[str, Any], *args, **kwargs) -> bool:
        del info, args, kwargs
        return self._sit_hold_count >= self.hold_steps

    def preload_objects(self) -> list[Actor]:
        return [ObjectActor(asset=AssetManager.get("objects").load(_CHAIR_LOAD))]


@TaskRegistry.register("g1_wholebody_condition_basket_to_table_apple_to_basket_teleop")
class G1WholebodyConditionBasketToTableAppleToBasketTaskTeleop(
    _InitialConditionMixin, G1WholebodyLongGroundBasketToTableAppleToBasketTaskTeleop
):
    uid = "g1_wholebody_condition_basket_to_table_apple_to_basket_teleop"
    label = "G1 TELEOP Condition Basket To Table Apple To Basket"
    description = "A task where the G1 robot must put the apple into the basket on the table, moving the basket from the ground to the table first if necessary."
    default_instruction = "if the basket is on the ground, put it on the table and then put the apple inside; if it is on the table, put the apple inside directly"
    initial_conditions = ("ground", "on_table")
    dr_cfgs = deepcopy(G1WholebodyLongGroundBasketToTableAppleToBasketTaskTeleop.dr_cfgs)
    dr_cfgs["language"] = LanguageDRCfg(instructions=[default_instruction])

    def _configure_condition_after_reset(self) -> None:
        if self.selected_initial_condition == "on_table":
            assert isinstance(self.target, ObjectActor)
            _place_on_surface(self.target, (0.05, 0.02), _LOW_TABLE)
            self._basket_on_table_phase = True


@TaskRegistry.register("g1_wholebody_condition_eraser_to_pencil_case_teleop")
class G1WholebodyConditionEraserToPencilCaseTaskTeleop(
    _InitialConditionMixin, G1WholebodyXMovePickTaskTeleop
):
    uid = "g1_wholebody_condition_eraser_to_pencil_case_teleop"
    label = "G1 TELEOP Condition Eraser To Pencil Case"
    description = (
        "if the pencil case is closed, open it and put the eraser inside; "
        "if it is open, put the eraser inside directly"
    )
    initial_conditions = ("closed", "open")

    def __init__(self, *args, target: str | None = None, hold_steps: int = _HOLD_STEPS, **kwargs):
        self.dr_cfgs = deepcopy(G1WholebodyXMovePickTaskTeleop.dr_cfgs)
        self.dr_cfgs["language"] = LanguageDRCfg(instructions=[self.description])
        self.dr_cfgs["target"] = TargetDRCfg(asset_id=_ERASER)
        self.dr_cfgs["articulated"] = ArticulatedObjectDrCfg(
            asset_id=_PENCIL_CASE,
            articulate_init_joint_qpos={_PENCIL_CASE_JOINT: _PENCIL_CASE_CLOSED},
        )
        self.dr_cfgs["distractors"] = DistractorDRCfg(
            res_id="objects", number_of_distractors=0, allow_duplicates=False
        )
        self.dr_cfgs["spatial"] = SpatialDRCfg(
            spatial_mode="fixed",
            robot_region=Box(low=[-0.95, 0.0, 0.0], high=[-0.95, 0.0, 0.0]),
            target_region=Box(low=[-0.275, -0.18], high=[-0.275, -0.18]),
            distractors_region=Box(low=[0.0, 0.0], high=[0.0, 0.0]),
            articulated_region=Box(low=[-0.22, 0.16, 0.651], high=[-0.22, 0.16, 0.651]),
            target_stable_indices=[0],
            target_rotate_z=Box(low=0.0, high=0.0),
            articulated_rotate_z=Box(low=-0.5 * np.pi, high=-0.5 * np.pi),
            fixed_stable_pose_idx=0,
            obj_surface_map={"target": "table"},
        )
        self.dr_cfgs["scene"] = _fixed_scene(_HIGH_TABLE)
        self.hold_steps = int(hold_steps)
        self._success_hold_count = 0
        super().__init__(*args, target=target, **kwargs)

    def _configure_condition_before_reset(self) -> None:
        qpos = (
            _PENCIL_CASE_CLOSED
            if self.selected_initial_condition == "closed"
            else _PENCIL_CASE_OPEN
        )
        cfg = self.dr.get_randomizer("articulated").cfg
        cfg.articulate_init_joint_qpos = {_PENCIL_CASE_JOINT: qpos}
        self.dr_cfgs["articulated"].articulate_init_joint_qpos = {_PENCIL_CASE_JOINT: qpos}

    def _configure_condition_after_reset(self) -> None:
        self._success_hold_count = 0

    def _eraser_in_case(self, mujoco_env) -> bool:
        points = _points_in_body_local(mujoco_env, self.target, _PENCIL_CASE_BASE)
        if points is None:
            return False
        qpos = float(mujoco_env.mjData.joint(_PENCIL_CASE_JOINT).qpos[0])
        return bool(
            qpos <= _PENCIL_CASE_OPEN_THRESHOLD
            and np.all(points >= _PENCIL_CASE_LOW)
            and np.all(points <= _PENCIL_CASE_HIGH)
            and _contact_between(mujoco_env, self.target.simulation_name, _PENCIL_CASE_BASE)
            and not _hand_contact(mujoco_env, self.target)
        )

    def compute_reward(self, info: dict[str, Any], *args, **kwargs) -> float:
        del info, args
        if self._eraser_in_case(kwargs.get("mujoco_env")):
            self._success_hold_count += 1
        else:
            self._success_hold_count = 0
        return float(0.1 * self._success_hold_count)

    def check_success(self, info: dict[str, Any], *args, **kwargs) -> bool:
        del info, args, kwargs
        return self._success_hold_count >= self.hold_steps

    def preload_objects(self) -> list[Actor]:
        return [ObjectActor(asset=AssetManager.get("objects").load(_ERASER_LOAD))]


@TaskRegistry.register("g1_wholebody_condition_bread_to_plate_or_tray_teleop")
class G1WholebodyConditionBreadToPlateOrTrayTaskTeleop(
    _InitialConditionMixin, G1WholebodyXMovePickTaskTeleop
):
    uid = "g1_wholebody_condition_bread_to_plate_or_tray_teleop"
    label = "G1 TELEOP Condition Bread To Plate Or Tray"
    description = "A task where the G1 robot must put the bread into the empty receptacle: use the plate when the cake is on the tray, or use the tray when the cake is on the plate."
    default_instruction = "put the bread into the empty receptacle: use the plate when the cake is on the tray, or use the tray when the cake is on the plate"
    initial_conditions = ("plate_empty", "plate_occupied")

    def __init__(self, *args, target: str | None = None, hold_steps: int = _HOLD_STEPS, **kwargs):
        self.dr_cfgs = deepcopy(G1WholebodyXMovePickTaskTeleop.dr_cfgs)
        self.dr_cfgs["language"] = LanguageDRCfg(instructions=[self.default_instruction])
        self.dr_cfgs["target"] = TargetDRCfg(asset_id=_BREAD)
        self.dr_cfgs["container"] = TargetDRCfg(asset_id=_PLATE)
        self.dr_cfgs["distractors"] = DistractorDRCfg(
            res_id="objects",
            number_of_distractors=2,
            allow_duplicates=False,
            include=[_TRAY_LOAD, _CAKE_LOAD],
        )
        self.dr_cfgs["spatial"] = SpatialDRCfg(
            spatial_mode="fixed",
            robot_region=Box(low=[-0.95, 0.0, 0.0], high=[-0.95, 0.0, 0.0]),
            target_region=Box(low=_C8_BREAD_XY, high=_C8_BREAD_XY),
            container_region=Box(low=_C8_PLATE_XY, high=_C8_PLATE_XY),
            distractors_region=[
                Box(low=_C8_TRAY_XY, high=_C8_TRAY_XY),
                Box(low=[0.30, 0.10], high=[0.30, 0.10]),
            ],
            target_stable_indices=[0],
            target_rotate_z=Box(low=0.0, high=0.0),
            container_rotate_z=Box(low=0.0, high=0.0),
            distractor_rotate_z=Box(low=0.0, high=0.0),
            fixed_stable_pose_idx=0,
            obj_surface_map={"target": "table", "container": "table", "distractor": "table"},
        )
        self.dr_cfgs["scene"] = _fixed_scene(_HIGH_TABLE)
        self.hold_steps = int(hold_steps)
        self._success_hold_count = 0
        super().__init__(*args, target=target, **kwargs)

    @property
    def plate(self) -> ObjectActor:
        actor = self.layout.actors["container"]
        assert isinstance(actor, ObjectActor)
        return actor

    @property
    def tray(self) -> ObjectActor:
        actor = self.layout.actors["distractor_0"]
        assert isinstance(actor, ObjectActor)
        return actor

    @property
    def cake(self) -> ObjectActor:
        actor = self.layout.actors["distractor_1"]
        assert isinstance(actor, ObjectActor)
        return actor

    def _configure_condition_after_reset(self) -> None:
        if self.selected_initial_condition == "plate_occupied":
            plate_position = np.asarray(self.plate.pose.position, dtype=np.float64)
            cake_stable_z, cake_quaternion = _stable_pose(self.cake)
            self.cake.pose.position = (
                plate_position + np.array([0.0, 0.0, cake_stable_z], dtype=np.float64)
            ).tolist()
            self.cake.pose.quaternion = cake_quaternion
        else:
            tray_position = np.asarray(self.tray.pose.position, dtype=np.float64)
            tray_rotation = t3d.quaternions.quat2mat(self.tray.pose.quaternion)
            self.cake.pose.position = (
                tray_position + tray_rotation @ _C8_CAKE_IN_TRAY_LOCAL
            ).tolist()
            self.cake.pose.quaternion = _stable_pose(self.cake)[1]
        self._success_hold_count = 0

    def _bread_on_destination(self, mujoco_env) -> bool:
        destination = self.plate if self.selected_initial_condition == "plate_empty" else self.tray
        local = _center_in_actor_local(mujoco_env, self.target, destination)
        if local is None:
            return False
        if destination is self.plate:
            inside = bool(
                np.linalg.norm(local[:2]) <= _PLATE_RADIUS
                and _PLATE_Z[0] <= local[2] <= _PLATE_Z[1]
            )
        else:
            inside = bool(
                np.all(np.abs(local[:2]) <= _TRAY_HALF_XY) and _TRAY_Z[0] <= local[2] <= _TRAY_Z[1]
            )
        return bool(
            inside
            and _contact_between(
                mujoco_env, self.target.simulation_name, destination.simulation_name
            )
            and not _hand_contact(mujoco_env, self.target)
        )

    def compute_reward(self, info: dict[str, Any], *args, **kwargs) -> float:
        del info, args
        if self._bread_on_destination(kwargs.get("mujoco_env")):
            self._success_hold_count += 1
        else:
            self._success_hold_count = 0
        return float(0.1 * self._success_hold_count)

    def check_success(self, info: dict[str, Any], *args, **kwargs) -> bool:
        del info, args, kwargs
        return self._success_hold_count >= self.hold_steps

    def preload_objects(self) -> list[Actor]:
        objects = AssetManager.get("objects")
        return [
            ObjectActor(asset=objects.load(_BREAD_LOAD)),
            ObjectActor(asset=objects.load(_PLATE_LOAD)),
            ObjectActor(asset=objects.load(_TRAY_LOAD)),
            ObjectActor(asset=objects.load(_CAKE_LOAD)),
        ]


@TaskRegistry.register("g1_wholebody_condition_clear_chair_and_push_dolly_teleop")
class G1WholebodyConditionClearChairAndPushDollyTaskTeleop(
    _InitialConditionMixin, G1WholebodyXMovePickTaskTeleop
):
    uid = "g1_wholebody_condition_clear_chair_and_push_dolly_teleop"
    label = "G1 TELEOP Condition Dolly Obstacle And Push Chair"
    description = "A task where the G1 robot must push the chair toward the table; the dolly may be blocking the path."
    default_instruction = "push the chair toward the table; the dolly may be blocking the path"
    initial_conditions = ("block", "clear")

    def __init__(self, *args, target: str | None = None, hold_steps: int = _HOLD_STEPS, **kwargs):
        self.dr_cfgs = deepcopy(G1WholebodyXMovePickTaskTeleop.dr_cfgs)
        self.dr_cfgs["language"] = LanguageDRCfg(instructions=[self.default_instruction])
        self.dr_cfgs["target"] = TargetDRCfg(asset_id=_CHAIR)
        self.dr_cfgs["articulated"] = ArticulatedObjectDrCfg(asset_id=_DOLLY)
        self.dr_cfgs["distractors"] = DistractorDRCfg(
            res_id="objects", number_of_distractors=0, allow_duplicates=False
        )
        self.dr_cfgs["spatial"] = SpatialDRCfg(
            spatial_mode="fixed",
            robot_region=Box(low=[-1.25, 0.0, 0.0], high=[-1.25, 0.0, 0.0]),
            target_region=Box(low=_C9_CHAIR_XY, high=_C9_CHAIR_XY),
            distractors_region=Box(low=[0.0, 0.0], high=[0.0, 0.0]),
            articulated_region=Box(low=[*_C9_DOLLY_BLOCK_XY, 0.0], high=[*_C9_DOLLY_BLOCK_XY, 0.0]),
            target_stable_indices=[0],
            target_rotate_z=Box(low=_C9_CHAIR_YAW, high=_C9_CHAIR_YAW),
            articulated_rotate_z=Box(low=_C9_DOLLY_YAW, high=_C9_DOLLY_YAW),
            fixed_stable_pose_idx=0,
            obj_surface_map={"target": "default", "articulated": "default"},
        )
        self.dr_cfgs["scene"] = _fixed_scene(
            _HIGH_TABLE, tuple(float(value) for value in _C9_TABLE_XY)
        )
        self.hold_steps = int(hold_steps)
        self._push_hold_count = 0
        self._init_chair_position: np.ndarray | None = None
        self._init_chair_up: np.ndarray | None = None
        super().__init__(*args, target=target, **kwargs)

    def _configure_condition_after_reset(self) -> None:
        assert isinstance(self.target, ObjectActor)
        _move_on_surface_preserve_orientation(self.target, _C9_CHAIR_XY.tolist(), 0.0)
        dolly = self.layout.actors["articulated"]
        dolly_xy = (
            _C9_DOLLY_BLOCK_XY if self.selected_initial_condition == "block" else _C9_DOLLY_CLEAR_XY
        )
        dolly.pose.position = [float(dolly_xy[0]), float(dolly_xy[1]), 0.0]
        dolly.pose.quaternion = [
            float(value) for value in t3d.euler.euler2quat(0.0, 0.0, _C9_DOLLY_YAW)
        ]
        self.target.mujoco_mass = _PUSH_CHAIR_MUJOCO_MASS_KG
        self.target.mujoco_friction = [0.35, 0.02, 0.001]
        self._push_hold_count = 0
        self._init_chair_position = None
        self._init_chair_up = None

    def _chair_metrics(self, mujoco_env) -> dict[str, float]:
        pose = _body_pose(mujoco_env, self.target.simulation_name)
        if pose is None:
            return {"displacement": 0.0, "tilt_cos": 0.0, "lift": 0.0}
        position, rotation = pose
        up = rotation[:, 2]
        if self._init_chair_position is None:
            self._init_chair_position = position.copy()
        if self._init_chair_up is None:
            self._init_chair_up = up / max(float(np.linalg.norm(up)), 1e-6)
        return {
            "displacement": float(np.linalg.norm(position[:2] - self._init_chair_position[:2])),
            "tilt_cos": float(
                np.dot(up / max(float(np.linalg.norm(up)), 1e-6), self._init_chair_up)
            ),
            "lift": float(position[2] - self._init_chair_position[2]),
        }

    def _dolly_metrics(self, mujoco_env) -> dict[str, float]:
        """Backward-compatible metric name; metrics track the pushed chair."""
        return self._chair_metrics(mujoco_env)

    def compute_reward(self, info: dict[str, Any], *args, **kwargs) -> float:
        del info, args
        mujoco_env = kwargs.get("mujoco_env")
        if mujoco_env is None:
            return 0.0
        metrics = self._chair_metrics(mujoco_env)
        pushed = bool(
            metrics["displacement"] >= 0.50
            and metrics["tilt_cos"] >= np.cos(np.deg2rad(35.0))
            and metrics["lift"] <= 0.15
        )
        if pushed:
            self._push_hold_count += 1
        else:
            self._push_hold_count = 0
        hold = np.clip(self._push_hold_count / self.hold_steps, 0.0, 1.0)
        move = np.clip(metrics["displacement"] / 0.50, 0.0, 1.0)
        return float(min(hold, move))

    def check_success(self, info: dict[str, Any], *args, **kwargs) -> bool:
        del info, args, kwargs
        return self._push_hold_count >= self.hold_steps

    def preload_objects(self) -> list[Actor]:
        return [ObjectActor(asset=AssetManager.get("objects").load(_CHAIR_LOAD))]
