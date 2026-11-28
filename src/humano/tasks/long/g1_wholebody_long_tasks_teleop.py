from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import transforms3d as t3d
import trimesh
from gymnasium import spaces

from humano.assets import AssetManager
from humano.core.actor import Actor, ObjectActor
from humano.core.layout import Layout
from humano.dr import (
    ArticulatedObjectDrCfg,
    DistractorDRCfg,
    LanguageDRCfg,
    SpatialDRCfg,
    TabletopSceneDRCfg,
    TargetDRCfg,
)
from humano.dr.types import Box
from humano.robots.protocols import Controllable
from humano.tasks.basic.g1_wholebody_ball_kick_teleop import G1WholebodyBallKickTaskTeleop
from humano.tasks.basic.g1_wholebody_chair_push_teleop import (
    _PUSH_CHAIR_MUJOCO_MASS_KG,
    G1WholebodyChairPushTaskTeleop,
)
from humano.tasks.basic.g1_wholebody_chair_sit_teleop import (
    _SIT_CHAIR_MUJOCO_FRICTION,
)
from humano.tasks.basic.g1_wholebody_dolly_push_teleop import (
    _DOLLY_BASE_BODY,
    _DOLLY_FLOATING_BASE_JOINT,
    G1WholebodyDollyPushTaskTeleop,
)
from humano.tasks.simple.g1_wholebody_xmove_pick_teleop import G1WholebodyXMovePickTaskTeleop
from humano.tasks.registry import TaskRegistry

_APPLE_ASSET_ID = "objects:apple/apple_001"
_APPLE_LOAD_ID = "apple/apple_001"
_PLATE_ASSET_ID = "objects:plate/plate_3"
_PLATE_LOAD_ID = "plate/plate_3"
_PENCIL_ASSET_ID = "objects:pencil002/pencil002"
_PENCIL_LOAD_ID = "pencil002/pencil002"
_ERASER_LOAD_ID = "eraser/eraser_002"
_PEN_ASSET_ID = "objects:pen/pen"
_PEN_LOAD_ID = "pen/pen"
_PEN_HOLDER_ASSET_ID = "objects:pen_holder001/pen_holder001"
_PEN_HOLDER_LOAD_ID = "pen_holder001/pen_holder001"
_BOX_ASSET_ID = "objects:box/box_007"
_BOX_LOAD_ID = "box/box_007"
_STORAGE_BOX_ASSET_ID = "objects:storage_box/storage_box_003"
_STORAGE_BOX_LOAD_ID = "storage_box/storage_box_003"
_BOTTLE_ASSET_ID = "objects:bottle/bottle_048"
_BOTTLE_LOAD_ID = "bottle/bottle_048"
_FOOTBALL_ASSET_ID = "objects:football/football"
_FOOTBALL_LOAD_ID = "football/football"
_LONG_FOOTBALL_CENTER_XY = (-0.05, -0.05)
_LONG_FOOTBALL_YAW = np.deg2rad(45.0)
_BASKET_ASSET_ID = "objects:picnic_basket/picnic_basket_009"
_BASKET_LOAD_ID = "picnic_basket/picnic_basket_009"
_BANANA_ASSET_ID = "objects:banana/banana_020"
_BANANA_LOAD_ID = "banana/banana_020"
_TRASH_CAN_ASSET_ID = "objects:dustbin/dustbin_011"
_TRASH_CAN_LOAD_ID = "dustbin/dustbin_011"
_BREAD_ASSET_ID = "objects:bread/bread_006"
_BREAD_LOAD_ID = "bread/bread_006"
_CAKE_ASSET_ID = "objects:cake/cake_001"
_CAKE_LOAD_ID = "cake/cake_001"
_TRAY_ASSET_ID = "objects:tray/tray_001"
_TRAY_LOAD_ID = "tray/tray_001"
_PEANUT_016_ASSET_ID = "objects:peanut/peanut_016"
_PEANUT_016_LOAD_ID = "peanut/peanut_016"
_PEANUT_018_LOAD_ID = "peanut/peanut_018"
_CUP_ASSET_ID = "objects:Cup_B0BR43SPKJ_Vanilla/Cup_B0BR43SPKJ_Vanilla"
_CUP_LOAD_ID = "Cup_B0BR43SPKJ_Vanilla/Cup_B0BR43SPKJ_Vanilla"
_DRAWER_ASSET_ID = "articulated:8"
_DRAWER_JOINT = "articulate_joint_0"
_DRAWER_BODY = "E_drawer_59"
_DRAWER_OPEN_THRESHOLD_M = 0.12
_DRAWER_LOCAL_LOW = np.array([-0.17, 0.025, -0.13], dtype=np.float64)
_DRAWER_LOCAL_HIGH = np.array([0.17, 0.335, 0.06], dtype=np.float64)
_DRAWER_INNER_BOTTOM_LOCAL_Z_RANGE = (-0.13, -0.085)

_PENCIL_CASE_ASSET_ID = "articulated:9"
_PENCIL_CASE_JOINT_QPOS = {"articulate_joint_0": -np.deg2rad(100.0)}
_PENCIL_CASE_POSITION = [-0.08, 0.16, 0.651]
_PENCIL_CASE_ROTATE_Z = np.pi / 2.0
_PENCIL_CASE_BASE_BODY = "009_base"
_PENCIL_CASE_LOCAL_LOW = np.array([-0.096, -0.023, -0.004], dtype=np.float64)
_PENCIL_CASE_LOCAL_HIGH = np.array([0.086, 0.023, 0.021], dtype=np.float64)

_PEN_HOLDER_XY = [-0.05, 0.18]
_PEN_HOLDER_RADIUS_M = 0.075
_PEN_HOLDER_MIN_Z_M = 0.02
_PEN_HOLDER_MAX_Z_M = 0.26
_PEN_HOLDER_VERTICAL_TILT_RAD = np.deg2rad(38.0)

_HIGH_TABLE_HEIGHT = 0.65
_LOW_TABLE_HEIGHT = 0.35
_SUCCESS_REWARD_INCREMENT = 0.1

_BOX_DOLLY_PUSH_THRESHOLD_M = 0.30
_BOX_TO_DOLLY_INITIAL_DOLLY_POSITION = [0.70, 0.0, 0.0]
_BOX_TO_DOLLY_INITIAL_DOLLY_YAW = 0.0

_CHAIR_PUSH_THRESHOLD_M = 0.35
_CHAIR_SIT_XY_RADIUS_M = 0.30
_CHAIR_SIT_PELVIS_Z_RANGE_M = (0.38, 0.72)
_BOX_THEN_CHAIR_ROBOT_POSITION = [-1.92, 0.0, 0.0]
_BOX_THEN_CHAIR_INITIAL_XY = [-1.30, -0.18]
_BOX_THEN_CHAIR_INITIAL_YAW = np.pi / 2
_BOX_THEN_CHAIR_BOX_XY = [-0.52, -0.18]

_BALL_GROUND_Z_MAX_M = 0.20
_BALL_KICK_DISTANCE_M = 0.50

_BASKET_GROUND_XY = [-0.62, 0.0]
_BASKET_TABLE_XY = [0.32, 0.02]
_BASKET_PLACE_XY = [0.05, 0.02]
_BASKET_APPLE_XY = [-0.20, -0.18]
_BASKET_MUJOCO_MASS_KG = 0.35
_BASKET_INSIDE_HALF_EXTENTS = np.array([0.14, 0.11], dtype=np.float64)
_BASKET_INSIDE_Z_RANGE = (-0.08, 0.18)
_TRASH_CAN_XY = [-0.52, 0.18]
_TRASH_CAN_INSIDE_HALF_EXTENTS = np.array([0.12, 0.12], dtype=np.float64)
_TRASH_CAN_INSIDE_Z_RANGE = (-0.14, 0.05)

_BOX_TABLE_XY = [0.32, 0.02]
_BOX_GROUND_XY = [-0.62, -0.12]
_BOTTLE_GROUND_XY = [-0.62, 0.16]
_BOX_PLACE_XY = [0.06, 0.02]
_STORAGE_BOX_INTERIOR_LOW = np.array([-0.215, -0.16, -0.005], dtype=np.float64)
_STORAGE_BOX_INTERIOR_HIGH = np.array([0.215, 0.16, 0.15], dtype=np.float64)
_STORAGE_BOX_MIN_INSIDE_FRACTION = 0.90
_L10_BOX_LIFT_THRESHOLD_M = 0.10

_TRAY_XY = [-0.05, 0.16]
_TRAY_INSIDE_HALF_EXTENTS = np.array([0.15, 0.11], dtype=np.float64)
_TRAY_INSIDE_Z_RANGE = (0.0, 0.10)
_L14_STORAGE_BOX_XY = [-0.04, 0.16]
_L14_TRAY_XY = [0.10, -0.20]
_L14_CAKE_IN_BOX_LOCAL = np.array([-0.13, 0.0, 0.055], dtype=np.float64)
_L14_BREAD_IN_BOX_LOCAL = np.array([0.08, 0.0, 0.003], dtype=np.float64)

_CUP_XY = [-0.17, 0.16]
_PLATE_XY = [0.08, 0.16]
_PLATE_INSIDE_RADIUS_M = 0.145
_PLATE_INSIDE_NORMAL_RANGE = (-0.02, 0.08)
_CUP_TIP_TILT_RAD = np.deg2rad(55.0)

_CABINET_TOP_BODY = "bedside_table014_base"
_CABINET_TOP_LOCAL_LOW = np.array([-0.28, -0.20, 0.14], dtype=np.float64)
_CABINET_TOP_LOCAL_HIGH = np.array([0.12, 0.20, 0.26], dtype=np.float64)
_BOOK_PARALLEL_MAX_RAD = np.deg2rad(20.0)
_BOOK_SPACING_RANGE_M = (0.04, 0.20)

_BOX_MOVE_AWAY_THRESHOLD_M = 0.35
_CHAIR_TO_TABLE_X_ERROR_M = 0.25
_CHAIR_TO_TABLE_Y_ERROR_M = 0.30


def _body_pose(mujoco_env, body_name: str) -> tuple[np.ndarray, np.ndarray] | None:
    try:
        body = mujoco_env.mjData.body(body_name)
    except KeyError:
        return None
    position = np.asarray(body.xpos, dtype=np.float64)
    rotation = np.asarray(body.xmat, dtype=np.float64).reshape(3, 3)
    return position, rotation


def _actor_body_name(actor: Actor) -> str:
    return str(actor.asset.label)


def _geom_body_name(model, geom) -> str:
    return model.body(int(np.asarray(geom.bodyid).item())).name


def _info_position(info: dict[str, Any], key: str, actor: Actor) -> np.ndarray:
    if key in info:
        return np.asarray(info[key][:3], dtype=np.float64)
    return np.asarray(actor.pose.position, dtype=np.float64)


def _info_quaternion(info: dict[str, Any], key: str, actor: Actor) -> np.ndarray:
    if key in info:
        return np.asarray(info[key][3:7], dtype=np.float64)
    return np.asarray(actor.pose.quaternion, dtype=np.float64)


def _contact_between(mujoco_env, label_a: str, label_b: str) -> bool:
    if mujoco_env is None:
        return False
    model = mujoco_env.mjModel
    data = mujoco_env.mjData
    for contact_id in range(data.ncon):
        contact = data.contact[contact_id]
        geom1 = model.geom(int(contact.geom1))
        geom2 = model.geom(int(contact.geom2))
        body1 = _geom_body_name(model, geom1)
        body2 = _geom_body_name(model, geom2)
        if (label_a in body1 and label_b in body2) or (label_a in body2 and label_b in body1):
            return True
    return False


def _contact_with_body_tree(mujoco_env, actor_label: str, root_body_name: str) -> bool:
    if mujoco_env is None:
        return False
    model = mujoco_env.mjModel
    data = mujoco_env.mjData
    root_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, root_body_name)
    if root_body_id < 0:
        return False

    def descends_from_root(body_id: int) -> bool:
        while body_id > 0:
            if body_id == root_body_id:
                return True
            body_id = int(model.body_parentid[body_id])
        return False

    for contact_id in range(data.ncon):
        contact = data.contact[contact_id]
        body1_id = int(model.geom(int(contact.geom1)).bodyid)
        body2_id = int(model.geom(int(contact.geom2)).bodyid)
        body1_name = model.body(body1_id).name
        body2_name = model.body(body2_id).name
        if (actor_label in body1_name and descends_from_root(body2_id)) or (
            actor_label in body2_name and descends_from_root(body1_id)
        ):
            return True
    return False


def _hand_contact(mujoco_env, actor: Actor) -> bool:
    return _contact_between(mujoco_env, _actor_body_name(actor), "hand")


def _contacts_between(mujoco_env, label_a: str, label_b: str) -> list[tuple[str, str, np.ndarray]]:
    if mujoco_env is None:
        return []
    model = mujoco_env.mjModel
    data = mujoco_env.mjData
    contacts: list[tuple[str, str, np.ndarray]] = []
    for contact_id in range(data.ncon):
        contact = data.contact[contact_id]
        geom1 = model.geom(int(contact.geom1))
        geom2 = model.geom(int(contact.geom2))
        body1 = _geom_body_name(model, geom1)
        body2 = _geom_body_name(model, geom2)
        if (label_a in body1 and label_b in body2) or (label_a in body2 and label_b in body1):
            contacts.append((body1, body2, np.asarray(contact.pos, dtype=np.float64)))
    return contacts


def _actor_center_in_actor_local(
    info: dict[str, Any], key: str, actor: Actor, container_key: str, container: Actor
) -> np.ndarray:
    actor_position = _info_position(info, key, actor)
    container_position = _info_position(info, container_key, container)
    container_quat = _info_quaternion(info, container_key, container)
    container_rotation = t3d.quaternions.quat2mat(container_quat)
    return container_rotation.T @ (actor_position - container_position)


def _center_in_plate_region(local: np.ndarray) -> bool:
    # plate_3 is authored with local y as its normal and local x/z as its surface.
    surface_radius = np.linalg.norm(local[[0, 2]])
    return bool(
        surface_radius <= _PLATE_INSIDE_RADIUS_M
        and _PLATE_INSIDE_NORMAL_RANGE[0] <= float(local[1]) <= _PLATE_INSIDE_NORMAL_RANGE[1]
    )


def _actor_center_in_body_local(mujoco_env, actor: Actor, body_name: str) -> np.ndarray | None:
    actor_pose = _body_pose(mujoco_env, _actor_body_name(actor))
    body_pose = _body_pose(mujoco_env, body_name)
    if actor_pose is None or body_pose is None:
        return None
    actor_pos, _ = actor_pose
    body_pos, body_rot = body_pose
    return (actor_pos - body_pos) @ body_rot


def _contact_in_body_local(mujoco_env, actor: Actor, body_name: str) -> bool:
    body_pose = _body_pose(mujoco_env, body_name)
    if body_pose is None:
        return False
    body_pos, body_rot = body_pose
    for _, _, contact_pos in _contacts_between(mujoco_env, _actor_body_name(actor), body_name):
        local = (contact_pos - body_pos) @ body_rot
        if np.all(local >= _CABINET_TOP_LOCAL_LOW) and np.all(local <= _CABINET_TOP_LOCAL_HIGH):
            return True
    return False


def _table_top(task) -> float:
    table = task.layout.actors["table"]
    return float(table.pose.position[2] + 0.5 * table.size[2])


def _table_top_named(task, table_name: str = "table") -> float:
    table = task.layout.actors[table_name]
    return float(table.pose.position[2] + 0.5 * table.size[2])


def _stable_z(actor: Actor) -> float:
    if isinstance(actor, ObjectActor):
        return float(actor.asset.stable_poses[0][2])
    return 0.0


def _load_collision_points(actor: Actor) -> np.ndarray:
    assert isinstance(actor, ObjectActor)
    points: list[np.ndarray] = []
    for mesh_path in actor.asset.collision_meshes_mujoco:
        mesh = trimesh.load_mesh(Path(mesh_path), force="mesh", process=False)
        if mesh.is_empty or len(mesh.vertices) == 0:
            continue
        points.append(np.asarray(mesh.vertices, dtype=np.float64))
    if not points:
        raise RuntimeError(f"No collision vertices found for {actor.uid!r}.")
    return np.vstack(points)


def _object_points_in_body_local(
    mujoco_env,
    actor: Actor,
    body_name: str,
    points: np.ndarray,
) -> np.ndarray | None:
    object_pose = _body_pose(mujoco_env, _actor_body_name(actor))
    body_pose = _body_pose(mujoco_env, body_name)
    if object_pose is None or body_pose is None:
        return None
    object_pos, object_rot = object_pose
    body_pos, body_rot = body_pose
    world_points = points @ object_rot.T + object_pos
    return (world_points - body_pos) @ body_rot


def _points_inside(points: np.ndarray | None, low: np.ndarray, high: np.ndarray) -> bool:
    if points is None:
        return False
    return bool(np.all(points >= low) and np.all(points <= high))


class _LongTableObjectToDrawerTask(G1WholebodyXMovePickTaskTeleop):
    metadata: dict[str, Any] = deepcopy(G1WholebodyXMovePickTaskTeleop.metadata)
    dr_cfgs = deepcopy(G1WholebodyXMovePickTaskTeleop.dr_cfgs)
    dr_cfgs["articulated"] = ArticulatedObjectDrCfg(
        asset_id=_DRAWER_ASSET_ID,
        articulate_init_joint_qpos={_DRAWER_JOINT: 0.0},
    )
    dr_cfgs["distractors"] = DistractorDRCfg(
        res_id="objects",
        number_of_distractors=0,
        allow_duplicates=False,
    )
    dr_cfgs["spatial"] = SpatialDRCfg(
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
    dr_cfgs["scene"] = TabletopSceneDRCfg(
        scene_mode="fixed",
        table_position=Box(low=[0.3, 0.0], high=[0.3, 0.0]),
        table_height=Box(low=_HIGH_TABLE_HEIGHT, high=_HIGH_TABLE_HEIGHT),
        room_choices=["hssd:scene0"],
        scene_manager="hssd",
    )

    target_asset_id: str
    target_load_id: str
    instruction: str

    def __init__(self, *args, target: str | None = None, hold_steps: int = 10, **kwargs) -> None:
        self.dr_cfgs = deepcopy(type(self).dr_cfgs)
        self.dr_cfgs["target"] = TargetDRCfg(asset_id=type(self).target_asset_id)
        self.dr_cfgs["language"] = LanguageDRCfg(instructions=[type(self).instruction])
        self.hold_steps = int(hold_steps)
        self._success_hold_count = 0
        self._target_collision_points: np.ndarray | None = None
        super().__init__(*args, target=target, **kwargs)

    @property
    def layout(self) -> Layout:
        assert self._layout is not None, "call reset() first"
        return self._layout

    @property
    def action_space(self) -> spaces.Space:
        assert isinstance(self.robot, Controllable)
        return self.robot.controller.action_space

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> None:
        super().reset(seed, options)
        self._success_hold_count = 0
        self.reward = 0.0

    def _drawer_qpos(self, mujoco_env=None) -> float:
        if mujoco_env is None:
            return 0.0
        return float(mujoco_env.mjData.joint(_DRAWER_JOINT).qpos[0])

    def _target_points_in_drawer_local(self, mujoco_env=None) -> np.ndarray | None:
        if mujoco_env is None:
            return None
        if self._target_collision_points is None:
            self._target_collision_points = _load_collision_points(self.target)
        return _object_points_in_body_local(
            mujoco_env,
            self.target,
            _DRAWER_BODY,
            self._target_collision_points,
        )

    def _target_in_drawer(self, info: dict[str, Any], mujoco_env=None) -> bool:
        del info
        local_points = self._target_points_in_drawer_local(mujoco_env=mujoco_env)
        if local_points is None:
            return False
        return bool(
            np.all(local_points >= _DRAWER_LOCAL_LOW) and np.all(local_points <= _DRAWER_LOCAL_HIGH)
        )

    def _target_contacts_drawer_inner_bottom(self, mujoco_env=None) -> bool:
        if mujoco_env is None:
            return False
        drawer_pose = _body_pose(mujoco_env, _DRAWER_BODY)
        if drawer_pose is None:
            return False
        drawer_position, drawer_rotation = drawer_pose
        target_name = _actor_body_name(self.target)
        model = mujoco_env.mjModel
        data = mujoco_env.mjData
        bottom_low, bottom_high = _DRAWER_INNER_BOTTOM_LOCAL_Z_RANGE

        for contact_id in range(data.ncon):
            contact = data.contact[contact_id]
            geom1 = model.geom(int(contact.geom1))
            geom2 = model.geom(int(contact.geom2))
            body1 = _geom_body_name(model, geom1)
            body2 = _geom_body_name(model, geom2)
            target_drawer_pair = (
                target_name in body1
                and _DRAWER_BODY in body2
                or target_name in body2
                and _DRAWER_BODY in body1
            )
            if not target_drawer_pair:
                continue
            local_contact = (
                np.asarray(contact.pos, dtype=np.float64) - drawer_position
            ) @ drawer_rotation
            if (
                _DRAWER_LOCAL_LOW[0] <= float(local_contact[0]) <= _DRAWER_LOCAL_HIGH[0]
                and _DRAWER_LOCAL_LOW[1] <= float(local_contact[1]) <= _DRAWER_LOCAL_HIGH[1]
                and bottom_low <= float(local_contact[2]) <= bottom_high
            ):
                return True
        return False

    def check_object_in_drawer(self, info: dict[str, Any] | None = None, *args, **kwargs) -> bool:
        mujoco_env = kwargs.get("mujoco_env", None)
        if mujoco_env is None or info is None or "target" not in info:
            return False
        drawer_open = -self._drawer_qpos(mujoco_env=mujoco_env) >= _DRAWER_OPEN_THRESHOLD_M
        return bool(
            drawer_open
            and self._target_in_drawer(info, mujoco_env=mujoco_env)
            and self._target_contacts_drawer_inner_bottom(mujoco_env=mujoco_env)
            and not _hand_contact(mujoco_env, self.target)
        )

    def check_object_in_container(self, *args, **kwargs) -> bool:
        return self.check_object_in_drawer(*args, **kwargs)

    def compute_reward(self, info: dict[str, Any], *args, **kwargs) -> float:
        if self.check_object_in_drawer(info, *args, **kwargs):
            self._success_hold_count += 1
        else:
            self._success_hold_count = 0
        self.reward = _SUCCESS_REWARD_INCREMENT * self._success_hold_count
        return float(self.reward)

    def check_success(self, info: dict[str, Any], *args, **kwargs) -> bool:
        del info, args, kwargs
        return self._success_hold_count >= self.hold_steps

    def preload_objects(self) -> list[Actor]:
        asset = AssetManager.get("objects").load(type(self).target_load_id)
        return [ObjectActor(asset=asset)]


@TaskRegistry.register("g1_wholebody_long_table_apple_to_drawer_teleop")
class G1WholebodyLongTableAppleToDrawerTaskTeleop(_LongTableObjectToDrawerTask):
    uid: str = "g1_wholebody_long_table_apple_to_drawer_teleop"
    label: str = "G1 TELEOP Long Table Apple To Drawer"
    description: str = "A task where the G1 robot must pick up the apple on the table and place it into the open drawer."

    target_asset_id = _APPLE_ASSET_ID
    target_load_id = _APPLE_LOAD_ID
    instruction = "open the drawer and put the apple from the table into the drawer"


@TaskRegistry.register("g1_wholebody_long_table_plate_to_drawer_teleop")
class G1WholebodyLongTablePlateToDrawerTaskTeleop(_LongTableObjectToDrawerTask):
    uid: str = "g1_wholebody_long_table_plate_to_drawer_teleop"
    label: str = "G1 TELEOP Long Table Plate To Drawer"
    description: str = "A task where the G1 robot must pick up the plate on the table and place it into the open drawer."

    target_asset_id = _PLATE_ASSET_ID
    target_load_id = _PLATE_LOAD_ID
    instruction = "open the drawer and put the plate from the table into the drawer"


@TaskRegistry.register("g1_wholebody_long_table_pencil_eraser_to_pencil_case_teleop")
class G1WholebodyLongTablePencilEraserToPencilCaseTaskTeleop(G1WholebodyXMovePickTaskTeleop):
    uid: str = "g1_wholebody_long_table_pencil_eraser_to_pencil_case_teleop"
    label: str = "G1 TELEOP Long Table Pencil Eraser To Pencil Case"
    description: str = "Put both the pencil and the eraser fully inside the open pencil case."

    metadata: dict[str, Any] = deepcopy(G1WholebodyXMovePickTaskTeleop.metadata)
    dr_cfgs = deepcopy(G1WholebodyXMovePickTaskTeleop.dr_cfgs)
    dr_cfgs["language"] = LanguageDRCfg(
        instructions=[
            "put the pencil and the eraser on the table fully inside the open pencil case",
        ]
    )
    dr_cfgs["target"] = TargetDRCfg(asset_id=_PENCIL_ASSET_ID)
    dr_cfgs["articulated"] = ArticulatedObjectDrCfg(
        asset_id=_PENCIL_CASE_ASSET_ID,
        articulate_init_joint_qpos=_PENCIL_CASE_JOINT_QPOS,
    )
    dr_cfgs["distractors"] = DistractorDRCfg(
        res_id="objects",
        number_of_distractors=1,
        allow_duplicates=False,
        include=[_ERASER_LOAD_ID],
    )
    dr_cfgs["spatial"] = SpatialDRCfg(
        spatial_mode="fixed",
        robot_region=Box(low=[-0.95, 0.0, 0.0], high=[-0.95, 0.0, 0.0]),
        target_region=Box(low=[-0.275, -0.20], high=[-0.275, -0.20]),
        distractors_region=Box(low=[-0.12, -0.22], high=[-0.12, -0.22]),
        articulated_region=Box(low=_PENCIL_CASE_POSITION, high=_PENCIL_CASE_POSITION),
        target_stable_indices=[0],
        target_rotate_z=Box(low=0.0, high=0.0),
        distractor_rotate_z=Box(low=0.0, high=0.0),
        articulated_rotate_z=Box(low=_PENCIL_CASE_ROTATE_Z, high=_PENCIL_CASE_ROTATE_Z),
        fixed_stable_pose_idx=0,
        obj_surface_map={"target": "table", "distractor": "table"},
    )
    dr_cfgs["scene"] = TabletopSceneDRCfg(
        scene_mode="fixed",
        table_position=Box(low=[0.3, 0.0], high=[0.3, 0.0]),
        table_height=Box(low=_HIGH_TABLE_HEIGHT, high=_HIGH_TABLE_HEIGHT),
        room_choices=["hssd:scene0"],
        scene_manager="hssd",
    )

    def __init__(self, *args, target: str | None = None, hold_steps: int = 10, **kwargs) -> None:
        self.dr_cfgs = deepcopy(type(self).dr_cfgs)
        super().__init__(*args, target=target, **kwargs)
        self.hold_steps = int(hold_steps)
        self._success_hold_count = 0
        self._pencil_points: np.ndarray | None = None
        self._eraser_points: np.ndarray | None = None

    @property
    def layout(self) -> Layout:
        assert self._layout is not None, "call reset() first"
        return self._layout

    @property
    def pencil_case(self) -> Actor:
        actor = self.layout.actors.get("articulated")
        assert actor is not None, "call reset() first"
        return actor

    @property
    def eraser(self) -> Actor:
        actor = self.layout.actors.get("distractor_0")
        assert actor is not None, "call reset() first"
        return actor

    @property
    def action_space(self) -> spaces.Space:
        assert isinstance(self.robot, Controllable)
        return self.robot.controller.action_space

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> None:
        super().reset(seed, options)
        self._success_hold_count = 0
        self.reward = 0.0

    def _object_inside_case(self, actor: Actor, points_cache: np.ndarray, mujoco_env) -> bool:
        points = _object_points_in_body_local(
            mujoco_env,
            actor,
            _PENCIL_CASE_BASE_BODY,
            points_cache,
        )
        return _points_inside(points, _PENCIL_CASE_LOCAL_LOW, _PENCIL_CASE_LOCAL_HIGH)

    def check_objects_in_case(self, info: dict[str, Any] | None = None, *args, **kwargs) -> bool:
        del info, args
        mujoco_env = kwargs.get("mujoco_env", None)
        if mujoco_env is None:
            return False
        if self._pencil_points is None:
            self._pencil_points = _load_collision_points(self.target)
        if self._eraser_points is None:
            self._eraser_points = _load_collision_points(self.eraser)
        return bool(
            self._object_inside_case(self.target, self._pencil_points, mujoco_env)
            and self._object_inside_case(self.eraser, self._eraser_points, mujoco_env)
            and not _hand_contact(mujoco_env, self.target)
            and not _hand_contact(mujoco_env, self.eraser)
        )

    def check_object_in_container(self, *args, **kwargs) -> bool:
        return self.check_objects_in_case(*args, **kwargs)

    def compute_reward(self, info: dict[str, Any], *args, **kwargs) -> float:
        if self.check_objects_in_case(info, *args, **kwargs):
            self._success_hold_count += 1
        else:
            self._success_hold_count = 0
        self.reward = _SUCCESS_REWARD_INCREMENT * self._success_hold_count
        return float(self.reward)

    def check_success(self, info: dict[str, Any], *args, **kwargs) -> bool:
        del info, args, kwargs
        return self._success_hold_count >= self.hold_steps

    def preload_objects(self) -> list[Actor]:
        objects = AssetManager.get("objects")
        return [
            ObjectActor(asset=objects.load(_PENCIL_LOAD_ID)),
            ObjectActor(asset=objects.load(_ERASER_LOAD_ID)),
        ]


@TaskRegistry.register("g1_wholebody_long_table_pencil_pen_to_pen_holder_teleop")
class G1WholebodyLongTablePencilPenToPenHolderTaskTeleop(G1WholebodyXMovePickTaskTeleop):
    uid: str = "g1_wholebody_long_table_pencil_pen_to_pen_holder_teleop"
    label: str = "G1 TELEOP Long Table Pencil Pen To Pen Holder"
    description: str = "Put both the pencil and the pen into the pen holder."

    metadata: dict[str, Any] = deepcopy(G1WholebodyXMovePickTaskTeleop.metadata)
    dr_cfgs = deepcopy(G1WholebodyXMovePickTaskTeleop.dr_cfgs)
    dr_cfgs["language"] = LanguageDRCfg(
        instructions=[
            "put the pencil and the pen on the table into the pen holder",
        ]
    )
    dr_cfgs["target"] = TargetDRCfg(asset_id=_PENCIL_ASSET_ID)
    dr_cfgs["container"] = TargetDRCfg(asset_id=_PEN_HOLDER_ASSET_ID)
    dr_cfgs["distractors"] = DistractorDRCfg(
        res_id="objects",
        number_of_distractors=1,
        allow_duplicates=False,
        include=[_PEN_LOAD_ID],
    )
    dr_cfgs["spatial"] = SpatialDRCfg(
        spatial_mode="fixed",
        robot_region=Box(low=[-0.95, 0.0, 0.0], high=[-0.95, 0.0, 0.0]),
        target_region=Box(low=[-0.275, -0.20], high=[-0.275, -0.20]),
        container_region=Box(low=_PEN_HOLDER_XY, high=_PEN_HOLDER_XY),
        distractors_region=Box(low=[-0.16, -0.24], high=[-0.16, -0.24]),
        target_stable_indices=[0],
        target_rotate_z=Box(low=0.0, high=0.0),
        container_rotate_z=Box(low=0.0, high=0.0),
        fixed_stable_pose_idx=0,
        obj_surface_map={"target": "table", "container": "table", "distractor": "table"},
    )
    dr_cfgs["scene"] = TabletopSceneDRCfg(
        scene_mode="fixed",
        table_position=Box(low=[0.3, 0.0], high=[0.3, 0.0]),
        table_height=Box(low=_HIGH_TABLE_HEIGHT, high=_HIGH_TABLE_HEIGHT),
        room_choices=["hssd:scene0"],
        scene_manager="hssd",
    )

    def __init__(self, *args, target: str | None = None, hold_steps: int = 10, **kwargs) -> None:
        self.dr_cfgs = deepcopy(type(self).dr_cfgs)
        super().__init__(*args, target=target, **kwargs)
        self.hold_steps = int(hold_steps)
        self._success_hold_count = 0

    @property
    def layout(self) -> Layout:
        assert self._layout is not None, "call reset() first"
        return self._layout

    @property
    def container(self) -> Actor:
        actor = self.layout.actors.get("container")
        assert actor is not None, "call reset() first"
        return actor

    @property
    def pen(self) -> Actor:
        actor = self.layout.actors.get("distractor_0")
        assert actor is not None, "call reset() first"
        return actor

    @property
    def action_space(self) -> spaces.Space:
        assert isinstance(self.robot, Controllable)
        return self.robot.controller.action_space

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> None:
        super().reset(seed, options)
        self._success_hold_count = 0
        self.reward = 0.0

    def _object_in_pen_holder(
        self,
        info: dict[str, Any],
        key: str,
        actor: Actor,
        long_axis_local: np.ndarray,
    ) -> bool:
        object_position = _info_position(info, key, actor)
        holder_position = _info_position(info, "container", self.container)
        holder_quat = _info_quaternion(info, "container", self.container)
        holder_rotation = t3d.quaternions.quat2mat(holder_quat)
        local_position = holder_rotation.T @ (object_position - holder_position)
        object_quat = _info_quaternion(info, key, actor)
        object_rotation = t3d.quaternions.quat2mat(object_quat)
        long_axis_world = object_rotation @ long_axis_local
        vertical_cos = float(abs(np.dot(long_axis_world, np.array([0.0, 0.0, 1.0]))))
        return bool(
            np.linalg.norm(local_position[:2]) <= _PEN_HOLDER_RADIUS_M
            and _PEN_HOLDER_MIN_Z_M <= float(local_position[2]) <= _PEN_HOLDER_MAX_Z_M
            and vertical_cos >= float(np.cos(_PEN_HOLDER_VERTICAL_TILT_RAD))
        )

    def check_objects_in_pen_holder(
        self,
        info: dict[str, Any] | None = None,
        *args,
        **kwargs,
    ) -> bool:
        del args
        mujoco_env = kwargs.get("mujoco_env", None)
        if (
            info is None
            or "target" not in info
            or "container" not in info
            or "distractor_0" not in info
        ):
            return False
        return bool(
            self._object_in_pen_holder(
                info,
                "target",
                self.target,
                np.array([0.0, 1.0, 0.0], dtype=np.float64),
            )
            and self._object_in_pen_holder(
                info,
                "distractor_0",
                self.pen,
                np.array([1.0, 0.0, 0.0], dtype=np.float64),
            )
            and not _hand_contact(mujoco_env, self.target)
            and not _hand_contact(mujoco_env, self.pen)
        )

    def check_object_in_container(self, *args, **kwargs) -> bool:
        return self.check_objects_in_pen_holder(*args, **kwargs)

    def compute_reward(self, info: dict[str, Any], *args, **kwargs) -> float:
        if self.check_objects_in_pen_holder(info, *args, **kwargs):
            self._success_hold_count += 1
        else:
            self._success_hold_count = 0
        self.reward = _SUCCESS_REWARD_INCREMENT * self._success_hold_count
        return float(self.reward)

    def check_success(self, info: dict[str, Any], *args, **kwargs) -> bool:
        del info, args, kwargs
        return self._success_hold_count >= self.hold_steps

    def preload_objects(self) -> list[Actor]:
        objects = AssetManager.get("objects")
        return [
            ObjectActor(asset=objects.load(_PENCIL_LOAD_ID)),
            ObjectActor(asset=objects.load(_PEN_HOLDER_LOAD_ID)),
            ObjectActor(asset=objects.load(_PEN_LOAD_ID)),
        ]


@TaskRegistry.register("g1_wholebody_long_ground_box_to_dolly_and_push_teleop")
class G1WholebodyLongGroundBoxToDollyAndPushTaskTeleop(G1WholebodyDollyPushTaskTeleop):
    uid: str = "g1_wholebody_long_ground_box_to_dolly_and_push_teleop"
    label: str = "G1 TELEOP Long Ground Box To Dolly And Push"
    description: str = "A task where the G1 robot must pick up a box from the ground, place it on the dolly, then push the dolly forward."

    metadata: dict[str, Any] = deepcopy(G1WholebodyDollyPushTaskTeleop.metadata)
    dr_cfgs = deepcopy(G1WholebodyDollyPushTaskTeleop.dr_cfgs)
    dr_cfgs["language"] = LanguageDRCfg(
        instructions=[
            "pick up the box from the ground, put it on the dolly, and push the dolly forward",
        ]
    )
    dr_cfgs["target"] = TargetDRCfg(asset_id=_BOX_ASSET_ID)
    dr_cfgs["spatial"] = deepcopy(G1WholebodyDollyPushTaskTeleop.dr_cfgs["spatial"])
    spatial_cfg = dr_cfgs["spatial"]
    assert isinstance(spatial_cfg, SpatialDRCfg)
    spatial_cfg.target_region = Box(low=[-0.60, 0.0], high=[-0.60, 0.0])
    spatial_cfg.articulated_region = Box(
        low=_BOX_TO_DOLLY_INITIAL_DOLLY_POSITION,
        high=_BOX_TO_DOLLY_INITIAL_DOLLY_POSITION,
    )
    spatial_cfg.target_stable_indices = [0]
    spatial_cfg.target_rotate_z = Box(low=0.0, high=0.0)
    spatial_cfg.articulated_rotate_z = Box(
        low=_BOX_TO_DOLLY_INITIAL_DOLLY_YAW,
        high=_BOX_TO_DOLLY_INITIAL_DOLLY_YAW,
    )
    spatial_cfg.obj_surface_map = {"target": "default", "articulated": "default"}

    def __init__(
        self,
        *args,
        move_threshold_m: float = _BOX_DOLLY_PUSH_THRESHOLD_M,
        hold_steps: int = 10,
        **kwargs,
    ) -> None:
        self._target = None
        super().__init__(*args, move_threshold_m=move_threshold_m, hold_steps=hold_steps, **kwargs)

    @property
    def target(self) -> Actor:
        assert self._target is not None, "call reset() first"
        return self._target

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> None:
        super().reset(seed, options)
        self._target = self.layout.actors.get("target")
        self._push_hold_count = 0
        self.reward = 0.0

    def _box_on_dolly(self, info: dict[str, Any], mujoco_env=None) -> bool:
        del info
        return _contact_with_body_tree(
            mujoco_env,
            _actor_body_name(self.target),
            _DOLLY_BASE_BODY,
        )

    def _is_pushed_now(self, metrics: dict[str, float]) -> bool:
        return bool(metrics["displacement"] >= self.move_threshold_m)

    def compute_reward(self, info: dict[str, Any], *args, **kwargs) -> float:
        del args
        mujoco_env = kwargs.get("mujoco_env")
        dolly_metrics = self._dolly_metrics(mujoco_env=mujoco_env)
        if self._box_on_dolly(info, mujoco_env=mujoco_env) and self._is_pushed_now(dolly_metrics):
            self._push_hold_count += 1
        else:
            self._push_hold_count = 0
        move_score = np.clip(
            dolly_metrics["displacement"] / max(self.move_threshold_m, 1e-6),
            0.0,
            1.0,
        )
        hold_score = np.clip(self._push_hold_count / max(1, self.hold_steps), 0.0, 1.0)
        self.reward = float(min(move_score, hold_score))
        return self.reward

    def check_success(self, info: dict[str, Any], *args, **kwargs) -> bool:
        del info, args, kwargs
        return self._push_hold_count >= self.hold_steps

    def preload_objects(self) -> list[Actor]:
        return [ObjectActor(asset=AssetManager.get("objects").load(_BOX_LOAD_ID))]


@TaskRegistry.register("g1_wholebody_long_chair_push_and_sit_teleop")
class G1WholebodyLongChairPushAndSitTaskTeleop(G1WholebodyChairPushTaskTeleop):
    uid: str = "g1_wholebody_long_chair_push_and_sit_teleop"
    label: str = "G1 TELEOP Long Chair Push And Sit"
    description: str = (
        "A task where the G1 robot must push the chair forward and then sit down on it."
    )

    metadata: dict[str, Any] = deepcopy(G1WholebodyChairPushTaskTeleop.metadata)
    dr_cfgs = deepcopy(G1WholebodyChairPushTaskTeleop.dr_cfgs)
    dr_cfgs["language"] = LanguageDRCfg(
        instructions=[
            "push the chair forward and then sit on it",
        ]
    )

    def __init__(
        self,
        *args,
        move_threshold_m: float = _CHAIR_PUSH_THRESHOLD_M,
        hold_steps: int = 30,
        **kwargs,
    ) -> None:
        self._chair_pushed_phase = False
        super().__init__(*args, move_threshold_m=move_threshold_m, hold_steps=hold_steps, **kwargs)

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> None:
        super().reset(seed, options)
        self._chair_pushed_phase = False
        self._push_hold_count = 0
        if isinstance(self._target, ObjectActor):
            self._target.mujoco_friction = _SIT_CHAIR_MUJOCO_FRICTION
            self._target.mujoco_mass = _PUSH_CHAIR_MUJOCO_MASS_KG

    def _chair_position(self, info: dict[str, Any] | None = None, mujoco_env=None) -> np.ndarray:
        if mujoco_env is not None and "target" in getattr(mujoco_env, "mj_objects", {}):
            chair_body = mujoco_env.mj_objects["target"]
            return np.asarray(chair_body.xpos, dtype=np.float32)
        if info is not None and "target" in info:
            return np.asarray(info["target"][:3], dtype=np.float32)
        return np.asarray(self.layout.actors["target"].pose.position, dtype=np.float32)

    def _sitting_now(self, info: dict[str, Any], mujoco_env=None) -> bool:
        if mujoco_env is None:
            return False
        chair_pos = self._chair_position(info=info, mujoco_env=mujoco_env)
        pelvis_pos = np.asarray(mujoco_env.mjData.body("pelvis").xpos, dtype=np.float64)
        qvel = np.asarray(mujoco_env.mjData.qvel, dtype=np.float64)
        xy_dist = float(np.linalg.norm(pelvis_pos[:2] - chair_pos[:2]))
        pelvis_z = float(pelvis_pos[2])
        linear_speed = float(np.linalg.norm(qvel[:3]))
        return bool(
            xy_dist <= _CHAIR_SIT_XY_RADIUS_M
            and _CHAIR_SIT_PELVIS_Z_RANGE_M[0] <= pelvis_z <= _CHAIR_SIT_PELVIS_Z_RANGE_M[1]
            and linear_speed <= 0.15
        )

    def compute_reward(self, info: dict[str, Any], *args, **kwargs) -> float:
        del args
        mujoco_env = kwargs.get("mujoco_env")
        metrics = self._chair_metrics(info, mujoco_env=mujoco_env)
        if self._is_pushed_now(metrics):
            self._chair_pushed_phase = True
        if self._chair_pushed_phase and self._sitting_now(info, mujoco_env=mujoco_env):
            self._push_hold_count += 1
        else:
            self._push_hold_count = 0
        push_score = np.clip(metrics["displacement"] / self.move_threshold_m, 0.0, 1.0)
        hold_score = np.clip(self._push_hold_count / max(1, self.hold_steps), 0.0, 1.0)
        return float(min(push_score, hold_score))

    def check_success(self, info: dict[str, Any], *args, **kwargs) -> bool:
        del info, args, kwargs
        return self._push_hold_count >= self.hold_steps


@TaskRegistry.register("g1_wholebody_long_table_football_to_ground_and_kick_teleop")
class G1WholebodyLongTableFootballToGroundAndKickTaskTeleop(G1WholebodyBallKickTaskTeleop):
    uid: str = "g1_wholebody_long_table_football_to_ground_and_kick_teleop"
    label: str = "G1 TELEOP Long Table Football To Ground And Kick"
    description: str = "A task where the G1 robot must move the football from the table to the ground and then kick it."
    _ball_center_xy = _LONG_FOOTBALL_CENTER_XY

    metadata: dict[str, Any] = deepcopy(G1WholebodyBallKickTaskTeleop.metadata)
    dr_cfgs = deepcopy(G1WholebodyBallKickTaskTeleop.dr_cfgs)
    dr_cfgs["language"] = LanguageDRCfg(
        instructions=[
            "move the football from the table to the ground and kick it forward",
        ]
    )
    dr_cfgs["target"] = TargetDRCfg(asset_id=_FOOTBALL_ASSET_ID)
    spatial_cfg = dr_cfgs["spatial"]
    assert isinstance(spatial_cfg, SpatialDRCfg)
    spatial_cfg.target_region = Box(
        low=list(_LONG_FOOTBALL_CENTER_XY),
        high=list(_LONG_FOOTBALL_CENTER_XY),
    )
    spatial_cfg.target_rotate_z = Box(low=_LONG_FOOTBALL_YAW, high=_LONG_FOOTBALL_YAW)
    spatial_cfg.obj_surface_map = {"target": "table", "distractor": "table"}
    dr_cfgs["scene"] = TabletopSceneDRCfg(
        scene_mode="fixed",
        table_position=Box(low=[0.3, 0.0], high=[0.3, 0.0]),
        table_height=Box(low=_HIGH_TABLE_HEIGHT, high=_HIGH_TABLE_HEIGHT),
        room_choices=["hssd:scene0"],
        scene_manager="hssd",
    )

    def __init__(self, *args, move_threshold_m: float = _BALL_KICK_DISTANCE_M, **kwargs) -> None:
        self._ball_grounded_phase = False
        self._ground_ball_xy: np.ndarray | None = None
        super().__init__(*args, move_threshold_m=move_threshold_m, **kwargs)

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> None:
        super().reset(seed, options)
        self._ball_grounded_phase = False
        self._ground_ball_xy = None

    def _ball_on_ground(self, info: dict[str, Any], mujoco_env=None) -> bool:
        if "target" not in info:
            return False
        target_position = np.asarray(info["target"][:3], dtype=np.float64)
        near_ground = float(target_position[2]) <= _BALL_GROUND_Z_MAX_M
        return bool(
            near_ground or _contact_between(mujoco_env, _actor_body_name(self.target), "floor")
        )

    def compute_reward(self, info: dict[str, Any], *args, **kwargs) -> float:
        del args
        mujoco_env = kwargs.get("mujoco_env")
        target_xy = np.asarray(info["target"][:2], dtype=np.float64)
        if self._ball_on_ground(info, mujoco_env=mujoco_env) and not self._ball_grounded_phase:
            self._ball_grounded_phase = True
            self._ground_ball_xy = target_xy.copy()
        if not self._ball_grounded_phase or self._ground_ball_xy is None:
            return 0.0
        displacement = float(np.linalg.norm(target_xy - self._ground_ball_xy))
        return float(np.clip(displacement / self.move_threshold_m, 0.0, 1.0))

    def check_success(self, info: dict[str, Any], *args, **kwargs) -> bool:
        return self.compute_reward(info, *args, **kwargs) >= 1.0

    def preload_objects(self) -> list[Actor]:
        return [ObjectActor(asset=AssetManager.get("objects").load(_FOOTBALL_LOAD_ID))]


@TaskRegistry.register("g1_wholebody_long_ground_basket_to_table_apple_to_basket_teleop")
class G1WholebodyLongGroundBasketToTableAppleToBasketTaskTeleop(G1WholebodyXMovePickTaskTeleop):
    uid: str = "g1_wholebody_long_ground_basket_to_table_apple_to_basket_teleop"
    label: str = "G1 TELEOP Long Ground Basket To Table Apple To Basket"
    description: str = "A task where the G1 robot must move the basket from the ground to the table, then put the apple into it."

    metadata: dict[str, Any] = deepcopy(G1WholebodyXMovePickTaskTeleop.metadata)
    dr_cfgs = deepcopy(G1WholebodyXMovePickTaskTeleop.dr_cfgs)
    dr_cfgs["language"] = LanguageDRCfg(
        instructions=[
            "put the basket from the ground onto the table, then put the apple on the table into the basket",
        ]
    )
    dr_cfgs["target"] = TargetDRCfg(asset_id=_BASKET_ASSET_ID)
    dr_cfgs["container"] = TargetDRCfg(asset_id=_APPLE_ASSET_ID)
    dr_cfgs["distractors"] = DistractorDRCfg(
        res_id="objects",
        number_of_distractors=0,
        allow_duplicates=False,
    )
    dr_cfgs["spatial"] = SpatialDRCfg(
        spatial_mode="fixed",
        robot_region=Box(low=[-0.95, 0.0, 0.0], high=[-0.95, 0.0, 0.0]),
        target_region=Box(low=_BASKET_GROUND_XY, high=_BASKET_GROUND_XY),
        container_region=Box(low=_BASKET_APPLE_XY, high=_BASKET_APPLE_XY),
        distractors_region=Box(low=[0.0, 0.0], high=[0.0, 0.0]),
        target_stable_indices=[0],
        target_rotate_z=Box(low=0.0, high=0.0),
        container_rotate_z=Box(low=0.0, high=0.0),
        fixed_stable_pose_idx=0,
        obj_surface_map={"target": "default", "container": "table"},
    )
    dr_cfgs["scene"] = TabletopSceneDRCfg(
        scene_mode="fixed",
        table_position=Box(low=_BASKET_TABLE_XY, high=_BASKET_TABLE_XY),
        table_height=Box(low=_LOW_TABLE_HEIGHT, high=_LOW_TABLE_HEIGHT),
        room_choices=["hssd:scene0"],
        scene_manager="hssd",
    )

    def __init__(self, *args, target: str | None = None, hold_steps: int = 10, **kwargs) -> None:
        self.dr_cfgs = deepcopy(type(self).dr_cfgs)
        super().__init__(*args, target=target, **kwargs)
        self.hold_steps = int(hold_steps)
        self._success_hold_count = 0
        self._basket_on_table_phase = False

    @property
    def layout(self) -> Layout:
        assert self._layout is not None, "call reset() first"
        return self._layout

    @property
    def apple(self) -> Actor:
        actor = self.layout.actors.get("container")
        assert actor is not None, "call reset() first"
        return actor

    @property
    def action_space(self) -> spaces.Space:
        assert isinstance(self.robot, Controllable)
        return self.robot.controller.action_space

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> None:
        super().reset(seed, options)
        self._success_hold_count = 0
        self._basket_on_table_phase = False
        self.reward = 0.0
        if isinstance(self._target, ObjectActor):
            stable_pose = self._target.asset.stable_poses[0]
            self._target.pose.position = [*_BASKET_GROUND_XY, float(stable_pose[2])]
            self._target.pose.quaternion = [float(v) for v in stable_pose[3:]]
            self._target.mujoco_mass = _BASKET_MUJOCO_MASS_KG

    def _basket_on_table(self, info: dict[str, Any], mujoco_env=None) -> bool:
        if "target" not in info:
            return False
        basket_position = np.asarray(info["target"][:3], dtype=np.float64)
        table_top = _table_top(self)
        min_height = table_top + _stable_z(self.target) - 0.04
        xy = np.asarray(_BASKET_PLACE_XY, dtype=np.float64)
        xy_close = float(np.linalg.norm(basket_position[:2] - xy)) <= 0.30
        high_enough = float(basket_position[2]) >= min_height
        return bool(xy_close and high_enough and not _hand_contact(mujoco_env, self.target))

    def _apple_in_basket(self, info: dict[str, Any], mujoco_env=None) -> bool:
        if "target" not in info or "container" not in info:
            return False
        basket_position = np.asarray(info["target"][:3], dtype=np.float64)
        apple_position = np.asarray(info["container"][:3], dtype=np.float64)
        basket_quat = np.asarray(info["target"][3:7], dtype=np.float64)
        basket_rotation = t3d.quaternions.quat2mat(basket_quat)
        apple_local = basket_rotation.T @ (apple_position - basket_position)
        return bool(
            np.all(np.abs(apple_local[:2]) <= _BASKET_INSIDE_HALF_EXTENTS)
            and _BASKET_INSIDE_Z_RANGE[0] <= float(apple_local[2]) <= _BASKET_INSIDE_Z_RANGE[1]
            and _contact_between(
                mujoco_env, _actor_body_name(self.apple), _actor_body_name(self.target)
            )
            and not _hand_contact(mujoco_env, self.apple)
        )

    def check_basket_and_apple_success(
        self,
        info: dict[str, Any] | None = None,
        *args,
        **kwargs,
    ) -> bool:
        del args
        if info is None:
            return False
        mujoco_env = kwargs.get("mujoco_env", None)
        if self._basket_on_table(info, mujoco_env=mujoco_env):
            self._basket_on_table_phase = True
        return bool(
            self._basket_on_table_phase
            and self._basket_on_table(info, mujoco_env=mujoco_env)
            and self._apple_in_basket(info, mujoco_env=mujoco_env)
        )

    def check_object_in_container(self, *args, **kwargs) -> bool:
        return self.check_basket_and_apple_success(*args, **kwargs)

    def compute_reward(self, info: dict[str, Any], *args, **kwargs) -> float:
        if self.check_basket_and_apple_success(info, *args, **kwargs):
            self._success_hold_count += 1
        else:
            self._success_hold_count = 0
        self.reward = _SUCCESS_REWARD_INCREMENT * self._success_hold_count
        return float(self.reward)

    def check_success(self, info: dict[str, Any], *args, **kwargs) -> bool:
        del info, args, kwargs
        return self._success_hold_count >= self.hold_steps

    def preload_objects(self) -> list[Actor]:
        objects = AssetManager.get("objects")
        return [
            ObjectActor(asset=objects.load(_BASKET_LOAD_ID)),
            ObjectActor(asset=objects.load(_APPLE_LOAD_ID)),
        ]


@TaskRegistry.register("g1_wholebody_long_ground_box_to_table_bottle_to_box_teleop")
class G1WholebodyLongGroundBoxToTableBottleToBoxTaskTeleop(G1WholebodyXMovePickTaskTeleop):
    uid: str = "g1_wholebody_long_ground_box_to_table_bottle_to_box_teleop"
    label: str = "G1 TELEOP Long Ground Bottle To Box And Lift"
    description: str = "A task where the G1 robot must put a bottle from the ground into a box on the table, then lift the box."

    metadata: dict[str, Any] = deepcopy(G1WholebodyXMovePickTaskTeleop.metadata)
    dr_cfgs = deepcopy(G1WholebodyXMovePickTaskTeleop.dr_cfgs)
    dr_cfgs["language"] = LanguageDRCfg(
        instructions=[
            "put the bottle from the ground into the box on the table, then lift the box",
        ]
    )
    dr_cfgs["target"] = TargetDRCfg(asset_id=_STORAGE_BOX_ASSET_ID)
    dr_cfgs["container"] = TargetDRCfg(asset_id=_BOTTLE_ASSET_ID)
    dr_cfgs["distractors"] = DistractorDRCfg(
        res_id="objects", number_of_distractors=0, allow_duplicates=False
    )
    dr_cfgs["spatial"] = SpatialDRCfg(
        spatial_mode="fixed",
        robot_region=Box(low=[-0.95, 0.0, 0.0], high=[-0.95, 0.0, 0.0]),
        target_region=Box(low=_BOX_PLACE_XY, high=_BOX_PLACE_XY),
        container_region=Box(low=_BOTTLE_GROUND_XY, high=_BOTTLE_GROUND_XY),
        distractors_region=Box(low=[0.0, 0.0], high=[0.0, 0.0]),
        rand_stable_pose=False,
        target_stable_indices=[0],
        target_rotate_z=Box(low=0.0, high=0.0),
        container_rotate_z=Box(low=0.0, high=0.0),
        fixed_stable_pose_idx=0,
        obj_surface_map={"target": "table", "container": "default"},
    )
    dr_cfgs["scene"] = TabletopSceneDRCfg(
        scene_mode="fixed",
        table_position=Box(low=_BOX_TABLE_XY, high=_BOX_TABLE_XY),
        table_height=Box(low=_LOW_TABLE_HEIGHT, high=_LOW_TABLE_HEIGHT),
        room_choices=["hssd:scene0"],
        scene_manager="hssd",
    )

    def __init__(self, *args, target: str | None = None, hold_steps: int = 10, **kwargs) -> None:
        self.dr_cfgs = deepcopy(type(self).dr_cfgs)
        super().__init__(*args, target=target, **kwargs)
        self.hold_steps = int(hold_steps)
        self._success_hold_count = 0
        self._bottle_in_box_phase = False
        self._box_initial_z: float | None = None
        self._bottle_collision_points: np.ndarray | None = None

    @property
    def layout(self) -> Layout:
        assert self._layout is not None, "call reset() first"
        return self._layout

    @property
    def bottle(self) -> Actor:
        actor = self.layout.actors.get("container")
        assert actor is not None, "call reset() first"
        return actor

    @property
    def action_space(self) -> spaces.Space:
        assert isinstance(self.robot, Controllable)
        return self.robot.controller.action_space

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> None:
        super().reset(seed, options)
        self._success_hold_count = 0
        self._bottle_in_box_phase = False
        self.reward = 0.0
        for actor, xy in ((self.target, _BOX_PLACE_XY), (self.bottle, _BOTTLE_GROUND_XY)):
            if isinstance(actor, ObjectActor):
                stable_pose = actor.asset.stable_poses[0]
                z = (
                    _table_top(self) + float(stable_pose[2])
                    if actor is self.target
                    else float(stable_pose[2])
                )
                actor.pose.position = [float(xy[0]), float(xy[1]), z]
                actor.pose.quaternion = [float(v) for v in stable_pose[3:]]
        self._box_initial_z = _table_top(self) + _stable_z(self.target)

    def _box_on_table(self, info: dict[str, Any], mujoco_env=None) -> bool:
        if "target" not in info:
            return False
        box_position = np.asarray(info["target"][:3], dtype=np.float64)
        table_top = _table_top(self)
        min_height = table_top + _stable_z(self.target) - 0.04
        xy_close = float(np.linalg.norm(box_position[:2] - np.asarray(_BOX_PLACE_XY))) <= 0.30
        return bool(
            xy_close
            and float(box_position[2]) >= min_height
            and _contact_between(mujoco_env, _actor_body_name(self.target), "table")
            and not _hand_contact(mujoco_env, self.target)
        )

    def _box_lifted(self, info: dict[str, Any], mujoco_env=None) -> bool:
        if self._box_initial_z is None or "target" not in info:
            return False
        box_z = float(info["target"][2])
        return bool(
            box_z - self._box_initial_z >= _L10_BOX_LIFT_THRESHOLD_M
            and not _contact_between(mujoco_env, _actor_body_name(self.target), "table")
        )

    def _bottle_inside_box(self, mujoco_env=None) -> bool:
        if mujoco_env is None:
            return False
        if self._bottle_collision_points is None:
            self._bottle_collision_points = _load_collision_points(self.bottle)
        bottle_local = _object_points_in_body_local(
            mujoco_env,
            self.bottle,
            _actor_body_name(self.target),
            self._bottle_collision_points,
        )
        bottle_center = _actor_center_in_body_local(
            mujoco_env,
            self.bottle,
            _actor_body_name(self.target),
        )
        if bottle_local is None or bottle_center is None:
            return False
        inside = np.all(
            (bottle_local >= _STORAGE_BOX_INTERIOR_LOW)
            & (bottle_local <= _STORAGE_BOX_INTERIOR_HIGH),
            axis=1,
        )
        return bool(
            np.all(bottle_center >= _STORAGE_BOX_INTERIOR_LOW)
            and np.all(bottle_center <= _STORAGE_BOX_INTERIOR_HIGH)
            and float(np.mean(inside)) >= _STORAGE_BOX_MIN_INSIDE_FRACTION
        )

    def _bottle_in_box(self, info: dict[str, Any], mujoco_env=None) -> bool:
        del info
        return bool(
            self._bottle_inside_box(mujoco_env=mujoco_env)
            and _contact_between(
                mujoco_env,
                _actor_body_name(self.bottle),
                _actor_body_name(self.target),
            )
            and not _hand_contact(mujoco_env, self.bottle)
        )

    def check_box_and_bottle_success(
        self, info: dict[str, Any] | None = None, *args, **kwargs
    ) -> bool:
        del args
        if info is None:
            return False
        mujoco_env = kwargs.get("mujoco_env", None)
        if self._box_on_table(info, mujoco_env=mujoco_env) and self._bottle_in_box(
            info, mujoco_env=mujoco_env
        ):
            self._bottle_in_box_phase = True
        return bool(
            self._bottle_in_box_phase
            and self._bottle_inside_box(mujoco_env=mujoco_env)
            and not _hand_contact(mujoco_env, self.bottle)
            and self._box_lifted(info, mujoco_env=mujoco_env)
        )

    def check_object_in_container(self, *args, **kwargs) -> bool:
        return self.check_box_and_bottle_success(*args, **kwargs)

    def compute_reward(self, info: dict[str, Any], *args, **kwargs) -> float:
        if self.check_box_and_bottle_success(info, *args, **kwargs):
            self._success_hold_count += 1
        else:
            self._success_hold_count = 0
        self.reward = _SUCCESS_REWARD_INCREMENT * self._success_hold_count
        return float(self.reward)

    def check_success(self, info: dict[str, Any], *args, **kwargs) -> bool:
        del info, args, kwargs
        return self._success_hold_count >= self.hold_steps

    def preload_objects(self) -> list[Actor]:
        objects = AssetManager.get("objects")
        return [
            ObjectActor(asset=objects.load(_STORAGE_BOX_LOAD_ID)),
            ObjectActor(asset=objects.load(_BOTTLE_LOAD_ID)),
        ]


@TaskRegistry.register("g1_wholebody_long_apple_banana_to_basket_teleop")
class G1WholebodyLongAppleBananaToBasketTaskTeleop(G1WholebodyXMovePickTaskTeleop):
    uid: str = "g1_wholebody_long_apple_banana_to_basket_teleop"
    label: str = "G1 TELEOP Long Apple Banana To Basket"
    description: str = (
        "A task where the G1 robot must put both the apple and the banana into the basket."
    )

    metadata: dict[str, Any] = deepcopy(G1WholebodyXMovePickTaskTeleop.metadata)
    dr_cfgs = deepcopy(G1WholebodyXMovePickTaskTeleop.dr_cfgs)
    dr_cfgs["language"] = LanguageDRCfg(
        instructions=["put the apple and the banana into the basket"]
    )
    dr_cfgs["target"] = TargetDRCfg(asset_id=_APPLE_ASSET_ID)
    dr_cfgs["container"] = TargetDRCfg(asset_id=_BASKET_ASSET_ID)
    dr_cfgs["distractors"] = DistractorDRCfg(
        res_id="objects",
        number_of_distractors=1,
        allow_duplicates=False,
        include=[_BANANA_LOAD_ID],
    )
    dr_cfgs["spatial"] = SpatialDRCfg(
        spatial_mode="fixed",
        robot_region=Box(low=[-0.95, 0.0, 0.0], high=[-0.95, 0.0, 0.0]),
        target_region=Box(low=[-0.275, -0.20], high=[-0.275, -0.20]),
        container_region=Box(low=[-0.04, 0.16], high=[-0.04, 0.16]),
        distractors_region=Box(low=[-0.17, -0.24], high=[-0.17, -0.24]),
        target_stable_indices=[0],
        target_rotate_z=Box(low=0.0, high=0.0),
        container_rotate_z=Box(low=0.0, high=0.0),
        fixed_stable_pose_idx=0,
        obj_surface_map={"target": "table", "container": "table", "distractor": "table"},
    )
    dr_cfgs["scene"] = TabletopSceneDRCfg(
        scene_mode="fixed",
        table_position=Box(low=[0.3, 0.0], high=[0.3, 0.0]),
        table_height=Box(low=_LOW_TABLE_HEIGHT, high=_LOW_TABLE_HEIGHT),
        room_choices=["hssd:scene0"],
        scene_manager="hssd",
    )

    def __init__(self, *args, target: str | None = None, hold_steps: int = 10, **kwargs) -> None:
        self.dr_cfgs = deepcopy(type(self).dr_cfgs)
        super().__init__(*args, target=target, **kwargs)
        self.hold_steps = int(hold_steps)
        self._success_hold_count = 0

    @property
    def layout(self) -> Layout:
        assert self._layout is not None, "call reset() first"
        return self._layout

    @property
    def basket(self) -> Actor:
        actor = self.layout.actors.get("container")
        assert actor is not None, "call reset() first"
        return actor

    @property
    def banana(self) -> Actor:
        actor = self.layout.actors.get("distractor_0")
        assert actor is not None, "call reset() first"
        return actor

    @property
    def action_space(self) -> spaces.Space:
        assert isinstance(self.robot, Controllable)
        return self.robot.controller.action_space

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> None:
        super().reset(seed, options)
        self._success_hold_count = 0
        self.reward = 0.0
        basket = self.layout.actors.get("container")
        if isinstance(basket, ObjectActor):
            basket.mujoco_mass = _BASKET_MUJOCO_MASS_KG

    def _object_in_basket(
        self, info: dict[str, Any], key: str, actor: Actor, mujoco_env=None
    ) -> bool:
        local = _actor_center_in_actor_local(info, key, actor, "container", self.basket)
        return bool(
            np.all(np.abs(local[:2]) <= _BASKET_INSIDE_HALF_EXTENTS)
            and _BASKET_INSIDE_Z_RANGE[0] <= float(local[2]) <= _BASKET_INSIDE_Z_RANGE[1]
            and _contact_between(mujoco_env, _actor_body_name(actor), _actor_body_name(self.basket))
            and not _hand_contact(mujoco_env, actor)
        )

    def check_objects_in_basket(self, info: dict[str, Any] | None = None, *args, **kwargs) -> bool:
        del args
        if (
            info is None
            or "target" not in info
            or "container" not in info
            or "distractor_0" not in info
        ):
            return False
        mujoco_env = kwargs.get("mujoco_env", None)
        return bool(
            self._object_in_basket(info, "target", self.target, mujoco_env=mujoco_env)
            and self._object_in_basket(info, "distractor_0", self.banana, mujoco_env=mujoco_env)
        )

    def check_object_in_container(self, *args, **kwargs) -> bool:
        return self.check_objects_in_basket(*args, **kwargs)

    def compute_reward(self, info: dict[str, Any], *args, **kwargs) -> float:
        if self.check_objects_in_basket(info, *args, **kwargs):
            self._success_hold_count += 1
        else:
            self._success_hold_count = 0
        self.reward = _SUCCESS_REWARD_INCREMENT * self._success_hold_count
        return float(self.reward)

    def check_success(self, info: dict[str, Any], *args, **kwargs) -> bool:
        del info, args, kwargs
        return self._success_hold_count >= self.hold_steps

    def preload_objects(self) -> list[Actor]:
        objects = AssetManager.get("objects")
        return [
            ObjectActor(asset=objects.load(_APPLE_LOAD_ID)),
            ObjectActor(asset=objects.load(_BASKET_LOAD_ID)),
            ObjectActor(asset=objects.load(_BANANA_LOAD_ID)),
        ]


@TaskRegistry.register("g1_wholebody_long_basket_apple_to_table_banana_to_trash_can_teleop")
class G1WholebodyLongBasketAppleToTableBananaToTrashCanTaskTeleop(G1WholebodyXMovePickTaskTeleop):
    uid = "g1_wholebody_long_basket_apple_to_table_banana_to_trash_can_teleop"
    label = "G1 TELEOP Long Basket Apple To Table Banana To Trash Can"
    description = "A task where the G1 robot must take the apple from the basket onto the high table, then put the banana in the trash can."

    metadata: dict[str, Any] = deepcopy(G1WholebodyXMovePickTaskTeleop.metadata)
    dr_cfgs = deepcopy(G1WholebodyXMovePickTaskTeleop.dr_cfgs)
    dr_cfgs["language"] = LanguageDRCfg(
        instructions=[
            "take the apple from the basket onto the high table, then put the banana in the trash can"
        ]
    )
    dr_cfgs["target"] = TargetDRCfg(asset_id=_APPLE_ASSET_ID)
    dr_cfgs["container"] = TargetDRCfg(asset_id=_BASKET_ASSET_ID)
    dr_cfgs["distractors"] = DistractorDRCfg(
        res_id="objects",
        number_of_distractors=2,
        allow_duplicates=False,
        include=[_BANANA_LOAD_ID, _TRASH_CAN_LOAD_ID],
    )
    dr_cfgs["spatial"] = SpatialDRCfg(
        spatial_mode="fixed",
        robot_region=Box(low=[-0.95, 0.0, 0.0], high=[-0.95, 0.0, 0.0]),
        target_region=Box(low=[-0.30, -0.24], high=[-0.30, -0.24]),
        container_region=Box(low=[-0.04, 0.16], high=[-0.04, 0.16]),
        distractors_region=[
            Box(low=[-0.17, -0.24], high=[-0.17, -0.24]),
            Box(low=_TRASH_CAN_XY, high=_TRASH_CAN_XY),
        ],
        target_stable_indices=[0],
        target_rotate_z=Box(low=0.0, high=0.0),
        container_rotate_z=Box(low=0.0, high=0.0),
        fixed_stable_pose_idx=0,
        obj_surface_map={
            "target": "table",
            "container": "table",
            "distractor": ["table", "ground"],
        },
    )
    dr_cfgs["scene"] = TabletopSceneDRCfg(
        scene_mode="fixed",
        table_position=Box(low=[0.3, 0.0], high=[0.3, 0.0]),
        table_height=Box(low=_HIGH_TABLE_HEIGHT, high=_HIGH_TABLE_HEIGHT),
        room_choices=["hssd:scene0"],
        scene_manager="hssd",
    )

    def __init__(self, *args, target: str | None = None, hold_steps: int = 10, **kwargs) -> None:
        self.dr_cfgs = deepcopy(type(self).dr_cfgs)
        self.hold_steps = int(hold_steps)
        self._success_hold_count = 0
        super().__init__(*args, target=target, **kwargs)

    @property
    def layout(self) -> Layout:
        assert self._layout is not None, "call reset() first"
        return self._layout

    @property
    def basket(self) -> Actor:
        actor = self.layout.actors.get("container")
        assert actor is not None, "call reset() first"
        return actor

    @property
    def banana(self) -> Actor:
        actor = self.layout.actors.get("distractor_0")
        assert actor is not None, "call reset() first"
        return actor

    @property
    def trash_can(self) -> Actor:
        actor = self.layout.actors.get("distractor_1")
        assert actor is not None, "call reset() first"
        return actor

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> None:
        super().reset(seed, options)
        self._success_hold_count = 0
        self.reward = 0.0
        if isinstance(self.basket, ObjectActor):
            self.basket.mujoco_mass = _BASKET_MUJOCO_MASS_KG
        if isinstance(self.trash_can, ObjectActor):
            self.trash_can.pose.position = [*_TRASH_CAN_XY, _stable_z(self.trash_can)]
            self.trash_can.pose.quaternion = [1.0, 0.0, 0.0, 0.0]
            self.trash_can.mujoco_freejoint = False
        basket_position = np.asarray(self.basket.pose.position, dtype=np.float64)
        basket_rotation = t3d.quaternions.quat2mat(
            np.asarray(self.basket.pose.quaternion, dtype=np.float64)
        )
        apple_stable_pose = self.target.asset.stable_poses[0]
        self.target.pose.position = (
            basket_position + basket_rotation @ np.array([-0.04, 0.0, 0.08])
        ).tolist()
        self.target.pose.quaternion = [float(value) for value in apple_stable_pose[3:]]
        banana_stable_pose = self.banana.asset.stable_poses[0]
        self.banana.pose.position = (
            basket_position + basket_rotation @ np.array([0.04, 0.0, 0.08])
        ).tolist()
        self.banana.pose.quaternion = [float(value) for value in banana_stable_pose[3:]]

    def _apple_in_basket(self, info: dict[str, Any], mujoco_env=None) -> bool:
        if "target" not in info or "container" not in info:
            return False
        local = _actor_center_in_actor_local(info, "target", self.target, "container", self.basket)
        return bool(
            np.all(np.abs(local[:2]) <= _BASKET_INSIDE_HALF_EXTENTS)
            and _BASKET_INSIDE_Z_RANGE[0] <= float(local[2]) <= _BASKET_INSIDE_Z_RANGE[1]
            and _contact_between(
                mujoco_env, _actor_body_name(self.target), _actor_body_name(self.basket)
            )
            and not _hand_contact(mujoco_env, self.target)
        )

    def _apple_on_table(self, info: dict[str, Any], mujoco_env=None) -> bool:
        del info
        return _contact_between(mujoco_env, _actor_body_name(self.target), "table")

    def _banana_in_trash_can(self, info: dict[str, Any], mujoco_env=None) -> bool:
        del mujoco_env
        if "distractor_0" not in info or "distractor_1" not in info:
            return False
        # Center-in-container is intentionally the only second-stage test.
        # The banana mesh is irregular and its collision vertices can extend
        # beyond the authored interior bounds even when it is visibly in the
        # trash can.
        local = _actor_center_in_actor_local(
            info,
            "distractor_0",
            self.banana,
            "distractor_1",
            self.trash_can,
        )
        return bool(
            np.all(np.abs(local[:2]) <= _TRASH_CAN_INSIDE_HALF_EXTENTS + 0.04)
            and _TRASH_CAN_INSIDE_Z_RANGE[0] - 0.04
            <= float(local[2])
            <= _TRASH_CAN_INSIDE_Z_RANGE[1] + 0.04
        )

    def check_object_in_container(
        self, info: dict[str, Any] | None = None, *args, **kwargs
    ) -> bool:
        del args
        if info is None or "target" not in info or "container" not in info:
            return False
        mujoco_env = kwargs.get("mujoco_env", None)
        return bool(
            self._apple_on_table(info, mujoco_env=mujoco_env)
            and not _contact_between(
                mujoco_env,
                _actor_body_name(self.target),
                _actor_body_name(self.basket),
            )
            and not _contact_between(
                mujoco_env,
                _actor_body_name(self.banana),
                _actor_body_name(self.basket),
            )
            and self._banana_in_trash_can(info, mujoco_env=mujoco_env)
        )

    def compute_reward(self, info: dict[str, Any], *args, **kwargs) -> float:
        if self.check_object_in_container(info, *args, **kwargs):
            self._success_hold_count += 1
        else:
            self._success_hold_count = 0
        self.reward = _SUCCESS_REWARD_INCREMENT * self._success_hold_count
        return float(self.reward)

    def check_success(self, info: dict[str, Any], *args, **kwargs) -> bool:
        del info, args, kwargs
        return self._success_hold_count >= self.hold_steps

    def preload_objects(self) -> list[Actor]:
        objects = AssetManager.get("objects")
        return [
            ObjectActor(asset=objects.load(_APPLE_LOAD_ID)),
            ObjectActor(asset=objects.load(_BASKET_LOAD_ID)),
            ObjectActor(asset=objects.load(_BANANA_LOAD_ID)),
            ObjectActor(asset=objects.load(_TRASH_CAN_LOAD_ID)),
        ]


@TaskRegistry.register("g1_wholebody_long_bread_cake_to_tray_teleop")
class G1WholebodyLongBreadCakeToTrayTaskTeleop(G1WholebodyXMovePickTaskTeleop):
    uid: str = "g1_wholebody_long_bread_cake_to_tray_teleop"
    label: str = "G1 TELEOP Long Bread Cake To Tray"
    description: str = (
        "A task where the G1 robot must put both the bread and the cake onto the tray."
    )

    metadata: dict[str, Any] = deepcopy(G1WholebodyXMovePickTaskTeleop.metadata)
    dr_cfgs = deepcopy(G1WholebodyXMovePickTaskTeleop.dr_cfgs)
    dr_cfgs["language"] = LanguageDRCfg(instructions=["put the bread and the cake onto the tray"])
    dr_cfgs["target"] = TargetDRCfg(asset_id=_BREAD_ASSET_ID)
    dr_cfgs["container"] = TargetDRCfg(asset_id=_TRAY_ASSET_ID)
    dr_cfgs["distractors"] = DistractorDRCfg(
        res_id="objects",
        number_of_distractors=1,
        allow_duplicates=False,
        include=[_CAKE_LOAD_ID],
    )
    dr_cfgs["spatial"] = SpatialDRCfg(
        spatial_mode="fixed",
        robot_region=Box(low=[-0.95, 0.0, 0.0], high=[-0.95, 0.0, 0.0]),
        target_region=Box(low=[-0.275, -0.24], high=[-0.275, -0.24]),
        container_region=Box(low=_TRAY_XY, high=_TRAY_XY),
        distractors_region=Box(low=[0.0, -0.24], high=[0.0, -0.24]),
        target_stable_indices=[0],
        target_rotate_z=Box(low=0.0, high=0.0),
        container_rotate_z=Box(low=0.0, high=0.0),
        fixed_stable_pose_idx=0,
        obj_surface_map={"target": "table", "container": "table", "distractor": "table"},
    )
    dr_cfgs["scene"] = TabletopSceneDRCfg(
        scene_mode="fixed",
        table_position=Box(low=[0.3, 0.0], high=[0.3, 0.0]),
        table_height=Box(low=_HIGH_TABLE_HEIGHT, high=_HIGH_TABLE_HEIGHT),
        room_choices=["hssd:scene0"],
        scene_manager="hssd",
    )

    def __init__(self, *args, target: str | None = None, hold_steps: int = 10, **kwargs) -> None:
        self.dr_cfgs = deepcopy(type(self).dr_cfgs)
        super().__init__(*args, target=target, **kwargs)
        self.hold_steps = int(hold_steps)
        self._success_hold_count = 0

    @property
    def layout(self) -> Layout:
        assert self._layout is not None, "call reset() first"
        return self._layout

    @property
    def tray(self) -> Actor:
        actor = self.layout.actors.get("container")
        assert actor is not None, "call reset() first"
        return actor

    @property
    def cake(self) -> Actor:
        actor = self.layout.actors.get("distractor_0")
        assert actor is not None, "call reset() first"
        return actor

    @property
    def action_space(self) -> spaces.Space:
        assert isinstance(self.robot, Controllable)
        return self.robot.controller.action_space

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> None:
        super().reset(seed, options)
        self._success_hold_count = 0
        self.reward = 0.0
        tray = self.layout.actors.get("container")
        if isinstance(tray, ObjectActor):
            tray.mujoco_mass = 0.25

    def _object_on_tray(
        self, info: dict[str, Any], key: str, actor: Actor, mujoco_env=None
    ) -> bool:
        local = _actor_center_in_actor_local(info, key, actor, "container", self.tray)
        return bool(
            np.all(np.abs(local[:2]) <= _TRAY_INSIDE_HALF_EXTENTS)
            and _TRAY_INSIDE_Z_RANGE[0] <= float(local[2]) <= _TRAY_INSIDE_Z_RANGE[1]
            and _contact_between(mujoco_env, _actor_body_name(actor), _actor_body_name(self.tray))
            and not _hand_contact(mujoco_env, actor)
        )

    def check_objects_on_tray(self, info: dict[str, Any] | None = None, *args, **kwargs) -> bool:
        del args
        if (
            info is None
            or "target" not in info
            or "container" not in info
            or "distractor_0" not in info
        ):
            return False
        mujoco_env = kwargs.get("mujoco_env", None)
        return bool(
            self._object_on_tray(info, "target", self.target, mujoco_env=mujoco_env)
            and self._object_on_tray(info, "distractor_0", self.cake, mujoco_env=mujoco_env)
        )

    def check_object_in_container(self, *args, **kwargs) -> bool:
        return self.check_objects_on_tray(*args, **kwargs)

    def compute_reward(self, info: dict[str, Any], *args, **kwargs) -> float:
        if self.check_objects_on_tray(info, *args, **kwargs):
            self._success_hold_count += 1
        else:
            self._success_hold_count = 0
        self.reward = _SUCCESS_REWARD_INCREMENT * self._success_hold_count
        return float(self.reward)

    def check_success(self, info: dict[str, Any], *args, **kwargs) -> bool:
        del info, args, kwargs
        return self._success_hold_count >= self.hold_steps

    def preload_objects(self) -> list[Actor]:
        objects = AssetManager.get("objects")
        return [
            ObjectActor(asset=objects.load(_BREAD_LOAD_ID)),
            ObjectActor(asset=objects.load(_TRAY_LOAD_ID)),
            ObjectActor(asset=objects.load(_CAKE_LOAD_ID)),
        ]


@TaskRegistry.register("g1_wholebody_long_peanuts_from_cup_to_plate_teleop")
class G1WholebodyLongPeanutsFromCupToPlateTaskTeleop(G1WholebodyXMovePickTaskTeleop):
    uid: str = "g1_wholebody_long_peanuts_from_cup_to_plate_teleop"
    label: str = "G1 TELEOP Long Peanuts From Cup To Plate"
    description: str = (
        "A task where the G1 robot must tip a cup so that two peanuts fall onto the plate."
    )

    metadata: dict[str, Any] = deepcopy(G1WholebodyXMovePickTaskTeleop.metadata)
    dr_cfgs = deepcopy(G1WholebodyXMovePickTaskTeleop.dr_cfgs)
    dr_cfgs["language"] = LanguageDRCfg(
        instructions=["pour two peanuts from the cup onto the plate"]
    )
    dr_cfgs["target"] = TargetDRCfg(asset_id=_PEANUT_016_ASSET_ID)
    dr_cfgs["container"] = TargetDRCfg(asset_id=_CUP_ASSET_ID)
    dr_cfgs["distractors"] = DistractorDRCfg(
        res_id="objects",
        number_of_distractors=2,
        allow_duplicates=False,
        include=["plate/plate_3", _PEANUT_018_LOAD_ID],
    )
    dr_cfgs["spatial"] = SpatialDRCfg(
        spatial_mode="fixed",
        robot_region=Box(low=[-0.95, 0.0, 0.0], high=[-0.95, 0.0, 0.0]),
        target_region=Box(low=[-0.20, 0.16], high=[-0.20, 0.16]),
        container_region=Box(low=_CUP_XY, high=_CUP_XY),
        distractors_region=[
            Box(low=_PLATE_XY, high=_PLATE_XY),
            Box(low=[-0.14, 0.16], high=[-0.14, 0.16]),
        ],
        target_stable_indices=[0],
        target_rotate_z=Box(low=0.0, high=0.0),
        container_rotate_z=Box(low=0.0, high=0.0),
        fixed_stable_pose_idx=0,
        obj_surface_map={"target": "table", "container": "table", "distractor": ["table", "table"]},
    )
    dr_cfgs["scene"] = TabletopSceneDRCfg(
        scene_mode="fixed",
        table_position=Box(low=[0.3, 0.0], high=[0.3, 0.0]),
        table_height=Box(low=_HIGH_TABLE_HEIGHT, high=_HIGH_TABLE_HEIGHT),
        room_choices=["hssd:scene0"],
        scene_manager="hssd",
    )

    def __init__(self, *args, target: str | None = None, hold_steps: int = 10, **kwargs) -> None:
        self.dr_cfgs = deepcopy(type(self).dr_cfgs)
        super().__init__(*args, target=target, **kwargs)
        self.hold_steps = int(hold_steps)
        self._success_hold_count = 0

    @property
    def layout(self) -> Layout:
        assert self._layout is not None, "call reset() first"
        return self._layout

    @property
    def cup(self) -> Actor:
        actor = self.layout.actors.get("container")
        assert actor is not None, "call reset() first"
        return actor

    @property
    def plate(self) -> Actor:
        actor = self.layout.actors.get("distractor_0")
        assert actor is not None, "call reset() first"
        return actor

    @property
    def peanut_b(self) -> Actor:
        actor = self.layout.actors.get("distractor_1")
        assert actor is not None, "call reset() first"
        return actor

    @property
    def action_space(self) -> spaces.Space:
        assert isinstance(self.robot, Controllable)
        return self.robot.controller.action_space

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> None:
        super().reset(seed, options)
        self._success_hold_count = 0
        self.reward = 0.0
        cup_pos = np.asarray(self.cup.pose.position, dtype=np.float64)
        for actor, offset in (
            (self.target, [-0.015, 0.0, 0.08]),
            (self.peanut_b, [0.015, 0.0, 0.08]),
        ):
            if isinstance(actor, ObjectActor):
                actor.pose.position = (cup_pos + np.asarray(offset, dtype=np.float64)).tolist()
                actor.pose.quaternion = [float(v) for v in actor.asset.stable_poses[0][3:]]

    def _cup_tipped(self, info: dict[str, Any]) -> bool:
        cup_quat = _info_quaternion(info, "container", self.cup)
        cup_rot = t3d.quaternions.quat2mat(cup_quat)
        cup_up = cup_rot @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
        return float(abs(np.dot(cup_up, np.array([0.0, 0.0, 1.0])))) <= float(
            np.cos(_CUP_TIP_TILT_RAD)
        )

    def _peanut_on_plate(
        self, info: dict[str, Any], key: str, actor: Actor, mujoco_env=None
    ) -> bool:
        local = _actor_center_in_actor_local(info, key, actor, "distractor_0", self.plate)
        return bool(
            _center_in_plate_region(local)
            and _contact_between(mujoco_env, _actor_body_name(actor), _actor_body_name(self.plate))
            and not _contact_between(
                mujoco_env, _actor_body_name(actor), _actor_body_name(self.cup)
            )
            and not _hand_contact(mujoco_env, actor)
        )

    def check_peanuts_on_plate(self, info: dict[str, Any] | None = None, *args, **kwargs) -> bool:
        del args
        if (
            info is None
            or "target" not in info
            or "container" not in info
            or "distractor_0" not in info
            or "distractor_1" not in info
        ):
            return False
        mujoco_env = kwargs.get("mujoco_env", None)
        return bool(
            self._cup_tipped(info)
            and self._peanut_on_plate(info, "target", self.target, mujoco_env=mujoco_env)
            and self._peanut_on_plate(info, "distractor_1", self.peanut_b, mujoco_env=mujoco_env)
        )

    def check_object_in_container(self, *args, **kwargs) -> bool:
        return self.check_peanuts_on_plate(*args, **kwargs)

    def compute_reward(self, info: dict[str, Any], *args, **kwargs) -> float:
        if self.check_peanuts_on_plate(info, *args, **kwargs):
            self._success_hold_count += 1
        else:
            self._success_hold_count = 0
        self.reward = _SUCCESS_REWARD_INCREMENT * self._success_hold_count
        return float(self.reward)

    def check_success(self, info: dict[str, Any], *args, **kwargs) -> bool:
        del info, args, kwargs
        return self._success_hold_count >= self.hold_steps

    def preload_objects(self) -> list[Actor]:
        objects = AssetManager.get("objects")
        return [
            ObjectActor(asset=objects.load(_PEANUT_016_LOAD_ID)),
            ObjectActor(asset=objects.load(_CUP_LOAD_ID)),
            ObjectActor(asset=objects.load(_PLATE_LOAD_ID)),
            ObjectActor(asset=objects.load(_PEANUT_018_LOAD_ID)),
        ]


@TaskRegistry.register("g1_wholebody_long_storage_box_cake_to_table_bread_to_tray_teleop")
class G1WholebodyLongStorageBoxCakeToTableBreadToTrayTaskTeleop(G1WholebodyXMovePickTaskTeleop):
    uid: str = "g1_wholebody_long_storage_box_cake_to_table_bread_to_tray_teleop"
    label: str = "G1 TELEOP Long Storage Box Cake To Table Bread To Tray"
    description: str = "A task where the G1 robot must take the cake from the storage box onto the table, then put the bread on the tray."

    metadata: dict[str, Any] = deepcopy(G1WholebodyXMovePickTaskTeleop.metadata)
    dr_cfgs = deepcopy(G1WholebodyXMovePickTaskTeleop.dr_cfgs)
    dr_cfgs["language"] = LanguageDRCfg(
        instructions=[
            "take the cake from the storage box onto the table, then put the bread on the tray"
        ]
    )
    dr_cfgs["target"] = TargetDRCfg(asset_id=_CAKE_ASSET_ID)
    dr_cfgs["container"] = TargetDRCfg(asset_id=_STORAGE_BOX_ASSET_ID)
    dr_cfgs["distractors"] = DistractorDRCfg(
        res_id="objects",
        number_of_distractors=2,
        allow_duplicates=False,
        include=[_BREAD_LOAD_ID, _TRAY_LOAD_ID],
    )
    dr_cfgs["spatial"] = SpatialDRCfg(
        spatial_mode="fixed",
        robot_region=Box(low=[-0.95, 0.0, 0.0], high=[-0.95, 0.0, 0.0]),
        target_region=Box(low=[-0.18, 0.16], high=[-0.18, 0.16]),
        container_region=Box(low=_L14_STORAGE_BOX_XY, high=_L14_STORAGE_BOX_XY),
        distractors_region=[
            Box(low=[0.08, 0.16], high=[0.08, 0.16]),
            Box(low=_L14_TRAY_XY, high=_L14_TRAY_XY),
        ],
        target_stable_indices=[0],
        target_rotate_z=Box(low=0.0, high=0.0),
        container_rotate_z=Box(low=0.0, high=0.0),
        distractor_rotate_z=Box(low=0.0, high=0.0),
        fixed_stable_pose_idx=0,
        obj_surface_map={
            "target": "table",
            "container": "table",
            "distractor": ["table", "table"],
        },
    )
    dr_cfgs["scene"] = TabletopSceneDRCfg(
        scene_mode="fixed",
        table_position=Box(low=[0.3, 0.0], high=[0.3, 0.0]),
        table_height=Box(low=_HIGH_TABLE_HEIGHT, high=_HIGH_TABLE_HEIGHT),
        room_choices=["hssd:scene0"],
        scene_manager="hssd",
    )

    def __init__(self, *args, target: str | None = None, hold_steps: int = 10, **kwargs) -> None:
        self.dr_cfgs = deepcopy(type(self).dr_cfgs)
        super().__init__(*args, target=target, **kwargs)
        self.hold_steps = int(hold_steps)
        self._success_hold_count = 0

    @property
    def layout(self) -> Layout:
        assert self._layout is not None, "call reset() first"
        return self._layout

    @property
    def storage_box(self) -> Actor:
        actor = self.layout.actors.get("container")
        assert actor is not None, "call reset() first"
        return actor

    @property
    def bread(self) -> Actor:
        actor = self.layout.actors.get("distractor_0")
        assert actor is not None, "call reset() first"
        return actor

    @property
    def tray(self) -> Actor:
        actor = self.layout.actors.get("distractor_1")
        assert actor is not None, "call reset() first"
        return actor

    @property
    def action_space(self) -> spaces.Space:
        assert isinstance(self.robot, Controllable)
        return self.robot.controller.action_space

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> None:
        super().reset(seed, options)
        self._success_hold_count = 0
        self.reward = 0.0
        if isinstance(self.tray, ObjectActor):
            self.tray.mujoco_freejoint = False

        box_position = np.asarray(self.storage_box.pose.position, dtype=np.float64)
        box_rotation = t3d.quaternions.quat2mat(
            np.asarray(self.storage_box.pose.quaternion, dtype=np.float64)
        )
        for actor, local_position in (
            (self.target, _L14_CAKE_IN_BOX_LOCAL),
            (self.bread, _L14_BREAD_IN_BOX_LOCAL),
        ):
            actor.pose.position = (box_position + box_rotation @ local_position).tolist()
            actor.pose.quaternion = [float(value) for value in actor.asset.stable_poses[0][3:]]

    def _cake_on_table(self, mujoco_env=None) -> bool:
        return _contact_between(mujoco_env, _actor_body_name(self.target), "table")

    def _bread_on_tray(self, info: dict[str, Any], mujoco_env=None) -> bool:
        del mujoco_env
        local = _actor_center_in_actor_local(
            info,
            "distractor_0",
            self.bread,
            "distractor_1",
            self.tray,
        )
        return bool(
            np.all(np.abs(local[:2]) <= _TRAY_INSIDE_HALF_EXTENTS + 0.02)
            and _TRAY_INSIDE_Z_RANGE[0] - 0.02 <= float(local[2]) <= _TRAY_INSIDE_Z_RANGE[1] + 0.02
        )

    def check_object_in_container(
        self,
        info: dict[str, Any] | None = None,
        *args,
        **kwargs,
    ) -> bool:
        del args
        if info is None or any(
            key not in info for key in ("target", "container", "distractor_0", "distractor_1")
        ):
            return False
        mujoco_env = kwargs.get("mujoco_env", None)
        return bool(
            self._cake_on_table(mujoco_env=mujoco_env)
            and not _contact_between(
                mujoco_env,
                _actor_body_name(self.target),
                _actor_body_name(self.storage_box),
            )
            and not _contact_between(
                mujoco_env,
                _actor_body_name(self.bread),
                _actor_body_name(self.storage_box),
            )
            and self._bread_on_tray(info, mujoco_env=mujoco_env)
        )

    def compute_reward(self, info: dict[str, Any], *args, **kwargs) -> float:
        if self.check_object_in_container(info, *args, **kwargs):
            self._success_hold_count += 1
        else:
            self._success_hold_count = 0
        self.reward = _SUCCESS_REWARD_INCREMENT * self._success_hold_count
        return float(self.reward)

    def check_success(self, info: dict[str, Any], *args, **kwargs) -> bool:
        del info, args, kwargs
        return self._success_hold_count >= self.hold_steps

    def preload_objects(self) -> list[Actor]:
        objects = AssetManager.get("objects")
        return [
            ObjectActor(asset=objects.load(_CAKE_LOAD_ID)),
            ObjectActor(asset=objects.load(_STORAGE_BOX_LOAD_ID)),
            ObjectActor(asset=objects.load(_BREAD_LOAD_ID)),
            ObjectActor(asset=objects.load(_TRAY_LOAD_ID)),
        ]


@TaskRegistry.register("g1_wholebody_long_apple_to_basket_then_basket_to_ground_teleop")
class G1WholebodyLongAppleToBasketThenBasketToGroundTaskTeleop(
    G1WholebodyLongAppleBananaToBasketTaskTeleop
):
    uid: str = "g1_wholebody_long_apple_to_basket_then_basket_to_ground_teleop"
    label: str = "G1 TELEOP Long Apple To Basket Then Basket To Ground"
    description: str = "A task where the G1 robot must put the apple into the basket, then lift the basket and put it on the ground."

    dr_cfgs = deepcopy(G1WholebodyLongAppleBananaToBasketTaskTeleop.dr_cfgs)
    dr_cfgs["language"] = LanguageDRCfg(
        instructions=["put the apple into the basket, then put the basket on the ground"]
    )
    dr_cfgs["distractors"] = DistractorDRCfg(
        res_id="objects", number_of_distractors=0, allow_duplicates=False
    )

    def __init__(self, *args, target: str | None = None, hold_steps: int = 10, **kwargs) -> None:
        self.dr_cfgs = deepcopy(type(self).dr_cfgs)
        G1WholebodyXMovePickTaskTeleop.__init__(self, *args, target=target, **kwargs)
        self.hold_steps = int(hold_steps)
        self._success_hold_count = 0
        self._apple_in_basket_phase = False

    @property
    def apple(self) -> Actor:
        return self.target

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> None:
        super().reset(seed, options)
        self._apple_in_basket_phase = False

    def _apple_in_basket(self, info: dict[str, Any], mujoco_env=None) -> bool:
        local = _actor_center_in_actor_local(info, "target", self.apple, "container", self.basket)
        return bool(
            np.all(np.abs(local[:2]) <= _BASKET_INSIDE_HALF_EXTENTS)
            and _BASKET_INSIDE_Z_RANGE[0] <= float(local[2]) <= _BASKET_INSIDE_Z_RANGE[1]
            and _contact_between(
                mujoco_env, _actor_body_name(self.apple), _actor_body_name(self.basket)
            )
            and not _hand_contact(mujoco_env, self.apple)
        )

    def _basket_on_ground(self, info: dict[str, Any], mujoco_env=None) -> bool:
        if "container" not in info:
            return False
        basket_z = float(info["container"][2])
        return bool(
            basket_z <= 0.15
            and (
                _contact_between(mujoco_env, _actor_body_name(self.basket), "floor")
                or _contact_between(mujoco_env, _actor_body_name(self.basket), "world")
            )
            and not _hand_contact(mujoco_env, self.basket)
        )

    def check_basket_ground_success(
        self, info: dict[str, Any] | None = None, *args, **kwargs
    ) -> bool:
        del args
        if info is None or "target" not in info or "container" not in info:
            return False
        mujoco_env = kwargs.get("mujoco_env", None)
        if self._apple_in_basket(info, mujoco_env=mujoco_env):
            self._apple_in_basket_phase = True
        return bool(
            self._apple_in_basket_phase
            and self._apple_in_basket(info, mujoco_env=mujoco_env)
            and self._basket_on_ground(info, mujoco_env=mujoco_env)
        )

    def check_object_in_container(self, *args, **kwargs) -> bool:
        return self.check_basket_ground_success(*args, **kwargs)

    def compute_reward(self, info: dict[str, Any], *args, **kwargs) -> float:
        if self.check_basket_ground_success(info, *args, **kwargs):
            self._success_hold_count += 1
        else:
            self._success_hold_count = 0
        self.reward = _SUCCESS_REWARD_INCREMENT * self._success_hold_count
        return float(self.reward)

    def preload_objects(self) -> list[Actor]:
        objects = AssetManager.get("objects")
        return [
            ObjectActor(asset=objects.load(_APPLE_LOAD_ID)),
            ObjectActor(asset=objects.load(_BASKET_LOAD_ID)),
        ]


@TaskRegistry.register("g1_wholebody_long_high_table_apple_to_low_table_drawer_teleop")
class G1WholebodyLongHighTableAppleToLowTableDrawerTaskTeleop(_LongTableObjectToDrawerTask):
    uid: str = "g1_wholebody_long_high_table_apple_to_low_table_drawer_teleop"
    label: str = "G1 TELEOP Long High Table Apple To Low Table Drawer"
    description: str = "A task where the G1 robot must move the apple from the high table into a drawer on the low table."

    dr_cfgs = deepcopy(_LongTableObjectToDrawerTask.dr_cfgs)
    dr_cfgs["spatial"] = SpatialDRCfg(
        spatial_mode="fixed",
        robot_region=Box(low=[-0.95, 0.0, 0.0], high=[-0.95, 0.0, 0.0]),
        target_region=Box(low=[-0.275, -0.20], high=[-0.275, -0.20]),
        distractors_region=Box(low=[0.0, 0.0], high=[0.0, 0.0]),
        articulated_region=Box(low=[-0.08, 0.62, 0.351], high=[-0.08, 0.62, 0.351]),
        target_stable_indices=[0],
        target_rotate_z=Box(low=0.0, high=0.0),
        articulated_rotate_z=Box(low=-0.5 * np.pi, high=-0.5 * np.pi),
        fixed_stable_pose_idx=0,
        obj_surface_map={"target": "table"},
    )
    dr_cfgs["scene"] = TabletopSceneDRCfg(
        scene_mode="fixed",
        table_position=Box(low=[0.3, -0.20], high=[0.3, -0.20]),
        table_height=Box(low=_HIGH_TABLE_HEIGHT, high=_HIGH_TABLE_HEIGHT),
        table2_size=Box(low=[1.2, 0.7, 0.1], high=[1.2, 0.7, 0.1]),
        table2_position=Box(low=[0.3, 0.62], high=[0.3, 0.62]),
        table2_height=Box(low=_LOW_TABLE_HEIGHT, high=_LOW_TABLE_HEIGHT),
        enable_table2=True,
        room_choices=["hssd:scene0"],
        scene_manager="hssd",
    )
    target_asset_id = _APPLE_ASSET_ID
    target_load_id = _APPLE_LOAD_ID
    instruction = "put the apple from the high table into the drawer on the low table"


@TaskRegistry.register("g1_wholebody_long_low_table_plate_to_high_table_drawer_teleop")
class G1WholebodyLongLowTablePlateToHighTableDrawerTaskTeleop(_LongTableObjectToDrawerTask):
    uid: str = "g1_wholebody_long_low_table_plate_to_high_table_drawer_teleop"
    label: str = "G1 TELEOP Long Low Table Plate To High Table Drawer"
    description: str = "A task where the G1 robot must move the plate from the low table into a drawer on the high table."

    dr_cfgs = deepcopy(_LongTableObjectToDrawerTask.dr_cfgs)
    dr_cfgs["spatial"] = SpatialDRCfg(
        spatial_mode="fixed",
        robot_region=Box(low=[-0.95, 0.0, 0.0], high=[-0.95, 0.0, 0.0]),
        target_region=Box(low=[-0.275, 0.62], high=[-0.275, 0.62]),
        distractors_region=Box(low=[0.0, 0.0], high=[0.0, 0.0]),
        articulated_region=Box(low=[-0.08, -0.20, 0.651], high=[-0.08, -0.20, 0.651]),
        target_stable_indices=[0],
        target_rotate_z=Box(low=0.0, high=0.0),
        articulated_rotate_z=Box(low=-0.5 * np.pi, high=-0.5 * np.pi),
        fixed_stable_pose_idx=0,
        obj_surface_map={"target": "table2"},
    )
    dr_cfgs["scene"] = TabletopSceneDRCfg(
        scene_mode="fixed",
        table_position=Box(low=[0.3, -0.20], high=[0.3, -0.20]),
        table_height=Box(low=_HIGH_TABLE_HEIGHT, high=_HIGH_TABLE_HEIGHT),
        table2_size=Box(low=[1.2, 0.7, 0.1], high=[1.2, 0.7, 0.1]),
        table2_position=Box(low=[0.3, 0.62], high=[0.3, 0.62]),
        table2_height=Box(low=_LOW_TABLE_HEIGHT, high=_LOW_TABLE_HEIGHT),
        enable_table2=True,
        room_choices=["hssd:scene0"],
        scene_manager="hssd",
    )
    target_asset_id = _PLATE_ASSET_ID
    target_load_id = _PLATE_LOAD_ID
    instruction = "put the plate from the low table into the drawer on the high table"


@TaskRegistry.register("g1_wholebody_long_move_box_then_chair_to_table_teleop")
class G1WholebodyLongMoveBoxThenChairToTableTaskTeleop(G1WholebodyChairPushTaskTeleop):
    uid: str = "g1_wholebody_long_move_box_then_chair_to_table_teleop"
    label: str = "G1 TELEOP Long Move Box Then Chair To Table"
    description: str = "A task where the G1 robot must move the box away, then push the chair to the front of the table."

    metadata: dict[str, Any] = deepcopy(G1WholebodyChairPushTaskTeleop.metadata)
    dr_cfgs = deepcopy(G1WholebodyChairPushTaskTeleop.dr_cfgs)
    dr_cfgs["language"] = LanguageDRCfg(
        instructions=["move the box away, then push the chair to the front of the table"]
    )
    dr_cfgs["distractors"] = DistractorDRCfg(
        res_id="objects",
        number_of_distractors=1,
        allow_duplicates=False,
        include=[_BOX_LOAD_ID],
    )
    dr_cfgs.pop("articulated", None)
    dr_cfgs["spatial"] = SpatialDRCfg(
        spatial_mode="fixed",
        robot_region=Box(low=_BOX_THEN_CHAIR_ROBOT_POSITION, high=_BOX_THEN_CHAIR_ROBOT_POSITION),
        target_region=Box(low=_BOX_THEN_CHAIR_INITIAL_XY, high=_BOX_THEN_CHAIR_INITIAL_XY),
        distractors_region=Box(low=_BOX_THEN_CHAIR_BOX_XY, high=_BOX_THEN_CHAIR_BOX_XY),
        target_stable_indices=[0],
        target_rotate_z=Box(low=_BOX_THEN_CHAIR_INITIAL_YAW, high=_BOX_THEN_CHAIR_INITIAL_YAW),
        distractor_rotate_z=Box(low=0.0, high=0.0),
        fixed_stable_pose_idx=0,
        obj_surface_map={"target": "ground", "distractor": "ground"},
    )
    dr_cfgs["scene"] = TabletopSceneDRCfg(
        scene_mode="fixed",
        table_position=Box(low=[0.28, 0.0], high=[0.28, 0.0]),
        table_height=Box(low=_HIGH_TABLE_HEIGHT, high=_HIGH_TABLE_HEIGHT),
        room_choices=["hssd:scene0"],
        scene_manager="hssd",
    )

    def __init__(
        self, *args, move_threshold_m: float = 0.0, hold_steps: int = 10, **kwargs
    ) -> None:
        self._box_moved_phase = False
        self._init_box_xy: np.ndarray | None = None
        super().__init__(*args, move_threshold_m=move_threshold_m, hold_steps=hold_steps, **kwargs)

    @property
    def layout(self) -> Layout:
        assert self._layout is not None, "call reset() first"
        return self._layout

    @property
    def box(self) -> Actor:
        actor = self.layout.actors.get("distractor_0")
        assert actor is not None, "call reset() first"
        return actor

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> None:
        super().reset(seed, options)
        self._box_moved_phase = False
        self._init_box_xy = np.asarray(self.box.pose.position[:2], dtype=np.float64)
        self._push_hold_count = 0
        self.reward = 0.0
        if isinstance(self._target, ObjectActor):
            stable_pose = self._target.asset.stable_poses[0]
            stable_ori_mat = t3d.quaternions.quat2mat(stable_pose[3:])
            yaw_ori_mat = t3d.euler.euler2mat(0.0, 0.0, _BOX_THEN_CHAIR_INITIAL_YAW)
            self._target.pose.position = [*_BOX_THEN_CHAIR_INITIAL_XY, float(stable_pose[2])]
            self._target.pose.quaternion = [
                float(v) for v in t3d.quaternions.mat2quat(yaw_ori_mat @ stable_ori_mat)
            ]

    def _box_moved_away(self, mujoco_env=None) -> bool:
        pose = _body_pose(mujoco_env, self.box.simulation_name)
        if pose is None:
            return False
        box_pos, _ = pose
        if self._init_box_xy is None:
            self._init_box_xy = box_pos[:2].copy()
        displacement = float(np.linalg.norm(box_pos[:2] - self._init_box_xy))
        return displacement >= _BOX_MOVE_AWAY_THRESHOLD_M

    def _chair_position(self, info: dict[str, Any] | None = None, mujoco_env=None) -> np.ndarray:
        if mujoco_env is not None and "target" in getattr(mujoco_env, "mj_objects", {}):
            chair_body = mujoco_env.mj_objects["target"]
            return np.asarray(chair_body.xpos, dtype=np.float32)
        if info is not None and "target" in info:
            return np.asarray(info["target"][:3], dtype=np.float32)
        return np.asarray(self.layout.actors["target"].pose.position, dtype=np.float32)

    def _chair_at_table(self, info: dict[str, Any], mujoco_env=None) -> bool:
        chair_pos = self._chair_position(info=info, mujoco_env=mujoco_env)
        table = self.layout.actors["table"]
        target_xy = np.asarray(
            [table.pose.position[0] - 0.35, table.pose.position[1]], dtype=np.float64
        )
        metrics = self._chair_metrics(info, mujoco_env=mujoco_env)
        return bool(
            abs(float(chair_pos[0] - target_xy[0])) <= _CHAIR_TO_TABLE_X_ERROR_M
            and abs(float(chair_pos[1] - target_xy[1])) <= _CHAIR_TO_TABLE_Y_ERROR_M
            and metrics["tilt_cos"] >= np.cos(self.max_tilt_rad)
            and metrics["lift"] <= self.max_lift_m
        )

    def compute_reward(self, info: dict[str, Any], *args, **kwargs) -> float:
        del args
        mujoco_env = kwargs.get("mujoco_env")
        self._chair_metrics(info, mujoco_env=mujoco_env)
        if self._box_moved_away(mujoco_env=mujoco_env):
            self._box_moved_phase = True
        if self._box_moved_phase and self._chair_at_table(info, mujoco_env=mujoco_env):
            self._push_hold_count += 1
        else:
            self._push_hold_count = 0
        self.reward = float(np.clip(self._push_hold_count / max(1, self.hold_steps), 0.0, 1.0))
        return self.reward

    def check_success(self, info: dict[str, Any], *args, **kwargs) -> bool:
        del info, args, kwargs
        return self._push_hold_count >= self.hold_steps

    def preload_objects(self) -> list[Actor]:
        objects = AssetManager.get("objects")
        return [
            ObjectActor(asset=objects.load("chair/chair_319_0p9")),
            ObjectActor(asset=objects.load(_BOX_LOAD_ID)),
        ]
