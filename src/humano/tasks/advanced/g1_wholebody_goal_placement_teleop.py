from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

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
from humano.tasks.advanced.g1_wholebody_ground_to_table_teleop import (
    _G1WholebodyGroundToTableTaskTeleop,
)
from humano.tasks.basic.initial_positions import (
    resolve_advanced_milk_to_trash_can_layout,
    resolve_advanced_table_pair_layout,
)
from humano.tasks.simple.g1_wholebody_xmove_pick_teleop import G1WholebodyXMovePickTaskTeleop
from humano.tasks.registry import TaskRegistry

_APPLE_ASSET_ID = "objects:apple/apple_001"
_APPLE_LOAD_ID = "apple/apple_001"
_PLATE_ASSET_ID = "objects:plate/plate_3"
_PLATE_LOAD_ID = "plate/plate_3"
_BREAD_ASSET_ID = "objects:bread/bread_006"
_BREAD_LOAD_ID = "bread/bread_006"
_MILK_ASSET_ID = "objects:Carton_Toy_3C923930_Milk/Carton_Toy_3C923930_Milk_1"
_MILK_LOAD_ID = "Carton_Toy_3C923930_Milk/Carton_Toy_3C923930_Milk_1"
_PENCIL_ASSET_ID = "objects:pencil002/pencil002"
_PENCIL_LOAD_ID = "pencil002/pencil002"
_BOOK_ASSET_ID = "objects:Book_9388144295_Homer/Book_9388144295_Homer"
_BOOK_LOAD_ID = "Book_9388144295_Homer/Book_9388144295_Homer"

_BASKET_ASSET_ID = "objects:picnic_basket/picnic_basket_009"
_BASKET_LOAD_ID = "picnic_basket/picnic_basket_009"
_TRAY_ASSET_ID = "objects:tray/tray_001"
_TRAY_LOAD_ID = "tray/tray_001"
_TRASH_CAN_ASSET_ID = "objects:dustbin/dustbin_011"
_TRASH_CAN_LOAD_ID = "dustbin/dustbin_011"
_PEN_HOLDER_ASSET_ID = "objects:pen_holder001/pen_holder001"
_PEN_HOLDER_LOAD_ID = "pen_holder001/pen_holder001"

_CABINET_ASSET_ID = "articulated:8"
_CABINET_JOINT = "articulate_joint_0"
_CABINET_DRAWER_BODY = "E_drawer_59"
_CABINET_POSITION = [-0.08, 0.18, 0.651]
_CABINET_ROTATE_Z = -0.5 * np.pi
_CABINET_DRAWER_OPEN_QPOS = -0.20
_CABINET_DRAWER_LOCAL_LOW = np.array([-0.17, 0.025, -0.13], dtype=np.float64)
_CABINET_DRAWER_LOCAL_HIGH = np.array([0.17, 0.335, 0.06], dtype=np.float64)
_CABINET_DRAWER_BOTTOM_Z_RANGE = (-0.13, -0.085)

_CART_ASSET_ID = "articulated:7"
_CART_BASE_BODY = "articulate_base"
_CART_POSITION = [-0.10, 1.45, 0.0]
_CART_LOCAL_LOW = np.array([-0.30, -0.10, 0.14], dtype=np.float64)
_CART_LOCAL_HIGH = np.array([0.30, 0.68, 0.34], dtype=np.float64)

_HIGH_TABLE_HEIGHT = 0.65
_LOW_TABLE_HEIGHT = 0.35
_TRASH_CAN_XY = [-0.52, 0.18]
_SUCCESS_REWARD_INCREMENT = 0.1

_RECEPTACLE_BOUNDS = {
    "basket": (np.array([0.14, 0.11]), (-0.08, 0.18)),
    "tray": (np.array([0.15, 0.11]), (-0.01, 0.10)),
    "trash_can": (np.array([0.12, 0.12]), (-0.14, 0.05)),
}
_PLACE_HEIGHTS = {
    "basket": 0.20,
    "tray": 0.08,
    "trash_can": 0.34,
    "pen_holder": 0.18,
}
_CONTAINER_MASSES = {
    "basket": 0.35,
    "tray": 0.25,
    "pen_holder": 0.12,
}
_PEN_HOLDER_RADIUS = 0.075
_PEN_HOLDER_Z_RANGE = (0.02, 0.26)
_PENCIL_VERTICAL_TILT_RAD = np.deg2rad(38.0)


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


def _geom_body_name(model, geom_id: int) -> str:
    geom = model.geom(int(geom_id))
    return model.body(int(np.asarray(geom.bodyid).item())).name


def _contact_between(mujoco_env, label_a: str, label_b: str) -> bool:
    if mujoco_env is None:
        return False
    model = mujoco_env.mjModel
    data = mujoco_env.mjData
    for contact_id in range(data.ncon):
        contact = data.contact[contact_id]
        body1 = _geom_body_name(model, contact.geom1)
        body2 = _geom_body_name(model, contact.geom2)
        if (label_a in body1 and label_b in body2) or (label_a in body2 and label_b in body1):
            return True
    return False


def _hand_contact(mujoco_env, actor: Actor) -> bool:
    return _contact_between(mujoco_env, str(actor.asset.label), "hand")


def _actor_center_in_actor_local(
    info: dict[str, Any],
    actor_key: str,
    actor: Actor,
    container_key: str,
    container: Actor,
) -> np.ndarray:
    actor_state = info.get(actor_key)
    container_state = info.get(container_key)
    actor_position = np.asarray(
        actor_state[:3] if actor_state is not None else actor.pose.position,
        dtype=np.float64,
    )
    container_position = np.asarray(
        container_state[:3] if container_state is not None else container.pose.position,
        dtype=np.float64,
    )
    container_quaternion = np.asarray(
        container_state[3:7] if container_state is not None else container.pose.quaternion,
        dtype=np.float64,
    )
    container_rotation = t3d.quaternions.quat2mat(container_quaternion)
    return container_rotation.T @ (actor_position - container_position)


def _load_collision_points(actor: Actor) -> np.ndarray:
    assert isinstance(actor, ObjectActor)
    points: list[np.ndarray] = []
    for mesh_path in actor.asset.collision_meshes_mujoco:
        mesh = trimesh.load_mesh(Path(mesh_path), force="mesh", process=False)
        if not mesh.is_empty and len(mesh.vertices):
            points.append(np.asarray(mesh.vertices, dtype=np.float64))
    if not points:
        raise RuntimeError(f"No collision vertices found for {actor.uid!r}.")
    return np.vstack(points)


class _G1WholebodyTableObjectToReceptacleTaskTeleop(G1WholebodyXMovePickTaskTeleop):
    metadata: dict[str, Any] = deepcopy(G1WholebodyXMovePickTaskTeleop.metadata)

    target_asset_id: str
    target_load_id: str
    target_label: str
    container_asset_id: str
    container_load_id: str
    container_label: str
    receptacle_kind: str
    instruction: str
    table_height: float = _HIGH_TABLE_HEIGHT
    container_on_ground: bool = False

    def __init__(
        self,
        *args,
        target: str | None = None,
        initial_position: str | None = None,
        **kwargs,
    ) -> None:
        self.dr_cfgs = deepcopy(G1WholebodyXMovePickTaskTeleop.dr_cfgs)
        self.dr_cfgs["language"] = LanguageDRCfg(instructions=[type(self).instruction])
        self.dr_cfgs["target"] = TargetDRCfg(asset_id=type(self).target_asset_id)
        self.dr_cfgs["container"] = TargetDRCfg(asset_id=type(self).container_asset_id)
        self.dr_cfgs["distractors"] = DistractorDRCfg(
            res_id="objects",
            number_of_distractors=0,
            allow_duplicates=False,
        )

        container_xy = list(_TRASH_CAN_XY if self.container_on_ground else [-0.05, 0.18])
        self.dr_cfgs["spatial"] = SpatialDRCfg(
            spatial_mode="fixed",
            robot_region=Box(low=[-0.95, 0.0, 0.0], high=[-0.95, 0.0, 0.0]),
            target_region=Box(low=[-0.275, -0.18], high=[-0.275, -0.18]),
            container_region=Box(low=container_xy, high=container_xy),
            distractors_region=Box(low=[0.0, 0.0], high=[0.0, 0.0]),
            target_stable_indices=[0],
            target_rotate_z=Box(low=0.0, high=0.0),
            container_rotate_z=Box(low=0.0, high=0.0),
            fixed_stable_pose_idx=0,
            obj_surface_map={
                "target": "table",
                "container": "ground" if self.container_on_ground else "table",
            },
        )
        self.dr_cfgs["scene"] = TabletopSceneDRCfg(
            scene_mode="fixed",
            table_position=Box(low=[0.3, 0.0], high=[0.3, 0.0]),
            table_height=Box(low=self.table_height, high=self.table_height),
            room_choices=["hssd:scene0"],
            scene_manager="hssd",
        )

        self._container_xy = container_xy
        if self.container_on_ground:
            layout = resolve_advanced_milk_to_trash_can_layout(initial_position)
        else:
            layout = resolve_advanced_table_pair_layout(initial_position)
        if layout is not None:
            resolved_layout, target_region, container_region = layout
            self._container_xy = list(resolved_layout.container_xy)
            spatial_cfg = self.dr_cfgs["spatial"]
            assert isinstance(spatial_cfg, SpatialDRCfg)
            spatial_cfg.target_region = target_region
            spatial_cfg.container_region = container_region

        self._container: Actor | None = None
        super().__init__(*args, target=target, **kwargs)

    @property
    def layout(self) -> Layout:
        assert self._layout is not None, "call reset() first"
        return self._layout

    @property
    def container(self) -> Actor:
        assert self._container is not None, "call reset() first"
        return self._container

    @property
    def action_space(self) -> spaces.Space:
        assert isinstance(self.robot, Controllable)
        return self.robot.controller.action_space

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> None:
        super().reset(seed, options)
        self._container = self.layout.actors.get("container")
        if isinstance(self._container, ObjectActor):
            mass = _CONTAINER_MASSES.get(self.receptacle_kind)
            if mass is not None:
                self._container.mujoco_mass = mass
            if self.container_on_ground:
                stable_z = float(self._container.asset.stable_poses[0][2])
                self._container.pose.position = [*self._container_xy, stable_z]
                self._container.pose.quaternion = [1.0, 0.0, 0.0, 0.0]
                self._container.mujoco_freejoint = False
        self.reward = 0.0

    def state_dict(self) -> dict[str, Any]:
        state_dict = super().state_dict()
        state_dict["container_uid"] = self.container.uid if self._container else None
        return state_dict

    def _pencil_in_pen_holder(self, info: dict[str, Any]) -> bool:
        local_position = _actor_center_in_actor_local(
            info,
            "target",
            self.target,
            "container",
            self.container,
        )
        target_state = info.get("target")
        target_quaternion = np.asarray(
            target_state[3:7] if target_state is not None else self.target.pose.quaternion,
            dtype=np.float64,
        )
        target_rotation = t3d.quaternions.quat2mat(target_quaternion)
        pencil_axis = target_rotation @ np.array([0.0, 1.0, 0.0], dtype=np.float64)
        vertical_cos = float(abs(np.dot(pencil_axis, [0.0, 0.0, 1.0])))
        return bool(
            np.linalg.norm(local_position[:2]) <= _PEN_HOLDER_RADIUS
            and _PEN_HOLDER_Z_RANGE[0] <= float(local_position[2]) <= _PEN_HOLDER_Z_RANGE[1]
            and vertical_cos >= float(np.cos(_PENCIL_VERTICAL_TILT_RAD))
        )

    def check_object_in_container(
        self,
        info: dict[str, Any] | None = None,
        *args,
        **kwargs,
    ) -> bool:
        del args
        mujoco_env = kwargs.get("mujoco_env")
        if mujoco_env is None or info is None or "target" not in info:
            return False

        if self.receptacle_kind == "pen_holder":
            placed = self._pencil_in_pen_holder(info)
        else:
            local_position = _actor_center_in_actor_local(
                info,
                "target",
                self.target,
                "container",
                self.container,
            )
            half_xy, z_range = _RECEPTACLE_BOUNDS[self.receptacle_kind]
            placed = bool(
                np.all(np.abs(local_position[:2]) <= half_xy)
                and z_range[0] <= float(local_position[2]) <= z_range[1]
                and _contact_between(
                    mujoco_env,
                    str(self.target.asset.label),
                    str(self.container.asset.label),
                )
            )
        return bool(placed and not _hand_contact(mujoco_env, self.target))

    def compute_reward(self, info: dict[str, Any], *args, **kwargs) -> float:
        if self.check_object_in_container(info, *args, **kwargs):
            self.reward += _SUCCESS_REWARD_INCREMENT
        else:
            self.reward = 0.0
        return float(self.reward)

    def preload_objects(self) -> list[Actor]:
        objects = AssetManager.get("objects")
        return [
            ObjectActor(asset=objects.load(type(self).target_load_id)),
            ObjectActor(asset=objects.load(type(self).container_load_id)),
        ]

    def decompose(self):
        from humano.datagen.subtask_spec import (
            GraspObjectSpec,
            LiftSpec,
            MoveEEFToPoseSpec,
            OpenGripperSpec,
            PhaseBreakSpec,
            StandSpec,
            WalkSpec,
        )

        container_rotation = t3d.quaternions.quat2mat(self.container.pose.quaternion)
        place_position = np.asarray(self.container.pose.position, dtype=np.float64) + (
            container_rotation
            @ np.array([0.0, 0.0, _PLACE_HEIGHTS[self.receptacle_kind]], dtype=np.float64)
        )
        place_orientation = None
        if self.receptacle_kind == "pen_holder":
            place_orientation = t3d.quaternions.mat2quat(
                container_rotation @ t3d.euler.euler2mat(0.5 * np.pi, 0.0, 0.0)
            )
        hand_uid = self._robot.hand_spec.uid_for("right")
        return [
            StandSpec("initialize"),
            WalkSpec("walk_to_table", vx=0.35, target_yaw=0.0, target_distance=0.45, steps=300),
            StandSpec("stand_at_table"),
            PhaseBreakSpec("phase_break_before_grasp", grasp_type="bodex"),
            GraspObjectSpec(
                f"grasp_{self.target_label}",
                target_uid=self.target.uid,
                pregrasp=False,
                grasp_type="bodex",
                hand_uid=hand_uid,
                lock_links=["left_hand_palm_link"],
            ),
            LiftSpec(f"lift_{self.target_label}", up=0.10, grasp_type="bodex", hand_uid=hand_uid),
            PhaseBreakSpec("phase_break_before_place", grasp_type="bodex"),
            MoveEEFToPoseSpec(
                f"move_to_{self.container_label}",
                position=place_position,
                orientation=place_orientation,
                grasp_type="bodex",
                hand_uid=hand_uid,
                lock_links=["waist_roll_link", "left_hand_palm_link"],
            ),
            OpenGripperSpec(f"release_{self.target_label}", hand_uid=hand_uid),
            StandSpec("stop"),
        ]


@TaskRegistry.register("g1_wholebody_table_apple_to_tray_teleop")
class G1WholebodyTableAppleToTrayTaskTeleop(_G1WholebodyTableObjectToReceptacleTaskTeleop):
    uid = "g1_wholebody_table_apple_to_tray_teleop"
    label = "G1 TELEOP Table Apple To Tray"
    description = "Pick up an apple on a low table and place it on a tray."
    target_asset_id = _APPLE_ASSET_ID
    target_load_id = _APPLE_LOAD_ID
    target_label = "apple"
    container_asset_id = _TRAY_ASSET_ID
    container_load_id = _TRAY_LOAD_ID
    container_label = "tray"
    receptacle_kind = "tray"
    instruction = "pick up the apple on the low table and place it on the tray"
    table_height = _LOW_TABLE_HEIGHT


@TaskRegistry.register("g1_wholebody_table_plate_to_basket_teleop")
class G1WholebodyTablePlateToBasketTaskTeleop(_G1WholebodyTableObjectToReceptacleTaskTeleop):
    uid = "g1_wholebody_table_plate_to_basket_teleop"
    label = "G1 TELEOP Table Plate To Basket"
    description = "Pick up a plate on a high table and place it in a basket."
    target_asset_id = _PLATE_ASSET_ID
    target_load_id = _PLATE_LOAD_ID
    target_label = "plate"
    container_asset_id = _BASKET_ASSET_ID
    container_load_id = _BASKET_LOAD_ID
    container_label = "basket"
    receptacle_kind = "basket"
    instruction = "pick up the plate on the high table and place it in the basket"


@TaskRegistry.register("g1_wholebody_table_bread_to_basket_teleop")
class G1WholebodyTableBreadToBasketTaskTeleop(_G1WholebodyTableObjectToReceptacleTaskTeleop):
    uid = "g1_wholebody_table_bread_to_basket_teleop"
    label = "G1 TELEOP Table Bread To Basket"
    description = "Pick up bread on a high table and place it in a basket."
    target_asset_id = _BREAD_ASSET_ID
    target_load_id = _BREAD_LOAD_ID
    target_label = "bread"
    container_asset_id = _BASKET_ASSET_ID
    container_load_id = _BASKET_LOAD_ID
    container_label = "basket"
    receptacle_kind = "basket"
    instruction = "pick up the bread on the high table and place it in the basket"


@TaskRegistry.register("g1_wholebody_table_bread_to_trash_can_teleop")
class G1WholebodyTableBreadToTrashCanTaskTeleop(_G1WholebodyTableObjectToReceptacleTaskTeleop):
    uid = "g1_wholebody_table_bread_to_trash_can_teleop"
    label = "G1 TELEOP Table Bread To Trash Can"
    description = "Pick up bread on a high table and place it in a trash can on the ground."
    target_asset_id = _BREAD_ASSET_ID
    target_load_id = _BREAD_LOAD_ID
    target_label = "bread"
    container_asset_id = _TRASH_CAN_ASSET_ID
    container_load_id = _TRASH_CAN_LOAD_ID
    container_label = "trash_can"
    receptacle_kind = "trash_can"
    instruction = "pick up the bread on the high table and place it in the trash can"
    container_on_ground = True


@TaskRegistry.register("g1_wholebody_table_milk_to_basket_teleop")
class G1WholebodyTableMilkToBasketTaskTeleop(_G1WholebodyTableObjectToReceptacleTaskTeleop):
    uid = "g1_wholebody_table_milk_to_basket_teleop"
    label = "G1 TELEOP Table Milk To Basket"
    description = "Pick up a milk carton on a high table and place it in a basket."
    target_asset_id = _MILK_ASSET_ID
    target_load_id = _MILK_LOAD_ID
    target_label = "milk"
    container_asset_id = _BASKET_ASSET_ID
    container_load_id = _BASKET_LOAD_ID
    container_label = "basket"
    receptacle_kind = "basket"
    instruction = "pick up the milk carton on the high table and place it in the basket"


@TaskRegistry.register("g1_wholebody_table_pencil_to_pen_holder_teleop")
class G1WholebodyTablePencilToPenHolderTaskTeleop(_G1WholebodyTableObjectToReceptacleTaskTeleop):
    uid = "g1_wholebody_table_pencil_to_pen_holder_teleop"
    label = "G1 TELEOP Table Pencil To Pen Holder"
    description = "Pick up a pencil on a high table and place it upright in a pen holder."
    target_asset_id = _PENCIL_ASSET_ID
    target_load_id = _PENCIL_LOAD_ID
    target_label = "pencil"
    container_asset_id = _PEN_HOLDER_ASSET_ID
    container_load_id = _PEN_HOLDER_LOAD_ID
    container_label = "pen_holder"
    receptacle_kind = "pen_holder"
    instruction = "pick up the pencil on the high table and place it upright in the pen holder"


@TaskRegistry.register("g1_wholebody_table_pencil_to_tray_teleop")
class G1WholebodyTablePencilToTrayTaskTeleop(_G1WholebodyTableObjectToReceptacleTaskTeleop):
    uid = "g1_wholebody_table_pencil_to_tray_teleop"
    label = "G1 TELEOP Table Pencil To Tray"
    description = "Pick up a pencil on a high table and place it on a tray."
    target_asset_id = _PENCIL_ASSET_ID
    target_load_id = _PENCIL_LOAD_ID
    target_label = "pencil"
    container_asset_id = _TRAY_ASSET_ID
    container_load_id = _TRAY_LOAD_ID
    container_label = "tray"
    receptacle_kind = "tray"
    instruction = "pick up the pencil on the high table and place it on the tray"


@TaskRegistry.register("g1_wholebody_table_plate_to_cabinet_teleop")
class G1WholebodyTablePlateToCabinetTaskTeleop(G1WholebodyXMovePickTaskTeleop):
    uid = "g1_wholebody_table_plate_to_cabinet_teleop"
    label = "G1 TELEOP Table Plate To Cabinet"
    description = "Pick up a plate on a high table and place it inside an open cabinet drawer."
    metadata: dict[str, Any] = deepcopy(G1WholebodyXMovePickTaskTeleop.metadata)

    def __init__(self, *args, target: str | None = None, **kwargs) -> None:
        self.dr_cfgs = deepcopy(G1WholebodyXMovePickTaskTeleop.dr_cfgs)
        self.dr_cfgs["language"] = LanguageDRCfg(
            instructions=[
                "pick up the plate on the high table and place it inside the open cabinet"
            ]
        )
        self.dr_cfgs["target"] = TargetDRCfg(asset_id=_PLATE_ASSET_ID)
        self.dr_cfgs["articulated"] = ArticulatedObjectDrCfg(
            asset_id=_CABINET_ASSET_ID,
            articulate_init_joint_qpos={_CABINET_JOINT: _CABINET_DRAWER_OPEN_QPOS},
        )
        self.dr_cfgs["distractors"] = DistractorDRCfg(
            res_id="objects", number_of_distractors=0, allow_duplicates=False
        )
        self.dr_cfgs["spatial"] = SpatialDRCfg(
            spatial_mode="fixed",
            robot_region=Box(low=[-0.95, 0.0, 0.0], high=[-0.95, 0.0, 0.0]),
            target_region=Box(low=[-0.275, -0.18], high=[-0.275, -0.18]),
            distractors_region=Box(low=[0.0, 0.0], high=[0.0, 0.0]),
            articulated_region=Box(low=_CABINET_POSITION, high=_CABINET_POSITION),
            target_stable_indices=[0],
            target_rotate_z=Box(low=0.0, high=0.0),
            articulated_rotate_z=Box(low=_CABINET_ROTATE_Z, high=_CABINET_ROTATE_Z),
            fixed_stable_pose_idx=0,
            obj_surface_map={"target": "table"},
        )
        self.dr_cfgs["scene"] = TabletopSceneDRCfg(
            scene_mode="fixed",
            table_position=Box(low=[0.3, 0.0], high=[0.3, 0.0]),
            table_height=Box(low=_HIGH_TABLE_HEIGHT, high=_HIGH_TABLE_HEIGHT),
            room_choices=["hssd:scene0"],
            scene_manager="hssd",
        )
        self._plate_collision_points: np.ndarray | None = None
        super().__init__(*args, target=target, **kwargs)

    @property
    def cabinet(self) -> Actor:
        cabinet = self.layout.actors.get("articulated")
        assert cabinet is not None, "call reset() first"
        return cabinet

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> None:
        super().reset(seed, options)
        self.reward = 0.0

    def _plate_points_in_drawer(self, mujoco_env) -> np.ndarray | None:
        plate_pose = _body_pose(mujoco_env, str(self.target.asset.label))
        drawer_pose = _body_pose(mujoco_env, _CABINET_DRAWER_BODY)
        if plate_pose is None or drawer_pose is None:
            return None
        if self._plate_collision_points is None:
            self._plate_collision_points = _load_collision_points(self.target)
        plate_position, plate_rotation = plate_pose
        drawer_position, drawer_rotation = drawer_pose
        world_points = self._plate_collision_points @ plate_rotation.T + plate_position
        return (world_points - drawer_position) @ drawer_rotation

    def _plate_contacts_drawer_bottom(self, mujoco_env) -> bool:
        drawer_pose = _body_pose(mujoco_env, _CABINET_DRAWER_BODY)
        if drawer_pose is None:
            return False
        drawer_position, drawer_rotation = drawer_pose
        model = mujoco_env.mjModel
        data = mujoco_env.mjData
        target_label = str(self.target.asset.label)
        for contact_id in range(data.ncon):
            contact = data.contact[contact_id]
            body1 = _geom_body_name(model, contact.geom1)
            body2 = _geom_body_name(model, contact.geom2)
            if not (
                target_label in body1
                and _CABINET_DRAWER_BODY in body2
                or target_label in body2
                and _CABINET_DRAWER_BODY in body1
            ):
                continue
            local = (np.asarray(contact.pos) - drawer_position) @ drawer_rotation
            if (
                _CABINET_DRAWER_LOCAL_LOW[0] <= local[0] <= _CABINET_DRAWER_LOCAL_HIGH[0]
                and _CABINET_DRAWER_LOCAL_LOW[1] <= local[1] <= _CABINET_DRAWER_LOCAL_HIGH[1]
                and _CABINET_DRAWER_BOTTOM_Z_RANGE[0]
                <= local[2]
                <= _CABINET_DRAWER_BOTTOM_Z_RANGE[1]
            ):
                return True
        return False

    def check_object_in_container(self, info=None, *args, **kwargs) -> bool:
        del info, args
        mujoco_env = kwargs.get("mujoco_env")
        if mujoco_env is None:
            return False
        points = self._plate_points_in_drawer(mujoco_env)
        drawer_qpos = float(mujoco_env.mjData.joint(_CABINET_JOINT).qpos[0])
        return bool(
            drawer_qpos <= -0.12
            and points is not None
            and np.all(points >= _CABINET_DRAWER_LOCAL_LOW)
            and np.all(points <= _CABINET_DRAWER_LOCAL_HIGH)
            and self._plate_contacts_drawer_bottom(mujoco_env)
            and not _hand_contact(mujoco_env, self.target)
        )

    def compute_reward(self, info: dict[str, Any], *args, **kwargs) -> float:
        if self.check_object_in_container(info, *args, **kwargs):
            self.reward += _SUCCESS_REWARD_INCREMENT
        else:
            self.reward = 0.0
        return float(self.reward)

    def preload_objects(self) -> list[Actor]:
        return [ObjectActor(asset=AssetManager.get("objects").load(_PLATE_LOAD_ID))]

    def decompose(self):
        from humano.datagen.subtask_spec import (
            GraspObjectSpec,
            LiftSpec,
            MoveEEFToPoseSpec,
            OpenGripperSpec,
            PhaseBreakSpec,
            StandSpec,
            WalkSpec,
        )

        cabinet_rotation = t3d.quaternions.quat2mat(self.cabinet.pose.quaternion)
        place_position = np.asarray(self.cabinet.pose.position) + cabinet_rotation @ np.array(
            [0.0, -0.02, 0.12]
        )
        hand_uid = self._robot.hand_spec.uid_for("right")
        return [
            StandSpec("initialize"),
            WalkSpec("walk_to_table", vx=0.35, target_yaw=0.0, target_distance=0.45, steps=300),
            PhaseBreakSpec("phase_break_before_grasp", grasp_type="bodex"),
            GraspObjectSpec(
                "grasp_plate",
                target_uid=self.target.uid,
                pregrasp=False,
                grasp_type="bodex",
                hand_uid=hand_uid,
                lock_links=["left_hand_palm_link"],
            ),
            LiftSpec("lift_plate", up=0.10, grasp_type="bodex", hand_uid=hand_uid),
            PhaseBreakSpec("phase_break_before_place", grasp_type="bodex"),
            MoveEEFToPoseSpec(
                "move_to_cabinet_drawer",
                position=place_position,
                grasp_type="bodex",
                hand_uid=hand_uid,
                lock_links=["waist_roll_link", "left_hand_palm_link"],
            ),
            OpenGripperSpec("release_plate", hand_uid=hand_uid),
            StandSpec("stop"),
        ]


@TaskRegistry.register("g1_wholebody_ground_book_to_table_teleop")
class G1WholebodyGroundBookToTableTaskTeleop(_G1WholebodyGroundToTableTaskTeleop):
    uid = "g1_wholebody_ground_book_to_table_teleop"
    label = "G1 TELEOP Ground Book To Table"
    description = "Pick up a book on the ground and place it on a high table."
    target_asset_id = _BOOK_ASSET_ID
    target_load_id = _BOOK_LOAD_ID
    target_label = "book"
    instruction = "pick up the book on the ground and place it on the table"


@TaskRegistry.register("g1_wholebody_table_book_to_cart_teleop")
class G1WholebodyTableBookToCartTaskTeleop(G1WholebodyXMovePickTaskTeleop):
    uid = "g1_wholebody_table_book_to_cart_teleop"
    label = "G1 TELEOP Table Book To Cart"
    description = "Pick up a book on a high table and place it on a dolly cart."
    metadata: dict[str, Any] = deepcopy(G1WholebodyXMovePickTaskTeleop.metadata)

    def __init__(self, *args, target: str | None = None, **kwargs) -> None:
        self.dr_cfgs = deepcopy(G1WholebodyXMovePickTaskTeleop.dr_cfgs)
        self.dr_cfgs["language"] = LanguageDRCfg(
            instructions=["pick up the book on the high table and place it on the cart"]
        )
        self.dr_cfgs["target"] = TargetDRCfg(asset_id=_BOOK_ASSET_ID)
        self.dr_cfgs["articulated"] = ArticulatedObjectDrCfg(
            asset_id=_CART_ASSET_ID,
            articulate_init_joint_qpos={},
        )
        self.dr_cfgs["distractors"] = DistractorDRCfg(
            res_id="objects", number_of_distractors=0, allow_duplicates=False
        )
        self.dr_cfgs["spatial"] = SpatialDRCfg(
            spatial_mode="fixed",
            robot_region=Box(low=[-0.95, 0.0, 0.0], high=[-0.95, 0.0, 0.0]),
            target_region=Box(low=[-0.275, -0.18], high=[-0.275, -0.18]),
            distractors_region=Box(low=[0.0, 0.0], high=[0.0, 0.0]),
            articulated_region=Box(low=_CART_POSITION, high=_CART_POSITION),
            target_stable_indices=[0],
            target_rotate_z=Box(low=0.0, high=0.0),
            articulated_rotate_z=Box(low=0.0, high=0.0),
            fixed_stable_pose_idx=0,
            obj_surface_map={"target": "table", "articulated": "default"},
        )
        self.dr_cfgs["scene"] = TabletopSceneDRCfg(
            scene_mode="fixed",
            table_position=Box(low=[0.3, 0.0], high=[0.3, 0.0]),
            table_height=Box(low=_HIGH_TABLE_HEIGHT, high=_HIGH_TABLE_HEIGHT),
            room_choices=["hssd:scene0"],
            scene_manager="hssd",
        )
        super().__init__(*args, target=target, **kwargs)

    @property
    def cart(self) -> Actor:
        cart = self.layout.actors.get("articulated")
        assert cart is not None, "call reset() first"
        return cart

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> None:
        super().reset(seed, options)
        self.reward = 0.0

    def check_object_in_container(self, info=None, *args, **kwargs) -> bool:
        del info, args
        mujoco_env = kwargs.get("mujoco_env")
        book_pose = _body_pose(mujoco_env, str(self.target.asset.label))
        cart_pose = _body_pose(mujoco_env, _CART_BASE_BODY)
        if book_pose is None or cart_pose is None:
            return False
        book_position, _ = book_pose
        cart_position, cart_rotation = cart_pose
        local = (book_position - cart_position) @ cart_rotation
        return bool(
            np.all(local >= _CART_LOCAL_LOW)
            and np.all(local <= _CART_LOCAL_HIGH)
            and _contact_between(mujoco_env, str(self.target.asset.label), _CART_BASE_BODY)
            and not _hand_contact(mujoco_env, self.target)
        )

    def compute_reward(self, info: dict[str, Any], *args, **kwargs) -> float:
        if self.check_object_in_container(info, *args, **kwargs):
            self.reward += _SUCCESS_REWARD_INCREMENT
        else:
            self.reward = 0.0
        return float(self.reward)

    def preload_objects(self) -> list[Actor]:
        return [ObjectActor(asset=AssetManager.get("objects").load(_BOOK_LOAD_ID))]

    def decompose(self):
        from humano.datagen.subtask_spec import (
            GraspObjectSpec,
            LiftSpec,
            MoveEEFToPoseSpec,
            OpenGripperSpec,
            PhaseBreakSpec,
            StandSpec,
            TurnSpec,
            WalkSpec,
        )

        cart_rotation = t3d.quaternions.quat2mat(self.cart.pose.quaternion)
        place_position = np.asarray(self.cart.pose.position) + cart_rotation @ np.array(
            [0.23, 0.30, 0.28]
        )
        hand_uid = self._robot.hand_spec.uid_for("right")
        return [
            StandSpec("initialize"),
            WalkSpec("walk_to_table", vx=0.35, target_yaw=0.0, target_distance=0.45, steps=300),
            PhaseBreakSpec("phase_break_before_grasp", grasp_type="bodex"),
            GraspObjectSpec(
                "grasp_book",
                target_uid=self.target.uid,
                pregrasp=False,
                grasp_type="bodex",
                hand_uid=hand_uid,
                lock_links=["left_hand_palm_link"],
            ),
            LiftSpec("lift_book", up=0.10, grasp_type="bodex", hand_uid=hand_uid),
            TurnSpec("turn_to_cart", vx=0.1, target_yaw=0.5 * np.pi),
            WalkSpec(
                "walk_to_cart",
                vx=0.25,
                target_yaw=0.5 * np.pi,
                target_distance=1.10,
            ),
            PhaseBreakSpec("phase_break_before_place", grasp_type="bodex"),
            MoveEEFToPoseSpec(
                "move_to_cart",
                position=place_position,
                grasp_type="bodex",
                hand_uid=hand_uid,
                lock_links=["waist_roll_link", "left_hand_palm_link"],
            ),
            OpenGripperSpec("release_book", hand_uid=hand_uid),
            StandSpec("stop"),
        ]
