from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
import transforms3d as t3d
from gymnasium import spaces

from humano.assets import AssetManager
from humano.core.actor import Actor, ObjectActor
from humano.core.layout import Layout
from humano.dr import (
    DistractorDRCfg,
    LanguageDRCfg,
    SpatialDRCfg,
    TabletopSceneDRCfg,
    TargetDRCfg,
)
from humano.dr.types import Box
from humano.robots.protocols import Controllable
from humano.tasks.basic.initial_positions import resolve_advanced_table_pair_layout
from humano.tasks.simple.g1_wholebody_xmove_pick_teleop import G1WholebodyXMovePickTaskTeleop
from humano.tasks.registry import TaskRegistry


_PLATE_ASSET_ID = "objects:plate/plate_3"
_SHELF_ASSET_ID = "objects:shelf/shelf_001"
_SHELF_LOAD_ID = "shelf/shelf_001"
_SHELF_XY = [-0.08, 0.90]
_SHELF_QUATERNION = [0.70710678, 0.0, 0.0, -0.70710678]
_SHELF_LOCAL_PLACE_OFFSET = np.array([-0.04, -0.02, 0.46], dtype=np.float64)
_SHELF_LOCAL_INSIDE_XY_LOW = np.array([-0.12, -0.14], dtype=np.float64)
_SHELF_LOCAL_INSIDE_XY_HIGH = np.array([0.20, 0.14], dtype=np.float64)
_SHELF_LOCAL_BOARD_Z_RANGES = (
    (0.02, 0.08),
    (0.40, 0.48),
    (0.81, 0.90),
    (1.22, 1.31),
)
_PLATE_PLACE_ORIENTATION = t3d.euler.euler2quat(np.pi / 2.0, 0.0, 0.0)
_PLATE_HORIZONTAL_TILT_RAD = np.deg2rad(20.0)
_HIGH_TABLE_HEIGHT = 0.65


@TaskRegistry.register("g1_wholebody_table_plate_to_shelf_teleop")
class G1WholebodyTablePlateToShelfTaskTeleop(G1WholebodyXMovePickTaskTeleop):
    uid: str = "g1_wholebody_table_plate_to_shelf_teleop"
    label: str = "G1 TELEOP Table Plate To Shelf"
    description: str = (
        "A task where the G1 robot must pick up a plate on a high table and place it "
        "in a shelf standing next to the table."
    )

    metadata: dict[str, Any] = deepcopy(G1WholebodyXMovePickTaskTeleop.metadata)
    dr_cfgs = deepcopy(G1WholebodyXMovePickTaskTeleop.dr_cfgs)
    dr_cfgs["language"] = LanguageDRCfg(
        instructions=[
            "pick up the plate on the high table and place it in the shelf",
        ]
    )
    dr_cfgs["target"] = TargetDRCfg(asset_id=_PLATE_ASSET_ID)
    dr_cfgs["container"] = TargetDRCfg(asset_id=_SHELF_ASSET_ID)
    dr_cfgs["distractors"] = DistractorDRCfg(
        res_id="objects",
        number_of_distractors=0,
        allow_duplicates=False,
    )
    dr_cfgs["spatial"] = SpatialDRCfg(
        spatial_mode="fixed",
        robot_region=Box(low=[-0.95, 0.0, 0.0], high=[-0.95, 0.0, 0.0]),
        target_region=Box(low=[-0.275, -0.12], high=[-0.275, -0.12]),
        container_region=Box(low=_SHELF_XY, high=_SHELF_XY),
        distractors_region=Box(low=[0.0, 0.0], high=[0.0, 0.0]),
        target_stable_indices=[0],
        target_rotate_z=Box(low=0.0, high=0.0),
        container_rotate_z=Box(low=0.0, high=0.0),
        fixed_stable_pose_idx=0,
        obj_surface_map={"target": "table", "container": "table"},
    )
    dr_cfgs["scene"] = TabletopSceneDRCfg(
        scene_mode="fixed",
        table_position=Box(low=[0.3, 0.0], high=[0.3, 0.0]),
        table_height=Box(low=_HIGH_TABLE_HEIGHT, high=_HIGH_TABLE_HEIGHT),
        room_choices=["hssd:scene0"],
        scene_manager="hssd",
    )

    def __init__(
        self,
        *args,
        target: str | None = None,
        initial_position: str | None = None,
        **kwargs,
    ) -> None:
        self.dr_cfgs = deepcopy(type(self).dr_cfgs)
        layout = resolve_advanced_table_pair_layout(initial_position)
        if layout is not None:
            _, target_region, _ = layout
            spatial_cfg = self.dr_cfgs["spatial"]
            assert isinstance(spatial_cfg, SpatialDRCfg)
            spatial_cfg.target_region = target_region
        super().__init__(*args, target=target, **kwargs)

    @property
    def layout(self) -> Layout:
        assert self._layout is not None, "call reset() first"
        return self._layout

    @property
    def shelf(self) -> Actor:
        shelf = self.layout.actors.get("container")
        assert shelf is not None, "call reset() first"
        return shelf

    @property
    def action_space(self) -> spaces.Space:
        assert isinstance(self.robot, Controllable)
        return self.robot.controller.action_space

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> None:
        super().reset(seed, options)
        shelf = self.shelf
        stable_z = 0.0
        if isinstance(shelf, ObjectActor):
            stable_z = float(shelf.asset.stable_poses[0][2])
            shelf.mujoco_freejoint = False
        shelf.pose.position = [_SHELF_XY[0], _SHELF_XY[1], stable_z]
        shelf.pose.quaternion = list(_SHELF_QUATERNION)

    def state_dict(self) -> dict[str, Any]:
        state_dict = super().state_dict()
        state_dict.update({"shelf_uid": self.shelf.uid if self._layout else None})
        return state_dict

    def _shelf_local_target_position(self, info: dict[str, Any]) -> np.ndarray:
        target_position = np.asarray(info["target"][:3], dtype=np.float64)
        shelf_position = np.asarray(self.shelf.pose.position, dtype=np.float64)
        shelf_rotation = t3d.quaternions.quat2mat(self.shelf.pose.quaternion)
        return shelf_rotation.T @ (target_position - shelf_position)

    def _target_contacts_shelf(self, mujoco_env) -> bool:
        target_name = str(self.target.asset.label)
        shelf_name = str(self.shelf.asset.label)
        mj_physics_data = mujoco_env.mjData
        mj_physics_model = mujoco_env.mjModel

        for i_contact in range(mj_physics_data.ncon):
            contact = mj_physics_data.contact[i_contact]
            g1 = mj_physics_model.geom(contact.geom1)
            g2 = mj_physics_model.geom(contact.geom2)
            body1 = mj_physics_model.body(g1.bodyid).name
            body2 = mj_physics_model.body(g2.bodyid).name
            if (target_name in body1 and shelf_name in body2) or (
                target_name in body2 and shelf_name in body1
            ):
                return True
        return False

    def check_hand_object_contact(self, *args, **kwargs) -> bool:
        mujoco_env = kwargs.get("mujoco_env", None)
        if mujoco_env is None:
            return False

        target_name = str(self.target.asset.label)
        mj_physics_data = mujoco_env.mjData
        mj_physics_model = mujoco_env.mjModel

        for i_contact in range(mj_physics_data.ncon):
            contact = mj_physics_data.contact[i_contact]
            g1 = mj_physics_model.geom(contact.geom1)
            g2 = mj_physics_model.geom(contact.geom2)
            body1 = mj_physics_model.body(g1.bodyid).name
            body2 = mj_physics_model.body(g2.bodyid).name
            if (target_name in body1 and "hand" in body2) or (
                target_name in body2 and "hand" in body1
            ):
                return True
        return False

    def check_plate_horizontal(self, info: dict[str, Any] | None = None) -> bool:
        if info is not None and "target" in info:
            target_quat = np.asarray(info["target"][3:7], dtype=np.float64)
        else:
            target_quat = np.asarray(self.target.pose.quaternion, dtype=np.float64)
        target_rotation = t3d.quaternions.quat2mat(target_quat)
        plate_normal_world = target_rotation @ np.array([0.0, 1.0, 0.0], dtype=np.float64)
        horizontal_cos = float(abs(np.dot(plate_normal_world, np.array([0.0, 0.0, 1.0]))))
        return horizontal_cos >= float(np.cos(_PLATE_HORIZONTAL_TILT_RAD))

    def check_object_in_shelf(self, info: dict[str, Any] | None = None, *args, **kwargs) -> bool:
        mujoco_env = kwargs.get("mujoco_env", None)
        if mujoco_env is None or info is None or "target" not in info:
            return False

        local_position = self._shelf_local_target_position(info)
        xy_inside = bool(
            np.all(local_position[:2] >= _SHELF_LOCAL_INSIDE_XY_LOW)
            and np.all(local_position[:2] <= _SHELF_LOCAL_INSIDE_XY_HIGH)
        )
        on_board = any(
            low <= float(local_position[2]) <= high
            for low, high in _SHELF_LOCAL_BOARD_Z_RANGES
        )
        return bool(
            xy_inside
            and on_board
            and self.check_plate_horizontal(info)
            and self._target_contacts_shelf(mujoco_env)
            and not self.check_hand_object_contact(*args, **kwargs)
        )

    def check_object_in_container(self, *args, **kwargs) -> bool:
        return self.check_object_in_shelf(*args, **kwargs)

    def compute_reward(self, info: dict[str, Any], *args, **kwargs) -> float:
        if self.check_object_in_shelf(info=info, *args, **kwargs):
            self.reward += 0.1
        else:
            self.reward = 0.0
        return self.reward

    def preload_objects(self) -> list[Actor]:
        objects = AssetManager.get("objects")
        plate = objects.load("plate/plate_3")
        shelf = objects.load(_SHELF_LOAD_ID)
        return [ObjectActor(asset=plate), ObjectActor(asset=shelf)]

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

        shelf_rotation = t3d.quaternions.quat2mat(self.shelf.pose.quaternion)
        place_position = np.asarray(self.shelf.pose.position) + (
            shelf_rotation @ _SHELF_LOCAL_PLACE_OFFSET
        )
        hand_uid = self._robot.hand_spec.uid_for("right")

        return [
            StandSpec("initialize"),
            WalkSpec("walk_to_table", vx=0.35, target_yaw=0.0, target_distance=0.45, steps=300),
            StandSpec("stand_at_table"),
            PhaseBreakSpec("phase_break_before_grasp", grasp_type="bodex"),
            GraspObjectSpec(
                "grasp_plate",
                target_uid=self.target.uid,
                pregrasp=False,
                grasp_type="bodex",
                hand_uid=hand_uid,
                lock_links=["left_hand_palm_link"],
            ),
            LiftSpec("lift_plate", up=0.08, grasp_type="bodex", hand_uid=hand_uid),
            PhaseBreakSpec("phase_break_before_place", grasp_type="bodex"),
            MoveEEFToPoseSpec(
                "move_to_shelf",
                position=place_position,
                orientation=_PLATE_PLACE_ORIENTATION,
                grasp_type="bodex",
                hand_uid=hand_uid,
                lock_links=["waist_roll_link", "left_hand_palm_link"],
            ),
            OpenGripperSpec("release_plate", hand_uid=hand_uid),
            StandSpec("stop"),
        ]
