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

_PEN_HOLDER_ASSET_ID = "objects:pen_holder001/pen_holder001"
_PEN_HOLDER_LOAD_ID = "pen_holder001/pen_holder001"
_ERASER_ASSET_ID = "objects:eraser/eraser_002_large"
_ERASER_LOAD_ID = "eraser/eraser_002_large"
_PEN_HOLDER_XY = [-0.24, -0.10]
_ERASER_STAGING_XY = [-0.10, -0.24]
_ERASER_LOCAL_OFFSET = np.array([0.0, 0.0, 0.070], dtype=np.float64)
_HIGH_TABLE_HEIGHT = 0.65
_SUCCESS_REWARD_INCREMENT = 0.1
_PEN_HOLDER_MIN_TIP_RAD = np.deg2rad(45.0)
_TABLE_XY_MARGIN = 0.03
_TABLE_Z_TOLERANCE = 0.04


@TaskRegistry.register("g1_wholebody_table_eraser_out_of_pen_holder_teleop")
class G1WholebodyTableEraserOutOfPenHolderTaskTeleop(G1WholebodyXMovePickTaskTeleop):
    uid: str = "g1_wholebody_table_eraser_out_of_pen_holder_teleop"
    label: str = "G1 TELEOP Table Eraser Out Of Pen Holder"
    description: str = (
        "A task where the G1 robot must tip a pen holder on a high table so that "
        "the eraser falls out onto the table."
    )

    metadata: dict[str, Any] = deepcopy(G1WholebodyXMovePickTaskTeleop.metadata)
    dr_cfgs = deepcopy(G1WholebodyXMovePickTaskTeleop.dr_cfgs)
    dr_cfgs["language"] = LanguageDRCfg(
        instructions=[
            "tip the pen holder and pour the eraser onto the table",
        ]
    )
    dr_cfgs["target"] = TargetDRCfg(asset_id=_PEN_HOLDER_ASSET_ID)
    dr_cfgs["container"] = TargetDRCfg(asset_id=_ERASER_ASSET_ID)
    dr_cfgs["distractors"] = DistractorDRCfg(
        res_id="objects",
        number_of_distractors=0,
        allow_duplicates=False,
    )
    dr_cfgs["spatial"] = SpatialDRCfg(
        spatial_mode="fixed",
        robot_region=Box(low=[-0.95, 0.0, 0.0], high=[-0.95, 0.0, 0.0]),
        target_region=Box(low=_PEN_HOLDER_XY, high=_PEN_HOLDER_XY),
        container_region=Box(low=_ERASER_STAGING_XY, high=_ERASER_STAGING_XY),
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
            _, target_region, container_region = layout
            spatial_cfg = self.dr_cfgs["spatial"]
            assert isinstance(spatial_cfg, SpatialDRCfg)
            spatial_cfg.target_region = target_region
            spatial_cfg.container_region = container_region
        super().__init__(*args, target=target, **kwargs)
        self._eraser = None

    @property
    def layout(self) -> Layout:
        assert self._layout is not None, "call reset() first"
        return self._layout

    @property
    def eraser(self) -> Actor:
        assert self._eraser is not None, "call reset() first"
        return self._eraser

    @property
    def action_space(self) -> spaces.Space:
        assert isinstance(self.robot, Controllable)
        return self.robot.controller.action_space

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> None:
        super().reset(seed, options)
        self._eraser = self.layout.actors.get("container")
        if isinstance(self._target, ObjectActor):
            self._target.mujoco_mass = 0.12
        if isinstance(self._eraser, ObjectActor):
            self._eraser.mujoco_mass = 0.04
            pen_holder_position = np.asarray(self.target.pose.position, dtype=np.float64)
            self._eraser.pose.position = (pen_holder_position + _ERASER_LOCAL_OFFSET).tolist()
            self._eraser.pose.quaternion = [1.0, 0.0, 0.0, 0.0]
        self.reward = 0.0

    def state_dict(self) -> dict[str, Any]:
        state_dict = super().state_dict()
        state_dict.update({"eraser_uid": self.eraser.uid if self._eraser else None})
        return state_dict

    def _contact_between_labels(
        self,
        mujoco_env,
        label_a: str,
        label_b: str,
    ) -> bool:
        mj_physics_data = mujoco_env.mjData
        mj_physics_model = mujoco_env.mjModel
        for i_contact in range(mj_physics_data.ncon):
            contact = mj_physics_data.contact[i_contact]
            g1 = mj_physics_model.geom(contact.geom1)
            g2 = mj_physics_model.geom(contact.geom2)
            body1 = mj_physics_model.body(g1.bodyid).name
            body2 = mj_physics_model.body(g2.bodyid).name
            if (label_a in body1 and label_b in body2) or (
                label_a in body2 and label_b in body1
            ):
                return True
        return False

    def check_eraser_on_table(self, *args, **kwargs) -> bool:
        mujoco_env = kwargs.get("mujoco_env", None)
        if mujoco_env is None:
            return False
        if not self._eraser_in_tabletop_region(kwargs.get("info")):
            return False
        eraser_name = str(self.eraser.asset.label)
        return self._contact_between_labels(mujoco_env, eraser_name, "table")

    def _eraser_in_tabletop_region(self, info: dict[str, Any] | None = None) -> bool:
        table = self.layout.actors.get("table")
        if table is None:
            return False

        eraser_state = (info or {}).get("container")
        if eraser_state is None:
            eraser_position = np.asarray(self.eraser.pose.position, dtype=np.float64)
        else:
            eraser_position = np.asarray(eraser_state[:3], dtype=np.float64)

        table_position = np.asarray(table.pose.position, dtype=np.float64)
        table_quaternion = np.asarray(table.pose.quaternion, dtype=np.float64)
        table_rotation = t3d.quaternions.quat2mat(table_quaternion)
        eraser_local = table_rotation.T @ (eraser_position - table_position)
        table_size = np.asarray(table.size, dtype=np.float64)

        half_xy = 0.5 * table_size[:2] + _TABLE_XY_MARGIN
        if np.any(np.abs(eraser_local[:2]) > half_xy):
            return False

        stable_z = 0.0
        if isinstance(self.eraser, ObjectActor):
            stable_z = float(self.eraser.asset.stable_poses[0][2])
        expected_z = 0.5 * float(table_size[2]) + stable_z
        return abs(float(eraser_local[2]) - expected_z) <= _TABLE_Z_TOLERANCE

    def check_eraser_touching_pen_holder(self, *args, **kwargs) -> bool:
        mujoco_env = kwargs.get("mujoco_env", None)
        if mujoco_env is None:
            return False
        eraser_name = str(self.eraser.asset.label)
        pen_holder_name = str(self.target.asset.label)
        return self._contact_between_labels(mujoco_env, eraser_name, pen_holder_name)

    def check_pen_holder_tipped(self, info: dict[str, Any] | None = None) -> bool:
        if info is not None and "target" in info:
            target_quat = np.asarray(info["target"][3:7], dtype=np.float64)
        else:
            target_quat = np.asarray(self.target.pose.quaternion, dtype=np.float64)
        target_rotation = t3d.quaternions.quat2mat(target_quat)
        pen_holder_axis_world = target_rotation @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
        upright_cos = float(np.dot(pen_holder_axis_world, np.array([0.0, 0.0, 1.0])))
        return upright_cos <= float(np.cos(_PEN_HOLDER_MIN_TIP_RAD))

    def check_object_in_container(self, *args, **kwargs) -> bool:
        del args
        return self.check_eraser_touching_pen_holder(**kwargs)

    def compute_reward(self, info: dict[str, Any], *args, **kwargs) -> float:
        del args
        eraser_on_table = self.check_eraser_on_table(info=info, **kwargs)
        eraser_touching_pen_holder = self.check_eraser_touching_pen_holder(**kwargs)
        pen_holder_tipped = self.check_pen_holder_tipped(info)
        if eraser_on_table and not eraser_touching_pen_holder and pen_holder_tipped:
            self.reward += _SUCCESS_REWARD_INCREMENT
        else:
            self.reward = 0.0
        return float(self.reward)

    def preload_objects(self) -> list[Actor]:
        objects = AssetManager.get("objects")
        pen_holder = objects.load(_PEN_HOLDER_LOAD_ID)
        eraser = objects.load(_ERASER_LOAD_ID)
        return [ObjectActor(asset=pen_holder), ObjectActor(asset=eraser)]

    def decompose(self):
        from humano.datagen.subtask_spec import (
            GraspObjectSpec,
            LiftSpec,
            PhaseBreakSpec,
            StandSpec,
            WalkSpec,
        )

        hand_uid = self._robot.hand_spec.uid_for("right")
        return [
            StandSpec("initialize"),
            WalkSpec("walk_to_table", vx=0.35, target_yaw=0.0, target_distance=0.45, steps=300),
            StandSpec("stand_at_table"),
            PhaseBreakSpec("phase_break_before_grasp", grasp_type="bodex"),
            GraspObjectSpec(
                "grasp_pen_holder",
                target_uid=self.target.uid,
                pregrasp=False,
                grasp_type="bodex",
                hand_uid=hand_uid,
                lock_links=["left_hand_palm_link"],
            ),
            LiftSpec("tip_pen_holder", up=0.08, grasp_type="bodex", hand_uid=hand_uid),
            StandSpec("stop"),
        ]
