from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
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


_BREAD_ASSET_ID = "objects:bread/bread_006"
_BREAD_LOAD_ID = "bread/bread_006"
_TRAY_ASSET_ID = "objects:tray/tray_001"
_TRAY_LOAD_ID = "tray/tray_001"
_TRAY_PLACE_HEIGHT = 0.08
_TRAY_INSIDE_HALF_EXTENTS = np.array([0.15, 0.11, 0.08], dtype=np.float32)


@TaskRegistry.register("g1_wholebody_table_bread_to_tray_teleop")
class G1WholebodyTableBreadToTrayTaskTeleop(G1WholebodyXMovePickTaskTeleop):
    uid: str = "g1_wholebody_table_bread_to_tray_teleop"
    label: str = "G1 TELEOP Table Bread To Tray"
    description: str = (
        "A task where the G1 robot must pick up bread on a high table and place it "
        "in a tray on the same high table."
    )

    metadata: dict[str, Any] = deepcopy(G1WholebodyXMovePickTaskTeleop.metadata)
    dr_cfgs = deepcopy(G1WholebodyXMovePickTaskTeleop.dr_cfgs)
    dr_cfgs["language"] = LanguageDRCfg(
        instructions=[
            "pick up the bread on the high table and place it in the tray",
        ]
    )
    dr_cfgs["target"] = TargetDRCfg(asset_id=_BREAD_ASSET_ID)
    dr_cfgs["container"] = TargetDRCfg(asset_id=_TRAY_ASSET_ID)
    dr_cfgs["distractors"] = DistractorDRCfg(
        res_id="objects",
        number_of_distractors=0,
        allow_duplicates=False,
    )
    dr_cfgs["spatial"] = SpatialDRCfg(
        spatial_mode="fixed",
        robot_region=Box(low=[-0.95, 0.0, 0.0], high=[-0.95, 0.0, 0.0]),
        target_region=Box(low=[-0.275, -0.24], high=[-0.275, -0.24]),
        container_region=Box(low=[-0.05, 0.16], high=[-0.05, 0.16]),
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
        table_height=Box(low=0.65, high=0.65),
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
        self._container = None

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
            self._container.mujoco_mass = 0.25

    def state_dict(self) -> dict[str, Any]:
        state_dict = super().state_dict()
        state_dict.update({"container_uid": self.container.uid if self._container else None})
        return state_dict

    def _target_contacts_container(self, mujoco_env) -> bool:
        target_name = str(self.target.asset.label)
        container_name = str(self.container.asset.label)
        mj_physics_data = mujoco_env.mjData
        mj_physics_model = mujoco_env.mjModel

        for i_contact in range(mj_physics_data.ncon):
            contact = mj_physics_data.contact[i_contact]
            g1 = mj_physics_model.geom(contact.geom1)
            g2 = mj_physics_model.geom(contact.geom2)
            body1 = mj_physics_model.body(g1.bodyid).name
            body2 = mj_physics_model.body(g2.bodyid).name
            if (target_name in body1 and container_name in body2) or (
                target_name in body2 and container_name in body1
            ):
                return True
        return False

    def check_object_in_container(self, info: dict[str, Any] | None = None, *args, **kwargs) -> bool:
        mujoco_env = kwargs.get("mujoco_env", None)
        if mujoco_env is None or info is None or "target" not in info or "container" not in info:
            return False

        target_position = np.asarray(info["target"][:3], dtype=np.float32)
        container_position = np.asarray(info["container"][:3], dtype=np.float32)
        delta = target_position - container_position
        inside_xy = np.all(np.abs(delta[:2]) <= _TRAY_INSIDE_HALF_EXTENTS[:2])
        inside_z = 0.0 <= delta[2] <= _TRAY_INSIDE_HALF_EXTENTS[2]
        return bool(inside_xy and inside_z and self._target_contacts_container(mujoco_env))

    def compute_reward(self, info: dict[str, Any], *args, **kwargs) -> float:
        if self.check_object_in_container(info=info, *args, **kwargs):
            self.reward += 0.1
        else:
            self.reward = 0.0
        return self.reward

    def preload_objects(self) -> list[Actor]:
        objects = AssetManager.get("objects")
        bread = objects.load(_BREAD_LOAD_ID)
        tray = objects.load(_TRAY_LOAD_ID)
        return [ObjectActor(asset=bread), ObjectActor(asset=tray)]

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

        place_position = np.asarray(self.container.pose.position) + np.array(
            [0.0, 0.0, _TRAY_PLACE_HEIGHT]
        )
        hand_uid = self._robot.hand_spec.uid_for("right")

        return [
            StandSpec("initialize"),
            WalkSpec("walk_to_table", vx=0.35, target_yaw=0.0, target_distance=0.45, steps=300),
            StandSpec("stand_at_table"),
            PhaseBreakSpec("phase_break_before_grasp", grasp_type="bodex"),
            GraspObjectSpec(
                "grasp_bread",
                target_uid=self.target.uid,
                pregrasp=False,
                grasp_type="bodex",
                hand_uid=hand_uid,
                lock_links=["left_hand_palm_link"],
            ),
            LiftSpec("lift_bread", up=0.08, grasp_type="bodex", hand_uid=hand_uid),
            PhaseBreakSpec("phase_break_before_place", grasp_type="bodex"),
            MoveEEFToPoseSpec(
                "move_to_tray",
                position=place_position,
                grasp_type="bodex",
                hand_uid=hand_uid,
                lock_links=["waist_roll_link", "left_hand_palm_link"],
            ),
            OpenGripperSpec("release_bread", hand_uid=hand_uid),
            StandSpec("stop"),
        ]
