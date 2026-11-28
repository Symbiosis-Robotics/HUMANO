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
from humano.tasks.basic.initial_positions import resolve_advanced_ground_to_table_layout
from humano.tasks.simple.g1_wholebody_xmove_pick_teleop import G1WholebodyXMovePickTaskTeleop
from humano.tasks.registry import TaskRegistry

_BOTTLE_ASSET_ID = "objects:bottle/bottle_048"
_BOX_ASSET_ID = "objects:box/box_007"
_BASKET_ASSET_ID = "objects:picnic_basket/picnic_basket_009"
_BASKET_LOAD_ID = "picnic_basket/picnic_basket_009"
_BASKET_MUJOCO_MASS_KG = 0.35

_HIGH_TABLE_HEIGHT = 0.65
_LOW_TABLE_HEIGHT = 0.35
_TARGET_XY = [-0.62, 0.0]
_TABLE_XY = [0.32, 0.02]
_PLACE_XY = [0.05, 0.02]
_TABLE_CONTACT_REWARD = 0.1


class _G1WholebodyGroundToTableTaskTeleop(G1WholebodyXMovePickTaskTeleop):
    metadata: dict[str, Any] = deepcopy(G1WholebodyXMovePickTaskTeleop.metadata)
    dr_cfgs = deepcopy(G1WholebodyXMovePickTaskTeleop.dr_cfgs)
    dr_cfgs["distractors"] = DistractorDRCfg(
        res_id="objects",
        number_of_distractors=0,
        allow_duplicates=False,
    )
    dr_cfgs["spatial"] = SpatialDRCfg(
        spatial_mode="fixed",
        robot_region=Box(low=[-0.95, 0.0, 0.0], high=[-0.95, 0.0, 0.0]),
        target_region=Box(low=_TARGET_XY, high=_TARGET_XY),
        distractors_region=Box(low=[0.0, 0.0], high=[0.0, 0.0]),
        target_stable_indices=[0],
        target_rotate_z=Box(low=0.0, high=0.0),
        fixed_stable_pose_idx=0,
    )
    dr_cfgs["scene"] = TabletopSceneDRCfg(
        scene_mode="fixed",
        table_position=Box(low=_TABLE_XY, high=_TABLE_XY),
        table_height=Box(low=_HIGH_TABLE_HEIGHT, high=_HIGH_TABLE_HEIGHT),
        room_choices=["hssd:scene0"],
        scene_manager="hssd",
    )

    target_asset_id: str
    target_load_id: str
    target_label: str
    default_instruction: str
    table_height: float = _HIGH_TABLE_HEIGHT
    target_mujoco_mass_kg: float | None = None

    def __init__(
        self,
        *args,
        target: str | None = None,
        initial_position: str | None = None,
        **kwargs,
    ) -> None:
        self.dr_cfgs = deepcopy(type(self).dr_cfgs)
        self.dr_cfgs["target"] = TargetDRCfg(asset_id=type(self).target_asset_id)
        self.dr_cfgs["language"] = LanguageDRCfg(instructions=[type(self).default_instruction])
        self.dr_cfgs["scene"].table_height = Box(
            low=type(self).table_height,
            high=type(self).table_height,
        )
        self._target_xy = tuple(_TARGET_XY)
        self._place_xy = tuple(_PLACE_XY)
        layout = resolve_advanced_ground_to_table_layout(initial_position)
        if layout is not None:
            resolved_layout, target_region, table_position = layout
            self._target_xy = resolved_layout.target_xy
            self._place_xy = resolved_layout.place_xy
            spatial_cfg = self.dr_cfgs["spatial"]
            assert isinstance(spatial_cfg, SpatialDRCfg)
            spatial_cfg.target_region = target_region
            self.dr_cfgs["scene"].table_position = table_position
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
        saved_scene = (options or {}).get("state_dict", {}).get("dr_state_dict", {}).get("scene")
        saved_table = saved_scene.get("table") if isinstance(saved_scene, dict) else None
        scene_dr = self.dr.get_randomizer("scene")
        scene_cfg = getattr(scene_dr, "cfg", None) or self.dr_cfgs.get("scene")
        original_height = None
        if isinstance(saved_table, dict) and scene_cfg is not None:
            original_height = scene_cfg.table_height
            saved_height = saved_table["pose"]["position"][2] + 0.5 * saved_table["size"][2]
            scene_cfg.table_height = Box(low=saved_height, high=saved_height)
        try:
            super().reset(seed, options)
        finally:
            if isinstance(saved_table, dict) and scene_cfg is not None:
                scene_cfg.table_height = original_height
        self._contact_started = False
        if isinstance(self._target, ObjectActor):
            stable_pose = self._target.asset.stable_poses[0]
            saved_spatial = (options or {}).get("state_dict", {}).get("dr_state_dict", {}).get("spatial", {})
            saved_target = saved_spatial.get(self._target.uid)
            target_xy = saved_target["position"][:2] if isinstance(saved_target, dict) else self._target_xy
            self._target.pose.position = [*target_xy, float(stable_pose[2])]
            self._target.pose.quaternion = [float(v) for v in stable_pose[3:]]
            if self.target_mujoco_mass_kg is not None:
                self._target.mujoco_mass = float(self.target_mujoco_mass_kg)
        self._init_target_height = None
        self.reward = 0.0

    def _table_top_height(self) -> float:
        table = self.layout.actors.get("table")
        if table is None:
            return float(type(self).table_height)
        return float(table.pose.position[2] + 0.5 * table.size[2])

    def check_object_on_table(self, info: dict[str, Any] | None = None, *args, **kwargs) -> bool:
        mujoco_env = kwargs.get("mujoco_env", None)
        if mujoco_env is None:
            return False

        target_state = (info or {}).get("target")
        if target_state is None:
            target_state = np.asarray(self.target.pose.position, dtype=np.float64)
        target_height = float(target_state[2])

        stable_z = 0.0
        if isinstance(self.target, ObjectActor):
            stable_z = float(self.target.asset.stable_poses[0][2])
        min_placed_height = self._table_top_height() + stable_z - 0.04
        if target_height < min_placed_height:
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
            if (target_name in body1 and "table" in body2) or (
                target_name in body2 and "table" in body1
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

    def check_success(self, info: dict[str, Any], *args, **kwargs) -> bool:
        reward = self.compute_reward(info, *args, **kwargs)
        return reward >= self.success_criteria

    def compute_reward(self, info: dict[str, Any], *args, **kwargs) -> float:
        is_object_on_table = self.check_object_on_table(info=info, *args, **kwargs)
        is_object_contacted_by_hand = self.check_hand_object_contact(*args, **kwargs)
        if is_object_on_table and not is_object_contacted_by_hand:
            self.reward += _TABLE_CONTACT_REWARD
        else:
            self.reward = 0.0
        return float(self.reward)

    def preload_objects(self) -> list[Actor]:
        asset = AssetManager.get("objects").load(type(self).target_load_id)
        return [ObjectActor(asset=asset)]

    def decompose(self):
        from humano.datagen.subtask_spec import (
            GraspObjectSpec,
            HeightAdjustSpec,
            MoveEEFToPoseSpec,
            OpenGripperSpec,
            PhaseBreakSpec,
            StandSpec,
            WalkSpec,
        )

        stable_z = 0.0
        if isinstance(self.target, ObjectActor):
            stable_z = float(self.target.asset.stable_poses[0][2])
        place_position = np.asarray(
            [self._place_xy[0], self._place_xy[1], self._table_top_height() + stable_z + 0.08],
            dtype=np.float64,
        )
        hand_uid = self._robot.hand_spec.uid_for("right")

        return [
            StandSpec("initialize"),
            HeightAdjustSpec("lower_to_ground_object", height=-0.30, keep_waist_pose=True),
            PhaseBreakSpec("phase_break_before_grasp", grasp_type="bodex"),
            GraspObjectSpec(
                f"grasp_{self.target_label}",
                target_uid=self.target.uid,
                pregrasp=False,
                grasp_type="bodex",
                hand_uid=hand_uid,
                lock_links=["left_hand_palm_link"],
            ),
            HeightAdjustSpec("lift_from_ground", height=0.0, keep_waist_pose=True),
            WalkSpec("walk_to_table", vx=0.25, target_yaw=0.0, target_distance=0.48, steps=220),
            PhaseBreakSpec("phase_break_before_place", grasp_type="bodex"),
            MoveEEFToPoseSpec(
                f"move_{self.target_label}_over_table",
                position=place_position,
                grasp_type="bodex",
                hand_uid=hand_uid,
                lock_links=["waist_roll_link", "left_hand_palm_link"],
            ),
            OpenGripperSpec(f"release_{self.target_label}", hand_uid=hand_uid),
            StandSpec("stop"),
        ]


@TaskRegistry.register("g1_wholebody_ground_bottle_to_table_teleop")
class G1WholebodyGroundBottleToTableTaskTeleop(_G1WholebodyGroundToTableTaskTeleop):
    uid: str = "g1_wholebody_ground_bottle_to_table_teleop"
    label: str = "G1 TELEOP Ground Bottle To Table"
    description: str = (
        "A task where the G1 robot must pick up a bottle on the ground and place it "
        "on a high table."
    )

    target_asset_id = _BOTTLE_ASSET_ID
    target_load_id = "bottle/bottle_048"
    target_label = "bottle"
    default_instruction = "pick up the bottle on the ground and place it on the table"


@TaskRegistry.register("g1_wholebody_ground_box_to_table_teleop")
class G1WholebodyGroundBoxToTableTaskTeleop(_G1WholebodyGroundToTableTaskTeleop):
    uid: str = "g1_wholebody_ground_box_to_table_teleop"
    label: str = "G1 TELEOP Ground Box To Table"
    description: str = (
        "A task where the G1 robot must pick up a box on the ground and place it "
        "on a low table."
    )

    target_asset_id = _BOX_ASSET_ID
    target_load_id = "box/box_007"
    target_label = "box"
    table_height = _LOW_TABLE_HEIGHT
    default_instruction = "pick up the box on the ground and place it on the low table"


@TaskRegistry.register("g1_wholebody_ground_basket_to_table_teleop")
class G1WholebodyGroundBasketToTableTaskTeleop(_G1WholebodyGroundToTableTaskTeleop):
    uid: str = "g1_wholebody_ground_basket_to_table_teleop"
    label: str = "G1 TELEOP Ground Basket To Table"
    description: str = (
        "A task where the G1 robot must pick up a basket on the ground and place it "
        "on a low table."
    )

    target_asset_id = _BASKET_ASSET_ID
    target_load_id = _BASKET_LOAD_ID
    target_label = "basket"
    table_height = _LOW_TABLE_HEIGHT
    target_mujoco_mass_kg = _BASKET_MUJOCO_MASS_KG
    default_instruction = "pick up the basket on the ground and place it on the low table"
