from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np

from humano.assets import AssetManager
from humano.core.actor import Actor, ObjectActor
from humano.dr import DistractorDRCfg, LanguageDRCfg, SpatialDRCfg, TargetDRCfg
from humano.dr.types import Box
from humano.tasks.basic.initial_positions import (
    MOVEMENT_MIN_ROBOT_TARGET_DISTANCE_M,
    resolve_robot_around_object_spawn,
)
from humano.tasks.simple.g1_wholebody_xmove_pick_teleop import G1WholebodyXMovePickTaskTeleop
from humano.tasks.registry import TaskRegistry

_BALL_ASSET_ID = "objects:football/football"
_BALL_MUJOCO_FRICTION = [2.0, 0.08, 0.01]
_BALL_CENTER_XY = (-0.275, 0.0)
_BALL_ROBOT_SPAWN_DISTANCE = 0.675


@TaskRegistry.register("g1_wholebody_ball_kick_teleop")
class G1WholebodyBallKickTaskTeleop(G1WholebodyXMovePickTaskTeleop):
    # Long table variants can override this without changing the basic task.
    _ball_center_xy = _BALL_CENTER_XY

    uid: str = "g1_wholebody_ball_kick_teleop"
    label: str = "G1 TELEOP Ball Kick"
    description: str = "A task where the G1 robot must kick a ball on the ground."

    metadata: dict[str, Any] = deepcopy(G1WholebodyXMovePickTaskTeleop.metadata)
    dr_cfgs = deepcopy(G1WholebodyXMovePickTaskTeleop.dr_cfgs)
    dr_cfgs.pop("scene", None)
    dr_cfgs["language"] = LanguageDRCfg(
        instructions=[
            "move forward and kick the ball",
        ]
    )
    dr_cfgs["target"] = TargetDRCfg(asset_id=_BALL_ASSET_ID)
    dr_cfgs["distractors"] = DistractorDRCfg(
        res_id="objects",
        number_of_distractors=0,
        allow_duplicates=False,
    )
    dr_cfgs["spatial"] = SpatialDRCfg(
        spatial_mode="fixed",
        robot_region=Box(low=[-0.95, 0.0, 0.0], high=[-0.95, 0.0, 0.0]),
        target_region=Box(low=[-0.275, 0.0], high=[-0.275, 0.0]),
        distractors_region=Box(low=[0.0, 0.0], high=[0.0, 0.0]),
        target_stable_indices=[0],
        target_rotate_z=Box(low=0.0, high=0.0),
        fixed_stable_pose_idx=0,
        obj_surface_map={"target": "default", "distractor": "default"},
    )

    def __init__(
        self,
        *args,
        move_threshold_m: float = 0.50,
        initial_position: str | None = None,
        movement: bool = False,
        **kwargs,
    ) -> None:
        self.dr_cfgs = deepcopy(type(self).dr_cfgs)
        _, robot_region, robot_orientation_region = resolve_robot_around_object_spawn(
            initial_position=initial_position,
            object_xy=self._ball_center_xy,
            distance=max(
                _BALL_ROBOT_SPAWN_DISTANCE,
                MOVEMENT_MIN_ROBOT_TARGET_DISTANCE_M + 1e-6 if movement else 0.0,
            ),
        )
        spatial_cfg = self.dr_cfgs["spatial"]
        assert isinstance(spatial_cfg, SpatialDRCfg)
        spatial_cfg.robot_region = robot_region
        if robot_orientation_region is not None:
            spatial_cfg.robot_orientation_region = robot_orientation_region
        self.move_threshold_m = float(move_threshold_m)
        self._init_target_xy: np.ndarray | None = None
        super().__init__(*args, **kwargs)

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> None:
        super().reset(seed, options)
        self._init_target_xy = np.asarray(self.target.pose.position[:2], dtype=np.float32)
        if isinstance(self._target, ObjectActor):
            self._target.mujoco_friction = _BALL_MUJOCO_FRICTION

    def compute_reward(self, info: dict[str, Any], *args, **kwargs) -> float:
        if self._init_target_xy is None:
            self._init_target_xy = np.asarray(info["target"][:2], dtype=np.float32)
        target_xy = np.asarray(info["target"][:2], dtype=np.float32)
        displacement = float(np.linalg.norm(target_xy - self._init_target_xy))
        return float(np.clip(displacement / self.move_threshold_m, 0.0, 1.0))

    def check_success(self, info: dict[str, Any], *args, **kwargs) -> bool:
        if self._init_target_xy is None:
            self._init_target_xy = np.asarray(info["target"][:2], dtype=np.float32)
        target_xy = np.asarray(info["target"][:2], dtype=np.float32)
        displacement = float(np.linalg.norm(target_xy - self._init_target_xy))
        return displacement >= self.move_threshold_m

    def preload_objects(self) -> list[Actor]:
        asset = AssetManager.get("objects").load("football/football")
        return [ObjectActor(asset=asset)]
