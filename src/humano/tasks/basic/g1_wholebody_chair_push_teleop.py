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

_CHAIR_ASSET_ID = "objects:chair/chair_319_0p9"
_CHAIR_CENTER_XY = (-0.28, 0.0)
_CHAIR_ROBOT_SPAWN_DISTANCE = 0.67
_PUSH_CHAIR_MUJOCO_FRICTION = [0.35, 0.02, 0.001]
_PUSH_CHAIR_BACK_MUJOCO_FRICTION = [1.2, 0.08, 0.005]
_PUSH_CHAIR_MUJOCO_MASS_KG = 3.0
_PUSH_CHAIR_BACK_COLLISION_INDICES = (
    6,
    7,
    8,
    12,
    13,
    14,
    16,
    17,
    18,
    19,
    20,
    21,
    23,
    25,
    26,
    27,
    28,
    30,
)


@TaskRegistry.register("g1_wholebody_chair_push_teleop")
class G1WholebodyChairPushTaskTeleop(G1WholebodyXMovePickTaskTeleop):
    uid: str = "g1_wholebody_chair_push_teleop"
    label: str = "G1 TELEOP Chair Push"
    description: str = "A task where the G1 robot must push a movable chair."

    metadata: dict[str, Any] = deepcopy(G1WholebodyXMovePickTaskTeleop.metadata)
    dr_cfgs = deepcopy(G1WholebodyXMovePickTaskTeleop.dr_cfgs)
    dr_cfgs.pop("scene", None)
    dr_cfgs["language"] = LanguageDRCfg(
        instructions=[
            "walk to the chair and push it forward",
        ]
    )
    dr_cfgs["target"] = TargetDRCfg(asset_id=_CHAIR_ASSET_ID)
    dr_cfgs["distractors"] = DistractorDRCfg(
        res_id="objects",
        number_of_distractors=0,
        allow_duplicates=False,
    )
    dr_cfgs["spatial"] = SpatialDRCfg(
        spatial_mode="fixed",
        robot_region=Box(low=[-0.95, 0.0, 0.0], high=[-0.95, 0.0, 0.0]),
        target_region=Box(low=[-0.28, 0.0], high=[-0.28, 0.0]),
        distractors_region=Box(low=[0.0, 0.0], high=[0.0, 0.0]),
        target_stable_indices=[0],
        target_rotate_z=Box(low=np.pi / 2, high=np.pi / 2),
        fixed_stable_pose_idx=0,
        obj_surface_map={"target": "default", "distractor": "default"},
    )

    def __init__(
        self,
        *args,
        move_threshold_m: float = 0.40,
        max_tilt_rad: float = np.deg2rad(45.0),
        max_lift_m: float = 0.35,
        hold_steps: int = 10,
        initial_position: str | None = None,
        movement: bool = False,
        **kwargs,
    ) -> None:
        self.dr_cfgs = deepcopy(self.dr_cfgs)
        spatial_cfg = self.dr_cfgs["spatial"]
        assert isinstance(spatial_cfg, SpatialDRCfg)
        if initial_position is not None or movement:
            _, robot_region, robot_orientation_region = resolve_robot_around_object_spawn(
                initial_position=initial_position,
                object_xy=_CHAIR_CENTER_XY,
                distance=max(
                    _CHAIR_ROBOT_SPAWN_DISTANCE,
                    MOVEMENT_MIN_ROBOT_TARGET_DISTANCE_M + 1e-6 if movement else 0.0,
                ),
            )
            spatial_cfg.robot_region = robot_region
            if robot_orientation_region is not None:
                spatial_cfg.robot_orientation_region = robot_orientation_region
        self.move_threshold_m = float(move_threshold_m)
        self.max_tilt_rad = float(max_tilt_rad)
        self.max_lift_m = float(max_lift_m)
        self.hold_steps = int(hold_steps)
        self._init_target_xy: np.ndarray | None = None
        self._init_target_z: float | None = None
        self._init_target_up_axis: np.ndarray | None = None
        self._push_hold_count = 0
        target = kwargs.get("target")
        if isinstance(target, str) and target.startswith("articulated:"):
            kwargs["target"] = _CHAIR_ASSET_ID
        super().__init__(*args, **kwargs)

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> None:
        super().reset(seed, options)
        self._init_target_xy = None
        self._init_target_z = None
        self._init_target_up_axis = None
        self._push_hold_count = 0
        if isinstance(self._target, ObjectActor):
            self._target.mujoco_freejoint = True
            self._target.mujoco_friction = _PUSH_CHAIR_MUJOCO_FRICTION
            self._target.mujoco_geom_frictions = {
                index: _PUSH_CHAIR_BACK_MUJOCO_FRICTION
                for index in _PUSH_CHAIR_BACK_COLLISION_INDICES
            }
            self._target.mujoco_mass = _PUSH_CHAIR_MUJOCO_MASS_KG

    def _chair_metrics(self, info: dict[str, Any], mujoco_env=None) -> dict[str, float]:
        if mujoco_env is not None and "target" in getattr(mujoco_env, "mj_objects", {}):
            chair_body = mujoco_env.mj_objects["target"]
            chair_pos = np.asarray(chair_body.xpos, dtype=np.float32)
            chair_up_axis = np.asarray(chair_body.xmat, dtype=np.float32).reshape(3, 3)[:, 2]
        else:
            chair_pos = np.asarray(info["target"][:3], dtype=np.float32)
            chair_up_axis = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)

        if self._init_target_xy is None:
            self._init_target_xy = chair_pos[:2].copy()
        if self._init_target_z is None:
            self._init_target_z = float(chair_pos[2])
        if self._init_target_up_axis is None:
            norm = max(float(np.linalg.norm(chair_up_axis)), 1e-6)
            self._init_target_up_axis = chair_up_axis / norm

        up_norm = max(float(np.linalg.norm(chair_up_axis)), 1e-6)
        normalized_up_axis = chair_up_axis / up_norm
        tilt_cos = float(np.clip(np.dot(normalized_up_axis, self._init_target_up_axis), -1.0, 1.0))
        displacement = float(np.linalg.norm(chair_pos[:2] - self._init_target_xy))
        lift = float(chair_pos[2] - self._init_target_z)
        return {
            "displacement": displacement,
            "tilt_cos": tilt_cos,
            "lift": lift,
        }

    def _is_pushed_now(self, metrics: dict[str, float]) -> bool:
        return (
            metrics["displacement"] >= self.move_threshold_m
            and metrics["tilt_cos"] >= np.cos(self.max_tilt_rad)
            and metrics["lift"] <= self.max_lift_m
        )

    def compute_reward(self, info: dict[str, Any], *args, **kwargs) -> float:
        metrics = self._chair_metrics(info, mujoco_env=kwargs.get("mujoco_env"))
        if self._is_pushed_now(metrics):
            self._push_hold_count += 1
        else:
            self._push_hold_count = 0

        move_score = np.clip(metrics["displacement"] / self.move_threshold_m, 0.0, 1.0)
        tilt_score = np.clip(
            (metrics["tilt_cos"] - np.cos(self.max_tilt_rad)) / (1.0 - np.cos(self.max_tilt_rad)),
            0.0,
            1.0,
        )
        hold_score = np.clip(self._push_hold_count / max(1, self.hold_steps), 0.0, 1.0)
        return float(min(move_score, tilt_score, hold_score))

    def check_success(self, info: dict[str, Any], *args, **kwargs) -> bool:
        return self._push_hold_count >= self.hold_steps

    def preload_objects(self) -> list[Actor]:
        asset = AssetManager.get("objects").load("chair/chair_319_0p9")
        return [ObjectActor(asset=asset)]
