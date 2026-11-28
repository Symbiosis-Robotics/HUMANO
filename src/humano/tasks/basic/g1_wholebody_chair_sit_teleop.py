from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np

from humano.assets import AssetManager
from humano.core.actor import Actor, ObjectActor
from humano.dr import DistractorDRCfg, LanguageDRCfg, SpatialDRCfg, TargetDRCfg
from humano.dr.types import Box
from humano.tasks.basic.initial_positions import (
    MOVEMENT_EXTRA_ROBOT_TARGET_DISTANCE_M,
    MOVEMENT_MIN_ROBOT_TARGET_DISTANCE_M,
    resolve_robot_around_object_spawn,
)
from humano.tasks.simple.g1_wholebody_xmove_pick_teleop import G1WholebodyXMovePickTaskTeleop
from humano.tasks.registry import TaskRegistry

_CHAIR_ASSET_ID = "objects:chair/chair_319_0p9"
_SIT_CHAIR_CENTER_XY = (-0.28, 0.0)
_SIT_CHAIR_ROBOT_SPAWN_DISTANCE = 0.67
_SIT_CHAIR_MUJOCO_FRICTION = [1.8, 0.08, 0.01]
_SIT_CHAIR_MUJOCO_MASS_KG = 12.0
_SIT_CHAIR_TOUCH_DISTANCE_M = 0.005
_SIT_CONTACT_BODY_KEYWORDS = ("pelvis", "hip", "waist", "torso")


@TaskRegistry.register("g1_wholebody_chair_sit_teleop")
class G1WholebodyChairSitTaskTeleop(G1WholebodyXMovePickTaskTeleop):
    uid: str = "g1_wholebody_chair_sit_teleop"
    label: str = "G1 TELEOP Chair Sit"
    description: str = "A task where the G1 robot must sit on a stable movable chair."

    metadata: dict[str, Any] = deepcopy(G1WholebodyXMovePickTaskTeleop.metadata)
    dr_cfgs = deepcopy(G1WholebodyXMovePickTaskTeleop.dr_cfgs)
    dr_cfgs.pop("scene", None)
    dr_cfgs["language"] = LanguageDRCfg(
        instructions=[
            "walk to the chair and sit down",
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
        target_rotate_z=Box(low=-np.pi / 2, high=-np.pi / 2),
        fixed_stable_pose_idx=0,
        obj_surface_map={"target": "default", "distractor": "default"},
    )

    def __init__(
        self,
        *args,
        sit_xy_radius_m: float = 0.28,
        pelvis_z_range_m: tuple[float, float] = (0.40, 0.68),
        base_linear_speed_max_mps: float = 0.12,
        base_angular_speed_max_rps: float = 0.8,
        chair_displacement_max_m: float = 0.13,
        chair_tilt_max_rad: float = np.deg2rad(20.0),
        chair_lift_max_m: float = 0.07,
        hold_steps: int = 30,
        initial_position: str | None = None,
        movement: bool = False,
        **kwargs,
    ) -> None:
        self.dr_cfgs = deepcopy(type(self).dr_cfgs)
        _, robot_region, robot_orientation_region = resolve_robot_around_object_spawn(
            initial_position=initial_position,
            object_xy=_SIT_CHAIR_CENTER_XY,
            distance=max(
                _SIT_CHAIR_ROBOT_SPAWN_DISTANCE,
                MOVEMENT_MIN_ROBOT_TARGET_DISTANCE_M + 1e-6 if movement else 0.0,
            )
            + (MOVEMENT_EXTRA_ROBOT_TARGET_DISTANCE_M if movement else 0.0),
        )
        spatial_cfg = self.dr_cfgs["spatial"]
        assert isinstance(spatial_cfg, SpatialDRCfg)
        spatial_cfg.robot_region = robot_region
        if robot_orientation_region is not None:
            spatial_cfg.robot_orientation_region = robot_orientation_region
        self.sit_xy_radius_m = float(sit_xy_radius_m)
        self.pelvis_z_range_m = (
            float(pelvis_z_range_m[0]),
            float(pelvis_z_range_m[1]),
        )
        self.base_linear_speed_max_mps = float(base_linear_speed_max_mps)
        self.base_angular_speed_max_rps = float(base_angular_speed_max_rps)
        self.chair_displacement_max_m = float(chair_displacement_max_m)
        self.chair_tilt_max_rad = float(chair_tilt_max_rad)
        self.chair_lift_max_m = float(chair_lift_max_m)
        self.hold_steps = int(hold_steps)
        self._sit_hold_count = 0
        self._init_target_xy: np.ndarray | None = None
        self._init_target_z: float | None = None
        self._init_target_up_axis: np.ndarray | None = None
        self._target_body_name: str | None = None
        super().__init__(*args, **kwargs)

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> None:
        super().reset(seed, options)
        self._sit_hold_count = 0
        self._init_target_xy = None
        self._init_target_z = None
        self._init_target_up_axis = None
        self._target_body_name = None
        if isinstance(self._target, ObjectActor):
            self._target.mujoco_freejoint = True
            self._target.mujoco_friction = _SIT_CHAIR_MUJOCO_FRICTION
            self._target.mujoco_mass = _SIT_CHAIR_MUJOCO_MASS_KG
            self._target_body_name = str(self._target.asset.label)

    def _chair_contact_metrics(self, mujoco_env=None) -> dict[str, float | int | bool]:
        if mujoco_env is None:
            return {
                "has_chair_contact": False,
                "chair_contact_count": 0,
                "min_chair_contact_dist": float("inf"),
            }

        model = mujoco_env.mjModel
        data = mujoco_env.mjData
        target_body_name = self._target_body_name
        if target_body_name is None and isinstance(self._target, ObjectActor):
            target_body_name = str(self._target.asset.label)

        contact_count = 0
        min_contact_dist = float("inf")
        for contact_id in range(data.ncon):
            contact = data.contact[contact_id]
            geom1 = model.geom(int(contact.geom1))
            geom2 = model.geom(int(contact.geom2))
            body1 = model.body(geom1.bodyid).name
            body2 = model.body(geom2.bodyid).name
            target_is_body1 = bool(target_body_name and target_body_name in body1)
            target_is_body2 = bool(target_body_name and target_body_name in body2)
            if target_is_body1 == target_is_body2:
                continue
            other_body = body2 if target_is_body1 else body1
            if not other_body or other_body == "world":
                continue
            if not any(key in other_body.lower() for key in _SIT_CONTACT_BODY_KEYWORDS):
                continue

            contact_dist = float(contact.dist)
            if contact_dist > _SIT_CHAIR_TOUCH_DISTANCE_M:
                continue
            contact_count += 1
            min_contact_dist = min(min_contact_dist, contact_dist)

        return {
            "has_chair_contact": contact_count > 0,
            "chair_contact_count": contact_count,
            "min_chair_contact_dist": min_contact_dist,
        }

    def _sit_metrics(self, info: dict[str, Any], mujoco_env=None) -> dict[str, float | int | bool]:
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

        if mujoco_env is not None:
            pelvis_pos = np.asarray(mujoco_env.mjData.body("pelvis").xpos, dtype=np.float32)
            qvel = np.asarray(mujoco_env.mjData.qvel, dtype=np.float32)
            linear_speed = float(np.linalg.norm(qvel[:3]))
            angular_speed = float(np.linalg.norm(qvel[3:6]))
        else:
            proprio = np.asarray(info["proprio"], dtype=np.float32)
            pelvis_pos = np.asarray([proprio[0], proprio[1], proprio[2]], dtype=np.float32)
            linear_speed = 0.0
            angular_speed = 0.0

        up_norm = max(float(np.linalg.norm(chair_up_axis)), 1e-6)
        normalized_up_axis = chair_up_axis / up_norm
        tilt_cos = float(np.clip(np.dot(normalized_up_axis, self._init_target_up_axis), -1.0, 1.0))
        chair_displacement = float(np.linalg.norm(chair_pos[:2] - self._init_target_xy))
        chair_lift = float(chair_pos[2] - self._init_target_z)
        xy_dist = float(np.linalg.norm(pelvis_pos[:2] - chair_pos[:2]))
        pelvis_z = float(pelvis_pos[2])
        contact_metrics = self._chair_contact_metrics(mujoco_env=mujoco_env)
        return {
            "xy_dist": xy_dist,
            "pelvis_z": pelvis_z,
            "linear_speed": linear_speed,
            "angular_speed": angular_speed,
            "chair_displacement": chair_displacement,
            "chair_tilt_cos": tilt_cos,
            "chair_lift": chair_lift,
            **contact_metrics,
        }

    def _is_sitting_now(self, metrics: dict[str, float | int | bool]) -> bool:
        z_low, z_high = self.pelvis_z_range_m
        return (
            metrics["xy_dist"] <= self.sit_xy_radius_m
            and z_low <= metrics["pelvis_z"] <= z_high
            and metrics["linear_speed"] <= self.base_linear_speed_max_mps
            and metrics["angular_speed"] <= self.base_angular_speed_max_rps
            and metrics["chair_displacement"] <= self.chair_displacement_max_m
            and metrics["chair_tilt_cos"] >= np.cos(self.chair_tilt_max_rad)
            and metrics["chair_lift"] <= self.chair_lift_max_m
            and bool(metrics["has_chair_contact"])
        )

    def compute_reward(self, info: dict[str, Any], *args, **kwargs) -> float:
        metrics = self._sit_metrics(info, mujoco_env=kwargs.get("mujoco_env"))
        if self._is_sitting_now(metrics):
            self._sit_hold_count += 1
        else:
            self._sit_hold_count = 0

        xy_score = 1.0 - np.clip(metrics["xy_dist"] / self.sit_xy_radius_m, 0.0, 1.0)
        z_low, z_high = self.pelvis_z_range_m
        z_mid = 0.5 * (z_low + z_high)
        z_half_width = max(1e-6, 0.5 * (z_high - z_low))
        z_score = 1.0 - np.clip(abs(metrics["pelvis_z"] - z_mid) / z_half_width, 0.0, 1.0)
        chair_move_score = 1.0 - np.clip(
            metrics["chair_displacement"] / self.chair_displacement_max_m,
            0.0,
            1.0,
        )
        chair_tilt_score = np.clip(
            (metrics["chair_tilt_cos"] - np.cos(self.chair_tilt_max_rad))
            / (1.0 - np.cos(self.chair_tilt_max_rad)),
            0.0,
            1.0,
        )
        hold_score = np.clip(self._sit_hold_count / max(1, self.hold_steps), 0.0, 1.0)
        contact_score = 1.0 if bool(metrics["has_chair_contact"]) else 0.0
        return float(
            min(xy_score, z_score, chair_move_score, chair_tilt_score, contact_score, hold_score)
        )

    def check_success(self, info: dict[str, Any], *args, **kwargs) -> bool:
        return self._sit_hold_count >= self.hold_steps

    def preload_objects(self) -> list[Actor]:
        asset = AssetManager.get("objects").load("chair/chair_319_0p9")
        return [ObjectActor(asset=asset)]
