from __future__ import annotations

from copy import deepcopy
from typing import Any

import mujoco
import numpy as np

from humano.dr import LanguageDRCfg, SpatialDRCfg, TabletopSceneDRCfg
from humano.dr.types import Box
from humano.tasks.basic.g1_wholebody_dolly_push_teleop import (
    _DOLLY_BASE_BODY,
    _DOLLY_FLOATING_BASE_JOINT,
    G1WholebodyDollyPushTaskTeleop,
)
from humano.tasks.basic.initial_positions import resolve_advanced_dolly_push_to_table_layout
from humano.tasks.registry import TaskRegistry

_TABLE_XY = [1.35, 0.0]
_TABLE_HEIGHT = 0.65
_MIN_UNDER_TABLE_OVERLAP_RATIO = 0.65
_MIN_UNDER_TABLE_X_DEPTH_M = 0.90
_MIN_TABLE_Y_OVERLAP_RATIO = 0.90
_TABLE_CONTACT_REWARD = 0.1


@TaskRegistry.register("g1_wholebody_dolly_push_to_table_teleop")
class G1WholebodyDollyPushToTableTaskTeleop(G1WholebodyDollyPushTaskTeleop):
    uid: str = "g1_wholebody_dolly_push_to_table_teleop"
    label: str = "G1 TELEOP Dolly Push To Table"
    description: str = (
        "A task where the G1 robot must push a dolly forward until most of the dolly "
        "footprint is under a high table."
    )

    metadata: dict[str, Any] = deepcopy(G1WholebodyDollyPushTaskTeleop.metadata)
    dr_cfgs = deepcopy(G1WholebodyDollyPushTaskTeleop.dr_cfgs)
    dr_cfgs["language"] = LanguageDRCfg(
        instructions=[
            "push the dolly forward until most of its footprint is under the high table",
        ]
    )
    dr_cfgs["scene"] = TabletopSceneDRCfg(
        scene_mode="fixed",
        table_position=Box(low=_TABLE_XY, high=_TABLE_XY),
        table_height=Box(low=_TABLE_HEIGHT, high=_TABLE_HEIGHT),
        room_choices=["hssd:scene0"],
        scene_manager="hssd",
    )
    spatial_cfg = dr_cfgs["spatial"]
    assert isinstance(spatial_cfg, SpatialDRCfg)
    spatial_cfg.robot_region = Box(low=[-1.15, 0.0, 0.0], high=[-1.15, 0.0, 0.0])
    spatial_cfg.robot_orientation_region = None
    spatial_cfg.articulated_region = Box(low=[0.25, 0.0, 0.0], high=[0.25, 0.0, 0.0])

    def __init__(
        self,
        *args,
        min_under_table_overlap_ratio: float = _MIN_UNDER_TABLE_OVERLAP_RATIO,
        min_under_table_x_depth_m: float = _MIN_UNDER_TABLE_X_DEPTH_M,
        min_table_y_overlap_ratio: float = _MIN_TABLE_Y_OVERLAP_RATIO,
        hold_steps: int = 10,
        initial_position: str | None = None,
        **kwargs,
    ) -> None:
        self.dr_cfgs = deepcopy(type(self).dr_cfgs)
        layout = resolve_advanced_dolly_push_to_table_layout(initial_position)
        resolved_initial_position = None
        if layout is not None:
            resolved_layout, _, articulated_region, table_position = layout
            resolved_initial_position = resolved_layout.name
            spatial_cfg = self.dr_cfgs["spatial"]
            assert isinstance(spatial_cfg, SpatialDRCfg)
            spatial_cfg.articulated_region = articulated_region
            self.dr_cfgs["scene"].table_position = table_position
        self.min_under_table_overlap_ratio = float(min_under_table_overlap_ratio)
        self.min_under_table_x_depth_m = float(min_under_table_x_depth_m)
        self.min_table_y_overlap_ratio = float(min_table_y_overlap_ratio)
        self.reward = 0.0
        super().__init__(
            *args,
            hold_steps=hold_steps,
            initial_position=resolved_initial_position,
            **kwargs,
        )

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> None:
        super().reset(seed, options)
        self._push_hold_count = 0
        self.reward = 0.0

    def _dolly_position(self, mujoco_env=None) -> np.ndarray:
        if mujoco_env is None:
            return np.asarray(
                self.layout.actors["articulated"].pose.position,
                dtype=np.float32,
            )

        model = mujoco_env.mjModel
        data = mujoco_env.mjData
        joint_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            _DOLLY_FLOATING_BASE_JOINT,
        )
        if joint_id >= 0:
            qpos_addr = int(model.jnt_qposadr[joint_id])
            return np.asarray(data.qpos[qpos_addr : qpos_addr + 3], dtype=np.float32)
        return np.asarray(data.body(_DOLLY_BASE_BODY).xpos, dtype=np.float32)

    def _table_bounds_xy(self) -> tuple[float, float, float, float]:
        table = self.layout.actors["table"]
        table_pos = np.asarray(table.pose.position, dtype=np.float32)
        table_size = np.asarray(table.size, dtype=np.float32)
        return (
            float(table_pos[0] - 0.5 * table_size[0]),
            float(table_pos[0] + 0.5 * table_size[0]),
            float(table_pos[1] - 0.5 * table_size[1]),
            float(table_pos[1] + 0.5 * table_size[1]),
        )

    def _dolly_footprint_bounds_xy(self, mujoco_env=None) -> tuple[float, float, float, float]:
        if mujoco_env is None:
            dolly_pos = self._dolly_position(mujoco_env=None)
            return (
                float(dolly_pos[0] - 0.35),
                float(dolly_pos[0] + 0.35),
                float(dolly_pos[1] - 0.25),
                float(dolly_pos[1] + 0.25),
            )

        model = mujoco_env.mjModel
        data = mujoco_env.mjData
        base_body_id = model.body(_DOLLY_BASE_BODY).id
        points = []
        for geom_id in range(model.ngeom):
            if int(model.geom_bodyid[geom_id]) == base_body_id:
                points.append(np.asarray(data.geom_xpos[geom_id][:2], dtype=np.float32))
        if not points:
            dolly_pos = self._dolly_position(mujoco_env=mujoco_env)
            points.append(dolly_pos[:2])
        points_xy = np.vstack(points)
        return (
            float(points_xy[:, 0].min()),
            float(points_xy[:, 0].max()),
            float(points_xy[:, 1].min()),
            float(points_xy[:, 1].max()),
        )

    def _under_table_metrics(self, mujoco_env=None) -> dict[str, float]:
        dolly_pos = self._dolly_position(mujoco_env=mujoco_env)
        table_x_min, table_x_max, table_y_min, table_y_max = self._table_bounds_xy()
        dolly_x_min, dolly_x_max, dolly_y_min, dolly_y_max = self._dolly_footprint_bounds_xy(
            mujoco_env=mujoco_env
        )

        overlap_x = max(0.0, min(dolly_x_max, table_x_max) - max(dolly_x_min, table_x_min))
        overlap_y = max(0.0, min(dolly_y_max, table_y_max) - max(dolly_y_min, table_y_min))
        dolly_width_x = max(1e-6, dolly_x_max - dolly_x_min)
        dolly_width_y = max(1e-6, dolly_y_max - dolly_y_min)
        overlap_ratio = float((overlap_x * overlap_y) / (dolly_width_x * dolly_width_y))
        y_overlap_ratio = float(overlap_y / dolly_width_y)
        x_depth = float(overlap_x)
        dolly_metrics = self._dolly_metrics(mujoco_env=mujoco_env)
        return {
            "under_table_overlap_ratio": overlap_ratio,
            "under_table_x_depth": x_depth,
            "table_y_overlap_ratio": y_overlap_ratio,
            "x_gap_to_table_front": float(table_x_min - dolly_pos[0]),
            "dolly_x_min": dolly_x_min,
            "dolly_x_max": dolly_x_max,
            "dolly_y_min": dolly_y_min,
            "dolly_y_max": dolly_y_max,
            "table_x_min": table_x_min,
            "table_x_max": table_x_max,
            "table_y_min": table_y_min,
            "table_y_max": table_y_max,
            "tilt_cos": dolly_metrics["tilt_cos"],
            "lift": dolly_metrics["lift"],
        }

    def _is_under_table_now(self, metrics: dict[str, float]) -> bool:
        return (
            metrics["under_table_overlap_ratio"] >= self.min_under_table_overlap_ratio
            and metrics["under_table_x_depth"] >= self.min_under_table_x_depth_m
            and metrics["table_y_overlap_ratio"] >= self.min_table_y_overlap_ratio
            and metrics["tilt_cos"] >= np.cos(self.max_tilt_rad)
            and metrics["lift"] <= self.max_lift_m
        )

    def compute_reward(self, info: dict[str, Any], *args, **kwargs) -> float:
        del info, args
        metrics = self._under_table_metrics(mujoco_env=kwargs.get("mujoco_env"))
        if self._is_under_table_now(metrics):
            self._push_hold_count += 1
        else:
            self._push_hold_count = 0

        overlap_score = np.clip(
            metrics["under_table_overlap_ratio"] / max(self.min_under_table_overlap_ratio, 1e-6),
            0.0,
            1.0,
        )
        depth_score = np.clip(
            metrics["under_table_x_depth"] / max(self.min_under_table_x_depth_m, 1e-6),
            0.0,
            1.0,
        )
        y_score = np.clip(
            metrics["table_y_overlap_ratio"] / max(self.min_table_y_overlap_ratio, 1e-6),
            0.0,
            1.0,
        )
        hold_score = np.clip(self._push_hold_count / max(1, self.hold_steps), 0.0, 1.0)
        self.reward = float(min(overlap_score, depth_score, y_score, hold_score))
        return self.reward

    def check_success(self, info: dict[str, Any], *args, **kwargs) -> bool:
        del info, args, kwargs
        return self._push_hold_count >= self.hold_steps
