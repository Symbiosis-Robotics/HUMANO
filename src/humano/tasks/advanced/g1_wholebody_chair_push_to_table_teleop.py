from __future__ import annotations

from copy import deepcopy
from typing import Any

import mujoco
import numpy as np

from humano.core.actor import ObjectActor
from humano.dr import LanguageDRCfg, SpatialDRCfg, TabletopSceneDRCfg
from humano.dr.types import Box
from humano.tasks.basic.g1_wholebody_chair_push_teleop import G1WholebodyChairPushTaskTeleop
from humano.tasks.basic.initial_positions import resolve_advanced_chair_push_to_table_layout
from humano.tasks.registry import TaskRegistry

_TABLE_XY = [1.0, 0.0]
_TABLE_HEIGHT = 0.65
_MAX_SURFACE_GAP_M = 0.05
_TABLE_Y_MARGIN_M = 0.20
_MAX_TILT_RAD = np.deg2rad(20.0)
_MAX_LIFT_M = 0.05
_CHAIR_FRONT_EXTENT_FALLBACK_M = 0.255


@TaskRegistry.register("g1_wholebody_chair_push_to_table_teleop")
class G1WholebodyChairPushToTableTaskTeleop(G1WholebodyChairPushTaskTeleop):
    uid: str = "g1_wholebody_chair_push_to_table_teleop"
    label: str = "G1 TELEOP Chair Push To Table"
    description: str = (
        "A task where the G1 robot must push a movable chair forward until it is "
        "close to the front edge of a high table."
    )

    metadata: dict[str, Any] = deepcopy(G1WholebodyChairPushTaskTeleop.metadata)
    dr_cfgs = deepcopy(G1WholebodyChairPushTaskTeleop.dr_cfgs)
    dr_cfgs["language"] = LanguageDRCfg(
        instructions=[
            "push the chair to the front of the table",
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
    spatial_cfg.robot_region = Box(low=[-0.95, 0.0, 0.0], high=[-0.95, 0.0, 0.0])
    spatial_cfg.robot_orientation_region = None

    def __init__(
        self,
        *args,
        max_surface_gap_m: float = _MAX_SURFACE_GAP_M,
        table_y_margin_m: float = _TABLE_Y_MARGIN_M,
        max_tilt_rad: float = _MAX_TILT_RAD,
        max_lift_m: float = _MAX_LIFT_M,
        hold_steps: int = 10,
        initial_position: str | None = None,
        **kwargs,
    ) -> None:
        self.dr_cfgs = deepcopy(type(self).dr_cfgs)
        layout = resolve_advanced_chair_push_to_table_layout(initial_position)
        resolved_initial_position = None
        if layout is not None:
            resolved_layout, _, target_region, table_position = layout
            resolved_initial_position = resolved_layout.name
            spatial_cfg = self.dr_cfgs["spatial"]
            assert isinstance(spatial_cfg, SpatialDRCfg)
            spatial_cfg.target_region = target_region
            self.dr_cfgs["scene"].table_position = table_position
        self.max_surface_gap_m = float(max_surface_gap_m)
        self.table_y_margin_m = float(table_y_margin_m)
        self.reward = 0.0
        super().__init__(
            *args,
            max_tilt_rad=max_tilt_rad,
            max_lift_m=max_lift_m,
            hold_steps=hold_steps,
            initial_position=resolved_initial_position,
            **kwargs,
        )

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> None:
        super().reset(seed, options)
        if isinstance(self._target, ObjectActor):
            stable_z = float(self._target.asset.stable_poses[0][2])
            self._target.pose.position[2] = stable_z
        self._push_hold_count = 0
        self.reward = 0.0

    def _chair_position(self, info: dict[str, Any] | None = None, mujoco_env=None) -> np.ndarray:
        if mujoco_env is not None and "target" in getattr(mujoco_env, "mj_objects", {}):
            chair_body = mujoco_env.mj_objects["target"]
            return np.asarray(chair_body.xpos, dtype=np.float32)
        if info is not None and "target" in info:
            return np.asarray(info["target"][:3], dtype=np.float32)
        return np.asarray(self.layout.actors["target"].pose.position, dtype=np.float32)

    def _table_front_center(self) -> np.ndarray:
        table = self.layout.actors["table"]
        table_pos = np.asarray(table.pose.position, dtype=np.float32)
        table_size = np.asarray(table.size, dtype=np.float32)
        return np.asarray(
            [table_pos[0] - 0.5 * table_size[0], table_pos[1]],
            dtype=np.float32,
        )

    @staticmethod
    def _body_is_descendant(model, body_id: int, ancestor_body_id: int) -> bool:
        while body_id > 0:
            if body_id == ancestor_body_id:
                return True
            body_id = int(model.body_parentid[body_id])
        return ancestor_body_id == 0

    def _chair_projected_bounds_xy(
        self,
        mujoco_env=None,
    ) -> tuple[float, float, float, float] | None:
        if mujoco_env is None or "target" not in getattr(mujoco_env, "mj_objects", {}):
            return None

        model = mujoco_env.mjModel
        data = mujoco_env.mjData
        chair_body_id = int(mujoco_env.mj_objects["target"].id)
        projected_points = []
        for geom_id in range(model.ngeom):
            geom_body_id = int(model.geom_bodyid[geom_id])
            if not self._body_is_descendant(model, geom_body_id, chair_body_id):
                continue
            if int(model.geom_contype[geom_id]) == 0 and int(model.geom_conaffinity[geom_id]) == 0:
                continue

            geom_position = np.asarray(data.geom_xpos[geom_id], dtype=np.float64)
            if int(model.geom_type[geom_id]) == int(mujoco.mjtGeom.mjGEOM_MESH):
                mesh_id = int(model.geom_dataid[geom_id])
                vertex_start = int(model.mesh_vertadr[mesh_id])
                vertex_count = int(model.mesh_vertnum[mesh_id])
                local_vertices = np.asarray(
                    model.mesh_vert[vertex_start : vertex_start + vertex_count],
                    dtype=np.float64,
                )
                geom_rotation = np.asarray(data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3)
                world_vertices = local_vertices @ geom_rotation.T + geom_position
                projected_points.append(world_vertices[:, :2])
            else:
                radius = float(model.geom_rbound[geom_id])
                projected_points.append(
                    np.asarray(
                        [geom_position[:2] - radius, geom_position[:2] + radius],
                        dtype=np.float64,
                    )
                )

        if not projected_points:
            return None
        points_xy = np.vstack(projected_points)
        return (
            float(points_xy[:, 0].min()),
            float(points_xy[:, 0].max()),
            float(points_xy[:, 1].min()),
            float(points_xy[:, 1].max()),
        )

    def _near_table_metrics(
        self,
        info: dict[str, Any],
        mujoco_env=None,
    ) -> dict[str, float]:
        chair_pos = self._chair_position(info=info, mujoco_env=mujoco_env)
        table_front_center = self._table_front_center()
        delta_xy = chair_pos[:2] - table_front_center
        x_gap = float(table_front_center[0] - chair_pos[0])
        y_error = float(abs(delta_xy[1]))
        chair_bounds = self._chair_projected_bounds_xy(mujoco_env=mujoco_env)
        if chair_bounds is None:
            chair_x_max = float(chair_pos[0] + _CHAIR_FRONT_EXTENT_FALLBACK_M)
            chair_y_min = float(chair_pos[1] - _CHAIR_FRONT_EXTENT_FALLBACK_M)
            chair_y_max = float(chair_pos[1] + _CHAIR_FRONT_EXTENT_FALLBACK_M)
        else:
            _, chair_x_max, chair_y_min, chair_y_max = chair_bounds
        signed_surface_gap = float(table_front_center[0] - chair_x_max)
        surface_gap = max(0.0, signed_surface_gap)
        chair_metrics = self._chair_metrics(info, mujoco_env=mujoco_env)
        return {
            "x_gap_to_table_front": x_gap,
            "y_error_to_table_front": y_error,
            "surface_gap_to_table": surface_gap,
            "signed_surface_gap_to_table": signed_surface_gap,
            "chair_x_max": chair_x_max,
            "chair_y_min": chair_y_min,
            "chair_y_max": chair_y_max,
            "displacement": chair_metrics["displacement"],
            "tilt_cos": chair_metrics["tilt_cos"],
            "lift": chair_metrics["lift"],
        }

    def _is_near_table_now(self, metrics: dict[str, float]) -> bool:
        return (
            metrics["x_gap_to_table_front"] >= 0.0
            and metrics["surface_gap_to_table"] <= self.max_surface_gap_m
            and metrics["y_error_to_table_front"] <= self.table_y_margin_m
            and metrics["tilt_cos"] >= np.cos(self.max_tilt_rad)
            and metrics["lift"] <= self.max_lift_m
        )

    def compute_reward(self, info: dict[str, Any], *args, **kwargs) -> float:
        del args
        metrics = self._near_table_metrics(info, mujoco_env=kwargs.get("mujoco_env"))
        is_near_table = self._is_near_table_now(metrics)
        if is_near_table:
            self._push_hold_count += 1
        else:
            self._push_hold_count = 0

        hold_score = np.clip(self._push_hold_count / max(1, self.hold_steps), 0.0, 1.0)
        self.reward = float(hold_score if is_near_table else 0.0)
        return self.reward

    def check_success(self, info: dict[str, Any], *args, **kwargs) -> bool:
        del info, args, kwargs
        return self._push_hold_count >= self.hold_steps
