from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
import transforms3d as t3d
import trimesh

from humano.assets import AssetManager
from humano.core.actor import Actor, ObjectActor
from humano.dr import (
    DistractorDRCfg,
    LanguageDRCfg,
    SpatialDRCfg,
    TabletopSceneDRCfg,
    TargetDRCfg,
)
from humano.dr.types import Box
from humano.tasks.basic.initial_positions import (
    apply_movement_layout,
    resolve_book_initial_position,
)
from humano.tasks.simple.g1_wholebody_xmove_pick_teleop import G1WholebodyXMovePickTaskTeleop
from humano.tasks.registry import TaskRegistry


@TaskRegistry.register("g1_wholebody_table_book_pick_teleop")
class G1WholebodyTableBookPickTaskTeleop(G1WholebodyXMovePickTaskTeleop):
    uid: str = "g1_wholebody_table_book_pick_teleop"
    label: str = "G1 TELEOP Table Book Pick"
    description: str = "A task where the G1 robot must move forward to pick up a book on the table."

    metadata: dict[str, Any] = deepcopy(G1WholebodyXMovePickTaskTeleop.metadata)
    dr_cfgs = deepcopy(G1WholebodyXMovePickTaskTeleop.dr_cfgs)
    dr_cfgs["language"] = LanguageDRCfg(
        instructions=[
            "move forward to the table and pick up the book",
        ]
    )
    dr_cfgs["target"] = TargetDRCfg(asset_id="objects:Book_9388144295_Homer/Book_9388144295_Homer")
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
    )
    dr_cfgs["scene"] = TabletopSceneDRCfg(
        scene_mode="fixed",
        table_position=Box(low=[0.3, 0.0], high=[0.3, 0.0]),
        table_height=Box(low=0.65, high=0.65),
        room_choices=["hssd:scene0"],
        scene_manager="hssd",
    )

    def preload_objects(self) -> list[Actor]:
        asset = AssetManager.get("objects").load("Book_9388144295_Homer/Book_9388144295_Homer")
        return [ObjectActor(asset=asset)]

    def __init__(
        self,
        *args,
        initial_position: str | None = None,
        book_initial_position: str | None = None,
        movement: bool = False,
        **kwargs,
    ) -> None:
        self.dr_cfgs = deepcopy(type(self).dr_cfgs)
        self._book_initial_position = resolve_book_initial_position(
            initial_position=initial_position,
            legacy_initial_position=book_initial_position,
            legacy_name="book_initial_position",
        )
        if movement:
            apply_movement_layout(self.dr_cfgs)
        super().__init__(*args, **kwargs)

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> None:
        super().reset(seed, options)
        self._apply_book_initial_pose()

    def _apply_book_initial_pose(self) -> None:
        if self._target is None:
            return

        rotation = self._book_initial_rotation()
        stable_pose = np.asarray(self._target.asset.stable_poses[0], dtype=np.float32)
        surface_height = float(self._target.pose.position[2]) - float(stable_pose[2])
        stable_z = self._stable_z_for_rotation(rotation)
        position = list(self._target.pose.position)
        position[2] = surface_height + stable_z
        self._target.pose.position = position
        self._target.pose.quaternion = t3d.quaternions.mat2quat(rotation).tolist()
        self._init_target_height = None

    def _book_initial_rotation(self) -> np.ndarray:
        if self._book_initial_position == "fallen_cross":
            return t3d.euler.euler2mat(0.0, 0.0, 0.5 * np.pi)
        if self._book_initial_position == "upright_tall":
            return t3d.euler.euler2mat(0.5 * np.pi, 0.0, 0.0)
        if self._book_initial_position == "upright_long_edge":
            return t3d.euler.euler2mat(0.0, 0.5 * np.pi, 0.0)
        return t3d.euler.euler2mat(0.0, 0.0, 0.0)

    def _stable_z_for_rotation(self, rotation: np.ndarray) -> float:
        mesh_paths = list(getattr(self._target.asset, "collision_meshes_mujoco", []) or [])
        if not mesh_paths:
            mesh_paths = [self._target.asset.collision_mesh_curobo]

        min_z: float | None = None
        for mesh_path in mesh_paths:
            mesh = trimesh.load_mesh(mesh_path, force="mesh")
            if mesh.is_empty:
                continue
            vertices = np.asarray(mesh.vertices, dtype=np.float32) @ rotation.T
            mesh_min_z = float(vertices[:, 2].min())
            min_z = mesh_min_z if min_z is None else min(min_z, mesh_min_z)
        if min_z is None:
            return 0.0
        return max(-min_z, 0.0)
