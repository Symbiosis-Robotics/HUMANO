from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import transforms3d as t3d
import trimesh
from gymnasium import spaces

from humano.assets import AssetManager
from humano.core.actor import Actor, ObjectActor
from humano.core.layout import Layout
from humano.dr import (
    ArticulatedObjectDrCfg,
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


_PEN_ASSET_ID = "objects:pen/pen"
_PEN_LOAD_ID = "pen/pen"
_PENCIL_CASE_ASSET_ID = "articulated:9"
_PENCIL_CASE_JOINT_QPOS = {"articulate_joint_0": -np.deg2rad(100.0)}
_PENCIL_CASE_POSITION = [-0.22, 0.16, 0.651]
_PENCIL_CASE_ROTATE_Z = -np.pi / 2.0
_PENCIL_CASE_BASE_BODY = "009_base"
_PENCIL_CASE_CONTACT_BODY_KEYWORDS = (
    "009_base",
    "E_pencil_case",
)
_PEN_LOCAL_INSIDE_LOW = np.array([-0.097, -0.023, -0.004], dtype=np.float64)
_PEN_LOCAL_INSIDE_HIGH = np.array([0.091, 0.023, 0.021], dtype=np.float64)
_PEN_PLACE_LOCAL_OFFSET = np.array([-0.004, 0.0, 0.0], dtype=np.float64)
_PEN_PLACE_LOCAL_ROTATION = np.eye(3, dtype=np.float64)
_HIGH_TABLE_HEIGHT = 0.65
_SUCCESS_REWARD_INCREMENT = 0.1


def _body_pose(mujoco_env, body_name: str) -> tuple[np.ndarray, np.ndarray] | None:
    try:
        body = mujoco_env.mjModel.body(body_name)
    except KeyError:
        return None
    body_id = body.id
    data = mujoco_env.mjData
    position = np.asarray(data.xpos[body_id], dtype=np.float64)
    rotation = np.asarray(data.xmat[body_id], dtype=np.float64).reshape(3, 3)
    return position, rotation


@TaskRegistry.register("g1_wholebody_table_pen_to_pencil_case_teleop")
class G1WholebodyTablePenToPencilCaseTaskTeleop(G1WholebodyXMovePickTaskTeleop):
    uid: str = "g1_wholebody_table_pen_to_pencil_case_teleop"
    label: str = "G1 TELEOP Table Pen To Pencil Case"
    description: str = (
        "A task where the G1 robot must pick up a pen on a high table and place it "
        "fully inside an open pencil case on the same high table."
    )

    metadata: dict[str, Any] = deepcopy(G1WholebodyXMovePickTaskTeleop.metadata)
    dr_cfgs = deepcopy(G1WholebodyXMovePickTaskTeleop.dr_cfgs)
    dr_cfgs["language"] = LanguageDRCfg(
        instructions=[
            "pick up the pen on the high table and place it fully inside the pencil case",
        ]
    )
    dr_cfgs["target"] = TargetDRCfg(asset_id=_PEN_ASSET_ID)
    dr_cfgs["articulated"] = ArticulatedObjectDrCfg(
        asset_id=_PENCIL_CASE_ASSET_ID,
        articulate_init_joint_qpos=_PENCIL_CASE_JOINT_QPOS,
    )
    dr_cfgs["distractors"] = DistractorDRCfg(
        res_id="objects",
        number_of_distractors=0,
        allow_duplicates=False,
    )
    dr_cfgs["spatial"] = SpatialDRCfg(
        spatial_mode="fixed",
        robot_region=Box(low=[-0.95, 0.0, 0.0], high=[-0.95, 0.0, 0.0]),
        target_region=Box(low=[-0.275, -0.18], high=[-0.275, -0.18]),
        distractors_region=Box(low=[0.0, 0.0], high=[0.0, 0.0]),
        articulated_region=Box(low=_PENCIL_CASE_POSITION, high=_PENCIL_CASE_POSITION),
        target_stable_indices=[0],
        target_rotate_z=Box(low=0.0, high=0.0),
        articulated_rotate_z=Box(low=_PENCIL_CASE_ROTATE_Z, high=_PENCIL_CASE_ROTATE_Z),
        fixed_stable_pose_idx=0,
        obj_surface_map={"target": "table"},
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
            resolved_layout, target_region, _ = layout
            pencil_case_position = [
                resolved_layout.container_xy[0],
                resolved_layout.container_xy[1],
                _PENCIL_CASE_POSITION[2],
            ]
            spatial_cfg = self.dr_cfgs["spatial"]
            assert isinstance(spatial_cfg, SpatialDRCfg)
            spatial_cfg.target_region = target_region
            spatial_cfg.articulated_region = Box(
                low=pencil_case_position,
                high=list(pencil_case_position),
            )
        super().__init__(*args, target=target, **kwargs)
        self._pen_collision_points: np.ndarray | None = None

    @property
    def layout(self) -> Layout:
        assert self._layout is not None, "call reset() first"
        return self._layout

    @property
    def action_space(self) -> spaces.Space:
        assert isinstance(self.robot, Controllable)
        return self.robot.controller.action_space

    @property
    def pencil_case(self) -> Actor:
        pencil_case = self.layout.actors.get("articulated")
        assert pencil_case is not None, "call reset() first"
        return pencil_case

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> None:
        super().reset(seed, options)
        self.reward = 0.0

    def state_dict(self) -> dict[str, Any]:
        state_dict = super().state_dict()
        state_dict.update({"pencil_case_uid": self.pencil_case.uid if self._layout else None})
        return state_dict

    def _load_pen_collision_points(self) -> np.ndarray:
        if self._pen_collision_points is not None:
            return self._pen_collision_points

        points: list[np.ndarray] = []
        for mesh_path in self.target.asset.collision_meshes_mujoco:
            mesh = trimesh.load_mesh(Path(mesh_path), force="mesh", process=False)
            if mesh.is_empty or len(mesh.vertices) == 0:
                continue
            points.append(np.asarray(mesh.vertices, dtype=np.float64))
        if not points:
            raise RuntimeError("No collision vertices found for pen success check.")
        self._pen_collision_points = np.vstack(points)
        return self._pen_collision_points

    def _pen_points_in_case_local(self, mujoco_env) -> np.ndarray | None:
        pen_name = str(self.target.asset.label)
        pen_pose = _body_pose(mujoco_env, pen_name)
        case_pose = _body_pose(mujoco_env, _PENCIL_CASE_BASE_BODY)
        if pen_pose is None or case_pose is None:
            return None

        pen_position, pen_rotation = pen_pose
        case_position, case_rotation = case_pose
        local_points = self._load_pen_collision_points()
        world_points = local_points @ pen_rotation.T + pen_position
        return (world_points - case_position) @ case_rotation

    def _target_contacts_pencil_case(self, mujoco_env) -> bool:
        target_name = str(self.target.asset.label)
        mj_physics_data = mujoco_env.mjData
        mj_physics_model = mujoco_env.mjModel

        for i_contact in range(mj_physics_data.ncon):
            contact = mj_physics_data.contact[i_contact]
            g1 = mj_physics_model.geom(contact.geom1)
            g2 = mj_physics_model.geom(contact.geom2)
            body1 = mj_physics_model.body(g1.bodyid).name
            body2 = mj_physics_model.body(g2.bodyid).name
            target_first = target_name in body1 and any(
                keyword in body2 for keyword in _PENCIL_CASE_CONTACT_BODY_KEYWORDS
            )
            target_second = target_name in body2 and any(
                keyword in body1 for keyword in _PENCIL_CASE_CONTACT_BODY_KEYWORDS
            )
            if target_first or target_second:
                return True
        return False

    def check_pen_fully_inside_case(self, *args, **kwargs) -> bool:
        del args
        mujoco_env = kwargs.get("mujoco_env", None)
        if mujoco_env is None:
            return False

        points_case_local = self._pen_points_in_case_local(mujoco_env)
        if points_case_local is None:
            return False

        inside = bool(
            np.all(points_case_local >= _PEN_LOCAL_INSIDE_LOW)
            and np.all(points_case_local <= _PEN_LOCAL_INSIDE_HIGH)
        )
        return bool(inside and self._target_contacts_pencil_case(mujoco_env))

    def check_object_in_container(self, *args, **kwargs) -> bool:
        return self.check_pen_fully_inside_case(*args, **kwargs)

    def compute_reward(self, info: dict[str, Any], *args, **kwargs) -> float:
        del info
        if self.check_pen_fully_inside_case(*args, **kwargs):
            self.reward += _SUCCESS_REWARD_INCREMENT
        else:
            self.reward = 0.0
        return float(self.reward)

    def preload_objects(self) -> list[Actor]:
        objects = AssetManager.get("objects")
        pen = objects.load(_PEN_LOAD_ID)
        return [ObjectActor(asset=pen)]

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

        case_rotation = t3d.quaternions.quat2mat(self.pencil_case.pose.quaternion)
        place_position = np.asarray(self.pencil_case.pose.position) + (
            case_rotation @ _PEN_PLACE_LOCAL_OFFSET
        )
        place_orientation = t3d.quaternions.mat2quat(
            case_rotation @ _PEN_PLACE_LOCAL_ROTATION
        )
        hand_uid = self._robot.hand_spec.uid_for("right")

        return [
            StandSpec("initialize"),
            WalkSpec("walk_to_table", vx=0.35, target_yaw=0.0, target_distance=0.45, steps=300),
            StandSpec("stand_at_table"),
            PhaseBreakSpec("phase_break_before_grasp", grasp_type="bodex"),
            GraspObjectSpec(
                "grasp_pen",
                target_uid=self.target.uid,
                pregrasp=False,
                grasp_type="bodex",
                hand_uid=hand_uid,
                lock_links=["left_hand_palm_link"],
            ),
            LiftSpec("lift_pen", up=0.08, grasp_type="bodex", hand_uid=hand_uid),
            PhaseBreakSpec("phase_break_before_place", grasp_type="bodex"),
            MoveEEFToPoseSpec(
                "move_to_pencil_case",
                position=place_position,
                orientation=place_orientation,
                grasp_type="bodex",
                hand_uid=hand_uid,
                lock_links=["waist_roll_link", "left_hand_palm_link"],
            ),
            OpenGripperSpec("release_pen", hand_uid=hand_uid),
            StandSpec("stop"),
        ]
