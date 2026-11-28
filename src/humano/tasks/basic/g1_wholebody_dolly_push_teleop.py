from __future__ import annotations

from copy import deepcopy
from typing import Any

import mujoco
import numpy as np
from gymnasium import spaces

from humano.core.actor import Actor
from humano.core.layout import Layout
from humano.core.randomizer import RandomizerCfg
from humano.core.scene import Scene
from humano.core.task import Task
from humano.dr import (
    ArticulatedObjectDrCfg,
    CameraDRCfg,
    LanguageDRCfg,
    LightingDRCfg,
    MaterialDRCfg,
    SpatialDRCfg,
)
from humano.dr.types import Box
from humano.robots.hands import get_hand_spec
from humano.robots.protocols import Controllable
from humano.robots.registry import RobotRegistry
from humano.sensors import SensorCfg, StereoCameraCfg
from humano.sensors.config import DEFAULT_HEAD_CAMERA_FOV
from humano.tasks.basic.initial_positions import (
    MOVEMENT_EXTRA_ROBOT_TARGET_DISTANCE_M,
    MOVEMENT_MIN_ROBOT_TARGET_DISTANCE_M,
    resolve_robot_around_object_spawn,
)
from humano.tasks.registry import TaskRegistry

_DOLLY_ASSET_ID = "articulated:7"
_DOLLY_FLOATING_BASE_JOINT = "articulate_floating_base"
_DOLLY_BASE_BODY = "articulate_base"
_DOLLY_CENTER_XY = (0.25, 0.0)
_DOLLY_ROBOT_SPAWN_DISTANCE = 1.40


class _DollyPushDRManager:
    def __init__(self, **kwargs) -> None:
        self.randomizers = {}
        for key, cfg in kwargs.items():
            if not isinstance(cfg, RandomizerCfg):
                raise TypeError(f"Expected RandomizerCfg for {key}, got {type(cfg)}")
            self.randomizers[key] = cfg.build()

    def get_randomizer(self, name: str):
        return self.randomizers.get(name)

    def reset(self, seed: int | None = None) -> None:
        del seed
        for rnd in self.randomizers.values():
            rnd._inner_state = None

    def state_dict(self) -> dict[str, Any]:
        return {name: rnd.state_dict() for name, rnd in self.randomizers.items()}

    def load_state_dict(self, state_dict: dict[str, Any], dr_level: int | None = None) -> None:
        del dr_level
        for name, rnd_state in state_dict.get("dr_state_dict", {}).items():
            rnd = self.randomizers.get(name)
            if rnd is not None:
                rnd.load_state_dict(rnd_state)


@TaskRegistry.register("g1_wholebody_dolly_push_teleop")
class G1WholebodyDollyPushTaskTeleop(Task):
    uid: str = "g1_wholebody_dolly_push_teleop"
    label: str = "G1 TELEOP Dolly Push"
    description: str = "A task where the G1 robot must push a dolly forward."

    metadata: dict[str, Any] = {
        "physics_dt": 0.005,
        "control_hz": 200,
        "render_hz": 50,
        "dr_level": 0,
        "version": 1.0,
        "reward_dt": 0.02,
        "image_dt": 0.033333,
        "need_gravity": True,
        "max_episode_steps": 800,
    }

    robot_cfg: dict[str, Any] = dict(uid="g1_sonic")

    sensor_cfgs: dict[str, SensorCfg] = dict(
        head_stereo=StereoCameraCfg(
            uid="Realsense_D435i",
            mount="eye_in_head",
            width=640,
            height=360,
            focal_length=1.93,
            fov=DEFAULT_HEAD_CAMERA_FOV,
            near=0.2,
            far=5,
            baseline=0.05,
            pose=dict(position=[0.0, 0.0, 0.0]),
        ),
    )

    dr_cfgs = dict(
        language=LanguageDRCfg(
            instructions=[
                "walk to the dolly and push it forward",
            ]
        ),
        articulated=ArticulatedObjectDrCfg(asset_id=_DOLLY_ASSET_ID),
        spatial=SpatialDRCfg(
            spatial_mode="fixed",
            robot_region=Box(low=[-1.15, 0.0, 0.0], high=[-1.15, 0.0, 0.0]),
            target_region=Box(low=[0.0, 0.0], high=[0.0, 0.0]),
            distractors_region=Box(low=[0.0, 0.0], high=[0.0, 0.0]),
            articulated_region=Box(low=[0.25, 0.0, 0.0], high=[0.25, 0.0, 0.0]),
            articulated_rotate_z=Box(low=-0.5 * np.pi, high=-0.5 * np.pi),
        ),
        camera=CameraDRCfg(cam_id="franka_camera"),
        lighting=LightingDRCfg(
            light_mode="random",
            light_num=(2, 3),
            light_color_temperature=Box(low=2001, high=8001),
            light_intensity=Box(low=1e4, high=1e4),
            light_radius=Box(0.08, 0.12),
            light_length=Box(0.51, 1.1),
            light_spacing=Box((1.0, 1.0), (2.0, 2.0)),
            light_position=Box((-0.5, -0.5, 1.3), (0.5, 0.5, 1.5)),
            light_eulers=Box((0, 0, -0.5 * np.pi), (0, 0, 0.5 * np.pi)),
        ),
        material=MaterialDRCfg(material_mode="rand_all"),
    )

    def __init__(
        self,
        robot_uid: str = "g1_sonic",
        scene_uid: str | Scene | None = None,
        target: str | None = None,
        controller_uid: str = "pd_joint_pos",
        split: str = "train",
        render_hz: int | None = None,
        dr_level: int = 0,
        move_threshold_m: float = 0.50,
        max_tilt_rad: float = np.deg2rad(35.0),
        max_lift_m: float = 0.15,
        hold_steps: int = 10,
        initial_position: str | None = None,
        movement: bool = False,
        hand_type: str = "dex3",
        *args,
        **kwargs,
    ) -> None:
        del controller_uid, scene_uid
        self._instruction = None
        self._layout = None
        self.move_threshold_m = float(move_threshold_m)
        self.max_tilt_rad = float(max_tilt_rad)
        self.max_lift_m = float(max_lift_m)
        self.hold_steps = int(hold_steps)
        self._init_dolly_xy: np.ndarray | None = None
        self._init_dolly_z: float | None = None
        self._init_dolly_up_axis: np.ndarray | None = None
        self._push_hold_count = 0

        self.robot_cfg.update(dict(uid=robot_uid))
        hand_spec = kwargs.pop("hand_spec", None) or get_hand_spec(hand_type)
        self._robot = RobotRegistry.make(**self.robot_cfg, hand_spec=hand_spec, **kwargs)

        dr_cfgs = deepcopy(self.dr_cfgs)
        spatial_cfg = dr_cfgs["spatial"]
        assert isinstance(spatial_cfg, SpatialDRCfg)
        if initial_position is not None or movement:
            _, robot_region, robot_orientation_region = resolve_robot_around_object_spawn(
                initial_position=initial_position,
                object_xy=_DOLLY_CENTER_XY,
                distance=(
                    max(
                        _DOLLY_ROBOT_SPAWN_DISTANCE,
                        MOVEMENT_MIN_ROBOT_TARGET_DISTANCE_M,
                    )
                    + MOVEMENT_EXTRA_ROBOT_TARGET_DISTANCE_M
                    if movement
                    else _DOLLY_ROBOT_SPAWN_DISTANCE
                ),
            )
            spatial_cfg.robot_region = robot_region
            if robot_orientation_region is not None:
                spatial_cfg.robot_orientation_region = robot_orientation_region
        if target is not None and target.startswith("articulated:"):
            dr_cfgs["articulated"].asset_id = target

        drmgr = _DollyPushDRManager(**dr_cfgs)
        super().__init__(
            *args,
            dr=drmgr,
            split=split,
            render_hz=render_hz,
            dr_level=dr_level,
            **kwargs,
        )

    @property
    def layout(self) -> Layout:
        assert self._layout is not None, "call reset() first"
        return self._layout

    @property
    def instruction(self) -> str:
        assert self._instruction is not None, "call reset() first"
        return self._instruction

    @property
    def action_space(self) -> spaces.Space:
        assert isinstance(self.robot, Controllable)
        return self.robot.controller.action_space

    @property
    def observation_space(self) -> spaces.Space:
        default_obs = super().observation_space
        obs: dict[str, Any] = {
            "joint_qpos": spaces.Box(
                -np.pi, np.pi, shape=(self.robot.wholebody_dof,), dtype=np.float32
            ),
        }
        if isinstance(default_obs, spaces.Dict):
            obs.update(dict(default_obs))
        return spaces.Dict(obs)

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> None:
        super().reset(seed, options)
        split = self.metadata.get("split", "train")
        lang_dr = self.dr.get_randomizer("language")
        assert lang_dr is not None
        self._instruction = lang_dr(split)
        self._init_dolly_xy = None
        self._init_dolly_z = None
        self._init_dolly_up_axis = None
        self._push_hold_count = 0
        self.robot.reset(spawn_pose=self.layout.robot.pose)

    def _dolly_metrics(self, mujoco_env=None) -> dict[str, float]:
        if mujoco_env is not None:
            model = mujoco_env.mjModel
            data = mujoco_env.mjData
            joint_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, _DOLLY_FLOATING_BASE_JOINT
            )
            if joint_id >= 0:
                qpos_addr = int(model.jnt_qposadr[joint_id])
                dolly_pos = np.asarray(data.qpos[qpos_addr : qpos_addr + 3], dtype=np.float32)
                dolly_quat = np.asarray(data.qpos[qpos_addr + 3 : qpos_addr + 7], dtype=np.float32)
                dolly_up_axis = np.empty(9, dtype=np.float64)
                mujoco.mju_quat2Mat(dolly_up_axis, dolly_quat)
            else:
                dolly_pos = np.asarray(data.body(_DOLLY_BASE_BODY).xpos, dtype=np.float32)
                dolly_up_axis = np.asarray(data.body(_DOLLY_BASE_BODY).xmat, dtype=np.float32)
            dolly_up_axis = dolly_up_axis.reshape(3, 3)[:, 2]
        else:
            dolly_pos = np.asarray(
                self.layout.actors["articulated"].pose.position, dtype=np.float32
            )
            dolly_up_axis = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)

        if self._init_dolly_xy is None:
            self._init_dolly_xy = dolly_pos[:2].copy()
        if self._init_dolly_z is None:
            self._init_dolly_z = float(dolly_pos[2])
        if self._init_dolly_up_axis is None:
            norm = max(float(np.linalg.norm(dolly_up_axis)), 1e-6)
            self._init_dolly_up_axis = dolly_up_axis / norm

        up_norm = max(float(np.linalg.norm(dolly_up_axis)), 1e-6)
        normalized_up_axis = dolly_up_axis / up_norm
        tilt_cos = float(np.clip(np.dot(normalized_up_axis, self._init_dolly_up_axis), -1.0, 1.0))
        displacement = float(np.linalg.norm(dolly_pos[:2] - self._init_dolly_xy))
        lift = float(dolly_pos[2] - self._init_dolly_z)
        return {
            "displacement": displacement,
            "forward_displacement": float(dolly_pos[0] - self._init_dolly_xy[0]),
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
        del info, args
        metrics = self._dolly_metrics(mujoco_env=kwargs.get("mujoco_env"))
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
        del info, args, kwargs
        return self._push_hold_count >= self.hold_steps

    def preload_objects(self) -> list[Actor]:
        return []
