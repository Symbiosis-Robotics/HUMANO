from __future__ import annotations

from copy import deepcopy
from typing import Any

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
    TabletopSceneDRCfg,
)
from humano.dr.types import Box
from humano.robots.hands import get_hand_spec
from humano.robots.protocols import Controllable
from humano.robots.registry import RobotRegistry
from humano.sensors import SensorCfg, StereoCameraCfg
from humano.sensors.config import DEFAULT_HEAD_CAMERA_FOV
from humano.tasks.basic.initial_positions import (
    apply_movement_layout,
    resolve_drawer_initial_position,
)
from humano.tasks.registry import TaskRegistry

_DRAWER_ASSET_ID = "articulated:8"
_DRAWER_JOINT = "articulate_joint_0"


class _DrawerOpenDRManager:
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


@TaskRegistry.register("g1_wholebody_drawer_open_teleop")
class G1WholebodyDrawerOpenTaskTeleop(Task):
    uid: str = "g1_wholebody_drawer_open_teleop"
    label: str = "G1 TELEOP Drawer Open"
    description: str = "A task where the G1 robot must open the drawer."

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
                "walk to the table and open the drawer",
            ]
        ),
        articulated=ArticulatedObjectDrCfg(
            asset_id=_DRAWER_ASSET_ID,
            articulate_init_joint_qpos={_DRAWER_JOINT: 0.0},
        ),
        spatial=SpatialDRCfg(
            spatial_mode="fixed",
            robot_region=Box(low=[-1.05, 0.0, 0.0], high=[-1.05, 0.0, 0.0]),
            target_region=Box(low=[0.0, 0.0], high=[0.0, 0.0]),
            distractors_region=Box(low=[0.0, 0.0], high=[0.0, 0.0]),
            articulated_region=Box(low=[-0.25, 0.0, 0.65], high=[-0.25, 0.0, 0.65]),
            articulated_rotate_z=Box(low=-0.5 * np.pi, high=-0.5 * np.pi),
        ),
        scene=TabletopSceneDRCfg(
            scene_mode="fixed",
            table_position=Box(low=[0.3, 0.0], high=[0.3, 0.0]),
            table_height=Box(low=0.65, high=0.65),
            room_choices=["hssd:scene0"],
            scene_manager="hssd",
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
        open_threshold_m: float = 0.12,
        hold_steps: int = 10,
        hand_type: str = "dex3",
        initial_position: str | None = None,
        movement: bool = False,
        *args,
        **kwargs,
    ) -> None:
        del controller_uid
        self._instruction = None
        self._layout = None
        self.open_threshold_m = float(open_threshold_m)
        self.hold_steps = int(hold_steps)
        self._open_hold_count = 0

        self.robot_cfg.update(dict(uid=robot_uid))
        hand_spec = kwargs.pop("hand_spec", None) or get_hand_spec(hand_type)
        self._robot = RobotRegistry.make(**self.robot_cfg, hand_spec=hand_spec, **kwargs)

        dr_cfgs = deepcopy(self.dr_cfgs)
        if target is not None and target.startswith("articulated:"):
            dr_cfgs["articulated"].asset_id = target

        if scene_uid is not None:
            dr_cfgs["scene"].room_choices = [str(scene_uid).split(":", 1)[-1]]
        drawer_position = resolve_drawer_initial_position(initial_position)
        if drawer_position is not None:
            _, table_height, articulated_region = drawer_position
            dr_cfgs["scene"].table_height = Box(low=table_height, high=table_height)
            dr_cfgs["spatial"].articulated_region = articulated_region
        if movement:
            apply_movement_layout(
                dr_cfgs,
                target_region_name="articulated_region",
                flip_articulated_yaw=True,
            )

        drmgr = _DrawerOpenDRManager(**dr_cfgs)
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
        self._open_hold_count = 0
        self.robot.reset(spawn_pose=self.layout.robot.pose)

    def _drawer_qpos(self, mujoco_env=None) -> float:
        if mujoco_env is None:
            return 0.0
        return float(mujoco_env.mjData.joint(_DRAWER_JOINT).qpos[0])

    def _is_open_now(self, drawer_qpos: float) -> bool:
        return -drawer_qpos >= self.open_threshold_m

    def compute_reward(self, info: dict[str, Any], *args, **kwargs) -> float:
        del info, args
        drawer_qpos = self._drawer_qpos(mujoco_env=kwargs.get("mujoco_env"))
        if self._is_open_now(drawer_qpos):
            self._open_hold_count += 1
        else:
            self._open_hold_count = 0

        open_score = np.clip((-drawer_qpos) / self.open_threshold_m, 0.0, 1.0)
        hold_score = np.clip(self._open_hold_count / max(1, self.hold_steps), 0.0, 1.0)
        return float(min(open_score, hold_score))

    def check_success(self, info: dict[str, Any], *args, **kwargs) -> bool:
        del info, args, kwargs
        return self._open_hold_count >= self.hold_steps

    def preload_objects(self) -> list[Actor]:
        return []
