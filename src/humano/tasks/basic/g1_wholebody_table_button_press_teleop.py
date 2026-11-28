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
    resolve_table_articulated_region,
)
from humano.tasks.registry import TaskRegistry

_BUTTON_ASSET_ID = "articulated:6"
_BUTTON_BODY_NAMES = (
    "E_button_01_13",
    "E_button_02_6",
)
_CONTACT_BODY_KEYWORDS = (
    "palm",
    "thumb",
    "index",
    "middle",
    "ring",
    "little",
    "pinky",
)
_NEAR_BODY_KEYWORDS = (
    "thumb_2",
    "index_1",
    "middle_1",
    "distal",
    "tip",
)
_BUTTON_GEOM_KEYWORDS = (
    "E_button_01_13",
    "E_button_02_6",
)


class _ButtonPressDRManager:
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


@TaskRegistry.register("g1_wholebody_table_button_press_teleop")
class G1WholebodyTableButtonPressTaskTeleop(Task):
    uid: str = "g1_wholebody_table_button_press_teleop"
    label: str = "G1 TELEOP Table Button Press"
    description: str = "A task where the G1 robot must press a button on the table."
    button_body_names: tuple[str, ...] = _BUTTON_BODY_NAMES
    button_geom_keywords: tuple[str, ...] = _BUTTON_GEOM_KEYWORDS
    button_press_direction_world: tuple[float, float, float] = (0.0, 0.0, -1.0)

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

    robot_cfg: dict[str, Any] = dict(
        uid="g1_sonic",
    )

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
            pose=dict(
                position=[0.0, 0.0, 0.0],
            ),
        ),
    )

    dr_cfgs = dict(
        language=LanguageDRCfg(
            instructions=[
                "move forward to the table and press the button",
            ]
        ),
        articulated=ArticulatedObjectDrCfg(
            asset_id=_BUTTON_ASSET_ID,
        ),
        spatial=SpatialDRCfg(
            spatial_mode="fixed",
            robot_region=Box(low=[-0.95, 0.0, 0.0], high=[-0.95, 0.0, 0.0]),
            target_region=Box(low=[0.0, 0.0], high=[0.0, 0.0]),
            distractors_region=Box(low=[0.0, 0.0], high=[0.0, 0.0]),
            articulated_region=Box(low=[-0.275, -0.06, 0.65], high=[-0.275, -0.06, 0.65]),
            articulated_rotate_z=Box(low=0.0, high=0.0),
        ),
        scene=TabletopSceneDRCfg(
            scene_mode="fixed",
            table_position=Box(low=[0.3, 0.0], high=[0.3, 0.0]),
            table_height=Box(low=0.65, high=0.65),
            room_choices=["hssd:scene0"],
            scene_manager="hssd",
        ),
        camera=CameraDRCfg(
            cam_id="franka_camera",
        ),
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
        material=MaterialDRCfg(
            material_mode="rand_all",
        ),
    )

    def __init__(
        self,
        robot_uid: str = "g1_sonic",
        scene_uid: str | Scene = "hssd:scene0",
        target: str | None = None,
        controller_uid: str = "pd_joint_pos",
        split: str = "train",
        render_hz: int | None = None,
        dr_level: int = 0,
        touch_distance_m: float = 0.001,
        min_downward_force_n: float = 20.0,
        min_contact_points: int = 1,
        hold_steps: int = 50,
        hand_type: str = "dex3",
        initial_position: str | None = None,
        movement: bool = False,
        *args,
        **kwargs,
    ):
        del controller_uid
        self._instruction = None
        self._layout = None
        self._button_hold_count = 0
        self.touch_distance_m = float(touch_distance_m)
        self.min_downward_force_n = float(min_downward_force_n)
        self.min_contact_points = int(min_contact_points)
        self.hold_steps = int(hold_steps)

        self.robot_cfg.update(dict(uid=robot_uid))

        dr_cfgs = deepcopy(self.dr_cfgs)
        if target is not None and target.startswith("articulated:"):
            dr_cfgs["articulated"].asset_id = target
        if scene_uid is not None:
            dr_cfgs["scene"].room_choices = [str(scene_uid).split(":", 1)[-1]]
        articulated_region = resolve_table_articulated_region(
            initial_position=initial_position,
            z=0.65,
        )
        if articulated_region is not None:
            dr_cfgs["spatial"].articulated_region = articulated_region
        if movement:
            apply_movement_layout(
                dr_cfgs,
                target_region_name="articulated_region",
                flip_articulated_yaw=True,
            )

        hand_spec = kwargs.pop("hand_spec", None) or get_hand_spec(hand_type)
        self._robot = RobotRegistry.make(**self.robot_cfg, hand_spec=hand_spec, **kwargs)

        drmgr = _ButtonPressDRManager(**dr_cfgs)
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
        self._button_hold_count = 0
        self.robot.reset(spawn_pose=self.layout.robot.pose)

    def _button_metrics(self, mujoco_env=None) -> dict[str, float | int | bool]:
        if mujoco_env is None:
            return {
                "min_hand_button_dist": float("inf"),
                "min_button_contact_dist": float("inf"),
                "has_button_contact": False,
                "button_contact_count": 0,
                "button_downward_force_n": 0.0,
                "has_downward_button_contact": False,
            }

        model = mujoco_env.mjModel
        data = mujoco_env.mjData
        button_positions = [
            np.asarray(data.body(name).xpos, dtype=np.float32) for name in self.button_body_names
        ]
        fingertip_positions = []
        for body_id in range(model.nbody):
            body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            if body_name and any(key in body_name.lower() for key in _NEAR_BODY_KEYWORDS):
                fingertip_positions.append(np.asarray(data.body(body_id).xpos, dtype=np.float32))

        if fingertip_positions:
            min_dist = min(
                float(np.linalg.norm(fingertip_pos - button_pos))
                for fingertip_pos in fingertip_positions
                for button_pos in button_positions
            )
        else:
            min_dist = float("inf")

        button_geom_ids = set()
        hand_body_ids = set()
        for geom_id in range(model.ngeom):
            geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
            body_id = int(model.geom_bodyid[geom_id])
            body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
            if any(key in geom_name for key in self.button_geom_keywords):
                button_geom_ids.add(geom_id)
            if any(key in body_name.lower() for key in _CONTACT_BODY_KEYWORDS):
                hand_body_ids.add(body_id)

        min_button_contact_dist = float("inf")
        button_contact_count = 0
        button_downward_force_n = 0.0
        press_direction = np.asarray(self.button_press_direction_world, dtype=np.float64)
        press_direction /= max(float(np.linalg.norm(press_direction)), 1e-6)
        for contact_id in range(data.ncon):
            contact = data.contact[contact_id]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            body1 = int(model.geom_bodyid[geom1])
            body2 = int(model.geom_bodyid[geom2])
            button_is_geom1 = geom1 in button_geom_ids and body2 in hand_body_ids
            button_is_geom2 = geom2 in button_geom_ids and body1 in hand_body_ids
            if not button_is_geom1 and not button_is_geom2:
                continue

            button_contact_count += 1
            min_button_contact_dist = min(min_button_contact_dist, float(contact.dist))

            contact_force = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(model, data, contact_id, contact_force)
            contact_frame = np.asarray(contact.frame, dtype=np.float64).reshape(3, 3)
            world_force_on_geom2 = contact_frame.T @ contact_force[:3]

            # MuJoCo reports force on geom2. Convert it to force on the button,
            # then project it onto the selected button's press direction.
            force_on_button = -world_force_on_geom2 if button_is_geom1 else world_force_on_geom2
            press_force = max(0.0, float(np.dot(force_on_button, press_direction)))
            button_downward_force_n += press_force

        has_button_contact = button_contact_count >= self.min_contact_points
        has_downward_button_contact = (
            has_button_contact
            and min_button_contact_dist <= self.touch_distance_m
            and button_downward_force_n >= self.min_downward_force_n
        )

        return {
            "min_hand_button_dist": min_dist,
            "min_button_contact_dist": min_button_contact_dist,
            "has_button_contact": has_button_contact,
            "button_contact_count": button_contact_count,
            "button_downward_force_n": button_downward_force_n,
            "has_downward_button_contact": has_downward_button_contact,
        }

    def compute_reward(self, info: dict[str, Any], *args, **kwargs) -> float:
        del info, args
        metrics = self._button_metrics(mujoco_env=kwargs.get("mujoco_env"))
        is_pressing_down = bool(metrics["has_downward_button_contact"])
        if is_pressing_down:
            self._button_hold_count += 1
        else:
            self._button_hold_count = 0

        touch_score = 1.0 if is_pressing_down else 0.0
        hold_score = np.clip(self._button_hold_count / max(1, self.hold_steps), 0.0, 1.0)
        return float(min(touch_score, hold_score))

    def check_success(self, info: dict[str, Any], *args, **kwargs) -> bool:
        del info, args, kwargs
        return self._button_hold_count >= self.hold_steps

    def preload_objects(self) -> list[Actor]:
        return []
