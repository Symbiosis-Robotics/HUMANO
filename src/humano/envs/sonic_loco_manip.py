"""
HUMANO: Humanoid Benchmark and Learning Framework

Originally derived from the SIMPLE project.
Copyright (c) 2025 Songlin Wei and Contributors
Copyright (c) 2026 HUMANO Contributors
Licensed under the terms in LICENSE file.
"""

from __future__ import annotations

import time
from threading import Lock
from typing import Any

import mujoco
import mujoco.viewer
import numpy as np
from regex import F
from unitree_sdk2py.core.channel import ChannelFactoryInitialize

from humano.core.task import Task
from humano.envs.base_dual_env import BaseDualSim
from humano.evals.ood_physics import (
    EVAL_ROBOT_INITIAL_PARAMS,
    EVAL_TASK_ROBOT_INITIAL_STATE,
)
from humano.evals.ood_sensor_noise import SensorNoise


def _yaw_quaternion(quaternion: np.ndarray, yaw: float) -> np.ndarray:
    w, x, y, z = quaternion
    c, s = np.cos(yaw / 2.0), np.sin(yaw / 2.0)
    return np.array([c * w - s * z, c * x - s * y, c * y + s * x, c * z + s * w])


def _apply_eval_robot_initial_params(layout, params: dict[str, Any]) -> None:
    actor = layout.actors["robot"]
    if str(actor.robot.uid) != params["robot_key"]:
        raise ValueError("robot_initial OOD params target a different robot")
    translation = np.asarray(params["translation_xy_m"], dtype=float)
    if translation.shape != (2,) or not np.all(np.isfinite(translation)):
        raise ValueError("robot_initial translation_xy_m must be two finite values")
    yaw = float(params["yaw_rad"])
    if not np.isfinite(yaw):
        raise ValueError("robot_initial yaw_rad must be finite")
    position = list(actor.pose.position)
    position[0] = float(position[0]) + float(translation[0])
    position[1] = float(position[1]) + float(translation[1])
    actor.pose.position = position
    actor.pose.quaternion = _yaw_quaternion(
        np.asarray(actor.pose.quaternion, dtype=float),
        yaw,
    ).tolist()


def _robot_eval_indices(robot) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    qpos_parts = []
    qvel_parts = []
    if robot.use_floating_root_link:
        qpos_parts.append(np.arange(7, dtype=int))
        qvel_parts.append(np.arange(6, dtype=int))
    elif robot.use_constrained_root_link:
        qpos_parts.append(np.arange(1, dtype=int))
        qvel_parts.append(np.arange(1, dtype=int))
    for name in ("body_joint_index", "left_hand_model_index", "right_hand_model_index"):
        joints = np.asarray(getattr(robot, name), dtype=int)
        qpos_parts.append(robot.mjModel.jnt_qposadr[joints])
        qvel_parts.append(robot.mjModel.jnt_dofadr[joints])
    actuator_parts = [
        np.asarray(getattr(robot, name), dtype=int)
        for name in (
            "body_actuator_index",
            "left_hand_model_actuator_index",
            "right_hand_model_actuator_index",
            "left_hand_actuated_actuator_index",
            "right_hand_actuated_actuator_index",
        )
    ]
    return (
        np.unique(np.concatenate(qpos_parts)),
        np.unique(np.concatenate(qvel_parts)),
        np.unique(np.concatenate(actuator_parts)),
    )


def _capture_eval_robot_initial_state(data, robot) -> dict[str, Any]:
    qpos_idx, qvel_idx, ctrl_idx = _robot_eval_indices(robot)
    ctrl_idx = ctrl_idx[ctrl_idx < data.ctrl.size]
    return {
        "qpos_idx": qpos_idx,
        "qvel_idx": qvel_idx,
        "ctrl_idx": ctrl_idx,
        "qpos": data.qpos[qpos_idx].copy(),
        "qvel": data.qvel[qvel_idx].copy(),
        "qacc_warmstart": data.qacc_warmstart[qvel_idx].copy(),
        "ctrl": data.ctrl[ctrl_idx].copy(),
        "time": float(data.time),
    }


def _restore_eval_robot_initial_state(data, snapshot: dict[str, Any]) -> None:
    data.qpos[snapshot["qpos_idx"]] = snapshot["qpos"]
    data.qvel[snapshot["qvel_idx"]] = snapshot["qvel"]
    data.qacc_warmstart[snapshot["qvel_idx"]] = snapshot["qacc_warmstart"]
    data.ctrl[snapshot["ctrl_idx"]] = snapshot["ctrl"]
    data.time = snapshot["time"]


class SonicLocoManipEnv(BaseDualSim):
    _success: bool
    _FAST_ISAAC_RENDER_WARMUP_FRAMES = 40

    def __init__(
        self,
        task: str | Task,
        sonic_config: dict,
        sim_mode="mujoco_isaac",
        headless=True,
        fast_mode: bool = False,
        isaac_discard_solver_result: bool = False,
        camera_render_dt: float | None = None,
        mujoco_excluded_cameras: tuple[str, ...] = (),
        *args,
        **kwargs,
    ) -> None:
        super().__init__(task, sim_mode, headless, *args, sonic_config=sonic_config, **kwargs)
        self.sonic_config = sonic_config
        if camera_render_dt is not None and camera_render_dt <= 0:
            raise ValueError("camera_render_dt must be positive")
        self._mujoco_excluded_cameras = mujoco_excluded_cameras
        self._camera_render_dt = camera_render_dt
        self._camera_observation = None
        self._next_camera_time = 0.0
        self._fast_mode = fast_mode and "isaac" in sim_mode
        self._isaac_discard_solver_result = isaac_discard_solver_result

        self.offscreen = "isaac" in sim_mode or headless
        self.onscreen = not self.offscreen

        self.reward_lock = Lock()

        self.last_reward = 0
        # if self.offscreen:
        #     self.init_renderers()

        try:
            if self.sonic_config.get("INTERFACE", None):
                ChannelFactoryInitialize(
                    self.sonic_config["DOMAIN_ID"], self.sonic_config["INTERFACE"]
                )
            else:
                ChannelFactoryInitialize(self.sonic_config["DOMAIN_ID"])
        except Exception as e:
            print(f"Note: Channel factory initialization attempt: {e}")

        self.sim_thread = None
        self.viewer = None
        self._last_mujoco_frame = None
        self._last_isaac_frame = None
        self._sensor_noise = SensorNoise()
        self._fast_isaac_warmup_pending = self._fast_mode and self.isaac is not None

    def init_publisher(self):
        pass

    # adapted from base_sim.update_viewer
    def update_viewer(self):  # FIXME
        if self.viewer is not None:
            self.viewer.sync()  # where .sync is called?

    def update_reward(self):
        with self.reward_lock:
            self.last_reward = 0

    def get_reward(self):
        with self.reward_lock:
            return self.last_reward

    def _get_obs(self):
        mujoco_joint_items = list(self.mujoco.get_robot_qpos().items())
        qpos = np.asarray([q for _, q in mujoco_joint_items], dtype=np.float32)
        # HACK: we assume qpos is the same in IsaacSim and MuJoCo, so we can just use MuJoCo qpos for IsaacSim observations.
        if self.isaac and not self._fast_mode:
            qpos = self._isaac_qpos_for_mujoco_joints(mujoco_joint_items)

        sim_time = float(self.mujoco.mjData.time)
        if (
            self._camera_render_dt is None
            or self._camera_observation is None
            or sim_time + 1e-9 >= self._next_camera_time
        ):
            self._camera_observation = self._render_frame()
            if self._camera_render_dt is not None:
                self._next_camera_time = sim_time + self._camera_render_dt
        return {"joint_qpos": qpos, **self._camera_observation}

    def _isaac_qpos_for_mujoco_joints(self, mujoco_joint_items):
        isaacsim_joint_indices = []
        for jname, _ in mujoco_joint_items:
            isaac_jname = self.task.robot.jname_mujoco_to_isaac(jname)
            isaacsim_joint_indices.append(self.isaac.robot.get_dof_index(isaac_jname))
        return np.asarray(
            self.isaac.robot.get_joint_positions(joint_indices=isaacsim_joint_indices),
            dtype=np.float32,
        ).reshape(-1)

    def _get_info(self):
        info = {}
        for k, v in self.mujoco.mj_objects.items():
            info[str(k)] = np.concatenate([v.xpos, v.xquat])

        # get wholebody proprioception info
        # currently, it is returned as part of info dict,
        # but we may want to merge into obs in the future
        proprio = self.task.robot.prepare_obs()
        return {**info, "proprio": proprio}

    def _step_isaac_from_mujoco(self) -> None:
        assert self.isaac is not None
        self.isaac.step(self.mujoco, advance_physics=not self._isaac_discard_solver_result)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:  # type: ignore
        super().reset(seed=seed, options=options)
        state = (options or {}).get("state_dict") or {}
        self._sensor_noise = SensorNoise(state.get("ood_sensor_noise"))
        # reset task layout in mujoco with domain randomization
        self.task.reset(seed, options)
        if state.get(EVAL_ROBOT_INITIAL_PARAMS):
            _apply_eval_robot_initial_params(self.task.layout, state[EVAL_ROBOT_INITIAL_PARAMS])
        # update mujoco layout with new task layout
        arena_bytes = (state.get("ood", {}).get("params", {}).get("sampling_mujoco_arena_bytes")
                       if state.get("ood_distractor") else None)
        self.mujoco.update_layout(sonic_config=self.sonic_config, arena_bytes=arena_bytes)

        # FIXME
        self.mjSpec = self.mujoco.mjSpec
        self.mjModel = self.mujoco.mjModel
        self.mjData = self.mujoco.mjData

        # set up viewer
        if self.onscreen:
            if self.viewer is not None:
                self.viewer.close()
            elastic_band = self.task.robot.elastic_band
            self.viewer = mujoco.viewer.launch_passive(
                self.mjModel,
                self.mjData,
                key_callback=elastic_band.MujuocoKeyCallback if elastic_band else None,
                show_left_ui=False,
                show_right_ui=False,
            )
        else:
            mujoco.mj_forward(self.mjModel, self.mjData)
            self.viewer = None

        if self.viewer:
            self.viewer.cam.azimuth = 100
            self.viewer.cam.elevation = -30
            self.viewer.cam.distance = 3
            self.viewer.cam.lookat = np.array([0, 0, 0.38])
            self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE  # mjCAMERA_TRACKING
            viewer_camera = self.task.teleop_viewer_camera
            if viewer_camera is not None:
                self.viewer.cam.azimuth = viewer_camera.get("azimuth", self.viewer.cam.azimuth)
                self.viewer.cam.elevation = viewer_camera.get("elevation", self.viewer.cam.elevation)
                self.viewer.cam.distance = viewer_camera["distance"]
                self.viewer.cam.lookat = np.asarray(
                    self.task.layout.actors["robot"].pose.position
                ) + np.asarray(viewer_camera["lookat_offset"])  # (3,), m, world

        eval_robot_initial = None
        if state.get(EVAL_TASK_ROBOT_INITIAL_STATE):
            eval_robot_initial = _capture_eval_robot_initial_state(self.mjData, self.task.robot)

        if state.get("ood_physics_state") is not None:
            from humano.evals.ood_physics import restore_physics

            restore_physics(self.mjData, state["ood_physics_state"])
            if eval_robot_initial is not None:
                _restore_eval_robot_initial_state(self.mjData, eval_robot_initial)
            mujoco.mj_forward(self.mjModel, self.mjData)
            if state.get("ood_distractor") and state.get("ood_paired_id_initial_step"):
                self.mujoco.step(render=False)
        elif state.get("ood_skip_initial_step", False):
            mujoco.mj_forward(self.mjModel, self.mjData)
        else:
            self.mujoco.step(render=False)

        if self.isaac is not None:
            self.isaac.reset()
            self.isaac.update_layout()
            self._step_isaac_from_mujoco()
            self._fast_isaac_warmup_pending = self._fast_mode

        self.control_decimal = int((1 / self.mujoco.physics_dt) / self.task.render_hz)
        self.step_count = 0
        self.close_cmd_received = False
        self._task_evaluation = None
        self._camera_observation = None
        self._next_camera_time = 0.0
        obs = self._get_obs()
        if self._fast_mode and self.isaac is not None:
            frame_isaac = self.isaac.render()
            self._last_isaac_frame = frame_isaac
            obs.update(self._sensor_noise.apply(frame_isaac))
        info = self._get_info()
        self._success = False
        return obs, info

    def evaluate_task(self, info: dict[str, Any]) -> tuple[float, bool]:
        sim_time = float(self.mujoco.mjData.time)
        if self._task_evaluation is None or self._task_evaluation[0] != sim_time:
            reward = float(self.task.compute_reward(info, mujoco_env=self.mujoco))
            # Preserve the task-progress updates used to label native episodes.
            self._success = bool(self.task.check_success(info, mujoco_env=self.mujoco))
            self._task_evaluation = (sim_time, reward, self._success)
        return self._task_evaluation[1:]

    def step(self, action):
        t_start = time.monotonic()
        for _ in range(self.control_decimal):
            self.mujoco.apply_action(action)
            self.mujoco.step(render=False)
            info = self._get_info()
            reward, terminated = self.evaluate_task(info)
            if self.isaac and not self._fast_mode:
                self._step_isaac_from_mujoco()
            if terminated:
                break

        t_physics = time.monotonic()
        self.step_count += 1
        obs = self._get_obs()

        t_render = time.monotonic()

        # Timing diagnostics (first 3 steps + every 200th, or when slow)
        t_total = time.monotonic() - t_start
        if self.step_count <= 3 or self.step_count % 200 == 0 or t_total > 0.005:
            physics_ms = (t_physics - t_start) * 1000
            render_ms = (t_render - t_physics) * 1000
            print(
                f"[SonicLocoManip] step={self.step_count} "
                f"physics={physics_ms:.1f}ms render={render_ms:.1f}ms "
                f"total={t_total * 1000:.1f}ms fast={self._fast_mode}",
                flush=True,
            )

        truncated = False
        self._success = terminated
        return obs, reward, terminated, truncated, info

    """ def check_fall(self):
        self.fall = False
        if self.mjData.qpos[2] < 0.2:
            self.fall = True
            print(f"Warning: Robot has fallen, height: {self.mjData.qpos[2]:.3f} m")

        if self.fall:
            self.__reset() """

    """ def __reset(self): # FIXME
        mujoco.mj_resetData(self.mjModel, self.mjData)
        # Re-arm the elastic band so the robot hangs safely again after a fall.
        if self.task.robot.elastic_band:
            self.task.robot.elastic_band.length = 0.0
            self.task.robot.elastic_band.enable = True """

    def render(self):
        return self._render_frame()

    def _zero_image_frame(self):
        frame = {}
        for key, space in self.observation_space.spaces.items():
            shape = getattr(space, "shape", None)
            if shape is not None and len(shape) == 3 and shape[-1] == 3:
                frame[key] = np.zeros(space.shape, dtype=space.dtype)
        return frame

    def _render_frame(self):
        if self.isaac and self._fast_mode:
            frame = self._zero_image_frame()
            self._last_mujoco_frame = frame
            return frame

        frame_mujoco = self.mujoco.render(excluded_cameras=self._mujoco_excluded_cameras)
        # Keep the observation contract without rendering the unused right camera.
        if "head_stereo_right" in self._mujoco_excluded_cameras:
            frame_mujoco["head_stereo_right"] = frame_mujoco["head_stereo_left"]
        self._last_mujoco_frame = frame_mujoco

        if self.isaac:
            # Otherwise, render IsaacSim and return its frames.
            frame_isaac = self.isaac.render()
            self._last_isaac_frame = frame_isaac
            return self._sensor_noise.apply(frame_isaac)
        else:
            # MuJoCo-only: filter to observation space keys only
            frame = {}
            for key in self.observation_space.spaces:
                if key in frame_mujoco:
                    frame[key] = frame_mujoco[key]
            return self._sensor_noise.apply(frame)

    def sync_isaac_and_render(self, *, reset_accumulation: bool = False) -> dict[str, np.ndarray]:
        """Sync IsaacSim state to MuJoCo and render one frame.

        Called externally (e.g. by GR00T policy agent) to get up-to-date
        camera images on demand when ``fast_mode`` is active.  The returned
        payload also includes the Isaac-side joint qpos sampled after the sync,
        so policy queries can use state from the same Isaac frame as the image.
        """
        if not self.isaac:
            return self._sensor_noise.apply(self._last_mujoco_frame or {})
        if reset_accumulation or self._fast_isaac_warmup_pending:
            # Loading can consume a variable number of renderer frames before capture.
            self.isaac.reset_render_accumulation()
            for _ in range(self._FAST_ISAAC_RENDER_WARMUP_FRAMES):
                self.isaac.step(self.mujoco, advance_physics=False)
            self._fast_isaac_warmup_pending = False
        self.isaac.step(self.mujoco, advance_physics=False)
        frame = self.isaac.render()
        self._last_isaac_frame = frame
        mujoco_joint_items = list(self.mujoco.get_robot_qpos().items())
        return {
            **self._sensor_noise.apply(frame),
            "isaac_joint_qpos": self._isaac_qpos_for_mujoco_joints(mujoco_joint_items),
        }

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
        super().close()
