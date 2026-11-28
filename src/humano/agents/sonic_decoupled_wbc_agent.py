"""
HUMANO: Humanoid Benchmark and Learning Framework

Originally derived from the SIMPLE project.
Copyright (c) 2025 Songlin Wei and Contributors
Copyright (c) 2026 HUMANO Contributors
Licensed under the terms in LICENSE file.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable

import numpy as np
from decoupled_wbc.control.main.teleop.configs.configs import ControlLoopConfig
from decoupled_wbc.control.policy.wbc_policy_factory import get_wbc_policy
from decoupled_wbc.control.robot_model.instantiation.g1 import instantiate_g1_robot_model
from scipy.spatial.transform import Rotation

from humano.core.action import ActionCmd
from humano.robots.g1_sonic import G1Sonic

from .wbc_agent import WholeBodyControlAgent


class SonicDecoupledWbcAgent(WholeBodyControlAgent):
    def __init__(
        self,
        robot: G1Sonic,
        sonic_config: dict,
        time_fn: Callable[[], float] = time.monotonic,
        preserve_spawn_heading: bool = False,
    ):
        self.robot = robot
        self.sim_dt = sonic_config["SIMULATE_DT"]
        self._time_fn = time_fn
        self._preserve_spawn_heading = preserve_spawn_heading
        self._spawn_heading_yaw: float | None = None

        # --- Decoupled WBC pipeline (no teleop/streamer) ---
        self._init_decoupled_policy(sonic_config)
        self._action_queue = deque()

    def _init_decoupled_policy(self, sonic_config: dict):

        enable_waist = sonic_config.get("enable_waist", False)
        waist_location = "lower_and_upper_body" if enable_waist else "lower_body"

        self._dwbc_robot_model = instantiate_g1_robot_model(
            waist_location=waist_location,
            high_elbow_pose=sonic_config.get("high_elbow_pose", False),
        )

        dwbc_config = ControlLoopConfig(
            sim_frequency=round(1.0 / self.sim_dt),
            enable_waist=enable_waist,
            high_elbow_pose=sonic_config.get("high_elbow_pose", False),
        )
        wbc_config = dwbc_config.load_wbc_yaml()
        assert wbc_config["SIMULATE_DT"] == self.sim_dt

        self._wbc_policy = get_wbc_policy(
            "g1",
            self._dwbc_robot_model,
            wbc_config,
            init_time=dwbc_config.upper_body_joint_speed,
        )

        self._upper_body_indices = self._dwbc_robot_model.get_joint_group_indices("upper_body")
        self._dwbc_left_hand_dof = len(
            self._dwbc_robot_model.get_hand_actuated_joint_indices("left")
        )
        self._dwbc_right_hand_dof = len(
            self._dwbc_robot_model.get_hand_actuated_joint_indices("right")
        )
        self._control_frequency = dwbc_config.control_frequency
        self._control_dt = 1.0 / self._control_frequency
        self._cached_target_q = None
        self._cached_left_hand_q = None
        self._cached_right_hand_q = None
        self._t_start = self._time_fn()

    @staticmethod
    def _fit_vector(values, size: int) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float64).reshape(-1)
        if arr.size == size:
            return arr
        fitted = np.zeros(size, dtype=np.float64)
        copy_len = min(size, arr.size)
        if copy_len:
            fitted[:copy_len] = arr[:copy_len]
        return fitted

    def _sim_hand_values(self, sim_obs: dict, side: str, suffix: str, size: int) -> np.ndarray:
        active_key = f"{side}_hand_actuated_{suffix}"
        model_key = f"{side}_hand_{suffix}"
        values = sim_obs.get(active_key)
        if values is None:
            values = sim_obs.get(model_key, np.zeros(self.robot.hand_spec.dof))
        return self._fit_vector(values, size)

    def _robot_hand_action(self, values, side: str) -> np.ndarray:
        if self.robot.hand_spec.name == "brainco":
            return np.asarray(self.robot.hand_spec.init_qpos, dtype=np.float64)  # (6,), rad
        return self._fit_vector(values, self.robot.hand_spec.dof)

    # ------------------------------------------------------------------
    # Observation / goal building
    # ------------------------------------------------------------------

    def _build_wbc_observation(self, sim_obs: dict) -> dict:
        """Convert HUMANO observation dict to decoupled_wbc observation format.

        """
        rm = self._dwbc_robot_model
        obs = {}

        left_hand_q = self._sim_hand_values(sim_obs, "left", "q", self._dwbc_left_hand_dof)
        right_hand_q = self._sim_hand_values(sim_obs, "right", "q", self._dwbc_right_hand_dof)
        left_hand_dq = self._sim_hand_values(sim_obs, "left", "dq", self._dwbc_left_hand_dof)
        right_hand_dq = self._sim_hand_values(sim_obs, "right", "dq", self._dwbc_right_hand_dof)

        obs["q"] = rm.get_configuration_from_actuated_joints(
            body_actuated_joint_values=sim_obs["body_q"],
            left_hand_actuated_joint_values=left_hand_q,
            right_hand_actuated_joint_values=right_hand_q,
        )
        obs["dq"] = rm.get_configuration_from_actuated_joints(
            body_actuated_joint_values=sim_obs["body_dq"],
            left_hand_actuated_joint_values=left_hand_dq,
            right_hand_actuated_joint_values=right_hand_dq,
        )
        obs["ddq"] = rm.get_configuration_from_actuated_joints(
            body_actuated_joint_values=sim_obs.get("body_ddq", np.zeros(29)),
            left_hand_actuated_joint_values=self._sim_hand_values(
                sim_obs, "left", "ddq", self._dwbc_left_hand_dof
            ),
            right_hand_actuated_joint_values=self._sim_hand_values(
                sim_obs, "right", "ddq", self._dwbc_right_hand_dof
            ),
        )
        obs["tau_est"] = rm.get_configuration_from_actuated_joints(
            body_actuated_joint_values=sim_obs.get("body_tau_est", np.zeros(29)),
            left_hand_actuated_joint_values=self._sim_hand_values(
                sim_obs, "left", "tau_est", self._dwbc_left_hand_dof
            ),
            right_hand_actuated_joint_values=self._sim_hand_values(
                sim_obs, "right", "tau_est", self._dwbc_right_hand_dof
            ),
        )

        base_pose = np.asarray(sim_obs["floating_base_pose"], dtype=np.float64).copy()
        if self._preserve_spawn_heading:
            quat_wxyz = base_pose[3:7]
            rotation = Rotation.from_quat(quat_wxyz[[1, 2, 3, 0]])
            if self._spawn_heading_yaw is None:
                self._spawn_heading_yaw = float(rotation.as_euler("xyz")[2])
            relative_quat = (
                Rotation.from_euler("z", -self._spawn_heading_yaw) * rotation
            ).as_quat()
            base_pose[3:7] = relative_quat[[3, 0, 1, 2]]
        obs["floating_base_pose"] = base_pose
        obs["floating_base_vel"] = sim_obs["floating_base_vel"]
        obs["floating_base_acc"] = sim_obs.get("floating_base_acc", np.zeros(6))

        obs["torso_quat"] = sim_obs.get("secondary_imu_quat", np.array([1, 0, 0, 0]))
        obs["torso_ang_vel"] = (
            sim_obs.get("secondary_imu_vel", np.zeros(6))[3:6]
            if "secondary_imu_vel" in sim_obs
            else np.zeros(3)
        )

        obs["wrist_pose"] = sim_obs.get("wrist_pose", np.zeros(14))
        return obs

    def get_stabilize_action(self, observation) -> ActionCmd:
        """Run the WBC pipeline ramping to the WBC default pose — used during stabilization.

        Drives the upper body toward the WBC default configuration (forearms pointing
        forward), matching what teleop does before the operator activates the teleop policy.
        Does NOT read from episode data or advance _data_row_index.
        """
        # Always run the full WBC pipeline (main loop now at 50Hz)
        from decoupled_wbc.control.main.constants import (
            DEFAULT_BASE_HEIGHT,
            DEFAULT_NAV_CMD,
        )

        t_now = self._time_fn()
        control_freq = self._control_frequency
        proprio = self.robot.prepare_obs()
        wbc_obs = self._build_wbc_observation(proprio)
        self._wbc_policy.set_observation(wbc_obs)

        # Use the WBC's own default upper body pose (forearms forward), not the
        # MJCF keyframe defaults (arms down).  On the first step give a 2s ramp
        # window — matching teleop's initial `target_time = t_now + 2.0` — so the
        # arms move smoothly to the default rather than snapping.
        is_first_step = self._cached_target_q is None
        default_upper_body = self._dwbc_robot_model.get_initial_upper_body_pose()
        goal = {
            "target_upper_body_pose": default_upper_body,
            "navigate_cmd": np.asarray(DEFAULT_NAV_CMD),
            "base_height_command": np.atleast_1d(np.asarray(DEFAULT_BASE_HEIGHT)),
            "target_time": t_now + (2.0 if is_first_step else 1 / control_freq),
            "interpolation_garbage_collection_time": t_now - 2 / control_freq,
            "timestamp": t_now,
        }
        self._wbc_policy.set_goal(goal)

        wbc_action = self._wbc_policy.get_action(time=t_now)
        self._cached_target_q = self._dwbc_robot_model.get_body_actuated_joints(wbc_action["q"])
        self._cached_left_hand_q = self._dwbc_robot_model.get_hand_actuated_joints(
            wbc_action["q"], side="left"
        )
        self._cached_right_hand_q = self._dwbc_robot_model.get_hand_actuated_joints(
            wbc_action["q"], side="right"
        )

        return ActionCmd(
            "decoupled_wbc",
            target_q=self._cached_target_q,
            left_hand_q=self._robot_hand_action(self._cached_left_hand_q, "left"),
            right_hand_q=self._robot_hand_action(self._cached_right_hand_q, "right"),
        )

    def queue_action(self, action_cmd: ActionCmd):
        self._action_queue.append(action_cmd)

    def get_action(self, observation, instruction=None, **kwargs):
        try:
            if observation is not None and "agent" in observation:
                self._last_qpos = dict(
                    zip(self.robot.joint_names, observation["agent"], strict=False)
                )
            action = self._action_queue.popleft()
            self._last_pred_action = action
            return action
        except IndexError:
            raise StopIteration("No more queued actions in PrimitiveAgent.") from None

    def __len__(self):
        return len(self._action_queue)

    def reset(self, **kwargs):
        self._action_queue = deque()
        return super().reset(**kwargs)
