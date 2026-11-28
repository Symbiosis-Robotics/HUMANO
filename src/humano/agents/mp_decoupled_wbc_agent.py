"""Motion-planning to decoupled-WBC adapter for G1Sonic datagen."""

from __future__ import annotations

import time
from copy import deepcopy
from typing import Any

import numpy as np
from decoupled_wbc.control.main.constants import DEFAULT_BASE_HEIGHT, DEFAULT_NAV_CMD

from humano.core.action import ActionCmd
from humano.robots.g1_sonic import LEFT_HAND_JOINTS, RIGHT_HAND_JOINTS

from .mp import MotionPlannerAgent
from .sonic_decoupled_wbc_agent import SonicDecoupledWbcAgent

_SOURCE_ACTION_IDS = {
    "unknown": 0,
    "move_qpos": 1,
    "move_qpos_with_eef": 2,
    "loco_command": 3,
    "open_eef": 4,
    "close_eef": 5,
}


def _as_finite_vector(value: Any, size: int, name: str) -> np.ndarray:
    vec = np.asarray(value, dtype=np.float32).reshape(-1)
    if vec.size != size or not np.isfinite(vec).all():
        raise ValueError(f"{name} must be a finite {size}D vector, got shape={vec.shape}")
    return vec


class MPDecoupledWbcAgent(SonicDecoupledWbcAgent):
    """Execute MP references through decoupled WBC at the policy control rate.

    Dataset labels are intentionally separated from the controller's internal
    upper-body vector. VLA predicts the two arm targets and hand targets while
    GEAR-WBC owns the waist and legs.
    """

    def __init__(
        self,
        robot,
        mp_agent: MotionPlannerAgent,
        sonic_config: dict,
        *,
        base_height_is_delta: bool = True,
        base_height_offset: float = 0.0,
        base_height_override: float | None = None,
        enable_lower_body_policy_action: bool = True,
        locomotion_arm_pose: str = "current",
        nav_vx_sign: float = 1.0,
        nav_vy_sign: float = 1.0,
        debug: bool = False,
    ):
        super().__init__(robot, sonic_config)
        self.mp_agent = mp_agent
        self.base_height_is_delta = bool(base_height_is_delta)
        self.base_height_offset = float(base_height_offset)
        self.base_height_override = None if base_height_override is None else float(base_height_override)
        self.locomotion_arm_pose = str(locomotion_arm_pose)
        self.nav_vx_sign = float(nav_vx_sign)
        self.nav_vy_sign = float(nav_vy_sign)
        self.debug = bool(debug)

        if self.nav_vx_sign not in {-1.0, 1.0} or self.nav_vy_sign not in {-1.0, 1.0}:
            raise ValueError("nav_vx_sign and nav_vy_sign must each be -1 or 1")
        if not np.isfinite(self.base_height_offset):
            raise ValueError("base_height_offset must be finite")
        if self.base_height_override is not None and not np.isfinite(self.base_height_override):
            raise ValueError("base_height_override must be finite when provided")
        if self.locomotion_arm_pose not in {"current", "wbc_initial", "robot_stand"}:
            raise ValueError(
                "locomotion_arm_pose must be one of: current, wbc_initial, robot_stand"
            )

        self.arm_joint_names = self._joint_names_for_group("arms")
        self.sonic_upper_joint_names = self._joint_names_for_group("upper_body")
        if len(self.arm_joint_names) != 14:
            raise RuntimeError(f"Expected 14 arm joints, got {self.arm_joint_names}")
        if any(name.startswith("waist_") for name in self.sonic_upper_joint_names):
            raise RuntimeError(
                "datagen requires waist ownership by GEAR-WBC; "
                "construct the agent with sonic_config['enable_waist']=False"
            )

        self._left_hand_index = {name: index for index, name in enumerate(LEFT_HAND_JOINTS)}
        self._right_hand_index = {name: index for index, name in enumerate(RIGHT_HAND_JOINTS)}
        missing_hands = set(LEFT_HAND_JOINTS + RIGHT_HAND_JOINTS) - set(self.sonic_upper_joint_names)
        if missing_hands:
            raise RuntimeError(f"WBC upper-body group is missing hand joints: {sorted(missing_hands)}")

        self._physics_dt = float(self.sim_dt)
        if self._physics_dt <= 0.0:
            raise ValueError(f"SIMULATE_DT must be positive, got {self._physics_dt}")
        self._wbc_interval_steps = max(1, int(round(1.0 / (self._control_frequency * self._physics_dt))))
        if not np.isclose(self._wbc_interval_steps * self._physics_dt, 1.0 / self._control_frequency):
            raise ValueError(
                "GEAR policy rate must be an integer multiple of MuJoCo physics dt: "
                f"physics_dt={self._physics_dt}, control_frequency={self._control_frequency}"
            )

        self._default_base_height = float(DEFAULT_BASE_HEIGHT)
        self._initial_upper_body_pose = self._initial_upper_body_pose_by_name()
        self._last_nav_cmd = np.asarray(DEFAULT_NAV_CMD, dtype=np.float32).copy()
        self._last_base_height = np.asarray([self._default_base_height], dtype=np.float32)
        self._target_qpos = self._read_robot_qpos()
        self._left_hand_q = self._hand_from_qpos(LEFT_HAND_JOINTS, self._target_qpos)
        self._right_hand_q = self._hand_from_qpos(RIGHT_HAND_JOINTS, self._target_qpos)

        self.last_source_action: ActionCmd | None = None
        self.last_pre_wbc_action: dict[str, Any] | None = None
        self.last_post_wbc_action: dict[str, Any] | None = None

        self._phase_index = 0
        self._source_remaining_s = 0.0
        self._wbc_step_index = 0
        self._wbc_inference_count = 0
        self._cached_wbc_action: ActionCmd | None = None

        lower_body_policy = getattr(self._wbc_policy, "lower_body_policy", None)
        if lower_body_policy is not None and hasattr(lower_body_policy, "use_policy_action"):
            lower_body_policy.use_policy_action = bool(enable_lower_body_policy_action)

    def _joint_names_for_group(self, group: str) -> list[str]:
        group_indices = set(self._dwbc_robot_model.get_joint_group_indices(group))
        names_and_indices = [
            (index, name)
            for name, index in self._dwbc_robot_model.joint_to_dof_index.items()
            if index in group_indices
        ]
        return [name for _index, name in sorted(names_and_indices)]

    def _initial_upper_body_pose_by_name(self) -> dict[str, float]:
        values = np.asarray(
            self._dwbc_robot_model.get_initial_upper_body_pose(), dtype=np.float32
        ).reshape(-1)
        if values.size != len(self.sonic_upper_joint_names):
            raise RuntimeError(
                "WBC initial upper-body pose length does not match upper-body joint group"
            )
        return {
            name: float(value)
            for name, value in zip(self.sonic_upper_joint_names, values, strict=True)
        }

    @property
    def source_action_ids(self) -> dict[str, int]:
        return dict(_SOURCE_ACTION_IDS)

    @property
    def wbc_interval_steps(self) -> int:
        return self._wbc_interval_steps

    @property
    def physics_dt(self) -> float:
        return self._physics_dt

    @property
    def wbc_inference_count(self) -> int:
        return self._wbc_inference_count

    def set_phase_index(self, phase_index: int) -> None:
        if phase_index < 0:
            raise ValueError("phase_index must be non-negative")
        self._phase_index = int(phase_index)

    def _simulation_time(self) -> float:
        mj_data = getattr(self.robot, "mjData", None)
        if mj_data is not None:
            return float(mj_data.time)
        return time.monotonic()

    def _read_robot_qpos(self) -> dict[str, float]:
        try:
            qpos = self.robot.get_robot_qpos()
            return {name: float(qpos.get(name, 0.0)) for name in self.robot.joint_names}
        except Exception:
            return {name: float(value) for name, value in self.robot.init_joint_states.items()}

    @staticmethod
    def _hand_from_qpos(names: list[str], qpos: dict[str, float]) -> np.ndarray:
        return np.asarray([qpos.get(name, 0.0) for name in names], dtype=np.float32)

    def _target_arm_pose(self) -> dict[str, float]:
        return {name: float(self._target_qpos.get(name, 0.0)) for name in self.arm_joint_names}

    def _controller_upper_body_target(self, target_arm_pose: dict[str, float]) -> dict[str, float]:
        target: dict[str, float] = {}
        for name in self.sonic_upper_joint_names:
            if name in self._left_hand_index:
                target[name] = float(self._left_hand_q[self._left_hand_index[name]])
            elif name in self._right_hand_index:
                target[name] = float(self._right_hand_q[self._right_hand_index[name]])
            else:
                target[name] = float(target_arm_pose[name])
        return target

    def _side_from_hand_uid(self, hand_uid: str | None) -> str | None:
        if hand_uid is None:
            return None
        if "left" in hand_uid:
            return "left"
        if "right" in hand_uid:
            return "right"
        return None

    def _open_hand_target(self, side: str) -> np.ndarray:
        names = LEFT_HAND_JOINTS if side == "left" else RIGHT_HAND_JOINTS
        return np.asarray([self.robot.init_joint_states.get(name, 0.0) for name in names], dtype=np.float32)

    def _close_hand_target(self, side: str) -> np.ndarray:
        eef_cfg = getattr(self.robot.controller_cfg, f"{side}_eef", None)
        close_qpos = getattr(eef_cfg, "close_qpos", None)
        if close_qpos is None:
            return self._open_hand_target(side)
        return _as_finite_vector(close_qpos, 7, f"{side}_close_qpos")

    def _target_hand_is_open_or_missing(
        self, side: str, target_qpos: dict[str, float] | None
    ) -> bool:
        if target_qpos is None:
            return True
        names = LEFT_HAND_JOINTS if side == "left" else RIGHT_HAND_JOINTS
        open_target = self._open_hand_target(side)
        values = []
        open_values = []
        for index, name in enumerate(names):
            if name not in target_qpos:
                continue
            try:
                values.append(float(target_qpos[name]))
            except (TypeError, ValueError):
                raise ValueError(f"invalid qpos target for {name}: {target_qpos[name]!r}") from None
            open_values.append(float(open_target[index]))
        if not values:
            return True
        return bool(np.allclose(values, open_values, atol=1e-4))

    def _apply_hand_state(
        self,
        eef_state: str | None,
        hand_uid: str | None,
        *,
        previous_hand_q: np.ndarray | None = None,
    ) -> None:
        if eef_state not in {"open_eef", "close_eef"}:
            return
        side = self._side_from_hand_uid(hand_uid) or "right"
        if (
            eef_state == "close_eef"
            and previous_hand_q is not None
            and not np.allclose(previous_hand_q, self._open_hand_target(side), atol=1e-4)
        ):
            target = np.asarray(previous_hand_q, dtype=np.float32).copy()
        else:
            target = self._close_hand_target(side) if eef_state == "close_eef" else self._open_hand_target(side)
        if side == "left":
            self._left_hand_q = target
        else:
            self._right_hand_q = target

    def _update_qpos_target(self, target_qpos: dict[str, float] | None) -> None:
        if target_qpos is None:
            return
        for name, value in target_qpos.items():
            try:
                self._target_qpos[name] = float(value)
            except (TypeError, ValueError):
                raise ValueError(f"invalid qpos target for {name}: {value!r}") from None
        if any(name in target_qpos for name in LEFT_HAND_JOINTS):
            self._left_hand_q = self._hand_from_qpos(LEFT_HAND_JOINTS, self._target_qpos)
        if any(name in target_qpos for name in RIGHT_HAND_JOINTS):
            self._right_hand_q = self._hand_from_qpos(RIGHT_HAND_JOINTS, self._target_qpos)

    def _apply_locomotion_arm_pose(self) -> None:
        if self.locomotion_arm_pose == "current":
            return
        if self.locomotion_arm_pose == "wbc_initial":
            source = self._initial_upper_body_pose
        elif self.locomotion_arm_pose == "robot_stand":
            source = self.robot.init_joint_states
        else:
            raise RuntimeError(f"Unsupported locomotion arm pose: {self.locomotion_arm_pose}")
        for name in self.arm_joint_names:
            if name in source:
                self._target_qpos[name] = float(source[name])

    def _nav_from_loco_command(self, command: Any) -> tuple[np.ndarray, np.ndarray]:
        cmd = np.asarray(command, dtype=np.float32).reshape(-1)
        if cmd.size < 4 or not np.isfinite(cmd).all():
            raise ValueError(f"loco_command must contain at least 4 finite values, got {cmd}")

        turn_flag = float(cmd[7]) if cmd.size > 7 else 0.0
        nav_cmd = np.asarray(
            [self.nav_vx_sign * cmd[0], self.nav_vy_sign * cmd[2], turn_flag, cmd[1]],
            dtype=np.float32,
        )
        height = float(cmd[3])
        if self.base_height_is_delta:
            base = (
                self._default_base_height
                if self.base_height_override is None
                else self.base_height_override
            )
            height = base + height
        elif self.base_height_override is not None:
            height = self.base_height_override
        height += self.base_height_offset
        return nav_cmd, np.asarray([height], dtype=np.float32)

    def _source_duration_s(self, source_action: ActionCmd) -> float:
        try:
            duration_s = float(source_action["source_dt"])
        except (TypeError, ValueError):
            duration_s = self._physics_dt
        if not np.isfinite(duration_s) or duration_s <= 0.0:
            return self._physics_dt
        return duration_s

    def _source_to_pre_wbc(self, source_action: ActionCmd) -> dict[str, Any]:
        action_type = source_action.type
        source_id = _SOURCE_ACTION_IDS.get(action_type, _SOURCE_ACTION_IDS["unknown"])

        if action_type == "loco_command":
            self._apply_locomotion_arm_pose()
            self._last_nav_cmd, self._last_base_height = self._nav_from_loco_command(source_action["command"])
        elif action_type in {"move_qpos", "move_qpos_with_eef", "open_eef", "close_eef"}:
            hand_uid = source_action["hand_uid"]
            eef_state = action_type if action_type in {"open_eef", "close_eef"} else source_action["eef_state"]
            side = self._side_from_hand_uid(hand_uid) or "right"
            previous_left_hand_q = self._left_hand_q.copy()
            previous_right_hand_q = self._right_hand_q.copy()
            preserve_move_left_hand = (
                action_type == "move_qpos_with_eef"
                and eef_state != "open_eef"
                and self._target_hand_is_open_or_missing("left", source_action["target_qpos"])
                and not np.allclose(previous_left_hand_q, self._open_hand_target("left"), atol=1e-4)
            )
            preserve_move_right_hand = (
                action_type == "move_qpos_with_eef"
                and eef_state != "open_eef"
                and self._target_hand_is_open_or_missing("right", source_action["target_qpos"])
                and not np.allclose(previous_right_hand_q, self._open_hand_target("right"), atol=1e-4)
            )
            previous_side_hand_q = previous_left_hand_q if side == "left" else previous_right_hand_q
            previous_hand_q = (
                previous_side_hand_q
                if eef_state == "close_eef"
                and not np.allclose(previous_side_hand_q, self._open_hand_target(side), atol=1e-4)
                else None
            )

            self._update_qpos_target(source_action["target_qpos"])
            if preserve_move_left_hand:
                self._left_hand_q = previous_left_hand_q
            if preserve_move_right_hand:
                self._right_hand_q = previous_right_hand_q
            self._apply_hand_state(eef_state, hand_uid, previous_hand_q=previous_hand_q)
        else:
            raise ValueError(f"Unsupported MP action type for decoupled WBC: {action_type}")

        target_arm_pose = self._target_arm_pose()
        controller_upper_body_pose = self._controller_upper_body_target(target_arm_pose)
        left_hand_q = _as_finite_vector(self._left_hand_q, 7, "left_hand_q")
        right_hand_q = _as_finite_vector(self._right_hand_q, 7, "right_hand_q")
        nav_cmd = _as_finite_vector(self._last_nav_cmd, 4, "navigate_cmd")
        base_height = _as_finite_vector(self._last_base_height, 1, "base_height_command")

        return {
            "source_action_type": action_type,
            "source_action_type_id": np.asarray([source_id], dtype=np.float32),
            "source_dt": np.asarray([self._source_duration_s(source_action)], dtype=np.float32),
            "phase_index": np.asarray([self._phase_index], dtype=np.float32),
            "target_arm_pose": target_arm_pose,
            "target_arm_pose_vector": np.asarray(
                [target_arm_pose[name] for name in self.arm_joint_names], dtype=np.float32
            ),
            "controller_upper_body_pose_vector": np.asarray(
                [controller_upper_body_pose[name] for name in self.sonic_upper_joint_names], dtype=np.float32
            ),
            "navigate_cmd": nav_cmd,
            "base_height_command": base_height,
            "left_hand_q": left_hand_q,
            "right_hand_q": right_hand_q,
            "hand_uid": source_action["hand_uid"],
            "eef_state": source_action["eef_state"] or action_type,
            "raw_parameters": deepcopy(source_action.parameters),
        }

    def _run_wbc(self, pre_wbc_action: dict[str, Any]) -> ActionCmd:
        proprio = self.robot.prepare_obs()
        wbc_obs = self._build_wbc_observation(proprio)
        self._wbc_policy.set_observation(wbc_obs)

        t_now = self._simulation_time()
        goal = {
            "target_upper_body_pose": pre_wbc_action["controller_upper_body_pose_vector"],
            "navigate_cmd": pre_wbc_action["navigate_cmd"],
            "base_height_command": pre_wbc_action["base_height_command"],
            "target_time": t_now + 1.0 / self._control_frequency,
            "interpolation_garbage_collection_time": t_now - 2.0 / self._control_frequency,
            "timestamp": t_now,
        }
        self._wbc_policy.set_goal(goal)
        wbc_action = self._wbc_policy.get_action(time=t_now)

        target_q = self._dwbc_robot_model.get_body_actuated_joints(wbc_action["q"])
        self.last_post_wbc_action = {
            "target_q": np.asarray(target_q, dtype=np.float32),
            "left_hand_q": self._robot_hand_action(pre_wbc_action["left_hand_q"], "left"),
            "right_hand_q": self._robot_hand_action(pre_wbc_action["right_hand_q"], "right"),
            "wbc_q": np.asarray(wbc_action["q"], dtype=np.float32),
            "wbc_navigate_cmd": np.asarray(wbc_action.get("navigate_cmd", []), dtype=np.float32),
            "wbc_base_height_command": np.asarray(
                wbc_action.get("base_height_command", []), dtype=np.float32
            ),
            "wbc_sim_time": np.asarray([t_now], dtype=np.float32),
        }
        return ActionCmd(
            "decoupled_wbc",
            target_q=self.last_post_wbc_action["target_q"],
            left_hand_q=self.last_post_wbc_action["left_hand_q"],
            right_hand_q=self.last_post_wbc_action["right_hand_q"],
            navigate_cmd=pre_wbc_action["navigate_cmd"],
            base_height_command=pre_wbc_action["base_height_command"],
            source_action_type=pre_wbc_action["source_action_type"],
            source_action_type_id=pre_wbc_action["source_action_type_id"],
            phase_index=pre_wbc_action["phase_index"],
            source_dt=pre_wbc_action["source_dt"],
        )

    def _next_wbc_action(self) -> ActionCmd:
        if self.last_pre_wbc_action is None:
            raise RuntimeError("No pre-WBC action is available")
        if self._cached_wbc_action is None or self._wbc_step_index % self._wbc_interval_steps == 0:
            self._cached_wbc_action = self._run_wbc(self.last_pre_wbc_action)
            self._wbc_inference_count += 1
        self._wbc_step_index += 1
        action = self._cached_wbc_action.copy()
        self._last_pred_action = action
        return action

    def _consume_next_source_action(self, observation, instruction, info, **kwargs) -> None:
        self.last_source_action = self.mp_agent.get_action(
            observation, instruction=instruction, info=info, **kwargs
        )
        self.last_pre_wbc_action = self._source_to_pre_wbc(self.last_source_action)
        self._source_remaining_s = self._source_duration_s(self.last_source_action)
        if self.debug:
            print(
                "[MPDecoupledWbcAgent] "
                f"phase={self._phase_index} {self.last_source_action.type} "
                f"dt={self._source_remaining_s:.3f}s "
                f"nav={self.last_pre_wbc_action['navigate_cmd'].tolist()}"
            )

    def get_action(self, observation, instruction=None, info=None, **kwargs):
        if self.last_source_action is None:
            self._consume_next_source_action(observation, instruction, info, **kwargs)
        else:
            self._source_remaining_s -= self._physics_dt
            if self._source_remaining_s <= 1e-9:
                self._consume_next_source_action(observation, instruction, info, **kwargs)
        return self._next_wbc_action()

    def get_stabilize_action(self, observation) -> ActionCmd:
        if self.last_pre_wbc_action is None:
            stand_action = ActionCmd("loco_command", command=[0.0] * 8, source_dt=self._physics_dt)
            self.last_source_action = stand_action
            self.last_pre_wbc_action = self._source_to_pre_wbc(stand_action)
        return self._next_wbc_action()

    def set_manual_locomotion_command(
        self,
        *,
        vx: float,
        vy: float = 0.0,
        target_yaw: float = 0.0,
        height: float = 0.0,
        turn_flag: float = 0.0,
    ) -> None:
        command = [float(vx), float(target_yaw), float(vy), float(height), 0.0, 0.0, 0.0, float(turn_flag)]
        source_action = ActionCmd("loco_command", command=command, source_dt=self._physics_dt)
        self.last_source_action = source_action
        self.last_pre_wbc_action = self._source_to_pre_wbc(source_action)

    def get_manual_locomotion_action(self) -> ActionCmd:
        return self._next_wbc_action()

    def reset(self, **kwargs):
        super().reset(**kwargs)
        self.mp_agent.reset()
        self._target_qpos = self._read_robot_qpos()
        self._left_hand_q = self._hand_from_qpos(LEFT_HAND_JOINTS, self._target_qpos)
        self._right_hand_q = self._hand_from_qpos(RIGHT_HAND_JOINTS, self._target_qpos)
        self._last_nav_cmd = np.asarray(DEFAULT_NAV_CMD, dtype=np.float32).copy()
        self._last_base_height = np.asarray([self._default_base_height], dtype=np.float32)
        self.last_source_action = None
        self.last_pre_wbc_action = None
        self.last_post_wbc_action = None
        self._phase_index = 0
        self._source_remaining_s = 0.0
        self._wbc_step_index = 0
        self._wbc_inference_count = 0
        self._cached_wbc_action = None
        self._wbc_policy.reset(init_time=self._simulation_time())
