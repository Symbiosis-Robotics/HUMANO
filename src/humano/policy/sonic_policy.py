from __future__ import annotations

import json
import struct
import threading
import time
from collections import deque
from collections.abc import Callable
from enum import Enum
from typing import Any

import numpy as np
import zmq
from gear_sonic.utils.mujoco_sim.unitree_sdk2py_bridge import UnitreeSdk2Bridge
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__HandCmd_ as HandCmd_default

from humano.core.action import ActionCmd
from humano.policy.client import HttpActionClient
from humano.robots.hands import brainco_command_gains
from humano.wbc.hand_control import (
    DEFAULT_HAND_KD,
    DEFAULT_HAND_KP,
    build_hand_cmd,
    dds_hand_feedback,
)
from humano.wbc.base import MotorCommand, RobotState
from humano.wbc.sonic.policy_parameters import (
    SONIC_STAND_PD_KD,
    SONIC_STAND_PD_KP,
)

HEADER_SIZE = 1280
LEFT_LEG_JOINTS = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
]
RIGHT_LEG_JOINTS = [
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
]
WAIST_JOINTS = ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
LEFT_ARM_JOINTS = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
]
RIGHT_ARM_JOINTS = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]


class ActionType(str, Enum):
    """Action type dispatched by get_action() template method."""

    POLICY = "policy"
    STABILIZE = "stabilize"
    BOOTSTRAP_STABILIZE = "bootstrap_stabilize"
    PREFETCH_WARMUP = "prefetch_warmup"


def _pack_message(topic: str, version: int, fields: list[dict[str, Any]], payload: bytes) -> bytes:
    header = {"v": version, "endian": "le", "count": 1, "fields": fields}
    header_bytes = json.dumps(header).encode("utf-8")
    if len(header_bytes) > HEADER_SIZE:
        raise ValueError(f"ZMQ header is too large: {len(header_bytes)} > {HEADER_SIZE}")
    return (
        topic.encode("utf-8")
        + header_bytes
        + (b"\x00" * (HEADER_SIZE - len(header_bytes)))
        + payload
    )


def _projected_gravity(root_quat_wxyz: np.ndarray) -> np.ndarray:
    q = np.asarray(root_quat_wxyz, dtype=np.float64)
    if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-8:
        return np.array([0.0, 0.0, -1.0], dtype=np.float32)
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    rot = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    return (rot.T @ np.array([0.0, 0.0, -1.0], dtype=np.float64)).astype(np.float32)


def _policy_state(robot, observation: dict[str, Any]) -> dict[str, np.ndarray]:
    spec = robot.hand_spec
    qpos = np.asarray(observation["joint_qpos"], dtype=np.float32).reshape(-1)
    by_name = dict(zip(robot.joint_names, qpos, strict=True))
    left_names, right_names = spec.left_actuated_joints, spec.right_actuated_joints
    if spec.name == "dex3":
        # Preserve the Dex3 dataset's coupled left index/middle feedback.
        by_name[left_names[5]], by_name[left_names[6]] = (
            by_name[left_names[3]],
            by_name[left_names[4]],
        )
    if spec.qpos_reorder_from_gr00t is not None:
        left_names = tuple(left_names[i] for i in spec.qpos_reorder_from_gr00t)
        right_names = tuple(right_names[i] for i in spec.qpos_reorder_from_gr00t)
    groups = {
        "left_leg": LEFT_LEG_JOINTS,
        "right_leg": RIGHT_LEG_JOINTS,
        "waist": WAIST_JOINTS,
        "left_arm": LEFT_ARM_JOINTS,
        "left_hand": left_names,
        "right_arm": RIGHT_ARM_JOINTS,
        "right_hand": right_names,
    }
    state = {
        key: np.asarray([by_name[name] for name in names], dtype=np.float32)
        for key, names in groups.items()
    }
    state["projected_gravity"] = _projected_gravity(robot.mjData.qpos[3:7])
    return state


def _validate_policy_actions(actions: np.ndarray, spec) -> None:
    expected = 64 + 2 * spec.dof
    if actions.shape[-1] != expected:
        raise ValueError(
            f"{spec.name} SONIC action must be {expected}D "
            f"[motion_token(64), left_hand_joints({spec.dof}), right_hand_joints({spec.dof})], "
            f"got {actions.shape}"
        )


def _hand_gains(spec) -> tuple[float, float]:
    if spec.kp_map is None:
        return DEFAULT_HAND_KP, DEFAULT_HAND_KD
    return next(iter(spec.kp_map.values())), spec.kd


def _brainco_command_gains(command, spec, *, kp_override=None, kd_override=None):
    if spec.name == "brainco":
        for motor, (kp, kd) in zip(
            command.motor_cmd[: spec.dof], brainco_command_gains(), strict=True
        ):
            motor.kp = kp if kp_override is None else kp_override
            motor.kd = kd if kd_override is None else kd_override
    return command


class SonicPolicyDeployAgent:
    """Bridge standard SONIC-token HTTP actions into the SONIC deploy ZMQ token stream."""

    def __init__(
        self,
        robot,
        host: str,
        port: int,
        *,
        sonic_config: dict[str, Any],
        zmq_bind_host: str = "*",
        zmq_port: int | None = None,
        pose_topic: str = "pose",
        command_topic: str = "command",
        query_dataset: str = "humano",
        action_exec_horizon: int | None = None,
        **_: Any,
    ):
        # ========== 1. 基础对象和 HTTP policy client ==========
        self.robot = robot
        self.client = HttpActionClient(host, port)  # 用 HTTP 请求 GR00T policy server
        self.sonic_config = sonic_config

        # ZMQ topic / 查询配置
        self.pose_topic = pose_topic
        self.command_topic = command_topic
        self.query_dataset = query_dataset
        self.action_exec_horizon = action_exec_horizon

        # ========== 2. Unitree SDK bridge ==========
        # 把 SONIC/GR00T 的中间表示转换到 Unitree SDK2 相关控制格式
        self.unitree_bridge = UnitreeSdk2Bridge(sonic_config)

        # ========== 3. 动作队列与异步 query 状态 ==========
        # GR00T 一次可能返回一段 action chunk，
        # 这里用队列缓存，控制循环每次取一小段执行。
        self.action_queue: deque[np.ndarray] = deque()

        # 多线程访问 action_queue 时用锁保护
        self._action_lock = threading.Lock()

        # 异步查询 GR00T policy 的线程
        self._async_query_thread: threading.Thread | None = None

        # 如果异步线程里报错，存在这里，主线程后面可以检查
        self._async_query_error: BaseException | None = None
        self._session_generation = 0
        self._closed = False

        # 是否重置 GR00T 历史上下文，比如 episode 刚开始时要 True
        self._reset_history = True

        # 当前控制/观测帧编号
        self._frame_index = 0
        self._low_state_publish_count = 0
        self._last_low_state_publish_time: float | None = None

        # 上一次发给 SONIC 的 token，用于平滑、限幅、debug
        self._last_token: np.ndarray | None = None

        # 上一次左右手命令，用于保持/平滑手部状态
        self._last_left_hand: np.ndarray | None = None
        self._last_right_hand: np.ndarray | None = None

        # 直接构造左右手命令的默认结构
        self._direct_left_hand_cmd = self._make_hand_cmd()
        self._direct_right_hand_cmd = self._make_hand_cmd()

        # ========== 4. 启动状态 / low-level command 状态 ==========
        self._started = False  # Agent 是否已经启动
        self._last_low_cmd_received = False  # 是否收到过低层控制命令
        # debug 打印计数，防止疯狂刷屏
        self._debug_low_cmd_prints = 0
        self._debug_token_prints = 0
        self._debug_state_printed = False
        self._last_token_source: str | None = None
        self._token_source_counts: dict[str, int] = {}
        self._last_fallback_action: str | None = None

        # ========== 5. 仿真步长和控制步长 ==========
        # 仿真 dt，比如 mujoco / isaac 的 simulation timestep
        self.sim_dt = float(sonic_config["SIMULATE_DT"])

        # SONIC stays at 50 Hz when hand contacts require smaller physics steps.
        self.control_dt = float(sonic_config.get("CONTROL_DT", 1.0 / 50.0))

        # ========== 6. 最小 SONIC stream 状态 ==========
        # 这里只保留核心链路：发 token stream，并确认 SONIC 回过 LowCmd。
        self._stream_start_time: float | None = None
        self._last_stream_command_time = 0.0
        self._stream_command_repeat_s = float(sonic_config.get("STREAM_COMMAND_REPEAT_S", 2.0))
        self._stream_command_heartbeat_s = float(
            sonic_config.get("STREAM_COMMAND_HEARTBEAT_S", 0.25)
        )
        self._stream_token_count = 0
        self._first_low_cmd_time: float | None = None
        self._sonic_ready_reported = False
        self._last_waiting_ready_log_time = 0.0

        # LowCmd 没回来前不能执行 wbc_torque，只保持当前关节位姿。
        self._ground_hold_target_q: np.ndarray | None = None
        self._ground_hold_logged = False
        self._sonic_stand_hold_logged = False

        # policy 是否已经拿到第一批 GR00T action
        self._policy_ready = False

        # ========== 8. GR00T token 发送频率 ==========
        self._token_target_hz = float(sonic_config.get("GR00T_TOKEN_TARGET_HZ", 30.0))
        self._token_period_s = 1.0 / self._token_target_hz if self._token_target_hz > 0.0 else 0.0
        self._policy_step_index = 0
        self._next_new_token_time_s = 0.0

        # ========== 13. ZMQ publisher 初始化 ==========
        # 创建 ZMQ context，主要用来向SONIC进程发布token
        self.context = zmq.Context()

        # 创建 PUB socket，用于把 pose/command token 发布出去
        self.publisher = self.context.socket(zmq.PUB)

        # 发送高水位线，最多积压 10 条，防止消息堆太多
        self.publisher.setsockopt(zmq.SNDHWM, 10)

        # socket close 时不等待未发送消息，避免退出卡住
        self.publisher.setsockopt(zmq.LINGER, 0)

        # 绑定端口：优先用传入的 zmq_port，否则用 sonic_config["SONIC_ZMQ_PORT"]，默认 5556
        bind_port = int(zmq_port or sonic_config["SONIC_ZMQ_PORT"])

        # 真正启动 ZMQ PUB 服务
        self.publisher.bind(f"tcp://{zmq_bind_host}:{bind_port}")

        # 等一小会儿，让 subscriber 有时间连上
        time.sleep(0.2)

    def reset(self, **_: Any) -> None:
        with self._action_lock:
            self._session_generation += 1
            self.action_queue.clear()
        self._async_query_thread = None
        self._async_query_error = None
        self.unitree_bridge.reset()
        self._reset_history = True
        self._frame_index = 0
        self._low_state_publish_count = 0
        self._last_low_state_publish_time = None
        self._last_token = None
        self._last_left_hand = None
        self._last_right_hand = None
        self._direct_left_hand_cmd = self._make_hand_cmd()
        self._direct_right_hand_cmd = self._make_hand_cmd()
        self._started = False
        self._last_low_cmd_received = False
        self._debug_low_cmd_prints = 0
        self._debug_token_prints = 0
        self._debug_state_printed = False
        self._last_token_source = None
        self._token_source_counts.clear()
        self._last_fallback_action = None
        self._stream_start_time = None
        self._last_stream_command_time = 0.0
        self._stream_token_count = 0
        self._policy_step_index = 0
        self._next_new_token_time_s = 0.0
        self._first_low_cmd_time = None
        self._sonic_ready_reported = False
        self._last_waiting_ready_log_time = 0.0
        self._ground_hold_target_q = None
        self._ground_hold_logged = False
        self._sonic_stand_hold_logged = False
        self._policy_ready = False
        self._send_stream_command(start=False, stop=False, planner=False)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.stop_stream()
        except Exception:
            pass
        query_thread = self._async_query_thread
        if (
            query_thread is not None
            and query_thread.is_alive()
            and query_thread is not threading.current_thread()
        ):
            query_thread.join(timeout=0.2)
        try:
            self.unitree_bridge.close()
        except Exception:
            pass
        self.publisher.close()
        self.context.term()

    def stop_stream(self) -> None:
        try:
            for _ in range(3):
                self._send_stream_command(start=False, stop=True, planner=False)
                time.sleep(0.02)
        except zmq.ZMQError:
            pass

    def _send_stream_command(
        self, *, start: bool, stop: bool = False, planner: bool = False
    ) -> None:
        # 主要是封装start stop和planner的command
        fields = [
            {"name": "start", "dtype": "u8", "shape": [1]},
            {"name": "stop", "dtype": "u8", "shape": [1]},
            {"name": "planner", "dtype": "u8", "shape": [1]},
        ]
        payload = struct.pack("BBB", 1 if start else 0, 1 if stop else 0, 1 if planner else 0)
        self.publisher.send(_pack_message(self.command_topic, 1, fields, payload))

    def _ensure_stream_started(self, low_cmd_received: bool | None = None) -> None:
        now = time.perf_counter()
        if not self._started:
            self._stream_start_time = time.perf_counter()
            self._last_stream_command_time = 0.0

        should_repeat_start = (
            self._stream_start_time is None
            or now - self._stream_start_time <= self._stream_command_repeat_s
        )
        should_heartbeat = (
            self._stream_command_heartbeat_s > 0.0
            and now - self._last_stream_command_time >= self._stream_command_heartbeat_s
        )
        if should_repeat_start or should_heartbeat:
            self._send_stream_command(start=True, stop=False, planner=False)
            self._last_stream_command_time = now
        if not self._started:
            self._started = True
            print("[SonicPolicyAgent] Started SONIC token stream", flush=True)

    def _make_hand_cmd(
        self,
        joints: np.ndarray | None = None,
        *,
        kp: float | None = None,
        kd: float | None = None,
    ):
        kp_override, kd_override = kp, kd
        spec = self.robot.hand_spec
        default_kp, default_kd = _hand_gains(spec)
        kp = default_kp if kp is None else kp
        kd = default_kd if kd is None else kd
        command = build_hand_cmd(
            spec.init_qpos if joints is None else joints,
            hand_dof=spec.dof,
            command_factory=HandCmd_default,
            kp=kp,
            kd=kd,
        )

        return _brainco_command_gains(
            command, spec, kp_override=kp_override, kd_override=kd_override
        )

    def _set_direct_hand_targets(
        self,
        left_hand: np.ndarray | None,
        right_hand: np.ndarray | None,
    ) -> None:
        if left_hand is not None:
            self._direct_left_hand_cmd = self._make_hand_cmd(left_hand)
        if right_hand is not None:
            self._direct_right_hand_cmd = self._make_hand_cmd(right_hand)

    def _limit_token_step(self, token: np.ndarray) -> np.ndarray:
        return np.asarray(token, dtype=np.float32)

    def _quantize_token(self, token: np.ndarray) -> np.ndarray:
        return np.asarray(token, dtype=np.float32)

    def _robot_debug_state(self) -> dict[str, float]:
        state: dict[str, float] = {}
        for name in ("pelvis_z", "pelvis_vz"):
            try:
                value = float(getattr(self.robot, name))
            except Exception:
                continue
            if np.isfinite(value):
                state[name] = value
        try:
            qvel = np.asarray(self.robot.mjData.qvel[:6], dtype=np.float32)
            if qvel.size:
                state["base_qvel_max"] = float(np.max(np.abs(qvel)))
        except Exception:
            pass
        return state

    @staticmethod
    def _format_debug_state(state: dict[str, float]) -> str:
        return " ".join(f"{key}={value:.3f}" for key, value in state.items())

    def _print_fallback_action(
        self,
        fallback_action: str,
        *,
        body_torque: np.ndarray | None = None,
    ) -> None:
        should_print = fallback_action != self._last_fallback_action
        if fallback_action == "wbc_torque" and self._debug_low_cmd_prints < 12:
            should_print = True
        if not should_print:
            return

        robot_state = self._robot_debug_state()
        state_text = self._format_debug_state(robot_state)
        state_text = f" {state_text}" if state_text else ""
        torque_text = ""
        if body_torque is not None:
            torque_text = (
                f" body_torque_norm={np.linalg.norm(body_torque):.3f}"
                f" body_torque_max={np.max(np.abs(body_torque)):.3f}"
            )
        print(
            "[SonicPolicyAgent] Fallback action "
            f"frame={self._frame_index} action={fallback_action}"
            f" low_cmd={self._low_cmd_received()}"
            f"{state_text}"
            f"{torque_text}",
            flush=True,
        )
        self._last_fallback_action = fallback_action

    def _send_token(
        self,
        token: np.ndarray,
        *,
        left_hand: np.ndarray | None = None,
        right_hand: np.ndarray | None = None,
        final_action: np.ndarray | None = None,
        token_source: str = "unknown",
        queue_before: int | None = None,
        queue_after: int | None = None,
        step_time_s: float | None = None,
        token_jump_norm: float | None = None,
        query_latency_s: float | None = None,
    ) -> None:
        send_start = time.perf_counter()
        token = np.asarray(token, dtype=np.float32).reshape(64)
        low_cmd_received = self._low_cmd_received()
        state_start = time.perf_counter()
        robot_state = self._robot_debug_state()
        state_ms = (time.perf_counter() - state_start) * 1000.0
        fallback_action = "wbc_torque" if low_cmd_received else "sonic_stand_pd_hold"
        source_count = self._token_source_counts.get(token_source, 0) + 1
        self._token_source_counts[token_source] = source_count
        should_print = (
            self._debug_token_prints < 16
            or token_source != self._last_token_source
            or (token_source.endswith("empty_queue") and source_count % 25 == 0)
        )
        print_start = time.perf_counter()
        if should_print:
            queue_text = ""
            if queue_before is not None or queue_after is not None:
                queue_text = f" queue={queue_before}->{queue_after}"
            time_text = "" if step_time_s is None else f" step_time={step_time_s:.3f}"
            jump_text = "" if token_jump_norm is None else f" token_jump={token_jump_norm:.4f}"
            latency_text = (
                "" if query_latency_s is None else f" query_latency={query_latency_s:.3f}"
            )
            state_text = self._format_debug_state(robot_state)
            state_text = f" {state_text}" if state_text else ""
            print(
                "[SonicPolicyAgent] Sending SONIC token "
                f"frame={self._frame_index} "
                f"source={token_source}"
                f"{queue_text}"
                f"{time_text}"
                f"{jump_text}"
                f"{latency_text} "
                f"low_cmd={low_cmd_received} fallback={fallback_action}"
                f"{state_text} "
                f"min={token.min():.4f} max={token.max():.4f} "
                f"mean={token.mean():.4f} norm={np.linalg.norm(token):.4f} "
                f"max_abs={np.max(np.abs(token)):.4f} "
                f"head={np.array2string(token[:8], precision=4, separator=',')}",
                flush=True,
            )
            self._debug_token_prints += 1
        print_ms = (time.perf_counter() - print_start) * 1000.0
        self._last_token_source = token_source
        pack_start = time.perf_counter()
        fields = [{"name": "token_state", "dtype": "f32", "shape": [64]}]
        payload = token.tobytes()  # token -> payload
        if left_hand is not None:
            left = np.asarray(left_hand, dtype=np.float32).reshape(self.robot.hand_spec.dof)
            fields.append(
                {"name": "left_hand_joints", "dtype": "f32", "shape": [self.robot.hand_spec.dof]}
            )
            payload += left.tobytes()
        if right_hand is not None:
            right = np.asarray(right_hand, dtype=np.float32).reshape(self.robot.hand_spec.dof)
            fields.append(
                {"name": "right_hand_joints", "dtype": "f32", "shape": [self.robot.hand_spec.dof]}
            )
            payload += right.tobytes()
        fields.append({"name": "frame_index", "dtype": "i64", "shape": [1]})
        payload += np.asarray([self._frame_index - 1], dtype=np.int64).tobytes()
        packed_message = _pack_message(self.pose_topic, 4, fields, payload)
        pack_ms = (time.perf_counter() - pack_start) * 1000.0
        zmq_start = time.perf_counter()
        self.publisher.send(packed_message)  # send token here
        zmq_send_ms = (time.perf_counter() - zmq_start) * 1000.0
        self._stream_token_count += 1
        total_ms = (time.perf_counter() - send_start) * 1000.0
        if total_ms > 20.0 or zmq_send_ms > 10.0:
            print(
                "[SonicPolicyAgent][SEND_TOKEN_TIMING] "
                f"frame={self._frame_index - 1} source={token_source} "
                f"total_ms={total_ms:.3f} state_ms={state_ms:.3f} "
                f"print_ms={print_ms:.3f} pack_ms={pack_ms:.3f} "
                f"zmq_send_ms={zmq_send_ms:.3f}",
                flush=True,
            )

    def _build_state(self, observation: dict[str, Any]) -> dict[str, np.ndarray]:
        state = _policy_state(self.robot, observation)
        if not self._debug_state_printed:
            print(
                "[SonicPolicyAgent] GR00T state built "
                f"left_hand="
                f"{np.array2string(state['left_hand'], precision=4, separator=',')} "
                f"right_hand="
                f"{np.array2string(state['right_hand'], precision=4, separator=',')} "
                f"projected_gravity={np.array2string(state['projected_gravity'], precision=4, separator=',')}",
                flush=True,
            )
            self._debug_state_printed = True
        return state

    def _filter_safe_actions(self, pred_action: np.ndarray) -> np.ndarray:
        return pred_action

    def _query_policy(
        self,
        observation: dict[str, Any],
        instruction: str,
        session_generation: int | None = None,
    ) -> None:
        query_start = time.perf_counter()
        query_frame = int(self._frame_index)
        robot_state = self._robot_debug_state()
        state_text = self._format_debug_state(robot_state)
        state_text = f" {state_text}" if state_text else ""
        print(
            f"[SonicPolicyAgent] Querying /act frame={query_frame}{state_text} "
            f"instruction={instruction!r}",
            flush=True,
        )
        image = {"ego_view": observation["head_stereo_left"]}
        state = self._build_state(observation)
        with self._action_lock:
            if session_generation is not None and session_generation != self._session_generation:
                return
            request_generation = (
                self._session_generation if session_generation is None else session_generation
            )
            history = {"session_generation": request_generation}
            if self._reset_history:
                history["reset"] = True
            self._reset_history = False
        pred_action, *_ = self.client.query_action(
            image,
            instruction,
            state,
            {},
            history=history,
            dataset=self.query_dataset,
        )
        if self._closed:
            return
        pred_action = np.asarray(pred_action, dtype=np.float32)
        if pred_action.ndim == 1:
            pred_action = pred_action[None, :]
        if self.action_exec_horizon is not None:
            pred_action = pred_action[: self.action_exec_horizon]
        _validate_policy_actions(pred_action, self.robot.hand_spec)
        token_chunk = pred_action[:, :64]
        left_hand_chunk = pred_action[:, 64 : 64 + self.robot.hand_spec.dof]
        right_hand_chunk = pred_action[:, 64 + self.robot.hand_spec.dof :]
        query_latency_s = time.perf_counter() - query_start
        print(
            "[SonicPolicyAgent] Received action chunk "
            f"shape={pred_action.shape} "
            f"latency={query_latency_s:.3f}s "
            f"token_min={token_chunk.min():.4f} token_max={token_chunk.max():.4f} "
            f"token_mean={token_chunk.mean():.4f} "
            f"token_max_abs={np.max(np.abs(token_chunk)):.4f} "
            f"token_norm0={np.linalg.norm(token_chunk[0]):.4f} "
            f"left_hand_range=({left_hand_chunk.min():.4f},{left_hand_chunk.max():.4f}) "
            f"right_hand_range=({right_hand_chunk.min():.4f},{right_hand_chunk.max():.4f})",
            flush=True,
        )
        pred_action = self._filter_safe_actions(pred_action)
        with self._action_lock:
            if self._closed or (
                session_generation is not None and session_generation != self._session_generation
            ):
                return
            self.action_queue.extend(pred_action)
            queue_len = len(self.action_queue)
        print(
            "[SonicPolicyAgent] Enqueued action chunk "
            f"shape={pred_action.shape} queue_len={queue_len}",
            flush=True,
        )

    def _publish_initial_pose_token(
        self,
        *,
        token_source: str = "initial_token",
        queue_before: int | None = None,
        queue_after: int | None = None,
        step_time_s: float | None = None,
    ) -> None:
        token = np.zeros(64, dtype=np.float32)
        previous_token = self._last_token
        self._last_left_hand = np.asarray(self.robot.hand_spec.init_qpos, dtype=np.float32)
        self._last_right_hand = np.asarray(self.robot.hand_spec.init_qpos, dtype=np.float32)
        self._set_direct_hand_targets(self._last_left_hand, self._last_right_hand)
        self._send_token(
            token,
            left_hand=self._last_left_hand,
            right_hand=self._last_right_hand,
            final_action=np.concatenate([token, self._last_left_hand, self._last_right_hand]),
            token_source=token_source,
            queue_before=queue_before,
            queue_after=queue_after,
            step_time_s=step_time_s,
            token_jump_norm=(
                None
                if previous_token is None
                else float(np.linalg.norm(token - np.asarray(previous_token, dtype=np.float32)))
            ),
        )
        self._ensure_stream_started()

    def _start_async_policy_query(
        self,
        observation: dict[str, Any],
        instruction: str,
    ) -> None:
        if self._closed:
            return
        if self._async_query_error is not None:
            raise self._async_query_error
        with self._action_lock:
            if self.action_queue:
                self._policy_ready = True
                return
            session_generation = self._session_generation
        if self._async_query_thread is not None and self._async_query_thread.is_alive():
            return

        obs_snapshot = {
            key: value.copy() if isinstance(value, np.ndarray) else value
            for key, value in observation.items()
        }

        def _worker() -> None:
            try:
                self._query_policy(
                    obs_snapshot,
                    instruction,
                    session_generation,
                )
                with self._action_lock:
                    is_current = not self._closed and session_generation == self._session_generation
                    if is_current:
                        self._policy_ready = True
                if is_current:
                    print("[SonicPolicyAgent] GR00T first action prefetched", flush=True)
            except BaseException as exc:
                with self._action_lock:
                    if not self._closed and session_generation == self._session_generation:
                        self._async_query_error = exc

        print("[SonicPolicyAgent] Prefetching GR00T first action", flush=True)
        self._async_query_thread = threading.Thread(
            target=_worker,
            name="gr00t_n17_prefetch",
            daemon=True,
        )
        self._async_query_thread.start()

    def prefetch_policy_action(self, observation: dict[str, Any], instruction: str) -> None:
        self._start_async_policy_query(observation, instruction)

    def _publish_low_state(self, info: dict[str, Any] | None) -> None:
        if info is None or "proprio" not in info:
            keys = list(info.keys()) if isinstance(info, dict) else None
            print(
                "[SonicPolicyAgent][LOWSTATE_SKIP] "
                f"frame={self._frame_index} "
                f"info_none={info is None} keys={keys}",
                flush=True,
            )
            return
        proprio = dds_hand_feedback(info["proprio"], self.robot.hand_spec.name)
        tick = None
        try:
            tick = int(float(proprio.get("time", 0.0)) * 1000.0)
        except Exception:
            pass

        publish_start = time.perf_counter()
        self.unitree_bridge.PublishLowState(proprio)
        publish_ms = (time.perf_counter() - publish_start) * 1000.0
        self._low_state_publish_count += 1
        now = time.perf_counter()
        dt_ms = (
            0.0
            if self._last_low_state_publish_time is None
            else (now - self._last_low_state_publish_time) * 1000.0
        )
        self._last_low_state_publish_time = now

        # publish_ms是self.unitree_bridge.PublishLowState执行时间
        # dt_ms是相邻两个LowState相邻的用时
        if publish_ms > 20.0:
            print(
                "[SonicPolicyAgent][LOWSTATE_PUBLISH_SLOW] "
                f"count={self._low_state_publish_count} "
                f"frame={self._frame_index} "
                f"tick={tick} "
                f"publish_ms={publish_ms:.3f}",
                flush=True,
            )
        if dt_ms > 50.0:
            print(
                "[SonicPolicyAgent][LOWSTATE_GAP] "
                f"count={self._low_state_publish_count} "
                f"frame={self._frame_index} "
                f"tick={tick} "
                f"dt_ms={dt_ms:.3f} "
                f"publish_ms={publish_ms:.3f}",
                flush=True,
            )
        if self._low_state_publish_count == 1 or self._low_state_publish_count % 50 == 0:
            print(
                "[SonicPolicyAgent] publish_lowstate "
                f"count={self._low_state_publish_count} "
                f"frame={self._frame_index} "
                f"tick={tick} "
                f"dt_ms={dt_ms:.3f} "
                f"publish_ms={publish_ms:.3f}",
                flush=True,
            )

    def _low_cmd_received(self) -> bool:
        lock = getattr(self.unitree_bridge, "low_cmd_lock", None)
        if lock is None:
            return bool(self.unitree_bridge.low_cmd_received)
        with lock:
            return bool(self.unitree_bridge.low_cmd_received)

    def _publish_action_token(self, observation: dict[str, Any], instruction: str) -> None:
        if self._async_query_error is not None:
            raise self._async_query_error
        step_index = self._policy_step_index
        self._policy_step_index += 1
        step_time_s = step_index * self.control_dt
        if (
            self._token_period_s > 0.0
            and self._last_token is not None
            and step_time_s + 1e-9 < self._next_new_token_time_s
        ):  # 没有到下一个发送的时间，直接重复发送之前的token
            with self._action_lock:
                queue_len = len(self.action_queue)
            self._set_direct_hand_targets(self._last_left_hand, self._last_right_hand)
            self._send_token(
                self._last_token,
                left_hand=self._last_left_hand,
                right_hand=self._last_right_hand,
                final_action=np.concatenate(
                    [self._last_token, self._last_left_hand, self._last_right_hand]
                ),
                token_source="last_token_rate_hold",
                queue_before=queue_len,
                queue_after=queue_len,
                step_time_s=step_time_s,
                token_jump_norm=0.0,
            )
            self._ensure_stream_started()
            return

        self._start_async_policy_query(observation, instruction)
        with self._action_lock:
            queue_before = len(self.action_queue)
            action = self.action_queue.popleft() if self.action_queue else None
            queue_after = len(self.action_queue)
        if action is None:
            if self._last_token is None:
                return
            self._set_direct_hand_targets(self._last_left_hand, self._last_right_hand)
            self._send_token(
                self._last_token,
                left_hand=self._last_left_hand,
                right_hand=self._last_right_hand,
                final_action=np.concatenate(
                    [self._last_token, self._last_left_hand, self._last_right_hand]
                ),
                token_source="last_token_empty_queue",
                queue_before=queue_before,
                queue_after=queue_after,
                step_time_s=step_time_s,
                token_jump_norm=0.0,
            )
            self._ensure_stream_started()
            return

        self._policy_ready = True
        raw_token = action[:64]
        token = self._quantize_token(self._limit_token_step(raw_token))
        previous_token = self._last_token
        left_hand = action[64 : 64 + self.robot.hand_spec.dof]
        right_hand = action[64 + self.robot.hand_spec.dof :]
        action = action.copy()
        action[:64] = token
        self._last_token = token
        self._last_left_hand = left_hand
        self._last_right_hand = right_hand
        self._set_direct_hand_targets(left_hand, right_hand)
        self._send_token(
            token,
            left_hand=left_hand,
            right_hand=right_hand,
            final_action=action,
            token_source="policy_action",
            queue_before=queue_before,
            queue_after=queue_after,
            step_time_s=step_time_s,
            token_jump_norm=(
                None
                if previous_token is None
                else float(np.linalg.norm(token - np.asarray(previous_token, dtype=np.float32)))
            ),
        )
        if self._token_period_s > 0.0:
            while self._next_new_token_time_s <= step_time_s + 1e-9:
                self._next_new_token_time_s += self._token_period_s
        self._ensure_stream_started()

    def _update_sonic_ready(self, low_cmd_received: bool) -> bool:
        now = time.perf_counter()
        if low_cmd_received and not self._last_low_cmd_received:
            print("[SonicPolicyAgent] First SONIC LowCmd received", flush=True)
            self._first_low_cmd_time = now
        self._last_low_cmd_received = low_cmd_received
        ready = bool(low_cmd_received and self._started)
        if ready and not self._sonic_ready_reported:
            self._sonic_ready_reported = True
            print(
                f"[SonicPolicyAgent] SONIC token stream ready: tokens={self._stream_token_count}",
                flush=True,
            )
        elif low_cmd_received and not ready and now - self._last_waiting_ready_log_time >= 1.0:
            self._last_waiting_ready_log_time = now
            print("[SonicPolicyAgent] Waiting for SONIC token stream start", flush=True)
        return ready

    def _ground_hold_action(self) -> ActionCmd:
        # CMT: 这个指令是让机器人保持不动
        # OPT1: 自己实现
        # if self._ground_hold_target_q is None:
        #     self._ground_hold_target_q = np.asarray(
        #         self.robot.mjData.qpos[self.robot.body_joint_index + self.robot.qpos_offset - 1],
        #         dtype=np.float32,
        #     ).copy()
        # if not self._ground_hold_logged:
        #     print(
        #         "[SonicPolicyAgent] Holding current pose until SONIC LowCmd arrives",
        #         flush=True,
        #     )
        #     self._ground_hold_logged = True
        # self._print_fallback_action("sonic_stand_pd_hold")
        # return ActionCmd(
        #     "sonic_stand_pd_hold",
        #     target_q=self._ground_hold_target_q,
        #     kp=np.asarray(
        #         self.robot.sonic_config.get("MOTOR_KP", [150.0] * self.robot.num_body_dof),
        #         dtype=np.float64,
        #     ),
        #     kd=np.asarray(
        #         self.robot.sonic_config.get("MOTOR_KD", [3.0] * self.robot.num_body_dof),
        #         dtype=np.float64,
        #     ),
        # )

        # OPT2：直接调用g1_sonic的实现方法
        from humano.robots.g1_sonic import SONIC_DEFAULT_STAND_Q

        if not self._sonic_stand_hold_logged:
            print(
                "[SonicPolicyAgent] Holding SONIC default stand before policy handoff",
                flush=True,
            )
            self._sonic_stand_hold_logged = True
        return ActionCmd(
            "sonic_stand_pd_hold",
            target_q=SONIC_DEFAULT_STAND_Q,
            kp=np.asarray(
                self.robot.sonic_config.get("MOTOR_KP", SONIC_STAND_PD_KP),
                dtype=np.float64,
            ),
            kd=np.asarray(
                self.robot.sonic_config.get("MOTOR_KD", SONIC_STAND_PD_KD),
                dtype=np.float64,
            ),
        )

    def ready_for_policy(self) -> bool:
        # CMT：准备好的条件是 SONIC stream 已经启动，并且至少收到过一次 LowCmd。
        return self._started and self._low_cmd_received()

    def policy_action_ready(self) -> bool:
        if self._async_query_error is not None:
            raise self._async_query_error
        with self._action_lock:
            return bool(self.action_queue)

    def _fallback_action(self, force_hold=False) -> ActionCmd:
        low_cmd_received = self._low_cmd_received()
        self._update_sonic_ready(low_cmd_received)
        if force_hold or not low_cmd_received:
            print("[Gr00TN17Agent][!WARNING] low_cmd not received, use ground hold action")
            return self._ground_hold_action()
        # debug日志打印
        if low_cmd_received and self._debug_low_cmd_prints < 12:
            low_cmd = self.unitree_bridge.low_cmd
            num_motor = int(getattr(self.unitree_bridge, "num_body_motor", 29))
            q = np.asarray([low_cmd.motor_cmd[i].q for i in range(num_motor)], dtype=np.float32)
            dq = np.asarray([low_cmd.motor_cmd[i].dq for i in range(num_motor)], dtype=np.float32)
            kp = np.asarray([low_cmd.motor_cmd[i].kp for i in range(num_motor)], dtype=np.float32)
            kd = np.asarray([low_cmd.motor_cmd[i].kd for i in range(num_motor)], dtype=np.float32)
            tau = np.asarray([low_cmd.motor_cmd[i].tau for i in range(num_motor)], dtype=np.float32)
            body_torque = self.robot.compute_body_torques(low_cmd, self.unitree_bridge.use_sensor)
            self._print_fallback_action("wbc_torque", body_torque=body_torque)
            print(
                "[SonicPolicyAgent] LowCmd/body torque "
                f"q_norm={np.linalg.norm(q):.3f} dq_norm={np.linalg.norm(dq):.3f} "
                f"kp_range=({kp.min():.3f},{kp.max():.3f}) "
                f"kd_range=({kd.min():.3f},{kd.max():.3f}) "
                f"tau_norm={np.linalg.norm(tau):.3f} "
                f"body_torque_norm={np.linalg.norm(body_torque):.3f} "
                f"body_torque_max={np.max(np.abs(body_torque)):.3f} "
                f"pelvis_z={self.robot.pelvis_z:.3f} pelvis_vz={self.robot.pelvis_vz:.3f}",
                flush=True,
            )
            self._debug_low_cmd_prints += 1
        else:
            self._print_fallback_action("wbc_torque")

        return ActionCmd(
            "wbc_torque",
            low_cmd=self.unitree_bridge.low_cmd,
            use_sensor=self.unitree_bridge.use_sensor,
            left_hand_cmd=self._direct_left_hand_cmd,
            right_hand_cmd=self._direct_right_hand_cmd,
        )

    # ------------------------------------------------------------------
    # Template method: get_action() unified entry
    # ------------------------------------------------------------------

    def _before_action_step(
        self,
        observation,
        instruction: str,
        info: dict[str, Any] | None,
        *,
        action_type: ActionType,
        **kwargs: Any,
    ) -> None:
        """Pre-action hook: advance frame counter, publish low state."""
        self._frame_index += 1
        self._publish_low_state(info)

    def _action_step(
        self,
        observation,
        instruction: str,
        info: dict[str, Any] | None,
        *,
        action_type: ActionType,
        **kwargs: Any,
    ) -> ActionCmd:
        handlers = {
            ActionType.POLICY: self._get_policy_action,
            ActionType.STABILIZE: self._get_stabilize_action,
            ActionType.BOOTSTRAP_STABILIZE: self._get_bootstrap_stabilize_action,
            ActionType.PREFETCH_WARMUP: self._get_prefetch_warmup_action,
        }
        try:
            handler = handlers[action_type]
        except KeyError:
            raise ValueError(f"Unsupported action_type: {action_type}") from None
        return handler(observation, instruction, info, **kwargs)

    def _after_action_step(
        self,
        action: ActionCmd,
        observation,
        instruction: str,
        info: dict[str, Any] | None,
        *,
        action_type: ActionType,
        **kwargs: Any,
    ) -> None:
        """Post-action hook: reserved for debug, stats, logging, etc."""
        pass

    # ------------------------------------------------------------------
    # Private action implementations (called by _action_step dispatch)
    # ------------------------------------------------------------------

    def _get_policy_action(
        self,
        observation,
        instruction: str,
        info: dict[str, Any] | None,
        **kwargs: Any,
    ) -> ActionCmd:
        self._publish_action_token(observation, instruction)
        return self._fallback_action()

    def _get_stabilize_action(
        self,
        observation,
        instruction: str,
        info: dict[str, Any] | None,
        **kwargs: Any,
    ) -> ActionCmd:
        low_cmd_received = self._low_cmd_received()
        if not low_cmd_received:
            self._publish_initial_pose_token()
            return self._fallback_action()

        self._publish_action_token(observation, instruction)
        return self._fallback_action()

    def _get_bootstrap_stabilize_action(
        self,
        observation,
        instruction: str,
        info: dict[str, Any] | None,
        **kwargs: Any,
    ) -> ActionCmd:
        return self._get_prefetch_warmup_action(
            observation, instruction, info, prefetch_policy=False
        )

    def _get_prefetch_warmup_action(
        self,
        observation,
        instruction: str,
        info: dict[str, Any] | None,
        *,
        prefetch_policy: bool = False,
        **kwargs: Any,
    ) -> ActionCmd:
        if prefetch_policy:
            self._start_async_policy_query(observation, instruction)
        self._publish_initial_pose_token()
        return self._fallback_action()

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    def get_action(
        self,
        observation: dict[str, Any],
        instruction: str | None = None,
        info: dict[str, Any] | None = None,
        *,
        action_type: ActionType = ActionType.POLICY,
        **kwargs: Any,
    ) -> ActionCmd:
        """Unified entry point: dispatches to the correct action via action_type.

        Usage::

            action = agent.get_action(obs, instruction=inst, info=info)               # policy (default)
            action = agent.get_action(obs, instruction=inst, info=info,
                                      action_type=ActionType.STABILIZE)               # stabilize
            action = agent.get_action(obs, instruction=inst, info=info,
                                      action_type=ActionType.PREFETCH_WARMUP,
                                      prefetch_policy=True)                           # prefetch warmup
        """
        instruction = instruction or "complete the task"

        self._before_action_step(
            observation,
            instruction,
            info,
            action_type=action_type,
            **kwargs,
        )

        action = self._action_step(
            observation,
            instruction,
            info,
            action_type=action_type,
            **kwargs,
        )

        self._after_action_step(
            action,
            observation,
            instruction,
            info,
            action_type=action_type,
            **kwargs,
        )

        return action

    def get_stabilize_action(
        self,
        observation,
        info: dict[str, Any] | None = None,
        instruction: str | None = None,
    ) -> ActionCmd:
        """Backward-compatible wrapper around get_action(STABILIZE)."""
        return self.get_action(
            observation,
            instruction=instruction,
            info=info,
            action_type=ActionType.STABILIZE,
        )

    def get_bootstrap_stabilize_action(
        self,
        observation,
        info: dict[str, Any] | None = None,
        instruction: str | None = None,
    ) -> ActionCmd:
        """Backward-compatible wrapper around get_action(BOOTSTRAP_STABILIZE)."""
        return self.get_action(
            observation,
            instruction=instruction,
            info=info,
            action_type=ActionType.BOOTSTRAP_STABILIZE,
        )

    def get_prefetch_warmup_action(
        self,
        observation,
        info: dict[str, Any] | None = None,
        instruction: str | None = None,
        *,
        prefetch_policy: bool = False,
    ) -> ActionCmd:
        """Backward-compatible wrapper around get_action(PREFETCH_WARMUP)."""
        return self.get_action(
            observation,
            instruction=instruction,
            info=info,
            action_type=ActionType.PREFETCH_WARMUP,
            prefetch_policy=prefetch_policy,
        )


# =========================================================================
# SonicPolicyNativeAgent — in-process SONIC 模式
# =========================================================================


class NativeMotorBridge:
    def _extract_robot_state(self) -> RobotState:
        """Build a :class:`RobotState` from the MuJoCo data of *self.robot* (G1Sonic)."""
        robot = self.robot

        body_q = robot.mjData.qpos[robot.body_joint_index + robot.qpos_offset - 1].copy()
        body_dq = robot.mjData.qvel[robot.body_joint_index + robot.qvel_offset - 1].copy()
        base_quat = robot.mjData.qpos[3:7].copy()
        base_ang_vel = robot.mjData.qvel[3:6].copy()

        return RobotState(
            body_q=body_q,
            body_dq=body_dq,
            base_quat=base_quat,
            base_ang_vel=base_ang_vel,
            pelvis_z=float(robot.pelvis_z),
        )

    def _motor_cmd_to_action(
        self,
        cmd: MotorCommand,
        left_hand: np.ndarray | None = None,
        right_hand: np.ndarray | None = None,
    ) -> ActionCmd:
        """Convert a :class:`MotorCommand` to an ``ActionCmd("wbc_torque", …)``.

        Uses :class:`MotorCommand` directly as ``low_cmd`` (it exposes the
        DDS-compatible ``.motor_cmd`` property), and
        :meth:`MotorCommand.hand_command` for hands — no mock classes needed.
        """
        spec = self.robot.hand_spec
        kp, kd = _hand_gains(spec)
        return ActionCmd(
            "wbc_torque",
            low_cmd=cmd,
            use_sensor=False,
            left_hand_cmd=_brainco_command_gains(
                MotorCommand.hand_command(
                    left_hand if left_hand is not None else spec.init_qpos, kp=kp, kd=kd
                ),
                spec,
            ),
            right_hand_cmd=_brainco_command_gains(
                MotorCommand.hand_command(
                    right_hand if right_hand is not None else spec.init_qpos, kp=kp, kd=kd
                ),
                spec,
            ),
        )


class SonicPolicyNativeAgent(NativeMotorBridge):
    """Native-mode SONIC-token policy agent: runs the SONIC decoder in-process via
    :class:`SonicController`, bypassing ZMQ and DDS entirely.

    Follows the same ``get_action(…, action_type=…)`` template-method pattern
    as :class:`SonicPolicyDeployAgent` so the two are interchangeable from the
    eval loop's perspective.
    """

    def __init__(
        self,
        robot,
        host: str,
        port: int,
        *,
        sonic_config: dict[str, Any],
        native_model_path: str | None = None,
        native_obs_config_path: str | None = None,
        native_control_dt: float = 0.02,
        native_provider: str = "cpu",
        query_dataset: str = "humano",
        action_exec_horizon: int | None = None,
        fast_isaac_sync_fn: Callable[[], dict[str, np.ndarray]] | None = None,
        policy_seed: int | None = None,
        **kwargs: Any,
    ):
        # ---- shared: HTTP policy client ----
        self.robot = robot
        self.client = HttpActionClient(host, port)
        self.sonic_config = sonic_config
        self._fast_isaac_sync_fn = fast_isaac_sync_fn
        self.query_dataset = query_dataset
        self.action_exec_horizon = action_exec_horizon
        self._policy_seed = policy_seed

        # ---- shared: action queue ----
        self.action_queue: deque[np.ndarray] = deque()
        self._action_lock = threading.Lock()
        self._closed = False
        self._reset_history = True
        self._frame_index = 0

        # ---- shared: last-token cache ----
        self._last_token: np.ndarray | None = None
        self._last_left_hand: np.ndarray | None = None
        self._last_right_hand: np.ndarray | None = None

        # ---- shared: timing ----
        self.sim_dt = float(sonic_config["SIMULATE_DT"])
        self.control_dt = float(sonic_config.get("CONTROL_DT", native_control_dt))

        # ---- shared: policy ready / debug ----
        self._policy_ready = False
        self._debug_state_printed = False
        self._debug_token_prints = 0
        self._last_token_source: str | None = None
        self._token_source_counts: dict[str, int] = {}
        self._last_fallback_action: str | None = None

        # ---- shared: ground-hold / stand-hold ----
        self._ground_hold_logged = False
        self._sonic_stand_hold_logged = False

        # ---- native: in-process SONIC controller ----
        from humano.wbc.sonic import SonicController
        from humano.wbc.sonic import WBC_SONIC_WARMUP_READY_S, WBC_SONIC_WARMUP_WBC_S

        if native_model_path is None:
            raise ValueError("native_model_path is required for SonicPolicyNativeAgent")
        if native_obs_config_path is None:
            raise ValueError("native_obs_config_path is required for SonicPolicyNativeAgent")

        self._sonic_controller = SonicController(
            model_path=native_model_path,
            obs_config_path=native_obs_config_path,
            control_dt=native_control_dt,
            provider=native_provider,
        )
        native_env_step_dt = float(kwargs.pop("native_env_step_dt", 0.0))
        if native_env_step_dt > 0:
            self._sonic_controller.set_env_step_dt(native_env_step_dt)
        env_step_dt = native_env_step_dt or self.sim_dt
        self._ready_wbc_steps = round(WBC_SONIC_WARMUP_WBC_S / env_step_dt)
        self.bootstrap_stabilize_steps = self._ready_wbc_steps + round(
            WBC_SONIC_WARMUP_READY_S / env_step_dt
        )
        self.prefetch_warmup_steps = 0
        self._wbc_stabilizer = None
        print(
            "[SonicPolicyAgent] Created native SONIC controller. "
            f"model={native_model_path}, obs_config={native_obs_config_path}, "
            f"control_dt={native_control_dt:.4f}s, "
            f"env_step_dt={native_env_step_dt:.4f}s, "
            f"inference_interval={self._sonic_controller._inference_interval}"
        )

    # ==================================================================
    #  Shared helpers (mirror SonicPolicyDeployAgent)
    # ==================================================================

    def _robot_debug_state(self) -> dict[str, float]:
        state: dict[str, float] = {}
        for name in ("pelvis_z", "pelvis_vz"):
            try:
                value = float(getattr(self.robot, name))
            except Exception:
                continue
            if np.isfinite(value):
                state[name] = value
        try:
            qvel = np.asarray(self.robot.mjData.qvel[:6], dtype=np.float32)
            if qvel.size:
                state["base_qvel_max"] = float(np.max(np.abs(qvel)))
        except Exception:
            pass
        return state

    @staticmethod
    def _format_debug_state(state: dict[str, float]) -> str:
        return " ".join(f"{key}={value:.3f}" for key, value in state.items())

    def _print_fallback_action(
        self,
        fallback_action: str,
        *,
        body_torque: np.ndarray | None = None,
    ) -> None:
        should_print = fallback_action != self._last_fallback_action
        if not should_print:
            return
        robot_state = self._robot_debug_state()
        state_text = self._format_debug_state(robot_state)
        state_text = f" {state_text}" if state_text else ""
        torque_text = ""
        if body_torque is not None:
            torque_text = (
                f" body_torque_norm={np.linalg.norm(body_torque):.3f}"
                f" body_torque_max={np.max(np.abs(body_torque)):.3f}"
            )
        print(
            "[SonicPolicyAgent] Fallback action "
            f"frame={self._frame_index} action={fallback_action}"
            f"{state_text}"
            f"{torque_text}",
            flush=True,
        )
        self._last_fallback_action = fallback_action

    # ==================================================================
    #  GR00T state building (mirrors DeployAgent)
    # ==================================================================

    def _build_state(self, observation: dict[str, Any]) -> dict[str, np.ndarray]:
        state = _policy_state(self.robot, observation)
        if not self._debug_state_printed:
            print(
                "[SonicPolicyAgent] GR00T state built "
                f"left_hand="
                f"{np.array2string(state['left_hand'], precision=4, separator=',')} "
                f"right_hand="
                f"{np.array2string(state['right_hand'], precision=4, separator=',')} "
                f"projected_gravity={np.array2string(state['projected_gravity'], precision=4, separator=',')}",
                flush=True,
            )
            self._debug_state_printed = True
        return state

    # ==================================================================
    #  GR00T policy query (mirrors DeployAgent)
    # ==================================================================
    def _query_policy(self, observation: dict[str, Any], instruction: str) -> None:
        query_start = time.perf_counter()
        query_frame = int(self._frame_index)
        robot_state = self._robot_debug_state()
        state_text = self._format_debug_state(robot_state)
        state_text = f" {state_text}" if state_text else ""
        print(
            f"[SonicPolicyAgent] Querying /act frame={query_frame}{state_text} "
            f"instruction={instruction!r}",
            flush=True,
        )

        # -- fast mode: trigger isaac sync + render, inject fresh frames
        #    into the observation dict so GR00T sees up-to-date camera images.
        if self._fast_isaac_sync_fn is not None:
            fresh_frames = self._fast_isaac_sync_fn()
            key = "head_stereo_left"
            img = fresh_frames[key]
            if "isaac_joint_qpos" in fresh_frames:
                observation["joint_qpos"] = np.asarray(
                    fresh_frames["isaac_joint_qpos"], dtype=np.float32
                )
            # DEBUG starts
            # import cv2
            # import shutil
            # import os
            # if query_frame < 510:
            #     shutil.rmtree("debug/native_sonic", ignore_errors=True)
            # os.makedirs("debug/native_sonic", exist_ok=True)
            # print(
            #     f"[SonicPolicyAgent][DEBUG] Saving debug frame {key}_{query_frame:06d} to debug/native_sonic/"
            # )
            # cv2.imwrite(
            #     f"debug/native_sonic/{key}_{query_frame:06d}.png",
            #     img[:, :, ::-1],
            # )
            # DEBUG ends
            observation[key] = img

        image = {"ego_view": observation["head_stereo_left"]}
        state = self._build_state(observation)
        history = {"reset": True} if self._reset_history else {}
        self._reset_history = False
        pred_action, *_ = self.client.query_action(
            image,
            instruction,
            state,
            {},
            history=history,
            dataset=self.query_dataset,
            inference_seed=self._policy_seed,
        )
        if self._closed:
            return
        pred_action = np.asarray(pred_action, dtype=np.float32)
        if pred_action.ndim == 1:
            pred_action = pred_action[None, :]
        if self.action_exec_horizon is not None:
            pred_action = pred_action[: self.action_exec_horizon]
        _validate_policy_actions(pred_action, self.robot.hand_spec)
        query_latency_s = time.perf_counter() - query_start
        print(
            "[SonicPolicyAgent] Received action chunk "
            f"shape={pred_action.shape} "
            f"latency={query_latency_s:.3f}s "
            f"token_norm0={np.linalg.norm(pred_action[0, :64]):.4f}",
            flush=True,
        )
        with self._action_lock:
            if self._closed:
                return
            self.action_queue.extend(pred_action)
            queue_len = len(self.action_queue)
        print(
            "[SonicPolicyAgent] Enqueued action chunk "
            f"shape={pred_action.shape} queue_len={queue_len}",
            flush=True,
        )

    # ==================================================================
    #  Readiness (always true — controller is in-process)
    # ==================================================================

    def _low_cmd_received(self) -> bool:
        return True

    def ready_for_policy(self) -> bool:
        return True

    def policy_action_ready(self) -> bool:
        with self._action_lock:
            return bool(self.action_queue)

    def prefetch_policy_action(self, observation: dict[str, Any], instruction: str) -> None:
        self._query_policy(observation, instruction)
        self._policy_ready = True

    # ==================================================================
    #  Ground-hold / fallback
    # ==================================================================

    def _ground_hold_action(self) -> ActionCmd:
        from humano.robots.g1_sonic import SONIC_DEFAULT_STAND_Q

        if not self._sonic_stand_hold_logged:
            print(
                "[SonicPolicyAgent] Holding SONIC default stand before policy handoff",
                flush=True,
            )
            self._sonic_stand_hold_logged = True
        return ActionCmd(
            "sonic_stand_pd_hold",
            target_q=SONIC_DEFAULT_STAND_Q,
            kp=np.asarray(
                self.robot.sonic_config.get("MOTOR_KP", SONIC_STAND_PD_KP),
                dtype=np.float64,
            ),
            kd=np.asarray(
                self.robot.sonic_config.get("MOTOR_KD", SONIC_STAND_PD_KD),
                dtype=np.float64,
            ),
        )

    def _fallback_action(self, force_hold: bool = False) -> ActionCmd:
        return self._ground_hold_action()

    # ==================================================================
    #  Native: robot state → wbc.base types
    # ==================================================================
    # ==================================================================
    #  Native: action-producing pipeline
    # ==================================================================
    def _get_action_use_native_sonic(self, observation, instruction=None, info=None) -> ActionCmd:
        # ---- one-shot: first POLICY phase call ----
        if not getattr(self, "_policy_phase_logged", False):
            self._policy_phase_logged = True
            _ql = len(self.action_queue) if hasattr(self, "action_queue") else -1
            print(
                f"[SonicPolicyAgent] POLICY phase started: "
                f"frame={self._frame_index} "
                f"ctrl_tick={self._sonic_controller._control_tick_count} "
                f"queue_len={_ql} "
                f"last_token={'None' if self._last_token is None else 'set'}",
                f"needs_new_token={self._sonic_controller.needs_new_token}",
                flush=True,
            )

        left_hand = self._last_left_hand
        right_hand = self._last_right_hand

        if self._sonic_controller.needs_new_token:
            with self._action_lock:
                queue_empty = not bool(self.action_queue)

            if queue_empty:
                self._query_policy(observation, instruction or "complete the task")

            with self._action_lock:
                if self.action_queue:
                    action = self.action_queue.popleft()
                else:
                    action = None

            if self._frame_index == 502:
                print(
                    f"[SonicPolicyAgent] Frame 502: action_queue_len={len(self.action_queue)} "
                    f"last_token={'None' if self._last_token is None else 'set'}",
                    f"action={'None' if action is None else 'set'}",
                    flush=True,
                )
            if action is not None:
                token = np.asarray(action[:64], dtype=np.float64)
                hand_dof = self.robot.hand_spec.dof
                left_hand = np.asarray(action[64 : 64 + hand_dof], dtype=np.float64)
                right_hand = np.asarray(action[64 + hand_dof :], dtype=np.float64)
                self._last_token = token.astype(np.float32, copy=False)
                self._last_left_hand = left_hand.astype(np.float32, copy=False)
                self._last_right_hand = right_hand.astype(np.float32, copy=False)
                # one-shot: first real policy token consumed
                if not getattr(self, "_first_policy_token_logged", False):
                    self._first_policy_token_logged = True
                    print(
                        f"[SonicPolicyAgent] First POLICY token consumed: "
                        f"frame={self._frame_index} "
                        f"ctrl_tick={self._sonic_controller._control_tick_count} "
                        f"token_head={token[:4].tolist()} "
                        f"queue_remaining={len(self.action_queue)}",
                        flush=True,
                    )
        rs = self._extract_robot_state()
        if self._last_token is None:
            self._sonic_controller.warmup(rs)
            return self._wbc_stabilizer.get_stabilize_action(observation)
        token = self._last_token
        if self._frame_index % 20 == 0:
            robot_state_debug = self._robot_debug_state()
            state_text = self._format_debug_state(robot_state_debug)
            state_text = f" {state_text}" if state_text else ""
            print(
                f"[SonicPolicyAgent] Native SONIC step: frame={self._frame_index} rs={state_text}",
                flush=True,
            )
        cmd = self._sonic_controller.step(rs, token)
        return self._motor_cmd_to_action(cmd, left_hand, right_hand)

    # ==================================================================
    #  Template method: get_action() dispatch  (mirrors DeployAgent)
    # ==================================================================

    _ACTION_HANDLERS: dict[ActionType, str] = {
        ActionType.POLICY: "_get_policy_action",
        ActionType.STABILIZE: "_get_stabilize_action",
        ActionType.BOOTSTRAP_STABILIZE: "_get_bootstrap_stabilize_action",
        ActionType.PREFETCH_WARMUP: "_get_prefetch_warmup_action",
    }

    def _before_action_step(
        self,
        observation,
        instruction: str,
        info: dict[str, Any] | None,
        *,
        action_type: ActionType,
        **kwargs: Any,
    ) -> None:
        """Advance frame counter (no DDS LowState publish needed in native mode)."""
        self._frame_index += 1

    def _action_step(
        self,
        observation,
        instruction: str,
        info: dict[str, Any] | None,
        *,
        action_type: ActionType,
        **kwargs: Any,
    ) -> ActionCmd:
        handler_name = self._ACTION_HANDLERS[action_type]
        return getattr(self, handler_name)(observation, instruction, info, **kwargs)

    def _after_action_step(
        self,
        action: ActionCmd,
        observation,
        instruction: str,
        info: dict[str, Any] | None,
        *,
        action_type: ActionType,
        **kwargs: Any,
    ) -> None:
        """No-op: reserved for debug/stats."""

    # ---- private action implementations (called by _action_step dispatch) ----

    def _get_policy_action(
        self, observation, instruction: str, info: dict[str, Any] | None, **kwargs: Any
    ) -> ActionCmd:
        return self._get_action_use_native_sonic(observation, instruction, info)

    def _get_stabilize_action(
        self, observation, instruction: str, info: dict[str, Any] | None, **kwargs: Any
    ) -> ActionCmd:
        return self._get_bootstrap_stabilize_action(observation, instruction, info)

    def _get_bootstrap_stabilize_action(
        self, observation, instruction: str, info: dict[str, Any] | None, **kwargs: Any
    ) -> ActionCmd:
        """Keep WBC in control throughout initialization."""
        if self._frame_index > self._ready_wbc_steps:
            self._sonic_controller.warmup(self._extract_robot_state())
        return self._wbc_stabilizer.get_stabilize_action(observation)

    def _get_prefetch_warmup_action(
        self,
        observation,
        instruction: str,
        info: dict[str, Any] | None,
        *,
        prefetch_policy: bool = False,
        **kwargs: Any,
    ) -> ActionCmd:
        if prefetch_policy:
            self._query_policy(observation, instruction or "complete the task")
        return self._get_bootstrap_stabilize_action(observation, instruction, info)

    # ---- public entry points ----

    def get_action(
        self,
        observation: dict[str, Any],
        instruction: str | None = None,
        info: dict[str, Any] | None = None,
        *,
        action_type: ActionType = ActionType.POLICY,
        **kwargs: Any,
    ) -> ActionCmd:
        """Template-method entry point (same signature as :class:`SonicPolicyDeployAgent`)."""
        instruction = instruction or "complete the task"

        self._before_action_step(observation, instruction, info, action_type=action_type, **kwargs)

        action = self._action_step(
            observation, instruction, info, action_type=action_type, **kwargs
        )

        self._after_action_step(
            action, observation, instruction, info, action_type=action_type, **kwargs
        )

        return action

    def get_stabilize_action(
        self,
        observation,
        info: dict[str, Any] | None = None,
        instruction: str | None = None,
    ) -> ActionCmd:
        return self.get_action(
            observation, instruction=instruction, info=info, action_type=ActionType.STABILIZE
        )

    def get_bootstrap_stabilize_action(
        self,
        observation,
        info: dict[str, Any] | None = None,
        instruction: str | None = None,
    ) -> ActionCmd:
        return self.get_action(
            observation,
            instruction=instruction,
            info=info,
            action_type=ActionType.BOOTSTRAP_STABILIZE,
        )

    def get_prefetch_warmup_action(
        self,
        observation,
        info: dict[str, Any] | None = None,
        instruction: str | None = None,
        *,
        prefetch_policy: bool = False,
    ) -> ActionCmd:
        return self.get_action(
            observation,
            instruction=instruction,
            info=info,
            action_type=ActionType.PREFETCH_WARMUP,
            prefetch_policy=prefetch_policy,
        )

    # ==================================================================
    #  Lifecycle
    # ==================================================================

    def reset(
        self,
        *,
        policy_seed: int | None = None,
        **_: Any,
    ) -> None:
        if policy_seed is not None:
            self._policy_seed = policy_seed
        with self._action_lock:
            self.action_queue.clear()
        self._reset_history = True
        self._frame_index = 0
        self._last_token = None
        self._last_left_hand = None
        self._last_right_hand = None
        self._debug_state_printed = False
        self._debug_token_prints = 0
        self._last_token_source = None
        self._token_source_counts.clear()
        self._last_fallback_action = None
        self._ground_hold_logged = False
        self._sonic_stand_hold_logged = False
        self._policy_ready = False
        self._bootstrap_transition_logged = False
        self._policy_phase_logged = False
        self._first_policy_token_logged = False
        self._sonic_controller.reset()
        if self._wbc_stabilizer is None:
            from humano.agents.sonic_decoupled_wbc_agent import SonicDecoupledWbcAgent

            self._wbc_stabilizer = SonicDecoupledWbcAgent(
                self.robot,
                self.sonic_config,
                time_fn=lambda: float(self.robot.mjData.time),
                preserve_spawn_heading=True,
            )
        if self._wbc_stabilizer is not None:
            self._wbc_stabilizer._spawn_heading_yaw = None
            self._wbc_stabilizer._cached_target_q = None
            self._wbc_stabilizer._cached_left_hand_q = None
            self._wbc_stabilizer._cached_right_hand_q = None
            self._wbc_stabilizer._wbc_policy.reset(init_time=float(self.robot.mjData.time))
            lower_body_policy = getattr(self._wbc_stabilizer._wbc_policy, "lower_body_policy", None)
            if lower_body_policy is not None:
                lower_body_policy.use_policy_action = True

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True

    def stop_stream(self) -> None:
        """Native mode: SONIC controller is in-process; no ZMQ stream to stop."""


# =========================================================================
#  Factory — 根据 use_native_sonic 参数分发到正确的实现类
# =========================================================================
def SonicPolicyAgent(
    robot,
    host: str,
    port: int,
    *,
    use_native_sonic: bool = False,
    **kwargs: Any,
):
    """Factory: return the right SONIC-token agent based on *use_native_sonic*.

    Parameters
    ----------
    use_native_sonic : bool
        False → :class:`SonicPolicyDeployAgent` (ZMQ + DDS service mode).
        True  → :class:`SonicPolicyNativeAgent` (in-process SONIC controller).

    All other keyword arguments are forwarded to the selected implementation's
    __init__.
    """
    if use_native_sonic:
        return SonicPolicyNativeAgent(robot, host, port, **kwargs)
    return SonicPolicyDeployAgent(robot, host, port, **kwargs)
