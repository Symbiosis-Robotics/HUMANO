"""
In-process SONIC controller — replaces the C++ ``g1_deploy_onnx_ref`` CONTROL state.

Implements the :class:`WholeBodyController` interface.  A full CONTROL tick
(state logging → observation building → ONNX inference → motor command) runs at
the SONIC control rate (default 50 Hz), throttled relative to the environment
step interval.  Between CONTROL ticks :meth:`step` returns the cached
:class:`MotorCommand` — matching the C++ ``LowCommandWriter`` behaviour.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml

from humano.wbc.base import MotorCommand, RobotState, WholeBodyController
from humano.wbc.sonic.observation import (
    OBS_REGISTRY,
    get_gatherer,
    set_shared_state,
)
from humano.wbc.sonic.policy import SonicPolicy
from humano.wbc.sonic.policy_parameters import (
    default_angles,
    g1_action_scale,
    isaaclab_to_mujoco,
    kds,
    kps,
    mujoco_to_isaaclab,
)
from humano.wbc.state_logger import StateLogger

_NUM_JOINTS = 29


def _make_default_command() -> MotorCommand:
    """PD-hold at ``default_angles`` (zero action).

    Used as the initial cached command before the first inference tick,
    mirroring the C++ behaviour where ``motor_command_buffer_`` starts
    with a safe stand-still command.
    """
    return MotorCommand(
        q_target=default_angles.copy(),
        dq_target=np.zeros(_NUM_JOINTS, dtype=np.float64),
        kp=kps.copy(),
        kd=kds.copy(),
        tau_ff=np.zeros(_NUM_JOINTS, dtype=np.float64),
    )


# ---------------------------------------------------------------------------
# YAML helper
# ---------------------------------------------------------------------------
def _parse_obs_config(path: str | Path) -> list[str]:
    """Return the list of *enabled* policy observation names from the YAML."""
    with open(path) as fh:
        raw = yaml.safe_load(fh)
    obs_list: list[dict[str, Any]] = raw.get("observations", [])
    return [o["name"] for o in obs_list if o.get("enabled", True)]


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------
class SonicController(WholeBodyController):
    """Synchronous, in-process replacement for the C++ CONTROL state.

    Parameters
    ----------
    model_path:
        Path to ``model_decoder.onnx``.
    obs_config_path:
        Path to ``observation_config.yaml`` (release config).
    control_dt:
        Control period in seconds (default 0.02 = 50 Hz).
    logger_capacity:
        StateLogger ring buffer capacity.
    """

    def __init__(
        self,
        model_path: str | Path,
        obs_config_path: str | Path,
        control_dt: float = 0.02,
        logger_capacity: int = 500,
        provider: str = "cpu",
    ) -> None:
        super().__init__()

        # -- ONNX policy --
        self._policy = SonicPolicy(model_path, provider=provider)

        # -- observation layout --
        self._enabled_obs_names = _parse_obs_config(obs_config_path)
        self._build_obs_layout()

        # -- validate dimensions --
        if self._policy.input_dim != self._total_obs_dim:
            raise ValueError(
                f"ONNX model input dim {self._policy.input_dim} != "
                f"observation total dim {self._total_obs_dim}. "
                f"Enabled observations: {self._enabled_obs_names}"
            )

        # -- state --
        self._control_dt = control_dt
        self._state_logger = StateLogger(capacity=logger_capacity)
        self._obs_buffer = np.zeros(self._total_obs_dim, dtype=np.float64)
        # Previous CONTROL tick's policy action (updated only at inference ticks,
        # so it never carries stale duplicated values between ticks).
        self._last_action = np.zeros(_NUM_JOINTS, dtype=np.float64)

        # token and frame counter
        self._token_data: np.ndarray | None = None
        self._frame_index: int = 0
        # Separate counter for CONTROL (inference) ticks only — used for
        # diagnostics so pre-stabilisation hold frames don't skew the count.
        self._control_tick_count: int = 0

        # -- inference throttling (matches C++ CONTROL 50 Hz / LowCmd 500 Hz) --
        # env.step() may run faster than the SONIC control rate.  The full
        # CONTROL tick (log → gather → infer) is throttled; between ticks we
        # return the cached MotorCommand (C++ LowCommandWriter behaviour).
        self._inference_interval: int = 1  # set via set_env_step_dt()
        self._last_inference_index: int = -(10**9)  # force first call to run CONTROL tick
        self._cached_motor_command: MotorCommand = _make_default_command()
        # Populated only on an actual 50 Hz inference tick for native recording.
        self.last_tick_trace: dict[str, np.ndarray] | None = None

    # ------------------------------------------------------------------
    # Observation layout
    # ------------------------------------------------------------------
    def _build_obs_layout(self) -> None:
        """Pre-compute (name, offset, dimension) for each enabled observation."""
        self._obs_layout: list[tuple[str, int, int]] = []
        offset = 0
        for name in self._enabled_obs_names:
            if name not in OBS_REGISTRY:
                raise KeyError(f"Unknown observation '{name}'. Available: {list(OBS_REGISTRY)}")
            dim, _ = OBS_REGISTRY[name]
            self._obs_layout.append((name, offset, dim))
            offset += dim
        self._total_obs_dim = offset

    # ------------------------------------------------------------------
    # Joint remapping (exact C++ GatherRobotStateToLogger equivalent)
    # ------------------------------------------------------------------
    @staticmethod
    def _remap_joints(
        body_q_raw: np.ndarray,
        body_dq_raw: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Remap MuJoCo-order joints to the scrambled C++ state-logger order.

        C++ equivalent::

            for (int i = 0; i < G1_NUM_MOTOR; i++) {
                body_q[i] = ls->motor_state()[mujoco_to_isaaclab[i]].q()
                          - default_angles[mujoco_to_isaaclab[i]];
                body_dq[i] = ls->motor_state()[mujoco_to_isaaclab[i]].dq();
            }
        """
        q = np.zeros(_NUM_JOINTS, dtype=np.float64)
        dq = np.zeros(_NUM_JOINTS, dtype=np.float64)
        for i in range(_NUM_JOINTS):
            isaac_idx = mujoco_to_isaaclab[i]
            q[i] = body_q_raw[isaac_idx] - default_angles[isaac_idx]
            dq[i] = body_dq_raw[isaac_idx]
        return q, dq

    # ------------------------------------------------------------------
    # WholeBodyController interface
    # ------------------------------------------------------------------
    def reload_model(self) -> None:
        """Close the current ONNX session and reload the decoder from disk."""
        self._policy.reload()

    def reset(self, reload_model: bool = True) -> None:
        """Discard all history and reload the model — call at the start of a new episode."""
        self._state_logger.clear()
        self._last_action.fill(0.0)
        self._obs_buffer.fill(0.0)
        self._token_data = None
        self._frame_index = 0
        self._control_tick_count = 0
        self._last_inference_index = -(10**9)
        self._cached_motor_command = _make_default_command()
        self.last_tick_trace = None
        if reload_model:
            self._policy.reload()

    def set_env_step_dt(self, env_step_dt: float) -> None:
        """Configure the env.step() interval so the controller can throttle the CONTROL tick.

        Must be called once before the first step.  When *env_step_dt* is
        smaller than ``control_dt`` (e.g. 0.005 s vs 0.020 s), the full
        CONTROL tick (state logging → observation gathering → inference)
        only runs every *N*-th call to :meth:`step` / :meth:`warmup`.
        Intermediate calls return the cached :class:`MotorCommand` without
        touching the StateLogger — matching the C++ ``LowCommandWriter``
        behaviour.
        """
        if env_step_dt > 0 and env_step_dt < self._control_dt:
            self._inference_interval = max(1, int(round(self._control_dt / env_step_dt)))
        else:
            self._inference_interval = 1

    @property
    def needs_new_token(self) -> bool:
        """Return True if the next ``step()`` call will trigger a CONTROL tick.

        The caller should only provide a new token when this returns True;
        otherwise the token will be ignored (the controller returns the
        cached MotorCommand without running inference or logging state).

        This mirrors the service-mode ``_publish_action_token`` rate-hold
        gate: tokens are only consumed from the queue at the SONIC CONTROL
        rate (50 Hz), not at the env.step() rate (200 Hz).
        """
        return (self._frame_index + 1) - self._last_inference_index >= self._inference_interval

    def warmup(self, robot_state: RobotState) -> MotorCommand:
        """Advance decoder history with fixed zero input during WBC control."""
        return self.step(robot_state, np.zeros(64, dtype=np.float64))

    def log_state_only(self, robot_state: RobotState) -> MotorCommand:
        """Log robot state to the StateLogger without running ONNX inference.

        Returns the default PD-hold motor command (stand-still at
        ``default_angles``).  Use this during a pre-stabilisation phase to
        pre-fill the history buffer with real, stable data so that the first
        CONTROL tick sees plausible history instead of zero-padded frames.

        The state is logged at the same throttled rate as CONTROL ticks
        (controlled by :meth:`set_env_step_dt`).  Between CONTROL-rate ticks
        this is a no-op and returns the cached command — matching the
        ``LowCommandWriter`` behaviour.
        """
        self.last_tick_trace = None
        body_q_raw = robot_state.body_q
        body_dq_raw = robot_state.body_dq
        base_quat = robot_state.base_quat
        base_ang_vel = robot_state.base_ang_vel

        self._frame_index += 1

        if self._frame_index - self._last_inference_index >= self._inference_interval:
            # Log state at CONTROL rate (same throttling as _step_impl)
            body_q_r, body_dq_r = self._remap_joints(body_q_raw, body_dq_raw)
            self._state_logger.log_full_state(
                base_quat=base_quat,
                base_ang_vel=base_ang_vel,
                body_q=body_q_r,
                body_dq=body_dq_r,
                last_action=self._last_action,  # zeros during hold phase
            )
            # LogPostState with zero token (no token during hold)
            self._state_logger.log_post_state(np.zeros(64, dtype=np.float64))
            self._last_inference_index = self._frame_index

        return self._cached_motor_command

    def step(self, robot_state: RobotState, motion_token) -> MotorCommand:
        """Execute one CONTROL tick.

        *motion_token* must be a ``np.ndarray`` of shape ``(64,)`` (the
        latent token).
        """
        token = np.asarray(motion_token, dtype=np.float64)
        if token.shape != (64,):
            raise ValueError(f"SONIC motion_token must have shape (64,); got shape {token.shape}")
        if not np.all(np.isfinite(token)):
            raise ValueError("SONIC motion_token contains NaN or infinite values")
        return self._step_impl(
            robot_state.body_q,
            robot_state.body_dq,
            robot_state.base_quat,
            robot_state.base_ang_vel,
            robot_state.pelvis_z,
            token,
        )

    # ------------------------------------------------------------------
    # Internal CONTROL tick (matches C++ G1Deploy::Control() CONTROL case)
    # ------------------------------------------------------------------
    def _step_impl(
        self,
        body_q_raw: np.ndarray,
        body_dq_raw: np.ndarray,
        base_quat: np.ndarray,
        base_ang_vel: np.ndarray,
        pelvis_z: float | None,
        token: np.ndarray,
    ) -> MotorCommand:
        self.last_tick_trace = None
        self._frame_index += 1

        # ---- CONTROL tick (throttled to control_dt, matches C++ 50 Hz) ----
        # State logging, observation gathering, post-state logging, and ONNX
        # inference all happen inside the same CONTROL tick — exactly like the
        # C++ G1Deploy::Control() flow.  Between CONTROL ticks we return the
        # cached MotorCommand without touching the StateLogger (C++
        # LowCommandWriter behaviour).
        if self._frame_index - self._last_inference_index >= self._inference_interval:
            # ---- 1. GatherRobotStateToLogger ----
            body_q_r, body_dq_r = self._remap_joints(body_q_raw, body_dq_raw)
            self._state_logger.log_full_state(
                base_quat=base_quat,
                base_ang_vel=base_ang_vel,
                body_q=body_q_r,
                body_dq=body_dq_r,
                last_action=self._last_action,
            )

            # ---- 2. Store token ----
            self._token_data = token.copy()

            # ---- 3. GatherObservations ----
            set_shared_state(self._state_logger, self._token_data)
            self._obs_buffer.fill(0.0)

            for name, offset, _ in self._obs_layout:
                gatherer = get_gatherer(name)
                ok = gatherer(self._obs_buffer, offset)
                if not ok:
                    raise RuntimeError(
                        f"Observation gatherer '{name}' failed at tick {self._frame_index}"
                    )

            # ---- 4. LogPostState ----
            self._state_logger.log_post_state(token.copy())

            # ---- 5. CreatePolicyCommand ----
            obs_f32 = self._obs_buffer.astype(np.float32, copy=False).reshape(1, -1)
            action = self._policy.infer(obs_f32)  # shape (29,) in isaaclab order

            # Save for next CONTROL tick's history.
            self._last_action = action.copy()

            # Compute q_target in MuJoCo (hardware) order.
            # C++: q_target[i] = default_angles[i]
            #                    + floatarr[isaaclab_to_mujoco[i]] * g1_action_scale[i]
            q_target = np.zeros(_NUM_JOINTS, dtype=np.float64)
            for i in range(_NUM_JOINTS):
                isaac_idx = isaaclab_to_mujoco[i]
                q_target[i] = default_angles[i] + action[isaac_idx] * g1_action_scale[i]

            self.last_tick_trace = {
                "token_state": token.copy(),
                "decoder_obs": self._obs_buffer.copy(),
                "decoder_raw_action": action.copy(),
                "decoder_target_action": q_target.copy(),
            }

            # ---- DIAGNOSTIC: first 20 ticks every tick, then every 50 ticks ----
            _DIAG_TICKS_DENSE = 20
            _DIAG_INTERVAL = 50
            _pelvis_z = None if pelvis_z is None else float(pelvis_z)
            _pelvis_z_low = _pelvis_z is not None and np.isfinite(_pelvis_z) and _pelvis_z < 0.7
            _should_diag = (
                self._control_tick_count < _DIAG_TICKS_DENSE
                or self._control_tick_count % _DIAG_INTERVAL == 0
                or _pelvis_z_low
            )
            if _should_diag:
                _log_sz = self._state_logger.size
                _q_err = q_target - default_angles
                _q_err_norm = float(np.linalg.norm(_q_err))
                _q_err_max = float(np.max(np.abs(_q_err)))
                _action_norm = float(np.linalg.norm(action))
                _action_max = float(np.max(np.abs(action)))

                # Per-observation-component stats
                _obs_parts = {}
                for name, off, dim in self._obs_layout:
                    chunk = self._obs_buffer[off : off + dim]
                    _obs_parts[name] = (
                        float(np.mean(np.abs(chunk))),
                        float(np.max(np.abs(chunk))),
                        float(np.count_nonzero(chunk)) / max(dim, 1),
                    )

                # Token first 4 dims
                _tk_head = self._token_data[:4].tolist() if self._token_data is not None else []
                _pelvis_z_str = "n/a" if _pelvis_z is None else f"{_pelvis_z:.3f}"

                _obs_parts_str = " | ".join(
                    f"{n}: μ={m:.3f} max={x:.3f} nz={nz:.1%}"
                    for n, (m, x, nz) in _obs_parts.items()
                )

                print(
                    f"[SONIC][DEBUG] tick={self._frame_index} "
                    f"ctrl_tick={self._control_tick_count} "
                    f"pelvis_z={_pelvis_z_str} "
                    f"logger_size={_log_sz} "
                    f"token_head={_tk_head} "
                    f"action_norm={_action_norm:.4f} action_max={_action_max:.4f} "
                    f"q_err_norm={_q_err_norm:.4f} q_err_max={_q_err_max:.4f} "
                    f"obs=[{_obs_parts_str}]",
                    flush=True,
                )

            # Increment CONTROL tick counter (for diagnostics — counts inference ticks only).
            self._control_tick_count += 1

            self._cached_motor_command = MotorCommand(
                q_target=q_target,
                dq_target=np.zeros(_NUM_JOINTS, dtype=np.float64),
                kp=kps.copy(),
                kd=kds.copy(),
                tau_ff=np.zeros(_NUM_JOINTS, dtype=np.float64),
            )
            self._last_inference_index = self._frame_index
        return self._cached_motor_command
