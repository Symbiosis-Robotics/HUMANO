"""
Base interface for whole-body controllers.

Defines standardised dataclasses for robot state input (RobotState) and motor command
output (MotorCommand), plus the WholeBodyController abstract base class that all
wbc/ implementations must subclass.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------------------
# Internal helpers — not exported
# ---------------------------------------------------------------------------


class _MotorCmdEntry:
    """Single motor command entry exposing the duck-type interface expected by
    :meth:`G1Sonic.compute_body_torques` and :meth:`G1Sonic.compute_hand_torques`.

    Mirrors the DDS ``MotorCmd`` struct: ``.q``, ``.dq``, ``.kp``, ``.kd``, ``.tau``.
    Internal — callers should use :class:`MotorCommand` or
    :meth:`MotorCommand.hand_command` instead.
    """

    __slots__ = ("q", "dq", "kp", "kd", "tau")

    def __init__(self, q: float, dq: float, kp: float, kd: float, tau: float) -> None:
        self.q = float(q)
        self.dq = float(dq)
        self.kp = float(kp)
        self.kd = float(kd)
        self.tau = float(tau)


# ---------------------------------------------------------------------------
# Public base types
# ---------------------------------------------------------------------------


@dataclass
class RobotState:
    """Standardised robot proprioceptive state for one control tick.

    All arrays use MuJoCo / hardware joint ordering (29-DOF G1).
    """

    body_q: np.ndarray  # [29] joint positions (rad)
    body_dq: np.ndarray  # [29] joint velocities (rad/s)
    base_quat: np.ndarray  # [4] floating-base quaternion (w, x, y, z)
    base_ang_vel: np.ndarray  # [3] base angular velocity (rad/s)
    pelvis_z: float | None = None  # optional floating-base height (m)
    base_accel: np.ndarray | None = None  # [3] optional base linear acceleration


@dataclass
class MotorCommand:
    """Standardised motor command output for env.step(action).

    Can be passed directly as the ``low_cmd`` argument to
    ``ActionCmd("wbc_torque")`` — the :attr:`motor_cmd` property
    exposes the DDS-compatible entry list that
    :meth:`G1Sonic.compute_body_torques` expects.

    Hand commands (7-DOF each) are created via the :meth:`hand_command`
    static factory.
    """

    q_target: np.ndarray = field(
        default_factory=lambda: np.zeros(29, dtype=np.float64)
    )  # [29] target joint positions
    dq_target: np.ndarray = field(
        default_factory=lambda: np.zeros(29, dtype=np.float64)
    )  # [29] target joint velocities
    kp: np.ndarray = field(
        default_factory=lambda: np.zeros(29, dtype=np.float64)
    )  # [29] position gains
    kd: np.ndarray = field(
        default_factory=lambda: np.zeros(29, dtype=np.float64)
    )  # [29] velocity gains
    tau_ff: np.ndarray = field(
        default_factory=lambda: np.zeros(29, dtype=np.float64)
    )  # [29] feed-forward torques

    # ------------------------------------------------------------------
    # DDS-compatible accessors (used internally by G1Sonic torque math)
    # ------------------------------------------------------------------

    @property
    def motor_cmd(self) -> list[_MotorCmdEntry]:
        """DDS-compatible motor command entries.

        Each element exposes ``.q`` / ``.dq`` / ``.kp`` / ``.kd`` / ``.tau``
        so this ``MotorCommand`` can be used wherever the sim expects a
        DDS ``LowCmd_``.
        """
        n = len(self.q_target)
        return [
            _MotorCmdEntry(
                float(self.q_target[i]),
                float(self.dq_target[i]),
                float(self.kp[i]),
                float(self.kd[i]),
                float(self.tau_ff[i]),
            )
            for i in range(n)
        ]

    @staticmethod
    def hand_command(
        joints: np.ndarray | None = None,
        *,
        kp: float = 1.5,
        kd: float = 0.1,
    ) -> _HandCmd:
        """Create a DDS-compatible hand command (7-DOF).

        The returned object exposes ``.motor_cmd[i].q/kp/kd/dq/tau``
        and can be used as ``left_hand_cmd`` / ``right_hand_cmd`` in
        ``ActionCmd("wbc_torque")``.
        """
        return _HandCmd._from_joints(joints, kp, kd)


class WholeBodyController(ABC):
    """Base interface for whole-body controllers.

    Each implementation converts an external motion signal into motor commands
    while maintaining internal state (history buffer) for temporal consistency.

    Lifecycle per episode::

        controller.reset()
        for each step:
            cmd = controller.step(robot_state, motion_signal)
            env.step(cmd)
    """

    @abstractmethod
    def reset(self) -> None:
        """Reset internal state at the start of a new episode."""
        ...

    @abstractmethod
    def step(self, robot_state: RobotState, motion_signal) -> MotorCommand:
        """Execute one control tick.

        Args:
            robot_state:
                Current robot state from MuJoCo / env.
            motion_signal:
                External motion input.  Type depends on implementation:
                  * SONIC: ``np.ndarray`` shape (64,) — latent token
                  * future: SMPL pose, trajectory waypoint, etc.

        Returns:
            MotorCommand ready for env.step(action).
        """
        ...

    def warmup(self, robot_state: RobotState, motion_signal) -> MotorCommand | None:
        """Optional bootstrap step — fill the history buffer before the main loop.

        The default implementation logs state and returns ``None`` (no command).
        Override if the controller needs an explicit multi-step warmup phase.
        """
        return None


# ---------------------------------------------------------------------------
# Internal hand-command helper — only used via MotorCommand.hand_command()
# ---------------------------------------------------------------------------


class _HandCmd:
    """Internal DDS ``HandCmd_``-compatible wrapper.

    Exposes ``.motor_cmd`` (list of :class:`_MotorCmdEntry`) so it can be
    passed as ``left_hand_cmd`` / ``right_hand_cmd`` to
    ``ActionCmd("wbc_torque")``.  Construct via :meth:`MotorCommand.hand_command`.
    """

    def __init__(self, motor_cmd: list[_MotorCmdEntry]) -> None:
        self.motor_cmd = motor_cmd

    @staticmethod
    def _from_joints(
        joints: np.ndarray | None,
        kp: float = 1.5,
        kd: float = 0.1,
    ) -> _HandCmd:
        """Build from a 7-DOF joint position array.

        Fewer than 7 joints are zero-padded; more than 7 are truncated.
        """
        if joints is None:
            joints = np.zeros(7, dtype=np.float32)
        raw = np.asarray(joints, dtype=np.float32).ravel()
        padded = np.zeros(7, dtype=np.float32)
        n = min(len(raw), 7)
        padded[:n] = raw[:n]
        return _HandCmd([_MotorCmdEntry(float(padded[i]), 0.0, kp, kd, 0.0) for i in range(7)])
