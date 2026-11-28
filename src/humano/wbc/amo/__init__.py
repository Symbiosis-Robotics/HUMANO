"""SIMPLE AMO inference in the HUMANO motor-command interface."""

from pathlib import Path

import numpy as np
import torch
from transforms3d.euler import quat2euler
from transforms3d.quaternions import qinverse, qmult

from humano.wbc.base import MotorCommand, RobotState, WholeBodyController


def _bilateral(parts):
    return [f"{side}_{part}_joint" for side in ("left", "right") for part in parts]


_LEGS = _bilateral(("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll"))
_WAIST = ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
_ARMS = _bilateral(
    (
        "shoulder_pitch",
        "shoulder_roll",
        "shoulder_yaw",
        "elbow",
        "wrist_roll",
        "wrist_pitch",
        "wrist_yaw",
    )
)
_BODY = _LEGS + _WAIST + _ARMS
_KP = [150, 150, 150, 300, 80, 20] * 2 + [400] * 3 + [80, 80, 40, 60, 40, 40, 40] * 2
_KD = [2, 2, 2, 4, 2, 1] * 2 + [15] * 3 + [2, 2, 1, 1, 1, 1, 1] * 2


class AmoController(WholeBodyController):
    """Advance AMO once for each 20 ms control tick."""

    control_dt = 0.02

    def __init__(self, weights_dir, joint_names, device="cuda:0"):
        if len(joint_names) != 29 or set(joint_names) != set(_BODY):
            raise ValueError("AMO requires the 29 named G1 body joints")
        names = list(joint_names)
        self.observed = [names.index(j) for j in _LEGS + _WAIST + _ARMS[:4] + _ARMS[7:11]]
        self.output_order = [_BODY.index(j) for j in names]
        self.device = device
        directory = Path(weights_dir)
        self.policy = torch.jit.load(str(directory / "amo_jit.pt"), map_location=device).eval()
        self.adapter = torch.jit.load(str(directory / "adapter_jit.pt"), map_location=device).eval()
        stats = torch.load(
            directory / "adapter_norm_stats.pt", map_location="cpu", weights_only=False
        )
        self.stats = {
            k: self._tensor(stats[k])
            for k in ("input_mean", "input_std", "output_mean", "output_std")
        }
        self.reset()

    def _tensor(self, array):
        return torch.as_tensor(array, dtype=torch.float32, device=self.device).reshape(1, -1)

    def reset(self):
        self.initial_quat = None
        self.history = np.zeros((25, 93), dtype=np.float32)
        self.last_action = np.zeros(23, dtype=np.float32)
        self.phase = np.array([0.25, 0.25])
        self.standing = True

    @torch.inference_mode()
    def step(self, robot_state: RobotState, motion_signal) -> MotorCommand:
        """Consume [vx, yaw, vy, height offset, torso yaw/pitch/roll, turn, L7, R7]."""
        command = np.asarray(motion_signal, dtype=np.float32)
        if command.shape != (22,) or not np.isfinite(command).all():
            raise ValueError("AMO expects 8 finite locomotion commands and 14 finite arm targets")
        vx, yaw, vy, height_delta, torso_yaw, pitch, roll, turn = command[:8]
        height = 0.75 + height_delta
        q = np.asarray(robot_state.body_q, dtype=np.float32)[self.observed]
        dq = np.asarray(robot_state.body_dq, dtype=np.float32)[self.observed]
        quat = np.asarray(robot_state.base_quat, dtype=np.float32)
        if self.initial_quat is None:
            self.initial_quat = quat.copy()
        rpy = quat2euler(qmult(qinverse(self.initial_quat), quat))
        yaw_error = 0.0 if self.standing else (rpy[2] - yaw + np.pi) % (2 * np.pi) - np.pi
        arms = q[15:]
        adapter_in = self._tensor(np.r_[height, torso_yaw, pitch, roll, arms])
        normalized = (adapter_in - self.stats["input_mean"]) / (self.stats["input_std"] + 1e-8)
        adapter_out = (
            self.adapter(normalized) * self.stats["output_std"] + self.stats["output_mean"]
        )
        proprio = np.r_[
            np.asarray(robot_state.base_ang_vel) * 0.25,
            rpy[:2],
            np.sin(yaw_error),
            np.cos(yaw_error),
            q,
            dq * 0.05,
            self.last_action,
            np.sin(self.phase * 2 * np.pi),
            adapter_out.cpu().numpy().ravel(),
        ]
        demo = np.r_[arms, vx, vy, 0, torso_yaw, pitch, roll, [height] * 3]
        # SIMPLE supplies prior ticks to the first history and includes this tick in the second.
        obs = self._tensor(np.r_[proprio, demo, np.zeros(3), self.history[-10:].ravel()])
        self.history = np.concatenate((self.history[1:], proprio.astype(np.float32)[None]))
        raw = self.policy(obs, self._tensor(self.history)).cpu().numpy().reshape(15).clip(-40, 40)
        self.last_action = np.r_[raw, arms / 0.25]
        self.standing = abs(vx) < 0.1 and abs(vy) < 0.1 and turn < 0.5
        self.phase = (self.phase + self.control_dt * 1.3) % 1
        near_quarter = np.abs(self.phase - 0.25) < 0.05
        if self.standing and near_quarter.any():
            self.phase[:] = 0.25
        elif not self.standing and near_quarter.all():
            self.phase[:] = (0.25, 0.75)
        return MotorCommand(
            q_target=np.r_[raw * 0.25, command[8:]][self.output_order],
            kp=np.asarray(_KP)[self.output_order],
            kd=np.asarray(_KD)[self.output_order],
        )
