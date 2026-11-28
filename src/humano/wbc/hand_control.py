"""DDS hand targets and feedback for policy evaluation."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

DEFAULT_HAND_KP = 1.5
DEFAULT_HAND_KD = 0.1


def dds_hand_feedback(proprio: dict, hand_type: str) -> dict:
    if hand_type != "brainco":
        return proprio
    return {
        **proprio,
        **{
            f"{side}_hand_{field}": proprio[f"{side}_hand_actuated_{field}"]
            for side in ("left", "right")
            for field in ("q", "dq")
        },
    }


def build_hand_cmd(
    q_target: np.ndarray,
    *,
    hand_dof: int,
    command_factory: Callable[[], Any],
    kp: float = DEFAULT_HAND_KP,
    kd: float = DEFAULT_HAND_KD,
) -> Any:
    """Build a fresh fixed-width DDS hand command from a hand-specific target."""
    targets = np.asarray(q_target, dtype=np.float32).reshape(-1)
    if targets.size != hand_dof or not np.all(np.isfinite(targets)):
        raise ValueError(f"Expected {hand_dof} finite hand targets, got {targets.size}")

    command = command_factory()
    if len(command.motor_cmd) < hand_dof:
        raise ValueError(f"Hand command has fewer than {hand_dof} motor slots")
    for index, target in enumerate(targets):
        motor = command.motor_cmd[index]
        motor.mode = index | (0x01 << 4)
        motor.q = float(target)
        motor.dq = 0.0
        motor.kp = kp
        motor.kd = kd
        motor.tau = 0.0
    return command
