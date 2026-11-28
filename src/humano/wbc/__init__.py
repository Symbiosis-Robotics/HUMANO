"""
Whole-body control (WBC) — in-process controllers for humanoid robots.

Each implementation converts an external motion signal (token, trajectory, velocity command,
SMPL pose, etc.) into motor commands, maintaining internal state for temporal consistency.
"""

from humano.wbc.base import MotorCommand, RobotState, WholeBodyController

__all__ = ["WholeBodyController", "RobotState", "MotorCommand"]
