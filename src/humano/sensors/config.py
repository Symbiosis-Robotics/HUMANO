"""
HUMANO: Humanoid Benchmark and Learning Framework

Originally derived from the SIMPLE project.
Copyright (c) 2025 Songlin Wei and Contributors
Copyright (c) 2026 HUMANO Contributors
Licensed under the terms in LICENSE file.
"""

from dataclasses import dataclass

import numpy as np
from gymnasium import spaces

from humano.core.types import Pose

DEFAULT_HEAD_CAMERA_FOV = np.deg2rad(110)


@dataclass
class SensorCfg:
    uid: str
    mount: str
    # parent_link: str

    @property
    def observation_space(self) -> spaces.Space: ...


@dataclass
class CameraCfg(SensorCfg):
    # uid: str

    width: int
    height: int

    fov: float
    near: float
    far: float

    focal_length: float

    pose: dict

    @property
    def observation_space(self) -> spaces.Space:
        return spaces.Box(0, 255, shape=(self.height, self.width, 3), dtype=np.uint8)

    @property
    def resolution(self) -> tuple[int, int]:  # [W, H]
        return (self.width, self.height)

    @property
    def fy(self) -> float:
        fx = self.width / (2 * np.tan(self.fov / 2))
        fy = fx
        return fy

    @property
    def position(self) -> list[float]:
        if "position" in self.pose:
            return self.pose["position"]

    @property
    def quaternion(self) -> list[float]:
        if "quaternion" in self.pose:
            return self.pose["quaternion"]


@dataclass
class StereoCameraCfg(CameraCfg):
    baseline: float

    @property
    def observation_space(self) -> spaces.Space:
        return spaces.Dict(
            {
                f"left": spaces.Box(0, 255, shape=(self.height, self.width, 3), dtype=np.uint8),
                f"right": spaces.Box(0, 255, shape=(self.height, self.width, 3), dtype=np.uint8),
            }
        )
