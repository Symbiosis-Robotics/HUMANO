"""
HUMANO: Humanoid Benchmark and Learning Framework

Originally derived from the SIMPLE project.
Copyright (c) 2025 Songlin Wei and Contributors
Copyright (c) 2026 HUMANO Contributors
Licensed under the terms in LICENSE file.
"""

from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import transforms3d as t3d

from humano.core.randomizer import Randomizer, RandomizerCfg
from humano.sensors.config import CameraCfg

from .types import Box


class CameraDR(Randomizer):
    def __init__(self, cfg: "CameraDRCfg") -> None:
        super().__init__(cfg)
        self._rng = np.random.default_rng(cfg.seed)
        for name in ("translation", "rotation_deg"):
            bounds = getattr(cfg, name)
            if bounds is not None:
                low, high = np.asarray(bounds.low), np.asarray(bounds.high)
                if (
                    low.shape != (3,)
                    or high.shape != (3,)
                    or not np.all(np.isfinite([low, high]))
                    or np.any(low > high)
                ):
                    raise ValueError(f"{name} requires three finite, ordered per-axis bounds")

    def reseed(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed)

    def __call__(self, split: str, cam_cfg: CameraCfg) -> CameraCfg:
        """Compose an episode-fixed offset in nominal camera axes (x forward, y left, z up).

        Rotations use static XYZ roll/pitch/yaw in degrees; quaternions are wxyz.
        Task.reset must resolve the nominal mounting quaternion before this call.
        DRManager clears the cached offsets at episode reset and restores them on replay.
        Explicit state for the actual uid takes priority over disabled or stale configs.
        """
        nominal = deepcopy(cam_cfg)
        state = self._inner_state or {}
        if cam_cfg.uid not in state:
            if self.cfg.cam_id is not None and self.cfg.cam_id != cam_cfg.uid:
                return nominal
            if self.cfg.translation is None and self.cfg.rotation_deg is None:
                return nominal
            offsets = {
                name: (
                    self._rng.uniform(bounds.low, bounds.high).tolist()
                    if (bounds := getattr(self.cfg, name)) is not None
                    else [0.0, 0.0, 0.0]
                )
                for name in ("translation", "rotation_deg")
            }
            state = {**state, cam_cfg.uid: offsets}
            self._inner_state = state
        offsets = state[cam_cfg.uid]
        translation = np.asarray(offsets["translation"], dtype=np.float64)
        rotation = np.asarray(offsets["rotation_deg"], dtype=np.float64)
        if not np.any(translation) and not np.any(rotation):
            return nominal
        if nominal.quaternion is None:
            raise ValueError("Resolve nominal camera quaternion in Task.reset before CameraDR")

        nominal_q = np.asarray(nominal.quaternion, dtype=np.float64)
        offset_q = t3d.euler.euler2quat(*np.deg2rad(rotation), axes="sxyz")
        pose = {
            **nominal.pose,
            "position": (
                np.asarray(nominal.position) + t3d.quaternions.quat2mat(nominal_q) @ translation
            ).tolist(),
            "quaternion": t3d.quaternions.qmult(nominal_q, offset_q).tolist(),
        }
        return replace(nominal, pose=pose)

    def state_dict(self) -> dict[str, Any]:
        return deepcopy(self._inner_state or {})

    def load_state_dict(self, state_dict: dict[str, Any] | None) -> None:
        self._inner_state = deepcopy(state_dict)


@dataclass
class CameraDRCfg(RandomizerCfg):
    cam_id: str | None = None
    translation: Box | None = None  # metres in nominal camera axes
    rotation_deg: Box | None = None  # roll, pitch, yaw; static XYZ
    seed: int | None = None
    randmizer_class: type[Randomizer] = CameraDR

    @classmethod
    def oasis_nico(cls, *, seed: int | None = None) -> "CameraDRCfg":
        """OASIS/NICO viewpoint preset, enabled explicitly for the actual head sensor uid."""
        return cls(
            cam_id="Realsense_D435i",
            translation=Box(low=[-0.01] * 3, high=[0.01] * 3),
            rotation_deg=Box(low=[-1.5, -15.0, -10.0], high=[1.5, 5.0, 10.0]),
            seed=seed,
        )
