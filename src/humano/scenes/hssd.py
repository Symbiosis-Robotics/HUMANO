"""
HUMANO: Humanoid Benchmark and Learning Framework

Originally derived from the SIMPLE project.
Copyright (c) 2025 Songlin Wei and Contributors
Copyright (c) 2026 HUMANO Contributors
Licensed under the terms in LICENSE file.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from humano.core.scene import Scene, TabletopScene
from humano.scenes.scene_manager import SceneManager

if TYPE_CHECKING:
    from humano.core import Asset, Scene

import os
import random

import numpy as np
import transforms3d as t3d
import yaml
from dotenv import load_dotenv

from humano.utils import resolve_data_path

load_dotenv()


def table_top_center_and_yaw(
    position: list[float], quaternion: list[float], size: list[float]
) -> tuple[np.ndarray, float]:
    """Return a box table's top-center position and world-frame yaw."""
    rotation = t3d.quaternions.quat2mat(np.asarray(quaternion, dtype=np.float64))
    top_center = np.asarray(position, dtype=np.float64) + rotation @ np.array(
        [0.0, 0.0, 0.5 * float(size[2])], dtype=np.float64
    )
    yaw = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
    return top_center, yaw


class HssdSuite(TabletopScene):
    name: str
    center_offset: list[float]
    center_orientation: list[float]

    def __init__(self, conf) -> None:
        self.uid = f"hssd:{conf['uid']}"
        self.conf = conf
        self.name = conf["name"]
        # self.table = table
        self.data_dir = f"{conf['data_dir']}/{conf['name']}"

        self.middle()

    def set_table(self, table: Asset) -> None:
        self.table = table

    def set_table2(self, table2: Asset) -> None:
        self.table2 = table2

    def dr(self) -> HssdSuite:  # MOVE TO scene DR
        """randomize with reasonable range"""
        self.center_offset = np.random.uniform(
            self.conf["center_offset_limit_up"], self.conf["center_offset_limit_down"]
        )
        self.center_orientation = np.random.uniform(
            self.conf["center_orientation_limit_up"], self.conf["center_orientation_limit_down"]
        )
        return self

    def middle(self) -> HssdSuite:
        self.center_offset = [
            (u + d) / 2.0
            for u, d in zip(
                self.conf["center_offset_limit_up"], self.conf["center_offset_limit_down"]
            )
        ]
        self.center_orientation = [
            (u + d) / 2.0
            for u, d in zip(
                self.conf["center_orientation_limit_up"], self.conf["center_orientation_limit_down"]
            )
        ]
        return self


@SceneManager.register("hssd")
class HssdSceneManager(SceneManager):
    def __init__(self) -> None:
        # load hssd scenes config
        config_file_path = resolve_data_path("scenes/hssd/config.yaml")
        with open(config_file_path) as f:
            try:
                self.hssd_scenes = yaml.safe_load(f)
            except yaml.YAMLError as exc:
                print(exc)

    def sample(self, exclude: list[str] | None = None) -> Scene:
        sampled = HssdSuite(random.choice(self.hssd_scenes))
        return sampled  # some dr ranges must be valid for this specific scene

    def load(self, scene_uid: str) -> Scene:
        """Load an asset by its ID."""
        if ":" in scene_uid:
            scene_uid = scene_uid.split(":")[1]

        hssd_scenes_dict = {s["uid"]: s for s in self.hssd_scenes}
        scene_name = hssd_scenes_dict[scene_uid]["name"]

        scene_dir = resolve_data_path(f"scenes/hssd/{scene_name}", auto_download=True)
        usd_path = os.path.join(scene_dir, f"{scene_name}.usd")
        return HssdSuite(hssd_scenes_dict[scene_uid])
