"""
HUMANO: Humanoid Benchmark and Learning Framework

Originally derived from the SIMPLE project.
Copyright (c) 2025 Songlin Wei and Contributors
Copyright (c) 2026 HUMANO Contributors
Licensed under the terms in LICENSE file.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Type

from humano.scenes.hssd import HssdSceneManager, HssdSuite

if TYPE_CHECKING:
    from humano.core import Asset, Scene
    from humano.core.scene import TabletopScene
    from humano.dr.types import Box
    from humano.scenes import SceneManager


# from humano.utils import wxyz2xyzw
import random
from dataclasses import dataclass

import numpy as np
import transforms3d as t3d

from humano.assets import AssetManager
from humano.core.randomizer import Randomizer, RandomizerCfg
from humano.scenes import SceneManager, ShowHouse


class SceneDR(Randomizer):
    def __init__(self, cfg: TabletopSceneDRCfg) -> None:
        super().__init__(cfg)


class TabletopSceneDR(SceneDR):
    def __init__(self, cfg: TabletopSceneDRCfg) -> None:
        super().__init__(cfg)
        self.scene_manager = None if cfg.scene_config else SceneManager.get(cfg.scene_manager)

    def state_dict(self) -> dict[str, Any]:
        return self._inner_state.to_dict() if self._inner_state is not None else {}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if self.cfg.scene_config:
            from copy import deepcopy
            from pathlib import Path

            from humano.scenes.indexed import (
                IndexedBackgroundScene,
                IndexedNavigationObstacle,
                IndexedSceneBounds,
                IndexedSceneTransform,
                IndexedVisualProfile,
            )

            self._inner_state = IndexedBackgroundScene(
                uid=state_dict["uid"],
                provider=state_dict["provider"],
                index_path=Path(state_dict["index_path"]),
                usd_path=Path(state_dict["usd_path"]),
                transform=IndexedSceneTransform.from_dict(state_dict["transform"]),
                bounds=IndexedSceneBounds.from_dict(state_dict["bounds"]),
                visual_profile=IndexedVisualProfile.from_dict(state_dict["visual_profile"]),
                immutable_obstacles=tuple(
                    IndexedNavigationObstacle.from_dict(item)
                    for item in state_dict["immutable_obstacles"]
                ),
                revision=state_dict["revision"],
                metadata=deepcopy(state_dict["metadata"]),
            )
            current_scene = self._configured_scene()
            current_profile = current_scene.visual_profile
            if "render_fill_lights" in current_scene.metadata:
                self._inner_state.metadata["render_fill_lights"] = current_scene.metadata["render_fill_lights"]
            self._inner_state.visual_profile.hidden_prim_paths = sorted(
                set(self._inner_state.visual_profile.hidden_prim_paths)
                | set(current_profile.hidden_prim_paths)
            )
            return
        scene_uid = state_dict["uid"].split(":")[1]
        table_cfg = state_dict["table"]
        self._inner_state = self.scene_manager[scene_uid]

        self._inner_state.center_offset = state_dict["center_offset"]
        self._inner_state.center_orientation = state_dict["center_orientation"]

        table = AssetManager.create(
            "primitive:box",
            size=table_cfg["size"],
            position=table_cfg["pose"]["position"],
            quaternion=table_cfg["pose"]["quaternion"],
        )
        self._inner_state.table = table

        if "table2" in state_dict and state_dict["table2"] is not None:
            table2_cfg = state_dict["table2"]
            table2 = AssetManager.create(
                "primitive:box",
                size=table2_cfg["size"],
                position=table2_cfg["pose"]["position"],
                quaternion=table2_cfg["pose"]["quaternion"],
            )
            self._inner_state.table2 = table2

    def _configured_scene(self):
        from humano.scenes.indexed import load_configured_background_scene

        return load_configured_background_scene(self.cfg.scene_config, self.cfg.scene_entry)

    def __call__(self, split: str, *args, **kwargs) -> TabletopScene:
        if self.cfg.scene_config:
            if self._inner_state is None:
                self._inner_state = self._configured_scene()
            return self._inner_state
        if self._inner_state is None:
            # scene randomization
            if self.cfg.scene_mode == "fixed":
                scene = self.scene_manager.load(self.cfg.room_choices[0])
            else:
                if isinstance(self.cfg.room_choices, list):
                    room_uid = random.choice(self.cfg.room_choices)
                    scene = self.scene_manager[room_uid]
                else:
                    scene = self.scene_manager.sample()

            # table randomization
            if isinstance(self.scene_manager, HssdSceneManager):
                assert isinstance(scene, HssdSuite)
                # table has fixed size in HSSD scenes
                # but table height can be overridden
                table_size = np.concatenate([scene.conf["table_size"][:2], [0.1]])

                table_position = [0.0, 0.0]  # default table position
                if self.cfg.table_position is not None:
                    table_position = self.cfg.table_position.middle()
                else:
                    table_position = [0.0, 0.0]
                if self.cfg.table_height is None:
                    table_height = scene.conf["table_size"][2]  # scene.conf["table_height"]
                else:
                    table_height = (
                        self.cfg.table_height.middle()
                        if self.cfg.scene_mode == "fixed"
                        else self.cfg.table_height.sample()
                    )
                rotation_z = 0.0
            else:
                if self.cfg.scene_mode == "fixed":
                    table_size = self.cfg.table_size.middle()
                    table_height = self.cfg.table_height.middle()
                    table_position = self.cfg.table_position.middle()
                    rotation_z = self.cfg.rotation_z.middle()
                else:
                    table_size = self.cfg.table_size.sample()
                    table_height = self.cfg.table_height.sample()
                    table_position = self.cfg.table_position.sample()
                    rotation_z = self.cfg.rotation_z.sample()

            # compute table height
            table_position.append(-0.5 * table_size[2] + table_height)
            quaternion = t3d.euler.euler2quat(0, 0, rotation_z).tolist()

            table = AssetManager.create(
                "primitive:box", size=table_size, position=table_position, quaternion=quaternion
            )

            # if not self.cfg.scene_mode == "fixed":
            #     scene.dr() # initialize dr params
            # else:
            #     scene.middle()

            # scene.table = table
            scene.set_table(table)  # FIXME

            if self.cfg.enable_table2:
                # table randomization
                if isinstance(self.scene_manager, HssdSceneManager):
                    assert isinstance(scene, HssdSuite)
                    # Prefer HSSD's authored second-table size when available, but
                    # allow task-local primitive tables for scenes that only define one table.
                    if "table2_size" in scene.conf:
                        table2_size = np.concatenate([scene.conf["table2_size"][:2], [0.1]])
                    elif self.cfg.table2_size is not None:
                        table2_size = self.cfg.table2_size.middle()
                    else:
                        raise KeyError(
                            "table2_size is required when enable_table2=True and the HSSD scene "
                            "does not define table2_size."
                        )
                    if self.cfg.table2_position is not None:
                        table2_position = self.cfg.table2_position.middle()
                    else:
                        table2_position = scene.conf["table2_position"][:2]
                    if self.cfg.table2_height is None:
                        table2_height = scene.conf["table2_size"][2]
                    else:
                        table2_height = (
                            self.cfg.table2_height.middle()
                            if self.cfg.scene_mode == "fixed"
                            else self.cfg.table2_height.sample()
                        )
                    if self.cfg.table2_rotation_z is None:
                        table2_rotation_z = 0.0
                    else:
                        table2_rotation_z = self.cfg.table2_rotation_z.sample()
                else:
                    if self.cfg.scene_mode == "fixed":
                        table2_size = self.cfg.table2_size.middle()
                        table2_height = self.cfg.table2_height.middle()
                        table2_position = self.cfg.table2_position.middle()
                        table2_rotation_z = self.cfg.table2_rotation_z.middle()
                    else:
                        table2_size = self.cfg.table2_size.sample()
                        table2_height = self.cfg.table2_height.sample()
                        table2_position = self.cfg.table2_position.sample()
                        table2_rotation_z = self.cfg.table2_rotation_z.sample()

                table2_position.append(-0.5 * table2_size[2] + table2_height)
                quaternion2 = t3d.euler.euler2quat(0, 0, table2_rotation_z).tolist()

                table2 = AssetManager.create(
                    "primitive:box",
                    size=table2_size,
                    position=table2_position,
                    quaternion=quaternion2,
                )

                scene.set_table2(table2)
        else:
            scene = self._inner_state
        # TODO scene = ShowHouse(table)
        return super()._transient(scene)

    def replicate_scene(
        self, split: str, table_size, table_position, table_orientation
    ) -> TabletopScene:
        table = AssetManager.create(
            "primitive:box", size=table_size, position=table_position, quaternion=table_orientation
        )

        if isinstance(self.cfg.room_choices, list):
            room_uid = random.choice(self.cfg.room_choices)
            scene = self.scene_manager[room_uid]
        else:
            scene = self.scene_manager.sample()

        # scene.table = table
        scene.set_table(table)  # FIXME

        # TODO scene = ShowHouse(table)
        return scene


@dataclass
class TabletopSceneDRCfg(RandomizerCfg):
    scene_config: str | None = None
    scene_entry: str | None = None
    scene_mode: str | None = None  # fixed, random
    table_size: Box | None = None
    table_position: Box | None = None
    table_height: Box | None = None
    rotation_z: Box | None = None
    room_choices: list | str | None = None
    scene_manager: str | None = None
    randmizer_class: type[Randomizer] = TabletopSceneDR

    enable_table2: bool = False
    table2_size: Box | None = None
    table2_position: Box | None = None
    table2_height: Box | None = None
    table2_rotation_z: Box | None = None

    def __post_init__(self):
        if self.scene_manager == "hssd":
            assert self.scene_mode == "fixed" or (
                self.table_size is None
                # and self.table_position is None
                # and self.table_height is None
                and self.rotation_z is None
            ), (
                "When using 'hssd' scene manager, table_size, table_position, table_height, and rotation_z are fixed."
            )
