from __future__ import annotations

from copy import deepcopy
from typing import Any

from humano.assets import AssetManager
from humano.core.actor import Actor, ObjectActor
from humano.dr import LanguageDRCfg, SpatialDRCfg, TargetDRCfg
from humano.tasks.basic.g1_wholebody_ground_apple_pick_teleop import (
    G1WholebodyGroundApplePickTaskTeleop,
)
from humano.tasks.basic.initial_positions import (
    MOVEMENT_GROUND_PICK_EXTRA_ROBOT_TARGET_DISTANCE_M,
    apply_movement_layout,
    resolve_surface_scene_cfg,
)
from humano.tasks.simple.g1_wholebody_xmove_pick_teleop import G1WholebodyXMovePickTaskTeleop
from humano.tasks.registry import TaskRegistry


@TaskRegistry.register("g1_wholebody_ground_box_pick_teleop")
class G1WholebodyGroundBoxPickTaskTeleop(G1WholebodyGroundApplePickTaskTeleop):
    uid: str = "g1_wholebody_ground_box_pick_teleop"
    label: str = "G1 TELEOP Ground Box Pick"
    description: str = "A task where the G1 robot must pick up a box on the ground."

    metadata: dict[str, Any] = deepcopy(G1WholebodyGroundApplePickTaskTeleop.metadata)
    dr_cfgs = deepcopy(G1WholebodyGroundApplePickTaskTeleop.dr_cfgs)
    dr_cfgs["language"] = LanguageDRCfg(
        instructions=[
            "pick up the box on the ground",
        ]
    )
    dr_cfgs["target"] = TargetDRCfg(asset_id="objects:box/box_007")

    def preload_objects(self) -> list[Actor]:
        asset = AssetManager.get("objects").load("box/box_007")
        return [ObjectActor(asset=asset)]

    def __init__(
        self,
        *args,
        initial_position: str | None = None,
        movement: bool = False,
        **kwargs,
    ) -> None:
        self.dr_cfgs = deepcopy(type(self).dr_cfgs)
        surface, scene_cfg = resolve_surface_scene_cfg(initial_position)
        if scene_cfg is None:
            self.dr_cfgs.pop("scene", None)
            surface_phrase = "on the ground"
        else:
            self.dr_cfgs["scene"] = scene_cfg
            surface_phrase = "on the low table" if surface == "low_table" else "on the high table"
        self.dr_cfgs["language"] = LanguageDRCfg(
            instructions=[
                f"pick up the box {surface_phrase}",
            ]
        )
        spatial_cfg = self.dr_cfgs["spatial"]
        assert isinstance(spatial_cfg, SpatialDRCfg)
        spatial_cfg.obj_surface_map = {"target": "table"} if scene_cfg is not None else None
        if movement:
            apply_movement_layout(
                self.dr_cfgs,
                extra_robot_target_distance_m=(MOVEMENT_GROUND_PICK_EXTRA_ROBOT_TARGET_DISTANCE_M),
            )
        G1WholebodyXMovePickTaskTeleop.__init__(self, *args, **kwargs)

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> None:
        G1WholebodyXMovePickTaskTeleop.reset(self, seed, options)
