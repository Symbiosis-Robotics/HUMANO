from __future__ import annotations

from copy import deepcopy
from typing import Any

from humano.assets import AssetManager
from humano.core.actor import Actor, ObjectActor
from humano.dr import (
    DistractorDRCfg,
    LanguageDRCfg,
    SpatialDRCfg,
    TabletopSceneDRCfg,
    TargetDRCfg,
)
from humano.dr.types import Box
from humano.tasks.basic.initial_positions import (
    apply_movement_layout,
    resolve_initial_position_region,
)
from humano.tasks.simple.g1_wholebody_xmove_pick_teleop import G1WholebodyXMovePickTaskTeleop
from humano.tasks.registry import TaskRegistry

_PLATE_ASSET_ID = "objects:plate/plate_3"


@TaskRegistry.register("g1_wholebody_table_plate_pick_teleop")
class G1WholebodyTablePlatePickTaskTeleop(G1WholebodyXMovePickTaskTeleop):
    uid: str = "g1_wholebody_table_plate_pick_teleop"
    label: str = "G1 TELEOP Table Plate Pick"
    description: str = (
        "A task where the G1 robot must move forward to pick up a large plate "
        "on the table with both hands."
    )

    metadata: dict[str, Any] = deepcopy(G1WholebodyXMovePickTaskTeleop.metadata)
    dr_cfgs = deepcopy(G1WholebodyXMovePickTaskTeleop.dr_cfgs)
    dr_cfgs["language"] = LanguageDRCfg(
        instructions=[
            "move forward to the table and pick up the large plate with both hands",
        ]
    )
    dr_cfgs["target"] = TargetDRCfg(asset_id=_PLATE_ASSET_ID)
    dr_cfgs["distractors"] = DistractorDRCfg(
        res_id="objects",
        number_of_distractors=0,
        allow_duplicates=False,
    )
    dr_cfgs["spatial"] = SpatialDRCfg(
        spatial_mode="fixed",
        robot_region=Box(low=[-0.95, 0.0, 0.0], high=[-0.95, 0.0, 0.0]),
        target_region=Box(low=[-0.275, -0.06], high=[-0.275, -0.06]),
        distractors_region=Box(low=[0.0, 0.0], high=[0.0, 0.0]),
        target_stable_indices=[0],
        target_rotate_z=Box(low=0.0, high=0.0),
        fixed_stable_pose_idx=0,
    )
    dr_cfgs["scene"] = TabletopSceneDRCfg(
        scene_mode="fixed",
        table_position=Box(low=[0.3, 0.0], high=[0.3, 0.0]),
        table_height=Box(low=0.65, high=0.65),
        room_choices=["hssd:scene0"],
        scene_manager="hssd",
    )

    def preload_objects(self) -> list[Actor]:
        asset = AssetManager.get("objects").load("plate/plate_3")
        return [ObjectActor(asset=asset)]

    def __init__(
        self,
        *args,
        initial_position: str | None = None,
        plate_initial_position: str | None = None,
        movement: bool = False,
        **kwargs,
    ) -> None:
        self.dr_cfgs = deepcopy(type(self).dr_cfgs)
        target_region = resolve_initial_position_region(
            initial_position=initial_position,
            legacy_initial_position=plate_initial_position,
            legacy_name="plate_initial_position",
        )
        if target_region is not None:
            spatial_cfg = self.dr_cfgs["spatial"]
            assert isinstance(spatial_cfg, SpatialDRCfg)
            spatial_cfg.target_region = target_region
        if movement:
            apply_movement_layout(self.dr_cfgs)
        super().__init__(*args, **kwargs)
