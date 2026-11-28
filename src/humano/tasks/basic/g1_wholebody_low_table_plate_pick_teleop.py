from __future__ import annotations

from copy import deepcopy
from typing import Any

from humano.dr import LanguageDRCfg, TabletopSceneDRCfg
from humano.dr.types import Box
from humano.tasks.basic.g1_wholebody_table_plate_pick_teleop import (
    G1WholebodyTablePlatePickTaskTeleop,
)
from humano.tasks.registry import TaskRegistry


@TaskRegistry.register("g1_wholebody_low_table_plate_pick_teleop")
class G1WholebodyLowTablePlatePickTaskTeleop(G1WholebodyTablePlatePickTaskTeleop):
    uid: str = "g1_wholebody_low_table_plate_pick_teleop"
    label: str = "G1 TELEOP Low Table Plate Pick"
    description: str = "A task where the G1 robot must pick up a large plate from a low table."

    metadata: dict[str, Any] = deepcopy(G1WholebodyTablePlatePickTaskTeleop.metadata)
    dr_cfgs = deepcopy(G1WholebodyTablePlatePickTaskTeleop.dr_cfgs)
    dr_cfgs["language"] = LanguageDRCfg(
        instructions=[
            "pick up the large plate from the low table with both hands",
        ]
    )
    dr_cfgs["scene"] = TabletopSceneDRCfg(
        scene_mode="fixed",
        table_position=Box(low=[0.3, 0.0], high=[0.3, 0.0]),
        table_height=Box(low=0.35, high=0.35),
        room_choices=["hssd:scene0"],
        scene_manager="hssd",
    )
