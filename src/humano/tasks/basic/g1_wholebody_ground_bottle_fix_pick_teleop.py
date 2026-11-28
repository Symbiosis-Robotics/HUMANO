from __future__ import annotations

from copy import deepcopy

from humano.assets import AssetManager
from humano.core.actor import Actor, ObjectActor
from humano.dr import TargetDRCfg
from humano.tasks.basic.g1_wholebody_ground_bottle_pick_teleop import (
    G1WholebodyGroundBottlePickTaskTeleop,
)
from humano.tasks.registry import TaskRegistry


@TaskRegistry.register("g1_wholebody_ground_bottle_fix_pick_teleop")
class G1WholebodyGroundBottleFixPickTaskTeleop(G1WholebodyGroundBottlePickTaskTeleop):
    uid: str = "g1_wholebody_ground_bottle_fix_pick_teleop"
    label: str = "G1 TELEOP Ground Bottle Fix Pick"
    description: str = "Pick up the collision-corrected bottle on the ground."

    metadata = deepcopy(G1WholebodyGroundBottlePickTaskTeleop.metadata)
    dr_cfgs = deepcopy(G1WholebodyGroundBottlePickTaskTeleop.dr_cfgs)
    dr_cfgs["target"] = TargetDRCfg(asset_id="objects:bottle/bottle_fix")

    def preload_objects(self) -> list[Actor]:
        asset = AssetManager.get("objects").load("bottle/bottle_fix")
        return [ObjectActor(asset=asset)]
