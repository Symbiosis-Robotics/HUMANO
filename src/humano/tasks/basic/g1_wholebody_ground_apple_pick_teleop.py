from __future__ import annotations

from copy import deepcopy
from typing import Any

from humano.core.actor import ObjectActor
from humano.dr import LanguageDRCfg
from humano.tasks.basic.g1_wholebody_table_apple_pick_teleop import (
    G1WholebodyTableApplePickTaskTeleop,
)
from humano.tasks.registry import TaskRegistry


@TaskRegistry.register("g1_wholebody_ground_apple_pick_teleop")
class G1WholebodyGroundApplePickTaskTeleop(G1WholebodyTableApplePickTaskTeleop):
    uid: str = "g1_wholebody_ground_apple_pick_teleop"
    label: str = "G1 TELEOP Ground Apple Pick"
    description: str = "A task where the G1 robot must pick up an apple on the ground."

    metadata: dict[str, Any] = deepcopy(G1WholebodyTableApplePickTaskTeleop.metadata)
    dr_cfgs = deepcopy(G1WholebodyTableApplePickTaskTeleop.dr_cfgs)
    dr_cfgs.pop("scene", None)
    dr_cfgs["language"] = LanguageDRCfg(
        instructions=[
            "pick up the apple on the ground",
        ]
    )

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> None:
        super().reset(seed, options)
        if isinstance(self._target, ObjectActor):
            stable_pose = self._target.asset.stable_poses[0]
            self._target.pose.position[2] = float(stable_pose[2])
            self._init_target_height = None
