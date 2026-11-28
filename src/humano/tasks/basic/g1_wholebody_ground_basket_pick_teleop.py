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
    apply_movement_layout,
    resolve_surface_scene_cfg,
)
from humano.tasks.simple.g1_wholebody_xmove_pick_teleop import G1WholebodyXMovePickTaskTeleop
from humano.tasks.registry import TaskRegistry

_BASKET_ASSET_ID = "picnic_basket/picnic_basket_009"
_BASKET_MUJOCO_MASS_KG = 0.35


@TaskRegistry.register("g1_wholebody_ground_basket_pick_teleop")
class G1WholebodyGroundBasketPickTaskTeleop(G1WholebodyGroundApplePickTaskTeleop):
    uid: str = "g1_wholebody_ground_basket_pick_teleop"
    label: str = "G1 TELEOP Ground Basket Pick"
    description: str = "A task where the G1 robot must pick up a basket on the ground."

    metadata: dict[str, Any] = deepcopy(G1WholebodyGroundApplePickTaskTeleop.metadata)
    dr_cfgs = deepcopy(G1WholebodyGroundApplePickTaskTeleop.dr_cfgs)
    dr_cfgs["language"] = LanguageDRCfg(
        instructions=[
            "pick up the basket on the ground",
        ]
    )
    dr_cfgs["target"] = TargetDRCfg(asset_id=f"objects:{_BASKET_ASSET_ID}")

    def preload_objects(self) -> list[Actor]:
        asset = AssetManager.get("objects").load(_BASKET_ASSET_ID)
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
                f"pick up the basket {surface_phrase}",
            ]
        )
        spatial_cfg = self.dr_cfgs["spatial"]
        assert isinstance(spatial_cfg, SpatialDRCfg)
        spatial_cfg.obj_surface_map = {"target": "table"} if scene_cfg is not None else None
        if movement:
            apply_movement_layout(self.dr_cfgs)
        G1WholebodyXMovePickTaskTeleop.__init__(self, *args, **kwargs)

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> None:
        G1WholebodyXMovePickTaskTeleop.reset(self, seed, options)
        if isinstance(self._target, ObjectActor):
            self._target.mujoco_mass = _BASKET_MUJOCO_MASS_KG
