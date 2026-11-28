"""
HUMANO: Humanoid Benchmark and Learning Framework

Originally derived from the SIMPLE project.
Copyright (c) 2025 Songlin Wei and Contributors
Copyright (c) 2026 HUMANO Contributors
Licensed under the terms in LICENSE file.
"""

from dataclasses import dataclass
from typing import Any, Dict, Type

from humano.assets import AssetManager
from humano.core.asset import Asset
from humano.core.randomizer import Randomizer, RandomizerCfg

# from typing import TYPE_CHECKING
# if TYPE_CHECKING:
#     from humano.dr.target import TargetDr


class TargetDR(Randomizer):
    # asset_id: str
    # object: "Asset"

    def __init__(self, cfg: "TargetDRCfg") -> None:
        super().__init__(cfg)
        self.res_id, self.obj_id = cfg.asset_id.split(":")

    def state_dict(self) -> dict[str, Any]:
        # state_dict = {}
        # for k, v in self._inner_state.items():
        #     state_dict[k] = v.to_dict()
        # return state_dict
        return self._inner_state.to_dict() if self._inner_state else {}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._inner_state = AssetManager.get(state_dict["res_id"]).load(state_dict["uid"])

    def __call__(self, split: str, **kwargs) -> Asset:

        # res_id, obj_id = self.asset_id.split(':')
        asset = AssetManager.get(self.res_id).load(self.obj_id)
        # return self.object

        # if self.rand_stable_pose:
        #     ...

        return super()._transient(asset)


@dataclass
class TargetDRCfg(RandomizerCfg):
    asset_id: str | None = None  # e.g., "res_id:obj_id"
    randmizer_class: type[Randomizer] = TargetDR

    # def __init__(self, asset_id) -> None:
    #     if asset_id is None or ':' not in asset_id:
    #         raise ValueError("Invalid asset_id format. Expected 'res_id:obj_id'.")
    #     self.res_id, self.obj_id = asset_id.split(':')
