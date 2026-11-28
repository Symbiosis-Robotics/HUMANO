from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
import transforms3d as t3d

from humano.core.actor import Actor, ObjectActor
from humano.dr import DistractorDRCfg, LanguageDRCfg, SpatialDRCfg
from humano.dr.types import Box
from humano.tasks.advanced.g1_wholebody_table_book_to_shelf_teleop import (
    _BOOK_PLACE_ORIENTATION,
    _SHELF_LOCAL_INSIDE_XY_HIGH,
    _SHELF_LOCAL_INSIDE_XY_LOW,
    _SHELF_LOCAL_LAYER_Z_RANGES,
    G1WholebodyTableBookToShelfTaskTeleop,
)
from humano.tasks.basic.initial_positions import resolve_book_shelf_layer_place_offset
from humano.tasks.condition.g1_wholebody_condition_tasks_teleop import (
    _InitialConditionMixin,
)
from humano.tasks.registry import TaskRegistry

_OCCUPANT_LOAD_ID = "Book_1788883705_Expectations/Book_1788883705_Expectations"
_LAYER_BY_CONDITION = {
    "lower_occupied": "layer2",
    "upper_occupied": "layer3",
}
_OCCUPANT_LOCAL_Y = 0.08
_TARGET_LOCAL_Y = -0.11


@TaskRegistry.register("g1_wholebody_condition_book_to_occupied_shelf_layer_teleop")
class G1WholebodyConditionBookToOccupiedShelfLayerTaskTeleop(
    _InitialConditionMixin, G1WholebodyTableBookToShelfTaskTeleop
):
    uid = "g1_wholebody_condition_book_to_occupied_shelf_layer_teleop"
    label = "G1 TELEOP Condition Book To Occupied Shelf Layer"
    description = "A task where the G1 robot must put the table book in shelf layer 2 when that layer contains a book; put it in shelf layer 3 when that layer contains a book."
    default_instruction = "put the table book in shelf layer 2 when that layer contains a book; put it in shelf layer 3 when that layer contains a book"
    initial_conditions = ("lower_occupied", "upper_occupied")
    metadata: dict[str, Any] = deepcopy(G1WholebodyTableBookToShelfTaskTeleop.metadata)
    dr_cfgs = deepcopy(G1WholebodyTableBookToShelfTaskTeleop.dr_cfgs)
    dr_cfgs["language"] = LanguageDRCfg(instructions=[default_instruction])
    dr_cfgs["distractors"] = DistractorDRCfg(
        res_id="objects",
        number_of_distractors=1,
        include=[_OCCUPANT_LOAD_ID],
        allow_duplicates=False,
    )
    spatial_cfg = dr_cfgs["spatial"]
    assert isinstance(spatial_cfg, SpatialDRCfg)
    spatial_cfg.distractors_region = Box(low=[0.05, 0.0], high=[0.05, 0.0])
    spatial_cfg.obj_surface_map = {
        "target": "table",
        "container": "ground",
        "distractor": "ground",
    }

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, initial_position="layer1", **kwargs)
        self._occupant: Actor | None = None

    @property
    def occupant(self) -> Actor:
        assert self._occupant is not None, "call reset() first"
        return self._occupant

    def _configure_condition_after_reset(self) -> None:
        layer = _LAYER_BY_CONDITION[self.selected_initial_condition]
        offset = np.asarray(resolve_book_shelf_layer_place_offset(layer), dtype=np.float64)
        self._shelf_layer_name = layer
        self._shelf_local_place_offset = offset.copy()
        self._shelf_local_place_offset[1] = _TARGET_LOCAL_Y
        self._occupant = self.layout.actors["distractor_0"]

        shelf_rotation = t3d.quaternions.quat2mat(self.shelf.pose.quaternion)
        occupant_offset = offset.copy()
        occupant_offset[1] = _OCCUPANT_LOCAL_Y
        self.occupant.pose.position = (
            np.asarray(self.shelf.pose.position, dtype=np.float64)
            + shelf_rotation @ occupant_offset
        ).tolist()
        if isinstance(self.occupant, ObjectActor):
            self.occupant.pose.quaternion = [float(value) for value in _BOOK_PLACE_ORIENTATION]
        self.reward = 0.0

    def state_dict(self) -> dict[str, Any]:
        state = super().state_dict()
        state["occupant_uid"] = self.occupant.uid if self._occupant else None
        return state

    def check_object_in_shelf(self, info=None, *args, **kwargs) -> bool:
        if not super().check_object_in_shelf(info, *args, **kwargs):
            return False
        if info is None or "target" not in info or "distractor_0" not in info:
            return False
        target_position = np.asarray(info["target"][:3], dtype=np.float64)
        occupant_position = np.asarray(info["distractor_0"][:3], dtype=np.float64)
        shelf_position = np.asarray(self.shelf.pose.position, dtype=np.float64)
        shelf_rotation = t3d.quaternions.quat2mat(self.shelf.pose.quaternion)
        target_local = (target_position - shelf_position) @ shelf_rotation
        occupant_local = (occupant_position - shelf_position) @ shelf_rotation
        layer = _LAYER_BY_CONDITION[self.selected_initial_condition]
        layer_index = int(layer.removeprefix("layer")) - 1
        layer_low, layer_high = _SHELF_LOCAL_LAYER_Z_RANGES[layer_index]
        target_layer = layer_low <= float(target_local[2]) <= layer_high
        occupant_layer = layer_low <= float(occupant_local[2]) <= layer_high
        occupant_inside = bool(
            np.all(occupant_local[:2] >= _SHELF_LOCAL_INSIDE_XY_LOW)
            and np.all(occupant_local[:2] <= _SHELF_LOCAL_INSIDE_XY_HIGH)
        )
        return bool(
            self.check_book_upright(info) and target_layer and occupant_layer and occupant_inside
        )
