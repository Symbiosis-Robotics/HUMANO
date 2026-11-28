"""Condition-aware HUMANO task registrations."""

from .g1_wholebody_condition_tasks_teleop import (
    G1WholebodyConditionAppleToDrawerTaskTeleop,
    G1WholebodyConditionAppleToFruitFreeBasketTaskTeleop,
    G1WholebodyConditionBasketToTableAppleToBasketTaskTeleop,
    G1WholebodyConditionBottleToEmptyBoxTaskTeleop,
    G1WholebodyConditionBreadToPlateOrTrayTaskTeleop,
    G1WholebodyConditionChairToTableAndSitTaskTeleop,
    G1WholebodyConditionClearChairAndPushDollyTaskTeleop,
    G1WholebodyConditionEraserToPencilCaseTaskTeleop,
    G1WholebodyConditionPlateToDrawerTaskTeleop,
)
from .g1_wholebody_condition_book_to_occupied_shelf_layer_teleop import (
    G1WholebodyConditionBookToOccupiedShelfLayerTaskTeleop,
)
from .g1_wholebody_condition_peanuts_to_empty_receptacle_teleop import (
    G1WholebodyConditionPeanutsToEmptyPlateOrTrayTaskTeleop,
)

__all__ = [
    "G1WholebodyConditionAppleToFruitFreeBasketTaskTeleop",
    "G1WholebodyConditionBasketToTableAppleToBasketTaskTeleop",
    "G1WholebodyConditionBottleToEmptyBoxTaskTeleop",
    "G1WholebodyConditionBreadToPlateOrTrayTaskTeleop",
    "G1WholebodyConditionChairToTableAndSitTaskTeleop",
    "G1WholebodyConditionClearChairAndPushDollyTaskTeleop",
    "G1WholebodyConditionEraserToPencilCaseTaskTeleop",
    "G1WholebodyConditionAppleToDrawerTaskTeleop",
    "G1WholebodyConditionPlateToDrawerTaskTeleop",
    "G1WholebodyConditionBookToOccupiedShelfLayerTaskTeleop",
    "G1WholebodyConditionPeanutsToEmptyPlateOrTrayTaskTeleop",
]
