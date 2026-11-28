"""Goal-conditioned HUMANO task registrations."""

from .g1_wholebody_goal_interaction_teleop import (
    G1WholebodyGoalEraserPourTaskTeleop,
    G1WholebodyGoalFootballPlacementTaskTeleop,
)
from .g1_wholebody_goal_placement_teleop import (
    G1WholebodyGoalApplePlacementTaskTeleop,
    G1WholebodyGoalBookPlacementTaskTeleop,
    G1WholebodyGoalBreadPlacementTaskTeleop,
    G1WholebodyGoalGroundBottlePlacementTaskTeleop,
    G1WholebodyGoalMilkPlacementTaskTeleop,
    G1WholebodyGoalObjectToBasketTaskTeleop,
    G1WholebodyGoalPlatePlacementTaskTeleop,
    G1WholebodyGoalStorageBoxPlacementTaskTeleop,
)

__all__ = [
    "G1WholebodyGoalApplePlacementTaskTeleop",
    "G1WholebodyGoalBookPlacementTaskTeleop",
    "G1WholebodyGoalBreadPlacementTaskTeleop",
    "G1WholebodyGoalEraserPourTaskTeleop",
    "G1WholebodyGoalFootballPlacementTaskTeleop",
    "G1WholebodyGoalGroundBottlePlacementTaskTeleop",
    "G1WholebodyGoalMilkPlacementTaskTeleop",
    "G1WholebodyGoalPlatePlacementTaskTeleop",
    "G1WholebodyGoalObjectToBasketTaskTeleop",
    "G1WholebodyGoalStorageBoxPlacementTaskTeleop",
]
