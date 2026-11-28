from .g1_wholebody_object_tasks_teleop import (
    OBJECT_TASK_TYPES,
    G1WholebodyFoodToTrayTaskTeleop,
    G1WholebodyGroundLargeObjectPickTaskTeleop,
    G1WholebodyGroundObjectPickTaskTeleop,
    G1WholebodyGroundSmallObjectPickTaskTeleop,
    G1WholebodyTableLargeObjectPickTaskTeleop,
    G1WholebodyTableLargeObjectToCabinetTaskTeleop,
    G1WholebodyTableObjectPickTaskTeleop,
    G1WholebodyTableObjectToTrashCanTaskTeleop,
    G1WholebodyTableSmallObjectPickTaskTeleop,
    G1WholebodyTableSmallObjectToBasketTaskTeleop,
)

__all__ = ["OBJECT_TASK_TYPES", *(task_type.__name__ for task_type in OBJECT_TASK_TYPES)]
