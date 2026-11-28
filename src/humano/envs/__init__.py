"""
HUMANO Gymnasium environment registration.

SONIC/G1 evaluation environments are registered here.
"""

from enum import Enum

from gymnasium.envs.registration import register


class SIM_MODE(Enum):
    MUJOCO = "mujoco"
    ISAAC = "isaac"
    MUJOCO_ISAAC = "mujoco_isaac"


_SONIC_ENTRY = "humano.envs.sonic_loco_manip:SonicLocoManipEnv"

_SONIC_TASKS = {
    "G1WholebodyBallKickTeleop-v0": "g1_wholebody_ball_kick_teleop",
    "G1WholebodyChairPushTeleop-v0": "g1_wholebody_chair_push_teleop",
    "G1WholebodyChairPushToTableTeleop-v0": "g1_wholebody_chair_push_to_table_teleop",
    "G1WholebodyChairSitTeleop-v0": "g1_wholebody_chair_sit_teleop",
    "G1WholebodyConditionAppleToFruitFreeBasketTeleop-v0": "g1_wholebody_condition_apple_to_fruit_free_basket_teleop",
    "G1WholebodyConditionBasketToTableAppleToBasketTeleop-v0": "g1_wholebody_condition_basket_to_table_apple_to_basket_teleop",
    "G1WholebodyConditionBottleToEmptyBoxTeleop-v0": "g1_wholebody_condition_bottle_to_empty_box_teleop",
    "G1WholebodyConditionBreadToPlateOrTrayTeleop-v0": "g1_wholebody_condition_bread_to_plate_or_tray_teleop",
    "G1WholebodyConditionChairToTableAndSitTeleop-v0": "g1_wholebody_condition_chair_to_table_and_sit_teleop",
    "G1WholebodyConditionClearChairAndPushDollyTeleop-v0": "g1_wholebody_condition_clear_chair_and_push_dolly_teleop",
    "G1WholebodyConditionEraserToPencilCaseTeleop-v0": "g1_wholebody_condition_eraser_to_pencil_case_teleop",
    "G1WholebodyConditionAppleToDrawerTeleop-v0": "g1_wholebody_condition_apple_to_drawer_teleop",
    "G1WholebodyConditionPeanutsToEmptyPlateOrTrayTeleop-v0": "g1_wholebody_condition_peanuts_to_empty_plate_or_tray_teleop",
    "G1WholebodyConditionPlateToDrawerTeleop-v0": "g1_wholebody_condition_plate_to_drawer_teleop",
    "G1WholebodyConditionBookToOccupiedShelfLayerTeleop-v0": "g1_wholebody_condition_book_to_occupied_shelf_layer_teleop",
    "G1WholebodyDollyPushTeleop-v0": "g1_wholebody_dolly_push_teleop",
    "G1WholebodyDollyPushToTableTeleop-v0": "g1_wholebody_dolly_push_to_table_teleop",
    "G1WholebodyGroundApplePickTeleop-v0": "g1_wholebody_ground_apple_pick_teleop",
    "G1WholebodyGroundBasketPickTeleop-v0": "g1_wholebody_ground_basket_pick_teleop",
    "G1WholebodyGroundBasketToTableTeleop-v0": "g1_wholebody_ground_basket_to_table_teleop",
    "G1WholebodyGroundBookToTableTeleop-v0": "g1_wholebody_ground_book_to_table_teleop",
    "G1WholebodyGroundBottleFixPickTeleop-v0": "g1_wholebody_ground_bottle_fix_pick_teleop",
    "G1WholebodyGroundBottlePickTeleop-v0": "g1_wholebody_ground_bottle_pick_teleop",
    "G1WholebodyGroundBottleToTableTeleop-v0": "g1_wholebody_ground_bottle_to_table_teleop",
    "G1WholebodyGroundBoxPickTeleop-v0": "g1_wholebody_ground_box_pick_teleop",
    "G1WholebodyGroundBoxToTableTeleop-v0": "g1_wholebody_ground_box_to_table_teleop",
    "G1WholebodyGoalApplePlacementTeleop-v0": "g1_wholebody_goal_apple_placement_teleop",
    "G1WholebodyGoalBookPlacementTeleop-v0": "g1_wholebody_goal_book_placement_teleop",
    "G1WholebodyGoalBreadPlacementTeleop-v0": "g1_wholebody_goal_bread_placement_teleop",
    "G1WholebodyGoalStorageBoxPlacementTeleop-v0": "g1_wholebody_goal_storage_box_placement_teleop",
    "G1WholebodyGoalEraserPourTeleop-v0": "g1_wholebody_goal_eraser_pour_teleop",
    "G1WholebodyGoalFootballPlacementTeleop-v0": "g1_wholebody_goal_football_placement_teleop",
    "G1WholebodyGoalGroundBottlePlacementTeleop-v0": "g1_wholebody_goal_ground_bottle_placement_teleop",
    "G1WholebodyGoalMilkPlacementTeleop-v0": "g1_wholebody_goal_milk_placement_teleop",
    "G1WholebodyGoalPlatePlacementTeleop-v0": "g1_wholebody_goal_plate_placement_teleop",
    "G1WholebodyGoalObjectToBasketTeleop-v0": "g1_wholebody_goal_object_to_basket_teleop",
    "G1WholebodyLowTableApplePickTeleop-v0": "g1_wholebody_low_table_apple_pick_teleop",
    "G1WholebodyLowTablePlatePickTeleop-v0": "g1_wholebody_low_table_plate_pick_teleop",
    "G1WholebodyDrawerOpenTeleop-v0": "g1_wholebody_drawer_open_teleop",
    "G1WholebodyTableButtonPressTeleop-v0": "g1_wholebody_table_button_press_teleop",
    "G1WholebodyTableApplePickTeleop-v0": "g1_wholebody_table_apple_pick_teleop",
    "G1WholebodyTableAppleToBasketTeleop-v0": "g1_wholebody_table_apple_to_basket_teleop",
    "G1WholebodyTableAppleToTrayTeleop-v0": "g1_wholebody_table_apple_to_tray_teleop",
    "G1WholebodyTableBreadToTrayTeleop-v0": "g1_wholebody_table_bread_to_tray_teleop",
    "G1WholebodyTableBookToCartTeleop-v0": "g1_wholebody_table_book_to_cart_teleop",
    "G1WholebodyTableBookPickTeleop-v0": "g1_wholebody_table_book_pick_teleop",
    "G1WholebodyTableBookToShelfTeleop-v0": "g1_wholebody_table_book_to_shelf_teleop",
    "G1WholebodyTableBreadToBasketTeleop-v0": "g1_wholebody_table_bread_to_basket_teleop",
    "G1WholebodyTableBreadToTrashCanTeleop-v0": "g1_wholebody_table_bread_to_trash_can_teleop",
    "G1WholebodyTableEraserOutOfPenHolderTeleop-v0": "g1_wholebody_table_eraser_out_of_pen_holder_teleop",
    "G1WholebodyTableMilkToBasketTeleop-v0": "g1_wholebody_table_milk_to_basket_teleop",
    "G1WholebodyTableMilkToTrashCanTeleop-v0": "g1_wholebody_table_milk_to_trash_can_teleop",
    "G1WholebodyTablePencilToPencilCaseTeleop-v0": "g1_wholebody_table_pencil_to_pencil_case_teleop",
    "G1WholebodyTablePencilToPenHolderTeleop-v0": "g1_wholebody_table_pencil_to_pen_holder_teleop",
    "G1WholebodyTablePencilToTrayTeleop-v0": "g1_wholebody_table_pencil_to_tray_teleop",
    "G1WholebodyTablePlatePickTeleop-v0": "g1_wholebody_table_plate_pick_teleop",
    "G1WholebodyTablePlateToBasketTeleop-v0": "g1_wholebody_table_plate_to_basket_teleop",
    "G1WholebodyTablePlateToCabinetTeleop-v0": "g1_wholebody_table_plate_to_cabinet_teleop",
    "G1WholebodyTablePlateToShelfTeleop-v0": "g1_wholebody_table_plate_to_shelf_teleop",
    "G1WholebodyXMoveAndPickMP-v0": "g1_wholebody_x_move_and_pick_mp",
    "G1WholebodyXMovePickTeleop-v0": "g1_wholebody_xmove_pick_teleop",
    "G1WholebodyXMoveBendPickTeleop-v0": "g1_wholebody_xmove_bend_pick_teleop",
    "G1WholebodyPickAndPlaceAndHugContainerTeleop-v0": "g1_wholebody_pick_and_place_and_hug_container_teleop",
    "G1WholebodyLocomotionPickBetweenTablesTeleop-v0": "g1_wholebody_locomotion_pick_between_tables_teleop",
    "G1WholebodyLongAppleBananaToBasketTeleop-v0": "g1_wholebody_long_apple_banana_to_basket_teleop",
    "G1WholebodyLongAppleToBasketThenBasketToGroundTeleop-v0": "g1_wholebody_long_apple_to_basket_then_basket_to_ground_teleop",
    "G1WholebodyLongBasketAppleToTableBananaToTrashCanTeleop-v0": "g1_wholebody_long_basket_apple_to_table_banana_to_trash_can_teleop",
    "G1WholebodyLongBreadCakeToTrayTeleop-v0": "g1_wholebody_long_bread_cake_to_tray_teleop",
    "G1WholebodyLongBreadToTrayAndLiftTeleop-v0": "g1_wholebody_long_bread_to_tray_and_lift_teleop",
    "G1WholebodyLongCakeToPlateAndLiftTeleop-v0": "g1_wholebody_long_cake_to_plate_and_lift_teleop",
    "G1WholebodyLongChairPushAndSitTeleop-v0": "g1_wholebody_long_chair_push_and_sit_teleop",
    "G1WholebodyLongGroundBasketToTableAppleToBasketTeleop-v0": "g1_wholebody_long_ground_basket_to_table_apple_to_basket_teleop",
    "G1WholebodyLongGroundBoxToDollyAndPushTeleop-v0": "g1_wholebody_long_ground_box_to_dolly_and_push_teleop",
    "G1WholebodyLongGroundBoxToTableBottleToBoxTeleop-v0": "g1_wholebody_long_ground_box_to_table_bottle_to_box_teleop",
    "G1WholebodyLongHighTableAppleToLowTableDrawerTeleop-v0": "g1_wholebody_long_high_table_apple_to_low_table_drawer_teleop",
    "G1WholebodyLongLowTablePlateToHighTableDrawerTeleop-v0": "g1_wholebody_long_low_table_plate_to_high_table_drawer_teleop",
    "G1WholebodyLongMoveBoxThenChairToTableTeleop-v0": "g1_wholebody_long_move_box_then_chair_to_table_teleop",
    "G1WholebodyLongPeanutsFromCupToPlateTeleop-v0": "g1_wholebody_long_peanuts_from_cup_to_plate_teleop",
    "G1WholebodyLongShelfBookToTableDrawerTeleop-v0": "g1_wholebody_long_shelf_book_to_table_drawer_teleop",
    "G1WholebodyLongTableAppleToDrawerTeleop-v0": "g1_wholebody_long_table_apple_to_drawer_teleop",
    "G1WholebodyLongTableFootballToGroundAndKickTeleop-v0": "g1_wholebody_long_table_football_to_ground_and_kick_teleop",
    "G1WholebodyLongTablePencilEraserToPencilCaseTeleop-v0": "g1_wholebody_long_table_pencil_eraser_to_pencil_case_teleop",
    "G1WholebodyLongTablePencilPenToPenHolderTeleop-v0": "g1_wholebody_long_table_pencil_pen_to_pen_holder_teleop",
    "G1WholebodyLongTablePlateToDrawerTeleop-v0": "g1_wholebody_long_table_plate_to_drawer_teleop",
    "G1WholebodyLongStorageBoxCakeToTableBreadToTrayTeleop-v0": "g1_wholebody_long_storage_box_cake_to_table_bread_to_tray_teleop",
    "G1WholebodyLongTrayBreadToPlateTeleop-v0": "g1_wholebody_long_tray_bread_to_plate_teleop",
    "G1WholebodyHandoverTeleop-v0": "g1_wholebody_handover_teleop",
    "G1WholebodyCloseDoorTeleop-v0": "g1_wholebody_close_door_teleop",
    "G1WholebodyOpenOvenTeleop-v0": "g1_wholebody_open_oven_teleop",
    "G1WholebodyOpenFaucetTeleop-v0": "g1_wholebody_open_faucet_teleop",
    "G1WholebodyPushOfficeChairTeleop-v0": "g1_wholebody_push_office_chair_teleop",
    "G1WholebodyOpenTrashCanTeleop-v0": "g1_wholebody_open_trash_can_teleop",
    "G1WholebodyTableSmallObjectPickTeleop-v0": "g1_wholebody_table_small_object_pick_teleop",
    "G1WholebodyTableLargeObjectPickTeleop-v0": "g1_wholebody_table_large_object_pick_teleop",
    "G1WholebodyTableObjectPickTeleop-v0": "g1_wholebody_table_object_pick_teleop",
    "G1WholebodyGroundSmallObjectPickTeleop-v0": "g1_wholebody_ground_small_object_pick_teleop",
    "G1WholebodyGroundLargeObjectPickTeleop-v0": "g1_wholebody_ground_large_object_pick_teleop",
    "G1WholebodyGroundObjectPickTeleop-v0": "g1_wholebody_ground_object_pick_teleop",
    "G1WholebodyTableSmallObjectToBasketTeleop-v0": "g1_wholebody_table_small_object_to_basket_teleop",
    "G1WholebodyTableLargeObjectToCabinetTeleop-v0": "g1_wholebody_table_large_object_to_cabinet_teleop",
    "G1WholebodyTableObjectToTrashCanTeleop-v0": "g1_wholebody_table_object_to_trash_can_teleop",
    "G1WholebodyFoodToTrayTeleop-v0": "g1_wholebody_food_to_tray_teleop",
}


def _register_sonic_envs(namespace: str) -> None:
    for name, task in _SONIC_TASKS.items():
        register(
            id=f"{namespace}/{name}",
            entry_point=_SONIC_ENTRY,
            kwargs={"task": task},
        )


_register_sonic_envs("humano")

from .base_dual_env import BaseDualSim
from .sonic_loco_manip import SonicLocoManipEnv

__all__ = ["SIM_MODE", "BaseDualSim", "SonicLocoManipEnv"]
