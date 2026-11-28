# Task Catalog

HUMANO defines 100 benchmark task entries across eight suites, using 70 distinct environment IDs. The [machine-readable catalog](../../resources/task_catalog_100.csv) lists the same task numbers and environment IDs. Download the archive for the suite and task you want to train or evaluate. See [task archives](../../data/README.md#download-and-extract-a-task).

| Suite | Tasks |
| --- | ---: |
| Basic | 10 |
| Advanced | 10 |
| Spatial | 20 |
| Long | 20 |
| Goal | 10 |
| Condition | 10 |
| Object | 10 |
| Movement | 10 |
| **Total** | **100** |

## Task identity

- Identify a task by **suite + task number**. Spatial and Movement reuse environment
  IDs from Basic/Advanced with different layouts or parameters.
- Environment IDs include `-v0`; dataset directory names may omit it. Normalize
  this suffix when matching entries.
- Spatial variants are defined by episode configurations. Movement uses
  `movement=true` and requires walking to the interaction position; use Movement
  data for this suite.
- Object contains 10 environments with 27 object branches, counted as 10 entries.
- B-1 does not require one-handed execution; B-2 explicitly requires both hands.
  In L-15, the apple must remain inside the basket after placement on the floor.

### C-9 naming

C-9 retains the registered ID `G1WholebodyConditionClearChairAndPushDollyTeleop-v0` for compatibility. Its task is to **push the chair toward the table with a dolly as an obstacle**. The `block` and `clear` branches change the dolly's initial position; moving the dolly is not required. Success checks chair displacement, tilt, lift, and sustained stability.

## Basic (10)

| Number | Environment ID | Task |
| --- | --- | --- |
| B-1 | `G1WholebodyTableApplePickTeleop-v0` | Pick up the apple from the table |
| B-2 | `G1WholebodyTablePlatePickTeleop-v0` | Pick up the plate from the table with both hands |
| B-3 | `G1WholebodyChairPushTeleop-v0` | Push the chair |
| B-4 | `G1WholebodyDollyPushTeleop-v0` | Push the dolly |
| B-5 | `G1WholebodyChairSitTeleop-v0` | Sit on the chair |
| B-6 | `G1WholebodyTableButtonPressTeleop-v0` | Press the button on the table |
| B-7 | `G1WholebodyGroundBottlePickTeleop-v0` | Pick up the bottle from the floor |
| B-8 | `G1WholebodyGroundBoxPickTeleop-v0` | Pick up the box from the floor |
| B-9 | `G1WholebodyBallKickTeleop-v0` | Kick the ball |
| B-10 | `G1WholebodyGroundBasketPickTeleop-v0` | Pick up the basket from the floor |

## Advanced (10)

| Number | Environment ID | Task |
| --- | --- | --- |
| A-1 | `G1WholebodyTableAppleToBasketTeleop-v0` | Put the apple from the table into the basket |
| A-2 | `G1WholebodyTablePlateToShelfTeleop-v0` | Put the plate from the table onto the shelf |
| A-3 | `G1WholebodyTableMilkToTrashCanTeleop-v0` | Put the milk carton from the table into the trash can |
| A-4 | `G1WholebodyTableBreadToTrayTeleop-v0` | Put the bread from the table onto the tray |
| A-5 | `G1WholebodyGroundBasketToTableTeleop-v0` | Move the basket from the floor to the table |
| A-6 | `G1WholebodyGroundBoxToTableTeleop-v0` | Move the box from the floor to the table |
| A-7 | `G1WholebodyGroundBottleToTableTeleop-v0` | Move the bottle from the floor to the table |
| A-8 | `G1WholebodyChairPushToTableTeleop-v0` | Push the chair toward the table |
| A-9 | `G1WholebodyDollyPushToTableTeleop-v0` | Push the dolly toward the table |
| A-10 | `G1WholebodyTableEraserOutOfPenHolderTeleop-v0` | Pour the eraser out of the pen holder |

## Spatial (20)

| Number | Environment ID | Task |
| --- | --- | --- |
| S-1 | `G1WholebodyTableApplePickTeleop-v0` | Pick up the apple from the table |
| S-2 | `G1WholebodyTablePlatePickTeleop-v0` | Pick up the plate from the table |
| S-3 | `G1WholebodyGroundBottlePickTeleop-v0` | Pick up the bottle from the floor |
| S-4 | `G1WholebodyGroundBoxPickTeleop-v0` | Pick up the box from the floor |
| S-5 | `G1WholebodyChairSitTeleop-v0` | Sit on the chair |
| S-6 | `G1WholebodyDollyPushTeleop-v0` | Push the dolly |
| S-7 | `G1WholebodyChairPushTeleop-v0` | Push the chair |
| S-8 | `G1WholebodyBallKickTeleop-v0` | Kick the ball |
| S-9 | `G1WholebodyTableButtonPressTeleop-v0` | Press the button on the table |
| S-10 | `G1WholebodyGroundBasketPickTeleop-v0` | Pick up the basket from the floor |
| S-11 | `G1WholebodyTableAppleToBasketTeleop-v0` | Put the apple from the table into the basket |
| S-12 | `G1WholebodyTablePlateToShelfTeleop-v0` | Put the plate from the table onto the shelf |
| S-13 | `G1WholebodyTableMilkToTrashCanTeleop-v0` | Put the milk carton from the table into the trash can |
| S-14 | `G1WholebodyTableBreadToTrayTeleop-v0` | Put the bread from the table onto the tray |
| S-15 | `G1WholebodyGroundBottleToTableTeleop-v0` | Move the bottle from the floor to the table |
| S-16 | `G1WholebodyGroundBoxToTableTeleop-v0` | Move the box from the floor to the table |
| S-17 | `G1WholebodyChairPushToTableTeleop-v0` | Push the chair toward the table |
| S-18 | `G1WholebodyDollyPushToTableTeleop-v0` | Push the dolly toward the table |
| S-19 | `G1WholebodyTableEraserOutOfPenHolderTeleop-v0` | Pour the eraser out of the pen holder |
| S-20 | `G1WholebodyGroundBasketToTableTeleop-v0` | Move the basket from the floor to the table |

## Long (20)

| Number | Environment ID | Task |
| --- | --- | --- |
| L-1 | `G1WholebodyLongTableAppleToDrawerTeleop-v0` | Put the apple from the table into the drawer |
| L-2 | `G1WholebodyLongTablePlateToDrawerTeleop-v0` | Put the plate from the table into the drawer |
| L-3 | `G1WholebodyLongBreadCakeToTrayTeleop-v0` | Put the bread and cake onto the tray |
| L-4 | `G1WholebodyLongBreadToTrayAndLiftTeleop-v0` | Put the bread onto the tray, then lift the tray |
| L-5 | `G1WholebodyLongCakeToPlateAndLiftTeleop-v0` | Put the cake onto the plate, then lift the plate |
| L-6 | `G1WholebodyLongGroundBoxToDollyAndPushTeleop-v0` | Put the box from the floor onto the dolly, then push it forward |
| L-7 | `G1WholebodyLongChairPushAndSitTeleop-v0` | Push the chair, then sit down |
| L-8 | `G1WholebodyLongTableFootballToGroundAndKickTeleop-v0` | Move the football from the table to the floor, then kick it |
| L-9 | `G1WholebodyLongGroundBasketToTableAppleToBasketTeleop-v0` | Move the basket from the floor to the table, then put the apple from the table into it |
| L-10 | `G1WholebodyLongGroundBoxToTableBottleToBoxTeleop-v0` | Put the bottle from the floor into the box on the low table, then lift the box |
| L-11 | `G1WholebodyLongAppleBananaToBasketTeleop-v0` | Put the apple and banana into the basket |
| L-12 | `G1WholebodyLongBasketAppleToTableBananaToTrashCanTeleop-v0` | Move the apple from the basket to the table, then put the banana into the trash can |
| L-13 | `G1WholebodyLongPeanutsFromCupToPlateTeleop-v0` | Pour the two peanuts from the jar onto the plate |
| L-14 | `G1WholebodyLongStorageBoxCakeToTableBreadToTrayTeleop-v0` | Move the cake from the storage box to the table and the bread to the tray |
| L-15 | `G1WholebodyLongAppleToBasketThenBasketToGroundTeleop-v0` | Put the apple into the basket, then lift and place the basket on the floor |
| L-16 | `G1WholebodyLongHighTableAppleToLowTableDrawerTeleop-v0` | Put the apple from the high table into the low table drawer |
| L-17 | `G1WholebodyLongLowTablePlateToHighTableDrawerTeleop-v0` | Put the plate from the low table into the high table drawer |
| L-18 | `G1WholebodyLongMoveBoxThenChairToTableTeleop-v0` | Move the box aside, then push the chair toward the table |
| L-19 | `G1WholebodyTableBookToShelfTeleop-v0` | Put the book from the table onto the bookshelf |
| L-20 | `G1WholebodyLongShelfBookToTableDrawerTeleop-v0` | Put the book from the bookshelf into the table drawer |

## Goal (10)

| Number | Environment ID | Task |
| --- | --- | --- |
| G-1 | `G1WholebodyGoalApplePlacementTeleop-v0` | Put the apple into the specified container |
| G-2 | `G1WholebodyGoalPlatePlacementTeleop-v0` | Put the plate into the specified container |
| G-3 | `G1WholebodyGoalBreadPlacementTeleop-v0` | Put the bread into the specified container |
| G-4 | `G1WholebodyGoalMilkPlacementTeleop-v0` | Put the milk carton into the specified container |
| G-5 | `G1WholebodyGoalObjectToBasketTeleop-v0` | Put the specified object into the basket |
| G-6 | `G1WholebodyGoalBookPlacementTeleop-v0` | Put the book into the specified container |
| G-7 | `G1WholebodyGoalGroundBottlePlacementTeleop-v0` | Move the bottle from the floor to the specified surface |
| G-8 | `G1WholebodyGoalEraserPourTeleop-v0` | Pour the eraser from the pen holder to the specified location |
| G-9 | `G1WholebodyGoalStorageBoxPlacementTeleop-v0` | Put the specified object into the storage box |
| G-10 | `G1WholebodyGoalFootballPlacementTeleop-v0` | Move the football to the specified location |

## Condition (10)

| Number | Environment ID | Task |
| --- | --- | --- |
| C-1 | `G1WholebodyConditionPlateToDrawerTeleop-v0` | Put the plate into the drawer according to the drawer state |
| C-2 | `G1WholebodyConditionAppleToDrawerTeleop-v0` | Put the apple into the drawer according to the drawer state |
| C-3 | `G1WholebodyConditionBottleToEmptyBoxTeleop-v0` | Put the bottle into the initially empty open storage box |
| C-4 | `G1WholebodyConditionAppleToFruitFreeBasketTeleop-v0` | Put the apple into the basket that initially contains no other fruit |
| C-5 | `G1WholebodyConditionChairToTableAndSitTeleop-v0` | Move the chair toward the table as needed, then sit down |
| C-6 | `G1WholebodyConditionBasketToTableAppleToBasketTeleop-v0` | Put the apple into the basket according to the basket position |
| C-7 | `G1WholebodyConditionPeanutsToEmptyPlateOrTrayTeleop-v0` | Pour the peanuts from the cup onto the empty plate or tray |
| C-8 | `G1WholebodyConditionBreadToPlateOrTrayTeleop-v0` | Choose the plate or tray according to whether the plate is empty |
| C-9 | `G1WholebodyConditionClearChairAndPushDollyTeleop-v0` | Push the chair with a dolly as an obstacle |
| C-10 | `G1WholebodyConditionBookToOccupiedShelfLayerTeleop-v0` | Place the book according to the occupied bookshelf layer |

## Object (10)

| Number | Environment ID | Task |
| --- | --- | --- |
| O-1 | `G1WholebodyTableSmallObjectPickTeleop-v0` | Pick up the small object from the table with one hand |
| O-2 | `G1WholebodyTableLargeObjectPickTeleop-v0` | Pick up the large object from the table with both hands |
| O-3 | `G1WholebodyTableObjectPickTeleop-v0` | Pick up the object from the table according to its type |
| O-4 | `G1WholebodyGroundSmallObjectPickTeleop-v0` | Pick up the small object from the floor with one hand |
| O-5 | `G1WholebodyGroundLargeObjectPickTeleop-v0` | Pick up the large object from the floor with both hands |
| O-6 | `G1WholebodyGroundObjectPickTeleop-v0` | Pick up the object from the floor according to its type |
| O-7 | `G1WholebodyTableSmallObjectToBasketTeleop-v0` | Put the small object from the table into the basket |
| O-8 | `G1WholebodyTableLargeObjectToCabinetTeleop-v0` | Put the large object from the table into the cabinet |
| O-9 | `G1WholebodyTableObjectToTrashCanTeleop-v0` | Put the object from the table into the trash can |
| O-10 | `G1WholebodyFoodToTrayTeleop-v0` | Put the specified food onto the tray |

## Movement (10)

| Number | Environment ID | Task |
| --- | --- | --- |
| M-1 | `G1WholebodyTableApplePickTeleop-v0` | Walk to the opposite side of the table and pick up the apple |
| M-2 | `G1WholebodyTablePlatePickTeleop-v0` | Walk to the opposite side of the table and pick up the plate |
| M-3 | `G1WholebodyChairPushTeleop-v0` | Walk to the chair and push it forward |
| M-4 | `G1WholebodyDollyPushTeleop-v0` | Walk to the dolly and push it forward |
| M-5 | `G1WholebodyChairSitTeleop-v0` | Walk to the chair and sit down |
| M-6 | `G1WholebodyTableButtonPressTeleop-v0` | Walk to the opposite side of the table and press the button |
| M-7 | `G1WholebodyGroundBottlePickTeleop-v0` | Walk to the bottle on the floor and pick it up |
| M-8 | `G1WholebodyGroundBoxPickTeleop-v0` | Walk to the box on the floor and pick it up |
| M-9 | `G1WholebodyBallKickTeleop-v0` | Walk to the football and kick it |
| M-10 | `G1WholebodyGroundBasketPickTeleop-v0` | Walk to the basket on the floor and pick it up |
