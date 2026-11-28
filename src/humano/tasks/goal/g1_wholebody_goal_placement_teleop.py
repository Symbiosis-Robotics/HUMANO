from __future__ import annotations

from typing import Any

import numpy as np
import transforms3d as t3d

from humano.tasks.goal._goal_base import (
    APPLE,
    BANANA,
    BASKET,
    BOOK,
    BOTTLE,
    BOX,
    BREAD,
    CABINET,
    CABINET_INTERIOR_HIGH,
    CABINET_INTERIOR_LOW,
    CABINET_XY,
    MILK,
    PLATE,
    SHELF,
    TRASH_CAN,
    TRAY,
    GoalObjectTask,
    actor_local_position,
    actor_points_in_body,
    body_name,
    body_pose,
    contact_between,
    hand_contact,
)
from humano.tasks.registry import TaskRegistry

_BASKET_HALF_XY = np.array([0.14, 0.11], dtype=np.float64)
_BOX_LOW = np.array([-0.215, -0.16, -0.005], dtype=np.float64)
_BOX_HIGH = np.array([0.215, 0.16, 0.15], dtype=np.float64)
_PLATE_BOX_MIN_XY_INSIDE_FRACTION = 0.90
_TRAY_HALF_XY = np.array([0.15, 0.11], dtype=np.float64)
_PEN_CASE_LOW = np.array([-0.097, -0.023, -0.004], dtype=np.float64)
_PEN_CASE_HIGH = np.array([0.091, 0.023, 0.021], dtype=np.float64)
_SHELF_XY_LOW = np.array([-0.08, -0.14], dtype=np.float64)
_SHELF_XY_HIGH = np.array([0.30, 0.18], dtype=np.float64)


def _plate_in_box(mujoco_env, target, box) -> bool:
    if hand_contact(mujoco_env, target):
        return False
    points = actor_points_in_body(mujoco_env, target, box.simulation_name)
    center = actor_local_position(mujoco_env, target, box.simulation_name)
    if points is None or center is None:
        return False
    low = _BOX_LOW.copy()
    low[2] = -0.015
    xy_inside = np.all((points[:, :2] >= low[:2]) & (points[:, :2] <= _BOX_HIGH[:2]), axis=1)
    geometry_inside = bool(
        np.all(center >= low)
        and np.all(center <= _BOX_HIGH)
        and np.mean(xy_inside) >= _PLATE_BOX_MIN_XY_INSIDE_FRACTION
    )
    supported = contact_between(mujoco_env, target.simulation_name, box.simulation_name) or (
        contact_between(mujoco_env, target.simulation_name, "table")
    )
    return geometry_inside and supported


class _ReceptacleGoalTask(GoalObjectTask):
    def _object_destination_success(self, goal: str, mujoco_env) -> bool:
        destination = self.destination(goal)
        local = actor_local_position(mujoco_env, self.target, destination.simulation_name)
        if local is None or hand_contact(mujoco_env, self.target):
            return False
        touching = contact_between(
            mujoco_env, self.target.simulation_name, destination.simulation_name
        )
        if goal == "basket":
            return bool(
                np.all(np.abs(local[:2]) <= _BASKET_HALF_XY)
                and -0.08 <= local[2] <= 0.18
                and touching
            )
        if goal == "tray":
            return bool(
                np.all(np.abs(local[:2]) <= _TRAY_HALF_XY)
                and -0.01 <= local[2] <= 0.10
                and touching
            )
        if goal in {"box", "cabinet"}:
            low, high = (
                (_BOX_LOW, _BOX_HIGH)
                if goal == "box"
                else (CABINET_INTERIOR_LOW, CABINET_INTERIOR_HIGH)
            )
            points = actor_points_in_body(mujoco_env, self.target, destination.simulation_name)
            return bool(
                points is not None and np.all(points >= low) and np.all(points <= high) and touching
            )
        if goal == "trash_can":
            points = actor_points_in_body(mujoco_env, self.target, destination.simulation_name)
            return bool(
                points is not None
                and np.all(np.abs(points[:, :2]) <= 0.12)
                and np.mean(points[:, 2] <= 0.125) >= 0.5
            )
        if goal == "pen_holder":
            target_pose = body_pose(mujoco_env, self.target.simulation_name)
            if target_pose is None:
                return False
            axis = target_pose[1] @ np.array([1.0, 0.0, 0.0])
            return bool(
                np.linalg.norm(local[:2]) <= 0.075
                and 0.02 <= local[2] <= 0.26
                and abs(float(axis[2])) >= np.cos(np.deg2rad(38.0))
            )
        raise KeyError(goal)

    def _articulated_destination_success(self, goal: str, mujoco_env) -> bool:
        if goal == "pencil_case":
            points = actor_points_in_body(mujoco_env, self.target, "009_base")
            return bool(
                points is not None
                and np.all(points >= _PEN_CASE_LOW)
                and np.all(points <= _PEN_CASE_HIGH)
                and contact_between(mujoco_env, self.target.simulation_name, "009_base")
                and not hand_contact(mujoco_env, self.target)
            )
        raise KeyError(goal)

    def goal_satisfied(self, info: dict[str, Any], mujoco_env) -> bool:
        del info
        if self.selected_goal in self.destination_objects:
            return self._object_destination_success(self.selected_goal, mujoco_env)
        return self._articulated_destination_success(self.selected_goal, mujoco_env)


@TaskRegistry.register("g1_wholebody_goal_apple_placement_teleop")
class G1WholebodyGoalApplePlacementTaskTeleop(_ReceptacleGoalTask):
    uid = "g1_wholebody_goal_apple_placement_teleop"
    label = "G1 TELEOP Goal Apple Placement"
    description = "A task where the G1 robot must place an apple in the instructed receptacle."
    goals = ("basket", "box", "tray")
    goal_instructions = {
        "basket": "put the apple in the basket",
        "box": "put the apple in the box",
        "tray": "put the apple on the tray",
    }
    target_asset = APPLE
    destination_objects = {
        "basket": (BASKET, "table", (-0.05, -0.02)),
        "box": (BOX, "table", (0.45, 0.30)),
        "tray": (TRAY, "table", (0.35, -0.28)),
    }


@TaskRegistry.register("g1_wholebody_goal_plate_placement_teleop")
class G1WholebodyGoalPlatePlacementTaskTeleop(_ReceptacleGoalTask):
    uid = "g1_wholebody_goal_plate_placement_teleop"
    label = "G1 TELEOP Goal Plate Placement"
    description = "A task where the G1 robot must place a plate in the instructed receptacle."
    goals = ("basket", "box", "cabinet")
    goal_instructions = {
        "basket": "put the plate in the basket",
        "box": "put the plate in the box",
        "cabinet": "put the plate fully inside the cabinet compartment",
    }
    target_asset = PLATE
    cabinet_xy = (0.70, -0.18)
    destination_objects = {
        "basket": (BASKET, "table", (-0.02, -0.02)),
        "box": (BOX, "table", (0.45, 0.30)),
        "cabinet": (CABINET, "table", cabinet_xy),
    }

    def goal_satisfied(self, info: dict[str, Any], mujoco_env) -> bool:
        if self.selected_goal == "box":
            del info
            return _plate_in_box(mujoco_env, self.target, self.destination("box"))
        if self.selected_goal != "cabinet":
            return super().goal_satisfied(info, mujoco_env)
        if hand_contact(mujoco_env, self.target):
            return False
        cabinet = self.destination("cabinet")
        local = actor_local_position(mujoco_env, self.target, cabinet.simulation_name)
        return bool(
            local is not None
            and np.all(local >= CABINET_INTERIOR_LOW)
            and np.all(local <= CABINET_INTERIOR_HIGH)
            and contact_between(mujoco_env, self.target.simulation_name, cabinet.simulation_name)
        )


@TaskRegistry.register("g1_wholebody_goal_bread_placement_teleop")
class G1WholebodyGoalBreadPlacementTaskTeleop(_ReceptacleGoalTask):
    uid = "g1_wholebody_goal_bread_placement_teleop"
    label = "G1 TELEOP Goal Bread Placement"
    description = "A task where the G1 robot must place bread in the instructed receptacle."
    goals = ("basket", "tray", "trash_can")
    goal_instructions = {
        "basket": "put the bread in the basket",
        "tray": "put the bread on the tray",
        "trash_can": "put the bread in the trash can",
    }
    target_asset = BREAD
    destination_objects = {
        "basket": (BASKET, "table", (-0.05, -0.02)),
        "tray": (TRAY, "table", (0.30, -0.28)),
        "trash_can": (TRASH_CAN, "ground", (-0.48, 0.48)),
    }


@TaskRegistry.register("g1_wholebody_goal_milk_placement_teleop")
class G1WholebodyGoalMilkPlacementTaskTeleop(_ReceptacleGoalTask):
    uid = "g1_wholebody_goal_milk_placement_teleop"
    label = "G1 TELEOP Goal Milk Placement"
    description = "A task where the G1 robot must place a milk carton in the instructed receptacle."
    goals = ("basket", "box", "trash_can")
    goal_instructions = {
        "basket": "put the milk carton in the basket",
        "box": "put the milk carton in the box",
        "trash_can": "put the milk carton in the trash can",
    }
    target_asset = MILK
    destination_objects = {
        "basket": (BASKET, "table", (-0.05, -0.02)),
        "box": (BOX, "table", (0.45, 0.30)),
        "trash_can": (TRASH_CAN, "ground", (-0.48, 0.48)),
    }


@TaskRegistry.register("g1_wholebody_goal_object_to_basket_teleop")
class G1WholebodyGoalObjectToBasketTaskTeleop(_ReceptacleGoalTask):
    uid = "g1_wholebody_goal_object_to_basket_teleop"
    label = "G1 TELEOP Goal Object To Basket"
    description = "A task where the G1 robot must put the instructed object into the basket."
    goals = ("apple", "banana", "bread")
    goal_instructions = {
        "apple": "put the apple in the basket",
        "banana": "put the banana in the basket",
        "bread": "put the bread in the basket",
    }
    goal_target_assets = {
        "apple": APPLE,
        "banana": BANANA,
        "bread": BREAD,
    }
    target_asset = APPLE
    target_xy = (-0.22, -0.22)
    destination_objects = {
        "banana": (BANANA, "table", (-0.22, 0.0)),
        "bread": (BREAD, "table", (-0.22, 0.22)),
        "basket": (BASKET, "table", (0.25, 0.0)),
    }

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> None:
        super().reset(seed, options)
        selected_asset = self.goal_target_assets[self.selected_goal]
        self._target = next(
            actor
            for actor in self.layout.actors.values()
            if str(getattr(getattr(actor, "asset", None), "uid", "")) == selected_asset[1]
        )

    def goal_satisfied(self, info: dict[str, Any], mujoco_env) -> bool:
        del info
        return self._object_destination_success("basket", mujoco_env)


@TaskRegistry.register("g1_wholebody_goal_storage_box_placement_teleop")
class G1WholebodyGoalStorageBoxPlacementTaskTeleop(_ReceptacleGoalTask):
    uid = "g1_wholebody_goal_storage_box_placement_teleop"
    label = "G1 TELEOP Goal Storage Box Placement"
    description = "A task where the G1 robot must put the instructed object into the storage box."
    goals = ("plate", "book", "ground_bottle")
    goal_instructions = {
        "plate": "put the plate into the storage box",
        "book": "put the book into the storage box",
        "ground_bottle": "pick up the bottle from the ground and put it into the storage box",
    }
    goal_target_assets = {
        "plate": PLATE,
        "book": BOOK,
        "ground_bottle": BOTTLE,
    }
    target_asset = PLATE
    target_xy = (-0.15, -0.22)
    destination_objects = {
        "book": (BOOK, "table", (0.15, -0.22)),
        "ground_bottle": (BOTTLE, "ground", (-0.55, -0.20)),
        "box": (BOX, "table", (0.45, 0.25)),
    }

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> None:
        super().reset(seed, options)
        targets = {
            "plate": self.target,
            "book": self.destination("book"),
            "ground_bottle": self.destination("ground_bottle"),
        }
        self._target = targets[self.selected_goal]

    def goal_satisfied(self, info: dict[str, Any], mujoco_env) -> bool:
        del info
        box = self.destination("box")
        if self.selected_goal == "plate":
            return _plate_in_box(mujoco_env, self.target, box)
        if hand_contact(mujoco_env, self.target):
            return False
        points = actor_points_in_body(mujoco_env, self.target, box.simulation_name)
        low = _BOX_LOW.copy()
        low[2] = -0.015
        if points is None:
            return False
        geometry_inside = bool(np.all(points >= low) and np.all(points <= _BOX_HIGH))
        return bool(
            geometry_inside
            and (
                contact_between(mujoco_env, self.target.simulation_name, box.simulation_name)
                or contact_between(mujoco_env, self.target.simulation_name, "table")
            )
        )


@TaskRegistry.register("g1_wholebody_goal_book_placement_teleop")
class G1WholebodyGoalBookPlacementTaskTeleop(GoalObjectTask):
    uid = "g1_wholebody_goal_book_placement_teleop"
    label = "G1 TELEOP Goal Book Placement"
    description = "A task where the G1 robot must place a book at the instructed destination."
    goals = ("shelf", "cabinet")
    goal_instructions = {
        "shelf": "put the book upright in the lower shelf",
        "cabinet": "put the book fully inside the cabinet compartment",
    }
    target_asset = BOOK
    target_surface = "table"
    target_xy = (-0.18, -0.22)
    destination_objects = {
        "shelf": (SHELF, "ground", (-0.08, 0.90)),
        "cabinet": (CABINET, "table", CABINET_XY),
    }

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> None:
        super().reset(seed, options)
        shelf = self.destination("shelf")
        stable_z = float(shelf.asset.stable_poses[0][2])
        shelf.pose.position = [-0.08, 0.90, stable_z]
        shelf.pose.quaternion = [0.70710678, 0.0, 0.0, -0.70710678]

    def goal_satisfied(self, info: dict[str, Any], mujoco_env) -> bool:
        del info
        if hand_contact(mujoco_env, self.target):
            return False
        if self.selected_goal == "table":
            return contact_between(mujoco_env, self.target.simulation_name, "table")
        if self.selected_goal == "shelf":
            shelf = self.destination("shelf")
            local = actor_local_position(mujoco_env, self.target, shelf.simulation_name)
            if local is None:
                return False
            target_pose = body_pose(mujoco_env, self.target.simulation_name)
            upright = False
            if target_pose is not None:
                # A book may be stood on either edge; accept either horizontal
                # local axis aligned with world up.
                upright = max(
                    abs(float((target_pose[1] @ [1.0, 0.0, 0.0])[2])),
                    abs(float((target_pose[1] @ [0.0, 1.0, 0.0])[2])),
                ) >= np.cos(np.deg2rad(30.0))
            return bool(
                np.all(local[:2] >= _SHELF_XY_LOW)
                and np.all(local[:2] <= _SHELF_XY_HIGH)
                and any(
                    (
                        0.24 <= local[2] <= 0.38,
                        0.46 <= local[2] <= 0.62,
                        0.86 <= local[2] <= 1.04,
                    )
                )
                and upright
                and contact_between(mujoco_env, self.target.simulation_name, shelf.simulation_name)
            )
        cabinet = self.destination("cabinet")
        points = actor_points_in_body(mujoco_env, self.target, cabinet.simulation_name)
        return bool(
            points is not None
            and np.all(points >= CABINET_INTERIOR_LOW)
            and np.all(points <= CABINET_INTERIOR_HIGH)
            and contact_between(mujoco_env, self.target.simulation_name, cabinet.simulation_name)
        )


@TaskRegistry.register("g1_wholebody_goal_ground_bottle_placement_teleop")
class G1WholebodyGoalGroundBottlePlacementTaskTeleop(GoalObjectTask):
    uid = "g1_wholebody_goal_ground_bottle_placement_teleop"
    label = "G1 TELEOP Goal Ground Bottle Placement"
    description = (
        "A task where the G1 robot must move a bottle from the ground to the instructed surface."
    )
    goals = ("table", "cabinet")
    goal_instructions = {
        "table": "pick up the bottle from the ground and put it on the table",
        "cabinet": "pick up the bottle from the ground and put it fully inside the cabinet compartment",
    }
    target_asset = BOTTLE
    target_surface = "ground"
    target_xy = (-0.55, -0.20)
    destination_objects = {"cabinet": (CABINET, "table", CABINET_XY)}

    def goal_satisfied(self, info: dict[str, Any], mujoco_env) -> bool:
        del info
        if hand_contact(mujoco_env, self.target):
            return False
        if self.selected_goal == "table":
            model = mujoco_env.mjModel
            target_name = self.target.simulation_name
            return any(
                (
                    body_name(model, contact.geom1) == target_name
                    and body_name(model, contact.geom2) == "table"
                )
                or (
                    body_name(model, contact.geom2) == target_name
                    and body_name(model, contact.geom1) == "table"
                )
                for contact in mujoco_env.mjData.contact[: mujoco_env.mjData.ncon]
            )
        cabinet = self.destination("cabinet")
        points = actor_points_in_body(mujoco_env, self.target, cabinet.simulation_name)
        return bool(
            points is not None
            and np.all(points >= CABINET_INTERIOR_LOW)
            and np.all(points <= CABINET_INTERIOR_HIGH)
            and contact_between(mujoco_env, self.target.simulation_name, cabinet.simulation_name)
        )
