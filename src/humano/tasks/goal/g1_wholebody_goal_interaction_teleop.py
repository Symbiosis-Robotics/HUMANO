from __future__ import annotations

from typing import Any

import numpy as np

from humano.assets import AssetManager
from humano.core.actor import ObjectActor
from humano.tasks.advanced.g1_wholebody_table_eraser_out_of_pen_holder_teleop import (
    G1WholebodyTableEraserOutOfPenHolderTaskTeleop,
)
from humano.tasks.goal._goal_base import (
    CABINET,
    CABINET_INTERIOR_HIGH,
    CABINET_INTERIOR_LOW,
    CABINET_XY,
    FOOTBALL,
    GoalObjectTask,
    GoalSelectionMixin,
    actor_points_in_body,
    body_pose,
    configure_fixed_cabinet,
    contact_between,
    hand_contact,
)
from humano.tasks.registry import TaskRegistry


def _add_fixed_cabinet(task, xy: tuple[float, float] = CABINET_XY) -> ObjectActor:
    cabinet = ObjectActor(
        asset=AssetManager.get("objects").load(CABINET[1]),
        instance_id="goal_white_cabinet_1",
    )
    configure_fixed_cabinet(cabinet, xy=xy)
    task.layout.add_primitive("goal_cabinet", cabinet)
    return cabinet


@TaskRegistry.register("g1_wholebody_goal_eraser_pour_teleop")
class G1WholebodyGoalEraserPourTaskTeleop(
    GoalSelectionMixin, G1WholebodyTableEraserOutOfPenHolderTaskTeleop
):
    uid = "g1_wholebody_goal_eraser_pour_teleop"
    label = "G1 TELEOP Goal Eraser Pour"
    description = "A task where the G1 robot must pour an eraser from a pen holder onto the instructed surface."
    goals = ("table", "cabinet")
    # Keep the cabinet at the shared goal position; move the source objects
    # toward the near table edge to provide more approach clearance.
    cabinet_xy = CABINET_XY
    source_y_shift = -0.15
    goal_instructions = {
        "table": "tip the pen holder and pour the eraser onto the table",
        "cabinet": "tip the pen holder and pour the eraser fully inside the cabinet compartment",
    }

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> None:
        super().reset(seed, options)
        for actor in (self.target, self.eraser):
            if isinstance(actor, ObjectActor):
                position = np.asarray(actor.pose.position, dtype=np.float64)
                position[1] += self.source_y_shift
                actor.pose.position = position.tolist()
        self._goal_cabinet = _add_fixed_cabinet(self, xy=self.cabinet_xy)
        self._success_hold_count = 0

    def _eraser_on_cabinet(self, mujoco_env) -> bool:
        cabinet_body = self._goal_cabinet.simulation_name
        points = actor_points_in_body(mujoco_env, self.eraser, cabinet_body)
        return bool(
            points is not None
            and np.all(points >= CABINET_INTERIOR_LOW)
            and np.all(points <= CABINET_INTERIOR_HIGH)
            and contact_between(mujoco_env, self.eraser.simulation_name, cabinet_body)
            and not hand_contact(mujoco_env, self.eraser)
        )

    def compute_reward(self, info: dict[str, Any], *args, **kwargs) -> float:
        del args
        mujoco_env = kwargs.get("mujoco_env")
        valid = self.goal_satisfied(info, mujoco_env)
        self._success_hold_count = self._success_hold_count + 1 if valid else 0
        self.reward = 0.1 * self._success_hold_count
        return float(self.reward)

    def goal_satisfied(self, info: dict[str, Any], mujoco_env) -> bool:
        on_destination = (
            self.check_eraser_on_table(info=info, mujoco_env=mujoco_env)
            if self.selected_goal == "table"
            else self._eraser_on_cabinet(mujoco_env)
        )
        return bool(
            on_destination
            and not self.check_eraser_touching_pen_holder(mujoco_env=mujoco_env)
            and self.check_pen_holder_tipped(info)
        )

    def check_success(self, info: dict[str, Any], *args, **kwargs) -> bool:
        del info, args, kwargs
        return self._success_hold_count >= 10


@TaskRegistry.register("g1_wholebody_goal_football_placement_teleop")
class G1WholebodyGoalFootballPlacementTaskTeleop(GoalObjectTask):
    uid = "g1_wholebody_goal_football_placement_teleop"
    label = "G1 TELEOP Goal Football Placement"
    description = "A task where the G1 robot must place a football at the instructed destination."
    goals = ("ground", "cabinet")
    goal_instructions = {
        "ground": "pick up the football from the table and put it on the ground",
        "cabinet": "pick up the football from the table and put it fully inside the cabinet compartment",
    }
    target_asset = FOOTBALL
    target_xy = (-0.15, -0.27)
    collision_sphere = ((0.0024, 0.0024, 0.11), 0.11)
    cabinet_xy = CABINET_XY
    destination_objects = {"cabinet": (CABINET, "table", cabinet_xy)}

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> None:
        super().reset(seed, options)
        if isinstance(self.target, ObjectActor):
            self.target.mujoco_mass = 0.45
            self.target.mujoco_friction = [0.5, 0.005, 0.0001]
            self.target.mujoco_collision_sphere = self.collision_sphere

    def goal_satisfied(self, info: dict[str, Any], mujoco_env) -> bool:
        del info
        if hand_contact(mujoco_env, self.target):
            return False
        if self.selected_goal == "ground":
            target_pose = body_pose(mujoco_env, self.target.simulation_name)
            return bool(
                target_pose is not None
                and target_pose[0][2] <= 0.20
                and (
                    contact_between(mujoco_env, self.target.simulation_name, "ground")
                    or contact_between(mujoco_env, self.target.simulation_name, "world")
                )
            )
        cabinet = self.destination("cabinet")
        points = actor_points_in_body(mujoco_env, self.target, cabinet.simulation_name)
        return bool(
            points is not None
            and np.all(points >= CABINET_INTERIOR_LOW)
            and np.all(points <= CABINET_INTERIOR_HIGH)
            and contact_between(mujoco_env, self.target.simulation_name, cabinet.simulation_name)
        )
