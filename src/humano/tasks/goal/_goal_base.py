from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import transforms3d as t3d
import trimesh
from gymnasium import spaces

from humano.assets import AssetManager
from humano.core.actor import Actor, ObjectActor
from humano.core.layout import Layout
from humano.dr import (
    ArticulatedObjectDrCfg,
    DistractorDRCfg,
    LanguageDRCfg,
    SpatialDRCfg,
    TabletopSceneDRCfg,
    TargetDRCfg,
)
from humano.dr.types import Box
from humano.robots.protocols import Controllable
from humano.tasks.simple.g1_wholebody_xmove_pick_teleop import G1WholebodyXMovePickTaskTeleop

APPLE = ("objects:apple/apple_001", "apple/apple_001")
BANANA = ("objects:banana/banana_020", "banana/banana_020")
PLATE = ("objects:plate/plate_3", "plate/plate_3")
BREAD = ("objects:bread/bread_006", "bread/bread_006")
MILK = (
    "objects:Carton_Toy_3C923930_Milk/Carton_Toy_3C923930_Milk_1",
    "Carton_Toy_3C923930_Milk/Carton_Toy_3C923930_Milk_1",
)
PEN = ("objects:pen/pen", "pen/pen")
BOOK = (
    "objects:Book_9388144295_Homer/Book_9388144295_Homer",
    "Book_9388144295_Homer/Book_9388144295_Homer",
)
BOTTLE = ("objects:bottle/bottle_048", "bottle/bottle_048")
FOOTBALL = ("objects:football/football", "football/football")

BASKET = ("objects:picnic_basket/picnic_basket_009", "picnic_basket/picnic_basket_009")
BOX = ("objects:storage_box/storage_box_003", "storage_box/storage_box_003")
TRAY = ("objects:tray/tray_001", "tray/tray_001")
TRASH_CAN = ("objects:dustbin/dustbin_011", "dustbin/dustbin_011")
PEN_HOLDER = ("objects:pen_holder001/pen_holder001", "pen_holder001/pen_holder001")
SHELF = ("objects:shelf/shelf_001", "shelf/shelf_001")

CABINET = (
    "objects:white_cabinet/white_cabinet_1",
    "white_cabinet/white_cabinet_1",
)
# Keep the cabinet fully on the fixed scene0 tabletop (1.25 x 0.79 m),
# including its collision bounds after the -90 degree yaw rotation.
CABINET_XY = (-0.08, 0.15)
CABINET_QUATERNION = (0.70710678, 0.0, 0.0, -0.70710678)
CABINET_INTERIOR_LOW = np.array([-0.18, -0.15, 0.02], dtype=np.float64)
CABINET_INTERIOR_HIGH = np.array([0.18, 0.11, 0.40], dtype=np.float64)

TABLE_HEIGHT = 0.65
HOLD_STEPS = 10


def body_pose(mujoco_env, body_name: str) -> tuple[np.ndarray, np.ndarray] | None:
    if mujoco_env is None:
        return None
    try:
        body = mujoco_env.mjData.body(body_name)
    except KeyError:
        return None
    return (
        np.asarray(body.xpos, dtype=np.float64),
        np.asarray(body.xmat, dtype=np.float64).reshape(3, 3),
    )


def body_name(model, geom_id: int) -> str:
    return model.body(int(model.geom(int(geom_id)).bodyid)).name


def contact_between(mujoco_env, name_a: str, name_b: str) -> bool:
    if mujoco_env is None:
        return False
    for contact_id in range(mujoco_env.mjData.ncon):
        contact = mujoco_env.mjData.contact[contact_id]
        first = body_name(mujoco_env.mjModel, contact.geom1)
        second = body_name(mujoco_env.mjModel, contact.geom2)
        if (name_a in first and name_b in second) or (name_a in second and name_b in first):
            return True
    return False


def hand_contact(mujoco_env, actor: Actor) -> bool:
    return contact_between(mujoco_env, actor.simulation_name, "hand")


def actor_local_position(mujoco_env, actor: Actor, body: str) -> np.ndarray | None:
    actor_pose = body_pose(mujoco_env, actor.simulation_name)
    destination_pose = body_pose(mujoco_env, body)
    if actor_pose is None or destination_pose is None:
        return None
    actor_position, _ = actor_pose
    destination_position, destination_rotation = destination_pose
    return (actor_position - destination_position) @ destination_rotation


@lru_cache(maxsize=32)
def collision_points(asset_uid: str) -> np.ndarray:
    asset = AssetManager.get("objects").load(asset_uid)
    points = []
    for mesh_path in asset.collision_meshes_mujoco:
        mesh = trimesh.load_mesh(Path(mesh_path), force="mesh", process=False)
        if not mesh.is_empty and len(mesh.vertices):
            points.append(np.asarray(mesh.vertices, dtype=np.float64))
    if not points:
        raise RuntimeError(f"No collision vertices found for objects:{asset_uid}")
    return np.vstack(points)


def actor_points_in_body(mujoco_env, actor: ObjectActor, body: str) -> np.ndarray | None:
    actor_pose = body_pose(mujoco_env, actor.simulation_name)
    destination_pose = body_pose(mujoco_env, body)
    if actor_pose is None or destination_pose is None:
        return None
    actor_position, actor_rotation = actor_pose
    destination_position, destination_rotation = destination_pose
    points = collision_points(str(actor.asset.uid)) * actor.asset.scale
    world = points @ actor_rotation.T + actor_position
    return (world - destination_position) @ destination_rotation


def configure_fixed_cabinet(
    cabinet: Actor, xy: tuple[float, float] = CABINET_XY
) -> None:
    cabinet.mujoco_freejoint = False
    stable_z = float(cabinet.asset.stable_poses[0][2])
    cabinet.pose.position = [xy[0], xy[1], TABLE_HEIGHT + stable_z]
    cabinet.pose.quaternion = list(CABINET_QUATERNION)


class GoalSelectionMixin:
    goals: tuple[str, ...]
    goal_instructions: dict[str, str]

    def __init__(self, *args, goal: str = "random", **kwargs) -> None:
        allowed = ("random", *self.goals)
        if goal not in allowed:
            raise ValueError(f"Invalid goal {goal!r}; expected one of {allowed}")
        self.goal_mode = goal
        self.selected_goal: str | None = None
        self._goal_rng = np.random.default_rng()
        super().__init__(*args, **kwargs)

    def _select_goal(self, seed: int | None, options: dict[str, Any] | None) -> None:
        saved = None
        if options is not None and isinstance(options.get("state_dict"), dict):
            saved = options["state_dict"].get("selected_goal")
        if saved is not None:
            if saved not in self.goals:
                raise ValueError(f"Saved goal {saved!r} is invalid for {self.uid}")
            self.selected_goal = str(saved)
        elif self.goal_mode != "random":
            self.selected_goal = self.goal_mode
        else:
            if seed is not None:
                self._goal_rng = np.random.default_rng(seed)
            self.selected_goal = self.goals[int(self._goal_rng.integers(0, len(self.goals)))]

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> None:
        self._select_goal(seed, options)
        instruction = self.goal_instructions[self.selected_goal]
        language_dr = self.dr.get_randomizer("language")
        assert language_dr is not None
        language_dr.cfg.instructions = [instruction]
        self.dr_cfgs["language"].instructions = [instruction]
        super().reset(seed, options)

    def state_dict(self) -> dict[str, Any]:
        state = super().state_dict()
        state["selected_goal"] = self.selected_goal
        return state


class GoalObjectTask(GoalSelectionMixin, G1WholebodyXMovePickTaskTeleop):
    metadata: dict[str, Any] = deepcopy(G1WholebodyXMovePickTaskTeleop.metadata)
    target_asset: tuple[str, str]
    target_surface = "table"
    destination_objects: dict[str, tuple[tuple[str, str], str, tuple[float, float]]] = {}
    articulated_asset: str | None = None
    articulated_goal: str | None = None
    articulated_qpos: dict[str, float] | None = None
    cabinet_xy: tuple[float, float] = CABINET_XY
    articulated_position = (-0.08, 0.72, 0.0)
    articulated_rotate_z = -0.5 * np.pi
    target_xy = (-0.30, -0.22)
    hold_steps = HOLD_STEPS

    def __init__(self, *args, target: str | None = None, **kwargs) -> None:
        self.dr_cfgs = deepcopy(G1WholebodyXMovePickTaskTeleop.dr_cfgs)
        self.dr_cfgs["language"] = LanguageDRCfg(
            instructions=[self.goal_instructions[self.goals[0]]]
        )
        self.dr_cfgs["target"] = TargetDRCfg(asset_id=self.target_asset[0])
        loads = [spec[0][1] for spec in self.destination_objects.values()]
        if loads:
            self.dr_cfgs["distractors"] = DistractorDRCfg(
                res_id="objects", number_of_distractors=len(loads), include=loads
            )
        else:
            self.dr_cfgs.pop("distractors", None)
        if self.articulated_asset is not None:
            self.dr_cfgs["articulated"] = ArticulatedObjectDrCfg(
                asset_id=self.articulated_asset,
                articulate_init_joint_qpos=self.articulated_qpos,
            )
        else:
            self.dr_cfgs.pop("articulated", None)
        regions = [
            Box(low=list(spec[2]), high=list(spec[2])) for spec in self.destination_objects.values()
        ]
        surfaces = [spec[1] for spec in self.destination_objects.values()]
        self.dr_cfgs["spatial"] = SpatialDRCfg(
            spatial_mode="fixed",
            robot_region=Box(low=[-0.95, 0.0, 0.0], high=[-0.95, 0.0, 0.0]),
            target_region=Box(low=list(self.target_xy), high=list(self.target_xy)),
            distractors_region=regions or Box(low=[0.0, 0.0], high=[0.0, 0.0]),
            articulated_region=Box(
                low=list(self.articulated_position), high=list(self.articulated_position)
            )
            if self.articulated_asset
            else None,
            articulated_rotate_z=Box(low=self.articulated_rotate_z, high=self.articulated_rotate_z)
            if self.articulated_asset
            else None,
            target_stable_indices=[0],
            target_rotate_z=Box(low=0.0, high=0.0),
            fixed_stable_pose_idx=0,
            obj_surface_map={"target": self.target_surface, "distractor": surfaces or "table"},
        )
        self.dr_cfgs["scene"] = TabletopSceneDRCfg(
            scene_mode="fixed",
            table_position=Box(low=[0.30, 0.0], high=[0.30, 0.0]),
            table_height=Box(low=TABLE_HEIGHT, high=TABLE_HEIGHT),
            room_choices=["hssd:scene0"],
            scene_manager="hssd",
        )
        self._destinations: dict[str, Actor] = {}
        self._success_hold_count = 0
        super().__init__(*args, target=target, **kwargs)

    @property
    def layout(self) -> Layout:
        assert self._layout is not None, "call reset() first"
        return self._layout

    @property
    def action_space(self) -> spaces.Space:
        assert isinstance(self.robot, Controllable)
        return self.robot.controller.action_space

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> None:
        super().reset(seed, options)
        actors = list(self.layout.actors.values())
        self._destinations = {}
        for goal, (asset, _, _) in self.destination_objects.items():
            match = next(
                actor
                for actor in actors
                if str(getattr(getattr(actor, "asset", None), "uid", "")) == asset[1]
            )
            self._destinations[goal] = match
        if self.articulated_asset is not None:
            assert self.articulated_goal is not None
            self._destinations[self.articulated_goal] = self.layout.actors["articulated"]
        cabinet = self._destinations.get("cabinet")
        if cabinet is not None and str(cabinet.asset.uid) == CABINET[1]:
            configure_fixed_cabinet(cabinet, xy=self.cabinet_xy)
        self._success_hold_count = 0
        self.reward = 0.0

    def destination(self, goal: str | None = None) -> Actor:
        return self._destinations[goal or self.selected_goal]

    def goal_satisfied(self, info: dict[str, Any], mujoco_env) -> bool:
        raise NotImplementedError

    def compute_reward(self, info: dict[str, Any], *args, **kwargs) -> float:
        del args
        if self.goal_satisfied(info, kwargs.get("mujoco_env")):
            self._success_hold_count += 1
        else:
            self._success_hold_count = 0
        self.reward = 0.1 * self._success_hold_count
        return float(self.reward)

    def check_success(self, info: dict[str, Any], *args, **kwargs) -> bool:
        del info, args, kwargs
        return self._success_hold_count >= self.hold_steps

    def preload_objects(self) -> list[Actor]:
        objects = AssetManager.get("objects")
        loads = [
            self.target_asset[1],
            *(spec[0][1] for spec in self.destination_objects.values()),
        ]
        return [ObjectActor(asset=objects.load(load)) for load in loads]
