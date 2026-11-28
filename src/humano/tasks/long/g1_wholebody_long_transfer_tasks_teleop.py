from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
import transforms3d as t3d

from humano.assets import AssetManager
from humano.core.actor import Actor, ObjectActor
from humano.core.layout import Layout
from humano.dr import (
    DistractorDRCfg,
    LanguageDRCfg,
    SpatialDRCfg,
    TabletopSceneDRCfg,
    TargetDRCfg,
)
from humano.dr.types import Box
from humano.tasks.advanced.g1_wholebody_table_book_to_shelf_teleop import (
    _BOOK_ASSET_ID,
    _BOOK_LOAD_ID,
    _BOOK_PLACE_ORIENTATION,
    _SHELF_ASSET_ID,
    _SHELF_LOAD_ID,
    _SHELF_QUATERNION,
    _SHELF_XY,
)
from humano.tasks.advanced.g1_wholebody_table_bread_to_tray_teleop import (
    _BREAD_ASSET_ID,
    _BREAD_LOAD_ID,
    _TRAY_ASSET_ID,
    _TRAY_LOAD_ID,
)
from humano.tasks.basic.initial_positions import resolve_book_shelf_layer_place_offset
from humano.tasks.simple.g1_wholebody_xmove_pick_teleop import G1WholebodyXMovePickTaskTeleop
from humano.tasks.long.g1_wholebody_long_tasks_teleop import (
    _CAKE_ASSET_ID,
    _CAKE_LOAD_ID,
    _HIGH_TABLE_HEIGHT,
    _PLATE_ASSET_ID,
    _PLATE_INSIDE_NORMAL_RANGE,
    _PLATE_INSIDE_RADIUS_M,
    _PLATE_LOAD_ID,
    _SUCCESS_REWARD_INCREMENT,
    _TRAY_INSIDE_HALF_EXTENTS,
    _TRAY_INSIDE_Z_RANGE,
    _actor_body_name,
    _actor_center_in_actor_local,
    _center_in_plate_region,
    _contact_between,
    _hand_contact,
    _LongTableObjectToDrawerTask,
)
from humano.tasks.registry import TaskRegistry

_SHELF_BOOK_DRAWER_POSITION = [-0.08, 0.18, 0.651]
_TRAY_POSITION = [-0.12, -0.18]
_PLATE_POSITION = [-0.04, 0.20]
_LIFT_TARGET_POSITION = [-0.30, -0.24]
_LIFT_TRAY_POSITION = [0.10, 0.20]
_LIFT_PLATE_POSITION = [0.10, 0.20]
_BREAD_IN_TRAY_HEIGHT_M = 0.07
_CONTAINER_LIFT_THRESHOLD_M = 0.10
_BREAD_TRAY_LIFT_THRESHOLD_M = 0.08
_BREAD_TRAY_INSIDE_XY_MARGIN_M = 0.02
_BREAD_TRAY_INSIDE_Z_RANGE = (-0.03, 0.14)
_SHELF_BOOK_INITIAL_LAYER = "layer3"


def _container_lifted_from_table(
    info: dict[str, Any],
    container: Actor,
    initial_z: float | None,
    mujoco_env=None,
) -> bool:
    if initial_z is None or "container" not in info:
        return False
    container_z = float(info["container"][2])
    return bool(
        container_z - initial_z >= _CONTAINER_LIFT_THRESHOLD_M
        and not _contact_between(mujoco_env, _actor_body_name(container), "table")
        and not _hand_contact(mujoco_env, container)
    )


@TaskRegistry.register("g1_wholebody_long_shelf_book_to_table_drawer_teleop")
class G1WholebodyLongShelfBookToTableDrawerTaskTeleop(_LongTableObjectToDrawerTask):
    uid = "g1_wholebody_long_shelf_book_to_table_drawer_teleop"
    label = "G1 TELEOP Long Shelf Book To Table Drawer"
    description = (
        "A task where the G1 robot must move a book from the shelf into a drawer on the high table."
    )

    dr_cfgs = deepcopy(_LongTableObjectToDrawerTask.dr_cfgs)
    dr_cfgs["container"] = TargetDRCfg(asset_id=_SHELF_ASSET_ID)
    dr_cfgs["spatial"] = SpatialDRCfg(
        spatial_mode="fixed",
        robot_region=Box(low=[-0.95, 0.0, 0.0], high=[-0.95, 0.0, 0.0]),
        target_region=Box(low=[-0.08, 0.90], high=[-0.08, 0.90]),
        container_region=Box(low=_SHELF_XY, high=_SHELF_XY),
        distractors_region=Box(low=[0.0, 0.0], high=[0.0, 0.0]),
        articulated_region=Box(
            low=_SHELF_BOOK_DRAWER_POSITION,
            high=_SHELF_BOOK_DRAWER_POSITION,
        ),
        target_stable_indices=[0],
        target_rotate_z=Box(low=0.0, high=0.0),
        container_rotate_z=Box(low=0.0, high=0.0),
        articulated_rotate_z=Box(low=-0.5 * np.pi, high=-0.5 * np.pi),
        fixed_stable_pose_idx=0,
        obj_surface_map={
            "target": "default",
            "container": "default",
            "articulated": "table",
        },
    )
    dr_cfgs["scene"] = TabletopSceneDRCfg(
        scene_mode="fixed",
        table_position=Box(low=[0.3, 0.0], high=[0.3, 0.0]),
        table_height=Box(low=_HIGH_TABLE_HEIGHT, high=_HIGH_TABLE_HEIGHT),
        room_choices=["hssd:scene0"],
        scene_manager="hssd",
    )
    target_asset_id = _BOOK_ASSET_ID
    target_load_id = _BOOK_LOAD_ID
    instruction = "take the book from the shelf and put it into the drawer on the table"

    @property
    def shelf(self) -> Actor:
        actor = self.layout.actors.get("container")
        assert actor is not None, "call reset() first"
        return actor

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> None:
        super().reset(seed, options)
        shelf = self.shelf
        shelf_z = 0.0
        if isinstance(shelf, ObjectActor):
            shelf_z = float(shelf.asset.stable_poses[0][2])
            shelf.mujoco_freejoint = False
        shelf.pose.position = [*_SHELF_XY, shelf_z]
        shelf.pose.quaternion = list(_SHELF_QUATERNION)

        shelf_rotation = t3d.quaternions.quat2mat(shelf.pose.quaternion)
        shelf_local_place_offset = np.asarray(
            resolve_book_shelf_layer_place_offset(_SHELF_BOOK_INITIAL_LAYER),
            dtype=np.float64,
        )
        book_position = np.asarray(shelf.pose.position) + shelf_rotation @ shelf_local_place_offset
        self.target.pose.position = book_position.tolist()
        self.target.pose.quaternion = [float(value) for value in _BOOK_PLACE_ORIENTATION]

    def preload_objects(self) -> list[Actor]:
        objects = AssetManager.get("objects")
        return [
            ObjectActor(asset=objects.load(_BOOK_LOAD_ID)),
            ObjectActor(asset=objects.load(_SHELF_LOAD_ID)),
        ]


@TaskRegistry.register("g1_wholebody_long_tray_bread_to_plate_teleop")
class G1WholebodyLongTrayBreadToPlateTaskTeleop(G1WholebodyXMovePickTaskTeleop):
    uid = "g1_wholebody_long_tray_bread_to_plate_teleop"
    label = "G1 TELEOP Long Tray Bread To Plate"
    description = "Move bread from a tray onto a plate on the same high table."

    metadata: dict[str, Any] = deepcopy(G1WholebodyXMovePickTaskTeleop.metadata)
    dr_cfgs = deepcopy(G1WholebodyXMovePickTaskTeleop.dr_cfgs)
    dr_cfgs["language"] = LanguageDRCfg(
        instructions=["take the bread from the tray and place it on the plate"]
    )
    dr_cfgs["target"] = TargetDRCfg(asset_id=_BREAD_ASSET_ID)
    dr_cfgs["container"] = TargetDRCfg(asset_id=_TRAY_ASSET_ID)
    dr_cfgs["distractors"] = DistractorDRCfg(
        res_id="objects",
        number_of_distractors=1,
        allow_duplicates=False,
        include=[_PLATE_LOAD_ID],
    )
    dr_cfgs["spatial"] = SpatialDRCfg(
        spatial_mode="fixed",
        robot_region=Box(low=[-0.95, 0.0, 0.0], high=[-0.95, 0.0, 0.0]),
        target_region=Box(low=_TRAY_POSITION, high=_TRAY_POSITION),
        container_region=Box(low=_TRAY_POSITION, high=_TRAY_POSITION),
        distractors_region=Box(low=_PLATE_POSITION, high=_PLATE_POSITION),
        target_stable_indices=[0],
        target_rotate_z=Box(low=0.0, high=0.0),
        container_rotate_z=Box(low=0.0, high=0.0),
        fixed_stable_pose_idx=0,
        obj_surface_map={"target": "table", "container": "table", "distractor": "table"},
    )
    dr_cfgs["scene"] = TabletopSceneDRCfg(
        scene_mode="fixed",
        table_position=Box(low=[0.3, 0.0], high=[0.3, 0.0]),
        table_height=Box(low=_HIGH_TABLE_HEIGHT, high=_HIGH_TABLE_HEIGHT),
        room_choices=["hssd:scene0"],
        scene_manager="hssd",
    )

    def __init__(self, *args, target: str | None = None, hold_steps: int = 10, **kwargs) -> None:
        self.dr_cfgs = deepcopy(type(self).dr_cfgs)
        self.hold_steps = int(hold_steps)
        self._success_hold_count = 0
        super().__init__(*args, target=target, **kwargs)

    @property
    def tray(self) -> Actor:
        actor = self.layout.actors.get("container")
        assert actor is not None, "call reset() first"
        return actor

    @property
    def plate(self) -> Actor:
        actor = self.layout.actors.get("distractor_0")
        assert actor is not None, "call reset() first"
        return actor

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> None:
        super().reset(seed, options)
        self._success_hold_count = 0
        self.reward = 0.0
        if isinstance(self.tray, ObjectActor):
            self.tray.mujoco_mass = 0.25
        tray_position = np.asarray(self.tray.pose.position, dtype=np.float64)
        self.target.pose.position = (
            tray_position + np.array([0.0, 0.0, _BREAD_IN_TRAY_HEIGHT_M])
        ).tolist()
        if isinstance(self.target, ObjectActor):
            self.target.pose.quaternion = [
                float(value) for value in self.target.asset.stable_poses[0][3:]
            ]

    def check_bread_on_plate(self, info: dict[str, Any] | None = None, *args, **kwargs) -> bool:
        del args
        if info is None or "target" not in info or "distractor_0" not in info:
            return False
        mujoco_env = kwargs.get("mujoco_env", None)
        local = _actor_center_in_actor_local(
            info,
            "target",
            self.target,
            "distractor_0",
            self.plate,
        )
        return bool(
            np.linalg.norm(local[[0, 2]]) <= _PLATE_INSIDE_RADIUS_M
            and _PLATE_INSIDE_NORMAL_RANGE[0] <= float(local[1]) <= _PLATE_INSIDE_NORMAL_RANGE[1]
            and _contact_between(
                mujoco_env, str(self.target.asset.label), str(self.plate.asset.label)
            )
            and not _contact_between(
                mujoco_env,
                str(self.target.asset.label),
                str(self.tray.asset.label),
            )
            and not _hand_contact(mujoco_env, self.target)
        )

    def check_object_in_container(self, *args, **kwargs) -> bool:
        return self.check_bread_on_plate(*args, **kwargs)

    def compute_reward(self, info: dict[str, Any], *args, **kwargs) -> float:
        if self.check_bread_on_plate(info, *args, **kwargs):
            self._success_hold_count += 1
        else:
            self._success_hold_count = 0
        self.reward = _SUCCESS_REWARD_INCREMENT * self._success_hold_count
        return float(self.reward)

    def check_success(self, info: dict[str, Any], *args, **kwargs) -> bool:
        del info, args, kwargs
        return self._success_hold_count >= self.hold_steps

    def preload_objects(self) -> list[Actor]:
        objects = AssetManager.get("objects")
        return [
            ObjectActor(asset=objects.load(_BREAD_LOAD_ID)),
            ObjectActor(asset=objects.load(_TRAY_LOAD_ID)),
            ObjectActor(asset=objects.load(_PLATE_LOAD_ID)),
        ]


class _LongObjectToContainerAndLiftTask(G1WholebodyXMovePickTaskTeleop):
    target_asset_id: str
    target_load_id: str
    container_asset_id: str
    container_load_id: str
    instruction: str

    def __init__(self, *args, target: str | None = None, hold_steps: int = 10, **kwargs) -> None:
        self.dr_cfgs = deepcopy(type(self).dr_cfgs)
        self.hold_steps = int(hold_steps)
        self._success_hold_count = 0
        self._placed_phase = False
        self._container_initial_z: float | None = None
        super().__init__(*args, target=target, **kwargs)

    @property
    def layout(self) -> Layout:
        assert self._layout is not None, "call reset() first"
        return self._layout

    @property
    def container(self) -> Actor:
        actor = self.layout.actors.get("container")
        assert actor is not None, "call reset() first"
        return actor

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> None:
        super().reset(seed, options)
        self._success_hold_count = 0
        self._placed_phase = False
        self.reward = 0.0
        self._container_initial_z = float(self.container.pose.position[2])

    def _placed(self, info: dict[str, Any], mujoco_env=None) -> bool:
        raise NotImplementedError

    def _lifted(self, info: dict[str, Any], mujoco_env=None) -> bool:
        return _container_lifted_from_table(
            info,
            self.container,
            self._container_initial_z,
            mujoco_env=mujoco_env,
        )

    def _contained_after_placement(self, info: dict[str, Any], mujoco_env=None) -> bool:
        return self._placed(info, mujoco_env=mujoco_env)

    def check_object_in_container(
        self, info: dict[str, Any] | None = None, *args, **kwargs
    ) -> bool:
        del args
        if info is None:
            return False
        mujoco_env = kwargs.get("mujoco_env", None)
        if self._placed(info, mujoco_env=mujoco_env):
            self._placed_phase = True
        return bool(
            self._placed_phase
            and self._contained_after_placement(info, mujoco_env=mujoco_env)
            and self._lifted(info, mujoco_env=mujoco_env)
        )

    def compute_reward(self, info: dict[str, Any], *args, **kwargs) -> float:
        if self.check_object_in_container(info, *args, **kwargs):
            self._success_hold_count += 1
        else:
            self._success_hold_count = 0
        self.reward = _SUCCESS_REWARD_INCREMENT * self._success_hold_count
        return float(self.reward)

    def check_success(self, info: dict[str, Any], *args, **kwargs) -> bool:
        del info, args, kwargs
        return self._success_hold_count >= self.hold_steps


@TaskRegistry.register("g1_wholebody_long_bread_to_tray_and_lift_teleop")
class G1WholebodyLongBreadToTrayAndLiftTaskTeleop(_LongObjectToContainerAndLiftTask):
    uid = "g1_wholebody_long_bread_to_tray_and_lift_teleop"
    label = "G1 TELEOP Long Bread To Tray And Lift"
    description = "A task where the G1 robot must put the bread into the tray, then lift the tray."
    target_asset_id = _BREAD_ASSET_ID
    target_load_id = _BREAD_LOAD_ID
    container_asset_id = _TRAY_ASSET_ID
    container_load_id = _TRAY_LOAD_ID
    instruction = "put the bread into the tray, then lift the tray"

    metadata: dict[str, Any] = deepcopy(G1WholebodyXMovePickTaskTeleop.metadata)
    dr_cfgs = deepcopy(G1WholebodyXMovePickTaskTeleop.dr_cfgs)
    dr_cfgs["language"] = LanguageDRCfg(instructions=[instruction])
    dr_cfgs["target"] = TargetDRCfg(asset_id=_BREAD_ASSET_ID)
    dr_cfgs["container"] = TargetDRCfg(asset_id=_TRAY_ASSET_ID)
    dr_cfgs["distractors"] = DistractorDRCfg(
        res_id="objects", number_of_distractors=0, allow_duplicates=False
    )
    dr_cfgs["spatial"] = SpatialDRCfg(
        spatial_mode="fixed",
        robot_region=Box(low=[-0.95, 0.0, 0.0], high=[-0.95, 0.0, 0.0]),
        target_region=Box(low=_LIFT_TARGET_POSITION, high=_LIFT_TARGET_POSITION),
        container_region=Box(low=_LIFT_TRAY_POSITION, high=_LIFT_TRAY_POSITION),
        distractors_region=Box(low=[0.0, 0.0], high=[0.0, 0.0]),
        target_stable_indices=[0],
        target_rotate_z=Box(low=0.0, high=0.0),
        container_rotate_z=Box(low=0.0, high=0.0),
        fixed_stable_pose_idx=0,
        obj_surface_map={"target": "table", "container": "table"},
    )
    dr_cfgs["scene"] = TabletopSceneDRCfg(
        scene_mode="fixed",
        table_position=Box(low=[0.3, 0.0], high=[0.3, 0.0]),
        table_height=Box(low=_HIGH_TABLE_HEIGHT, high=_HIGH_TABLE_HEIGHT),
        room_choices=["hssd:scene0"],
        scene_manager="hssd",
    )

    def _bread_inside_tray(self, info: dict[str, Any], *, relaxed: bool) -> bool:
        if "target" not in info or "container" not in info:
            return False
        local = _actor_center_in_actor_local(
            info, "target", self.target, "container", self.container
        )
        half_extents = _TRAY_INSIDE_HALF_EXTENTS
        z_range = _TRAY_INSIDE_Z_RANGE
        if relaxed:
            half_extents = half_extents + _BREAD_TRAY_INSIDE_XY_MARGIN_M
            z_range = _BREAD_TRAY_INSIDE_Z_RANGE
        return bool(
            np.all(np.abs(local[:2]) <= half_extents)
            and z_range[0] <= float(local[2]) <= z_range[1]
        )

    def _placed(self, info: dict[str, Any], mujoco_env=None) -> bool:
        return bool(
            self._bread_inside_tray(info, relaxed=False)
            and _contact_between(
                mujoco_env, _actor_body_name(self.target), _actor_body_name(self.container)
            )
            and not _hand_contact(mujoco_env, self.target)
        )

    def _contained_after_placement(self, info: dict[str, Any], mujoco_env=None) -> bool:
        return bool(
            self._bread_inside_tray(info, relaxed=True)
            and not _hand_contact(mujoco_env, self.target)
        )

    def _lifted(self, info: dict[str, Any], mujoco_env=None) -> bool:
        del mujoco_env
        if self._container_initial_z is None or "container" not in info:
            return False
        return bool(
            float(info["container"][2]) - self._container_initial_z >= _BREAD_TRAY_LIFT_THRESHOLD_M
        )

    def preload_objects(self) -> list[Actor]:
        objects = AssetManager.get("objects")
        return [
            ObjectActor(asset=objects.load(_BREAD_LOAD_ID)),
            ObjectActor(asset=objects.load(_TRAY_LOAD_ID)),
        ]


@TaskRegistry.register("g1_wholebody_long_cake_to_plate_and_lift_teleop")
class G1WholebodyLongCakeToPlateAndLiftTaskTeleop(_LongObjectToContainerAndLiftTask):
    uid = "g1_wholebody_long_cake_to_plate_and_lift_teleop"
    label = "G1 TELEOP Long Cake To Plate And Lift"
    description = "A task where the G1 robot must put the cake into the plate, then lift the plate."
    target_asset_id = _CAKE_ASSET_ID
    target_load_id = _CAKE_LOAD_ID
    container_asset_id = _PLATE_ASSET_ID
    container_load_id = _PLATE_LOAD_ID
    instruction = "put the cake into the plate, then lift the plate"

    metadata: dict[str, Any] = deepcopy(G1WholebodyXMovePickTaskTeleop.metadata)
    dr_cfgs = deepcopy(G1WholebodyXMovePickTaskTeleop.dr_cfgs)
    dr_cfgs["language"] = LanguageDRCfg(instructions=[instruction])
    dr_cfgs["target"] = TargetDRCfg(asset_id=_CAKE_ASSET_ID)
    dr_cfgs["container"] = TargetDRCfg(asset_id=_PLATE_ASSET_ID)
    dr_cfgs["distractors"] = DistractorDRCfg(
        res_id="objects", number_of_distractors=0, allow_duplicates=False
    )
    dr_cfgs["spatial"] = SpatialDRCfg(
        spatial_mode="fixed",
        robot_region=Box(low=[-0.95, 0.0, 0.0], high=[-0.95, 0.0, 0.0]),
        target_region=Box(low=_LIFT_TARGET_POSITION, high=_LIFT_TARGET_POSITION),
        container_region=Box(low=_LIFT_PLATE_POSITION, high=_LIFT_PLATE_POSITION),
        distractors_region=Box(low=[0.0, 0.0], high=[0.0, 0.0]),
        target_stable_indices=[0],
        target_rotate_z=Box(low=0.0, high=0.0),
        container_rotate_z=Box(low=0.0, high=0.0),
        fixed_stable_pose_idx=0,
        obj_surface_map={"target": "table", "container": "table"},
    )
    dr_cfgs["scene"] = TabletopSceneDRCfg(
        scene_mode="fixed",
        table_position=Box(low=[0.3, 0.0], high=[0.3, 0.0]),
        table_height=Box(low=_HIGH_TABLE_HEIGHT, high=_HIGH_TABLE_HEIGHT),
        room_choices=["hssd:scene0"],
        scene_manager="hssd",
    )

    def _placed(self, info: dict[str, Any], mujoco_env=None) -> bool:
        if "target" not in info or "container" not in info:
            return False
        local = _actor_center_in_actor_local(
            info, "target", self.target, "container", self.container
        )
        return bool(
            _center_in_plate_region(local)
            and _contact_between(
                mujoco_env, _actor_body_name(self.target), _actor_body_name(self.container)
            )
            and not _hand_contact(mujoco_env, self.target)
        )

    def _lifted(self, info: dict[str, Any], mujoco_env=None) -> bool:
        """A plate is expected to remain in the operator's hands while lifted."""
        del mujoco_env
        if self._container_initial_z is None or "container" not in info:
            return False
        return bool(
            float(info["container"][2]) - self._container_initial_z >= _CONTAINER_LIFT_THRESHOLD_M
        )

    def preload_objects(self) -> list[Actor]:
        objects = AssetManager.get("objects")
        return [
            ObjectActor(asset=objects.load(_CAKE_LOAD_ID)),
            ObjectActor(asset=objects.load(_PLATE_LOAD_ID)),
        ]
