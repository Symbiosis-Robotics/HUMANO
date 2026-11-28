"""
HUMANO: Humanoid Benchmark and Learning Framework

Originally derived from the SIMPLE project.
Copyright (c) 2025 Songlin Wei and Contributors
Copyright (c) 2026 HUMANO Contributors
Licensed under the terms in LICENSE file.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, List, Type

import numpy as np
import transforms3d as t3d
import trimesh

from humano.core.actor import ArticulatedObjectActor, ObjectActor, RobotActor

# if TYPE_CHECKING:
from humano.core.layout import Layout
from humano.core.object import Object, SpatialAnnotated
from humano.core.randomizer import Randomizer, RandomizerCfg

from .types import Box

# def __init__(
#     self,
#     robot_region: Box | None = None,
#     target_region: Box | None = None,
#     distractors_region: Box | None = None,
#     rand_stable_pose: bool = True,
#     # rand_grasp_psoe: bool = False
#     collide_threshold: float = 0.005
# ) -> None:

#     self.robot_region = robot_region
#     self.target_region = target_region
#     self.distractors_region = distractors_region

#     self.rand_stable_pose = rand_stable_pose
#     # self.rand_grasp_pose = rand_grasp_psoe
#     self.collide_threshold = collide_threshold


class SpatialDR(Randomizer):
    cfg: SpatialDRCfg  # type: ignore

    def __init__(
        self,
        # robot_region: Box | None = None,
        # target_region: Box | None = None,
        # distractors_region: Box | None = None,
        # rand_stable_pose: bool = True,
        # # rand_grasp_psoe: bool = False
        # collide_threshold: float = 0.005
        cfg: SpatialDRCfg,
    ) -> None:
        super().__init__(cfg)
        # self.robot_region = robot_region
        # self.target_region = target_region
        # self.distractors_region = distractors_region

        # self.rand_stable_pose = rand_stable_pose
        # # self.rand_grasp_pose = rand_grasp_psoe
        # self.collide_threshold = collide_threshold
        # self.cfg = cfg
        self.spatial_mode = cfg.spatial_mode
        self.fixed_stable_pose_idx = cfg.fixed_stable_pose_idx
        # self._inner_state = {} # FIXME

    def state_dict(self) -> dict[str, Any]:
        return super().state_dict()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        return super().load_state_dict(state_dict)

    def __call__(self, split: str, layout: Layout, table_height: float = 0.0, *, seed=None) -> None:
        """Randomize the spatial arrangement of robot and objects in the layout.

        Args:
            split (str): The data split, e.g., "train", "val", "test".
            layout (Layout): The layout containing the robot and objects to be arranged.
            table_height (float): The height of the table surface.

            HACK: we always has ground plane at z=0, and objects are on the GROUND plane.
            So table_height is only used to adjust the robot base height.

        """
        if self.cfg.placements_from_scene:
            return self._place_configured_actors(layout, seed)

        self.collision_manager = trimesh.collision.CollisionManager()

        # self._inner_state = {} # keep track of placed objects
        if self._inner_state is None:
            self._inner_state = {}

        # Determine surface heights for different regions
        # Ground objects use the world floor, while the default surface remains
        # the tabletop height for legacy configurations without an explicit map.
        surface_heights = {"default": table_height, "ground": 0.0}
        scene = getattr(layout, "scene", None)
        if scene is not None and hasattr(scene, "table") and scene.table is not None:
            surface_heights["table"] = (
                scene.table.pose.position[2] + 0.5 * scene.table.size[2]
            )

        if scene is not None and hasattr(scene, "table2") and scene.table2 is not None:
            surface_heights["table2"] = (
                scene.table2.pose.position[2] + 0.5 * scene.table2.size[2]
            )

        distractor_count = 0
        for objtype, obj in layout.actors.items():
            if objtype == "articulated":
                assert isinstance(obj, ArticulatedObjectActor)
                if (
                    self._inner_state is not None
                    and f"articulated_{str(obj.uid)}" in self._inner_state
                ):
                    state = self._inner_state[f"articulated_{str(obj.uid)}"]
                    obj.pose.position = state["position"]
                    obj.pose.quaternion = state["quaternion"]
                else:
                    if self.cfg.articulated_region is not None:
                        if self.spatial_mode == "fixed":
                            position = self.cfg.articulated_region.middle()

                            rand_z = self.cfg.articulated_rotate_z.sample()
                            rand_ori_mat = t3d.euler.euler2mat(0.0, 0.0, rand_z)
                            obj.pose.position = position
                            obj.pose.quaternion = t3d.quaternions.mat2quat(rand_ori_mat).tolist()

                            self._inner_state[f"articulated_{str(obj.uid)}"] = {
                                "position": position,
                                "quaternion": obj.pose.quaternion,
                            }

                        else:
                            position = self.cfg.articulated_region.sample()
                            rand_z = self.cfg.articulated_rotate_z.sample()
                            rand_ori_mat = t3d.euler.euler2mat(0.0, 0.0, rand_z)
                            surface_height = surface_heights.get(
                                surface_name, surface_heights["default"]
                            )
                            if len(position) == 2:
                                position.append(0.0)
                                position[2] = surface_height

                            obj.pose.position = position
                            obj.pose.quaternion = t3d.quaternions.mat2quat(rand_ori_mat).tolist()

                            self._inner_state[f"articulated_{str(obj.uid)}"] = {
                                "position": position,
                                "quaternion": obj.pose.quaternion,
                            }

            if objtype == "robot":
                assert isinstance(obj, RobotActor)
                if self._inner_state is not None and str(obj.robot.uid) in self._inner_state:
                    state = self._inner_state[str(obj.robot.uid)]
                    robot_position = state["position"]
                    obj.pose.quaternion = state["quaternion"]

                else:
                    if self.cfg.robot_region is not None:
                        if self.spatial_mode == "fixed":
                            robot_position = self.cfg.robot_region.middle()
                        else:
                            robot_position = self.cfg.robot_region.sample()
                        if len(robot_position) == 2:
                            assert isinstance(robot_position, list)
                            # obj.pose.position = [robot_position[0], robot_position[1], 0.0]
                            # robot_position = obj.pose.position
                            robot_position.append(0.0)
                            robot_position[2] = -table_height
                        else:
                            robot_position = robot_position
                    else:
                        robot_position = [0.0, 0.0, 0.0]
                    # print(f"Placed robot at {robot_position}")
                obj.pose.position = robot_position

                if (
                    self.cfg.robot_orientation_region is not None
                    and str(obj.robot.uid) not in self._inner_state
                ):
                    if self.spatial_mode == "fixed":
                        robot_orientation = self.cfg.robot_orientation_region.middle()
                    else:
                        robot_orientation = self.cfg.robot_orientation_region.sample()
                    obj.pose.quaternion = robot_orientation
                self._inner_state[obj.robot.uid] = {
                    "position": robot_position,
                    "quaternion": obj.pose.quaternion,
                }
            elif objtype == "container":
                if self.cfg.rand_stable_pose and isinstance(obj.asset, SpatialAnnotated):
                    if (
                        self._inner_state is not None
                        and f"container_{obj.uid}" in self._inner_state
                    ):
                        state = self._inner_state[f"container_{obj.uid}"]
                        obj.pose.position = state["position"]
                        obj.pose.quaternion = state["quaternion"]
                    else:
                        surface_name = (
                            self.cfg.obj_surface_map.get("container", "table")
                            if self.cfg.obj_surface_map
                            else "table"
                        )
                        surface_height = surface_heights.get(
                            surface_name, surface_heights["default"]
                        )
                        self._random_place_one_object(
                            obj, self.cfg.container_region, objtype, surface_height=surface_height
                        )
            elif objtype == "target":  # isinstance(obj, Object):
                # if self.cfg.target_region is not None:
                #     #TODO Randomize target position within the specified region
                #     pass
                # # obj.pose.position = [0.3, 0.0, 0.0]

                if self.cfg.rand_stable_pose and isinstance(obj.asset, SpatialAnnotated):
                    if self._inner_state is not None and str(obj.uid) in self._inner_state:
                        state = self._inner_state[str(obj.uid)]
                        obj.pose.position = state["position"]
                        obj.pose.quaternion = state["quaternion"]
                    else:
                        surface_name = (
                            self.cfg.obj_surface_map.get("target", "table")
                            if self.cfg.obj_surface_map
                            else "table"
                        )
                        surface_height = surface_heights.get(
                            surface_name, surface_heights["default"]
                        )
                        self._random_place_one_object(
                            obj, self.cfg.target_region, objtype, surface_height=surface_height
                        )
                        # self._inner_state[obj.uid] = {
                        #     "position": [-0.4, 0.0, 0.04],
                        #     "quaternion": [1,0,0,0]
                        # }

            elif "distractor" in objtype:
                if self.cfg.distractors_region is not None:
                    pass

                if self.cfg.rand_stable_pose and isinstance(obj.asset, SpatialAnnotated):
                    if self._inner_state is not None and str(obj.uid) in self._inner_state:
                        state = self._inner_state[str(obj.uid)]
                        obj.pose.position = state["position"]
                        obj.pose.quaternion = state["quaternion"]
                    else:
                        surface_entry = (
                            self.cfg.obj_surface_map.get(objtype)
                            or self.cfg.obj_surface_map.get("distractor", "table")
                            if self.cfg.obj_surface_map
                            else "table"
                        )
                        if isinstance(surface_entry, list):
                            surface_name = surface_entry[distractor_count % len(surface_entry)]
                        else:
                            surface_name = surface_entry

                        surface_height = surface_heights.get(
                            surface_name, surface_heights["default"]
                        )

                        region = self.cfg.distractors_region
                        if isinstance(region, list):
                            region = region[distractor_count % len(region)]

                        self._random_place_one_object(
                            obj, region, objtype, surface_height=surface_height
                        )
                        distractor_count += 1

    def _place_configured_actors(self, layout, seed):
        from humano.core.types import Pose

        rules = layout.scene.metadata["placements"]
        saved_robot = (self._inner_state or {}).get(layout.robot.robot.uid)
        if self._inner_state is None or "poses" not in self._inner_state:
            rng = np.random.default_rng(seed)
            vertices = {}
            for key, rule in rules.items():
                if "stable_pose" in rule:
                    asset = layout.actors[key].asset
                    vertices[key] = np.concatenate(
                        [
                            np.asarray(trimesh.load_mesh(path).vertices)
                            for path in asset.collision_meshes_mujoco
                        ]
                    )
            for _ in range(100):
                poses = {}
                for key, rule in rules.items():
                    offset = np.asarray(
                        rule["fixed"]
                        if self.spatial_mode == "fixed"
                        else rng.uniform(rule["low"], rule["high"])
                    )
                    yaw = (
                        rule.get("fixed_yaw", 0.0)
                        if self.spatial_mode == "fixed"
                        else rng.uniform(*rule["yaw"])
                    )
                    parent = rule.get("relative_to")
                    if parent:
                        parent_pose = poses[parent]
                        parent_yaw = t3d.euler.quat2euler(parent_pose["quaternion"])[2]
                        parent_yaw += rule.get("parent_yaw_offset", 0.0)
                        rotation = t3d.euler.euler2mat(0, 0, parent_yaw)
                        position = np.asarray(parent_pose["position"]) + rotation @ offset
                        yaw += parent_yaw
                        if rule.get("face_parent"):
                            yaw = parent_yaw + np.arctan2(-offset[1], -offset[0])
                            if self.spatial_mode != "fixed":
                                yaw += rng.uniform(*rule["heading_jitter"])
                    else:
                        position = offset.copy()
                    quaternion = t3d.euler.euler2quat(0, 0, yaw)
                    if "stable_pose" in rule:
                        stable = layout.actors[key].asset.stable_poses[rule["stable_pose"]]
                        quaternion = t3d.quaternions.qmult(
                            quaternion, rule.get("stable_quaternion", stable[3:])
                        )  # (4,), unit quaternion, world
                        if "stable_quaternion" in rule:
                            points = (
                                vertices[key] @ t3d.quaternions.quat2mat(quaternion).T
                            )  # (N, 3), m, world axes
                            position[:2] -= (points[:, :2].min(0) + points[:, :2].max(0)) / 2
                            position[2] += (
                                rule["support_z"] - points[:, 2].min() + rule.get("gap", 0)
                            )
                        else:
                            position[2] += rule["support_z"] + stable[2] + rule.get("gap", 0)
                    if "world_z" in rule:
                        position[2] = rule["world_z"]
                    poses[key] = {"position": position.tolist(), "quaternion": quaternion.tolist()}
                if self._configured_poses_valid(layout, poses, vertices):
                    if saved_robot is not None:
                        poses["robot"] = saved_robot
                    self._inner_state = {
                        "seed": seed,
                        "poses": poses,
                        layout.robot.robot.uid: poses["robot"],
                    }
                    break
            else:
                raise ValueError(
                    "Could not sample a non-overlapping configured layout after 100 attempts"
                )
        for key, pose in self._inner_state["poses"].items():
            if key in layout.actors:
                layout.actors[key].pose = Pose(pose["position"], pose["quaternion"])

    def _configured_poses_valid(self, layout, poses, vertices):
        rules = layout.scene.metadata["placements"]
        placed = []
        for key, points in vertices.items():
            rule, pose = rules[key], poses[key]
            parent = poses[rule["relative_to"]]
            rotation = t3d.quaternions.quat2mat(pose["quaternion"])
            parent_rotation = t3d.quaternions.quat2mat(parent["quaternion"])
            local = (
                points @ rotation.T + pose["position"] - np.asarray(parent["position"])
            ) @ parent_rotation
            low, high = local[:, :2].min(0), local[:, :2].max(0)
            half = np.asarray(rule["support_half"])
            if np.any(low < -half) or np.any(high > half):
                return False
            if any(
                other_parent == rule["relative_to"]
                and np.all(high + self.cfg.collide_threshold > a)
                and np.all(b + self.cfg.collide_threshold > low)
                for other_parent, a, b in placed
            ):
                return False
            placed.append((rule["relative_to"], low, high))
        if self.spatial_mode == "fixed":
            return True
        obstacles = np.asarray(layout.scene.metadata.get("obstacles", []))
        if not obstacles.size:
            return True
        for key, rule in rules.items():
            if "clearance_half" not in rule:
                continue
            pose = poses[key]
            half = np.asarray(rule["clearance_half"])
            rotation = t3d.quaternions.quat2mat(pose["quaternion"])[:2, :2]
            extent = np.abs(rotation) @ half if rule.get("clearance_rotate", True) else half
            center = np.asarray(pose["position"])[:2]
            if np.any(
                np.all(center + extent > obstacles[:, 0], axis=1)
                & np.all(obstacles[:, 1] > center - extent, axis=1)
            ):
                return False
        return True

    def _random_place_one_object(
        self, obj: Object, region: Box, objtype: str, surface_height: float = 0.0
    ):

        object_msh = trimesh.load_mesh(obj.asset.collision_mesh_curobo)
        for _ in range(100):
            if objtype == "container":
                stable_pose = obj.asset.stable_poses[
                    0
                ]  # only use the first stable pose for container
            elif self.spatial_mode == "fixed":
                stable_pose = obj.asset.stable_poses[self.fixed_stable_pose_idx]
            else:
                # stable_pose = random.choice(obj.asset.stable_poses)
                if self.cfg.target_stable_indices is not None:
                    stable_pose = obj.asset.stable_poses[
                        random.choice(self.cfg.target_stable_indices)
                    ]
                else:
                    stable_pose = random.choice(obj.asset.stable_poses)
            p = np.zeros((3,), dtype=np.float32)
            p[:2] += np.asarray(region.sample(), dtype=np.float32)  # random xy
            p[2] = stable_pose[2] + surface_height
            obj.pose.position = p.tolist()
            # print(f"Trying to place object {obj.uid} at position {p.tolist()}")

            stable_ori_mat = t3d.quaternions.quat2mat(stable_pose[3:])  # 3x3
            if objtype == "target":
                rand_z = self.cfg.target_rotate_z.sample()

                rand_ori_mat = t3d.euler.euler2mat(0.0, 0.0, rand_z)

            elif objtype == "container" and self.cfg.container_rotate_z is not None:
                rand_z = self.cfg.container_rotate_z.sample()
                rand_ori_mat = t3d.euler.euler2mat(0.0, 0.0, rand_z)
            elif objtype.startswith("distractor") and self.cfg.distractor_rotate_z is not None:
                rand_z = self.cfg.distractor_rotate_z.sample()
                rand_ori_mat = t3d.euler.euler2mat(0.0, 0.0, rand_z)
            else:
                rand_ori_mat = t3d.euler.euler2mat(0.0, 0.0, random.uniform(0.0, 2.0 * np.pi))
            rand_ori_mat = rand_ori_mat @ stable_ori_mat

            transformation = np.eye(4)
            transformation[:3, :3] = rand_ori_mat
            transformation[:3, 3] = p
            distance = self.collision_manager.min_distance_single(object_msh, transformation)
            if distance > self.cfg.collide_threshold:  # type: ignore
                self.collision_manager.add_object(obj.uid, object_msh, transformation)
                # obj.pose.quaternion = [0.6013860816541081, 9.584801678524222e-05, -0.0020859967168959104, -0.7989558312094444 ] # FIXME #t3d.quaternions.mat2quat(rand_ori_mat).tolist()
                obj.pose.quaternion = t3d.quaternions.mat2quat(rand_ori_mat).tolist()
                if objtype == "container":
                    self._inner_state[f"container_{obj.uid}"] = {
                        "position": obj.pose.position,
                        "quaternion": obj.pose.quaternion,
                    }
                else:
                    self._inner_state[obj.uid] = {
                        "position": obj.pose.position,
                        "quaternion": obj.pose.quaternion,
                    }
                return True
        print(f"Warning: Failed to place object {obj.uid} without collision after 100 attempts.")
        return None


@dataclass
class SpatialDRCfg(RandomizerCfg):
    placements_from_scene: bool = False
    robot_region: Box | None = None
    robot_orientation_region: Box | None = None

    target_region: Box | None = None
    container_region: Box | None = None
    distractors_region: Box | list[Box] | None = None
    rand_stable_pose: bool = True
    collide_threshold: float = 0.005
    spatial_mode: str = "random"  # fixed, random
    fixed_stable_pose_idx: int = 0
    target_stable_indices: list[int] | None = None
    target_rotate_z: Box = Box(low=0.0, high=2.0 * np.pi)
    distractor_rotate_z: Box | None = None

    articulated_region: Box | None = None
    articulated_rotate_z: Box | None = None

    container_rotate_z: Box | None = None
    # container_rotate_z: Box = Box(low=0.0, high=0.0)
    obj_surface_map: dict[str, str | list[str]] | None = None
    randmizer_class: type[Randomizer] = SpatialDR
