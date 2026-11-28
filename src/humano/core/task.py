"""
HUMANO: Humanoid Benchmark and Learning Framework

Originally derived from the SIMPLE project.
Copyright (c) 2025 Songlin Wei and Contributors
Copyright (c) 2026 HUMANO Contributors
Licensed under the terms in LICENSE file.
"""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any, Dict, List, Optional

# from humano.core.task import SubtaskSpec

if TYPE_CHECKING:
    from humano.core.actor import Actor
    from humano.core.randomizer import Randomizer, RandomizerCfg
    from humano.datagen.subtask_spec import SubtaskSpec
    from humano.dr.manager import DRManager
    from humano.sensors.config import SensorCfg
    # from humano.core.layout import Layout
    # from humano.core.robot import Robot
# from gymnasium.envs.registration import EnvSpec


from abc import ABC, abstractmethod

import numpy as np
from gymnasium import spaces

from humano.core.layout import Layout
from humano.core.robot import Robot
from humano.dr.scene import TabletopSceneDR
from humano.robots.protocols import HeadCamMountable, WristCamMountable
from humano.sensors.config import CameraCfg


class Task(ABC):
    metadata: dict[str, Any] = {
        "physics_dt": 0.002,
        "render_hz": 50,
        "split": "train",  # train, val, test
    }

    uid: str
    label: str
    description: str

    teleop_viewer_camera: dict[str, Any] | None = None

    robot_cfg: dict[str, Any]

    sensor_cfgs: dict[str, SensorCfg]

    dr_cfgs: dict[str, RandomizerCfg]

    def __init__(
        self,
        dr: DRManager,
        split: str | None = None,
        render_hz: int | None = None,
        dr_level: int | None = None,
        physics_dt: float | None = None,
        *args,
        **kwargs,
    ) -> None:
        self.metadata.update(
            {
                k: v
                for k, v in {
                    "split": split,
                    "render_hz": render_hz,
                    "dr_level": dr_level,
                    "physics_dt": physics_dt,
                }.items()
                if v is not None
            }
        )
        self.dr = dr
        self.split = split

    @property
    def render_hz(self) -> int:
        return self.metadata.get("render_hz", 30)

    @property
    # @abstractmethod
    def robot(self) -> Robot:
        # TODO change to RobotConfig
        return self._robot

    @robot.setter
    def robot(self, value: Robot):
        print(f"Setting robot: {value}")
        self._robot = value

    @property
    @abstractmethod
    def layout(self) -> Layout: ...

    @property
    @abstractmethod
    def instruction(self) -> str:
        """A natural language instruction describing the task."""
        # return self.description

    @property
    def observation_space(self) -> spaces.Space:
        """default observation space, can be overridden by subclasses"""
        obs = {}
        for key, sensorCfg in self.sensor_cfgs.items():
            sense_obs = sensorCfg.observation_space
            if isinstance(sense_obs, spaces.Dict):
                for sub_key, sub_space in sense_obs.spaces.items():
                    obs[f"{key}_{sub_key}"] = sub_space
            else:
                obs[key] = sense_obs
        return spaces.Dict(obs)

    @property
    @abstractmethod
    def action_space(self) -> spaces.Space: ...

    def save_config(self) -> dict[str, Any]:
        info = {
            "uid": self.uid,
            "label": self.label,
            "description": self.description,
            "metadata": self.metadata,
            "robot_cfg": self.robot_cfg,
            "sensor_cfgs": {k: v.__dict__ for k, v in self.sensor_cfgs.items()},
            "dr_cfgs": {k: v.to_dict() for k, v in self.dr_cfgs.items()},
        }
        return info

    def load_config(self, path: str) -> None:
        """Load task config from a json file."""
        raise NotImplementedError

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> None:
        """Resets the task state.

        Reset the task to RANDOM initial state.
        if options is given it will DUPLICATE the states specified by the options.

        This happens before reseting the environment, and can be used to set up the task-specific state,
        such as randomizing the target object and its position.
        The reset of the environment will call the task's DR to randomize the scene,
        and then call the task's reset() to set up the task-specific state based on the randomized scene.
        """
        split = self.metadata.get("split", "train")
        replay = (options or {}).get("state_dict") or {}
        self._ood_environment_fields = {
            key: deepcopy(replay[key])
            for key in ("ood", "ood_distractor", "ood_sensor_noise", "ood_physics_state", "ood_skip_initial_step", "ood_robot_joint_qpos", "language_source")
            if key in replay
        }
        if replay.get("ood_distractor") and self.dr.get_randomizer("distractors") is None:
            from humano.dr.distractor import DistractorDRCfg

            self.dr.randomizers["distractors"] = DistractorDRCfg(
                res_id="objects", number_of_distractors=0
            ).build()

        replay_dr_state = replay.get("dr_state_dict", {})
        has_replay_scene = "scene" in replay_dr_state
        dynamic_scene = getattr(self, "_replay_scene_randomizer_added", False)
        if has_replay_scene and self.dr.get_randomizer("scene") is None:
            from humano.dr.scene import TabletopSceneDRCfg

            scene_cfg = replay.get("dr_cfgs", {}).get("scene", {})
            scene_manager = scene_cfg.get("scene_manager")
            if not scene_manager:
                raise ValueError(
                    "Replay scene state requires dr_cfgs.scene.scene_manager"
                )
            self.dr.randomizers["scene"] = TabletopSceneDRCfg(
                scene_manager=scene_manager
            ).build()
            self._replay_scene_randomizer_added = True
        elif not has_replay_scene and dynamic_scene:
            self.dr.randomizers.pop("scene", None)
            self._replay_scene_randomizer_added = False

        if options is not None and "state_dict" in options and options["state_dict"] is not None:
            assert self.uid == options["state_dict"]["uid"], (
                f"load the wrong state dict for the {self.uid} task ?!"
            )
            dr_level = options.get("dr_level", None)
            self.dr.load_state_dict(options["state_dict"], dr_level=dr_level)
        else:
            self.dr.reset(seed=seed)

        if replay.get("language_source") == "description" or (options or {}).get("language_source") == "description":
            language = self.dr.get_randomizer("language")
            if language is not None:
                language.load_state_dict(self.description)
            self._ood_environment_fields["language_source"] = "description"

        self._layout = Layout()
        self._layout.add_robot(self.robot)

        # scene randomization
        scene_dr = self.dr.get_randomizer("scene")
        surface_height_deltas: dict[str, float] = {}
        if isinstance(scene_dr, TabletopSceneDR):
            scene = scene_dr(split)
            self._layout.scene = scene
            if scene_dr.cfg.scene_config:
                self._layout.actors.update(scene.make_contact_actors())
                table_height = 0.0
            else:
                self._layout.add_primitive("table", scene.table)

                if hasattr(scene, "table2") and scene.table2 is not None:
                    self._layout.add_primitive("table2", scene.table2)
                surface_height_deltas = self._enforce_fixed_tabletop_scene_heights()
                # table_info = {
                #     "table_size": scene.table.size,
                #     "table_position": scene.table.pose.position,
                #     "table_orientation": scene.table.pose.quaternion,
                # }
                # scene_info["table_info"] = table_info
                table_height = scene.table.pose.position[2] + 0.5 * scene.table.size[2]
        else:
            table_height = 0.0

        # container randomization
        container_dr = self.dr.get_randomizer("container")
        if container_dr is not None:
            container_asset = container_dr(split)
            self._layout.add_object("container", container_asset)

        # target randomization
        target_dr = self.dr.get_randomizer("target")
        if target_dr is not None:
            target_asset = target_dr(split)
            self._layout.add_object("target", target_asset)

        # distractor randomization
        distractors_dr = self.dr.get_randomizer("distractors")
        if distractors_dr is not None:
            distractors = distractors_dr(split)
            # print(f"Adding {len(distractors)} distractors to layout.")
            for idx, (obj_id, obj) in enumerate(distractors.items()):
                self._layout.add_object(f"distractor_{idx}", obj)

        articulated_dr = self.dr.get_randomizer("articulated")
        if articulated_dr is not None:
            articulated_asset = articulated_dr(split)
            self._layout.add_articulated_object("articulated", articulated_asset)

        # sptial randomization
        spatial_dr = self.dr.get_randomizer("spatial")
        if spatial_dr is not None:
            if getattr(spatial_dr.cfg, "placements_from_scene", False):
                spatial_dr(split, self._layout, table_height=table_height, seed=seed)
            else:
                spatial_dr(split, self._layout, table_height=table_height)
            self._shift_loaded_table_surface_actors(surface_height_deltas, spatial_dr)

        # lighting randomization
        light_dr = self.dr.get_randomizer("lighting")
        if light_dr is not None:
            light_dr.load_ood_policy(replay.get("ood", {}))
            for light in light_dr(split):
                self._layout.add_light(light)
        # light_example = self._layout.lights[0] if len(self._layout.lights) > 0 else None
        # light_info = {
        #     "light_num": light_dr.cfg.light_num if light_dr is not None else 0,
        #     "light_color_temperature":light_example.light_color_temperature if light_example is not None else 6500,
        #     "light_intensity":light_example.light_intensity if light_example is not None else 100000,
        #     "light_radius":light_example.light_radius if light_example is not None else 0.1,
        #     "light_length":light_example.light_length if light_example is not None else 1.0,
        #     "light_spacing":light_example.pose.position,
        #     "light_position":light_example.center_light_postion if light_example is not None else [0.0,0.0,3.0],
        #     "light_orientation":light_example.center_light_orientation if light_example is not None else [0.0,0.0,0.0,1.0],
        # }
        # scene_info["light_info"] = light_info

        # camera randomization
        camera_dr = self.dr.get_randomizer("camera")
        if camera_dr is not None:
            for cam_id, cam_info in self.sensor_cfgs.items():
                if isinstance(cam_info, CameraCfg):
                    cam_cfg = deepcopy(cam_info)
                    if cam_cfg.quaternion is None and cam_cfg.mount == "eye_in_hand":
                        assert isinstance(self.robot, WristCamMountable), (
                            "Robot does not support wrist camera mounting."
                        )
                        cam_cfg.pose["quaternion"] = self.robot.wrist_camera_orientation
                    if (
                        cam_cfg.quaternion is None
                        and cam_cfg.mount == "eye_in_head"
                        and isinstance(self.robot, HeadCamMountable)
                    ):
                        cam_cfg.pose["quaternion"] = self.robot.head_camera_orientation

                    self._layout.add_camera(cam_id, camera_dr(split, cam_cfg))

        # material randomization
        material_dr = self.dr.get_randomizer("material")
        if material_dr is not None:
            material_info = material_dr(split, self._layout)
            # scene_info["material_info"] = material_info

        # self._layout.scene_info = scene_info # FIXME

    def _fixed_table_height_from_cfg(self, table_name: str) -> float | None:
        scene_dr = self.dr.get_randomizer("scene")
        scene_cfg = getattr(scene_dr, "cfg", None)
        if scene_cfg is None:
            scene_cfg = self.dr_cfgs.get("scene")
        if scene_cfg is None:
            return None

        height_attr = "table_height" if table_name == "table" else f"{table_name}_height"
        height_box = getattr(scene_cfg, height_attr, None)
        if height_box is None:
            return None

        low = np.asarray(height_box.low, dtype=np.float64)
        high = np.asarray(height_box.high, dtype=np.float64)
        if low.shape != () or high.shape != () or not np.allclose(low, high):
            return None
        return float(low)

    def _enforce_fixed_tabletop_scene_heights(self) -> dict[str, float]:
        height_deltas: dict[str, float] = {}
        for table_name in ("table", "table2"):
            table = self._layout.actors.get(table_name)
            if table is None:
                continue

            desired_height = self._fixed_table_height_from_cfg(table_name)
            if desired_height is None:
                continue

            current_height = float(table.pose.position[2] + 0.5 * table.size[2])
            delta = desired_height - current_height
            if abs(delta) <= 1e-9:
                continue

            position = list(table.pose.position)
            position[2] = float(position[2]) + delta
            table.pose.position = position
            height_deltas[table_name] = delta

        return height_deltas

    def _surface_name_for_actor(self, actor_name: str, spatial_dr: Randomizer) -> str | None:
        spatial_cfg = getattr(spatial_dr, "cfg", None)
        surface_map = getattr(spatial_cfg, "obj_surface_map", None) or {}

        if actor_name == "target":
            surface_entry = surface_map.get("target", "table")
        elif actor_name == "container":
            surface_entry = surface_map.get("container", "table")
        elif actor_name.startswith("distractor"):
            surface_entry = surface_map.get(actor_name) or surface_map.get("distractor", "table")
        elif actor_name == "articulated":
            surface_entry = surface_map.get("articulated")
        else:
            return None

        if isinstance(surface_entry, list):
            return surface_entry[0] if surface_entry else None
        return surface_entry

    def _spatial_state_key_for_actor(self, actor_name: str, actor: Actor) -> str | None:
        if actor_name == "container":
            return f"container_{actor.uid}"
        if actor_name == "articulated":
            return f"articulated_{actor.uid}"
        if hasattr(actor, "uid"):
            return str(actor.uid)
        return None

    def _shift_loaded_table_surface_actors(
        self,
        height_deltas: dict[str, float],
        spatial_dr: Randomizer,
    ) -> None:
        if not height_deltas:
            return

        spatial_state = getattr(spatial_dr, "_inner_state", None)
        for actor_name, actor in self._layout.actors.items():
            surface_name = self._surface_name_for_actor(actor_name, spatial_dr)
            if surface_name not in height_deltas or not hasattr(actor, "pose"):
                continue

            delta = height_deltas[surface_name]
            position = list(actor.pose.position)
            position[2] = float(position[2]) + delta
            actor.pose.position = position

            state_key = self._spatial_state_key_for_actor(actor_name, actor)
            if state_key is not None and isinstance(spatial_state, dict):
                actor_state = spatial_state.get(state_key)
                if isinstance(actor_state, dict) and "position" in actor_state:
                    state_position = list(actor_state["position"])
                    state_position[2] = float(state_position[2]) + delta
                    actor_state["position"] = state_position

    def state_dict(self) -> dict[str, Any]:
        """Dump the current state of the task.

        Returns:
            A dictionary containing the current state of the task.
        """
        rand_state_dict = self.dr.state_dict()
        return {
            "uid": self.uid,
            "label": self.label,
            "description": self.description,
            "metadata": self.metadata,
            "robot_cfg": self.robot_cfg,
            "sensor_cfgs": {k: v.__dict__ for k, v in self.sensor_cfgs.items()},
            "dr_cfgs": {k: v.to_dict() for k, v in self.dr_cfgs.items()},
            "dr_state_dict": rand_state_dict,
            "layout": self._layout.to_dict() if self._layout is not None else {},
            **getattr(self, "_ood_environment_fields", {}),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Clone the current layout with given options."""
        return self.reset(options={"state_dict": state_dict})

    def check_success(self, *args, **kwargs) -> bool:
        """Check if the task is successfully completed."""
        raise NotImplementedError

    def compute_reward(self, info: dict[str, Any], *args, **kwargs) -> float:
        """Compute the reward for the current state of the task."""
        return 0.0

    def preload_objects(self) -> list[Actor]:
        """Preloads all assets required by the task.

        This is needed for recurrent reset of the environment in IsaacSim
        if each episode might need different assets.


        Returns:
            list[Actor] a list of Actor instances that need to be preloaded.
        """
        raise NotImplementedError

    def decompose(self) -> list[SubtaskSpec]:
        """Decompose the task into a sequence of subtasks.

        Returns:
            list[SubtaskSpec]: A list of SubtaskSpec instances representing the subtasks.
        """
        raise NotImplementedError
