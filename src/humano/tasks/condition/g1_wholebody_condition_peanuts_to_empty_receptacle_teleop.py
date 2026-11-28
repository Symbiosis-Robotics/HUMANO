from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
import transforms3d as t3d

from humano.assets import AssetManager
from humano.core.actor import Actor, ObjectActor
from humano.dr import DistractorDRCfg, LanguageDRCfg, SpatialDRCfg
from humano.dr.types import Box
from humano.tasks.long.g1_wholebody_long_tasks_teleop import (
    G1WholebodyLongPeanutsFromCupToPlateTaskTeleop,
    _center_in_plate_region,
)
from humano.tasks.registry import TaskRegistry

from .g1_wholebody_condition_tasks_teleop import (
    _C8_PLATE_XY,
    _C8_TRAY_XY,
    _CAKE_LOAD,
    _TRAY_HALF_XY,
    _TRAY_LOAD,
    _TRAY_Z,
    _center_in_actor_local,
    _contact_between,
    _hand_contact,
    _InitialConditionMixin,
    _stable_pose,
)

_PEANUT_016_LOAD = "peanut/peanut_016"
_PEANUT_018_LOAD = "peanut/peanut_018"
_CUP_LOAD = "Cup_B0BR43SPKJ_Vanilla/Cup_B0BR43SPKJ_Vanilla"
_PLATE_LOAD = "plate/plate_3"

_C7_CUP_XY = np.array([-0.22, -0.18], dtype=np.float64)
_C7_PEANUT_A_XY = np.array([-0.30, 0.02], dtype=np.float64)
_C7_PEANUT_B_XY = np.array([-0.20, 0.02], dtype=np.float64)
_C7_CAKE_STAGING_XY = np.array([0.48, 0.20], dtype=np.float64)
_C7_CAKE_IN_TRAY_LOCAL = np.array([0.0, 0.0, 0.03], dtype=np.float64)


@TaskRegistry.register("g1_wholebody_condition_peanuts_to_empty_plate_or_tray_teleop")
class G1WholebodyConditionPeanutsToEmptyPlateOrTrayTaskTeleop(
    _InitialConditionMixin, G1WholebodyLongPeanutsFromCupToPlateTaskTeleop
):
    uid = "g1_wholebody_condition_peanuts_to_empty_plate_or_tray_teleop"
    label = "G1 TELEOP Condition Peanuts To Empty Plate Or Tray"
    description = "A task where the G1 robot must pour both peanuts from the cup into the empty receptacle: use the plate when the cake is on the tray, or use the tray when the cake is on the plate."
    default_instruction = "pour both peanuts from the cup into the empty receptacle: use the plate when the cake is on the tray, or use the tray when the cake is on the plate"
    initial_conditions = ("plate_empty", "tray_empty")

    dr_cfgs = deepcopy(G1WholebodyLongPeanutsFromCupToPlateTaskTeleop.dr_cfgs)
    dr_cfgs["language"] = LanguageDRCfg(instructions=[default_instruction])
    dr_cfgs["distractors"] = DistractorDRCfg(
        res_id="objects",
        number_of_distractors=4,
        allow_duplicates=False,
        include=[_PLATE_LOAD, _PEANUT_018_LOAD, _TRAY_LOAD, _CAKE_LOAD],
    )
    dr_cfgs["spatial"] = SpatialDRCfg(
        spatial_mode="fixed",
        robot_region=Box(low=[-0.95, 0.0, 0.0], high=[-0.95, 0.0, 0.0]),
        target_region=Box(low=_C7_PEANUT_A_XY, high=_C7_PEANUT_A_XY),
        container_region=Box(low=_C7_CUP_XY, high=_C7_CUP_XY),
        distractors_region=[
            Box(low=_C8_PLATE_XY, high=_C8_PLATE_XY),
            Box(low=_C7_PEANUT_B_XY, high=_C7_PEANUT_B_XY),
            Box(low=_C8_TRAY_XY, high=_C8_TRAY_XY),
            Box(low=_C7_CAKE_STAGING_XY, high=_C7_CAKE_STAGING_XY),
        ],
        target_stable_indices=[0],
        target_rotate_z=Box(low=0.0, high=0.0),
        container_rotate_z=Box(low=0.0, high=0.0),
        distractor_rotate_z=Box(low=0.0, high=0.0),
        fixed_stable_pose_idx=0,
        obj_surface_map={
            "target": "table",
            "container": "table",
            "distractor": "table",
        },
    )

    @property
    def tray(self) -> ObjectActor:
        actor = self.layout.actors["distractor_2"]
        assert isinstance(actor, ObjectActor)
        return actor

    @property
    def cake(self) -> ObjectActor:
        actor = self.layout.actors["distractor_3"]
        assert isinstance(actor, ObjectActor)
        return actor

    @property
    def selected_destination(self) -> ObjectActor:
        actor = self.plate if self.selected_initial_condition == "plate_empty" else self.tray
        assert isinstance(actor, ObjectActor)
        return actor

    @property
    def selected_destination_key(self) -> str:
        return (
            "distractor_0" if self.selected_initial_condition == "plate_empty" else "distractor_2"
        )

    def _configure_condition_after_reset(self) -> None:
        cake_stable_z, cake_quaternion = _stable_pose(self.cake)
        if self.selected_initial_condition == "plate_empty":
            tray_position = np.asarray(self.tray.pose.position, dtype=np.float64)
            tray_rotation = t3d.quaternions.quat2mat(self.tray.pose.quaternion)
            self.cake.pose.position = (
                tray_position + tray_rotation @ _C7_CAKE_IN_TRAY_LOCAL
            ).tolist()
        else:
            plate_position = np.asarray(self.plate.pose.position, dtype=np.float64)
            self.cake.pose.position = (
                plate_position + np.array([0.0, 0.0, cake_stable_z], dtype=np.float64)
            ).tolist()
        self.cake.pose.quaternion = cake_quaternion
        self._success_hold_count = 0
        self.reward = 0.0

    def _peanut_in_selected_destination(
        self,
        info: dict[str, Any],
        key: str,
        peanut: Actor,
        mujoco_env=None,
    ) -> bool:
        destination = self.selected_destination
        local = _center_in_actor_local(mujoco_env, peanut, destination)
        if local is None:
            return False
        if self.selected_initial_condition == "plate_empty":
            inside = _center_in_plate_region(local)
        else:
            inside = bool(
                np.all(np.abs(local[:2]) <= _TRAY_HALF_XY)
                and _TRAY_Z[0] - 0.02 <= local[2] <= _TRAY_Z[1]
            )
        del info, key
        return bool(
            inside
            and not _contact_between(mujoco_env, peanut.simulation_name, self.cup.simulation_name)
            and not _hand_contact(mujoco_env, peanut)
        )

    def check_peanuts_in_empty_receptacle(
        self, info: dict[str, Any] | None = None, *args, **kwargs
    ) -> bool:
        del args
        required = {"target", "container", "distractor_0", "distractor_1", "distractor_2"}
        if info is None or not required.issubset(info):
            return False
        mujoco_env = kwargs.get("mujoco_env")
        if not (
            self._peanut_in_selected_destination(info, "target", self.target, mujoco_env=mujoco_env)
            and self._peanut_in_selected_destination(
                info, "distractor_1", self.peanut_b, mujoco_env=mujoco_env
            )
        ):
            return False
        destination_name = self.selected_destination.simulation_name
        supported_a = _contact_between(mujoco_env, self.target.simulation_name, destination_name)
        supported_b = _contact_between(mujoco_env, self.peanut_b.simulation_name, destination_name)
        # One peanut may rest on the other without directly touching the receptacle.
        if not (supported_a or supported_b):
            return False
        return bool(
            supported_a
            and supported_b
            or _contact_between(
                mujoco_env, self.target.simulation_name, self.peanut_b.simulation_name
            )
        )

    def check_peanuts_on_plate(self, info=None, *args, **kwargs) -> bool:
        return self.check_peanuts_in_empty_receptacle(info, *args, **kwargs)

    def check_object_in_container(self, *args, **kwargs) -> bool:
        return self.check_peanuts_in_empty_receptacle(*args, **kwargs)

    def compute_reward(self, info: dict[str, Any], *args, **kwargs) -> float:
        if self.check_peanuts_in_empty_receptacle(info, *args, **kwargs):
            self._success_hold_count += 1
        else:
            self._success_hold_count = 0
        self.reward = 0.1 * self._success_hold_count
        return float(self.reward)

    def preload_objects(self) -> list[Actor]:
        objects = AssetManager.get("objects")
        return [
            ObjectActor(asset=objects.load(_PEANUT_016_LOAD)),
            ObjectActor(asset=objects.load(_CUP_LOAD)),
            ObjectActor(asset=objects.load(_PLATE_LOAD)),
            ObjectActor(asset=objects.load(_PEANUT_018_LOAD)),
            ObjectActor(asset=objects.load(_TRAY_LOAD)),
            ObjectActor(asset=objects.load(_CAKE_LOAD)),
        ]
