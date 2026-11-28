"""
HUMANO: Humanoid Benchmark and Learning Framework

Originally derived from the SIMPLE project.
Copyright (c) 2025 Songlin Wei and Contributors
Copyright (c) 2026 HUMANO Contributors
Licensed under the terms in LICENSE file.
"""

import os
import xml.etree.ElementTree as ET
from copy import deepcopy
from pathlib import Path
from threading import Lock
from typing import Any

import mujoco
import numpy as np
import transforms3d as t3d
from gear_sonic.utils.mujoco_sim.configs import SimLoopConfig
from gear_sonic.utils.mujoco_sim.robot import Robot as GearSonicRobot
from gear_sonic.utils.mujoco_sim.unitree_sdk2py_bridge import ElasticBand

from humano.core.action import ActionCmd
from humano.core.controller import Controller, ControllerCfg
from humano.core.robot import Robot
from humano.core.types import Pose
from humano.robots.controllers.combo import WholeBodyEEFController, WholeBodyEEFControllerCfg
from humano.robots.controllers.eef import DexHandEEFControllerCfg
from humano.robots.controllers.qpos import PDJointPosControllerCfg
from humano.robots.hands import (
    BRAINCO_CONFIG_PATH,
    INSPIRE_DFQ_PATH,
    INSPIRE_EFFORT_LIMIT,
    HandSpec,
    _brainco_values,
    get_hand_spec,
)
from humano.robots.mixin import CuRoboMixin
from humano.robots.protocols import HasDexterousHand, HeadCamMountable, Humanoid
from humano.robots.registry import RobotRegistry
from humano.utils import load_yaml, resolve_data_path

LEFT_LEG_JOINTS = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
]
RIGHT_LEFT_JOINTS = [
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
]
WAIST_JOINTS = ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
LEFT_ARM_JOINTS = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
]
RIGHT_ARM_JOINTS = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]
LEFT_HAND_JOINTS = [
    "left_hand_thumb_0_joint",
    "left_hand_thumb_1_joint",
    "left_hand_thumb_2_joint",
    "left_hand_index_0_joint",
    "left_hand_index_1_joint",
    "left_hand_middle_0_joint",
    "left_hand_middle_1_joint",
]
RIGHT_HAND_JOINTS = [
    "right_hand_thumb_0_joint",
    "right_hand_thumb_1_joint",
    "right_hand_thumb_2_joint",
    "right_hand_index_0_joint",
    "right_hand_index_1_joint",
    "right_hand_middle_0_joint",
    "right_hand_middle_1_joint",
]

WHOLE_BODY_JOINTS = (
    LEFT_LEG_JOINTS
    + RIGHT_LEFT_JOINTS
    + WAIST_JOINTS
    + LEFT_ARM_JOINTS
    + RIGHT_ARM_JOINTS
    + LEFT_HAND_JOINTS
    + RIGHT_HAND_JOINTS
)
SONIC_DEFAULT_STAND_Q = np.array(
    [
        -0.312,
        0.0,
        0.0,
        0.669,
        -0.363,
        0.0,
        -0.312,
        0.0,
        0.0,
        0.669,
        -0.363,
        0.0,
        0.0,
        0.0,
        0.0,
        0.2,
        0.2,
        0.0,
        0.6,
        0.0,
        0.0,
        0.0,
        0.2,
        -0.2,
        0.0,
        0.6,
        0.0,
        0.0,
        0.0,
    ],
    dtype=np.float64,
)
STABILIZE_VEL_THRESHOLD: float = 1e-4  # max |qvel[0:6]| to consider robot stable
MIN_STABILIZE_STEPS: int = 100  # min steps before velocity check is valid (1s at 200Hz)
GROUND_STAND_ANKLE_Z: float = 0.0710542


def _hand_model_init_qpos(
    *,
    model_joints: tuple[str, ...],
    actuated_joints: tuple[str, ...],
    mimic_map: dict[str, tuple[str, float, float]],
    actuated_qpos: tuple[float, ...],
) -> list[float]:
    """Expand active hand qpos to model-joint qpos, including mimic joints."""
    if len(actuated_qpos) != len(actuated_joints):
        raise ValueError(
            f"hand init_qpos len={len(actuated_qpos)} does not match "
            f"actuated joint count={len(actuated_joints)}"
        )

    active_qpos = dict(zip(actuated_joints, actuated_qpos, strict=True))
    model_qpos: list[float] = []
    for joint_name in model_joints:
        if joint_name in active_qpos:
            model_qpos.append(float(active_qpos[joint_name]))
            continue
        if joint_name in mimic_map:
            parent_joint, ratio, offset = mimic_map[joint_name]
            if parent_joint not in active_qpos:
                raise ValueError(
                    f"mimic joint {joint_name!r} references unknown parent {parent_joint!r}"
                )
            parent_qpos = float(active_qpos[parent_joint])
            if "thumb_" in joint_name and parent_qpos == 0.0:
                offset = 0.0
            model_qpos.append(max(0.0, parent_qpos * ratio + offset))
            continue
        raise ValueError(f"model joint {joint_name!r} is neither actuated nor configured as mimic")
    return model_qpos


@RobotRegistry.register("g1_sonic")
class G1Sonic(CuRoboMixin, Humanoid, Robot, HeadCamMountable, HasDexterousHand):
    uid: str = "g1_sonic"
    label: str = "Unitree G1 Wholebody"

    # #TODO verify
    wholebody_dof: int
    dof: int = 29

    robot_cfg: Any
    mjcf_path: str
    usd_path: str

    robot_ns: str
    hand_yaml: str
    hand_uid: str
    hand_dof: int

    pregrasp_distance: list[float] = [0.05, 0.08]

    wrist_camera_orientation: list[float] = [1, 0, 0, 0]
    head_camera_orientation: list[float] = [1, 0, 0, 0]
    # head_cam_link: str = "d435_link"

    init_joint_states: dict[str, float]
    joint_names: tuple[str, ...]
    hand_names: list[str]
    robot_eef_offset: float = 0
    # TODO May have problem here,not all joint are pd control

    controller_cfg: ControllerCfg

    hand_prim_path = "right_hand_base_link"
    eef_prim_path: str
    wrist_cam_link: str = "right_hand_camera_base_link"
    # z_offset= 0.785

    LEFT_ARM_EE_LINK: str
    RIGHT_ARM_EE_LINK: str

    joint_limits: dict[str, tuple[float, float]] = {}

    # FRANKA_FINGER_LENGTH=0

    _controller: Controller | None = None

    visulize_spheres: bool = False

    sonic_config: dict = {}

    spawn_pose: Pose

    left_hand_model_joints: tuple[str, ...]
    right_hand_model_joints: tuple[str, ...]
    left_hand_actuated_joints: tuple[str, ...]
    right_hand_actuated_joints: tuple[str, ...]

    def __init__(self, sonic_config: dict, hand_spec: HandSpec | None = None, **kwargs) -> None:
        if hand_spec is None:
            hand_spec = get_hand_spec("dex3")
        self.hand_spec = hand_spec

        # Derive instance attributes from hand_spec
        self.mjcf_path = hand_spec.mjcf_path
        self.usd_path = hand_spec.usd_path
        self.robot_ns = (
            "g1_29dof_with_inspire_hand"
            if hand_spec.name == "inspire"
            else f"g1_29dof_wholebody_{hand_spec.name}"
        )
        self.hand_yaml = hand_spec.hand_yaml
        self.hand_uid = hand_spec.uid_for("right")
        self.hand_dof = hand_spec.dof
        self.joint_limits = {}
        self.robot_cfg = hand_spec.cu_robot_yaml

        # Dynamic joint names
        self.left_hand_model_joints = hand_spec.left_model_joints
        self.right_hand_model_joints = hand_spec.right_model_joints
        self.left_hand_actuated_joints = hand_spec.left_actuated_joints
        self.right_hand_actuated_joints = hand_spec.right_actuated_joints

        BODY_JOINTS = (
            LEFT_LEG_JOINTS + RIGHT_LEFT_JOINTS + WAIST_JOINTS + LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS
        )
        self._body_joint_names = BODY_JOINTS
        self._whole_body_joint_names = (
            BODY_JOINTS + list(hand_spec.left_model_joints) + list(hand_spec.right_model_joints)
        )
        self.joint_names = tuple(self._whole_body_joint_names)
        self.hand_names = list(hand_spec.left_model_joints) + list(hand_spec.right_model_joints)

        self.wholebody_dof = len(self._whole_body_joint_names)
        left_hand_init_qpos = _hand_model_init_qpos(
            model_joints=hand_spec.left_model_joints,
            actuated_joints=hand_spec.left_actuated_joints,
            mimic_map=hand_spec.left_mimic_map,
            actuated_qpos=hand_spec.init_qpos,
        )
        right_hand_init_qpos = _hand_model_init_qpos(
            model_joints=hand_spec.right_model_joints,
            actuated_joints=hand_spec.right_actuated_joints,
            mimic_map=hand_spec.right_mimic_map,
            actuated_qpos=hand_spec.init_qpos,
        )

        self.init_joint_states = dict(
            zip(
                self.joint_names,
                [
                    *SONIC_DEFAULT_STAND_Q.tolist(),
                    *left_hand_init_qpos,
                    *right_hand_init_qpos,
                ],
                strict=True,
            )
        )

        self.LEFT_ARM_EE_LINK = hand_spec.left_ee_link
        self.RIGHT_ARM_EE_LINK = hand_spec.right_ee_link

        self.eef_prim_path = hand_spec.right_ee_link
        if hand_spec.name == "brainco":
            self.eef_prim_path = f"pelvis/{hand_spec.right_ee_link}"
            self.hand_prim_path = self.eef_prim_path
            self.wrist_cam_link = f"{self.hand_prim_path}/right_hand_camera_base_link"

        # Controller config built from hand_spec
        self.controller_cfg = WholeBodyEEFControllerCfg(
            left_leg=PDJointPosControllerCfg(
                joint_names=LEFT_LEG_JOINTS,
                init_qpos=SONIC_DEFAULT_STAND_Q[0:6].tolist(),
            ),
            right_leg=PDJointPosControllerCfg(
                joint_names=RIGHT_LEFT_JOINTS,
                init_qpos=SONIC_DEFAULT_STAND_Q[6:12].tolist(),
            ),
            waist=PDJointPosControllerCfg(
                joint_names=WAIST_JOINTS,
                init_qpos=SONIC_DEFAULT_STAND_Q[12:15].tolist(),
            ),
            left_arm=PDJointPosControllerCfg(
                joint_names=LEFT_ARM_JOINTS,
                init_qpos=SONIC_DEFAULT_STAND_Q[15:22].tolist(),
            ),
            right_arm=PDJointPosControllerCfg(
                joint_names=RIGHT_ARM_JOINTS,
                init_qpos=SONIC_DEFAULT_STAND_Q[22:29].tolist(),
            ),
            left_eef=DexHandEEFControllerCfg(
                joint_names=list(hand_spec.left_actuated_joints),
                init_qpos=list(hand_spec.init_qpos),
                close_qpos=list(hand_spec.close_qpos_left),
            ),
            right_eef=DexHandEEFControllerCfg(
                joint_names=list(hand_spec.right_actuated_joints),
                init_qpos=list(hand_spec.init_qpos),
                close_qpos=list(hand_spec.close_qpos_right),
            ),
        )

        super().__init__(self.uid, self.dof)
        self.sonic_config = (
            {**sonic_config, "SIMULATE_DT": hand_spec.physics_dt}
            if hand_spec.physics_dt is not None
            else sonic_config
        )

        self.command = (
            None  # [vx, yaw, vy, d_height, torso_yaw, torso_pitch, torso_roll] shape: (1,7)
        )
        self._debug_ctrl_err = {}

        self.sonic_robot_model = None

        # adapted from BaseSimulator.__init__
        self.env_name = self.sonic_config["ENV_NAME"]
        self.sim_dt = self.sonic_config["SIMULATE_DT"]
        self.reward_dt = self.sonic_config.get("REWARD_DT", 0.02)
        self.image_dt = self.sonic_config.get("IMAGE_DT", 0.033333)
        self.viewer_dt = self.sonic_config.get("VIEWER_DT", 0.02)
        self.sonic_robot = GearSonicRobot(self.sonic_config)  # FIXME rename

        # adapated from DefaultEnv.__init__
        self.num_body_dof = self.sonic_robot.NUM_JOINTS
        self.num_hand_dof = self.sonic_robot.NUM_HAND_MOTORS
        self.num_hand_model_dof = len(self.left_hand_model_joints)
        self.obs = None
        self.torques = np.zeros(self.num_body_dof + self.num_hand_dof * 2)
        self.torque_limit = np.array(self.sonic_robot.MOTOR_EFFORT_LIMIT_LIST)
        self.camera_configs = {}
        self.reward_lock = Lock()
        self.unitree_bridge = None
        self.elastic_band = None
        self.band_attached_link = None

    def reset(self, **kwargs):
        self.command = None  # FIXME
        self._debug_ctrl_err = {}
        self.spawn_pose = kwargs["spawn_pose"]
        self._stabilized = False
        self._stabilize_step_count = 0
        if self.hand_spec.name == "brainco":
            self._reset_brainco_hand_reference()

    def initialize_ground_standing_pose(self, log_prefix: str = "[G1Sonic]") -> dict[str, Any]:
        """Put the SONIC model in its supported ground standing pose."""
        if not hasattr(self, "mjData") or not hasattr(self, "mjModel"):
            raise RuntimeError("setup_control must run before initializing ground standing pose")
        if len(self._body_joint_names) != len(SONIC_DEFAULT_STAND_Q):
            raise RuntimeError("SONIC standing pose does not match the 29-body-joint order")

        if self.elastic_band is not None:
            self.elastic_band.enable = False
        self.mjData.qvel[:] = 0.0
        self.mjData.qacc_warmstart[:] = 0.0
        for joint_name, target in zip(self._body_joint_names, SONIC_DEFAULT_STAND_Q, strict=True):
            self.mjData.joint(joint_name).qpos[0] = float(target)
        self._apply_mimic_joints(initialize=True)
        root_z = self.sonic_config.get("INITIAL_GROUND_STAND_ROOT_Z")
        if self.use_floating_root_link and root_z is not None:
            self.mjData.qpos[2] = float(root_z)
        mujoco.mj_forward(self.mjModel, self.mjData)

        foot_body_names = ("left_ankle_roll_link", "right_ankle_roll_link")
        foot_positions: dict[str, list[float]] = {}
        for body_name in foot_body_names:
            try:
                foot_positions[body_name] = np.asarray(
                    self.mjData.xpos[self.mjModel.body(body_name).id],
                    dtype=np.float64,
                ).tolist()
            except Exception:
                continue
        if self.use_floating_root_link and foot_positions and root_z is None:
            min_ankle_z = min(pos[2] for pos in foot_positions.values())
            root_lift = GROUND_STAND_ANKLE_Z - float(min_ankle_z)
            if abs(root_lift) > 1e-6:
                self.mjData.qpos[2] += root_lift
                mujoco.mj_forward(self.mjModel, self.mjData)
                foot_positions = {}
                for body_name in foot_body_names:
                    try:
                        foot_positions[body_name] = np.asarray(
                            self.mjData.xpos[self.mjModel.body(body_name).id],
                            dtype=np.float64,
                        ).tolist()
                    except Exception:
                        continue

        state = {
            "mode": "ground_standing_pose",
            "root_qpos": np.asarray(self.mjData.qpos[:7], dtype=np.float64).tolist(),
            "body_q": np.asarray(
                [self.mjData.joint(name).qpos[0] for name in self._body_joint_names],
                dtype=np.float64,
            ).tolist(),
            "foot_positions": foot_positions,
            "elastic_band_enabled": bool(
                getattr(getattr(self, "elastic_band", None), "enable", False)
            ),
        }
        print(
            f"{log_prefix} Ground standing reset: "
            f"root={np.round(np.asarray(state['root_qpos'][:3]), 3)} "
            f"left_foot_z={foot_positions.get('left_ankle_roll_link', [None, None, None])[2]} "
            f"right_foot_z={foot_positions.get('right_ankle_roll_link', [None, None, None])[2]}"
        )
        return state

    def update_ee_link(self, hand_uid: str) -> None:
        if self.robot_cfg is None:
            raise NotImplementedError(
                f"CuRobo planning is not configured for {self.hand_spec.name}"
            )
        # FIXME(inspire): link_names, collision links, camera links, and hand
        # prim paths below are Dex3-specific; provide them through HandSpec
        # before using this planner path with Inspire.
        if "left" in hand_uid:
            ee_link = self.LEFT_ARM_EE_LINK
            link_names = [
                "waist_yaw_link",
                "waist_roll_link",
                "torso_link",
                "left_hand_palm_link",
                "left_hand_index_1_link",
                "left_hand_middle_1_link",
                "left_hand_thumb_2_link",
                "right_hand_palm_link",
                "right_hand_index_1_link",
                "right_hand_middle_1_link",
                "right_hand_thumb_2_link",
            ]
            collision_link_names = [
                "left_shoulder_pitch_link",
                "left_shoulder_roll_link",
                "left_shoulder_yaw_link",
                "left_elbow_link",
                "left_wrist_pitch_link",
                "torso_link",
                "right_shoulder_pitch_link",
                "right_shoulder_roll_link",
                "right_shoulder_yaw_link",
                "right_elbow_link",
                "right_wrist_pitch_link",
                "left_hand_palm_link",
                "left_hand_thumb_0_link",
                "left_hand_thumb_1_link",
                "left_hand_thumb_2_link",
                "left_hand_middle_0_link",
                "left_hand_middle_1_link",
                "left_hand_index_0_link",
                "left_hand_index_1_link",
            ]
            self.hand_prim_path = "left_hand_base_link"
            self.eef_prim_path = "left_hand_palm_link"
            self.wrist_cam_link = "left_hand_camera_base_link"
            self.hand_yaml = self.hand_spec.hand_yaml.replace("_right.yml", "_left.yml")
        elif "right" in hand_uid:
            ee_link = self.RIGHT_ARM_EE_LINK
            link_names = [
                "waist_yaw_link",
                "waist_roll_link",
                "torso_link",
                "right_hand_palm_link",
                "right_hand_index_1_link",
                "right_hand_middle_1_link",
                "right_hand_thumb_2_link",
                "left_hand_palm_link",
                "left_hand_index_1_link",
                "left_hand_middle_1_link",
                "left_hand_thumb_2_link",
            ]
            collision_link_names = [
                "left_shoulder_pitch_link",
                "left_shoulder_roll_link",
                "left_shoulder_yaw_link",
                "left_elbow_link",
                "left_wrist_pitch_link",
                "torso_link",
                "right_shoulder_pitch_link",
                "right_shoulder_roll_link",
                "right_shoulder_yaw_link",
                "right_elbow_link",
                "right_wrist_pitch_link",
                "right_hand_palm_link",
                "right_hand_thumb_0_link",
                "right_hand_thumb_1_link",
                "right_hand_thumb_2_link",
                "right_hand_middle_0_link",
                "right_hand_middle_1_link",
                "right_hand_index_0_link",
                "right_hand_index_1_link",
            ]
            self.hand_prim_path = "right_hand_base_link"
            self.eef_prim_path = "right_hand_palm_link"
            self.wrist_cam_link = "right_hand_camera_base_link"
            self.hand_yaml = self.hand_spec.hand_yaml.replace("_left.yml", "_right.yml")
        else:
            raise ValueError(f"Unsupported G1Sonic hand_uid: {hand_uid!r}")

        self.robot_cfg["kinematics"]["ee_link"] = ee_link
        self.robot_cfg["kinematics"]["link_names"] = link_names
        self.robot_cfg["kinematics"]["collision_link_names"] = collision_link_names
        self.hand_uid = hand_uid
        self._kin_model = None
        self._ik_solver = None
        print(f"[G1Sonic] now the ee_link is {self.robot_cfg['kinematics']['ee_link']}")

    def get_grasp_pose_wrt_robot(
        self,
        grasp_info: dict,
        pregrasp: bool = False,
        robot_pose: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convert a cached grasp pose from world coordinates to the robot base frame."""
        if robot_pose is None:
            raise ValueError("robot_pose is required to plan a grasp with G1Sonic")

        grasp_position = np.asarray(grasp_info["position"], dtype=np.float32)
        grasp_orientation = np.asarray(grasp_info["orientation"], dtype=np.float32)
        if grasp_position.shape != (3,) or grasp_orientation.shape != (4,):
            raise ValueError("grasp_info must contain position[3] and orientation[4]")

        T_world_grasp = np.eye(4, dtype=np.float32)
        grasp_rotation = t3d.quaternions.quat2mat(grasp_orientation)
        T_world_grasp[:3, :3] = grasp_rotation
        grasp_depth_offset_m = float(
            getattr(self, "sonic_config", {}).get("PLANNER_GRASP_DEPTH_OFFSET_M", 0.0)
        )
        T_world_grasp[:3, 3] = (
            grasp_position
            + (float(grasp_info.get("depth", 0.0)) + grasp_depth_offset_m) * grasp_rotation[:, 0]
        )

        T_grasp_ee = np.eye(4, dtype=np.float32)
        T_ee_hand = np.eye(4, dtype=np.float32)
        T_ee_hand[2, 3] = -float(self.robot_eef_offset)

        if pregrasp:
            distance = float(np.mean(self.pregrasp_distance))
            T_grasp_ee[0, 3] = -distance

        T_robot_hand = (
            np.linalg.inv(np.asarray(robot_pose, dtype=np.float32))
            @ T_world_grasp
            @ T_grasp_ee
            @ T_ee_hand
        )
        return (
            T_robot_hand[:3, 3].copy(),
            t3d.quaternions.mat2quat(T_robot_hand[:3, :3]).astype(np.float32),
        )

    @property
    def stabilized(self) -> bool:
        """True once max(|floating-base qvel[0:6]|) drops below STABILIZE_VEL_THRESHOLD.

        Requires MIN_STABILIZE_STEPS accesses before the velocity check is
        evaluated, preventing an immediate latch when qvel=0 right after reset.
        Latches to True and stays there until the next reset().
        """
        self._stabilize_step_count += 1
        if not self._stabilized and self._stabilize_step_count >= MIN_STABILIZE_STEPS:
            self._stabilized = bool(np.max(np.abs(self.mjData.qvel[0:6])) < STABILIZE_VEL_THRESHOLD)
        return self._stabilized

    def setup_control(self, mjData, mjModel, **kwargs) -> tuple[dict[str, Any], dict[str, Any]]:
        """Called after environment reset, ..."""
        self.mjData = mjData
        # self.mjData.qvel = 0
        self.mjModel = mjModel

        # adapted from base_sim.init_scene
        self.torso_index = mujoco.mj_name2id(mjModel, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
        self.root_body = "pelvis"
        self.root_body_id = mjModel.body(self.root_body).id
        self.joint_class_map = self._get_dof_indices_by_class(kwargs["mjSpec"], mjModel)
        self.perform_sysid_search = self.sonic_config.get("perform_sysid_search", False)

        # Check for static root link (fixed base)
        self.use_floating_root_link = "floating_base_joint" in [
            mjModel.joint(i).name for i in range(mjModel.njnt)
        ]
        self.use_constrained_root_link = "constrained_base_joint" in [
            mjModel.joint(i).name for i in range(mjModel.njnt)
        ]

        # Enable the elastic band
        if self.sonic_config.get("ENABLE_ELASTIC_BAND", False) and self.use_floating_root_link:
            band_z = float(self.sonic_config.get("ELASTIC_BAND_Z", 1.0))
            self.elastic_band = ElasticBand(point=np.array(self.spawn_pose.position[:2] + [band_z]))
            print(
                f"[G1Sonic] Elastic band enabled: "
                f"point=({self.spawn_pose.position[0]:.3f}, "
                f"{self.spawn_pose.position[1]:.3f}, {band_z:.3f})"
            )

            if "g1" in self.sonic_config["ROBOT_TYPE"]:
                if self.sonic_config["enable_waist"]:
                    self.band_attached_link = mjModel.body("pelvis").id
                else:
                    self.band_attached_link = mjModel.body("torso_link").id
            elif "h1" in self.sonic_config["ROBOT_TYPE"]:
                self.band_attached_link = mjModel.body("torso_link").id
            else:
                self.band_attached_link = mjModel.body("base_link").id
        else:
            self.elastic_band = None
            self.band_attached_link = None
            print("[G1Sonic] Elastic band disabled")

        # MuJoCo qpos/qvel arrays start with root DOFs before joint DOFs:
        # floating base has 7 qpos (pos + quat) and 6 qvel (lin + ang velocity)
        if self.use_floating_root_link:
            self.qpos_offset = 7
            self.qvel_offset = 6
        else:
            if self.use_constrained_root_link:
                self.qpos_offset = 1
                self.qvel_offset = 1
            else:
                raise ValueError(
                    "No root link found --"
                    "The absolute static root will make the simulation unstable."
                )

        body_joint_index = []
        left_hand_model_index = []
        right_hand_model_index = []
        left_hand_actuated_index = []
        right_hand_actuated_index = []
        for i in range(self.mjModel.njnt):
            name = self.mjModel.joint(i).name
            if (
                any(
                    [
                        part_name in name
                        for part_name in [
                            "hip",
                            "knee",
                            "ankle",
                            "waist",
                            "shoulder",
                            "elbow",
                            "wrist",
                        ]
                    ]
                )
                and name in self.joint_names
            ):
                body_joint_index.append(i)
            elif name in self.left_hand_model_joints:
                left_hand_model_index.append(i)
                if name in self.left_hand_actuated_joints:
                    left_hand_actuated_index.append(i)
            elif name in self.right_hand_model_joints:
                right_hand_model_index.append(i)
                if name in self.right_hand_actuated_joints:
                    right_hand_actuated_index.append(i)

        assert len(body_joint_index) == self.sonic_robot.NUM_JOINTS
        assert len(left_hand_model_index) == len(self.left_hand_model_joints)
        assert len(right_hand_model_index) == len(self.right_hand_model_joints)
        assert len(left_hand_actuated_index) == self.sonic_robot.NUM_HAND_MOTORS
        assert len(right_hand_actuated_index) == self.sonic_robot.NUM_HAND_MOTORS

        self.body_joint_index = np.array(body_joint_index)
        self.left_hand_model_index = np.array(left_hand_model_index)
        self.right_hand_model_index = np.array(right_hand_model_index)
        self.left_hand_actuated_index = np.array(left_hand_actuated_index)
        self.right_hand_actuated_index = np.array(right_hand_actuated_index)
        if self.hand_spec.name == "brainco":
            # Revo2 nests thumb rotation before flexion; commands use SDK channel order.
            self.left_hand_actuated_index = np.array(
                [mjModel.joint(name).id for name in self.left_hand_actuated_joints]
            )  # (6,), model joint ids
            self.right_hand_actuated_index = np.array(
                [mjModel.joint(name).id for name in self.right_hand_actuated_joints]
            )  # (6,), model joint ids
        self.body_actuator_index = self._actuator_ids_for_joint_ids(self.body_joint_index)
        self.left_hand_model_actuator_index = self._actuator_ids_for_joint_ids(
            self.left_hand_actuated_index
            if self.hand_spec.name == "brainco"
            else self.left_hand_model_index
        )
        self.right_hand_model_actuator_index = self._actuator_ids_for_joint_ids(
            self.right_hand_actuated_index
            if self.hand_spec.name == "brainco"
            else self.right_hand_model_index
        )
        self.left_hand_actuated_actuator_index = self._actuator_ids_for_joint_ids(
            self.left_hand_actuated_index
        )
        self.right_hand_actuated_actuator_index = self._actuator_ids_for_joint_ids(
            self.right_hand_actuated_index
        )

        self.ctrl_actuator_offset = 6 if self.sonic_config["FREE_BASE"] else 0
        self.body_ctrl_index = self.body_actuator_index - self.ctrl_actuator_offset
        self.left_hand_actuated_ctrl_index = (
            self.left_hand_actuated_actuator_index - self.ctrl_actuator_offset
        )
        self.right_hand_actuated_ctrl_index = (
            self.right_hand_actuated_actuator_index - self.ctrl_actuator_offset
        )
        self._resize_ctrl_buffers_for_model()
        self._configure_sonic_actuation_model()

        actuators = {}
        joints = {}

        for name in self.joint_names:
            joints[name] = mjData.joint(name)
            actuator_id = mujoco.mj_name2id(mjModel, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if actuator_id != -1:
                actuators[name] = mjData.actuator(name)

        self.joints = joints
        self.actuators = actuators
        # self.last_action=[0 for _ in range(len(actuators))]
        self.controller.set_initial_qpos(actuators, joints)
        controlled_actuator_ids = np.concatenate(
            (
                self.body_actuator_index,
                self.left_hand_model_actuator_index,
                self.right_hand_model_actuator_index,
            )
        )
        self.mjData.ctrl[controlled_actuator_ids] = 0.0
        if self.hand_spec.name in {"inspire", "brainco"}:
            self.mjData.ctrl[self.left_hand_actuated_actuator_index] = self.hand_spec.init_qpos
            self.mjData.ctrl[self.right_hand_actuated_actuator_index] = self.hand_spec.init_qpos
        if self.sonic_config.get("INITIALIZE_GROUND_STANDING_POSE_BEFORE_VIEWER", False):
            self.initialize_ground_standing_pose(log_prefix="[G1Sonic/setup_control]")

        if not self.joint_limits:
            for j, v in self.joints.items():
                limits = mjModel.jnt_range[v.id]
                self.joint_limits[j] = (limits[0], limits[1])
        return self.joints, self.actuators

    def _actuator_ids_for_joint_ids(self, joint_ids: np.ndarray) -> np.ndarray:
        actuator_ids = []
        for joint_id in joint_ids:
            joint_name = self.mjModel.joint(int(joint_id)).name
            actuator_id = mujoco.mj_name2id(self.mjModel, mujoco.mjtObj.mjOBJ_ACTUATOR, joint_name)
            if actuator_id == -1:
                raise RuntimeError(f"SONIC joint {joint_name!r} has no named MuJoCo actuator")
            actuator_ids.append(actuator_id)
        return np.asarray(actuator_ids, dtype=np.int64)

    def _resize_ctrl_buffers_for_model(self) -> None:
        """Match torque arrays to the MuJoCo actuator vector.

        SONIC hand commands can be active-joint only, while the MJCF may expose
        additional mimic actuators. The ctrl vector includes both active torque
        commands and mimic position targets, so it must match ``mjModel.nu``.
        """
        ctrl_len = self.mjModel.nu
        if self.sonic_config["FREE_BASE"]:
            ctrl_len = max(0, ctrl_len - 6)
        if len(self.torques) != ctrl_len:
            self.torques = np.zeros(ctrl_len, dtype=np.float64)
        source_limits = np.asarray(self.sonic_robot.MOTOR_EFFORT_LIMIT_LIST, dtype=np.float64)
        left_body_joint_names = self._body_joint_names[:22]
        right_arm_joint_names = self._body_joint_names[22:]
        source_joint_names = (
            left_body_joint_names
            + list(self.left_hand_actuated_joints)
            + right_arm_joint_names
            + list(self.right_hand_actuated_joints)
        )
        if len(source_limits) != len(source_joint_names):
            raise RuntimeError(
                "SONIC effort limits do not match the active actuator layout: "
                f"limits={len(source_limits)} joints={len(source_joint_names)}"
            )

        remapped = np.full(ctrl_len, np.inf, dtype=np.float64)
        for joint_name, effort_limit in zip(source_joint_names, source_limits, strict=True):
            actuator_id = mujoco.mj_name2id(self.mjModel, mujoco.mjtObj.mjOBJ_ACTUATOR, joint_name)
            if actuator_id == -1:
                raise RuntimeError(f"SONIC effort limit references missing actuator {joint_name!r}")
            ctrl_index = actuator_id - self.ctrl_actuator_offset
            if not 0 <= ctrl_index < ctrl_len:
                raise RuntimeError(f"SONIC actuator {joint_name!r} maps outside the ctrl buffer")
            remapped[ctrl_index] = float(effort_limit)
        if self.hand_spec.name == "brainco":
            for side in ("left", "right"):
                ids = (
                    getattr(self, f"{side}_hand_actuated_actuator_index")
                    - self.ctrl_actuator_offset
                )
                remapped[ids] = self.hand_spec.effort_limits
        self.torque_limit = remapped

    def prepare_inspire_mjcf(self, path: str) -> mujoco.MjSpec:
        """Apply the repository DFQ config to downloaded G1 assets before compilation."""
        root = ET.parse(path).getroot()
        dfq = ET.parse(INSPIRE_DFQ_PATH).getroot()
        compiler = root.find("compiler")
        compiler.set("meshdir", str(Path(path).parent / compiler.get("meshdir", "")))
        for option in list(root.findall("option")):
            root.remove(option)
        root.append(deepcopy(dfq.find("option")))
        for side, prefix in (("left", "L"), ("right", "R")):
            wrist = root.find(f".//body[@name='{side}_wrist_yaw_link']")
            base_name = f"{prefix}_hand_base_link"
            for geom in list(wrist.findall("geom")):
                if geom.get("mesh") == base_name:
                    wrist.remove(geom)
            for body in list(wrist.findall("body")):
                if body.get("name", "").startswith(f"{prefix}_"):
                    wrist.remove(body)
            wrist.append(deepcopy(dfq.find(f"worldbody/body[@name='{base_name}']")))
        for mesh in dfq.findall("asset/mesh"):
            current = root.find(f"asset/mesh[@name='{mesh.get('name')}']")
            if current is not None:
                root.find("asset").remove(current)
            root.find("asset").append(deepcopy(mesh))
        for tag in ("actuator", "contact", "equality", "custom"):
            section = root.find(tag)
            if section is None:
                section = ET.SubElement(root, tag)
            for node in list(section):
                if any(
                    value.startswith(("L_", "R_", "collision_hand_"))
                    for value in node.attrib.values()
                ):
                    section.remove(node)
            section.extend(deepcopy(list(dfq.find(tag))))
        spec = mujoco.MjSpec.from_string(ET.tostring(root, encoding="unicode"))
        spec.option.timestep = get_hand_spec("inspire").physics_dt
        return spec

    def prepare_brainco_mjcf(self, path: str) -> mujoco.MjSpec:
        """Replace the stock wrist forks and hands with the bound BC2/Revo2 assembly."""
        root = ET.parse(path).getroot()
        overlay = ET.parse(BRAINCO_CONFIG_PATH).getroot()
        compiler = root.find("compiler")
        compiler.set("meshdir", str(Path(path).parent / compiler.get("meshdir", "")))
        compiler.set("balanceinertia", "true")
        removed_names = set()
        for replacement in overlay.findall("worldbody/body"):
            wrist = root.find(f".//body[@name='{replacement.get('name')}']")
            side = replacement.get("name").split("_")[0]
            for node in list(wrist):
                if node.tag in {"inertial", "geom"} or node.get("name", "").startswith(
                    f"{side}_hand_"
                ):
                    removed_names.update(
                        item.get("name") for item in node.iter() if item.get("name")
                    )
                    wrist.remove(node)
            wrist.extend(deepcopy(list(replacement)))
        for section in root:
            if section.tag in {"actuator", "sensor", "contact", "equality"}:
                for node in list(section):
                    if removed_names.intersection(node.attrib.values()):
                        section.remove(node)
        for mesh in overlay.findall("asset/mesh[@file]"):
            filename = mesh.get("file")
            bundled_path = BRAINCO_CONFIG_PATH.parent / filename
            mesh.set(
                "file",
                str(bundled_path)
                if bundled_path.is_file()
                else resolve_data_path(filename, auto_download=False),
            )
        for tag in ("default", "asset", "custom", "actuator", "contact", "equality"):
            section = root.find(tag)
            if section is None:
                section = ET.SubElement(root, tag)
            section.extend(deepcopy(list(overlay.find(tag))))
        for node in list(root.findall("option")) + list(root.findall("keyframe")):
            root.remove(node)
        root.append(deepcopy(overlay.find("option")))
        return mujoco.MjSpec.from_string(ET.tostring(root, encoding="unicode"))

    def _configure_sonic_actuation_model(self) -> None:
        active_actuator_ids = np.concatenate(
            (
                self.body_actuator_index,
                self.left_hand_actuated_actuator_index,
                self.right_hand_actuated_actuator_index,
            )
        )

        if self.hand_spec.name == "inspire":
            for actuator_id in self.body_actuator_index:
                ctrl_index = int(actuator_id) - self.ctrl_actuator_offset
                effort_limit = float(self.torque_limit[ctrl_index])
                self._set_direct_torque_actuator(int(actuator_id), effort_limit)

            active_hand_actuator_ids = np.concatenate(
                (self.left_hand_actuated_actuator_index, self.right_hand_actuated_actuator_index)
            )
            for actuator_id in active_hand_actuator_ids:
                actuator_id = int(actuator_id)
                self._set_direct_torque_actuator(actuator_id, INSPIRE_EFFORT_LIMIT)
                self.torque_limit[actuator_id - self.ctrl_actuator_offset] = INSPIRE_EFFORT_LIMIT

            active_hand_names = set(self.left_hand_actuated_joints) | set(
                self.right_hand_actuated_joints
            )
            mimic_actuator_ids = []
            for joint_name in self.hand_names:
                if joint_name in active_hand_names:
                    continue
                actuator_id = mujoco.mj_name2id(
                    self.mjModel, mujoco.mjtObj.mjOBJ_ACTUATOR, joint_name
                )
                if actuator_id != -1:
                    # DFQ equality constraints drive followers; a second PD fights the coupling.
                    self.mjModel.actuator_gainprm[actuator_id, :] = 0.0
                    self.mjModel.actuator_biasprm[actuator_id, :] = 0.0
                    mimic_actuator_ids.append(actuator_id)

            for joint_name in self._body_joint_names:
                joint_id = self.mjModel.joint(joint_name).id
                dof_id = int(self.mjModel.jnt_dofadr[joint_id])
                self.mjModel.dof_damping[dof_id] = 0.05
                self.mjModel.dof_armature[dof_id] = 0.01
                self.mjModel.dof_frictionloss[dof_id] = 0.1 if "wrist" in joint_name else 0.2

            closed_limits = dict(
                zip(
                    (*self.left_hand_actuated_joints, *self.right_hand_actuated_joints),
                    (*self.hand_spec.close_qpos_left, *self.hand_spec.close_qpos_right),
                    strict=True,
                )
            )
            for mimic, (parent, ratio, offset) in (
                self.hand_spec.left_mimic_map | self.hand_spec.right_mimic_map
            ).items():
                closed_limits[mimic] = closed_limits[parent] * ratio + offset
            for joint_name in self.hand_names:
                joint_id = self.mjModel.joint(joint_name).id
                dof_id = int(self.mjModel.jnt_dofadr[joint_id])
                self.mjModel.jnt_actfrclimited[joint_id] = 1
                self.mjModel.jnt_actfrcrange[joint_id] = (
                    -INSPIRE_EFFORT_LIMIT,
                    INSPIRE_EFFORT_LIMIT,
                )
                self.mjModel.jnt_solref[joint_id] = (0.002, 1.0)
                self.mjModel.jnt_margin[joint_id] = 0.0
                self.mjModel.jnt_solimp[joint_id] = (0.9999, 0.9999, 0.001, 0.5, 2.0)
                self.mjModel.jnt_range[joint_id, 0] = 0.0
                self.mjModel.jnt_range[joint_id, 1] = min(
                    self.mjModel.jnt_range[joint_id, 1], closed_limits[joint_name]
                )
                self.mjModel.dof_damping[dof_id] = 0.0
                self.mjModel.dof_frictionloss[dof_id] = 0.0

            # Armature changes invalidate the inverse weights used by joint constraints.
            mujoco.mj_setConst(self.mjModel, self.mjData)
            self._inspire_hand_targets = np.tile(self.hand_spec.init_qpos, (2, 1))  # (2, 6), rad

            print(
                "[G1Sonic] SPIDER Inspire hand dynamics with SONIC torque control: "
                f"active_actuators={len(active_actuator_ids)} "
                f"mimic_position_actuators={len(mimic_actuator_ids)}"
            )

        if self.hand_spec.name == "brainco":
            self._reset_brainco_hand_reference()
        self._validate_direct_torque_actuators(
            self.body_actuator_index if self.hand_spec.name == "brainco" else active_actuator_ids
        )

    def _set_direct_torque_actuator(self, actuator_id: int, effort_limit: float | None) -> None:
        self.mjModel.actuator_dyntype[actuator_id] = mujoco.mjtDyn.mjDYN_NONE
        self.mjModel.actuator_gaintype[actuator_id] = mujoco.mjtGain.mjGAIN_FIXED
        self.mjModel.actuator_biastype[actuator_id] = mujoco.mjtBias.mjBIAS_NONE
        self.mjModel.actuator_dynprm[actuator_id, :] = 0.0
        self.mjModel.actuator_dynprm[actuator_id, 0] = 1.0
        self.mjModel.actuator_gainprm[actuator_id, :] = 0.0
        self.mjModel.actuator_gainprm[actuator_id, 0] = 1.0
        self.mjModel.actuator_biasprm[actuator_id, :] = 0.0
        self.mjModel.actuator_ctrllimited[actuator_id] = int(effort_limit is not None)
        if effort_limit is None:
            self.mjModel.actuator_forcelimited[actuator_id] = 0
        else:
            self.mjModel.actuator_ctrlrange[actuator_id, :] = (-effort_limit, effort_limit)

    def _validate_direct_torque_actuators(self, actuator_ids: np.ndarray) -> None:
        invalid = []
        for actuator_id in actuator_ids:
            actuator_id = int(actuator_id)
            is_direct_torque = (
                self.mjModel.actuator_dyntype[actuator_id] == mujoco.mjtDyn.mjDYN_NONE
                and self.mjModel.actuator_gaintype[actuator_id] == mujoco.mjtGain.mjGAIN_FIXED
                and self.mjModel.actuator_biastype[actuator_id] == mujoco.mjtBias.mjBIAS_NONE
                and np.isclose(self.mjModel.actuator_gainprm[actuator_id, 0], 1.0)
                and np.allclose(self.mjModel.actuator_biasprm[actuator_id], 0.0)
            )
            if not is_direct_torque:
                invalid.append(self.mjModel.actuator(actuator_id).name)
        if invalid:
            raise RuntimeError(
                "SONIC requires direct-torque MuJoCo actuators; invalid actuators: "
                + ", ".join(invalid)
            )

    def _get_dof_indices_by_class(self, mjSpec, mjModel) -> dict[str, list[int]]:
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".xml") as f:
            # mujoco.mj_saveLastXML(f.name, mjModel)
            f.write(mjSpec.to_xml())
            temp_xml_path = f.name

        try:
            tree = ET.parse(temp_xml_path)
            root = tree.getroot()

            joint_class_map = {}
            for joint_element in root.findall(".//joint[@class]"):
                joint_name = joint_element.get("name")
                joint_class = joint_element.get("class")
                if joint_name and joint_class:
                    joint_id = mujoco.mj_name2id(mjModel, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                    if joint_id != -1:
                        dof_adr = mjModel.jnt_dofadr[joint_id]
                        if joint_class not in joint_class_map:
                            joint_class_map[joint_class] = []
                        joint_class_map[joint_class].append(dof_adr)
        finally:
            os.remove(temp_xml_path)

        return joint_class_map

    @property
    def pelvis_vz(self) -> float:
        return self.mjData.qvel[2]

    @property
    def pelvis_z(self) -> float:
        return self.mjData.qpos[2]

    def _apply_elastic_band_force(self) -> None:
        """Apply suspension force if the elastic band is enabled.

        This is intentionally independent from the action type so that the robot
        can be suspended while still executing WBC torque.
        """
        if (
            self.elastic_band is None
            or self.band_attached_link is None
            or not self.elastic_band.enable
        ):
            return

        pose = np.concatenate(
            [
                self.mjData.xpos[self.band_attached_link],
                self.mjData.xquat[self.band_attached_link],
                np.zeros(6),
            ]
        )

        mujoco.mj_objectVelocity(
            self.mjModel,
            self.mjData,
            mujoco.mjtObj.mjOBJ_BODY,
            self.band_attached_link,
            pose[7:13],
            0,
        )
        pose[7:10], pose[10:13] = pose[10:13], pose[7:10].copy()

        self.mjData.xfrc_applied[self.band_attached_link] = self.elastic_band.Advance(pose)

    def prepare_obs(self) -> dict[str, Any]:
        obs = {}
        if self.use_floating_root_link:  # move to this class
            obs["floating_base_pose"] = self.mjData.qpos[:7]
            obs["floating_base_vel"] = self.mjData.qvel[:6]
            obs["floating_base_acc"] = self.mjData.qacc[:6]
        else:
            obs["floating_base_pose"] = np.zeros(7)
            obs["floating_base_vel"] = np.zeros(6)
            obs["floating_base_acc"] = np.zeros(6)

        obs["secondary_imu_quat"] = self.mjData.xquat[self.torso_index]

        pose = np.zeros(13)
        torso_link = self.mjModel.body("torso_link").id
        # mj_objectVelocity returns [ang_vel, lin_vel]; swap to [lin_vel, ang_vel]
        mujoco.mj_objectVelocity(
            self.mjModel, self.mjData, mujoco.mjtObj.mjOBJ_BODY, torso_link, pose[7:13], 1
        )
        pose[7:10], pose[10:13] = (
            pose[10:13],
            pose[7:10].copy(),
        )
        obs["secondary_imu_vel"] = pose[7:13]

        obs["body_q"] = self.mjData.qpos[self.body_joint_index + 7 - 1]
        obs["body_dq"] = self.mjData.qvel[self.body_joint_index + 6 - 1]
        obs["body_ddq"] = self.mjData.qacc[self.body_joint_index + 6 - 1]
        obs["body_tau_est"] = self.mjData.actuator_force[self.body_actuator_index]
        if self.num_hand_dof > 0:
            obs["left_hand_q"] = self.mjData.qpos[self.left_hand_model_index + self.qpos_offset - 1]
            obs["left_hand_dq"] = self.mjData.qvel[
                self.left_hand_model_index + self.qvel_offset - 1
            ]
            obs["left_hand_ddq"] = self.mjData.qacc[
                self.left_hand_model_index + self.qvel_offset - 1
            ]
            obs["left_hand_tau_est"] = (
                self.mjData.qfrc_actuator[self.mjModel.jnt_dofadr[self.left_hand_model_index]]
                if self.hand_spec.name == "brainco"
                else self.mjData.actuator_force[self.left_hand_model_actuator_index]
            )
            obs["right_hand_q"] = self.mjData.qpos[
                self.right_hand_model_index + self.qpos_offset - 1
            ]
            obs["right_hand_dq"] = self.mjData.qvel[
                self.right_hand_model_index + self.qvel_offset - 1
            ]
            obs["right_hand_ddq"] = self.mjData.qacc[
                self.right_hand_model_index + self.qvel_offset - 1
            ]
            obs["right_hand_tau_est"] = (
                self.mjData.qfrc_actuator[self.mjModel.jnt_dofadr[self.right_hand_model_index]]
                if self.hand_spec.name == "brainco"
                else self.mjData.actuator_force[self.right_hand_model_actuator_index]
            )
            obs["left_hand_actuated_q"] = self.mjData.qpos[
                self.left_hand_actuated_index + self.qpos_offset - 1
            ]
            obs["left_hand_actuated_dq"] = self.mjData.qvel[
                self.left_hand_actuated_index + self.qvel_offset - 1
            ]
            obs["left_hand_actuated_ddq"] = self.mjData.qacc[
                self.left_hand_actuated_index + self.qvel_offset - 1
            ]
            obs["left_hand_actuated_tau_est"] = self.mjData.actuator_force[
                self.left_hand_actuated_actuator_index
            ]
            obs["right_hand_actuated_q"] = self.mjData.qpos[
                self.right_hand_actuated_index + self.qpos_offset - 1
            ]
            obs["right_hand_actuated_dq"] = self.mjData.qvel[
                self.right_hand_actuated_index + self.qvel_offset - 1
            ]
            obs["right_hand_actuated_ddq"] = self.mjData.qacc[
                self.right_hand_actuated_index + self.qvel_offset - 1
            ]
            obs["right_hand_actuated_tau_est"] = self.mjData.actuator_force[
                self.right_hand_actuated_actuator_index
            ]
        obs["time"] = self.mjData.time
        return obs  # joints in mjcf order (thumb, middle, index)

    def get_actuators_action(self) -> dict[str, float]:
        """Get the current actuator actions of the robot."""
        return {a: v.ctrl[0] for a, v in self.actuators.items()}

    def get_robot_qpos(self) -> dict[str, float]:
        """Get the current joint positions of the robot."""
        # return np.array([j.qpos[0] for j in self.joints])
        return {j: v.qpos[0] for j, v in self.joints.items()}

    def step(self, command, eval=False):  # FIXME remove
        ...

    def get_robot_pose(self):
        return np.round(self.mjData.qpos[:7], 3)

    def apply_action(self, action_cmd: ActionCmd) -> None:
        assert isinstance(self.controller, WholeBodyEEFController)

        self._apply_elastic_band_force()

        match action_cmd.type:
            case "elastic_band":
                pass

            case "wbc_torque":
                low_cmd = action_cmd["low_cmd"]
                use_sensor = action_cmd["use_sensor"]
                left_hand_cmd = action_cmd["left_hand_cmd"]
                right_hand_cmd = action_cmd["right_hand_cmd"]

                body_torques = self.compute_body_torques(low_cmd, use_sensor)
                if self.hand_spec.name in {"inspire", "brainco"}:
                    hand_controls = self._compute_position_hand_controls(
                        [motor.q for motor in left_hand_cmd.motor_cmd[: self.hand_spec.dof]],
                        [motor.q for motor in right_hand_cmd.motor_cmd[: self.hand_spec.dof]],
                        action_cmd["brainco_feedback_q"],
                    )
                else:
                    hand_controls = self.compute_hand_torques(left_hand_cmd, right_hand_cmd)

                self.torques[:] = 0.0
                self.torques[self.body_ctrl_index] = body_torques
                if self.num_hand_dof > 0:
                    left_count = len(self.left_hand_actuated_index)
                    self.torques[self.left_hand_actuated_ctrl_index] = hand_controls[:left_count]
                    self.torques[self.right_hand_actuated_ctrl_index] = hand_controls[left_count:]

                self._apply_mimic_joints()
                self._write_actuator_controls()

            case "sonic_stand_pd_hold":
                target_q = np.asarray(
                    action_cmd["target_q"]
                    if action_cmd["target_q"] is not None
                    else SONIC_DEFAULT_STAND_Q,
                    dtype=np.float64,
                )
                kp = np.asarray(
                    action_cmd["kp"]
                    if action_cmd["kp"] is not None
                    else self.sonic_config.get("MOTOR_KP", [150.0] * self.num_body_dof),
                    dtype=np.float64,
                )
                kd = np.asarray(
                    action_cmd["kd"]
                    if action_cmd["kd"] is not None
                    else self.sonic_config.get("MOTOR_KD", [3.0] * self.num_body_dof),
                    dtype=np.float64,
                )
                q_cur = self.mjData.qpos[self.body_joint_index + self.qpos_offset - 1]
                dq_cur = self.mjData.qvel[self.body_joint_index + self.qvel_offset - 1]
                body_torques = kp[: self.num_body_dof] * (target_q[: self.num_body_dof] - q_cur)
                body_torques += kd[: self.num_body_dof] * (0.0 - dq_cur)
                self.torques[:] = 0.0
                self.torques[self.body_ctrl_index] = body_torques
                if self.hand_spec.name in {"inspire", "brainco"}:
                    hand_controls = self._compute_position_hand_controls(
                        None, None, action_cmd["brainco_feedback_q"]
                    )
                    count = self.hand_spec.dof
                    self.torques[self.left_hand_actuated_ctrl_index] = hand_controls[:count]
                    self.torques[self.right_hand_actuated_ctrl_index] = hand_controls[count:]
                self._apply_mimic_joints()
                self._write_actuator_controls()

            case "decoupled_wbc":
                target_q = action_cmd["target_q"]  # 29 body joints in actuator order
                # PD position control using per-joint gains from decoupled_wbc config
                kp = np.array(self.sonic_config.get("MOTOR_KP", [100.0] * self.num_body_dof))
                kd = np.array(self.sonic_config.get("MOTOR_KD", [5.0] * self.num_body_dof))

                q_cur = self.mjData.qpos[self.body_joint_index + self.qpos_offset - 1]
                dq_cur = self.mjData.qvel[self.body_joint_index + self.qvel_offset - 1]

                body_torques = kp * (target_q - q_cur) + kd * (0 - dq_cur)
                self.torques[:] = 0.0
                self.torques[self.body_ctrl_index] = body_torques

                # Hand PD control (driven by trigger/grip via decoupled WBC teleop IK)
                if self.hand_spec.name in {"inspire", "brainco"}:
                    hand_controls = self._compute_position_hand_controls(
                        action_cmd["left_hand_q"],
                        action_cmd["right_hand_q"],
                        action_cmd["brainco_feedback_q"],
                    )
                    count = self.hand_spec.dof
                    self.torques[self.left_hand_actuated_ctrl_index] = hand_controls[:count]
                    self.torques[self.right_hand_actuated_ctrl_index] = hand_controls[count:]
                elif self.num_hand_dof > 0:
                    left_hand_q = action_cmd["left_hand_q"]
                    right_hand_q = action_cmd["right_hand_q"]
                    if self.hand_spec.kp_map is not None:
                        hand_kp = np.array(
                            [
                                self.hand_spec.kp_map.get(j, 5.0)
                                for j in self.left_hand_actuated_joints
                            ]
                        )
                    else:
                        # Default: 5.0 for thumb, 2.5 for others (balancing grip forces)
                        hand_kp = (
                            np.array([5.0] + [2.5] * (self.hand_spec.dof - 1))
                            if self.hand_spec.dof > 1
                            else np.array([5.0])
                        )
                    hand_kd = self.hand_spec.kd
                    if left_hand_q is not None:
                        lh_q_cur = self.mjData.qpos[
                            self.left_hand_actuated_index + self.qpos_offset - 1
                        ]
                        lh_dq_cur = self.mjData.qvel[
                            self.left_hand_actuated_index + self.qvel_offset - 1
                        ]
                        self.torques[self.left_hand_actuated_ctrl_index] = hand_kp * (
                            left_hand_q - lh_q_cur
                        ) + hand_kd * (0 - lh_dq_cur)
                    if right_hand_q is not None:
                        rh_q_cur = self.mjData.qpos[
                            self.right_hand_actuated_index + self.qpos_offset - 1
                        ]
                        rh_dq_cur = self.mjData.qvel[
                            self.right_hand_actuated_index + self.qvel_offset - 1
                        ]
                        self.torques[self.right_hand_actuated_ctrl_index] = hand_kp * (
                            right_hand_q - rh_q_cur
                        ) + hand_kd * (0 - rh_dq_cur)

                self._apply_mimic_joints()

                self._write_actuator_controls()

    def _write_actuator_controls(self) -> None:
        controls = np.clip(self.torques, -self.torque_limit, self.torque_limit)  # (nu-offset,)
        if self.hand_spec.name == "brainco":
            # Position references are radians; force limits live on the actuators.
            indexes = np.concatenate(
                (self.left_hand_actuated_ctrl_index, self.right_hand_actuated_ctrl_index)
            )  # (12,)
            controls[indexes] = self.torques[indexes]
        self.mjData.ctrl = (
            np.concatenate((np.zeros(6), controls)) if self.sonic_config["FREE_BASE"] else controls
        )  # (nu,)

    def compute_body_torques(self, low_cmd, use_sensor) -> np.ndarray:
        # PD control: tau = tau_ff + kp * (q_des - q) + kd * (dq_des - dq)
        body_torques = np.zeros(self.num_body_dof)  # (29,)
        # Native commands materialize all motor entries on every property access.
        motor_cmd = low_cmd.motor_cmd if self.num_body_dof else ()
        # if self.unitree_bridge is not None and self.unitree_bridge.low_cmd:
        for i in range(self.num_body_dof):  # self.unitree_bridge.num_body_motor
            if use_sensor:  # self.unitree_bridge.use_sensor:
                body_torques[i] = (
                    motor_cmd[i].tau  # type:ignore
                    + motor_cmd[i].kp  # type:ignore
                    * (motor_cmd[i].q - self.mjData.sensordata[i])  # type:ignore
                    + motor_cmd[i].kd  # type:ignore
                    * (
                        motor_cmd[i].dq  # type:ignore
                        - self.mjData.sensordata[i + self.num_body_dof]
                    )
                )
            else:
                body_torques[i] = (
                    motor_cmd[i].tau  # type:ignore
                    + motor_cmd[i].kp  # type:ignore
                    * (
                        motor_cmd[i].q  # type:ignore
                        - self.mjData.qpos[self.body_joint_index[i] + self.qpos_offset - 1]
                    )
                    + motor_cmd[i].kd  # type:ignore
                    * (
                        motor_cmd[i].dq  # type:ignore
                        - self.mjData.qvel[self.body_joint_index[i] + self.qvel_offset - 1]
                    )
                )
        return body_torques

    def _compute_position_hand_controls(
        self, left_target, right_target, brainco_feedback_q=None
    ) -> np.ndarray:
        if brainco_feedback_q is not None:
            if self.hand_spec.name != "brainco":
                raise ValueError("Feedback targets require the BrainCo hand")
            feedback = np.asarray(brainco_feedback_q, dtype=np.float64)  # (2, 6), rad
            if feedback.shape != (2, self.hand_spec.dof) or not np.isfinite(feedback).all():
                raise ValueError("BrainCo feedback must be finite with shape (2, 6), in radians")
            joints = np.array(
                [self.left_hand_actuated_index, self.right_hand_actuated_index]
            )  # (2, 6)
            ranges = self.mjModel.jnt_range[joints]  # (2, 6, 2), rad
            reference = np.clip(feedback, ranges[:, :, 0], ranges[:, :, 1])  # (2, 6), rad
            # Measured trajectories already contain their motion timing; do not rate-limit again.
            self._brainco_hand_targets = reference.copy()  # (2, 6), rad
            self._brainco_hand_reference = reference.copy()  # (2, 6), rad
            self._brainco_reference_velocity = np.zeros_like(reference)  # (2, 6), rad/s
            return reference.reshape(-1).copy()  # (12,), rad
        if self.hand_spec.name == "brainco":
            return self._compute_brainco_hand_reference(left_target, right_target)
        return self._compute_inspire_hand_torques(left_target, right_target)

    def _reset_brainco_hand_reference(self) -> None:
        self._brainco_hand_targets = np.tile(self.hand_spec.init_qpos, (2, 1))  # (2, 6), rad
        self._brainco_hand_reference = self._brainco_hand_targets.copy()  # (2, 6), rad
        self._brainco_reference_velocity = np.zeros((2, 6))  # (2, 6), rad/s
        self._brainco_close_delay = np.asarray(_brainco_values("brainco_close_delay"))  # (6,), s
        self._brainco_command_history = [[], []]
        self._brainco_previous_command = self._brainco_hand_targets.copy()  # (2, 6), rad
        self._brainco_closing = np.zeros((2, 6), dtype=bool)  # (2, 6)

    def _delay_brainco_closing(self, side: int, target: np.ndarray) -> np.ndarray:
        now = float(self.mjData.time)
        history = self._brainco_command_history[side]
        if not history:
            history.append(
                (
                    now - float(self._brainco_close_delay.max()),
                    self._brainco_previous_command[side].copy(),
                )
            )
        change = target - self._brainco_previous_command[side]  # (6,), rad
        self._brainco_closing[side] = np.where(
            change > 0, True, np.where(change < 0, False, self._brainco_closing[side])
        )  # (6,)
        self._brainco_previous_command[side] = target.copy()  # (6,), rad
        history.append((now, target.copy()))
        delayed = target.copy()  # (6,), rad
        for channel in np.flatnonzero(
            self._brainco_closing[side] & (self._brainco_close_delay > 0)
        ):
            cutoff = now - self._brainco_close_delay[channel]
            delayed[channel] = next(
                value[channel] for stamp, value in reversed(history) if stamp <= cutoff + 1e-9
            )
        while (
            len(history) > 2 and history[1][0] < now - float(self._brainco_close_delay.max()) - 1e-9
        ):
            history.pop(0)
        return delayed

    def _compute_brainco_hand_reference(self, left_target, right_target) -> np.ndarray:
        dt = self.mjModel.opt.timestep
        close_vmax = np.asarray(self.hand_spec.reference_velocity)  # (6,), rad/s
        open_vmax = np.asarray(self.hand_spec.reference_open_velocity)  # (6,), rad/s
        close_acceleration = np.asarray(self.hand_spec.reference_acceleration)  # (6,), rad/s2
        for side, target, indexes in (
            (0, left_target, self.left_hand_actuated_index),
            (1, right_target, self.right_hand_actuated_index),
        ):
            ranges = self.mjModel.jnt_range[indexes]  # (6, 2), rad
            if target is not None:
                target = np.clip(
                    target, self.hand_spec.init_qpos, self.hand_spec.close_qpos_left
                )  # (6,), rad
                target = self._delay_brainco_closing(side, target)  # (6,), rad
                actual = self.mjData.qpos[self.mjModel.jnt_qposadr[indexes]]  # (6,), rad
                opening = (target < self._brainco_hand_targets[side]) & (target < actual)
                opening[1] = False
                self._brainco_hand_reference[side, opening] = actual[opening]
                speed = self.mjData.qvel[self.mjModel.jnt_dofadr[indexes]]  # (6,), rad/s
                self._brainco_reference_velocity[side, opening] = np.clip(
                    speed, -open_vmax, open_vmax
                )[opening]
                self._brainco_hand_targets[side] = target
            delta = self._brainco_hand_targets[side] - self._brainco_hand_reference[side]
            vmax = np.where(delta < 0, open_vmax, close_vmax)  # (6,), rad/s
            acceleration = close_acceleration * vmax / close_vmax  # (6,), rad/s2
            desired = np.clip(delta / dt, -vmax, vmax)  # (6,), rad/s
            velocity = self._brainco_reference_velocity[side] + np.clip(
                desired - self._brainco_reference_velocity[side],
                -acceleration * dt,
                acceleration * dt,
            )  # (6,), rad/s
            step = velocity * dt  # (6,), rad
            reached = (step * delta >= 0) & (np.abs(step) >= np.abs(delta))
            proposed = self._brainco_hand_reference[side] + np.where(reached, delta, step)
            clipped = np.clip(proposed, ranges[:, 0], ranges[:, 1])  # (6,), rad
            self._brainco_hand_reference[side] = clipped
            self._brainco_reference_velocity[side] = np.where(
                reached | (clipped != proposed), 0, velocity
            )
        return self._brainco_hand_reference.reshape(-1).copy()  # (12,), rad

    def _compute_inspire_hand_torques(self, left_target, right_target) -> np.ndarray:
        for side, target, close in (
            (0, left_target, self.hand_spec.close_qpos_left),
            (1, right_target, self.hand_spec.close_qpos_right),
        ):
            if target is not None:
                self._inspire_hand_targets[side] = np.clip(target, self.hand_spec.init_qpos, close)
        indexes = np.concatenate(
            (self.left_hand_actuated_index, self.right_hand_actuated_index)
        )  # (12,)
        q = self.mjData.qpos[indexes + self.qpos_offset - 1]  # (12,), rad
        dq = self.mjData.qvel[indexes + self.qvel_offset - 1]  # (12,), rad/s
        kp = np.tile(
            [self.hand_spec.kp_map[joint] for joint in self.left_hand_actuated_joints], 2
        )  # (12,)
        targets = self._inspire_hand_targets.reshape(-1)  # (12,), rad
        return np.clip(
            kp * (targets - q) - self.hand_spec.kd * dq, -INSPIRE_EFFORT_LIMIT, INSPIRE_EFFORT_LIMIT
        )

    def compute_hand_torques(self, left_hand_cmd, right_hand_cmd) -> np.ndarray:
        num_actuated = len(self.left_hand_actuated_joints)
        left_hand_torques = np.zeros(num_actuated)
        right_hand_torques = np.zeros(num_actuated)
        # if self.unitree_bridge is not None and self.unitree_bridge.low_cmd:
        for i in range(num_actuated):  # self.unitree_bridge.num_hand_motor
            left_hand_torques[i] = (
                left_hand_cmd.motor_cmd[i].tau  # type:ignore
                + left_hand_cmd.motor_cmd[i].kp  # type:ignore
                * (
                    left_hand_cmd.motor_cmd[i].q  # type:ignore
                    - self.mjData.qpos[self.left_hand_actuated_index[i] + self.qpos_offset - 1]
                )
                + left_hand_cmd.motor_cmd[i].kd  # type:ignore
                * (
                    left_hand_cmd.motor_cmd[i].dq  # type:ignore
                    - self.mjData.qvel[self.left_hand_actuated_index[i] + self.qvel_offset - 1]
                )  # type:ignore
            )
            right_hand_torques[i] = (
                right_hand_cmd.motor_cmd[i].tau  # type:ignore
                + right_hand_cmd.motor_cmd[i].kp  # type:ignore
                * (
                    right_hand_cmd.motor_cmd[i].q  # type:ignore
                    - self.mjData.qpos[self.right_hand_actuated_index[i] + self.qpos_offset - 1]
                )
                + right_hand_cmd.motor_cmd[i].kd  # type:ignore
                * (
                    right_hand_cmd.motor_cmd[i].dq  # type:ignore
                    - self.mjData.qvel[self.right_hand_actuated_index[i] + self.qvel_offset - 1]
                )
            )
        return np.concatenate((left_hand_torques, right_hand_torques))

    def _apply_mimic_joints(self, *, initialize: bool = False) -> None:
        """Propagate active hand positions to mimic position targets.

        Initialization synchronizes qpos once. DFQ equality constraints enforce
        coupling during simulation; follower actuator slots remain unpowered.
        """
        if self.hand_spec.name == "brainco":
            self._apply_brainco_coupling(initialize=initialize)
            return
        for side_mimic_map, close_qpos in (
            (self.hand_spec.left_mimic_map, self.hand_spec.close_qpos_left),
            (self.hand_spec.right_mimic_map, self.hand_spec.close_qpos_right),
        ):
            if not side_mimic_map:
                continue
            for mimic_joint, (parent_joint, ratio, offset) in side_mimic_map.items():
                try:
                    parent_adr = self.mjModel.joint(parent_joint).qposadr[0]
                    mimic_adr = self.mjModel.joint(mimic_joint).qposadr[0]
                    parent_qpos = float(self.mjData.qpos[parent_adr])
                    if "thumb_" in mimic_joint and offset > 0.0:
                        offset *= float(np.clip(parent_qpos / close_qpos[1], 0.0, 1.0))
                    target = parent_qpos * ratio + offset
                    target = float(np.clip(target, *self.mjModel.joint(mimic_joint).range))
                    if initialize:
                        self.mjData.qpos[mimic_adr] = target

                    actuator_id = mujoco.mj_name2id(
                        self.mjModel,
                        mujoco.mjtObj.mjOBJ_ACTUATOR,
                        mimic_joint,
                    )
                    if actuator_id != -1:
                        ctrl_index = actuator_id - self.ctrl_actuator_offset
                        if 0 <= ctrl_index < len(self.torques):
                            self.torques[ctrl_index] = target
                        self.mjData.ctrl[actuator_id] = target
                except (IndexError, AttributeError):
                    pass

    def _apply_brainco_coupling(self, *, initialize: bool = False) -> None:
        """Use actual proximal motion to constrain unpowered distal joints."""
        model, data = self.mjModel, self.mjData
        for name, (parent, ratio, offset) in (
            self.hand_spec.left_mimic_map | self.hand_spec.right_mimic_map
        ).items():
            joint = model.joint(name)
            angle = float(data.qpos[model.joint(parent).qposadr[0]])
            raw = ratio * angle + offset
            target = float(np.clip(raw, *joint.range))
            eq = np.flatnonzero(
                (model.eq_type == mujoco.mjtEq.mjEQ_JOINT) & (model.eq_obj1id == joint.id)
            )[0]
            # At a distal limit the proximal can still advance; reversal restores the ratio.
            model.eq_data[eq, :5] = (
                [target, 0, 0, 0, 0]
                if raw >= joint.range[1] or raw < joint.range[0]
                else [offset, ratio, 0, 0, 0]
            )
            if initialize:
                data.qpos[joint.qposadr[0]] = target

    @property
    def head_cam_link(self) -> str:
        """Get the head camera link name."""
        if self.hand_spec.name == "brainco":
            return "pelvis/torso_link/d435_link"
        return "d435_link"
