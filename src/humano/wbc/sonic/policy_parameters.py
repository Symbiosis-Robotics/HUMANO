"""
Python port of ``policy_parameters.hpp`` — G1 29-DOF policy constants.

All values are bit-identical to the C++ header.  The joint ordering conventions
are documented in that file; in short, **MuJoCo (hardware) order** is used for
output commands and **IsaacLab order** for the policy network.
"""

from __future__ import annotations

import math

import numpy as np

# ---------------------------------------------------------------------------
# Motor armature constants
# ---------------------------------------------------------------------------
ARMATURE_5020 = 0.003609725
ARMATURE_7520_14 = 0.010177520
ARMATURE_7520_22 = 0.025101925
ARMATURE_4010 = 0.00425

# ---------------------------------------------------------------------------
# PID gain computation
# ---------------------------------------------------------------------------
NATURAL_FREQ = 10.0 * 2.0 * math.pi  # 10 Hz
DAMPING_RATIO = 2.0

STIFFNESS_5020 = ARMATURE_5020 * NATURAL_FREQ * NATURAL_FREQ
STIFFNESS_7520_14 = ARMATURE_7520_14 * NATURAL_FREQ * NATURAL_FREQ
STIFFNESS_7520_22 = ARMATURE_7520_22 * NATURAL_FREQ * NATURAL_FREQ
STIFFNESS_4010 = ARMATURE_4010 * NATURAL_FREQ * NATURAL_FREQ

DAMPING_5020 = 2.0 * DAMPING_RATIO * ARMATURE_5020 * NATURAL_FREQ
DAMPING_7520_14 = 2.0 * DAMPING_RATIO * ARMATURE_7520_14 * NATURAL_FREQ
DAMPING_7520_22 = 2.0 * DAMPING_RATIO * ARMATURE_7520_22 * NATURAL_FREQ
DAMPING_4010 = 2.0 * DAMPING_RATIO * ARMATURE_4010 * NATURAL_FREQ

# ---------------------------------------------------------------------------
# Effort limits (used for action-scale computation)
# ---------------------------------------------------------------------------
EFFORT_LIMIT_5020 = 25.0
EFFORT_LIMIT_7520_14 = 88.0
EFFORT_LIMIT_7520_22 = 139.0
EFFORT_LIMIT_4010 = 5.0

# ---------------------------------------------------------------------------
# Joint mapping arrays
# ---------------------------------------------------------------------------
isaaclab_to_mujoco: np.ndarray = np.array(
    [
        0,
        3,
        6,
        9,
        13,
        17,
        1,
        4,
        7,
        10,
        14,
        18,
        2,
        5,
        8,
        11,
        15,
        19,
        21,
        23,
        25,
        27,
        12,
        16,
        20,
        22,
        24,
        26,
        28,
    ],
    dtype=np.int64,
)
"""Map isaaclab-order index → mujoco-order index."""

mujoco_to_isaaclab: np.ndarray = np.array(
    [
        0,
        6,
        12,
        1,
        7,
        13,
        2,
        8,
        14,
        3,
        9,
        15,
        22,
        4,
        10,
        16,
        23,
        5,
        11,
        17,
        24,
        18,
        25,
        19,
        26,
        20,
        27,
        21,
        28,
    ],
    dtype=np.int64,
)
"""Map mujoco-order index → isaaclab-order index."""

# ---------------------------------------------------------------------------
# Action scales (mujoco order)
#
# Computed as:  0.25 * effort_limit / stiffness  (see C++ header)
# ---------------------------------------------------------------------------
g1_action_scale: np.ndarray = np.array(
    [
        0.25 * EFFORT_LIMIT_7520_22 / STIFFNESS_7520_22,  # left_hip_pitch
        0.25 * EFFORT_LIMIT_7520_22 / STIFFNESS_7520_22,  # left_hip_roll
        0.25 * EFFORT_LIMIT_7520_14 / STIFFNESS_7520_14,  # left_hip_yaw
        0.25 * EFFORT_LIMIT_7520_22 / STIFFNESS_7520_22,  # left_knee
        0.25 * EFFORT_LIMIT_5020 / STIFFNESS_5020,  # left_ankle_pitch
        0.25 * EFFORT_LIMIT_5020 / STIFFNESS_5020,  # left_ankle_roll
        0.25 * EFFORT_LIMIT_7520_22 / STIFFNESS_7520_22,  # right_hip_pitch
        0.25 * EFFORT_LIMIT_7520_22 / STIFFNESS_7520_22,  # right_hip_roll
        0.25 * EFFORT_LIMIT_7520_14 / STIFFNESS_7520_14,  # right_hip_yaw
        0.25 * EFFORT_LIMIT_7520_22 / STIFFNESS_7520_22,  # right_knee
        0.25 * EFFORT_LIMIT_5020 / STIFFNESS_5020,  # right_ankle_pitch
        0.25 * EFFORT_LIMIT_5020 / STIFFNESS_5020,  # right_ankle_roll
        0.25 * EFFORT_LIMIT_7520_14 / STIFFNESS_7520_14,  # waist_yaw
        0.25 * EFFORT_LIMIT_5020 / STIFFNESS_5020,  # waist_roll
        0.25 * EFFORT_LIMIT_5020 / STIFFNESS_5020,  # waist_pitch
        0.25 * EFFORT_LIMIT_5020 / STIFFNESS_5020,  # left_shoulder_pitch
        0.25 * EFFORT_LIMIT_5020 / STIFFNESS_5020,  # left_shoulder_roll
        0.25 * EFFORT_LIMIT_5020 / STIFFNESS_5020,  # left_shoulder_yaw
        0.25 * EFFORT_LIMIT_5020 / STIFFNESS_5020,  # left_elbow
        0.25 * EFFORT_LIMIT_5020 / STIFFNESS_5020,  # left_wrist_roll
        0.25 * EFFORT_LIMIT_4010 / STIFFNESS_4010,  # left_wrist_pitch
        0.25 * EFFORT_LIMIT_4010 / STIFFNESS_4010,  # left_wrist_yaw
        0.25 * EFFORT_LIMIT_5020 / STIFFNESS_5020,  # right_shoulder_pitch
        0.25 * EFFORT_LIMIT_5020 / STIFFNESS_5020,  # right_shoulder_roll
        0.25 * EFFORT_LIMIT_5020 / STIFFNESS_5020,  # right_shoulder_yaw
        0.25 * EFFORT_LIMIT_5020 / STIFFNESS_5020,  # right_elbow
        0.25 * EFFORT_LIMIT_5020 / STIFFNESS_5020,  # right_wrist_roll
        0.25 * EFFORT_LIMIT_4010 / STIFFNESS_4010,  # right_wrist_pitch
        0.25 * EFFORT_LIMIT_4010 / STIFFNESS_4010,  # right_wrist_yaw
    ],
    dtype=np.float64,
)

# ---------------------------------------------------------------------------
# PD gains (mujoco order)
# ---------------------------------------------------------------------------
kps: np.ndarray = np.array(
    [
        STIFFNESS_7520_22,  # left_hip_pitch
        STIFFNESS_7520_22,  # left_hip_roll
        STIFFNESS_7520_14,  # left_hip_yaw
        STIFFNESS_7520_22,  # left_knee
        2.0 * STIFFNESS_5020,  # left_ankle_pitch
        2.0 * STIFFNESS_5020,  # left_ankle_roll
        STIFFNESS_7520_22,  # right_hip_pitch
        STIFFNESS_7520_22,  # right_hip_roll
        STIFFNESS_7520_14,  # right_hip_yaw
        STIFFNESS_7520_22,  # right_knee
        2.0 * STIFFNESS_5020,  # right_ankle_pitch
        2.0 * STIFFNESS_5020,  # right_ankle_roll
        STIFFNESS_7520_14,  # waist_yaw
        2.0 * STIFFNESS_5020,  # waist_roll
        2.0 * STIFFNESS_5020,  # waist_pitch
        STIFFNESS_5020,  # left_shoulder_pitch
        STIFFNESS_5020,  # left_shoulder_roll
        STIFFNESS_5020,  # left_shoulder_yaw
        STIFFNESS_5020,  # left_elbow
        STIFFNESS_5020,  # left_wrist_roll
        STIFFNESS_4010,  # left_wrist_pitch
        STIFFNESS_4010,  # left_wrist_yaw
        STIFFNESS_5020,  # right_shoulder_pitch
        STIFFNESS_5020,  # right_shoulder_roll
        STIFFNESS_5020,  # right_shoulder_yaw
        STIFFNESS_5020,  # right_elbow
        STIFFNESS_5020,  # right_wrist_roll
        STIFFNESS_4010,  # right_wrist_pitch
        STIFFNESS_4010,  # right_wrist_yaw
    ],
    dtype=np.float64,
)

kds: np.ndarray = np.array(
    [
        DAMPING_7520_22,  # left_hip_pitch
        DAMPING_7520_22,  # left_hip_roll
        DAMPING_7520_14,  # left_hip_yaw
        DAMPING_7520_22,  # left_knee
        2.0 * DAMPING_5020,  # left_ankle_pitch
        2.0 * DAMPING_5020,  # left_ankle_roll
        DAMPING_7520_22,  # right_hip_pitch
        DAMPING_7520_22,  # right_hip_roll
        DAMPING_7520_14,  # right_hip_yaw
        DAMPING_7520_22,  # right_knee
        2.0 * DAMPING_5020,  # right_ankle_pitch
        2.0 * DAMPING_5020,  # right_ankle_roll
        DAMPING_7520_14,  # waist_yaw
        2.0 * DAMPING_5020,  # waist_roll
        2.0 * DAMPING_5020,  # waist_pitch
        DAMPING_5020,  # left_shoulder_pitch
        DAMPING_5020,  # left_shoulder_roll
        DAMPING_5020,  # left_shoulder_yaw
        DAMPING_5020,  # left_elbow
        DAMPING_5020,  # left_wrist_roll
        DAMPING_4010,  # left_wrist_pitch
        DAMPING_4010,  # left_wrist_yaw
        DAMPING_5020,  # right_shoulder_pitch
        DAMPING_5020,  # right_shoulder_roll
        DAMPING_5020,  # right_shoulder_yaw
        DAMPING_5020,  # right_elbow
        DAMPING_5020,  # right_wrist_roll
        DAMPING_4010,  # right_wrist_pitch
        DAMPING_4010,  # right_wrist_yaw
    ],
    dtype=np.float64,
)

# ---------------------------------------------------------------------------
# Default standing angles (mujoco order)
# ---------------------------------------------------------------------------
default_angles: np.ndarray = np.array(
    [
        -0.312,  # left_hip_pitch
        0.0,  # left_hip_roll
        0.0,  # left_hip_yaw
        0.669,  # left_knee
        -0.363,  # left_ankle_pitch
        0.0,  # left_ankle_roll
        -0.312,  # right_hip_pitch
        0.0,  # right_hip_roll
        0.0,  # right_hip_yaw
        0.669,  # right_knee
        -0.363,  # right_ankle_pitch
        0.0,  # right_ankle_roll
        0.0,  # waist_yaw
        0.0,  # waist_roll
        0.0,  # waist_pitch
        0.2,  # left_shoulder_pitch
        0.2,  # left_shoulder_roll
        0.0,  # left_shoulder_yaw
        0.6,  # left_elbow
        0.0,  # left_wrist_roll
        0.0,  # left_wrist_pitch
        0.0,  # left_wrist_yaw
        0.2,  # right_shoulder_pitch
        -0.2,  # right_shoulder_roll
        0.0,  # right_shoulder_yaw
        0.6,  # right_elbow
        0.0,  # right_wrist_roll
        0.0,  # right_wrist_pitch
        0.0,  # right_wrist_yaw
    ],
    dtype=np.float64,
)

# ---------------------------------------------------------------------------
# Stand-hold PD gains (identical to C++ InitControl)
# ---------------------------------------------------------------------------
SONIC_STAND_PD_KP: np.ndarray = np.array(
    [
        99.1,
        99.1,
        40.2,
        99.1,
        28.5,
        28.5,  # 左腿 hip_pitch/roll/yaw, knee, ankle_pitch/roll
        99.1,
        99.1,
        40.2,
        99.1,
        28.5,
        28.5,  # 右腿
        40.2,
        28.5,
        28.5,  # 腰 yaw/roll/pitch
        14.3,
        14.3,
        14.3,
        14.3,
        14.3,
        16.8,
        16.8,  # 左臂 shld_pitch/roll/yaw, elbow, wrist_roll/pitch/yaw
        14.3,
        14.3,
        14.3,
        14.3,
        14.3,
        16.8,
        16.8,  # 右臂
    ],
    dtype=np.float64,
)

SONIC_STAND_PD_KD: np.ndarray = np.array(
    [
        6.31,
        6.31,
        2.56,
        6.31,
        1.81,
        1.81,
        6.31,
        6.31,
        2.56,
        6.31,
        1.81,
        1.81,
        2.56,
        1.81,
        1.81,
        0.91,
        0.91,
        0.91,
        0.91,
        0.91,
        1.07,
        1.07,
        0.91,
        0.91,
        0.91,
        0.91,
        0.91,
        1.07,
        1.07,
    ],
    dtype=np.float64,
)
