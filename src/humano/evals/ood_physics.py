"""Replay the exact validated initial physics state before any integration step."""

import numpy as np

_FIELDS = ("qpos", "qvel", "act", "ctrl", "mocap_pos", "mocap_quat", "eq_active")
EVAL_TASK_ROBOT_INITIAL_STATE = "_eval_task_robot_initial_state"
EVAL_ROBOT_INITIAL_PARAMS = "_eval_robot_initial_params"


def capture_physics(data):
    return {**{field: getattr(data, field).tolist() for field in _FIELDS}, "time": float(data.time)}


def restore_physics(data, state):
    values = {
        field: np.asarray(state[field], dtype=getattr(data, field).dtype) for field in _FIELDS
    }
    for field, value in values.items():
        if value.size == 0 and getattr(data, field).size == 0:
            value = values[field] = value.reshape(getattr(data, field).shape)
        if value.shape != getattr(data, field).shape or not np.all(np.isfinite(value)):
            raise ValueError(f"Frozen OOD physics {field} does not match the task model")
    if not np.isfinite(state["time"]):
        raise ValueError("Frozen OOD time must be finite")
    for field, value in values.items():
        getattr(data, field)[:] = value
    data.time = float(state["time"])


def apply_robot_joint_positions(data, names, positions):
    values = np.asarray(positions, dtype=float)
    if values.shape != (len(names),) or not np.all(np.isfinite(values)):
        raise ValueError("Recorded OOD robot joints do not match the selected hand model")
    joints = [data.joint(name) for name in names]
    if any(joint.qpos.shape != (1,) for joint in joints):
        raise ValueError("Recorded robot joints must be scalar joints")
    for joint, value in zip(joints, values, strict=True):
        joint.qpos[0] = value
