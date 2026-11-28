"""
SONIC observation registry and history gatherers.

Mirrors the C++ ``GetObservationRegistry()`` pattern — each observation is
registered with a name, dimension, and a gatherer closure that reads from the
shared ``StateLogger`` and external token buffer, writing into the observation
vector at a given offset.

The registry is *extensible*: to add or remove observations, edit
``observation_config.yaml`` and add the corresponding entry here.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from humano.wbc.state_logger import StateLogger

# ---------------------------------------------------------------------------
# Type alias for gatherer functions
# ---------------------------------------------------------------------------
ObsGatherer = Callable[[np.ndarray, int], bool]
"""Signature: ``(target_buffer: np.ndarray, offset: int) -> bool``."""

# ---------------------------------------------------------------------------
# Quaternion utilities (ported from math_utils.hpp)
# ---------------------------------------------------------------------------


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    """Quaternion conjugate: (w, x, y, z) → (w, -x, -y, -z)."""
    r = q.copy()
    r[..., 1:] *= -1.0
    return r


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate 3-D vector *v* by quaternion *q* (w, x, y, z).

    Matches the C++ ``quat_rotate_d`` implementation exactly.
    """
    q_w = q[..., 0]
    q_vec = q[..., 1:]

    # term_a = v * (2*w² - 1)
    scale_a = 2.0 * q_w * q_w - 1.0
    a = v * scale_a

    # term_b = cross(q_vec, v) * 2*w
    cross_qv = np.cross(q_vec, v)
    b = cross_qv * (q_w * 2.0)

    # term_c = q_vec * (2 * dot(q_vec, v))
    dot_qv = np.dot(q_vec, v)
    c = q_vec * (dot_qv * 2.0)

    return a + b + c


# ---------------------------------------------------------------------------
# Shared state held by the observation pipeline.
# Each gatherer closes over this module-level singleton.
# ---------------------------------------------------------------------------
_state_logger: StateLogger | None = None
_token_data: np.ndarray | None = None

_NUM_JOINTS = 29
_LOG_DT = 0.02  # 50 Hz


def set_shared_state(logger: StateLogger, token_data: np.ndarray | None) -> None:
    """Called by SonicController before each tick's observation gathering."""
    global _state_logger, _token_data
    _state_logger = logger
    _token_data = token_data


# ---------------------------------------------------------------------------
# Token gatherer
# ---------------------------------------------------------------------------
def _gather_token_state(target: np.ndarray, offset: int) -> bool:
    if _token_data is None:
        return False
    n = len(_token_data)
    target[offset : offset + n] = _token_data
    return True


# ---------------------------------------------------------------------------
# History gatherers (match C++ GatherHis* functions)
# ---------------------------------------------------------------------------
def _gather_his_body_joint_positions(
    target: np.ndarray,
    offset: int,
    num_frames: int = 10,
    step_size: int = 1,
) -> bool:
    if _state_logger is None:
        return False
    sample_dt = _LOG_DT * step_size
    hist = _state_logger.get_latest(num_frames, sample_dt, newest_first=False)
    if not hist:
        return False
    n_joints = len(hist[0].body_q)
    for f in range(num_frames):
        entry = hist[f] if f < len(hist) else hist[-1]
        start = offset + f * n_joints
        target[start : start + n_joints] = entry.body_q
    return True


def _gather_his_body_joint_velocities(
    target: np.ndarray,
    offset: int,
    num_frames: int = 10,
    step_size: int = 1,
) -> bool:
    if _state_logger is None:
        return False
    sample_dt = _LOG_DT * step_size
    hist = _state_logger.get_latest(num_frames, sample_dt, newest_first=False)
    if not hist:
        return False
    n_joints = len(hist[0].body_dq)
    for f in range(num_frames):
        entry = hist[f] if f < len(hist) else hist[-1]
        start = offset + f * n_joints
        target[start : start + n_joints] = entry.body_dq
    return True


def _gather_his_last_actions(
    target: np.ndarray,
    offset: int,
    num_frames: int = 10,
    step_size: int = 1,
) -> bool:
    if _state_logger is None:
        return False
    sample_dt = _LOG_DT * step_size
    hist = _state_logger.get_latest(num_frames, sample_dt, newest_first=False)
    if not hist:
        return False
    n_actions = len(hist[0].last_action) if hist else _NUM_JOINTS
    for f in range(num_frames):
        entry = hist[f] if f < len(hist) else hist[-1]
        start = offset + f * n_actions
        target[start : start + n_actions] = entry.last_action
    return True


def _gather_his_base_angular_velocity(
    target: np.ndarray,
    offset: int,
    num_frames: int = 10,
    step_size: int = 1,
) -> bool:
    if _state_logger is None:
        return False
    sample_dt = _LOG_DT * step_size
    hist = _state_logger.get_latest(num_frames, sample_dt, newest_first=False)
    if not hist:
        return False
    for f in range(num_frames):
        entry = hist[f] if f < len(hist) else hist[-1]
        start = offset + f * 3
        target[start : start + 3] = entry.base_ang_vel
    return True


def _gather_his_gravity_dir(
    target: np.ndarray,
    offset: int,
    num_frames: int = 10,
    step_size: int = 1,
) -> bool:
    if _state_logger is None:
        return False
    sample_dt = _LOG_DT * step_size
    hist = _state_logger.get_latest(num_frames, sample_dt, newest_first=False)
    if not hist:
        return False
    for f in range(num_frames):
        entry = hist[f] if f < len(hist) else hist[-1]
        # C++: gravity_dir = quat_rotate(quat_conjugate(base_quat), {0,0,-1})
        gravity_dir_inv = quat_conjugate(entry.base_quat)
        gravity_dir = quat_rotate(gravity_dir_inv, np.array([0.0, 0.0, -1.0]))
        start = offset + f * 3
        target[start : start + 3] = gravity_dir
    return True


# ---------------------------------------------------------------------------
# Observation registry
# ---------------------------------------------------------------------------
# Each entry:  (name, dimension, gatherer_factory)
#
# *dimension* is the pre-computed output size.  *gatherer_factory* is a
# callable that returns a gatherer closing over the required parameters.
# For history-based observations the factory takes no arguments; the
# gatherer reads shared module state.
# ---------------------------------------------------------------------------


def _his_q_10f_1s() -> ObsGatherer:
    return lambda tgt, off: _gather_his_body_joint_positions(tgt, off, 10, 1)


def _his_dq_10f_1s() -> ObsGatherer:
    return lambda tgt, off: _gather_his_body_joint_velocities(tgt, off, 10, 1)


def _his_act_10f_1s() -> ObsGatherer:
    return lambda tgt, off: _gather_his_last_actions(tgt, off, 10, 1)


def _his_angvel_10f_1s() -> ObsGatherer:
    return lambda tgt, off: _gather_his_base_angular_velocity(tgt, off, 10, 1)


def _his_grav_10f_1s() -> ObsGatherer:
    return lambda tgt, off: _gather_his_gravity_dir(tgt, off, 10, 1)


OBS_REGISTRY: dict[str, tuple[int, Callable[[], ObsGatherer]]] = {
    "token_state": (64, lambda: _gather_token_state),
    "his_body_joint_positions_10frame_step1": (290, _his_q_10f_1s),
    "his_body_joint_velocities_10frame_step1": (290, _his_dq_10f_1s),
    "his_last_actions_10frame_step1": (290, _his_act_10f_1s),
    "his_base_angular_velocity_10frame_step1": (30, _his_angvel_10f_1s),
    "his_gravity_dir_10frame_step1": (30, _his_grav_10f_1s),
}
"""Maps observation name → (dimension, factory).  Add entries here to extend."""


def get_total_obs_dim(enabled_names: list[str]) -> int:
    """Compute the total observation vector size for a set of enabled observations."""
    return sum(OBS_REGISTRY[name][0] for name in enabled_names)


def get_gatherer(name: str) -> ObsGatherer:
    """Return a gatherer callable for *name*."""
    _, factory = OBS_REGISTRY[name]
    return factory()
