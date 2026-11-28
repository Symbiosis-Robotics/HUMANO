"""Frozen HUMANO request and action contracts for LeRobot baselines."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from humano.policy.server import (
    ACTION_DIM,
    ACTION_HORIZON,
    IMAGE_SHAPE,
    state_parts,
    validate_action_chunk,
)


@dataclass(frozen=True)
class BaselineSpec:
    name: str
    policy_type: str
    requires_task: bool = False
    objective: str | None = None


BASELINE_SPECS = {
    "act": BaselineSpec("act", "act"),
    "diffusion": BaselineSpec("diffusion", "diffusion"),
    "pi05": BaselineSpec("pi05", "pi05", requires_task=True),
    "flow_matching": BaselineSpec(
        "flow_matching",
        "multi_task_dit",
        requires_task=True,
        objective="flow_matching",
    ),
}


def _vector(values: Any, name: str, size: int) -> np.ndarray:
    result = np.asarray(values, dtype=np.float32)
    # (D,) source units and frame defined by the named contract field
    if result.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {result.shape}")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    return result


def build_runtime_observation(request: Mapping[str, Any], hand_type: str = "dex3") -> dict[str, Any]:
    """Convert the GR00T-aligned HTTP request into the Track B schema."""

    state = request.get("state")
    if not isinstance(state, Mapping):
        raise TypeError("state must be a mapping")
    joint_parts = [_vector(state.get(name), name, size) for name, size in state_parts(hand_type)]
    # 7 x (part_dim,) rad, GR00T joint order
    joint_state = np.concatenate(joint_parts).astype(np.float32, copy=False)
    # (43,) rad, GR00T joint order
    projected_gravity = _vector(
        state.get("projected_gravity"),
        "projected_gravity",
        3,
    )
    # (3,) unitless, robot body frame

    image_fields = request.get("image")
    if not isinstance(image_fields, Mapping):
        raise TypeError("image must be a mapping")
    image = np.asarray(image_fields.get("ego_view"))
    # (360, 640, 3) uint8 or normalized RGB, ego-camera frame
    if image.shape != IMAGE_SHAPE:
        raise ValueError(f"ego_view must have shape {IMAGE_SHAPE}, got {image.shape}")
    if not np.isfinite(image).all():
        raise ValueError("ego_view must contain only finite values")

    instruction = request.get("instruction", "")
    if not isinstance(instruction, str):
        raise TypeError("instruction must be a string")

    return {
        "observation.images.ego_view": image,
        "observation.state": joint_state,
        "observation.projected_gravity": projected_gravity,
        "task": instruction,
    }
