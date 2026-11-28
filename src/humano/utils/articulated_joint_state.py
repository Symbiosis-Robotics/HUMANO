"""Helpers for initializing articulated-object joint state."""

from __future__ import annotations


def resolve_articulated_joint_positions(
    joint_qpos: dict[str, float] | None,
    *,
    num_dofs: int,
) -> list[float]:
    """Return explicit joint positions or a full zero-state fallback."""
    if joint_qpos:
        return list(joint_qpos.values())
    return [0.0] * num_dofs
