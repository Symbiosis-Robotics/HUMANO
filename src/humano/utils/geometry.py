from __future__ import annotations

import numpy as np
from transforms3d.quaternions import qmult, quat2mat


def is_valid_rotation_matrix(R, tol=1e-6):
    return (
        R.shape == (3, 3)
        and np.allclose(R.T @ R, np.eye(3), atol=tol)
        and np.isclose(np.linalg.det(R), 1.0, atol=tol)
    )


def compose_pose_with_offset(
    position: np.ndarray,
    orientation: np.ndarray,
    offset_position: np.ndarray,
    offset_orientation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compose a world pose with a local pose offset using wxyz quaternions."""
    world_position = np.asarray(position, dtype=np.float64)
    world_orientation = np.asarray(orientation, dtype=np.float64)
    local_position = np.asarray(offset_position, dtype=np.float64)
    local_orientation = np.asarray(offset_orientation, dtype=np.float64)
    composed_position = world_position + quat2mat(world_orientation) @ local_position
    composed_orientation = qmult(world_orientation, local_orientation)
    return composed_position, composed_orientation / np.linalg.norm(
        composed_orientation
    )


def local_aabbs_to_world(
    local_centers: np.ndarray,
    local_half_extents: np.ndarray,
    world_positions: np.ndarray,
    world_rotations: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Transform local axis-aligned boxes into world axis-aligned bounds."""
    centers = np.asarray(local_centers, dtype=np.float64)
    half_extents = np.asarray(local_half_extents, dtype=np.float64)
    positions = np.asarray(world_positions, dtype=np.float64)
    rotations = np.asarray(world_rotations, dtype=np.float64)
    world_centers = positions + np.einsum("nij,nj->ni", rotations, centers)
    world_half_extents = np.einsum("nij,nj->ni", np.abs(rotations), half_extents)
    return world_centers - world_half_extents, world_centers + world_half_extents


def overlapping_aabb_indices(
    candidate_min: np.ndarray,
    candidate_max: np.ndarray,
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    clearance: float = 0.0,
) -> np.ndarray:
    """Return indices of bounds overlapping a candidate after applying clearance."""
    if clearance < 0:
        raise ValueError("clearance must be non-negative")
    mins = np.asarray(bounds_min, dtype=np.float64)
    maxs = np.asarray(bounds_max, dtype=np.float64)
    if mins.size == 0:
        return np.empty(0, dtype=np.int64)
    overlaps = np.all(mins - clearance <= candidate_max, axis=1) & np.all(
        maxs + clearance >= candidate_min, axis=1
    )
    return np.flatnonzero(overlaps)


def polygon_intersects_aabb_xy(
    polygon: np.ndarray, bounds_min: np.ndarray, bounds_max: np.ndarray
) -> bool:
    """Test a projected mesh face against an XY box after their AABBs overlap."""
    points = np.asarray(polygon, dtype=np.float64)
    lower = np.asarray(bounds_min, dtype=np.float64)
    upper = np.asarray(bounds_max, dtype=np.float64)
    if np.any(np.all((points >= lower) & (points <= upper), axis=1)):
        return True

    def edge_hits_box(start: np.ndarray, end: np.ndarray) -> bool:
        direction = end - start
        enter, leave = 0.0, 1.0
        for axis in range(2):
            if abs(direction[axis]) < 1e-12:
                if start[axis] < lower[axis] or start[axis] > upper[axis]:
                    return False
                continue
            near = (lower[axis] - start[axis]) / direction[axis]
            far = (upper[axis] - start[axis]) / direction[axis]
            enter = max(enter, min(near, far))
            leave = min(leave, max(near, far))
        return enter <= leave

    if any(edge_hits_box(start, end) for start, end in zip(points, np.roll(points, -1, axis=0), strict=True)):
        return True
    corner = lower
    inside = False
    for start, end in zip(points, np.roll(points, -1, axis=0), strict=True):
        if (start[1] > corner[1]) != (end[1] > corner[1]):
            cross_x = start[0] + (corner[1] - start[1]) * (end[0] - start[0]) / (end[1] - start[1])
            if corner[0] < cross_x:
                inside = not inside
    return inside
