"""
HUMANO: Humanoid Benchmark and Learning Framework

Originally derived from the SIMPLE project.
Copyright (c) 2025 Songlin Wei and Contributors
Copyright (c) 2026 HUMANO Contributors
Licensed under the terms in LICENSE file.
"""

from typing import List, Protocol, runtime_checkable

from .actor import Actor
from .asset import ArticulatedAsset, Asset
from .types import GraspPose, Pose, Vec3, Vec7


class Object(Actor):
    asset: Asset

    def __init__(self, asset: Asset) -> None:
        self.asset = asset


class ArticulatedObject(Actor):
    asset: ArticulatedAsset

    def __init__(self, asset: ArticulatedAsset) -> None:
        self.asset = asset


@runtime_checkable
class SemanticAnnotated(Protocol):
    label: str
    name: str
    description: str | None


@runtime_checkable
class SpatialAnnotated(Protocol):
    """Spatial annotation for the object."""

    keypoints: dict[str, Vec3]
    axes: dict[str, Vec3]
    stable_poses: list  # [Vec7] #: np.ndarray #dict[str, Pose]

    # class Graspable:
    canonical_grasps: list  # [GraspPose]
    functional_grasps: dict  # [str, GraspPose]


# class
