"""
HUMANO: Humanoid Benchmark and Learning Framework

Originally derived from the SIMPLE project.
Copyright (c) 2025 Songlin Wei and Contributors
Copyright (c) 2026 HUMANO Contributors
Licensed under the terms in LICENSE file.
"""

import os
from pathlib import Path
from typing import Any

import numpy as np

from humano.core.asset import ArticulatedAsset
from humano.core.object import SemanticAnnotated, SpatialAnnotated
from humano.core.types import GraspPose, Pose
from humano.utils import resolve_data_path

from .asset_manager import AssetManager

Articulated_Object_Names = {
    0: "office_chair",
    1: "bottle",
    2: "oven",
    3: "door",
    4: "faucet",
    5: "trash_can",
    6: "button",
    7: "dolly",
    8: "bedside_table",
    9: "pencil_case",
}

Default_Articulated_Init_Joint_Qpos = {
    9: {"articulate_joint_0": -np.deg2rad(100.0)},
}

_LIBRARY_ARTICULATED_OBJECTS = {
    "fridge001": {
        "name": "fridge",
        "mjcf": "fridge001_usd_collision.xml",
        "usd": "fridge001.usd",
    },
}


class ArticulatedObjectAsset(ArticulatedAsset, SemanticAnnotated):
    def __init__(
        self,
        uid: str,
        label: str,
        name: str,
        usd_path: str | None,
        mjcf_path: str,
        description: str | None = None,
        articulate_init_joint_qpos: dict[str, float] | None = None,
    ) -> None:
        super().__init__(uid=uid, usd_path=usd_path, mjcf_path=mjcf_path)
        self.label = label
        self.name = name
        self.description = description
        self.articulate_init_joint_qpos = articulate_init_joint_qpos

    def __repr__(self) -> str:
        return f"ArticulatedObjectAsset(uid={self.uid}, name={self.name})"

    def to_dict(self) -> dict[str, Any]:
        return {
            "res_id": "articulated",
            "uid": self.uid,
            "label": self.label,
            "name": self.name,
            "usd_path": self.usd_path,
            "mjcf_path": self.mjcf_path,
            "description": self.description,
            "articulate_init_joint_qpos": self.articulate_init_joint_qpos,
        }


@AssetManager.register("articulated")
class ArticulatedAssetManager(AssetManager):
    src_dir: str = "objects/articulated"

    def __init__(self, src_dir: str | None = None) -> None:
        if src_dir is not None:
            self.src_dir = src_dir
        self.src_dir = "objects/articulated"

    def load(
        self, asset_id: str, articulate_init_joint_qpos: dict[str, float] | None = None
    ) -> ArticulatedAsset:
        """Load an asset by its ID."""
        if not asset_id.isdigit():
            return self._load_library_object(asset_id, articulate_init_joint_qpos)

        assert int(asset_id) in Articulated_Object_Names, f"Invalid asset ID: {asset_id}"

        asset_id_int = int(asset_id)

        mjcf_path = resolve_data_path(
            os.path.join(
                self.src_dir, f"{asset_id_int:03d}", "output_mjcf", f"{asset_id_int:03d}.xml"
            ),
            auto_download=True,
        )

        # TODO: Load usd_path
        usd_path = resolve_data_path(
            os.path.join(
                self.src_dir, f"{asset_id_int:03d}", "output_usd", f"{asset_id_int:03d}.usd"
            ),
            auto_download=True,
        )

        name = Articulated_Object_Names[asset_id_int]
        if articulate_init_joint_qpos is None:
            articulate_init_joint_qpos = Default_Articulated_Init_Joint_Qpos.get(asset_id_int)
        return ArticulatedObjectAsset(
            uid=asset_id,
            label="_".join(name.split(" ")).replace("-", "_"),
            name=name,
            usd_path=usd_path,
            mjcf_path=mjcf_path,
            articulate_init_joint_qpos=articulate_init_joint_qpos,
        )

    def _load_library_object(
        self,
        asset_id: str,
        articulate_init_joint_qpos: dict[str, float] | None,
    ) -> ArticulatedAsset:
        spec = _LIBRARY_ARTICULATED_OBJECTS.get(asset_id)
        if spec is None:
            choices = ", ".join(sorted(_LIBRARY_ARTICULATED_OBJECTS))
            raise ValueError(
                f"Invalid articulated asset ID {asset_id!r}; named choices: {choices}."
            )

        asset_dir = Path(resolve_data_path(f"{self.src_dir}/{asset_id}", auto_download=True))
        mjcf_path = asset_dir / "output_mjcf" / spec["mjcf"]
        usd_path = asset_dir / "output_usd" / spec["usd"]
        if not mjcf_path.exists():
            raise FileNotFoundError(mjcf_path)
        if not usd_path.exists():
            raise FileNotFoundError(usd_path)
        return ArticulatedObjectAsset(
            uid=asset_id,
            label=asset_id,
            name=spec["name"],
            usd_path=str(usd_path),
            mjcf_path=str(mjcf_path),
            articulate_init_joint_qpos=articulate_init_joint_qpos,
        )

    def sample(self, exclude: list[str] | None = None) -> ArticulatedAsset:
        import random

        all_ids = [str(i) for i in Articulated_Object_Names.keys()]
        all_ids.extend(_LIBRARY_ARTICULATED_OBJECTS)

        if exclude is not None:
            all_ids = [id for id in all_ids if id not in exclude]

        asset_id = random.choice(all_ids)
        return self.load(asset_id)

    def __len__(self) -> int:
        return len(Articulated_Object_Names) + len(_LIBRARY_ARTICULATED_OBJECTS)
