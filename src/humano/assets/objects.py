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
import trimesh

from humano.core.asset import Asset
from humano.core.object import SemanticAnnotated, SpatialAnnotated
from humano.core.types import GraspPose, Pose
from humano.utils import resolve_data_path

from .asset_manager import AssetManager

Objects_Names = {
    0: "brasket",
}

_FIXED_OBJECT_PHYSICS = {
    "dustbin/dustbin_011": {"mujoco_mass": 10.0, "mujoco_freejoint": False},
    "shelf/shelf_001": {"mujoco_mass": 30.0, "mujoco_freejoint": False},
}

_SCALED_LIBRARY_OBJECTS = {
    "storage_box/storage_box_003_object_pick_1p3": ("storage_box/storage_box_003", 1.3),
}


def _resolve_visual_usd_path(usd_dir: str | Path, asset_name: str) -> Path:
    usd_dir = Path(usd_dir)
    for filename in (f"{asset_name}.extracted_visual.usd", f"{asset_name}.usd"):
        visual_usd = usd_dir / filename
        if visual_usd.exists():
            return visual_usd
    raise FileNotFoundError(f"No visual USD found for objects:{asset_name}")


class ObjectsAsset(Asset, SemanticAnnotated, SpatialAnnotated):
    def __init__(
        self,
        uid: str,
        label: str,
        name: str,
        usd_path: str,
        collision_mesh_curobo: str,
        collision_meshes_mujoco: list[str],
        description: str | None = None,
        stable_poses: dict[str, Pose] | None = None,
        scale: float = 1.0,
    ) -> None:
        super().__init__(
            uid=uid,
            usd_path=usd_path,
            collision_mesh_curobo=collision_mesh_curobo,
            collision_meshes_mujoco=collision_meshes_mujoco,
        )

        # SpatialAnnotated.__init__(self, uid=uid, name=name, usd_path=usd_path, collision_mesh_curobo=collision_mesh_curobo, collision_meshes_mujoco=collision_meshes_mujoco)

        self.name = name
        self.label = label
        self.description = description
        self.stable_poses = stable_poses
        self.scale = float(scale)
        for name, value in _FIXED_OBJECT_PHYSICS.get(uid, {}).items():
            setattr(self, name, value)

    def __repr__(self) -> str:
        return f"Objects1BAsset(uid={self.uid}, name={self.name})"

    def to_dict(self) -> dict[str, Any]:
        return {
            "res_id": "objects",
            "uid": self.uid,
            "label": self.label,
            "name": self.name,
            "usd_path": self.usd_path,
            "scale": self.scale,
            # "collision_mesh_curobo": self.collision_mesh_curobo,
            # "collision_meshes_mujoco": self.collision_meshes_mujoco,
            "description": self.description,
        }


@AssetManager.register("objects")
class ObjectsAssetManager(AssetManager):
    src_dir: str = "objects/rigid"
    _legacy_dirs = {"collision_models_mujoco", "models", "ruled_models", "stable"}

    def __init__(self, src_dir: str | None = None) -> None:
        if src_dir is not None:
            self.src_dir = src_dir

        self.src_dir = "objects/rigid"
        # self._load_graspnet_object_assets()

    def load(self, asset_id: str) -> Asset:
        """Load an asset by its ID."""
        if not asset_id.isdigit():
            return self._load_library_object(asset_id)

        return self._load_legacy_object(asset_id)

    def _load_legacy_object(self, asset_id: str) -> Asset:
        assert int(asset_id) in Objects_Names, f"Asset ID {asset_id} not found in GraspNet 1B."

        asset_id_int = int(asset_id)
        collision_mesh_dir = resolve_data_path(
            f"{self.src_dir}/collision_models_mujoco/{asset_id_int:03d}/",
            auto_download=True,
        )

        collision_mesh_curobo = os.path.join(collision_mesh_dir, f"{asset_id_int:03d}_32x8.obj")
        collision_mesh_new_dir = resolve_data_path(
            f"{self.src_dir}/models/{asset_id_int:03d}/coacd_0.05/",
            auto_download=True,
        )
        collision_meshes_mujoco = []
        for dir, _dirs, files in os.walk(collision_mesh_new_dir):
            collision_meshes_mujoco = [
                dir + "/" + file for file in files if file.startswith("convex_piece")
            ]

        stable_poses = np.array([[4.99879344e-01, -3.56605453e-05, 0.012, 0.7071, 0.7071, 0, 0]])

        name = Objects_Names[asset_id_int]
        return ObjectsAsset(
            uid=asset_id,
            label="_".join(name.split(" ")).replace("-", "_"),
            name=Objects_Names[asset_id_int],
            usd_path=resolve_data_path(
                f"{self.src_dir}/ruled_models/{asset_id_int:03d}_ruled.usd",
                auto_download=True,
            ),
            collision_mesh_curobo=collision_mesh_curobo,
            collision_meshes_mujoco=collision_meshes_mujoco,
            description="todo",
            stable_poses=stable_poses,
        )

    def _iter_library_asset_ids(self) -> list[str]:
        root = Path(resolve_data_path(self.src_dir, auto_download=True))
        asset_ids: list[str] = []
        for category_dir in sorted(root.iterdir()):
            if not category_dir.is_dir() or category_dir.name in self._legacy_dirs:
                continue
            for asset_dir in sorted(category_dir.iterdir()):
                if not asset_dir.is_dir():
                    continue
                asset_name = asset_dir.name
                xml = asset_dir / "original" / "output_mjcf" / f"{asset_name}_usd_collision.xml"
                usd = asset_dir / "original" / "output_usd" / f"{asset_name}.extracted_visual.usd"
                if xml.exists() and usd.exists():
                    asset_ids.append(f"{category_dir.name}/{asset_name}")
        return asset_ids

    def _resolve_library_asset_id(self, asset_id: str) -> tuple[str, str]:
        if "/" in asset_id:
            category, asset_name = asset_id.split("/", 1)
            return category, asset_name

        matches = [
            candidate
            for candidate in self._iter_library_asset_ids()
            if candidate.rsplit("/", 1)[-1] == asset_id
        ]
        if len(matches) == 1:
            return matches[0].split("/", 1)
        if len(matches) > 1:
            raise ValueError(
                f"Ambiguous objects asset ID '{asset_id}'. Use category/name, e.g. objects:{matches[0]}."
            )
        raise ValueError(f"Asset ID {asset_id} not found in objects library.")

    def _load_library_object(self, asset_id: str) -> Asset:
        scaled_spec = _SCALED_LIBRARY_OBJECTS.get(asset_id)
        if scaled_spec is not None:
            source_asset_id, scale = scaled_spec
            source = self._load_library_object(source_asset_id)
            assert isinstance(source, ObjectsAsset)
            stable_poses = np.asarray(source.stable_poses, dtype=np.float32).copy()
            stable_poses[:, :3] *= scale
            return ObjectsAsset(
                uid=asset_id,
                label=asset_id.rsplit("/", 1)[-1],
                name=f"{source.name} object pick {scale:g}x",
                usd_path=source.usd_path,
                collision_mesh_curobo=source.collision_mesh_curobo,
                collision_meshes_mujoco=list(source.collision_meshes_mujoco),
                description=f"O-5/O-6 {scale:g}x variant of storage_box_003",
                stable_poses=stable_poses,
                scale=scale,
            )

        category, asset_name = self._resolve_library_asset_id(asset_id)
        asset_rel = f"{self.src_dir}/{category}/{asset_name}/original"
        mjcf_dir = resolve_data_path(f"{asset_rel}/output_mjcf", auto_download=True)
        usd_dir = resolve_data_path(f"{asset_rel}/output_usd", auto_download=True)

        collision_dir = Path(mjcf_dir) / "meshes" / "usd_collision"
        collision_meshes_mujoco = sorted(str(p) for p in collision_dir.glob("*.obj"))
        if not collision_meshes_mujoco:
            meshes_dir = Path(mjcf_dir) / "meshes"
            collision_meshes_mujoco = sorted(str(p) for p in meshes_dir.rglob("*.obj"))
        if not collision_meshes_mujoco:
            raise FileNotFoundError(f"No MuJoCo collision OBJ found for objects:{asset_id}")

        visual_usd = _resolve_visual_usd_path(usd_dir, asset_name)

        stable_rotation = self._stable_rotation_for_category(category)
        stable_z = self._estimate_stable_z(collision_meshes_mujoco, rotation=stable_rotation)
        stable_quaternion = self._stable_quaternion_for_category(category)
        stable_poses = np.array([[0.0, 0.0, stable_z, *stable_quaternion]], dtype=np.float32)
        label = asset_name.replace("-", "_")
        return ObjectsAsset(
            uid=f"{category}/{asset_name}",
            label=label,
            name=asset_name.replace("_", " "),
            usd_path=str(visual_usd),
            collision_mesh_curobo=collision_meshes_mujoco[0],
            collision_meshes_mujoco=collision_meshes_mujoco,
            description="imported from Psi0 test_asset/assets_simplified_x10",
            stable_poses=stable_poses,
        )

    def _stable_rotation_for_category(self, category: str) -> np.ndarray | None:
        if category == "plate":
            return np.array(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 0.0, -1.0],
                    [0.0, 1.0, 0.0],
                ],
                dtype=np.float32,
            )
        return None

    def _stable_quaternion_for_category(self, category: str) -> tuple[float, float, float, float]:
        if category == "plate":
            s = float(np.sqrt(0.5))
            return (s, s, 0.0, 0.0)
        return (1.0, 0.0, 0.0, 0.0)

    def _estimate_stable_z(
        self, mesh_paths: list[str], rotation: np.ndarray | None = None
    ) -> float:
        min_z: float | None = None
        for mesh_path in mesh_paths:
            mesh = trimesh.load_mesh(mesh_path, force="mesh")
            if mesh.is_empty:
                continue
            vertices = np.asarray(mesh.vertices)
            if rotation is not None:
                vertices = vertices @ rotation.T
            mesh_min_z = float(vertices[:, 2].min())
            min_z = mesh_min_z if min_z is None else min(min_z, mesh_min_z)
        if min_z is None:
            return 0.0
        return max(-min_z, 0.0)

    def sample(self, exclude: list[str] | None = None) -> Asset:
        import random

        all_ids = [str(i) for i in Objects_Names.keys()] + self._iter_library_asset_ids()

        if exclude:
            exclude = [str(e) for e in exclude]
            candidates = [aid for aid in all_ids if aid not in exclude]

        else:
            candidates = all_ids
        sampled_id = random.choice(candidates)
        return self.load(sampled_id)

    def __len__(self) -> int:
        return len(Objects_Names) + len(self._iter_library_asset_ids())

    def _load_objects_assets(self):
        """
        Returns:
            dict: {model_id: {name, usd_path, collision_mesh_curobo, collision_meshes_mujoco}}
        """
        # FIXME need download the graspnet assets first
        assets_info = dict()
        for model_id in Objects_Names.keys():
            model_usd = resolve_data_path(f"{self.src_dir}/ruled_models/{model_id:03d}_ruled.usd")
            collision_mesh_dir = resolve_data_path(
                f"{self.src_dir}/collision_models_mujoco/{model_id:03d}/"
            )
            collision_mesh_curobo = os.path.join(collision_mesh_dir, f"{model_id:03d}_32x8.obj")

            collision_mesh_new_dir = resolve_data_path(
                f"{self.src_dir}/models/{model_id:03d}/coacd_0.05/",
                auto_download=True,
            )
            for dir, _dirs, files in os.walk(collision_mesh_new_dir):
                collision_meshes_mujoco = [
                    dir + "/" + file for file in files if file.startswith("convex_piece")
                ]

            assets_info[model_id] = {
                "name": Objects_Names[model_id].replace(" ", "_").replace("-", "_"),
                "usd_path": model_usd,
                "collision_mesh_curobo": collision_mesh_curobo,
                "collision_meshes_mujoco": collision_meshes_mujoco,
            }
        return assets_info
