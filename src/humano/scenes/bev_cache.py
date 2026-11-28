"""Read the Advanced scene and training-trajectory BEV cache in data/cache."""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def resolve_project_path(value: str | Path) -> Path:
    """Resolve serialized project paths independently of the caller's cwd."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def project_path(value: str | Path) -> str:
    """Expose physical dataset locations through the project's data entries."""
    path = resolve_project_path(value)
    if path.is_relative_to(PROJECT_ROOT):
        return path.relative_to(PROJECT_ROOT).as_posix()
    physical = path.resolve()
    entries = sorted((PROJECT_ROOT / "data").iterdir())
    aliases = [
        child
        for entry in entries
        if entry.is_symlink() and entry.is_dir()
        for child in entry.iterdir()
        if child.is_dir()
    ]
    task_aliases = [
        child
        for entry in aliases
        for child in entry.iterdir()
        if child.is_symlink() and child.is_dir()
    ]
    for entry in entries + aliases + task_aliases:
        if entry.is_dir() and physical.is_relative_to(entry.resolve()):
            return (
                entry.relative_to(PROJECT_ROOT) / physical.relative_to(entry.resolve())
            ).as_posix()
    raise ValueError(f"Path is not exposed through project data: {value}")


@lru_cache(maxsize=4)
def _manifests(root: Path) -> tuple[dict, dict]:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != "humano.training-eval-bev-cache.v1":
        raise ValueError(f"Unsupported BEV cache: {root}")
    scenes = json.loads((root / manifest["scene_manifest"]).read_text(encoding="utf-8"))
    trajectories = json.loads((root / manifest["trajectory_manifest"]).read_text(encoding="utf-8"))
    if scenes.get("schema") != "humano.selected-scene-bev-cache.v1":
        raise ValueError(f"Unsupported public scene BEV cache: {root}")
    if trajectories.get("schema") != "humano.advanced-train-trajectory-bev-cache.v1":
        raise ValueError(f"Unsupported public trajectory BEV cache: {root}")
    if (
        float(scenes["cell_m"]) != 0.05
        or float(trajectories["cell_m"]) != 0.05
        or float(trajectories["margin_m"]) != 0.5
        or trajectories["coordinate_frame"] != "mujoco_world_xy"
    ):
        raise ValueError(f"BEV cache grid or trajectory frame differs: {root}")
    return scenes, trajectories


def cache_manifest_id(root: str | Path) -> str:
    """Identify the published level-one cache without reading every NPZ file."""
    root = Path(root).expanduser().resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    _manifests(root)
    digest = hashlib.sha256()
    for name in ("manifest.json", manifest["scene_manifest"], manifest["trajectory_manifest"]):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update((root / name).read_bytes())
    support_name = manifest.get("scene_support_manifest")
    if support_name is not None:
        support_path = root / support_name
        support_bytes = support_path.read_bytes()
        digest.update(str(support_name).encode("utf-8"))
        digest.update(b"\0")
        digest.update(support_bytes)
        support_manifest = json.loads(support_bytes)
        for scene in support_manifest["scenes"]:
            scene_name = str(scene["path"])
            digest.update(scene_name.encode("utf-8"))
            digest.update(b"\0")
            digest.update((support_path.parent / scene_name).read_bytes())
    return digest.hexdigest()


def scene_sidecars(root: str | Path, scene_index: str | Path) -> dict[str, tuple[str, Path]]:
    root = Path(root).expanduser().resolve()
    index = Path(scene_index).expanduser().resolve()
    scenes, _ = _manifests(root)
    if hashlib.sha256(index.read_bytes()).hexdigest() != scenes["scene_index_sha256"]:
        raise ValueError("BEV cache uses a different scene index")
    indexed = {scene["uid"] for scene in json.loads(index.read_text())["scenes"]}
    if indexed != {scene["uid"] for scene in scenes["scenes"]}:
        raise ValueError("Scene catalog and level-one cache coverage differ")
    selected = {}
    for scene in scenes["scenes"]:
        uid = str(scene["uid"])
        if uid in selected:
            raise ValueError(f"Duplicate scene in BEV cache: {uid}")
        path = root / scene["asset_sidecar"]
        if not path.is_file():
            raise FileNotFoundError(path)
        structure = root / scene["structure_sidecar"]
        if not structure.is_file():
            raise FileNotFoundError(structure)
        selected[uid] = str(scene["revision"]), path
    return selected


def scene_support_relations(root: str | Path) -> dict[str, tuple[tuple[str, str], ...]]:
    """Read precomputed parent-to-child support edges for offline final hiding."""
    root = Path(root).expanduser().resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    support_name = manifest.get("scene_support_manifest")
    if support_name is None:
        return {}
    support_path = root / support_name
    supports = json.loads(support_path.read_text(encoding="utf-8"))
    if supports.get("schema") != "humano.scene-asset-support-manifest.v1":
        raise ValueError(f"Unsupported scene support cache: {support_path}")
    scenes, _ = _manifests(root)
    expected = {scene["uid"]: scene["revision"] for scene in scenes["scenes"]}
    result = {}
    for scene in supports["scenes"]:
        uid = scene["uid"]
        if uid in result or expected.get(uid) != scene["revision"]:
            raise ValueError(f"Scene support revision or duplicate differs: {uid}")
        path = support_path.parent / scene["path"]
        data = json.loads(path.read_text(encoding="utf-8"))
        if (
            data.get("schema") != "humano.scene-asset-support.v1"
            or data.get("uid") != uid
            or data.get("revision") != scene["revision"]
        ):
            raise ValueError(f"Scene support identity differs: {path}")
        paths = {asset["prim_path"] for asset in data["assets"]}
        edges = tuple((edge["parent_path"], edge["child_path"]) for edge in data["relations"])
        if any(
            parent == child or parent not in paths or child not in paths for parent, child in edges
        ):
            raise ValueError(f"Scene support refers to an invalid asset: {path}")
        result[uid] = edges
    if set(result) != set(expected):
        raise ValueError(f"Scene support cache misses BEV scenes: {support_path}")
    return result


def load_trajectory_points(
    root: str | Path, suite: str, task: str, *, episode_index: int | None = None
) -> np.ndarray:
    root = Path(root).expanduser().resolve()
    _, trajectories = _manifests(root)
    if suite != trajectories["suite"]:
        raise ValueError(f"BEV cache does not cover suite {suite}")
    matching = [item for item in trajectories["tasks"] if item["task"] == task]
    if len(matching) != 1:
        raise ValueError(f"BEV cache does not cover task {suite}/{task}")
    selected = matching[0]
    episodes = {int(item["episode_index"]): item for item in selected["episodes"]}
    if episode_index is None:
        path = root / selected["union_path"]
    else:
        if episode_index not in episodes:
            raise ValueError(f"BEV cache has no {task} episode {episode_index}")
        path = root / episodes[episode_index]["path"]
    with np.load(path, allow_pickle=False) as data:
        if str(data["suite"]) != suite or str(data["task"]) != task:
            raise ValueError(f"BEV trajectory task mismatch: {path}")
        if str(data["coordinate_frame"]) != trajectories["coordinate_frame"]:
            raise ValueError(f"BEV trajectory coordinate mismatch: {path}")
        if float(data["cell_m"]) != float(trajectories["cell_m"]) or float(
            data["margin_m"]
        ) != float(trajectories["margin_m"]):
            raise ValueError(f"BEV trajectory grid mismatch: {path}")
        if episode_index is None:
            if set(map(int, data["episode_indices"])) != set(episodes):
                raise ValueError(f"BEV task union misses training episodes: {path}")
        elif int(data["episode_index"]) != episode_index:
            raise ValueError(f"BEV trajectory episode mismatch: {path}")
        points = np.asarray(data["occupied_xy"], dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) == 0:
        raise ValueError(f"BEV trajectory has no XY occupancy: {path}")
    return points


def training_episode_indices(root: str | Path, suite: str, task: str) -> set[int]:
    _, trajectories = _manifests(Path(root).expanduser().resolve())
    if suite != trajectories["suite"]:
        raise ValueError(f"BEV cache does not cover suite {suite}")
    matching = [item for item in trajectories["tasks"] if item["task"] == task]
    if len(matching) != 1:
        raise ValueError(f"BEV cache does not cover task {suite}/{task}")
    return {int(item["episode_index"]) for item in matching[0]["episodes"]}
