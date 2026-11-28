"""Report missing Advanced scene BEV and per-episode scene bindings."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from humano.scenes.bev_cache import (
    cache_manifest_id,
    load_trajectory_points,
    resolve_project_path,
    scene_sidecars,
    training_episode_indices,
)


def _episode_records(dataset: Path) -> list[dict]:
    path = dataset / "meta" / "episodes.jsonl"
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    indices = [int(record["episode_index"]) for record in records]
    if not indices or len(indices) != len(set(indices)):
        raise ValueError(f"Dataset has no unique episode indices: {path}")
    return records


def _training_binding(path: Path, indices: set[int], manifest_id: str, index_hash: str) -> dict:
    if not path.is_file():
        return {"status": "missing", "path": str(path), "missing_episodes": sorted(indices)}
    data = json.loads(path.read_text(encoding="utf-8"))
    if (
        data.get("schema") != "humano.training-scene-assignments.v1"
        or data.get("bev_cache_manifest_id") != manifest_id
        or data.get("scene_index_hash") != index_hash
    ):
        return {
            "status": "stale",
            "path": str(path),
            "reason": "schema or level-one identity differs",
        }
    candidate_path = resolve_project_path(data.get("candidate_manifest") or "")
    if not candidate_path.is_file():
        return {
            "status": "incomplete",
            "path": str(path),
            "reason": f"Candidate manifest is missing: {candidate_path}",
        }
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    if candidate.get("bev_cache", {}).get("manifest_id") != manifest_id:
        return {
            "status": "stale",
            "path": str(path),
            "reason": "Candidate manifest uses a different level-one cache",
        }
    episodes = data.get("episodes", {})
    missing = sorted(
        index for index in indices if not _valid_training_placements(episodes.get(str(index)))
    )
    extra = sorted(set(episodes) - {str(index) for index in indices})
    if missing or extra:
        return {
            "status": "incomplete",
            "path": str(path),
            "missing_episodes": missing,
            "extra_episodes": extra,
        }
    return {"status": "ready", "path": str(path), "episodes": len(indices)}


def _valid_training_placements(placements: object) -> bool:
    return (
        isinstance(placements, list)
        and bool(placements)
        and all(
            isinstance(item, dict)
            and item.get("scene_uid")
            and isinstance(item.get("translation_xy"), list)
            and len(item["translation_xy"]) == 2
            and "yaw_rad" in item
            and isinstance(item.get("auto_hidden_prim_paths"), list)
            for item in placements
        )
    )


def _eval_binding(
    dataset: Path,
    manifest_id: str,
    index_hash: str,
    binding_root: Path | None,
    suite: str,
    task: str,
) -> dict:
    if not (dataset / "meta" / "episodes.jsonl").is_file():
        return {"status": "missing", "path": str(dataset)}
    records = _episode_records(dataset)
    missing = []
    embedded = {}
    for record in records:
        config = record.get("environment_config")
        if isinstance(config, str):
            config = json.loads(config)
        scene = config.get("eval_scene") if isinstance(config, dict) else None
        scene_uid = (
            scene.get("scene_uid") or (scene.get("scene") or {}).get("uid")
            if isinstance(scene, dict)
            else None
        )
        index_matches = (
            scene.get("scene_index_hash") == index_hash
            if isinstance(scene, dict) and "scene_index_hash" in scene
            else True
        )
        if (
            not isinstance(scene, dict)
            or scene.get("hidden_prim_paths_final") is not True
            or not index_matches
            or not scene_uid
            or not isinstance(scene.get("placement"), dict)
            or not isinstance(scene.get("hidden_prim_paths"), list)
        ):
            missing.append(int(record["episode_index"]))
        else:
            embedded[str(int(record["episode_index"]))] = scene
    if missing:
        return {"status": "incomplete", "path": str(dataset), "missing_episodes": missing}
    if binding_root is None or not (binding_root / "manifest.json").is_file():
        return {
            "status": "missing",
            "path": str(dataset),
            "reason": "Eval level-two binding manifest is missing",
        }
    binding = json.loads((binding_root / "manifest.json").read_text(encoding="utf-8"))
    if (
        binding.get("schema") != "humano.final-eval-scene-bindings.v1"
        or binding.get("cache_level") != 2
        or cache_manifest_id(resolve_project_path(binding["level1_cache_root"])) != manifest_id
        or (
            binding.get("level1_manifest_id") is not None
            and binding["level1_manifest_id"] != manifest_id
        )
        or resolve_project_path(binding["level2_cache_root"]).resolve() != binding_root.resolve()
    ):
        return {
            "status": "stale",
            "path": str(dataset),
            "reason": "Eval binding uses a different level-one cache",
        }
    placements = {
        item["binding_id"]: item
        for item in json.loads((binding_root / "placements.json").read_text(encoding="utf-8"))
    }
    condition = dataset.parent.parent.name if dataset.parent.name == suite else dataset.parent.name
    episode_bindings = {
        str(item["episode_index"]): item["binding_id"]
        for line in (binding_root / "episodes.jsonl").read_text(encoding="utf-8").splitlines()
        if (item := json.loads(line))["condition"] == condition and item["task"] == task
    }
    mismatched = []
    for index, scene in embedded.items():
        placement = placements.get(episode_bindings.get(index))
        if (
            placement is None
            or placement["scene_uid"] != scene.get("scene", {}).get("uid")
            or placement["scene_revision"] != scene.get("scene", {}).get("revision")
            or placement["hidden_prim_paths"] != scene["hidden_prim_paths"]
            or any(
                placement["placement"].get(key) != scene["placement"].get(key)
                for key in ("scene_uid", "translation_xy", "yaw_rad")
            )
        ):
            mismatched.append(int(index))
    if mismatched or set(episode_bindings) != set(embedded):
        return {"status": "stale", "path": str(dataset), "mismatched_episodes": mismatched}
    return {
        "status": "ready",
        "path": str(dataset),
        "binding_manifest": str(binding_root / "manifest.json"),
        "episodes": len(records),
    }


def check_levels(
    cache_root: Path,
    scene_index: Path,
    suite: str,
    task: str,
    train_dataset: Path,
    assignment: Path | None = None,
    eval_datasets: tuple[Path, ...] = (),
    eval_binding_root: Path | None = None,
) -> dict:
    records = _episode_records(train_dataset)
    indices = {int(record["episode_index"]) for record in records}
    try:
        manifest_id = cache_manifest_id(cache_root)
        scenes = scene_sidecars(cache_root, scene_index)
        if not scenes:
            raise ValueError("Scene BEV manifest contains no scene")
        if training_episode_indices(cache_root, suite, task) != indices:
            raise ValueError("Training episodes differ from the level-one trajectory manifest")
        load_trajectory_points(cache_root, suite, task)
        for index in sorted(indices):
            load_trajectory_points(cache_root, suite, task, episode_index=index)
    except (OSError, ValueError, KeyError) as exc:
        return {
            "level_one": {"status": "missing_or_invalid", "reason": str(exc)},
            "level_two": {"status": "waiting_for_level_one"},
        }
    index_hash = hashlib.sha256(scene_index.read_bytes()).hexdigest()
    training = _training_binding(
        assignment or cache_root / "bindings" / "train" / suite / task / "assignment.json",
        indices,
        manifest_id,
        index_hash,
    )
    evaluation = [
        _eval_binding(dataset, manifest_id, index_hash, eval_binding_root, suite, task)
        for dataset in eval_datasets
    ]
    return {
        "level_one": {
            "status": "ready",
            "manifest_id": manifest_id,
            "scenes": len(scenes),
            "training_episodes": len(indices),
        },
        "level_two": {"training": training, "evaluation": evaluation},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bev-cache-root", type=Path, default=Path("data/cache"))
    parser.add_argument("--scene-index", type=Path, required=True)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--train-dataset", type=Path, required=True)
    parser.add_argument("--training-assignment", type=Path)
    parser.add_argument("--eval-dataset", type=Path, action="append", default=[])
    parser.add_argument("--eval-binding-root", type=Path)
    args = parser.parse_args()
    report = check_levels(
        args.bev_cache_root,
        args.scene_index,
        args.suite,
        args.task,
        args.train_dataset,
        args.training_assignment,
        tuple(args.eval_dataset),
        args.eval_binding_root,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    level_two = report["level_two"]
    return int(
        report["level_one"]["status"] != "ready"
        or level_two["training"]["status"] != "ready"
        or any(item["status"] != "ready" for item in level_two["evaluation"])
    )


if __name__ == "__main__":
    raise SystemExit(main())
