"""Generate first-frame eval episodes with frozen task-aware scene candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import subprocess
from collections.abc import Iterable
from dataclasses import asdict, replace
from functools import cache
from itertools import combinations
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import humano.envs  # noqa: F401, E402
from humano.datasets.episode_metadata import (  # noqa: E402
    _load_episode_configs,
    _load_lerobot_episodes,
)
from humano.wbc.sonic.config import load_sonic_config  # noqa: E402
from humano.dr.material import (  # noqa: E402
    training_table_material_choices,
    with_task_table_material,
)
from humano.evals.eval_scene_protocol import build_eval_scene_selection  # noqa: E402
from humano.robots.hands import get_hand_spec  # noqa: E402
from humano.scenes.bev_cache import (  # noqa: E402
    cache_manifest_id,
    load_trajectory_points,
    project_path,
    resolve_project_path,
    training_episode_indices,
)
from humano.scenes.floorplan_split import (  # noqa: E402
    MERGED_GLOBAL_FLOORPLAN_SPLIT_SCHEMA,
    floor_plan_room_type,
    load_global_floorplan_split,
    split_floor_plan_sets,
)
from humano.scenes.scene_selection import (  # noqa: E402
    CANDIDATE_MANIFEST_SCHEMA,
    AxisAlignedBounds,
    ScenePlacement,
    SceneRecord,
    TaskLayout,
    _task_intersects_bounds,
    freeze_placement_hidden_paths,
    load_scene_records,
    parse_task_layout,
    rank_scene_placements,
    transform_bounds,
)
from humano.scenes.scene_split_catalog import validate_scene_split_catalog  # noqa: E402
from humano.utils import NumpyArrayEncoder  # noqa: E402

DEFAULT_ENV_ID = "humano/G1WholebodyPushOfficeChairTeleop-v0"
DEFAULT_OUTPUT_ROOT = Path("data/eval_generated")
DEFAULT_MIN_CANDIDATES = 5
DEFAULT_MAX_CANDIDATES = 20
DEFAULT_TRAJECTORY_MARGIN_M = 0.5
DEFAULT_FREE_SPACE_GRID_M = 0.5
ID_OOD_EPISODE_COUNT = 50
SINGLE_DISTRIBUTION_EPISODE_COUNT = 25
ID_OOD_SCENE_POOL_SIZE = 5
GROUPED_CANDIDATE_MANIFEST_SCHEMA = "humano.eval-scene-candidate-groups.v2"
TASK_ROOM_CONFIG_SCHEMA = "humano.task-scene-room-types.v1"
REPO_ROOT = Path(__file__).resolve().parents[2]


def _union_bounds(first: AxisAlignedBounds, second: AxisAlignedBounds) -> AxisAlignedBounds:
    return AxisAlignedBounds(
        (min(first.min_xy[0], second.min_xy[0]), min(first.min_xy[1], second.min_xy[1])),
        (max(first.max_xy[0], second.max_xy[0]), max(first.max_xy[1], second.max_xy[1])),
    )


def aggregate_task_layouts(
    layouts: Iterable[TaskLayout],
    trajectory_bounds: Iterable[AxisAlignedBounds],
    *,
    margin_m: float,
) -> TaskLayout:
    layout_values = tuple(layouts)
    trajectory_values = tuple(trajectory_bounds)
    if not layout_values or len(layout_values) != len(trajectory_values):
        raise ValueError("Layouts and trajectory bounds must be non-empty and have equal length")
    if margin_m < 0.0:
        raise ValueError("Trajectory margin must be non-negative")
    anchor = layout_values[0].anchor
    if any(
        (layout.anchor, layout.secondary_anchor) != (anchor, layout_values[0].secondary_anchor)
        for layout in layout_values[1:]
    ):
        raise ValueError("Demonstrations have different task anchors; build one pool per layout")
    protected = layout_values[0].protected_bounds
    for layout, trajectory in zip(layout_values, trajectory_values, strict=True):
        protected = _union_bounds(protected, layout.protected_bounds)
        protected = _union_bounds(protected, trajectory)
    expanded = AxisAlignedBounds(
        (protected.min_xy[0] - margin_m, protected.min_xy[1] - margin_m),
        (protected.max_xy[0] + margin_m, protected.max_xy[1] + margin_m),
    )
    return replace(
        layout_values[0],
        episode_index=0,
        protected_bounds=expanded,
        static_bounds=tuple(box for layout in layout_values for box in layout.static_bounds),
    )


def _layout_anchor_payload(layout: TaskLayout) -> dict[str, Any] | None:
    anchor = layout.anchor
    if anchor is None:
        return None
    return {
        "center_xy": list(anchor.center_xy),
        "size_xy": list(anchor.size_xy),
        "yaw_rad": anchor.yaw_rad,
    }


def _layout_group_key(layout: TaskLayout) -> str:
    anchors = (
        _layout_anchor_payload(layout),
        asdict(layout.secondary_anchor) if layout.secondary_anchor is not None else None,
    )
    serialized = json.dumps(anchors, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode()).hexdigest()


def _task_parameter_key(suite: str, config: dict[str, Any]) -> str:
    if suite == "spatial":
        spatial = config.get("dr_state_dict", {}).get("spatial") or {}
        value = tuple(
            sorted(
                (name, tuple(round(float(x), 3) for x in pose.get("position", [])[:3]))
                for name, pose in spatial.items()
                if not name.startswith("g1_")
            )
        )
    elif suite == "goal":
        value = config.get("selected_goal")
    elif suite in {"condition", "object"}:
        value = (
            config.get("initial_condition"),
            config.get("dr_state_dict", {}).get("target", {}).get("uid")
            if suite == "object"
            else None,
        )
    else:
        value = None
    return json.dumps(value, sort_keys=True)


def _parameter_group_key(layout: TaskLayout, suite: str, config: dict[str, Any]) -> str:
    value = f"{_layout_group_key(layout)}:{_task_parameter_key(suite, config)}"
    return hashlib.sha256(value.encode()).hexdigest()


def _group_layouts_by_anchor(
    layouts: Iterable[TaskLayout],
    trajectory_bounds: Iterable[AxisAlignedBounds],
    *,
    configs: dict[int, dict[str, Any]] | None = None,
    parameter_suite: str | None = None,
) -> dict[str, tuple[tuple[TaskLayout, ...], tuple[AxisAlignedBounds, ...]]]:
    groups: dict[str, list[tuple[TaskLayout, AxisAlignedBounds]]] = {}
    for layout, trajectory in zip(layouts, trajectory_bounds, strict=True):
        key = (
            _parameter_group_key(layout, parameter_suite, configs[layout.episode_index])
            if parameter_suite is not None and configs is not None
            else _layout_group_key(layout)
        )
        groups.setdefault(key, []).append((layout, trajectory))
    return {
        key: (tuple(layout for layout, _ in values), tuple(trajectory for _, trajectory in values))
        for key, values in sorted(groups.items())
    }


def _candidate_manifest_digest(manifest: dict[str, Any]) -> str:
    payload = {key: value for key, value in manifest.items() if key != "content_hash"}
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode()).hexdigest()


def validate_candidate_manifest(manifest: dict[str, Any]) -> None:
    schema = manifest.get("schema")
    if schema == CANDIDATE_MANIFEST_SCHEMA:
        candidates = manifest.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            raise ValueError("Candidate manifest must contain at least one scene")
    elif schema == GROUPED_CANDIDATE_MANIFEST_SCHEMA:
        groups = manifest.get("groups")
        if not isinstance(groups, list) or not groups:
            raise ValueError("Grouped candidate manifest must contain at least one layout group")
        group_keys = [group.get("layout_group_key") for group in groups if isinstance(group, dict)]
        if len(group_keys) != len(groups) or len(set(group_keys)) != len(groups):
            raise ValueError("Grouped candidate manifest must contain unique layout group keys")
        if any(
            not isinstance(group.get("candidates"), list) or not group["candidates"]
            for group in groups
        ):
            raise ValueError("Each layout group must contain at least one scene")
    else:
        raise ValueError(f"Unsupported candidate manifest schema: {schema!r}")
    candidate_groups = (
        [manifest.get("candidates", [])]
        if schema == CANDIDATE_MANIFEST_SCHEMA
        else [group["candidates"] for group in manifest["groups"]]
    )
    if any(
        candidate.get("immutable_obstacle_collision_free") is not True
        for candidates in candidate_groups
        for candidate in candidates
    ):
        raise ValueError("Candidate manifest contains a scene without a wall-collision check")
    if manifest.get("content_hash") != _candidate_manifest_digest(manifest):
        raise ValueError("Candidate manifest content hash does not match its contents")


def finalize_candidate_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    frozen = {key: value for key, value in manifest.items() if key != "content_hash"}
    frozen["content_hash"] = _candidate_manifest_digest(frozen)
    validate_candidate_manifest(frozen)
    return frozen


def _candidate_group_for_layout(
    manifest: dict[str, Any],
    layout: TaskLayout | None,
    environment_config: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], str | None]:
    if manifest["schema"] == CANDIDATE_MANIFEST_SCHEMA:
        return manifest, None
    if layout is None:
        raise ValueError("Grouped candidate manifest selection requires the eval task layout")
    parameter_suite = manifest.get("parameter_suite")
    if parameter_suite is not None and environment_config is None:
        raise ValueError("Parameter-group selection requires the eval environment config")
    group_key = (
        _parameter_group_key(layout, parameter_suite, environment_config)
        if parameter_suite is not None
        else _layout_group_key(layout)
    )
    matching = [group for group in manifest["groups"] if group["layout_group_key"] == group_key]
    if not matching:
        raise ValueError(
            "Eval task layout does not match any frozen candidate group; rebuild the pool with "
            "a compatible successful demonstration layout"
        )
    return matching[0], group_key


def _floor_plan_id(scene_uid: str) -> str:
    match = re.search(r"FloorPlan\d+", scene_uid)
    if match is None:
        raise ValueError(f"Scene uid has no physical FloorPlan id: {scene_uid}")
    return match.group(0)


def _select_scene_candidate(
    manifest: dict[str, Any],
    *,
    episode_index: int,
    seed: int,
    layout: TaskLayout | None,
    environment_config: dict[str, Any] | None = None,
    distribution: str | None = None,
    training_floor_plans: frozenset[str] | None = None,
    floor_plan: str | None = None,
) -> tuple[dict[str, Any], str | None]:
    validate_candidate_manifest(manifest)
    group, group_key = _candidate_group_for_layout(manifest, layout, environment_config)
    candidates = group["candidates"]
    if floor_plan is not None:
        candidates = [
            candidate
            for candidate in candidates
            if _floor_plan_id(candidate["scene_uid"]) == floor_plan
        ]
        if not candidates:
            raise ValueError(f"Candidate group has no selected FloorPlan: {floor_plan}")
    if distribution is not None:
        if distribution not in {"id", "ood"} or training_floor_plans is None:
            raise ValueError("ID/OOD selection requires a distribution and training FloorPlans")
        candidates = [
            candidate
            for candidate in candidates
            if (_floor_plan_id(candidate["scene_uid"]) in training_floor_plans)
            == (distribution == "id")
        ]
        if not candidates:
            raise ValueError(f"Candidate group has no {distribution.upper()} FloorPlan")
    group_suffix = "" if group_key is None else f":{group_key}"
    distribution_suffix = "" if distribution is None else f":{distribution}"
    floor_plan_suffix = "" if floor_plan is None else f":{floor_plan}"
    digest = hashlib.sha256(
        f"{manifest['content_hash']}{group_suffix}{distribution_suffix}{floor_plan_suffix}:"
        f"{seed}:{episode_index}".encode()
    ).digest()
    index = int.from_bytes(digest[:8], "big") % len(candidates)
    return dict(candidates[index]), group_key


def select_scene_candidate(
    manifest: dict[str, Any],
    *,
    episode_index: int,
    seed: int,
    layout: TaskLayout | None = None,
    distribution: str | None = None,
    training_floor_plans: frozenset[str] | None = None,
    floor_plan: str | None = None,
) -> dict[str, Any]:
    candidate, _group_key = _select_scene_candidate(
        manifest,
        episode_index=episode_index,
        seed=seed,
        layout=layout,
        distribution=distribution,
        training_floor_plans=training_floor_plans,
        floor_plan=floor_plan,
    )
    return candidate


def _manifest_group_floorplan_ranks(
    manifest: dict[str, Any],
) -> dict[str | None, dict[str, int]]:
    groups = manifest.get("groups")
    candidate_groups = groups if isinstance(groups, list) and groups else [manifest]
    group_ranks = {}
    for group in candidate_groups:
        ranks = {}
        for rank, candidate in enumerate(group["candidates"]):
            ranks.setdefault(_floor_plan_id(candidate["scene_uid"]), rank)
        group_key = str(group["layout_group_key"]) if groups is not None else None
        group_ranks[group_key] = ranks
    return group_ranks


def _balanced_floorplan_assignment(
    group_keys: tuple[str | None, ...],
    pool: tuple[str, ...],
    group_ranks: dict[str | None, dict[str, int]],
) -> tuple[str, ...] | None:
    if not pool:
        raise ValueError("FloorPlan pool must not be empty")
    if any(group_key not in group_ranks for group_key in group_keys):
        raise ValueError("Eval episode layout does not match the candidate manifest")
    target_count, extra_count = divmod(len(group_keys), len(pool))
    indexed_groups = tuple(
        sorted(
            enumerate(group_keys),
            key=lambda item: sum(floor_plan in group_ranks[item[1]] for floor_plan in pool),
        )
    )

    @cache
    def assign(position: int, remaining: tuple[int, ...]) -> tuple[int, ...] | None:
        if position == len(indexed_groups):
            return () if not any(remaining) else None
        _episode_index, group_key = indexed_groups[position]
        choices = [
            index
            for index, floor_plan in enumerate(pool)
            if remaining[index] > 0 and floor_plan in group_ranks[group_key]
        ]
        choices.sort(key=lambda index: (-remaining[index], index))
        for index in choices:
            updated = tuple(
                value - 1 if value_index == index else value
                for value_index, value in enumerate(remaining)
            )
            suffix = assign(position + 1, updated)
            if suffix is not None:
                return (index, *suffix)
        return None

    ordered_assignment = None
    for extra_indices in combinations(range(len(pool)), extra_count):
        extra_index_set = frozenset(extra_indices)
        remaining = tuple(target_count + (index in extra_index_set) for index in range(len(pool)))
        ordered_assignment = assign(0, remaining)
        if ordered_assignment is not None:
            break
    if ordered_assignment is None:
        return None
    result = [""] * len(group_keys)
    for (episode_index, _group_key), floor_plan_index in zip(
        indexed_groups, ordered_assignment, strict=True
    ):
        result[episode_index] = pool[floor_plan_index]
    return tuple(result)


def _select_floorplan_pool(
    group_ranks: dict[str | None, dict[str, int]],
    allowed_floor_plans: frozenset[str],
    *,
    label: str,
    seed: int,
    pool_size: int,
    group_keys: tuple[str | None, ...] | None = None,
) -> tuple[str, ...]:
    required_group_keys = (
        tuple(group_ranks) if group_keys is None else tuple(dict.fromkeys(group_keys))
    )
    eligible = sorted(
        {
            floor_plan
            for ranks in group_ranks.values()
            for floor_plan in ranks
            if floor_plan in allowed_floor_plans
        }
    )
    eligible.sort(
        key=lambda floor_plan: (
            -sum(floor_plan in group_ranks[group_key] for group_key in required_group_keys),
            sum(
                group_ranks[group_key].get(floor_plan, len(group_ranks[group_key]) + 1)
                for group_key in required_group_keys
            ),
            hashlib.sha256(f"{seed}:{label}:{floor_plan}".encode()).hexdigest(),
        )
    )
    if len(eligible) < pool_size:
        raise ValueError(
            f"Task candidate manifest has only {len(eligible)} usable {label} FloorPlans; "
            f"requires {pool_size}"
        )
    if group_keys is None:
        pool = tuple(eligible[:pool_size])
        if any(not set(pool) & set(group_ranks[group_key]) for group_key in required_group_keys):
            raise ValueError(f"Selected {label} FloorPlan pool does not cover every layout group")
        return pool
    shortlist = eligible[: min(20, len(eligible))]
    for values in combinations(shortlist, pool_size):
        pool = tuple(values)
        if _balanced_floorplan_assignment(group_keys, pool, group_ranks) is not None:
            return pool
    shortlist_set = frozenset(shortlist)
    for values in combinations(eligible, pool_size):
        if set(values) <= shortlist_set:
            continue
        pool = tuple(values)
        if _balanced_floorplan_assignment(group_keys, pool, group_ranks) is not None:
            return pool
    raise ValueError(
        f"No {pool_size}-scene {label} pool can assign episodes evenly "
        "for the generated layout groups"
    )


def _build_distribution_floorplan_pools(
    manifest: dict[str, Any],
    *,
    training_floor_plans: frozenset[str],
    global_train_floor_plans: frozenset[str],
    global_eval_floor_plans: frozenset[str],
    seed: int,
    pool_size: int = ID_OOD_SCENE_POOL_SIZE,
    episode_group_keys: dict[str, tuple[str | None, ...]] | None = None,
    distributions: tuple[str, ...] = ("id", "scene_ood"),
) -> dict[str, tuple[str, ...]]:
    group_ranks = _manifest_group_floorplan_ranks(manifest)
    allowed_floor_plans = {
        "id": training_floor_plans & global_train_floor_plans,
        "scene_ood": global_eval_floor_plans,
    }
    labels = {"id": "ID", "scene_ood": "scene-OOD"}
    return {
        distribution: _select_floorplan_pool(
            group_ranks,
            allowed_floor_plans[distribution],
            label=labels[distribution],
            seed=seed,
            pool_size=pool_size,
            group_keys=(episode_group_keys or {}).get(distribution),
        )
        for distribution in distributions
    }


def _build_episode_floorplan_schedule(
    manifest: dict[str, Any],
    episode_group_keys: tuple[str | None, ...],
    pools: dict[str, tuple[str, ...]],
    episode_distributions: tuple[str, ...] | None = None,
) -> dict[int, str]:
    distributions = episode_distributions or tuple(
        _episode_distribution(index) for index in range(len(episode_group_keys))
    )
    if len(distributions) != len(episode_group_keys):
        raise ValueError("Episode distributions and layout groups must have equal length")
    group_ranks = _manifest_group_floorplan_ranks(manifest)
    schedule = {}
    for distribution in dict.fromkeys(distributions):
        indices = tuple(
            index
            for index, episode_distribution in enumerate(distributions)
            if episode_distribution == distribution
        )
        group_keys = tuple(episode_group_keys[index] for index in indices)
        assignments = _balanced_floorplan_assignment(group_keys, pools[distribution], group_ranks)
        if assignments is None:
            raise ValueError(
                f"Cannot balance {distribution} episodes across {len(pools[distribution])} "
                "FloorPlans"
            )
        schedule.update(dict(zip(indices, assignments, strict=True)))
    return schedule


def _episode_distribution(episode_index: int) -> str:
    if not 0 <= episode_index < ID_OOD_EPISODE_COUNT:
        raise ValueError(f"ID/OOD episode index must be in [0, {ID_OOD_EPISODE_COUNT})")
    return "id" if episode_index < ID_OOD_EPISODE_COUNT // 2 else "scene_ood"


def _episode_floor_plan(episode_index: int, pools: dict[str, tuple[str, ...]]) -> str:
    distribution = _episode_distribution(episode_index)
    distribution_index = (
        episode_index if distribution == "id" else episode_index - ID_OOD_EPISODE_COUNT // 2
    )
    pool = pools[distribution]
    if len(pool) != ID_OOD_SCENE_POOL_SIZE:
        raise ValueError(f"{distribution} FloorPlan pool must contain exactly 5 scenes")
    return pool[distribution_index % len(pool)]


def _normalize_env_id(value: str) -> str:
    return value if value.startswith("humano/") else f"humano/{value}"


def _task_name(env_id: str) -> str:
    return env_id.split("/", maxsplit=1)[-1].removesuffix("-v0")


def _load_task_room_policy(path: Path | None, env_id: str) -> dict[str, Any] | None:
    if path is None:
        return None
    expanded = path.expanduser().resolve()
    content = json.loads(expanded.read_text(encoding="utf-8"))
    if content.get("schema") != TASK_ROOM_CONFIG_SCHEMA:
        raise ValueError(f"Unsupported task room config schema: {content.get('schema')!r}")
    task_name = _task_name(env_id)
    policy = content.get("tasks", {}).get(task_name)
    if not isinstance(policy, dict):
        raise ValueError(f"Task room config has no policy for {task_name}")
    primary = str(policy.get("primary", ""))
    fallback = tuple(str(value) for value in policy.get("fallback", ()))
    allowed = (primary, *fallback)
    valid_room_types = {"bathroom", "bedroom", "kitchen", "living_room"}
    if not primary or len(set(allowed)) != len(allowed) or not set(allowed) <= valid_room_types:
        raise ValueError(f"Invalid room type policy for {task_name}: {policy}")
    excluded = {str(value) for value in content.get("excluded_room_types", ())}
    if set(allowed) & excluded:
        raise ValueError(f"Task room policy for {task_name} uses excluded room types")
    return {
        "config_path": project_path(expanded),
        "config_sha256": hashlib.sha256(expanded.read_bytes()).hexdigest(),
        "task": task_name,
        "suite": str(policy.get("suite", "")),
        "primary": primary,
        "fallback": list(fallback),
        "allowed_room_types": list(allowed),
        "reason": str(policy.get("reason", "")),
    }


def _flat_action_dim(space: gym.Space) -> int:
    if isinstance(space, gym.spaces.Dict):
        return sum(_flat_action_dim(subspace) for subspace in space.spaces.values())
    if isinstance(space, gym.spaces.Box) and space.shape is not None:
        return int(np.prod(space.shape))
    raise ValueError(f"Unsupported action space: {space}")


def _build_features(task: Any) -> dict[str, dict[str, Any]]:
    action_space = task.action_space
    action_names = (
        list(action_space.spaces) if isinstance(action_space, gym.spaces.Dict) else ["action"]
    )
    features: dict[str, dict[str, Any]] = {
        "action": {
            "dtype": "float32",
            "shape": (_flat_action_dim(action_space),),
            "names": action_names,
        }
    }
    for key, space in task.observation_space.items():
        if len(space.shape) == 1:
            features[f"observation.{key}"] = {
                "dtype": space.dtype.name,
                "shape": space.shape,
                "names": [key],
            }
    return features


def _build_frame(task: Any, observation: dict[str, Any]) -> dict[str, np.ndarray]:
    frame = {"action": np.zeros(_flat_action_dim(task.action_space), dtype=np.float32)}
    for key, space in task.observation_space.items():
        if len(space.shape) == 1:
            frame[f"observation.{key}"] = np.asarray(observation[key])
    return frame


def _write_environment_config(
    dataset_root: Path, *, episode_index: int, environment_config: dict[str, Any]
) -> None:
    metadata_path = dataset_root / "meta" / "episodes.jsonl"
    episodes = tuple(
        json.loads(line) for line in metadata_path.read_text(encoding="utf-8").splitlines() if line
    )
    if episode_index >= len(episodes):
        raise ValueError(f"Episode metadata missing index {episode_index}")
    serialized = json.dumps(environment_config, cls=NumpyArrayEncoder)
    updated = tuple(
        {**episode, "environment_config": serialized} if index == episode_index else episode
        for index, episode in enumerate(episodes)
    )
    metadata_path.write_text(
        "".join(json.dumps(episode) + "\n" for episode in updated), encoding="utf-8"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("env_id", nargs="?", default=DEFAULT_ENV_ID)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--num-episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dr-level", type=int, default=0)
    parser.add_argument("--hand-type", choices=("dex3", "inspire", "brainco"), default="dex3")
    parser.add_argument("--candidate-manifest", type=Path)
    parser.add_argument(
        "--training-render-manifest",
        type=Path,
        help="Successful training render records used to define ID versus OOD FloorPlans",
    )
    parser.add_argument(
        "--floorplan-split",
        type=Path,
        help="Frozen global FloorPlan split required for ID/scene-OOD generation",
    )
    parser.add_argument(
        "--floorplan-split-view",
        help="Named view required by the merged v3 FloorPlan split",
    )
    parser.add_argument(
        "--floorplan-pool-size",
        type=int,
        default=ID_OOD_SCENE_POOL_SIZE,
        help="Number of FloorPlans used for each requested eval distribution",
    )
    parser.add_argument(
        "--eval-distribution",
        choices=("id", "scene_ood"),
        help="Generate only 25 episodes from one eval scene distribution",
    )
    parser.add_argument(
        "--scene-split-catalog",
        type=Path,
        help="Combined scene catalog used to verify the active split and formal scene index",
    )
    parser.add_argument("--build-candidates-from", type=Path)
    parser.add_argument("--scene-index", type=Path)
    parser.add_argument("--layout-key")
    parser.add_argument(
        "--task-room-config",
        type=Path,
        help="Task-to-room policy used to filter and prioritize candidate scenes",
    )
    parser.add_argument(
        "--bev-cache-root",
        type=Path,
        help="Use data/cache scene masks and Advanced training trajectory BEV",
    )
    parser.add_argument("--min-candidates", type=int, default=DEFAULT_MIN_CANDIDATES)
    parser.add_argument("--max-candidates", type=int, default=DEFAULT_MAX_CANDIDATES)
    parser.add_argument("--free-space-grid-m", type=float, default=DEFAULT_FREE_SPACE_GRID_M)
    parser.add_argument(
        "--build-candidates-only",
        action="store_true",
        help="Build and freeze a candidate manifest without writing eval episodes",
    )
    return parser.parse_args()


def _validate_id_ood_episode_count(num_episodes: int) -> None:
    if num_episodes != ID_OOD_EPISODE_COUNT:
        raise ValueError(f"ID/scene-OOD eval requires exactly {ID_OOD_EPISODE_COUNT} episodes")


def _validate_single_distribution_episode_count(num_episodes: int) -> None:
    if num_episodes != SINGLE_DISTRIBUTION_EPISODE_COUNT:
        raise ValueError(
            "Single-distribution eval requires exactly "
            f"{SINGLE_DISTRIBUTION_EPISODE_COUNT} episodes"
        )


def _load_training_floor_plans(path: Path) -> frozenset[str]:
    records = (
        json.loads(line)
        for line in path.expanduser().read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    floor_plans = frozenset(
        _floor_plan_id(str(record["scene_uid"]))
        for record in records
        if record.get("status") == "success" and record.get("scene_uid")
    )
    if not floor_plans:
        raise ValueError(f"Training render manifest has no successful FloorPlan scenes: {path}")
    return floor_plans


def _validate_args(args: argparse.Namespace) -> None:
    if args.num_episodes <= 0:
        raise ValueError("--num-episodes must be positive")
    if args.floorplan_pool_size <= 0:
        raise ValueError("--floorplan-pool-size must be positive")
    if not (1 <= args.min_candidates <= args.max_candidates <= DEFAULT_MAX_CANDIDATES):
        raise ValueError("Candidate counts must satisfy 1 <= min <= max <= 20")
    if args.free_space_grid_m <= 0.0:
        raise ValueError("Grid size must be positive")
    global_split = (
        load_global_floorplan_split(args.floorplan_split)
        if args.floorplan_split is not None and args.floorplan_split.is_file()
        else None
    )
    uses_merged_split = (
        global_split is not None
        and global_split.get("schema") == MERGED_GLOBAL_FLOORPLAN_SPLIT_SCHEMA
    )
    split_has_scene_index = global_split is not None and isinstance(
        global_split.get("scene_index"), dict
    )
    if uses_merged_split and args.floorplan_split_view is None:
        raise ValueError("Merged FloorPlan split requires --floorplan-split-view")
    if not uses_merged_split and args.floorplan_split_view is not None:
        raise ValueError("Named FloorPlan views require the merged v3 split")
    if split_has_scene_index and args.scene_split_catalog is not None:
        raise ValueError("FloorPlan split with scene_index replaces --scene-split-catalog")
    if args.scene_split_catalog is not None:
        if not args.scene_split_catalog.is_file():
            raise FileNotFoundError(args.scene_split_catalog)
        validate_scene_split_catalog(
            args.scene_split_catalog,
            active_split=args.floorplan_split,
        )
    if args.build_candidates_only and args.build_candidates_from is None:
        raise ValueError("--build-candidates-only requires --build-candidates-from")
    if args.task_room_config is not None:
        if args.build_candidates_from is None:
            raise ValueError("--task-room-config is only valid while building candidates")
        if not args.task_room_config.is_file():
            raise FileNotFoundError(args.task_room_config)
    if args.build_candidates_from is not None and args.bev_cache_root is None:
        raise ValueError("Candidate generation requires --bev-cache-root data/cache")
    if args.bev_cache_root is not None and args.build_candidates_from is None:
        raise ValueError("--bev-cache-root is only valid while building candidates")
    build_values = (args.build_candidates_from, args.scene_index, args.layout_key)
    if any(value is not None for value in build_values):
        if args.candidate_manifest is None or not all(value is not None for value in build_values):
            raise ValueError(
                "Building a pool requires --candidate-manifest, --build-candidates-from, "
                "--scene-index, and --layout-key"
            )
        if args.candidate_manifest.exists():
            raise FileExistsError(
                f"Frozen candidate manifest already exists: {args.candidate_manifest}"
            )
    elif args.candidate_manifest is not None and not args.candidate_manifest.is_file():
        raise FileNotFoundError(args.candidate_manifest)
    id_ood_values = (args.training_render_manifest, args.floorplan_split)
    if args.eval_distribution is not None and not all(value is not None for value in id_ood_values):
        raise ValueError(
            "--eval-distribution requires --training-render-manifest and --floorplan-split"
        )
    if any(value is not None for value in id_ood_values):
        if not all(value is not None for value in id_ood_values):
            raise ValueError(
                "ID/scene-OOD eval requires both --training-render-manifest and --floorplan-split"
            )
        if args.build_candidates_only or args.candidate_manifest is None:
            raise ValueError("ID/OOD eval requires an existing candidate manifest")
        if not args.training_render_manifest.is_file():
            raise FileNotFoundError(args.training_render_manifest)
        if not args.floorplan_split.is_file():
            raise FileNotFoundError(args.floorplan_split)
        if args.floorplan_pool_size > args.num_episodes:
            raise ValueError("--floorplan-pool-size must not exceed --num-episodes")
        if args.eval_distribution is None:
            _validate_id_ood_episode_count(args.num_episodes)
        else:
            _validate_single_distribution_episode_count(args.num_episodes)


def _output_dir(args: argparse.Namespace, env_id: str) -> Path:
    if args.output_dir is not None:
        return args.output_dir.expanduser()
    task_name = env_id.split("/", maxsplit=1)[-1]
    return DEFAULT_OUTPUT_ROOT / task_name / f"level-{args.dr_level}"


def _successful_episode_indices(data_dir: Path, episodes: dict[int, Any]) -> tuple[int, ...]:
    records = {
        int(record["episode_index"]): record
        for line in (data_dir / "meta" / "episodes.jsonl").read_text().splitlines()
        if line
        for record in (json.loads(line),)
    }
    missing_metadata = sorted(set(episodes) - set(records))
    if missing_metadata:
        raise ValueError(f"Episode metadata missing for indices: {missing_metadata}")
    selected = tuple(
        index
        for index in sorted(episodes)
        if not records[index].get("partial_reason") and records[index].get("success") is not False
    )
    if not selected:
        raise ValueError(f"No successful demonstrations found in {data_dir}")
    return selected


def _hash_files(root: Path, paths: tuple[Path, ...]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(root)).encode())
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def _source_hash(data_dir: Path) -> str:
    paths = (
        *sorted((data_dir / "meta").glob("*.json*")),
        *sorted((data_dir / "data").rglob("*.parquet")),
    )
    return _hash_files(data_dir, paths)


def _code_revision() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()


def _collect_cached_layouts(args: argparse.Namespace, env_id: str):
    data_dir = args.build_candidates_from.expanduser().resolve()
    episodes = _load_lerobot_episodes(data_dir)
    configs = _load_episode_configs(data_dir)
    indices = _successful_episode_indices(data_dir, episodes)
    missing = sorted(set(indices) - set(configs))
    if missing:
        raise ValueError(f"Candidate source lacks environment configs: {missing}")
    suite = data_dir.parent.name
    task = env_id.split("/", 1)[-1].removesuffix("-v0")
    cache_root = args.bev_cache_root.expanduser().resolve()
    if set(indices) != training_episode_indices(cache_root, suite, task):
        raise ValueError(f"Public BEV cache episodes differ from successful source {suite}/{task}")
    points_by_episode = {
        index: load_trajectory_points(cache_root, suite, task, episode_index=index)
        for index in indices
    }
    layouts = tuple(
        parse_task_layout(configs[index], episode_index=index, clearance_m=0.0) for index in indices
    )
    bounds = tuple(
        AxisAlignedBounds(
            tuple(map(float, points.min(axis=0))), tuple(map(float, points.max(axis=0)))
        )
        for points in points_by_episode.values()
    )
    return data_dir, indices, layouts, bounds


def _candidate_records(
    scenes: tuple[SceneRecord, ...],
    placements: tuple[ScenePlacement, ...],
    revisions: dict[str, str],
    aggregate: TaskLayout,
) -> list[dict[str, Any]]:
    scene_by_uid = {scene.uid: scene for scene in scenes}
    records = []
    for rank, placement in enumerate(placements, start=1):
        scene = scene_by_uid[placement.scene_uid]
        room = transform_bounds(
            scene.bounds,
            placement.yaw_rad,
            placement.translation_xy,
        )
        if not room.contains_bounds(aggregate.protected_bounds):
            raise ValueError(f"Candidate {placement.scene_uid} does not cover all demonstrations")
        hidden_paths = list(freeze_placement_hidden_paths(scene, placement))
        obstacle_overlaps = [
            obstacle.prim_path
            for obstacle in scene.immutable_obstacles
            if _task_intersects_bounds(
                aggregate,
                transform_bounds(obstacle.bounds, placement.yaw_rad, placement.translation_xy),
            )
        ]
        if obstacle_overlaps:
            raise ValueError(
                f"Candidate {placement.scene_uid} overlaps immutable obstacles: "
                f"{obstacle_overlaps[:3]}"
            )
        records.append(
            {
                "rank": rank,
                "scene_uid": placement.scene_uid,
                "scene_revision": revisions[placement.scene_uid],
                "placement": placement.to_dict(),
                "hidden_prim_paths": hidden_paths,
                "covers_all_demonstrations": True,
                "immutable_obstacle_collision_free": True,
                "immutable_obstacle_count": len(scene.immutable_obstacles),
            }
        )
    return records


def _manifest_payload(
    *,
    args: argparse.Namespace,
    env_id: str,
    data_dir: Path,
    indices: tuple[int, ...],
    aggregate: TaskLayout,
    candidates: list[dict[str, Any]],
    scene_index: Path,
    room_type_policy: dict[str, Any] | None,
) -> dict[str, Any]:
    payload = {
        "schema": CANDIDATE_MANIFEST_SCHEMA,
        "layout_key": args.layout_key,
        "env_id": env_id,
        "hand_type": args.hand_type,
        "code_revision": _code_revision(),
        "source": {
            "dataset": project_path(data_dir),
            "sha256": _source_hash(data_dir),
            "episode_indices": list(indices),
        },
        "scene_index": {
            "path": project_path(scene_index),
            "sha256": hashlib.sha256(scene_index.read_bytes()).hexdigest(),
        },
        "algorithm": {
            "name": "fixed_layout_table_first_no_extra_hiding_v2",
            "seed": args.seed,
            "trajectory_margin_m": DEFAULT_TRAJECTORY_MARGIN_M,
            "free_space_grid_m": args.free_space_grid_m,
            "min_candidates": args.min_candidates,
            "max_candidates": args.max_candidates,
            "immutable_obstacle_policy": "reject_candidate",
        },
        "protected_bounds": {
            "min_xy": list(aggregate.protected_bounds.min_xy),
            "max_xy": list(aggregate.protected_bounds.max_xy),
        },
        "candidates": candidates,
    }
    if room_type_policy is not None:
        payload["room_type_policy"] = room_type_policy
    return payload


def _grouped_manifest_payload(
    *,
    args: argparse.Namespace,
    env_id: str,
    data_dir: Path,
    indices: tuple[int, ...],
    groups: list[dict[str, Any]],
    scene_index: Path,
    room_type_policy: dict[str, Any] | None,
    parameter_suite: str | None = None,
) -> dict[str, Any]:
    payload = {
        "schema": GROUPED_CANDIDATE_MANIFEST_SCHEMA,
        "layout_key": args.layout_key,
        "env_id": env_id,
        "hand_type": args.hand_type,
        "code_revision": _code_revision(),
        "source": {
            "dataset": project_path(data_dir),
            "sha256": _source_hash(data_dir),
            "episode_indices": list(indices),
        },
        "scene_index": {
            "path": project_path(scene_index),
            "sha256": hashlib.sha256(scene_index.read_bytes()).hexdigest(),
        },
        "algorithm": {
            "name": "fixed_layout_table_first_no_extra_hiding_by_layout_v2",
            "seed": args.seed,
            "trajectory_margin_m": DEFAULT_TRAJECTORY_MARGIN_M,
            "free_space_grid_m": args.free_space_grid_m,
            "min_candidates": args.min_candidates,
            "max_candidates": args.max_candidates,
            "immutable_obstacle_policy": "reject_candidate",
        },
        "groups": groups,
    }
    if room_type_policy is not None:
        payload["room_type_policy"] = room_type_policy
    if parameter_suite is not None:
        payload["parameter_suite"] = parameter_suite
    return payload


def _rank_candidate_records(
    *,
    scenes: tuple[SceneRecord, ...],
    revisions: dict[str, str],
    aggregate: TaskLayout,
    args: argparse.Namespace,
    room_type_policy: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    room_type_priority = (
        tuple(room_type_policy["allowed_room_types"]) if room_type_policy is not None else (None,)
    )
    room_priority = {
        room_type: index
        for index, room_type in enumerate(room_type_priority)
        if room_type is not None
    }
    room_scenes = tuple(
        scene
        for scene in scenes
        if not room_priority or floor_plan_room_type(_floor_plan_id(scene.uid)) in room_priority
    )
    if not room_scenes:
        raise ValueError("No scenes satisfy the task room type policy")
    if room_priority:
        room_scenes = tuple(
            sorted(
                room_scenes,
                key=lambda scene: room_priority[floor_plan_room_type(_floor_plan_id(scene.uid))],
            )
        )
    placements = rank_scene_placements(
        room_scenes,
        aggregate,
        max_candidates=len(room_scenes),
        seed=args.seed,
        free_space_grid_m=args.free_space_grid_m,
        allow_all_hide_fallback=False,
    )
    if room_priority:
        placements = tuple(
            sorted(
                placements,
                key=lambda placement: room_priority[
                    floor_plan_room_type(_floor_plan_id(placement.scene_uid))
                ],
            )
        )
    placements = placements[: args.max_candidates]
    if len(placements) < args.min_candidates:
        raise ValueError(
            f"Only {len(placements)} candidates cover all demonstrations; need {args.min_candidates}"
        )
    return _candidate_records(scenes, tuple(placements), revisions, aggregate)


def _write_candidate_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")


def _build_candidate_manifest(args: argparse.Namespace, env_id: str) -> dict[str, Any]:
    data_dir, indices, layouts, trajectories = _collect_cached_layouts(args, env_id)
    scene_index = args.scene_index.expanduser().resolve()
    scene_content = json.loads(scene_index.read_text(encoding="utf-8"))
    revisions = {
        str(scene["uid"]): str(scene.get("revision", "unknown"))
        for scene in scene_content["scenes"]
    }
    scenes = load_scene_records(scene_index, bev_cache_root=args.bev_cache_root)
    room_type_policy = _load_task_room_policy(args.task_room_config, env_id)
    suite = data_dir.parent.name
    task_name = env_id.split("/", 1)[-1].removesuffix("-v0")
    task_occupied_xy = load_trajectory_points(args.bev_cache_root, suite, task_name)
    parameter_suite = suite if suite in {"spatial", "goal", "condition", "object"} else None
    configs = _load_episode_configs(data_dir) if parameter_suite is not None else None
    grouped = _group_layouts_by_anchor(
        layouts, trajectories, configs=configs, parameter_suite=parameter_suite
    )
    if len(grouped) == 1 and parameter_suite is None:
        aggregate = aggregate_task_layouts(layouts, trajectories, margin_m=0.0)
        aggregate = replace(
            aggregate, occupied_xy=task_occupied_xy, hide_occupied_xy=task_occupied_xy
        )
        payload = _manifest_payload(
            args=args,
            env_id=env_id,
            data_dir=data_dir,
            indices=indices,
            aggregate=aggregate,
            candidates=_rank_candidate_records(
                scenes=scenes,
                revisions=revisions,
                aggregate=aggregate,
                args=args,
                room_type_policy=room_type_policy,
            ),
            scene_index=scene_index,
            room_type_policy=room_type_policy,
        )
    else:
        group_records = []
        for group_key, (group_layouts, group_trajectories) in grouped.items():
            group_occupied_xy = np.concatenate(
                [
                    load_trajectory_points(
                        args.bev_cache_root,
                        suite,
                        task_name,
                        episode_index=layout.episode_index,
                    )
                    for layout in group_layouts
                ]
            )
            aggregate = aggregate_task_layouts(group_layouts, group_trajectories, margin_m=0.0)
            aggregate = replace(
                aggregate,
                occupied_xy=group_occupied_xy,
                hide_occupied_xy=group_occupied_xy,
                protected_bounds=_union_bounds(
                    aggregate.protected_bounds,
                    AxisAlignedBounds(
                        tuple(map(float, group_occupied_xy.min(axis=0))),
                        tuple(map(float, group_occupied_xy.max(axis=0))),
                    ),
                ),
            )
            group_records.append(
                {
                    "layout_group_key": group_key,
                    **(
                        {
                            "parameter_key": _task_parameter_key(
                                parameter_suite, configs[group_layouts[0].episode_index]
                            )
                        }
                        if parameter_suite is not None
                        else {}
                    ),
                    "anchor": _layout_anchor_payload(group_layouts[0]),
                    "source_episode_indices": [layout.episode_index for layout in group_layouts],
                    "protected_bounds": {
                        "min_xy": list(aggregate.protected_bounds.min_xy),
                        "max_xy": list(aggregate.protected_bounds.max_xy),
                    },
                    "candidates": _rank_candidate_records(
                        scenes=scenes,
                        revisions=revisions,
                        aggregate=aggregate,
                        args=args,
                        room_type_policy=room_type_policy,
                    ),
                }
            )
        payload = _grouped_manifest_payload(
            args=args,
            env_id=env_id,
            data_dir=data_dir,
            indices=indices,
            groups=group_records,
            scene_index=scene_index,
            room_type_policy=room_type_policy,
            parameter_suite=parameter_suite,
        )
    payload["bev_cache"] = {
        "root": project_path(args.bev_cache_root),
        "manifest_id": cache_manifest_id(args.bev_cache_root),
        "suite": suite,
        "task": task_name,
    }
    payload["task_bev_occupancy"] = {
        "bundle": project_path(args.bev_cache_root),
        "suite": suite,
        "task": task_name,
        "point_count": len(task_occupied_xy),
    }
    manifest = finalize_candidate_manifest(payload)
    _write_candidate_manifest(args.candidate_manifest.expanduser(), manifest)
    return manifest


def _load_candidate_manifest(path: Path, env_id: str, hand_type: str) -> dict[str, Any]:
    manifest = json.loads(path.expanduser().read_text(encoding="utf-8"))
    validate_candidate_manifest(manifest)
    bev_cache = manifest.get("bev_cache")
    if not isinstance(bev_cache, dict) or not bev_cache.get("manifest_id"):
        raise ValueError("Candidate manifest must identify its level-one BEV cache")
    if cache_manifest_id(resolve_project_path(bev_cache["root"])) != bev_cache["manifest_id"]:
        raise ValueError("Candidate manifest uses a different BEV cache manifest")
    if manifest.get("env_id") != env_id or manifest.get("hand_type") != hand_type:
        raise ValueError(
            "Candidate manifest env_id or hand_type does not match this generation run"
        )
    return manifest


def _attach_scene_selection(
    environment_config: dict[str, Any],
    manifest: dict[str, Any],
    *,
    episode_index: int,
    seed: int,
    selected_floor_plan: str | None = None,
    training_render_manifest: Path | None = None,
    global_split: dict[str, Any] | None = None,
    global_split_path: Path | None = None,
    global_split_view: str | None = None,
    distribution: str | None = None,
) -> dict[str, Any]:
    layout = parse_task_layout(environment_config, episode_index=episode_index, clearance_m=0.0)
    if selected_floor_plan is not None and distribution is None:
        distribution = _episode_distribution(episode_index)
    candidate, group_key = _select_scene_candidate(
        manifest,
        episode_index=episode_index,
        seed=seed,
        layout=layout,
        environment_config=environment_config,
        floor_plan=selected_floor_plan,
    )
    split_has_scene_index = global_split is not None and isinstance(
        global_split.get("scene_index"), dict
    )
    if split_has_scene_index:
        if distribution is None or global_split_path is None:
            raise ValueError("Eval scene selection requires split and distribution")
        if (
            global_split.get("schema") == MERGED_GLOBAL_FLOORPLAN_SPLIT_SCHEMA
            and global_split_view is None
        ):
            raise ValueError("Merged eval scene selection requires a split view")
        if manifest["scene_index"]["sha256"] != global_split["scene_index"]["sha256"]:
            raise ValueError("Candidate manifest scene index does not match the FloorPlan split")
        selection = build_eval_scene_selection(
            candidate,
            split=global_split,
            split_manifest_path=Path(project_path(global_split_path)),
            view=global_split_view,
            distribution=distribution,
            candidate_manifest_hash=manifest["content_hash"],
            layout_group_key=group_key,
        )
    else:
        selection = {
            "schema": "humano.eval-scene-selection.v2",
            "candidate_manifest_hash": manifest["content_hash"],
            "scene_index": manifest["scene_index"]["path"],
            "scene_index_hash": manifest["scene_index"]["sha256"],
            "scene_uid": candidate["scene_uid"],
            "scene_revision": candidate["scene_revision"],
            "placement": candidate["placement"],
            "hidden_prim_paths": candidate["hidden_prim_paths"],
            "immutable_obstacle_collision_free": candidate["immutable_obstacle_collision_free"],
            "immutable_obstacle_count": candidate["immutable_obstacle_count"],
            **(
                {
                    "distribution": distribution,
                    "floor_plan": _floor_plan_id(candidate["scene_uid"]),
                    "training_render_manifest": project_path(training_render_manifest),
                    "global_floorplan_split": project_path(global_split_path),
                    "global_floorplan_split_hash": global_split["content_hash"],
                }
                if distribution is not None
                else {}
            ),
            **({"layout_group_key": group_key} if group_key is not None else {}),
        }
    selection["hidden_prim_paths_final"] = True
    if "bev_cache" in manifest:
        selection["bev_cache"] = manifest["bev_cache"]
    return {**environment_config, "eval_scene": selection}


def _generate(args: argparse.Namespace) -> Path:
    _validate_args(args)
    env_id = _normalize_env_id(args.env_id)
    output_dir = _output_dir(args, env_id)
    if not args.build_candidates_only and output_dir.exists():
        raise FileExistsError(f"Output dataset already exists: {output_dir}")
    random.seed(args.seed)
    np.random.seed(args.seed)
    hand_spec = get_hand_spec(args.hand_type)
    sonic_config = load_sonic_config(hand_spec)
    env = gym.make(
        env_id,
        sim_mode="mujoco",
        headless=True,
        sonic_config=sonic_config,
        hand_type=hand_spec.name,
        dr_level=args.dr_level,
    )
    try:
        manifest = None
        if args.build_candidates_from is not None:
            manifest = _build_candidate_manifest(args, env_id)
        elif args.candidate_manifest is not None:
            manifest = _load_candidate_manifest(args.candidate_manifest, env_id, args.hand_type)
        training_floor_plans = (
            _load_training_floor_plans(args.training_render_manifest)
            if args.training_render_manifest is not None
            else None
        )
        training_materials = None
        if args.training_render_manifest is not None:
            assignment_path = (
                args.training_render_manifest.parents[2] / "meta" / "table_material_assignment.json"
            )
            assignment = (
                json.loads(assignment_path.read_text(encoding="utf-8"))
                if assignment_path.is_file()
                else None
            )
            training_materials = training_table_material_choices(assignment)
        global_split = (
            load_global_floorplan_split(args.floorplan_split)
            if args.floorplan_split is not None
            else None
        )
        if args.build_candidates_only:
            assert args.candidate_manifest is not None
            return args.candidate_manifest.expanduser()
        task = env.unwrapped.task
        episode_records = []
        episode_group_keys = []
        for episode_index in range(args.num_episodes):
            observation, _info = env.reset(seed=args.seed + episode_index)
            config = task.state_dict()
            group_key = None
            if manifest is not None:
                layout = parse_task_layout(config, episode_index=episode_index, clearance_m=0.0)
                _group, group_key = _candidate_group_for_layout(manifest, layout, config)
            episode_records.append((_build_frame(task, observation), task.instruction, config))
            episode_group_keys.append(group_key)
        floor_plan_schedule = None
        episode_distributions = None
        if global_split is not None:
            assert manifest is not None and training_floor_plans is not None
            global_train_floor_plans, global_eval_floor_plans = split_floor_plan_sets(
                global_split,
                view=args.floorplan_split_view,
            )
            episode_distributions = (
                (args.eval_distribution,) * args.num_episodes
                if args.eval_distribution is not None
                else tuple(_episode_distribution(index) for index in range(args.num_episodes))
            )
            requested_distributions = tuple(dict.fromkeys(episode_distributions))
            grouped_episode_keys = {
                distribution: tuple(
                    group_key
                    for group_key, episode_distribution in zip(
                        episode_group_keys, episode_distributions, strict=True
                    )
                    if episode_distribution == distribution
                )
                for distribution in requested_distributions
            }
            distribution_pools = _build_distribution_floorplan_pools(
                manifest,
                training_floor_plans=training_floor_plans,
                global_train_floor_plans=global_train_floor_plans,
                global_eval_floor_plans=global_eval_floor_plans,
                seed=args.seed,
                pool_size=args.floorplan_pool_size,
                episode_group_keys=grouped_episode_keys,
                distributions=requested_distributions,
            )
            floor_plan_schedule = _build_episode_floorplan_schedule(
                manifest,
                tuple(episode_group_keys),
                distribution_pools,
                episode_distributions,
            )
        dataset = LeRobotDataset.create(
            repo_id=env_id,
            root=output_dir,
            fps=int(task.metadata["render_hz"]),
            features=_build_features(task),
        )
        if manifest is not None:
            assert args.candidate_manifest is not None
        for episode_index, (frame, instruction, original_config) in enumerate(episode_records):
            config = original_config
            if manifest is not None:
                config = _attach_scene_selection(
                    config,
                    manifest,
                    episode_index=episode_index,
                    seed=args.seed,
                    selected_floor_plan=(
                        floor_plan_schedule[episode_index]
                        if floor_plan_schedule is not None
                        else None
                    ),
                    training_render_manifest=args.training_render_manifest,
                    global_split=global_split,
                    global_split_path=args.floorplan_split,
                    global_split_view=args.floorplan_split_view,
                    distribution=(
                        episode_distributions[episode_index]
                        if episode_distributions is not None
                        else None
                    ),
                )
            if training_materials is not None:
                material = training_materials[episode_index % len(training_materials)]
                config = with_task_table_material(config, material)
                config = with_task_table_material(config, material, table_name="table2")
            dataset.add_frame(frame, task=instruction)
            dataset.save_episode()
            _write_environment_config(
                output_dir, episode_index=episode_index, environment_config=config
            )
            print(f"Generated eval episode {episode_index + 1}/{args.num_episodes}")
    finally:
        env.close()
    return output_dir


def main() -> None:
    args = _parse_args()
    output_dir = _generate(args)
    label = "Candidate manifest" if args.build_candidates_only else "Eval dataset"
    print(f"{label}: {output_dir}")


if __name__ == "__main__":
    main()
