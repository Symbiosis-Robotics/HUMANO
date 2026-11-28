"""Load and validate the frozen global FloorPlan train/eval split."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

GLOBAL_FLOORPLAN_SPLIT_SCHEMA = "humano.global-floorplan-split-90.v2"
LEGACY_GLOBAL_FLOORPLAN_SPLIT_SCHEMA = "humano.global-floorplan-split-100.v1"
MERGED_GLOBAL_FLOORPLAN_SPLIT_SCHEMA = "humano.global-floorplan-split.v3"

LEGACY_BASIC_VIEW = "legacy_basic"
STANDARD_NO_BATHROOM_VIEW = "standard_no_bathroom"

_ROOM_TYPES = ("bathroom", "bedroom", "kitchen", "living_room")
_MERGED_ROOM_TYPE_COUNTS = {
    "bathroom": {"source": 21, "train": 14, "eval": 7},
    "bedroom": {"source": 30, "train": 20, "eval": 10},
    "kitchen": {"source": 30, "train": 20, "eval": 10},
    "living_room": {"source": 30, "train": 20, "eval": 10},
}
_MERGED_VIEW_SPECS = {
    LEGACY_BASIC_VIEW: {
        "include_room_types": list(_ROOM_TYPES),
        "counts": (65, 35),
    },
    STANDARD_NO_BATHROOM_VIEW: {
        "include_room_types": ["bedroom", "kitchen", "living_room"],
        "counts": (60, 30),
    },
}

_SPLIT_SPECS = {
    GLOBAL_FLOORPLAN_SPLIT_SCHEMA: {
        "counts": (60, 30, 0),
        "room_type_counts": {
            "bedroom": {"source": 30, "excluded": 0, "train": 20, "eval": 10},
            "kitchen": {"source": 30, "excluded": 0, "train": 20, "eval": 10},
            "living_room": {"source": 30, "excluded": 0, "train": 20, "eval": 10},
        },
        "excluded_room_types": ["bathroom"],
    },
    LEGACY_GLOBAL_FLOORPLAN_SPLIT_SCHEMA: {
        "counts": (65, 35, 3),
        "room_type_counts": {
            "bathroom": {"source": 21, "excluded": 0, "train": 14, "eval": 7},
            "bedroom": {"source": 29, "excluded": 1, "train": 18, "eval": 10},
            "kitchen": {"source": 26, "excluded": 1, "train": 16, "eval": 9},
            "living_room": {"source": 27, "excluded": 1, "train": 17, "eval": 9},
        },
        "excluded_room_types": None,
    },
}


def floor_plan_id(scene_uid: str) -> str:
    match = re.search(r"FloorPlan\d+", scene_uid)
    if match is None:
        raise ValueError(f"Scene uid has no physical FloorPlan id: {scene_uid}")
    return match.group(0)


def floor_plan_room_type(floor_plan: str) -> str:
    number = int(floor_plan.removeprefix("FloorPlan"))
    if 1 <= number <= 30:
        return "kitchen"
    if 201 <= number <= 230:
        return "living_room"
    if 301 <= number <= 330:
        return "bedroom"
    if 401 <= number <= 430:
        return "bathroom"
    raise ValueError(f"FloorPlan is outside the iTHOR 120 catalog: {floor_plan}")


def _split_digest(split: dict[str, Any]) -> str:
    payload = {key: value for key, value in split.items() if key != "content_hash"}
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode()).hexdigest()


def _sort_floor_plans(values: set[str] | frozenset[str]) -> list[str]:
    return sorted(values, key=lambda value: int(value.removeprefix("FloorPlan")))


def _merged_assignment_sets(
    split: dict[str, Any],
) -> tuple[frozenset[str], frozenset[str]]:
    assignments = split.get("assignments")
    if not isinstance(assignments, dict):
        raise ValueError("Merged FloorPlan split assignments must be a mapping")
    train = tuple(str(value) for value in assignments.get("train_floor_plans", ()))
    eval_floor_plans = tuple(
        str(value) for value in assignments.get("eval_floor_plans", ())
    )
    if len(train) != 74 or len(eval_floor_plans) != 37:
        raise ValueError(
            "Merged FloorPlan split must contain exactly 74 train and 37 eval scenes"
        )
    if len(set(train)) != len(train) or len(set(eval_floor_plans)) != len(eval_floor_plans):
        raise ValueError("Merged FloorPlan split contains duplicate FloorPlans")
    if set(train) & set(eval_floor_plans):
        raise ValueError("Merged FloorPlan train and eval assignments overlap")
    return frozenset(train), frozenset(eval_floor_plans)


def _project_merged_view(
    split: dict[str, Any], view: str
) -> tuple[frozenset[str], frozenset[str]]:
    views = split.get("views")
    if not isinstance(views, dict) or view not in views:
        raise ValueError(f"Unsupported merged FloorPlan split view: {view!r}")
    view_spec = views[view]
    if not isinstance(view_spec, dict):
        raise ValueError(f"Merged FloorPlan split view must be a mapping: {view}")
    include_room_types = frozenset(
        str(value) for value in view_spec.get("include_room_types", ())
    )
    excluded = frozenset(
        str(value) for value in view_spec.get("excluded_floor_plans", ())
    )
    train, eval_floor_plans = _merged_assignment_sets(split)

    def included(floor_plan: str) -> bool:
        return floor_plan not in excluded and floor_plan_room_type(floor_plan) in include_room_types

    return (
        frozenset(value for value in train if included(value)),
        frozenset(value for value in eval_floor_plans if included(value)),
    )


def _validate_merged_global_floorplan_split(split: dict[str, Any]) -> None:
    train, eval_floor_plans = _merged_assignment_sets(split)
    if len(train | eval_floor_plans) != 111:
        raise ValueError("Merged FloorPlan split must assign exactly 111 FloorPlans")

    room_type_counts = {
        room_type: {
            "source": sum(
                floor_plan_room_type(value) == room_type
                for value in train | eval_floor_plans
            ),
            "train": sum(floor_plan_room_type(value) == room_type for value in train),
            "eval": sum(
                floor_plan_room_type(value) == room_type for value in eval_floor_plans
            ),
        }
        for room_type in _ROOM_TYPES
    }
    if room_type_counts != _MERGED_ROOM_TYPE_COUNTS:
        raise ValueError("Merged FloorPlan split does not match the frozen room-type allocation")
    if split.get("room_type_counts") != room_type_counts:
        raise ValueError("Merged FloorPlan room type counts do not match its assignments")

    scene_index = split.get("scene_index")
    if not isinstance(scene_index, dict):
        raise ValueError("Merged FloorPlan split scene index must be a mapping")
    required_scene_index = {
        "path": str,
        "schema": str,
        "sha256": str,
        "scene_uid_count": int,
        "floor_plan_count": int,
    }
    if any(
        not isinstance(scene_index.get(key), expected_type)
        for key, expected_type in required_scene_index.items()
    ):
        raise ValueError("Merged FloorPlan split scene index metadata is incomplete")
    if scene_index["scene_uid_count"] != 114 or scene_index["floor_plan_count"] != 111:
        raise ValueError("Merged FloorPlan split scene index counts do not match the release")
    if len(scene_index["sha256"]) != 64:
        raise ValueError("Merged FloorPlan split scene index hash must be SHA-256")

    views = split.get("views")
    if not isinstance(views, dict) or set(views) != set(_MERGED_VIEW_SPECS):
        raise ValueError("Merged FloorPlan split must define both frozen views")
    for view, expected in _MERGED_VIEW_SPECS.items():
        view_spec = views[view]
        if view_spec.get("include_room_types") != expected["include_room_types"]:
            raise ValueError(f"Merged FloorPlan split view room types do not match: {view}")
        excluded = tuple(str(value) for value in view_spec.get("excluded_floor_plans", ()))
        if len(excluded) != len(set(excluded)) or not set(excluded) <= train | eval_floor_plans:
            raise ValueError(f"Merged FloorPlan split view exclusions are invalid: {view}")
        view_train, view_eval = _project_merged_view(split, view)
        expected_train, expected_eval = expected["counts"]
        expected_counts = {"train": expected_train, "eval": expected_eval}
        if view_spec.get("expected_counts") != expected_counts:
            raise ValueError(f"Merged FloorPlan split view counts do not match: {view}")
        if len(view_train) != expected_train or len(view_eval) != expected_eval:
            raise ValueError(f"Merged FloorPlan split view projection counts do not match: {view}")

    provenance = split.get("provenance")
    if not isinstance(provenance, dict) or set(provenance) != {
        "legacy_basic_split",
        "advanced_split",
    }:
        raise ValueError("Merged FloorPlan split provenance is incomplete")
    for label in ("legacy_basic_split", "advanced_split"):
        source = provenance[label]
        if not isinstance(source, dict) or not all(
            isinstance(source.get(key), str)
            for key in ("schema", "content_hash", "scene_index_sha256")
        ):
            raise ValueError(f"Merged FloorPlan split provenance is invalid: {label}")


def validate_global_floorplan_split(split: dict[str, Any]) -> None:
    schema = split.get("schema")
    if schema == MERGED_GLOBAL_FLOORPLAN_SPLIT_SCHEMA:
        if split.get("content_hash") != _split_digest(split):
            raise ValueError("Global FloorPlan split content hash does not match its contents")
        _validate_merged_global_floorplan_split(split)
        return
    if not isinstance(schema, str) or schema not in _SPLIT_SPECS:
        raise ValueError(f"Unsupported global FloorPlan split schema: {split.get('schema')!r}")
    spec = _SPLIT_SPECS[schema]
    if split.get("content_hash") != _split_digest(split):
        raise ValueError("Global FloorPlan split content hash does not match its contents")
    train = tuple(str(value) for value in split.get("train_floor_plans", ()))
    eval_floor_plans = tuple(str(value) for value in split.get("eval_floor_plans", ()))
    excluded = tuple(str(value) for value in split.get("excluded_floor_plans", ()))
    train_count, eval_count, excluded_count = spec["counts"]
    if (
        len(train) != train_count
        or len(eval_floor_plans) != eval_count
        or len(excluded) != excluded_count
    ):
        raise ValueError(
            "Global FloorPlan split must contain exactly "
            f"{train_count} train, {eval_count} eval, and {excluded_count} excluded scenes"
        )
    if any(len(set(values)) != len(values) for values in (train, eval_floor_plans, excluded)):
        raise ValueError("Global FloorPlan split contains duplicate FloorPlans")
    if (
        set(train) & set(eval_floor_plans)
        or set(train) & set(excluded)
        or set(eval_floor_plans) & set(excluded)
    ):
        raise ValueError("Global FloorPlan train, eval, and excluded sets overlap")
    expected_room_type_counts = spec["room_type_counts"]
    room_type_counts = {
        room_type: {
            "source": sum(
                floor_plan_room_type(value) == room_type
                for value in (*train, *eval_floor_plans, *excluded)
            ),
            "excluded": sum(
                floor_plan_room_type(value) == room_type for value in excluded
            ),
            "train": sum(floor_plan_room_type(value) == room_type for value in train),
            "eval": sum(floor_plan_room_type(value) == room_type for value in eval_floor_plans),
        }
        for room_type in expected_room_type_counts
    }
    if room_type_counts != expected_room_type_counts:
        raise ValueError("Global FloorPlan split does not match the frozen room-type allocation")
    if split.get("room_type_counts") != room_type_counts:
        raise ValueError("Global FloorPlan room type counts do not match the split contents")
    expected_excluded_room_types = spec["excluded_room_types"]
    if (
        expected_excluded_room_types is not None
        and split.get("excluded_room_types") != expected_excluded_room_types
    ):
        raise ValueError("Global FloorPlan split does not match the excluded room types")


def finalize_global_floorplan_split(split: dict[str, Any]) -> dict[str, Any]:
    frozen = {key: value for key, value in split.items() if key != "content_hash"}
    frozen["content_hash"] = _split_digest(frozen)
    validate_global_floorplan_split(frozen)
    return frozen


def build_merged_global_floorplan_split(
    legacy_basic: dict[str, Any],
    advanced: dict[str, Any],
    *,
    scene_index: dict[str, Any],
) -> dict[str, Any]:
    validate_global_floorplan_split(legacy_basic)
    validate_global_floorplan_split(advanced)
    if legacy_basic.get("schema") != LEGACY_GLOBAL_FLOORPLAN_SPLIT_SCHEMA:
        raise ValueError("Merged split requires the frozen legacy Basic split")
    if advanced.get("schema") != GLOBAL_FLOORPLAN_SPLIT_SCHEMA:
        raise ValueError("Merged split requires the frozen Advanced split")
    validate_split_compatibility(legacy_basic, advanced)

    basic_train, basic_eval = split_floor_plan_sets(legacy_basic)
    advanced_train, advanced_eval = split_floor_plan_sets(advanced)
    canonical_train = basic_train | advanced_train
    canonical_eval = basic_eval | advanced_eval
    if canonical_train & canonical_eval:
        raise ValueError("Merged split sources assign a FloorPlan to both train and eval")

    basic_selected = basic_train | basic_eval
    legacy_excluded = (canonical_train | canonical_eval) - basic_selected
    room_type_counts = {
        room_type: {
            "source": sum(
                floor_plan_room_type(value) == room_type
                for value in canonical_train | canonical_eval
            ),
            "train": sum(
                floor_plan_room_type(value) == room_type for value in canonical_train
            ),
            "eval": sum(
                floor_plan_room_type(value) == room_type for value in canonical_eval
            ),
        }
        for room_type in _ROOM_TYPES
    }
    return finalize_global_floorplan_split(
        {
            "schema": MERGED_GLOBAL_FLOORPLAN_SPLIT_SCHEMA,
            "scene_index": dict(scene_index),
            "assignments": {
                "train_floor_plans": _sort_floor_plans(canonical_train),
                "eval_floor_plans": _sort_floor_plans(canonical_eval),
            },
            "room_type_counts": room_type_counts,
            "views": {
                LEGACY_BASIC_VIEW: {
                    "usage": "Historical Basic train/eval protocol",
                    "include_room_types": list(_ROOM_TYPES),
                    "excluded_floor_plans": _sort_floor_plans(legacy_excluded),
                    "expected_counts": {"train": 65, "eval": 35},
                },
                STANDARD_NO_BATHROOM_VIEW: {
                    "usage": "Advanced and future suite train/eval protocol",
                    "include_room_types": ["bedroom", "kitchen", "living_room"],
                    "excluded_floor_plans": [],
                    "expected_counts": {"train": 60, "eval": 30},
                },
            },
            "provenance": {
                "legacy_basic_split": {
                    "schema": legacy_basic["schema"],
                    "content_hash": legacy_basic["content_hash"],
                    "scene_index_sha256": legacy_basic["scene_index_sha256"],
                },
                "advanced_split": {
                    "schema": advanced["schema"],
                    "content_hash": advanced["content_hash"],
                    "scene_index_sha256": advanced["scene_index_sha256"],
                },
            },
        }
    )


def load_global_floorplan_split(path: Path) -> dict[str, Any]:
    split = json.loads(path.expanduser().read_text(encoding="utf-8"))
    validate_global_floorplan_split(split)
    return split


def split_floor_plan_sets(
    split: dict[str, Any], *, view: str | None = None
) -> tuple[frozenset[str], frozenset[str]]:
    validate_global_floorplan_split(split)
    if split["schema"] == MERGED_GLOBAL_FLOORPLAN_SPLIT_SCHEMA:
        if view is None:
            raise ValueError("Merged FloorPlan split requires an explicit view")
        return _project_merged_view(split, view)
    if view is not None:
        raise ValueError("Legacy FloorPlan split does not support named views")
    return (
        frozenset(str(value) for value in split["train_floor_plans"]),
        frozenset(str(value) for value in split["eval_floor_plans"]),
    )


def validate_split_compatibility(reference: dict[str, Any], candidate: dict[str, Any]) -> None:
    reference_train, reference_eval = split_floor_plan_sets(reference)
    candidate_train, candidate_eval = split_floor_plan_sets(candidate)
    train_to_eval = sorted(reference_train & candidate_eval)
    eval_to_train = sorted(reference_eval & candidate_train)
    if train_to_eval or eval_to_train:
        raise ValueError(
            "Global FloorPlan splits change an existing train/eval assignment: "
            f"train_to_eval={train_to_eval}, eval_to_train={eval_to_train}"
        )
