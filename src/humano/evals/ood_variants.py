"""Paired, single-factor OOD configuration derivation (no simulator at import)."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

VARIANT_SCHEMA = "humano.ood-variant.v1"
DIMENSIONS = {
    "lighting": ("rgb", "local", "size"),
    "camera": ("x", "y", "z", "roll", "pitch", "yaw"),
    "language": ("T1", "T2", "T3"),
    "sensor_noise": ("gaussian", "blur"),
    "robot_initial": ("translation", "yaw"),
    "distractor": ("objects",),
    "background": ("id", "scene_ood"),
}


def task_key(env_id: str) -> str:
    return env_id.rsplit("/", 1)[-1].removesuffix("-v0")


def language_tasks(payload: dict) -> dict[str, dict]:
    return {task_key(row["task"]): row for row in payload["tasks"]}


def native_instruction(env_id: str, tasks: dict[str, dict]) -> str:
    return tasks[task_key(env_id)]["native_description"]


def head_camera_uid(config: dict) -> str:
    sensors = config.get("sensor_cfgs", {})
    candidates = [
        value.get("uid", key)
        for key, value in sensors.items()
        if "head" in str(value.get("mount", value.get("mount_to", ""))).lower()
    ]
    if not candidates:
        candidates = [
            value.get("uid", key)
            for key, value in sensors.items()
            if "Realsense_D435i" in (key, value.get("uid"))
        ]
    if len(candidates) != 1:
        raise ValueError(f"Cannot resolve unique actual head camera UID: {candidates}")
    return candidates[0]


def derive_camera_variant(base_state, subtype, rng, *, camera_uid, ranges=None):
    if subtype not in DIMENSIONS["camera"]:
        raise ValueError(f"Unknown camera axis: {subtype}")
    ranges = ranges or {}
    state = deepcopy(base_state or {})
    offsets = state.setdefault(camera_uid, {"translation": [0.0] * 3, "rotation_deg": [0.0] * 3})
    translation = subtype in ("x", "y", "z")
    field = "translation" if translation else "rotation_deg"
    axis = (
        ("x", "y", "z").index(subtype) if translation else ("roll", "pitch", "yaw").index(subtype)
    )
    defaults = {
        "translation_m": (-0.01, 0.01),
        "roll_deg": (-1.5, 1.5),
        "pitch_deg": (-15, 5),
        "yaw_deg": (-10, 10),
    }
    range_key = "translation_m" if translation else f"{subtype}_deg"
    bounds = ranges.get(range_key, defaults[range_key])
    import numpy as np

    if len(bounds) != 2 or not np.all(np.isfinite(bounds)) or bounds[0] >= bounds[1]:
        raise ValueError(f"Invalid camera bounds: {bounds}")
    for name in ("translation", "rotation_deg"):
        values = offsets[name]
        if len(values) != 3 or not np.all(np.isfinite(values)):
            raise ValueError(f"Invalid baseline camera {name}")
    delta = float(rng.uniform(*bounds))
    baseline = float(offsets[field][axis])
    offsets[field][axis] = baseline + delta
    return state, {
        "axis": subtype,
        "camera_uid": camera_uid,
        "delta": delta,
        "baseline": baseline,
        "range": list(bounds),
        "units": "m" if translation else "degrees",
    }


def background_condition(config: dict, train: set, scene_ood: set) -> dict:
    from humano.scenes.floorplan_split import floor_plan_id

    selection = config.get("eval_scene") or {}
    uid = selection.get("scene_uid") or selection.get("scene", {}).get("uid")
    if not uid:
        return {"accepted": False, "reason": "missing_eval_scene"}
    floorplan = floor_plan_id(uid)
    global_distribution = (
        "id" if floorplan in train else "scene_ood" if floorplan in scene_ood else None
    )
    declared = selection.get("distribution") or selection.get("split", {}).get("distribution")
    mismatch = declared is not None and declared != global_distribution
    accepted = declared in ("id", "scene_ood") and global_distribution == declared
    reason = None
    if global_distribution is None:
        reason = "outside_standard_no_bathroom"
    elif declared is None:
        reason = "needs_training_manifest"
    elif mismatch:
        reason = "source_split_mismatch"
    return {
        "accepted": accepted,
        "floor_plan": floorplan,
        "distribution": declared if accepted else None,
        "global_distribution": global_distribution,
        "source_distribution": declared,
        "source_split_mismatch": mismatch,
        "task_distribution_evidence": "source_eval_scene_declaration"
        if declared in ("id", "scene_ood")
        else None,
        "view": "standard_no_bathroom",
        "reason": reason,
    }


def derive_background_variant(
    config: dict, subtype: str, rng: Any, settings: dict
) -> tuple[dict, dict]:
    from humano.dr.material import _BLACK_TABLE_MATERIAL_NAMES
    from humano.scenes.floorplan_split import floor_plan_id, floor_plan_room_type

    material_distribution = "train" if subtype == "id" else "test"
    selection = config.get("eval_scene") or {}
    uid = selection.get("scene_uid") or selection.get("scene", {}).get("uid")
    if not uid:
        raise ValueError("Background variants require an indexed scene UID")
    room_type = floor_plan_room_type(floor_plan_id(uid))
    materials = settings["table_materials"]

    def family(item: dict) -> str:
        if item.get("type") == "usd_preview_surface":
            return item["family"]
        return item["path"].split("/")[2]

    families = {split: {family(item) for item in materials[split]} for split in ("train", "test")}
    if families["train"] != {"SolidColor"} or families["test"] not in (
        {"Plastic", "Metal"},
        {"Plastic", "Metal", "Leather", "Glass"},
    ):
        raise ValueError(f"Background material families violate the frozen split: {families}")
    for split in ("train", "test"):
        for item in materials[split]:
            if item.get("name") in _BLACK_TABLE_MATERIAL_NAMES:
                raise ValueError(f"Background pool contains blacklisted material: {item['name']}")
    pool = materials[material_distribution]
    if not pool:
        raise ValueError(f"Empty background {material_distribution} table material pool")
    family_names = list(dict.fromkeys(family(item) for item in pool))
    selected_family = family_names[int(rng.integers(len(family_names)))]
    family_pool = [item for item in pool if family(item) == selected_family]
    material = deepcopy(family_pool[int(rng.integers(len(family_pool)))])
    state = config.setdefault("dr_state_dict", {})
    scene = state.get("scene")
    table = scene.get("table") if isinstance(scene, dict) else None
    source_material = (
        deepcopy((state.get("material") or {}).get("table_material") or table.get("material"))
        if isinstance(table, dict)
        else None
    )
    state.setdefault("material", {})["table_material"] = deepcopy(material)
    if isinstance(table, dict):
        table["material"] = deepcopy(material)
    layout = config.get("layout") or {}
    for layout_table in (
        (layout.get("actors") or {}).get("table"),
        (layout.get("scene") or {}).get("table"),
    ):
        if isinstance(layout_table, dict):
            layout_table["material"] = deepcopy(material)
    return config, {
        "scene_distribution": subtype,
        "material_distribution": material_distribution,
        "material_split_policy": "family_disjoint",
        "material_family": family(material),
        "room_type": room_type,
        "material_target": "table",
        "source_has_task_table": isinstance(table, dict),
        "source_table_material": source_material,
        "table_material": material,
    }


def derive_variant(
    base_config: dict,
    dimension: str,
    subtype: str,
    rng: Any,
    *,
    defaults: dict,
    env_id: str,
    tasks: dict,
    env=None,
    indexed_scene=None,
) -> tuple[dict, dict]:
    if dimension not in DIMENSIONS or subtype not in DIMENSIONS[dimension]:
        raise ValueError(f"Unsupported OOD variant: {dimension}/{subtype}")
    config = deepcopy(base_config)
    state = config.setdefault("dr_state_dict", {})
    if (
        dimension in ("robot_initial", "distractor")
        and env is not None
        and config.get("eval_scene")
    ):
        scene_state = state.get("scene")
        has_table = isinstance(scene_state, dict) and isinstance(scene_state.get("table"), dict)
        if not has_table and "scene" not in env.task.dr.randomizers:
            state.pop("scene", None)
    settings = defaults.get(dimension, {})
    if dimension == "camera":
        state["camera"], params = derive_camera_variant(
            state.get("camera") or {},
            subtype,
            rng,
            camera_uid=head_camera_uid(config),
            ranges=settings,
        )
    elif dimension == "lighting":
        from humano.evals.ood_lighting import derive_lighting_variant

        state["lighting"], params = derive_lighting_variant(
            state.get("lighting") or {}, subtype, rng, **settings.get(subtype, {})
        )
    elif dimension == "sensor_noise":
        from humano.evals.ood_sensor_noise import (
            LIBERO_PLUS_BLUR_SIGMAS,
            derive_sensor_noise_variant,
        )

        params = derive_sensor_noise_variant(
            subtype,
            rng,
            gaussian_range=settings.get("gaussian_range", (0, 20)),
            blur_range=settings.get("blur_range"),
            blur_levels=settings.get("blur_levels", LIBERO_PLUS_BLUR_SIGMAS),
        )
        config["ood_sensor_noise"] = deepcopy(params)
    elif dimension == "language":
        row = tasks[task_key(env_id)]
        instruction = row["test_instructions"][DIMENSIONS["language"].index(subtype)]
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("Language variant requires a nonempty saved instruction")
        state["language"] = instruction
        params = {
            "instruction": instruction,
            "native_description": row["native_description"],
            "source": subtype,
        }
    elif dimension == "robot_initial":
        from humano.evals.ood_robot_initial import derive_robot_initial_variant

        if env is None:
            raise ValueError("Robot initial variants require MuJoCo")
        config, params = derive_robot_initial_variant(
            env,
            config,
            subtype,
            rng,
            max_attempts=settings.get("max_attempts", 100),
            xy_radius_m=settings.get("radius_m", 0.1),
            yaw_limit_deg=settings.get("yaw_deg", 10),
            max_support_gap_m=settings.get("max_support_gap_m", 0.04),
            task_overrides=settings.get("task_overrides", {}),
            indexed_scene=indexed_scene,
        )
    elif dimension == "distractor":
        from humano.evals.ood_distractor import derive_distractor_variant

        if env is None:
            raise ValueError("Distractor variants require MuJoCo")
        count = int(
            rng.integers(
                settings.get("count_range", [1, 5])[0], settings.get("count_range", [1, 5])[1] + 1
            )
        )
        if settings.get("category_uniform", True) is not True:
            raise ValueError("Distractor helper currently supports category_uniform=True only")
        layout_attempts = settings.get("layout_attempts", 1)
        if not isinstance(layout_attempts, int) or layout_attempts < 1:
            raise ValueError("distractor.layout_attempts must be a positive integer")
        for layout_attempt in range(1, layout_attempts + 1):
            try:
                config, params = derive_distractor_variant(
                    env,
                    config,
                    rng,
                    count=count,
                    max_attempts=settings.get("max_attempts", 100),
                    pool_exclusions=list(
                        dict.fromkeys(
                            [
                                *settings.get("exclude_categories", []),
                                *settings.get("pool_exclusions", []),
                            ]
                        )
                    ),
                    placement_regions=settings.get("placement_regions"),
                    require_visible=settings.get("require_visible", True),
                    min_projected_pixels=settings.get("min_projected_pixels", 4),
                    min_projected_short_pixels=settings.get("min_projected_short_pixels", 1),
                    **{
                        key: settings[key]
                        for key in ("near_category_groups", "asset_pool")
                        if key in settings
                    },
                )
                params["layout_attempts"] = layout_attempt
                break
            except ValueError as exc:
                if (
                    not str(exc).startswith("Cannot place count=")
                    or layout_attempt == layout_attempts
                ):
                    raise
    elif dimension == "background":
        config, params = derive_background_variant(config, subtype, rng, settings)
    else:
        raise AssertionError(f"Unhandled OOD dimension: {dimension}")
    return config, params
