"""Opt-in distractors for frozen SONIC initial configs; never runs a policy.

The small-object UID inventory comes from refinement2/distractor_task_allowlists.
Conditional size categories and unapproved furniture/balls are not in this pool.
Near-category groups below are conservative exclusions pending user review.
Only existing table/table2 or ground support is used; default retains SpatialDR's
meaning (table_height supplied by CoreTask, or zero without a tabletop scene).
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

# Audited library IDs, grouped by semantic category, including book/cup aliases.
_POOL = {
    "Carton_Toy_3C923930_Milk": ("Carton_Toy_3C923930_Milk/Carton_Toy_3C923930_Milk_1",),
    "apple": tuple(
        sorted(
            ("apple/apple_001", "apple/apple_4c19ae47", *(f"apple/apple_{i}" for i in range(1, 31)))
        )
    ),
    "banana": ("banana/banana_020",),
    "book": (
        "Book_1788883705_Expectations/Book_1788883705_Expectations",
        "Book_9388144295_Homer/Book_9388144295_Homer",
        "book/book_001",
        "comic_book/comic_book_728d1778",
    ),
    "bottle": ("bottle/bottle_048",),
    "bread": ("bread/bread_006",),
    "cake": ("cake/cake_001",),
    "cup": (
        "Cup_B0BR43SPKJ_Vanilla/Cup_B0BR43SPKJ_Vanilla",
        "Cup_B0CPX832ZP_Floral/Cup_B0CPX832ZP_Floral",
    ),
    "eraser": ("eraser/eraser_001", "eraser/eraser_002", "eraser/eraser_002_large"),
    "marker002": ("marker002/marker002",),
    "peanut": ("peanut/peanut_016", "peanut/peanut_018"),
    "pen": ("pen/pen", "pen/pen_001", "pen/pen_1", "pen/pen_2", "pen/pen_3", "pen/pen_4"),
    "pen_holder001": ("pen_holder001/pen_holder001",),
    "pencil002": ("pencil002/pencil002",),
    "plate": tuple(sorted(f"plate/plate_{i}" for i in range(1, 31))),
    "tray": ("tray/tray_001",),
}

_NEAR_GROUPS = (
    frozenset({"apple", "banana", "peanut"}),
    frozenset({"bread", "cake"}),
    frozenset({"pen", "pencil002", "marker002"}),
    frozenset({"cup", "bottle", "Carton_Toy_3C923930_Milk", "pen_holder001"}),
    frozenset({"plate", "tray"}),
    frozenset({"box", "storage_box", "picnic_basket", "dustbin"}),
    frozenset({"ball", "football"}),
)
_CLEARANCE = 0.005
_TASK_CLEARANCE = 0.08
_CONTACT_TOL = 0.0002
_GROUND_TOO_SMALL = frozenset({"eraser", "marker002", "peanut", "pen", "pencil002"})


def _category(uid: str) -> str:
    category = uid.split(":")[-1].split("/")[0]
    if category.startswith("Book_") or category == "comic_book":
        return "book"
    if category.startswith("Cup_"):
        return "cup"
    return category


def _interaction_objects(task, config):
    # Inspect ALL asset actors, including objects added by task.reset overrides.
    objects = {}
    for role, actor in task.layout.actors.items():
        asset = getattr(actor, "asset", None)
        if asset is not None and not role.startswith("distractor"):
            uid = str(asset.uid)
            objects[(role, uid)] = {
                "role": role,
                "uid": uid,
                "category": _category(uid),
                "name": getattr(asset, "name", uid),
            }
    for role in ("target", "container", "articulated"):
        state = config["dr_state_dict"].get(role)
        if isinstance(state, dict) and state.get("uid"):
            uid = str(state["uid"])
            objects.setdefault(
                (role, uid),
                {
                    "role": role,
                    "uid": uid,
                    "category": _category(uid),
                    "name": state.get("name", uid),
                },
            )
    return list(objects.values())


def _candidate_pool(interactions, exclusions, near_category_groups=None, asset_pool=None):
    excluded = {_category(value) for value in exclusions if "/" not in value}
    excluded.update(obj["category"] for obj in interactions)
    groups = _NEAR_GROUPS if near_category_groups is None else near_category_groups
    for group in groups:
        group = {_category(value) for value in group}
        if group & excluded:
            excluded.update(group)
    uids = {obj["uid"] for obj in interactions} | {value.split(":")[-1] for value in exclusions}
    pool = _POOL if asset_pool is None else asset_pool
    available = {}
    for category, ids in pool.items():
        category = _category(category)
        if category not in excluded:
            available.setdefault(category, []).extend(
                uid for uid in ids if uid not in uids and _category(uid) not in excluded
            )
    return {category: sorted(set(ids)) for category, ids in available.items()}, sorted(excluded)


def _sample_assets(manager, pool, rng, count, used):
    available = {cat: [uid for uid in ids if uid not in used] for cat, ids in pool.items()}
    selected = []
    failures = []
    while len(selected) < count:
        categories = sorted(cat for cat, ids in available.items() if ids)
        if not categories:
            raise ValueError(f"Not enough stable approved assets for count={count}: {failures}")
        category = categories[int(rng.integers(len(categories)))]
        ids = available[category]
        uid = ids.pop(int(rng.integers(len(ids))))
        try:
            asset = manager.load(uid)
        except (FileNotFoundError, ValueError) as exc:
            failures.append({"uid": uid, "reason": str(exc)})
            continue
        poses = np.asarray(asset.stable_poses, dtype=float)
        if poses.ndim != 2 or poses.shape[1] != 7 or not len(poses) or not np.isfinite(poses).all():
            failures.append({"uid": uid, "reason": "missing finite stable poses"})
            continue
        if not np.all(np.linalg.norm(poses[:, 3:], axis=1) > 0):
            failures.append({"uid": uid, "reason": "zero stable quaternion"})
            continue
        selected.append((category, asset, poses[int(rng.integers(len(poses)))].copy()))
    return selected, failures


def _world_bounds(model, data, ids):
    rotations = data.geom_xmat[ids].reshape(-1, 3, 3)
    aabb = model.geom_aabb[ids]
    centers = data.geom_xpos[ids] + np.einsum("nij,nj->ni", rotations, aabb[:, :3])
    half = np.einsum("nij,nj->ni", np.abs(rotations), aabb[:, 3:])
    return centers - half, centers + half


def _geom_ids(model, body_id):
    bodies = {int(body_id)}
    for i in range(int(body_id) + 1, model.nbody):
        if int(model.body_parentid[i]) in bodies:
            bodies.add(i)
    return np.flatnonzero(np.isin(model.geom_bodyid, list(bodies)))


def _indexed_scene(config):
    selection = config.get("eval_scene")
    if not selection:
        return None
    from humano.scenes.indexed import (
        align_indexed_background_scene,
        load_indexed_background_scene,
    )

    if selection.get("schema") == "humano.eval-scene-selection.v3":
        from humano.evals.eval_scene_protocol import resolve_eval_scene_selection

        selection = resolve_eval_scene_selection(
            selection, repository_root=Path(__file__).resolve().parents[3]
        )
    scene = load_indexed_background_scene(selection["scene_index"], selection["scene_uid"])
    placement = selection["placement"]
    return align_indexed_background_scene(
        scene,
        translation_xy=tuple(placement["translation_xy"]),
        yaw_rad=float(placement["yaw_rad"]),
        hidden_anchor_prim_path=placement.get("anchor_prim_path"),
        hide_all_background_assets=placement.get("strategy") == "all_hide_fallback",
    )


def _surface(sim, spatial_cfg, instance, index, scene, placement_regions):
    import mujoco

    mapping = spatial_cfg.get("obj_surface_map") or {}
    entry = mapping.get(instance) or mapping.get("distractor", "table")
    name = entry[index % len(entry)] if isinstance(entry, list) and entry else entry
    heights = {"default": 0.0}
    for table_name in ("table", "table2"):
        table = getattr(getattr(sim.task.layout, "scene", None), table_name, None)
        if table is not None:
            heights[table_name] = float(table.pose.position[2] + 0.5 * table.size[2])
    heights["default"] = heights.get("table", 0.0)
    height = heights.get(name, heights["default"])
    # Resolve support at the ORIGINAL mapped height, never reinterpret default.
    support = (
        name
        if name in ("table", "table2") and name in heights
        else ("table" if "table" in heights else "ground")
    )
    model, data = sim.mjModel, sim.mjData
    if model.opt.disableflags & int(mujoco.mjtDisableBit.mjDSBL_NATIVECCD):
        raise ValueError("Accurate signed geom distance requires native CCD")
    geom = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, f"{support}_geom" if support != "ground" else "ground"
    )
    if geom < 0:
        raise ValueError(f"No existing support for {instance}: surface={name}, height={height}")
    rotation = data.geom_xmat[geom].reshape(3, 3)
    if not np.allclose(rotation[:, 2], [0, 0, 1], atol=1e-6):
        raise ValueError(f"Support {support} is not horizontal")
    top = float(data.geom_xpos[geom, 2] + (model.geom_size[geom, 2] if support != "ground" else 0))
    if abs(top - height) > _CONTACT_TOL:
        raise ValueError(
            f"Mapped surface {name} height={height} disagrees with actual {support} top={top}"
        )
    if support != "ground":
        corners = np.array([[x, y, 0] for x in (-1, 1) for y in (-1, 1)], dtype=float)
        corners[:, :2] *= model.geom_size[geom, :2]
        xy = (corners @ rotation.T + data.geom_xpos[geom])[:, :2]
        low, high = xy.min(axis=0), xy.max(axis=0)
    elif scene is not None:
        low = np.asarray(scene.bounds.min_xy) + scene.bounds.margin_m
        high = np.asarray(scene.bounds.max_xy) - scene.bounds.margin_m
    else:
        # Bound ground sampling by the existing robot/task scene footprint.
        distractor_ids = {
            int(body.id) for key, body in sim.mj_objects.items() if key.startswith("distractor")
        }
        ids = np.array(
            [
                i
                for i in range(model.ngeom)
                if model.geom_type[i] != mujoco.mjtGeom.mjGEOM_PLANE
                and int(model.geom_bodyid[i]) not in distractor_ids
            ]
        )
        if not len(ids):
            raise ValueError(
                "Ground placement needs a finite existing scene footprint or placement_regions"
            )
        mins, maxs = _world_bounds(model, data, ids)
        low, high = mins[:, :2].min(axis=0) - 0.5, maxs[:, :2].max(axis=0) + 0.5
    support_low, support_high = low.copy(), high.copy()
    overrides = placement_regions or {}
    region = overrides.get(instance, overrides.get(name, overrides.get(support)))
    source = "override" if region is not None else "existing_support"
    if region is None:
        region = spatial_cfg.get("distractors_region")
        if isinstance(region, list):
            region = region[index % len(region)] if region else None
        if region and np.all(np.asarray(region["high"])[:2] > np.asarray(region["low"])[:2]):
            source = "configured_region"
        else:
            region = None
    if region is not None:
        lo, hi = np.asarray(region["low"], dtype=float), np.asarray(region["high"], dtype=float)
        if (
            lo.shape != (2,)
            or hi.shape != (2,)
            or not np.isfinite([lo, hi]).all()
            or np.any(hi < lo)
        ):
            raise ValueError(f"Invalid XY placement region for {instance}")
        if source == "override" and support == "ground" and scene is None:
            low, high = lo, hi
        else:
            low, high = np.maximum(low, lo), np.minimum(high, hi)
    if np.any(high < low):
        raise ValueError(f"No available XY region for {instance} on {support}")
    return {
        "original_surface": name,
        "resolved_height_m": height,
        "support": support,
        "support_geom": int(geom),
        "low": low.tolist(),
        "high": high.tolist(),
        "support_low": support_low.tolist(),
        "support_high": support_high.tolist(),
        "region_source": source,
    }


def _task_anchors(sim, surface):
    """Use the task's existing actors, rather than a global table/room region."""
    model, data = sim.mjModel, sim.mjData
    anchors = []
    for role in sim.task.layout.actors:
        if role.startswith("distractor") or role == "robot":
            continue
        body = sim.mj_objects.get(role)
        if body is None:
            continue
        xy = np.asarray(data.xpos[body.id, :2], dtype=float)
        if np.any(xy < surface["support_low"]) or np.any(xy > surface["support_high"]):
            continue
        ids = _geom_ids(model, body.id)
        if not len(ids):
            continue
        mins, maxs = _world_bounds(model, data, ids)
        if mins[:, 2].min() > surface["resolved_height_m"] + 0.35:
            continue
        anchors.append({"role": role, "xy": xy.tolist()})
    robot_xy = None
    if surface["support"] == "ground":
        robot_body = sim.mj_objects.get("robot")
        if robot_body is not None:
            robot_xy = np.asarray(data.xpos[robot_body.id, :2], dtype=float)
        else:
            try:
                robot_xy = np.asarray(data.body("robot").xpos[:2], dtype=float)
            except KeyError:
                robot_actor = sim.task.layout.actors.get("robot")
                if robot_actor is not None:
                    robot_xy = np.asarray(robot_actor.pose.position[:2], dtype=float)
        if (
            robot_xy is not None
            and np.all(robot_xy >= surface["support_low"])
            and np.all(robot_xy <= surface["support_high"])
        ):
            anchors.append({"role": "robot", "xy": robot_xy.tolist()})
    return anchors


def _xy_clearance(a_low, a_high, b_low, b_high):
    gap = np.maximum(np.maximum(b_low - a_high, a_low - b_high), 0)
    return float(np.linalg.norm(gap))


def _segment_clearance(point, start, end):
    direction = end - start
    length_sq = float(direction @ direction)
    if length_sq < 1e-8:
        return float(np.linalg.norm(point - start))
    fraction = np.clip(float((point - start) @ direction) / length_sq, 0, 1)
    return float(np.linalg.norm(point - (start + fraction * direction)))


def _protected_geometry(sim, instance):
    protected = {}
    for role in sim.task.layout.actors:
        if role == instance or role == "robot":
            continue
        body = sim.mj_objects.get(role)
        if body is not None:
            protected[role] = _geom_ids(sim.mjModel, body.id)
    return protected


def _is_added_distractor(role, surface):
    return role in set(surface.get("added_distractor_instances", ()))


def _set_pose(sim, instance, pose):
    import mujoco

    body = sim.mj_objects[instance]
    joint = int(sim.mjModel.body_jntadr[body.id])
    if joint < 0 or sim.mjModel.jnt_type[joint] != mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError(f"Distractor {instance} needs a free joint")
    address = int(sim.mjModel.jnt_qposadr[joint])
    sim.mjData.qpos[address : address + 7] = pose
    mujoco.mj_forward(sim.mjModel, sim.mjData)


def _check_placement(sim, instance, surface, scene):
    import mujoco

    model, data = sim.mjModel, sim.mjData
    ids = _geom_ids(model, sim.mj_objects[instance].id)
    mins, maxs = _world_bounds(model, data, ids)
    geom = surface["support_geom"]
    rotation = data.geom_xmat[geom].reshape(3, 3)
    corners = np.array(
        [
            [x, y, 0]
            for x in (mins[:, 0].min(), maxs[:, 0].max())
            for y in (mins[:, 1].min(), maxs[:, 1].max())
        ]
    )
    if surface["support"] != "ground":
        local = (corners - data.geom_xpos[geom]) @ rotation
        if np.any(np.abs(local[:, :2]) > model.geom_size[geom, :2] - _CLEARANCE):
            return "footprint outside support"
    for role, other_ids in _protected_geometry(sim, instance).items():
        if not len(other_ids):
            continue
        other_mins, other_maxs = _world_bounds(model, data, other_ids)
        if other_mins[:, 2].min() > surface["resolved_height_m"] + 0.5:
            continue  # Unplaced distractor placeholder, not an occupied slot.
        for other_id, low, high in zip(other_ids, other_mins, other_maxs, strict=True):
            if (
                not _is_added_distractor(instance, surface)
                or not _is_added_distractor(role, surface)
            ) and (
                _xy_clearance(mins[:, :2].min(axis=0), maxs[:, :2].max(axis=0), low[:2], high[:2])
                < _TASK_CLEARANCE
            ):
                return f"task/distractor clearance role={role} geom={model.geom(int(other_id)).name} below {_TASK_CLEARANCE}m"
    distance_to_support = []
    for i in ids:
        for j in range(model.ngeom):
            if j in ids or not (model.geom_contype[j] or model.geom_conaffinity[j]):
                continue
            distance = mujoco.mj_geomDistance(model, data, int(i), j, 0.02, None)
            if j == geom:
                distance_to_support.append(distance)
                if distance < -_CONTACT_TOL:
                    return "support penetration"
            elif distance < _CLEARANCE:
                return f"collision/clearance geom={model.geom(j).name} distance={distance}"
    if not distance_to_support or abs(min(distance_to_support)) > _CONTACT_TOL:
        return "no support contact"
    contacts = [
        c
        for c in data.contact
        if (int(c.geom1) in ids and c.geom2 == geom) or (int(c.geom2) in ids and c.geom1 == geom)
    ]
    if not any(
        abs(c.frame[2]) > 0.9
        and abs(c.pos[2] - surface["resolved_height_m"]) < _CONTACT_TOL
        and abs(c.dist) <= _CONTACT_TOL
        for c in contacts
    ):
        return "no legal MuJoCo upward support contact"
    corridor = surface.get("reserved_corridor")
    if corridor:
        center = (mins[:, :2].min(axis=0) + maxs[:, :2].max(axis=0)) / 2
        half_diagonal = float(np.linalg.norm(maxs[:, :2].max(axis=0) - mins[:, :2].min(axis=0))) / 2
        for segment in corridor:
            if (
                _segment_clearance(center, np.asarray(segment[0]), np.asarray(segment[1]))
                < _TASK_CLEARANCE + half_diagonal
            ):
                return "reserved task path clearance"
    if scene is not None:
        # Same bounds/immutable-obstacle semantics as the offline render guard,
        # without importing its renderer/Parquet CLI dependencies.
        margin = scene.bounds.margin_m
        if np.any(mins[:, :2] < np.asarray(scene.bounds.min_xy) + margin) or np.any(
            maxs[:, :2] > np.asarray(scene.bounds.max_xy) - margin
        ):
            return "indexed scene bounds violation"
        for obstacle in scene.immutable_obstacles:
            overlaps = np.all(maxs >= np.asarray(obstacle.min_xyz) - margin, axis=1) & np.all(
                mins <= np.asarray(obstacle.max_xyz) + margin, axis=1
            )
            if np.any(overlaps):
                return f"immutable wall/obstacle overlap: {obstacle.prim_path}"
    return None


def _visibility(sim, instance, min_projected_pixels, min_projected_short_pixels=1):
    """Project compiled collision bounds into actual head cameras; no raycast."""
    model, data = sim.mjModel, sim.mjData
    mins, maxs = _world_bounds(model, data, _geom_ids(model, sim.mj_objects[instance].id))
    low, high = mins.min(axis=0), maxs.max(axis=0)
    center = (low + high) / 2
    corners = np.array(
        [[x, y, z] for x in (low[0], high[0]) for y in (low[1], high[1]) for z in (low[2], high[2])]
    )
    near, far = (
        float(model.vis.map.znear * model.stat.extent),
        float(model.vis.map.zfar * model.stat.extent),
    )
    evidence = {
        "visible": False,
        "bbox_center_world": center.tolist(),
        "occlusion_checked": False,
        "min_projected_pixels": min_projected_pixels,
        "min_projected_short_pixels": min_projected_short_pixels,
        "size_metric": "max_bbox_side_pixels",
        "clipped_to_image": True,
        "cameras": [],
    }
    for name, camera in sim.task.layout.cameras.items():
        if camera.mount != "eye_in_head":
            continue
        camera_id = model.camera(name).id
        width, height = camera.resolution
        rotation = data.cam_xmat[camera_id].reshape(3, 3)
        local = (center - data.cam_xpos[camera_id]) @ rotation
        depth = float(-local[2])
        fovy = float(model.cam_fovy[camera_id])
        focal = height / (2 * np.tan(np.deg2rad(fovy) / 2))
        result = {
            "camera": name,
            "resolution": [int(width), int(height)],
            "fovy_deg": fovy,
            "near_m": near,
            "far_m": far,
            "depth_m": depth,
            "visible": False,
        }
        if near < depth < far:
            pixel = np.array(
                [width / 2 + focal * local[0] / depth, height / 2 - focal * local[1] / depth]
            )
            camera_corners = (corners - data.cam_xpos[camera_id]) @ rotation
            valid = -camera_corners[:, 2] > near
            camera_corners = camera_corners[valid]
            depths = -camera_corners[:, 2]
            projected = np.column_stack(
                (
                    width / 2 + focal * camera_corners[:, 0] / depths,
                    height / 2 - focal * camera_corners[:, 1] / depths,
                )
            )
            span = np.minimum(projected.max(axis=0), [width, height]) - np.maximum(
                projected.min(axis=0), [0, 0]
            )
            result.update(
                center_pixel=pixel.tolist(),
                projected_bbox_pixels=np.maximum(span, 0).tolist(),
                visible=bool(
                    np.all(pixel >= 0)
                    and np.all(pixel < [width, height])
                    and np.max(span) >= min_projected_pixels
                    and np.min(span) >= min_projected_short_pixels
                ),
            )
        evidence["cameras"].append(result)
        evidence["visible"] |= result["visible"]
    if not evidence["cameras"]:
        raise ValueError(
            "require_visible needs an actual MuJoCo eye_in_head camera and its image resolution"
        )
    return evidence


def _replay(env, config):
    import mujoco

    if "distractors" not in config.get("dr_cfgs", {}):
        # An opt-in variant can leave its randomizer registered after a failed
        # placement; the base state has no matching object shader entries.
        env.task.dr.randomizers.pop("distractors", None)
    env.task.reset(options={"state_dict": deepcopy(config)})
    env.mujoco.update_layout(sonic_config=env.sonic_config)
    if config.get("ood_physics_state") is not None:
        from humano.evals.ood_physics import restore_physics

        restore_physics(env.mujoco.mjData, config["ood_physics_state"])
    mujoco.mj_forward(env.mujoco.mjModel, env.mujoco.mjData)
    for name in ("mjSpec", "mjModel", "mjData"):
        setattr(env, name, getattr(env.mujoco, name))


def _preserve_base_physics(old_model, old_data, sim):
    """New distractors shift joint addresses; preserve existing named state."""
    import mujoco

    model, data = sim.mjModel, sim.mjData
    for old_joint in range(old_model.njnt):
        body_id = int(old_model.jnt_bodyid[old_joint])
        body = model.body(old_model.body(body_id).name).id
        joint = int(model.body_jntadr[body]) + old_joint - int(old_model.body_jntadr[body_id])
        kind = old_model.jnt_type[old_joint]
        if model.jnt_type[joint] != kind:
            raise ValueError(f"Joint type changed in body {old_model.body(body_id).name}")
        nq = (
            7
            if kind == mujoco.mjtJoint.mjJNT_FREE
            else 4
            if kind == mujoco.mjtJoint.mjJNT_BALL
            else 1
        )
        nv = (
            6
            if kind == mujoco.mjtJoint.mjJNT_FREE
            else 3
            if kind == mujoco.mjtJoint.mjJNT_BALL
            else 1
        )
        for field, address, length in (("qpos", "jnt_qposadr", nq), ("qvel", "jnt_dofadr", nv)):
            source = int(getattr(old_model, address)[old_joint])
            target = int(getattr(model, address)[joint])
            getattr(data, field)[target : target + length] = getattr(old_data, field)[
                source : source + length
            ]
    # Distractors add no actuators, equalities, or mocap bodies; those arrays
    # retain their ordering while qpos/qvel addresses above shift.
    for field in ("ctrl", "act", "eq_active", "mocap_pos", "mocap_quat"):
        getattr(data, field)[:] = getattr(old_data, field)
    data.time = old_data.time
    mujoco.mj_forward(model, data)


def derive_distractor_variant(
    env: Any,
    base_config: dict[str, Any],
    rng: np.random.Generator,
    count: int,
    max_attempts: int = 100,
    *,
    pool_exclusions: list[str] | None = None,
    placement_regions: dict[str, dict[str, list[float]]] | None = None,
    near_category_groups: list[list[str]] | None = None,
    asset_pool: dict[str, list[str]] | None = None,
    require_visible: bool = True,
    min_projected_pixels: float = 4,
    min_projected_short_pixels: float = 1,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Add exactly count approved assets to a frozen SONIC initialization config.

    env is the unwrapped SONIC environment (task, mujoco, sonic_config). It is
    left at the verified variant initialization; failure replays base_config.
    No env.reset/step, Isaac rendering or policy inference is invoked.
    pool_exclusions contains additional semantic categories or exact UIDs.
    near_category_groups overrides the provisional groups; [] disables them.
    asset_pool overrides category->UID lists, including user-selected larger
    assets; the same stable-pose and native placement checks still apply.
    require_visible checks collision-bbox center projection in a head camera,
    with minimum clipped bbox long/short sides in pixels; this does not prove no occlusion.
    placement_regions maps original surface names or distractor_N to {low,high}
    world XY, clipped to the existing support. All parameters are recorded for
    leader review; near groups are provisional conservative exclusions.
    Replay with full state (no dr_level): spatial stores distractor_N for the
    manifest AND asset UID because existing SpatialDR reads the UID pose.
    This proves static contact/clearance, not dynamic stability or policy success.
    """
    if not isinstance(count, int) or isinstance(count, bool) or count < 0:
        raise ValueError("count must be a nonnegative integer")
    if not isinstance(max_attempts, int) or max_attempts < 1:
        raise ValueError("max_attempts must be a positive integer")
    if (
        not isinstance(require_visible, bool)
        or not np.isfinite(min_projected_pixels)
        or min_projected_pixels <= 0
        or not np.isfinite(min_projected_short_pixels)
        or min_projected_short_pixels <= 0
    ):
        raise ValueError(
            "require_visible must be bool and projected pixel thresholds finite and positive"
        )
    if base_config.get("uid") != env.task.uid:
        raise ValueError("base_config task UID does not match env")
    if not base_config.get("dr_state_dict", {}).get("spatial"):
        raise ValueError("A frozen spatial initialization state is required")
    if not base_config["dr_cfgs"]["spatial"].get("rand_stable_pose", True):
        raise ValueError("SpatialDR must restore stable object poses for replay")
    state = deepcopy(base_config)
    if count:
        # CoreTask installs the opt-in randomizer before loading this state.
        # Dolly tasks have neither its config nor a distractor state by default.
        state["ood_distractor"] = True
        state["dr_state_dict"].setdefault("distractors", {})
        state["dr_cfgs"].setdefault(
            "distractors",
            {
                "randmizer_class": "humano.dr.distractor.DistractorDR",
                "res_id": "objects",
                "number_of_distractors": 0,
                "allow_duplicates": False,
                "include": None,
                "exclude": None,
            },
        )
    params = {
        "count": count,
        "max_attempts": max_attempts,
        "pool_exclusions": list(pool_exclusions or []),
        "placement_regions": deepcopy(placement_regions),
        "require_visible": require_visible,
        "min_projected_pixels": float(min_projected_pixels),
        "min_projected_short_pixels": float(min_projected_short_pixels),
        "visibility_metric": "max_bbox_side_pixels",
        "near_category_review_pending": True,
        "near_category_exclusion_groups": [
            sorted(g)
            for g in (_NEAR_GROUPS if near_category_groups is None else near_category_groups)
        ],
        "asset_pool": {
            category: list(ids)
            for category, ids in (_POOL if asset_pool is None else asset_pool).items()
        },
        "clearance_m": _CLEARANCE,
        "task_and_distractor_xy_clearance_m": _TASK_CLEARANCE,
        "added_distractor_pair_clearance": "mujoco_geometry_only_no_fixed_xy_gap",
        "ground_sampling_anchors": "task_objects_plus_robot_when_available",
        "contact_tolerance_m": _CONTACT_TOL,
        "distractors": [],
        "collision_attempts": 0,
        "validation": "static_mujoco_initialization_and_replay",
    }
    if not count:
        return state, params
    try:
        _replay(env, state)
        old_model, old_data = env.mujoco.mjModel, env.mujoco.mjData
        baseline_bodies = {
            env.mujoco.mjModel.body(i).name: (
                env.mujoco.mjData.xpos[i].copy(),
                env.mujoco.mjData.xquat[i].copy(),
            )
            for i in range(env.mujoco.mjModel.nbody)
        }
        existing = state["dr_state_dict"].get("distractors") or {}
        prior_added = set(state.get("metadata", {}).get("ood_added_distractor_instances", ()))
        added_instances = sorted(
            [
                *prior_added,
                *[f"distractor_{index}" for index in range(len(existing), len(existing) + count)],
            ]
        )
        interactions = _interaction_objects(env.task, base_config)
        scene = _indexed_scene(base_config)
        ground_support = any(
            _surface(
                env.mujoco,
                state["dr_cfgs"]["spatial"],
                f"distractor_{index}",
                index,
                scene,
                placement_regions,
            )["support"]
            == "ground"
            for index in range(len(existing), len(existing) + count)
        )
        support_exclusions = _GROUND_TOO_SMALL if ground_support and asset_pool is None else ()
        pool, excluded = _candidate_pool(
            interactions,
            [*(pool_exclusions or []), *support_exclusions],
            near_category_groups,
            asset_pool,
        )
        params.update(
            interaction_objects=interactions,
            excluded_categories=excluded,
            ground_size_excluded_categories=sorted(support_exclusions),
            candidate_categories=sorted(cat for cat, ids in pool.items() if ids),
        )
        randomizer = env.task.dr.get_randomizer("distractors")
        if randomizer is None:
            raise ValueError(
                "Task has no distractors randomizer; cannot replay without modifying core"
            )
        selected, failures = _sample_assets(
            randomizer.asset_manager, pool, rng, count, {str(a["uid"]) for a in existing.values()}
        )
        params["asset_rejections"] = failures
        new_assets = dict(existing)
        spatial = state["dr_state_dict"]["spatial"]
        for index, (_, asset, pose) in enumerate(selected, start=len(existing)):
            instance = f"distractor_{index}"
            new_assets[instance] = asset.to_dict()
            # Separate placeholders until every body has been compiled.
            placeholder = {
                "position": [0.0, 0.0, 10.0 + 2 * index],
                "quaternion": pose[3:].tolist(),
            }
            spatial[str(asset.uid)] = deepcopy(placeholder)
            spatial[instance] = deepcopy(placeholder)
        state["dr_state_dict"]["distractors"] = new_assets
        material = state["dr_state_dict"].get("material")
        if isinstance(material, dict):
            # MaterialDR replays shader entries by ObjectActor insertion order:
            # container, target, existing distractors, then the added distractors.
            # Preserve every original entry; append fixed shaders only for D.
            shader = {
                "reflection_roughness_constant": 0.5,
                "metallic_constant": 0.0,
                "specular_level": 0.0,
            }
            material["object_shader_params"].extend(deepcopy(shader) for _ in selected)
            params["new_distractor_shader_params"] = [deepcopy(shader) for _ in selected]
        cfg = state["dr_cfgs"]["distractors"]
        cfg.update(
            include=[a["uid"] for a in new_assets.values()], number_of_distractors=len(new_assets)
        )
        state.pop("ood_physics_state", None)  # Rebuild the snapshot for the enlarged model.
        _replay(env, state)
        _preserve_base_physics(old_model, old_data, env.mujoco)
        sim = env.mujoco
        for index, (category, asset, stable) in enumerate(selected, start=len(existing)):
            instance = f"distractor_{index}"
            surface = _surface(
                sim, state["dr_cfgs"]["spatial"], instance, index, scene, placement_regions
            )
            surface["added_distractor_instances"] = added_instances
            anchors = _task_anchors(sim, surface)
            if not anchors:
                anchors = [
                    {
                        "role": "support_center_fallback",
                        "xy": np.mean([surface["low"], surface["high"]], axis=0).tolist(),
                    }
                ]
            surface["task_anchors"] = anchors
            if anchors:
                robot = sim.mj_objects.get("robot")
                if robot is not None:
                    robot_xy = sim.mjData.xpos[robot.id, :2].tolist()
                else:
                    robot_actor = sim.task.layout.actors.get("robot")
                    robot_xy = robot_actor.pose.position[:2] if robot_actor is not None else None
                corridor_anchors = [anchor for anchor in anchors if anchor["role"] != "robot"]
                corridors = []
                if robot_xy is not None:
                    corridors.extend([robot_xy, anchor["xy"]] for anchor in corridor_anchors)
                corridors.extend(
                    [left["xy"], right["xy"]]
                    for left, right in zip(corridor_anchors, corridor_anchors[1:], strict=False)
                )
                surface["reserved_corridor"] = corridors
            reason = "untried"
            visibility = {"checked": False}
            for _attempt in range(1, max_attempts + 1):
                params["collision_attempts"] += 1
                pose = stable.copy()
                pose[3:] /= np.linalg.norm(pose[3:])
                angle_region = state["dr_cfgs"]["spatial"].get("distractor_rotate_z")
                if angle_region is not None:
                    from transforms3d.quaternions import qmult

                    angle = float(rng.uniform(angle_region["low"], angle_region["high"]))
                    pose[3:] = qmult([np.cos(angle / 2), 0, 0, np.sin(angle / 2)], pose[3:])
                if surface["region_source"] == "existing_support" and anchors:
                    anchor = np.asarray(anchors[int(rng.integers(len(anchors)))]["xy"])
                    # First try task-near slots, then use the remaining support if needed.
                    initial_radius = 0.35 if surface["support"] != "ground" else 0.6
                    max_radius = (
                        float(np.linalg.norm(np.asarray(surface["high"]) - surface["low"]))
                        if surface["support"] != "ground"
                        else 1.0
                    )
                    radius = initial_radius + max(0.0, max_radius - initial_radius) * min(
                        1.0, 2 * (_attempt - 1) / max_attempts
                    )
                    low = np.maximum(surface["low"], anchor - radius)
                    high = np.minimum(surface["high"], anchor + radius)
                    if np.any(high < low):
                        reason = "no task-near slot on support"
                        continue
                    pose[:2] = rng.uniform(low, high)
                    if np.linalg.norm(pose[:2] - anchor) > radius:
                        reason = "outside task-near radius"
                        continue
                else:
                    pose[:2] = rng.uniform(surface["low"], surface["high"])
                pose[2] += surface["resolved_height_m"]
                _set_pose(sim, instance, pose)
                import mujoco

                ids = _geom_ids(sim.mjModel, sim.mj_objects[instance].id)
                gap = min(
                    mujoco.mj_geomDistance(
                        sim.mjModel, sim.mjData, int(i), surface["support_geom"], 0.02, None
                    )
                    for i in ids
                )
                # Align the annotated stable pose to the actual compiled geoms.
                # A micron of penetration ensures an active contact at init.
                if abs(gap) < 0.01:
                    pose[2] -= gap + 1e-6
                    _set_pose(sim, instance, pose)
                reason = _check_placement(sim, instance, surface, scene)
                if reason is None and require_visible:
                    visibility = _visibility(
                        sim, instance, min_projected_pixels, min_projected_short_pixels
                    )
                    if not visibility["visible"]:
                        reason = f"not visible in head camera at long/short thresholds {min_projected_pixels}/{min_projected_short_pixels} pixels"
                if reason is None:
                    if surface["region_source"] == "existing_support":
                        surface["sampled_anchor_xy"] = anchor.tolist()
                        surface["sampled_radius_m"] = float(radius)
                    break
            else:
                raise ValueError(
                    f"Cannot place count={count}: {instance} uid={asset.uid} after {max_attempts} attempts: {reason}"
                )
            body = sim.mj_objects[instance]
            actual = {"position": body.xpos.tolist(), "quaternion": body.xquat.tolist()}
            spatial[instance] = deepcopy(actual)
            spatial[str(asset.uid)] = deepcopy(actual)
            actor = env.task.layout.actors[instance]
            actor.pose.position, actor.pose.quaternion = actual["position"], actual["quaternion"]
            params["distractors"].append(
                {
                    "instance": instance,
                    "uid": str(asset.uid),
                    "category": category,
                    "name": getattr(asset, "name", str(asset.uid)),
                    "worldpose": deepcopy(actual),
                    "surface": surface,
                    "collision_attempts": _attempt,
                    "simulator_name": actor.simulation_name,
                    "visibility": deepcopy(visibility),
                }
            )
            state.setdefault("layout", {}).setdefault("actors", {})[instance] = actor.to_dict()
        from humano.evals.ood_physics import capture_physics

        state["ood_physics_state"] = capture_physics(sim.mjData)
        if added_instances:
            state.setdefault("metadata", {})["ood_added_distractor_instances"] = added_instances
        _replay(env, state)
        for name, (position, quaternion) in baseline_bodies.items():
            body = env.mujoco.mjData.body(name)
            if not np.allclose(body.xpos, position, atol=1e-7, rtol=0) or not np.allclose(
                body.xquat, quaternion, atol=1e-7, rtol=0
            ):
                raise ValueError(f"Non-distractor body changed on replay: {name}")
        for record in params["distractors"]:
            body = env.mujoco.mj_objects[record["instance"]]
            expected = record["worldpose"]
            if not np.allclose(
                body.xpos, expected["position"], atol=1e-7, rtol=0
            ) or not np.allclose(body.xquat, expected["quaternion"], atol=1e-7, rtol=0):
                raise ValueError(f"Distractor pose changed on replay: {record['instance']}")
            reason = _check_placement(env.mujoco, record["instance"], record["surface"], scene)
            if reason is not None:
                raise ValueError(f"Distractor invalid on replay: {record['instance']}: {reason}")
            if require_visible:
                record["visibility"] = _visibility(
                    env.mujoco, record["instance"], min_projected_pixels, min_projected_short_pixels
                )
                if not record["visibility"]["visible"]:
                    raise ValueError(f"Distractor not visible on replay: {record['instance']}")
        verified_physics = capture_physics(env.mujoco.mjData)
        if verified_physics != state["ood_physics_state"]:
            raise ValueError("Full physics state changed on distractor replay")
        state["ood_physics_state"] = verified_physics
        params.update(full_physics_replay_verified=True, steady_sonic_handoff_verified=False)
        return state, params
    except Exception:
        # Do not leave a partial count in the caller's environment on failure.
        _replay(env, base_config)
        raise
