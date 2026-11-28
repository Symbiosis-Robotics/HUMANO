"""Frozen, collision-validated G1 initial translation or yaw variants.

The caller owns environment creation and indexed-scene installation. This helper
resets to base once, then validates scratch MjData without policy execution or
settling. The live environment is left at base, even on sampling failure.

The baseline is AFTER env.reset, including any physics step performed by reset.
The returned config stores ood_physics_state via capture_physics for environment
reset to restore before stepping. Persist the returned config explicitly.
Replaying spatial pose alone can change the world's post-reset physics state.
Long-term stability and task goal reachability are not certified here.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from typing import Any

import numpy as np

from humano.evals.ood_physics import capture_physics

XY_RADIUS_M = 0.1
YAW_LIMIT_RAD = float(np.deg2rad(10.0))
MAX_FLOOR_PENETRATION_M = 0.005
MAX_SUPPORT_GAP_M = 0.04  # Real Inspire prestep sole colliders are ~36 mm up.
_CONTACT_EPS_M = 1e-6
_FOOT_BODIES = frozenset({"left_ankle_roll_link", "right_ankle_roll_link"})


class RobotInitialSamplingError(ValueError):
    """No candidate passed; counts are available for generation reports."""

    def __init__(self, attempt_count: int, rejections: dict[str, int], last_evidence: dict):
        self.attempt_count = attempt_count
        self.rejections = rejections
        self.last_evidence = last_evidence
        super().__init__(f"No valid robot initial pose in {attempt_count} attempts: {rejections}")


def _sample_delta(
    subtype: str,
    rng: np.random.Generator,
    xy_radius_m: float = XY_RADIUS_M,
    yaw_limit_rad: float = YAW_LIMIT_RAD,
) -> tuple[float, float, float]:
    if subtype == "translation":
        radius = xy_radius_m * np.sqrt(rng.uniform(0.0, 1.0))
        angle = rng.uniform(0.0, 2.0 * np.pi)
        return float(radius * np.cos(angle)), float(radius * np.sin(angle)), 0.0
    if subtype == "yaw":
        return 0.0, 0.0, float(rng.uniform(-yaw_limit_rad, yaw_limit_rad))
    raise ValueError("subtype must be 'translation' or 'yaw'; joint jitter is unsupported")


def _yaw_quaternion(quaternion: np.ndarray, yaw: float) -> np.ndarray:
    # World-axis yaw left-multiplies the wxyz orientation; preserve roll/pitch.
    w, x, y, z = quaternion
    c, s = np.cos(yaw / 2.0), np.sin(yaw / 2.0)
    return np.array([c * w - s * z, c * x - s * y, c * y + s * x, c * z + s * w])


def _contact_depths(data) -> dict[tuple[int, int], float]:
    depths = {}
    for contact in data.contact[: data.ncon]:
        if contact.dist <= _CONTACT_EPS_M:
            pair = tuple(sorted((int(contact.geom1), int(contact.geom2))))
            depths[pair] = min(depths.get(pair, 0.0), float(contact.dist))
    return depths


def _foot_floor_distances(model, data, robot_bodies, tolerance) -> dict:
    import mujoco

    floors = [
        g
        for g in range(model.ngeom)
        if model.geom(g).name == "ground"
        and model.geom_bodyid[g] == 0
        and model.geom_type[g] == 0
        and data.geom_xmat[g].reshape(3, 3)[2, 2] > 0.99
    ]
    feet = [
        g
        for g in range(model.ngeom)
        if int(model.geom_bodyid[g]) in robot_bodies
        and model.body(int(model.geom_bodyid[g])).name in _FOOT_BODIES
        and (model.geom_contype[g] or model.geom_conaffinity[g])
    ]
    # Query ABOVE the acceptance tolerance: distmax itself is returned for far geoms.
    cutoff = max(0.1, tolerance + 0.01)
    return {
        (floor, foot): float(mujoco.mj_geomDistance(model, data, floor, foot, cutoff, None))
        for floor in floors
        for foot in feet
    }


def _validate_contacts(
    model,
    data,
    robot_bodies,
    base_contacts,
    max_support_gap_m=MAX_SUPPORT_GAP_M,
    base_foot_floor_distances=None,
) -> tuple[str | None, dict]:
    foot_contacts = 0
    self_contacts = 0
    rejected = []
    for contact in data.contact[: data.ncon]:
        if contact.dist > _CONTACT_EPS_M:
            continue  # MuJoCo margins can generate separated contact records.
        g1, g2 = int(contact.geom1), int(contact.geom2)
        b1, b2 = int(model.geom_bodyid[g1]), int(model.geom_bodyid[g2])
        r1, r2 = b1 in robot_bodies, b2 in robot_bodies
        if not (r1 or r2):
            continue
        reason = None
        if r1 and r2:
            pair = tuple(sorted((g1, g2)))
            if pair in base_contacts and contact.dist >= base_contacts[pair] - _CONTACT_EPS_M:
                self_contacts += 1
            else:
                reason = "new_or_deeper_robot_self_contact"
        else:
            other_geom, other_body = (g2, b2) if r1 else (g1, b1)
            robot_body = b1 if r1 else b2
            is_floor = (
                other_body == 0
                and model.geom(other_geom).name == "ground"
                and int(model.geom_type[other_geom]) == 0  # mjGEOM_PLANE
                and data.geom_xmat[other_geom].reshape(3, 3)[2, 2] > 0.99
            )
            if not is_floor:
                reason = "robot_external_contact"
            elif model.body(robot_body).name not in _FOOT_BODIES:
                reason = "nonfoot_ground_contact"
            elif contact.dist < -MAX_FLOOR_PENETRATION_M:
                reason = "deep_foot_ground_penetration"
            elif abs(float(contact.frame[2])) < 0.9:
                reason = "nonvertical_foot_ground_contact"
            else:
                foot_contacts += 1
        if reason:
            rejected.append(
                {
                    "reason": reason,
                    "geoms": [
                        model.geom(g1).name or f"geom_{g1}",
                        model.geom(g2).name or f"geom_{g2}",
                    ],
                    "distance_m": float(contact.dist),
                }
            )
    evidence = {
        "foot_ground_contact_count": foot_contacts,
        "expected_self_contact_count": self_contacts,
        "rejected_contacts": rejected,
        "max_floor_penetration_m": MAX_FLOOR_PENETRATION_M,
    }
    reason = rejected[0]["reason"] if rejected else None
    distances = _foot_floor_distances(model, data, robot_bodies, max_support_gap_m)
    reference = distances if base_foot_floor_distances is None else base_foot_floor_distances
    support_geoms = 0
    proximity_geoms = 0
    if reason is None:
        for pair, distance in distances.items():
            if distance < -MAX_FLOOR_PENETRATION_M:
                reason = "deep_foot_ground_penetration"
            elif distance <= _CONTACT_EPS_M:
                support_geoms += 1
            elif (
                distance <= max_support_gap_m + _CONTACT_EPS_M
                and pair in reference
                and 0 <= reference[pair] <= max_support_gap_m + _CONTACT_EPS_M
                and abs(distance - reference[pair]) <= _CONTACT_EPS_M
            ):
                proximity_geoms += 1
        if support_geoms == 0 and foot_contacts == 0 and proximity_geoms == 0:
            reason = reason or "no_foot_ground_support"
    evidence["foot_ground_support_geom_count"] = support_geoms
    evidence["foot_ground_proximity_geom_count"] = proximity_geoms
    evidence["max_support_gap_m"] = max_support_gap_m
    evidence["grounding_status"] = (
        "foot_contact_present"
        if foot_contacts
        else "geometric_floor_contact"
        if support_geoms
        else "near_floor_preserving_base_gap"
        if proximity_geoms
        else "unsupported"
    )
    evidence["foot_floor_distances"] = [
        {
            "foot_geom_id": foot,
            "foot_body": model.body(int(model.geom_bodyid[foot])).name,
            "floor_geom_id": floor,
            "signed_distance_m": distance,
            "base_signed_distance_m": reference.get((floor, foot)),
        }
        for (floor, foot), distance in distances.items()
    ]
    return reason, evidence


def _validate_indexed_scene(scene, model, data, geom_ids) -> tuple[str | None, dict]:
    # Same world AABB transform and obstacle clearance as the eval wall guard,
    # using loaded geometry directly to avoid importing CLI/Isaac dependencies.
    from humano.utils.geometry import local_aabbs_to_world

    aabbs = model.geom_aabb[geom_ids]
    lower, upper = local_aabbs_to_world(
        aabbs[:, :3],
        aabbs[:, 3:],
        data.geom_xpos[geom_ids],
        data.geom_xmat[geom_ids].reshape(-1, 3, 3),
    )
    evidence = {
        "scene_uid": scene.uid,
        "checked_aabb_count": len(geom_ids),
        "immutable_obstacle_count": len(scene.immutable_obstacles),
        "clearance_m": float(scene.bounds.margin_m),
    }
    if not (np.all(np.isfinite(lower)) and np.all(np.isfinite(upper))):
        return "nonfinite_indexed_bounds", evidence
    safe_min = np.asarray(scene.bounds.min_xy) + scene.bounds.margin_m
    safe_max = np.asarray(scene.bounds.max_xy) - scene.bounds.margin_m
    if not (np.all(lower[:, :2] >= safe_min) and np.all(upper[:, :2] <= safe_max)):
        return "indexed_scene_bounds", evidence
    for obstacle in scene.immutable_obstacles:
        obstacle_min = np.asarray(obstacle.min_xyz) - scene.bounds.margin_m
        obstacle_max = np.asarray(obstacle.max_xyz) + scene.bounds.margin_m
        overlap = np.all(upper >= obstacle_min, axis=1) & np.all(lower <= obstacle_max, axis=1)
        if np.any(overlap):
            evidence["obstacle_prim_path"] = obstacle.prim_path
            return "indexed_obstacle_overlap", evidence
    evidence["immutable_obstacle_collision_free"] = True
    return None, evidence


def derive_robot_initial_variant(
    env: Any,
    base_config: dict[str, Any],
    subtype: str,
    rng: np.random.Generator,
    max_attempts: int = 100,
    *,
    xy_radius_m: float = XY_RADIUS_M,
    yaw_limit_deg: float = 10.0,
    task_overrides: dict[str, dict[str, float]] | None = None,
    indexed_scene: Any | None = None,
    max_support_gap_m: float = MAX_SUPPORT_GAP_M,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return a separate translation/yaw variant and frozen constraint evidence.

    Requires env.unwrapped.{mjModel,mjData,task} after reset, a G1 root freejoint,
    and foot contact or unchanged baseline foot/ground proximity within the
    configurable limit (40 mm default for actual Inspire prestep colliders).
    Near-floor acceptance is not a contact or stability certificate. Expected
    base self-contact pairs may persist without deepening; all external robot
    contacts except sole/ground are rejected, including pairs present at base.
    XY radius and yaw limit are adjustable defaults. task_overrides maps task
    UID to xy_radius_m, yaw_limit_deg and/or max_support_gap_m, overriding defaults.
    Indexed eval_scene requires aligned geometry in task.layout.scene or the
    indexed_scene argument (for a background installed separately by the CLI).
    Raises RobotInitialSamplingError on exhaustion; never substitutes base.
    """
    if subtype not in {"translation", "yaw"}:
        raise ValueError("subtype must be 'translation' or 'yaw'; joint jitter is unsupported")
    if isinstance(max_attempts, bool) or not isinstance(max_attempts, int) or max_attempts < 1:
        raise ValueError("max_attempts must be a positive integer")
    overrides = (task_overrides or {}).get(base_config.get("uid"), {})
    if set(overrides) - {"xy_radius_m", "yaw_limit_deg", "max_support_gap_m"}:
        raise ValueError("Unsupported robot initial task override")
    xy_radius_m = float(overrides.get("xy_radius_m", xy_radius_m))
    yaw_limit_deg = float(overrides.get("yaw_limit_deg", yaw_limit_deg))
    max_support_gap_m = float(overrides.get("max_support_gap_m", max_support_gap_m))
    if not np.isfinite(max_support_gap_m) or max_support_gap_m < 0:
        raise ValueError("max_support_gap_m must be nonnegative and finite")
    if (
        not np.isfinite(xy_radius_m)
        or xy_radius_m <= 0
        or not np.isfinite(yaw_limit_deg)
        or yaw_limit_deg <= 0
    ):
        raise ValueError("xy_radius_m and yaw_limit_deg must be positive finite values")
    yaw_limit_rad = float(np.deg2rad(yaw_limit_deg))
    spatial = base_config.get("dr_state_dict", {}).get("spatial", {})
    keys = [key for key in ("g1_sonic", "g1_wholebody") if key in spatial]
    if len(keys) != 1:
        raise ValueError("spatial must contain exactly one of g1_sonic or g1_wholebody")
    robot_key = keys[0]
    position = np.asarray(spatial[robot_key]["position"], dtype=np.float64)
    quaternion = np.asarray(spatial[robot_key]["quaternion"], dtype=np.float64)
    if position.shape != (3,) or not np.all(np.isfinite(position)):
        raise ValueError("robot position must contain three finite values")
    if (
        quaternion.shape != (4,)
        or not np.all(np.isfinite(quaternion))
        or not np.isclose(np.linalg.norm(quaternion), 1.0, atol=1e-5)
    ):
        raise ValueError("robot quaternion must be a unit wxyz quaternion")

    import mujoco

    raw = env.unwrapped
    raw.reset(options={"state_dict": deepcopy(base_config)})
    model, source = raw.mjModel, raw.mjData
    root_body = int(model.body(raw.task.robot.root_body).id)
    root_joints = [
        j
        for j in range(model.njnt)
        if int(model.jnt_bodyid[j]) == root_body
        and int(model.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE)
    ]
    if len(root_joints) != 1:
        raise ValueError("G1 robot root must have exactly one freejoint")
    root_joint = root_joints[0]
    address = int(model.jnt_qposadr[root_joint])
    robot_bodies = {root_body}
    for body in range(root_body + 1, model.nbody):
        if int(model.body_parentid[body]) in robot_bodies:
            robot_bodies.add(body)
    robot_geom_ids = np.asarray(
        [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) in robot_bodies], dtype=int
    )
    if robot_geom_ids.size == 0:
        raise ValueError("Robot has no MuJoCo geometry to validate")
    scene = indexed_scene if indexed_scene is not None else getattr(raw.task.layout, "scene", None)
    indexed = scene is not None and hasattr(scene, "immutable_obstacles")
    if base_config.get("eval_scene") is not None and not indexed:
        raise ValueError("eval_scene requires a loaded indexed scene in task.layout.scene")
    if indexed and base_config.get("eval_scene") is not None:
        selection = base_config["eval_scene"]
        scene_uid = selection.get("scene_uid", selection.get("scene", {}).get("uid"))
        if scene_uid != scene.uid:
            raise ValueError("Loaded indexed scene does not match eval_scene uid")

    data = mujoco.MjData(model)
    # Preserve world state and auxiliary kinematic inputs; never touch live data.
    for field in ("qpos", "qvel", "act", "ctrl", "mocap_pos", "mocap_quat", "eq_active"):
        getattr(data, field)[:] = getattr(source, field)
    data.time = source.time
    baseline = source.qpos.copy()
    root_pose = baseline[address : address + 7].copy()
    mujoco.mj_forward(model, data)
    base_contacts = _contact_depths(data)
    base_distances = _foot_floor_distances(model, data, robot_bodies, max_support_gap_m)
    base_reason, baseline_constraints = _validate_contacts(
        model, data, robot_bodies, base_contacts, max_support_gap_m, base_distances
    )
    baseline_constraints.update(
        {
            "rejection_reason": base_reason,
            "root_position_m": root_pose[:3].tolist(),
            "source_sim_time_s": float(source.time),
        }
    )
    limited_joints = [
        j
        for j in range(model.njnt)
        if int(model.jnt_bodyid[j]) in robot_bodies and model.jnt_limited[j]
    ]
    rejections: Counter[str] = Counter()
    last_evidence = {}
    for attempt in range(1, max_attempts + 1):
        dx, dy, yaw = _sample_delta(subtype, rng, xy_radius_m, yaw_limit_rad)
        data.qpos[:] = baseline
        data.qpos[address : address + 2] += [dx, dy]
        data.qpos[address + 3 : address + 7] = _yaw_quaternion(root_pose[3:], yaw)
        mujoco.mj_forward(model, data)
        reason, evidence = _validate_contacts(
            model, data, robot_bodies, base_contacts, max_support_gap_m, base_distances
        )
        if not np.all(np.isfinite(data.qpos)) or not np.all(np.isfinite(data.qvel)):
            reason = "nonfinite_physical_state"
        q = data.qpos[address + 3 : address + 7]
        up_z = 1.0 - 2.0 * (q[1] ** 2 + q[2] ** 2)
        evidence["pelvis_tilt_deg"] = float(np.rad2deg(np.arccos(np.clip(up_z, -1, 1))))
        if not np.isclose(np.linalg.norm(q), 1.0, atol=1e-5) or up_z < 0.5:
            reason = "invalid_root_orientation"
        for joint in limited_joints:
            value = data.qpos[int(model.jnt_qposadr[joint])]
            low, high = model.jnt_range[joint]
            if value < low - 1e-6 or value > high + 1e-6:
                reason = "robot_joint_limit"
        if indexed:
            wall_reason, evidence["indexed_scene"] = _validate_indexed_scene(
                scene, model, data, robot_geom_ids
            )
            reason = reason or wall_reason
        evidence.update(
            {
                "validation": "frozen_mujoco_forward",
                "baseline_phase": "post_env_reset",
                "source_sim_time_s": float(source.time),
                "extra_validation_physics_steps": 0,
                "robot_height_unchanged": bool(data.qpos[address + 2] == root_pose[2]),
                "joint_qpos_unchanged": bool(
                    np.array_equal(data.qpos[address + 7 :], baseline[address + 7 :])
                ),
                "world_qpos_unchanged": bool(
                    np.array_equal(
                        np.concatenate((data.qpos[:address], data.qpos[address + 7 :])),
                        np.concatenate((baseline[:address], baseline[address + 7 :])),
                    )
                ),
                "stability_rollout_checked": False,
                "goal_reachability_checked": False,
                "policy_success_filter_used": False,
                "collision_coverage": "model_configured_geom_contacts",
            }
        )
        last_evidence = evidence
        if reason:
            rejections[reason] += 1
            continue
        variant = deepcopy(base_config)
        pose = variant["dr_state_dict"]["spatial"][robot_key]
        pose["position"] = (position + [dx, dy, 0]).tolist()
        pose["quaternion"] = _yaw_quaternion(quaternion, yaw).tolist()
        variant["ood_physics_state"] = capture_physics(data)
        return variant, {
            "subtype": subtype,
            "robot_key": robot_key,
            "translation_xy_m": [dx, dy],
            "yaw_rad": yaw,
            "xy_radius_m": xy_radius_m,
            "yaw_limit_rad": yaw_limit_rad,
            "yaw_limit_deg": yaw_limit_deg,
            "attempt_count": attempt,
            "rejections": dict(rejections),
            "accepted_pose": deepcopy(pose),
            "root_qpos_address": address,
            "root_pose": {
                "position": data.qpos[address : address + 3].tolist(),
                "quaternion": data.qpos[address + 3 : address + 7].tolist(),
            },
            "constraints": evidence,
            "baseline_constraints": baseline_constraints,
            "max_support_gap_m": max_support_gap_m,
        }
    raise RobotInitialSamplingError(max_attempts, dict(rejections), last_evidence)
