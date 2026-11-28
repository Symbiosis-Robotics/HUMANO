#!/usr/bin/env python3
"""Validate HandSpec consistency for G1Sonic hand types.

Usage:
    python scripts/custom/validate_hand.py --hand dex3
    python scripts/custom/validate_hand.py --all
"""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml


def validate_explicit_contact_profile(spec, mjcf_path: Path) -> list[tuple[bool, str]]:
    """Validate the static MJCF portion of an explicit hand-contact profile."""
    if spec.contact_collision_prefix is None:
        return []
    root = ET.parse(mjcf_path).getroot()
    geoms = root.findall(".//geom")
    collision_geoms = [
        geom for geom in geoms if (geom.get("name") or "").startswith(spec.contact_collision_prefix)
    ]
    hand_mesh_geoms = [
        geom
        for geom in geoms
        if (geom.get("mesh") or "").startswith("Link")
        or (geom.get("mesh") or "").endswith("_hand_base_link")
    ]
    collision_disabled = all(
        geom.get("contype") == "0" and geom.get("conaffinity") == "0"
        for geom in collision_geoms
    )
    meshes_disabled = all(
        geom.get("contype") == "0" and geom.get("conaffinity") == "0"
        for geom in hand_mesh_geoms
    )
    import mujoco

    mujoco.MjModel.from_xml_path(str(mjcf_path))
    return [
        (len(collision_geoms) == 22, f"  {'PASS' if len(collision_geoms) == 22 else 'FAIL'}  explicit collision geoms={len(collision_geoms)} (expected 22)"),
        (collision_disabled, f"  {'PASS' if collision_disabled else 'FAIL'}  explicit collision geoms disable automatic contacts"),
        (meshes_disabled, f"  {'PASS' if meshes_disabled else 'FAIL'}  Inspire visual meshes disable automatic contacts"),
        (True, "  PASS  Inspire MJCF compiles with the explicit contact geometry"),
    ]


def resolve_project_root() -> Path:
    """Return the HUMANO project root (two levels up from this script)."""
    return Path(__file__).resolve().parents[2]


def resolve_data_path(project_root: Path, rel_path: str) -> Path:
    """Resolve a data asset path (mjcf, cu_robot, hand_yaml, usd) under data/."""
    return project_root / "data" / rel_path


def resolve_wbc_yaml_path(project_root: Path, filename: str) -> Path:
    """Resolve a WBC YAML path under gear_sonic's wbc_configs directory."""
    return project_root / "third_party" / "gear_sonic" / "utils" / "mujoco_sim" / "wbc_configs" / filename


def check_file_exists(label: str, path: Path) -> tuple[bool, str]:
    """Check if a file exists. Returns (pass, message)."""
    if path.exists():
        return True, f"  PASS  {label}: {path}"
    # usd_path might be a directory containing .usd files
    if path.is_dir() and list(path.glob("*.usd")):
        return True, f"  PASS  {label}: {path} (directory with .usd files)"
    return False, f"  FAIL  {label}: not found: {path}"


def validate_spec(spec, project_root: Path) -> bool:
    """Run all validation checks on a single HandSpec. Returns True if all pass."""
    all_pass = True

    def check(passed: bool, msg: str) -> None:
        nonlocal all_pass
        print(msg)
        if not passed:
            all_pass = False

    # ── 1. File existence ────────────────────────────────────────────────
    print("\n--- File existence ---")
    ok, msg = check_file_exists("mjcf_path", resolve_data_path(project_root, spec.mjcf_path))
    check(ok, msg)
    ok, msg = check_file_exists("cu_robot_yaml", resolve_data_path(project_root, spec.cu_robot_yaml))
    check(ok, msg)
    ok, msg = check_file_exists("hand_yaml", resolve_data_path(project_root, spec.hand_yaml))
    check(ok, msg)
    ok, msg = check_file_exists("usd_path", resolve_data_path(project_root, spec.usd_path))
    check(ok, msg)

    # ── 2. WBC YAML existence ────────────────────────────────────────────
    print("\n--- WBC YAML ---")
    wbc_path = resolve_wbc_yaml_path(project_root, spec.wbc_yaml_path)
    ok = wbc_path.exists()
    check(ok, f"  {'PASS' if ok else 'FAIL'}  wbc_yaml_path: {wbc_path}")
    wbc_config = {}
    if ok:
        with wbc_path.open() as f:
            wbc_config = yaml.safe_load(f) or {}
        ok_motors = wbc_config.get("NUM_HAND_MOTORS") == spec.dof
        check(
            ok_motors,
            f"  {'PASS' if ok_motors else 'FAIL'}  NUM_HAND_MOTORS={wbc_config.get('NUM_HAND_MOTORS')} matches dof={spec.dof}",
        )
        ok_joints = wbc_config.get("NUM_HAND_JOINTS") == spec.dof
        check(
            ok_joints,
            f"  {'PASS' if ok_joints else 'FAIL'}  NUM_HAND_JOINTS={wbc_config.get('NUM_HAND_JOINTS')} matches active dof={spec.dof}",
        )
        effort_limits = wbc_config.get("motor_effort_limit_list", [])
        expected_effort_len = 29 + spec.dof * 2
        ok_effort = len(effort_limits) == expected_effort_len
        check(
            ok_effort,
            f"  {'PASS' if ok_effort else 'FAIL'}  motor_effort_limit_list len={len(effort_limits)} matches active ctrl len={expected_effort_len}",
        )

    mjcf_joints: set[str] = set()
    mjcf_actuated_joints: set[str] = set()
    mjcf_path = resolve_data_path(project_root, spec.mjcf_path)
    if mjcf_path.exists():
        root = ET.parse(mjcf_path).getroot()
        mjcf_joints = {joint.get("name") for joint in root.findall(".//joint") if joint.get("name")}
        mjcf_actuated_joints = {
            actuator.get("joint")
            for actuator in root.findall(".//actuator/*")
            if actuator.get("joint")
        }

    # ── Explicit contact profile ───────────────────────────────────────────
    print("\n--- Explicit contact profile ---")
    for ok, msg in validate_explicit_contact_profile(spec, mjcf_path):
        check(ok, msg)

    # ── 3. Joint consistency ─────────────────────────────────────────────
    print("\n--- Joint consistency ---")
    for side in ("left", "right"):
        actuated = getattr(spec, f"{side}_actuated_joints")
        model = getattr(spec, f"{side}_model_joints")
        mimic = getattr(spec, f"{side}_mimic_map")

        # Length matches dof
        ok_len = len(actuated) == spec.dof
        check(
            ok_len,
            f"  {'PASS' if ok_len else 'FAIL'}  {side}: actuated_joints len={len(actuated)} matches dof={spec.dof}",
        )

        # Actuated are subset of model
        actuated_set = set(actuated)
        model_set = set(model)
        extra = actuated_set - model_set
        ok_sub = not extra
        check(
            ok_sub,
            f"  {'PASS' if ok_sub else 'FAIL'}  {side}: actuated_joints subset of model_joints"
            + (f" (extra: {extra})" if extra else ""),
        )

        # No duplicates in actuated or model
        ok_dup_a = len(actuated) == len(actuated_set)
        check(
            ok_dup_a,
            f"  {'PASS' if ok_dup_a else 'FAIL'}  {side}: no duplicate actuated joints",
        )
        ok_dup_m = len(model) == len(model_set)
        check(
            ok_dup_m,
            f"  {'PASS' if ok_dup_m else 'FAIL'}  {side}: no duplicate model joints",
        )

        # Mimic joint names appear in model_joints
        mimic_joints = set(mimic.keys())
        missing_model = mimic_joints - model_set
        ok_mimic_model = not missing_model
        check(
            ok_mimic_model,
            f"  {'PASS' if ok_mimic_model else 'FAIL'}  {side}: mimic joints in model_joints"
            + (f" (missing: {missing_model})" if missing_model else ""),
        )

        # Mimic parent joints appear in actuated_joints
        parent_joints = {v[0] for v in mimic.values()}
        missing_act = parent_joints - actuated_set
        ok_parent = not missing_act
        check(
            ok_parent,
            f"  {'PASS' if ok_parent else 'FAIL'}  {side}: mimic parents in actuated_joints"
            + (f" (missing: {missing_act})" if missing_act else ""),
        )

        # Model joints must exist in the MJCF; active joints must have actuators.
        missing_mjcf = model_set - mjcf_joints
        ok_mjcf = not missing_mjcf
        check(
            ok_mjcf,
            f"  {'PASS' if ok_mjcf else 'FAIL'}  {side}: model_joints exist in MJCF"
            + (f" (missing: {missing_mjcf})" if missing_mjcf else ""),
        )
        missing_actuator = actuated_set - mjcf_actuated_joints
        ok_actuator = not missing_actuator
        check(
            ok_actuator,
            f"  {'PASS' if ok_actuator else 'FAIL'}  {side}: actuated_joints have MJCF actuators"
            + (f" (missing: {missing_actuator})" if missing_actuator else ""),
        )

    # ── 4. Qpos consistency ──────────────────────────────────────────────
    print("\n--- Qpos consistency ---")
    for label, qpos in [
        ("init_qpos", spec.init_qpos),
        ("close_qpos_left", spec.close_qpos_left),
        ("close_qpos_right", spec.close_qpos_right),
    ]:
        ok_q = len(qpos) == spec.dof
        check(
            ok_q,
            f"  {'PASS' if ok_q else 'FAIL'}  {label}: len={len(qpos)} matches dof={spec.dof}",
        )

    # ── 5. Controller init ───────────────────────────────────────────────
    print("\n--- Controller parameters ---")
    ok_has_name = bool(spec.name and spec.name.strip())
    check(ok_has_name, f"  {'PASS' if ok_has_name else 'FAIL'}  name is set: {spec.name!r}")
    ok_dof = spec.dof > 0
    check(ok_dof, f"  {'PASS' if ok_dof else 'FAIL'}  dof > 0: {spec.dof}")
    ok_has_ee_left = bool(spec.left_ee_link and spec.left_ee_link.strip())
    check(ok_has_ee_left, f"  {'PASS' if ok_has_ee_left else 'FAIL'}  left_ee_link is set")
    ok_has_ee_right = bool(spec.right_ee_link and spec.right_ee_link.strip())
    check(ok_has_ee_right, f"  {'PASS' if ok_has_ee_right else 'FAIL'}  right_ee_link is set")
    ok_kd = spec.kd > 0
    check(ok_kd, f"  {'PASS' if ok_kd else 'FAIL'}  kd > 0: {spec.kd}")

    if spec.kp_map is not None:
        # kp_map keys should be in actuated joints
        all_actuated = set(spec.left_actuated_joints) | set(spec.right_actuated_joints)
        kp_keys = set(spec.kp_map.keys())
        extra_kp = kp_keys - all_actuated
        ok_kp = not extra_kp
        check(
            ok_kp,
            f"  {'PASS' if ok_kp else 'FAIL'}  kp_map keys in actuated_joints"
            + (f" (extra: {extra_kp})" if extra_kp else ""),
        )

    return all_pass


def validate_registry_runtime(project_root: Path) -> bool:
    """Check that RobotRegistry can hold dex3 and inspire at the same time."""
    from gear_sonic.utils.mujoco_sim.configs import SimLoopConfig

    from humano.robots.hands import get_hand_spec
    from humano.robots.registry import RobotRegistry

    all_pass = True
    print(f"\n{'=' * 60}")
    print("Validating registry runtime")
    print(f"{'=' * 60}")
    robots = []
    for hand_name in ("dex3", "inspire"):
        spec = get_hand_spec(hand_name)
        sonic_config = SimLoopConfig(wbc_version=spec.wbc_version).load_wbc_yaml()
        sonic_config["WBC_VERSION"] = spec.wbc_version
        sonic_config["ENV_NAME"] = "humano"
        robot = RobotRegistry.make("g1_sonic", sonic_config, hand_spec=spec)
        robots.append(robot)
        ok = robot.hand_spec.name == hand_name
        print(f"  {'PASS' if ok else 'FAIL'}  RobotRegistry.make(... {hand_name}) returns {robot.hand_spec.name}")
        all_pass = all_pass and ok

    ok_distinct = robots[0] is not robots[1]
    print(f"  {'PASS' if ok_distinct else 'FAIL'}  dex3 and inspire robot instances are distinct")
    all_pass = all_pass and ok_distinct
    ok_joints = robots[0].joint_names != robots[1].joint_names
    print(f"  {'PASS' if ok_joints else 'FAIL'}  dex3 and inspire joint_names differ")
    all_pass = all_pass and ok_joints
    return all_pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate HandSpec configurations")
    parser.add_argument("--hand", type=str, default=None, help="Hand type to validate")
    parser.add_argument("--all", action="store_true", help="Validate all registered hands")
    args = parser.parse_args()

    # Add src to path for import
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

    from humano.robots.hands import _HAND_REGISTRY, get_hand_spec

    project_root = resolve_project_root()

    if args.all:
        specs = list(_HAND_REGISTRY.items())
    elif args.hand:
        try:
            spec = get_hand_spec(args.hand)
        except KeyError as e:
            print(f"Error: {e}")
            sys.exit(1)
        specs = [(args.hand, spec)]
    else:
        parser.print_help()
        sys.exit(1)

    all_pass = True
    for name, spec in specs:
        print(f"\n{'=' * 60}")
        print(f"Validating hand: {name}")
        print(f"{'=' * 60}")
        passed = validate_spec(spec, project_root)
        if not passed:
            all_pass = False

    if args.all:
        all_pass = validate_registry_runtime(project_root) and all_pass

    print(f"\n{'=' * 60}")
    if all_pass:
        print("OVERALL: ALL CHECKS PASSED")
    else:
        print("OVERALL: SOME CHECKS FAILED")
    print(f"{'=' * 60}")

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
