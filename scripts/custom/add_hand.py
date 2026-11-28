#!/usr/bin/env python3
"""Scaffold a new HandSpec by copying from an existing hand type.

Usage:
    python scripts/custom/add_hand.py --name my_hand --from dex3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _repr_tuple(values: tuple, indent: int = 8) -> str:
    """Format a tuple for display with per-element comments."""
    inner = ",\n" + " " * indent
    items = [repr(v) for v in values]
    return "(" + inner.join(items) + ",\n" + " " * (indent - 4) + ")"


def _repr_dict(d: dict, indent: int = 8) -> str:
    """Format a dict for display with per-entry comments."""
    if not d:
        return "{}"
    inner = ",\n" + " " * indent
    items = [f"{repr(k)}: {repr(v)}" for k, v in d.items()]
    return "{" + inner.join(items) + ",\n" + " " * (indent - 4) + "}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scaffold a new HandSpec from an existing hand type"
    )
    parser.add_argument("--name", type=str, required=True, help="Name of the new hand type")
    parser.add_argument(
        "--from",
        type=str,
        default="dex3",
        dest="source",
        help="Source hand type to copy from (default: dex3)",
    )
    args = parser.parse_args()

    # Add src to path so we can import humano
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

    from humano.robots.hands import HandSpec, get_hand_spec

    try:
        source = get_hand_spec(args.source)
    except KeyError:
        print(f"Error: unknown source hand type {args.source!r}")
        print(f"Available: dex3, inspire")
        sys.exit(1)

    name_upper = args.name.upper().replace("-", "_")
    indent = " " * 4

    # Map field values: keep numerics but mark as TODO, set strings to "TODO: fill me"
    def make_hand_spec_lines(spec: HandSpec, new_name: str) -> list[str]:
        lines = []
        lines.append(f"{name_upper} = HandSpec(")
        lines.append(f'{indent}name="{new_name}",')
        lines.append(f"{indent}dof={spec.dof},  # TODO: set correct DOF")

        # Joint tuples: copy structure, mark with TODOs
        lines.append(f"{indent}# TODO: fill in correct joint names")
        lines.append(f"{indent}left_actuated_joints=(")
        for _ in spec.left_actuated_joints:
            lines.append(f'{indent * 2}"TODO: fill me",')
        lines.append(f"{indent}),")
        lines.append(f"{indent}left_model_joints=(")
        for _ in spec.left_model_joints:
            lines.append(f'{indent * 2}"TODO: fill me",')
        lines.append(f"{indent}),")
        if spec.left_mimic_map:
            lines.append(f"{indent}left_mimic_map={{")
            for k in spec.left_mimic_map:
                lines.append(f'{indent * 2}"TODO: fill me": ("TODO: fill me", 1.0, 0.0),')
            lines.append(f"{indent}}},")
        else:
            lines.append(f"{indent}left_mimic_map={{}},")

        lines.append(f"{indent}right_actuated_joints=(")
        for _ in spec.right_actuated_joints:
            lines.append(f'{indent * 2}"TODO: fill me",')
        lines.append(f"{indent}),")
        lines.append(f"{indent}right_model_joints=(")
        for _ in spec.right_model_joints:
            lines.append(f'{indent * 2}"TODO: fill me",')
        lines.append(f"{indent}),")
        if spec.right_mimic_map:
            lines.append(f"{indent}right_mimic_map={{")
            for k in spec.right_mimic_map:
                lines.append(f'{indent * 2}"TODO: fill me": ("TODO: fill me", 1.0, 0.0),')
            lines.append(f"{indent}}},")
        else:
            lines.append(f"{indent}right_mimic_map={{}},")

        # qpos_reorder
        if spec.qpos_reorder_from_gr00t is not None:
            lines.append(f"{indent}# TODO: set correct qpos reorder indices (or None)")
            lines.append(f"{indent}qpos_reorder_from_gr00t=(")
            for _ in spec.qpos_reorder_from_gr00t:
                lines.append(f"{indent * 2}0,  # TODO: correct index")
            lines.append(f"{indent}),")
        else:
            lines.append(f"{indent}qpos_reorder_from_gr00t=None,")

        # String fields
        lines.append(f'{indent}left_hand_uid="TODO: fill me",')
        lines.append(f'{indent}right_hand_uid="TODO: fill me",')
        lines.append(f'{indent}mjcf_path="TODO: fill me",')
        lines.append(f'{indent}wbc_yaml_path="TODO: fill me",')
        lines.append(f'{indent}wbc_version="TODO: fill me",')
        lines.append(f'{indent}cu_robot_yaml="TODO: fill me",')
        lines.append(f'{indent}hand_yaml="TODO: fill me",')
        lines.append(f'{indent}usd_path="TODO: fill me",')
        lines.append(f'{indent}left_ee_link="TODO: fill me",')
        lines.append(f'{indent}right_ee_link="TODO: fill me",')

        # Qpos: copy length, mark with TODO
        lines.append(f"{indent}# TODO: set correct init/close qpos values")
        lines.append(f"{indent}init_qpos=(")
        for _ in spec.init_qpos:
            lines.append(f"{indent * 2}0.0,  # TODO: correct value")
        lines.append(f"{indent}),")
        lines.append(f"{indent}close_qpos_left=(")
        for _ in spec.close_qpos_left:
            lines.append(f"{indent * 2}0.0,  # TODO: correct value")
        lines.append(f"{indent}),")
        lines.append(f"{indent}close_qpos_right=(")
        for _ in spec.close_qpos_right:
            lines.append(f"{indent * 2}0.0,  # TODO: correct value")
        lines.append(f"{indent}),")

        # Gains
        lines.append(f"{indent}kp_map=None,  # TODO: set per-joint kp overrides or None")
        lines.append(f"{indent}kd={spec.kd},  # TODO: set correct kd")

        lines.append(")")
        return lines

    lines = make_hand_spec_lines(source, args.name)

    print(f"# Generated HandSpec for {args.name!r} (based on {args.source!r})")
    print("# Paste into src/humano/robots/hands.py and fill in the TODOs.")
    print()
    for line in lines:
        print(line)

    print()
    print("# TODO Checklist:")
    print("#   [ ] Create MJCF/URDF for the new hand")
    print("#   [ ] Create CuRobo YAML configs (cu_robot_yaml, hand_yaml)")
    print("#   [ ] Create WBC YAML "
          "(copy from source and modify NUM_HAND_MOTORS/NUM_HAND_JOINTS)")
    print("#   [ ] Create USD file")
    print("#   [ ] Fill in all \"TODO: fill me\" fields in the HandSpec")
    print("#   [ ] Register in `_HAND_REGISTRY` in `hands.py`")
    print(f"#   [ ] Run `validate_hand.py --hand {args.name}` to verify")


if __name__ == "__main__":
    main()
