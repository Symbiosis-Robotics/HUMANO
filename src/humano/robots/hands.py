"""
HUMANO: Hand specification for pluggable hand types on G1Sonic.

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

# Compiled SPIDER Inspire position-actuator gains in the apple MuJoCo example.
INSPIRE_SPIDER_KP = 300.0
INSPIRE_SPIDER_KD = 34.64188167
INSPIRE_EFFORT_LIMIT = 1.0

INSPIRE_DFQ_PATH = Path(__file__).resolve().parents[3] / "configs/robots/inspire_dfq.xml"
_DFQ = ET.parse(INSPIRE_DFQ_PATH).getroot()
_DFQ_CONTACT = _DFQ.find("contact/pair")


def _dfq_values(name: str) -> tuple[float, ...]:
    return tuple(map(float, _DFQ.find(f"custom/numeric[@name='{name}']").get("data").split()))


@dataclass(frozen=True)
class HandSpec:
    """Pluggable hand specification injected into G1Sonic.

    Encapsulates all hand-specific configuration so that switching hand types
    requires nothing more than passing a different HandSpec instance.
    """

    name: str  # "dex3" | "inspire" | "brainco"
    dof: int  # actuated DOF per hand (used for WBC, controllers, MotorCmd)

    # --- Joint classification (actuated vs model-level) ---
    left_actuated_joints: tuple[str, ...]  # actively controlled joints (length = dof)
    left_model_joints: tuple[str, ...]  # all hand joints in the model (incl. mimic)
    left_mimic_map: dict[str, tuple[str, float, float]]  # mimic joint -> (parent, ratio, offset)

    right_actuated_joints: tuple[str, ...]
    right_model_joints: tuple[str, ...]
    right_mimic_map: dict[str, tuple[str, float, float]]

    # --- Data reorder (optional, for external qpos formats) ---
    # Some external data formats (e.g. GR00T) store hand joints in a different
    # order than the MJCF. This mapping reorders raw qpos into model order.
    qpos_reorder_from_gr00t: tuple[int, ...] | None

    # --- Task / grasp spec identifiers ---
    left_hand_uid: str
    right_hand_uid: str

    # --- Model / config paths ---
    mjcf_path: str
    wbc_yaml_path: str  # filename under gear_sonic/utils/mujoco_sim/wbc_configs/
    wbc_version: str  # key for configs.py load_wbc_yaml
    cu_robot_yaml: str | None  # CuRobo full-body config
    hand_yaml: str | None  # CuRobo single-hand config
    usd_path: str | None

    # --- Arm -> Hand connection ---
    left_ee_link: str
    right_ee_link: str

    # --- Controller parameters (length = dof, actuated joints only) ---
    init_qpos: tuple[float, ...]
    close_qpos_left: tuple[float, ...]
    close_qpos_right: tuple[float, ...]

    # --- PD gains (actuated joints; None = use defaults) ---
    kp_map: dict[str, float] | None  # per-joint kp overrides
    kd: float

    # --- MuJoCo explicit hand-contact profile (None keeps legacy contacts) ---
    contact_collision_prefix: str | None
    contact_solref: tuple[float, float] | None
    contact_friction: tuple[float, float, float, float, float] | None
    contact_primary_condim: int | None
    contact_other_condim: int | None
    physics_dt: float | None = None
    reference_velocity: tuple[float, ...] | None = None
    binary_transition_duration: float | None = None
    effort_limits: tuple[float, ...] | None = None
    assembly_id: str | None = None
    reference_acceleration: tuple[float, ...] | None = None
    reference_open_velocity: tuple[float, ...] | None = None

    # --- Convenience properties ---

    @property
    def left_mimic_joints(self) -> tuple[str, ...]:
        return tuple(self.left_mimic_map.keys())

    @property
    def right_mimic_joints(self) -> tuple[str, ...]:
        return tuple(self.right_mimic_map.keys())

    def uid_for(self, side: str) -> str:
        """Return the hand_uid used in task / grasp specs."""
        if side == "left":
            return self.left_hand_uid
        if side == "right":
            return self.right_hand_uid
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")


# ── Dex3-1 Hand (7 actuated = 7 model, no mimic) ──────────────────────────

_DEX3_ACTUATED = (
    "left_hand_thumb_0_joint",
    "left_hand_thumb_1_joint",
    "left_hand_thumb_2_joint",
    "left_hand_index_0_joint",
    "left_hand_index_1_joint",
    "left_hand_middle_0_joint",
    "left_hand_middle_1_joint",
)
_DEX3_ACTUATED_R = tuple(n.replace("left_", "right_") for n in _DEX3_ACTUATED)

DEX3_HAND = HandSpec(
    name="dex3",
    dof=7,
    left_actuated_joints=_DEX3_ACTUATED,
    left_model_joints=_DEX3_ACTUATED,
    left_mimic_map={},
    right_actuated_joints=_DEX3_ACTUATED_R,
    right_model_joints=_DEX3_ACTUATED_R,
    right_mimic_map={},
    qpos_reorder_from_gr00t=(3, 4, 5, 6, 0, 1, 2),
    left_hand_uid="dex3_left",
    right_hand_uid="dex3_right",
    mjcf_path="robots/g1_sonic/g1_29dof_with_hand.xml",
    wbc_yaml_path="g1_29dof_sonic_model12.yaml",
    wbc_version="sonic_model12",
    cu_robot_yaml="robots/g1/curobo/g1_29dof_with_dex3.yml",
    hand_yaml="robots/g1/curobo/dex3_right.yml",
    usd_path="robots/g1/g1_29dof_wholebody_dex3.usd",
    left_ee_link="left_hand_palm_link",
    right_ee_link="right_hand_palm_link",
    init_qpos=(0, 0, 0, 0, 0, 0, 0),
    close_qpos_left=(
        0.3523,
        -0.0964,
        0.2790,
        -0.5058,
        -1.1950,
        -0.5389,
        -0.9835,
    ),
    close_qpos_right=(
        0.02331954,
        -0.02398408,
        -0.22170663,
        0.25662386,
        1.3371105,
        0.3085137,
        0.9805285,
    ),
    kp_map=None,
    kd=1.0,
    contact_collision_prefix=None,
    contact_solref=None,
    contact_friction=None,
    contact_primary_condim=None,
    contact_other_condim=None,
)

# ── Inspire Hand (6 actuated, 12 model, with mimic) ───────────────────────

_INSPIRE_ACTUATED_L = (
    "L_thumb_proximal_yaw_joint",
    "L_thumb_proximal_pitch_joint",
    "L_index_proximal_joint",
    "L_middle_proximal_joint",
    "L_ring_proximal_joint",
    "L_pinky_proximal_joint",
)
_INSPIRE_MIMIC_L = {
    joint.get("joint1"): (
        joint.get("joint2"),
        float(joint.get("polycoef").split()[1]),
        float(joint.get("polycoef").split()[0]),
    )
    for joint in _DFQ.findall("equality/joint")
    if joint.get("joint1").startswith("L_")
}
_INSPIRE_ACTUATED_R = tuple(n.replace("L_", "R_") for n in _INSPIRE_ACTUATED_L)
_INSPIRE_MIMIC_R = {
    k.replace("L_", "R_"): (v[0].replace("L_", "R_"), v[1], v[2])
    for k, v in _INSPIRE_MIMIC_L.items()
}

INSPIRE_HAND = HandSpec(
    name="inspire",
    dof=6,
    left_actuated_joints=_INSPIRE_ACTUATED_L,
    left_model_joints=_INSPIRE_ACTUATED_L + tuple(_INSPIRE_MIMIC_L.keys()),
    left_mimic_map=_INSPIRE_MIMIC_L,
    right_actuated_joints=_INSPIRE_ACTUATED_R,
    right_model_joints=_INSPIRE_ACTUATED_R + tuple(_INSPIRE_MIMIC_R.keys()),
    right_mimic_map=_INSPIRE_MIMIC_R,
    qpos_reorder_from_gr00t=None,
    left_hand_uid="inspire_left",
    right_hand_uid="inspire_right",
    mjcf_path="robots/g1/g1_29dof_with_inspire_hand.xml",
    wbc_yaml_path="g1_29dof_sonic_model12_inspire.yaml",
    wbc_version="sonic_model12_inspire",
    cu_robot_yaml="robots/g1/curobo/g1_29dof_with_inspire_hand.yml",
    hand_yaml="robots/g1/curobo/inspire_right.yml",
    usd_path="robots/g1/g1_29dof_with_inspire_hand.usd",
    left_ee_link="L_hand_base_link",
    right_ee_link="R_hand_base_link",
    init_qpos=_dfq_values("dfq_open"),
    close_qpos_left=_dfq_values("dfq_close_left"),
    close_qpos_right=_dfq_values("dfq_close_right"),
    kp_map={joint: INSPIRE_SPIDER_KP for joint in _INSPIRE_ACTUATED_L},
    kd=INSPIRE_SPIDER_KD,
    contact_collision_prefix="collision_hand_",
    contact_solref=tuple(map(float, _DFQ_CONTACT.get("solref").split())),
    contact_friction=tuple(map(float, _DFQ_CONTACT.get("friction").split())),
    contact_primary_condim=int(_DFQ_CONTACT.get("condim")),
    contact_other_condim=int(_DFQ_CONTACT.get("condim")),
    physics_dt=_dfq_values("dfq_physics_dt")[0],
    reference_velocity=_dfq_values("dfq_reference_velocity"),
)

BRAINCO_CONFIG_PATH = Path(__file__).resolve().parents[3] / "configs/robots/brainco_revo2.xml"
_BRAINCO = ET.parse(BRAINCO_CONFIG_PATH).getroot()
_BC_ACTIVE = tuple(_BRAINCO.find("custom/text[@name='brainco_joint_order']").get("data").split())
_BC_POSITION = _BRAINCO.find("default/default/position")
_BC_CONTACT = _BRAINCO.find("default/default/default[@class='brainco_collision']/geom")
_BC_FRICTION = tuple(map(float, _BC_CONTACT.get("friction").split()))
_BC_RANGES = tuple(
    tuple(map(float, _BRAINCO.find(f"worldbody/.//joint[@name='{name}']").get("range").split()))
    for name in _BC_ACTIVE
)
_BC_MIMIC = {
    node.get("joint1"): (node.get("joint2"), *map(float, node.get("polycoef").split()[1::-1]))
    for node in _BRAINCO.findall("equality/joint")
}


def _brainco_values(name: str) -> tuple[float, ...]:
    return tuple(map(float, _BRAINCO.find(f"custom/numeric[@name='{name}']").get("data").split()))


def brainco_command_gains() -> tuple[tuple[float, float], ...]:
    return tuple(
        (float(node.get("kp")), float(node.get("kv")))
        for name in _BC_ACTIVE
        for node in [_BRAINCO.find(f"actuator/position[@joint='{name}']")]
    )


_BC_CLOSE = tuple(
    lower + value * (upper - lower)
    for (lower, upper), value in zip(
        _BC_RANGES, _brainco_values("brainco_close_normalized"), strict=True
    )
)


BRAINCO_HAND = HandSpec(
    name="brainco",
    assembly_id=_BRAINCO.get("model"),
    dof=len(_BC_ACTIVE),
    left_actuated_joints=_BC_ACTIVE,
    right_actuated_joints=tuple(name.replace("left_", "right_") for name in _BC_ACTIVE),
    left_model_joints=tuple(
        j.get("name")
        for j in _BRAINCO.findall("worldbody/.//joint")
        if j.get("name").startswith("left_") and not j.get("name").endswith("_splay_joint")
    ),
    right_model_joints=tuple(
        j.get("name")
        for j in _BRAINCO.findall("worldbody/.//joint")
        if j.get("name").startswith("right_") and not j.get("name").endswith("_splay_joint")
    ),
    left_mimic_map={key: value for key, value in _BC_MIMIC.items() if key.startswith("left_")},
    right_mimic_map={key: value for key, value in _BC_MIMIC.items() if key.startswith("right_")},
    qpos_reorder_from_gr00t=None,
    left_hand_uid="brainco_left",
    right_hand_uid="brainco_right",
    mjcf_path=DEX3_HAND.mjcf_path,
    wbc_yaml_path=DEX3_HAND.wbc_yaml_path,
    wbc_version="sonic_model12_brainco",
    cu_robot_yaml=None,
    hand_yaml=None,
    usd_path="robots/g1/g1_29dof_wholebody_brainco.usd",
    left_ee_link="left_hand_base_link",
    right_ee_link="right_hand_base_link",
    init_qpos=tuple(
        lower + value * (upper - lower)
        for (lower, upper), value in zip(
            _BC_RANGES, _brainco_values("brainco_open_normalized"), strict=True
        )
    ),
    close_qpos_left=_BC_CLOSE,
    close_qpos_right=_BC_CLOSE,
    kp_map={
        name: float(_BRAINCO.find(f"actuator/position[@joint='{name}']").get("kp"))
        for name in _BC_ACTIVE
    },
    kd=float(_BC_POSITION.get("kv")),
    contact_collision_prefix="collision_brainco_",
    contact_solref=tuple(map(float, _BC_CONTACT.get("solref").split())),
    contact_friction=(
        _BC_FRICTION[0],
        _BC_FRICTION[0],
        _BC_FRICTION[1],
        _BC_FRICTION[2],
        _BC_FRICTION[2],
    ),
    contact_primary_condim=int(_BC_CONTACT.get("condim")),
    contact_other_condim=int(_BC_CONTACT.get("condim")),
    physics_dt=float(_BRAINCO.find("option").get("timestep")),
    reference_velocity=_brainco_values("brainco_reference_velocity"),
    reference_open_velocity=_brainco_values("brainco_reference_open_velocity"),
    reference_acceleration=tuple(
        value / _brainco_values("brainco_acceleration_time")[0]
        for value in _brainco_values("brainco_reference_velocity")
    ),
    binary_transition_duration=_brainco_values("brainco_binary_transition_duration")[0],
    effort_limits=tuple(
        float(_BRAINCO.find(f"actuator/position[@joint='{name}']").get("forcerange").split()[1])
        for name in _BC_ACTIVE
    ),
)


# ── Registry ──────────────────────────────────────────────────────────────

_HAND_REGISTRY: dict[str, HandSpec] = {
    "dex3": DEX3_HAND,
    "inspire": INSPIRE_HAND,
    "brainco": BRAINCO_HAND,
}


def get_hand_spec(hand_type: str) -> HandSpec:
    """Look up a HandSpec by name."""
    if hand_type not in _HAND_REGISTRY:
        raise KeyError(
            f"Unknown hand type: {hand_type!r}. Available: {list(_HAND_REGISTRY.keys())}"
        )
    return _HAND_REGISTRY[hand_type]


def register_hand_spec(name: str, spec: HandSpec) -> None:
    """Register a custom HandSpec so ``get_hand_spec(name)`` can find it."""
    _HAND_REGISTRY[name] = spec
