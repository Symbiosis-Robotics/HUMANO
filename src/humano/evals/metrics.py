"""Phase-3-only evaluation metrics for HUMANO whole-body rollouts."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

# FIXME(eval-metrics-validation): These fall thresholds are provisional heuristics.
# Validate them on labeled HUMANO rollouts that include normal walking, deep crouching,
# bending, chair sitting, true falls, and fall recovery before leaderboard use.
HARD_PELVIS_TILT_DEG = 75.0
HARD_TILT_SECONDS = 0.1
CORE_GROUND_CONTACT_SECONDS = 0.5

_CORE_BODY_TOKENS = ("pelvis", "waist", "torso")
_GROUND_GEOM_NAMES = frozenset({"ground"})


@dataclass(frozen=True)
class FallEvent:
    """The first fall trigger observed in one policy-execution phase."""

    reasons: tuple[str, ...]
    pelvis_tilt_deg: float
    hard_tilt_seconds: float
    core_ground_contact_seconds: float
    core_ground_contact_body: str | None


@dataclass(frozen=True)
class FallDetector:
    """Immutable duration-gated fall detector updated once per Phase-3 step."""

    sample_dt_s: float
    hard_tilt_seconds: float = 0.0
    core_ground_contact_seconds: float = 0.0
    event: FallEvent | None = None

    def __post_init__(self) -> None:
        if not np.isfinite(self.sample_dt_s) or self.sample_dt_s <= 0.0:
            raise ValueError("sample_dt_s must be a positive finite value")

    @property
    def fall_detected(self) -> bool:
        return self.event is not None

    def update(
        self,
        pelvis_quaternion_wxyz: np.ndarray,
        core_ground_contact_body: str | None,
    ) -> tuple[FallDetector, FallEvent | None]:
        """Return the next detector state and a newly-triggered event, if any."""
        if self.event is not None:
            return replace(self), None

        tilt_deg = pelvis_tilt_degrees(pelvis_quaternion_wxyz)
        hard_seconds = (
            self.hard_tilt_seconds + self.sample_dt_s if tilt_deg > HARD_PELVIS_TILT_DEG else 0.0
        )
        contact_seconds = (
            self.core_ground_contact_seconds + self.sample_dt_s
            if core_ground_contact_body is not None
            else 0.0
        )

        reasons = tuple(
            reason
            for triggered, reason in (
                (hard_seconds >= HARD_TILT_SECONDS, "hard_pelvis_tilt"),
                (
                    contact_seconds >= CORE_GROUND_CONTACT_SECONDS,
                    "sustained_core_ground_contact",
                ),
            )
            if triggered
        )
        event = (
            FallEvent(
                reasons=reasons,
                pelvis_tilt_deg=tilt_deg,
                hard_tilt_seconds=hard_seconds,
                core_ground_contact_seconds=contact_seconds,
                core_ground_contact_body=core_ground_contact_body,
            )
            if reasons
            else None
        )
        return (
            replace(
                self,
                hard_tilt_seconds=hard_seconds,
                core_ground_contact_seconds=contact_seconds,
                event=event,
            ),
            event,
        )


def pelvis_tilt_degrees(quaternion_wxyz: np.ndarray) -> float:
    """Return the angle between pelvis up and world up for a MuJoCo quaternion."""
    quaternion = np.asarray(quaternion_wxyz, dtype=np.float64).reshape(-1)
    if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
        raise ValueError("pelvis quaternion must contain four finite values")
    norm = float(np.linalg.norm(quaternion))
    if norm <= 0.0:
        raise ValueError("pelvis quaternion must have non-zero norm")
    _, x, y, _ = quaternion / norm
    pelvis_up_z = float(np.clip(1.0 - 2.0 * (x * x + y * y), -1.0, 1.0))
    return float(np.rad2deg(np.arccos(pelvis_up_z)))


def core_ground_contact_body(model: Any, data: Any) -> str | None:
    """Return the first pelvis/waist/torso body touching the MuJoCo ground."""
    for contact_index in range(int(data.ncon)):
        contact = data.contact[contact_index]
        geom_ids = (int(contact.geom1), int(contact.geom2))
        geom_names = tuple(str(model.geom(geom_id).name or "") for geom_id in geom_ids)
        for ground_side, geom_name in enumerate(geom_names):
            if geom_name not in _GROUND_GEOM_NAMES:
                continue
            robot_geom_id = geom_ids[1 - ground_side]
            body_id = int(model.geom_bodyid[robot_geom_id])
            body_name = str(model.body(body_id).name or "")
            if any(token in body_name.lower() for token in _CORE_BODY_TOKENS):
                return body_name
    return None


def dimensionless_jerk(
    joint_positions: np.ndarray,
    *,
    sample_dt_s: float,
) -> float | None:
    """Return mean duration/amplitude-normalized joint jerk; lower is smoother.

    This follows the dimensionless jerk family analyzed by Hogan and Sternad
    (2009): ``T**5 / A**2 * integral(jerk**2 dt)``. Each moving joint is
    normalized by its own peak-to-peak amplitude before scores are averaged.

    FIXME(eval-metrics-validation): Validate the episode-wide joint aggregation
    against human smoothness ratings and phase-segmented scores. Complex whole-body
    tasks contain gait, contact, and intentional stops that can inflate one global
    jerk score even when each task phase is well controlled.
    """
    positions = np.asarray(joint_positions, dtype=np.float64)
    if positions.ndim != 2:
        raise ValueError("joint_positions must have shape [steps, joints]")
    if not np.isfinite(sample_dt_s) or sample_dt_s <= 0.0:
        raise ValueError("sample_dt_s must be a positive finite value")
    if positions.shape[0] < 7 or positions.shape[1] == 0:
        return None
    if not np.all(np.isfinite(positions)):
        raise ValueError("joint_positions must contain only finite values")

    amplitudes = np.ptp(positions, axis=0)
    moving = amplitudes > 1e-6
    if not np.any(moving):
        return None

    velocity = np.gradient(positions[:, moving], sample_dt_s, axis=0, edge_order=2)
    acceleration = np.gradient(velocity, sample_dt_s, axis=0, edge_order=2)
    jerk = np.gradient(acceleration, sample_dt_s, axis=0, edge_order=2)
    squared_jerk = jerk * jerk
    jerk_energy = sample_dt_s * (
        0.5 * squared_jerk[0] + np.sum(squared_jerk[1:-1], axis=0) + 0.5 * squared_jerk[-1]
    )
    duration_s = (positions.shape[0] - 1) * sample_dt_s
    scores = duration_s**5 * jerk_energy / np.square(amplitudes[moving])
    score = float(np.mean(scores))
    return 0.0 if score < 1e-20 else score


def summarize_episode_metrics(
    episode_metrics: dict[str, dict[str, Any]],
) -> dict[str, float | None]:
    """Aggregate fall rate over all episodes and quality over successes only."""
    if not episode_metrics:
        return {
            "fall_rate": None,
            "timeout_rate": None,
            "median_success_completion_time_s": None,
            "median_success_dimensionless_jerk": None,
        }

    records = tuple(episode_metrics.values())
    successful = tuple(record for record in records if bool(record["success"]))
    completion_times = tuple(
        float(record["completion_time_s"])
        for record in successful
        if record.get("completion_time_s") is not None
    )
    jerk_scores = tuple(
        float(record["dimensionless_jerk"])
        for record in successful
        if record.get("dimensionless_jerk") is not None
    )
    return {
        "fall_rate": sum(bool(record["fall_detected"]) for record in records) / len(records),
        "timeout_rate": sum(bool(record["timed_out"]) for record in records)
        / len(records),
        "median_success_completion_time_s": (
            float(np.median(completion_times)) if completion_times else None
        ),
        "median_success_dimensionless_jerk": (
            float(np.median(jerk_scores)) if jerk_scores else None
        ),
    }


def episode_timed_out(
    *,
    success: bool,
    env_truncated: bool,
    reached_step_limit: bool,
) -> bool:
    """Return whether an unsuccessful Phase-3 episode ended by its time limit."""
    return not success and (env_truncated or reached_step_limit)


def summarize_evaluation_health(
    *,
    expected_episode_ids: tuple[str, ...],
    episode_metrics: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Summarize evaluation completeness separately from policy metrics."""
    missing_episode_ids = sorted(set(expected_episode_ids) - set(episode_metrics))
    termination_reasons = tuple(
        str(record["termination_reason"])
        for record in episode_metrics.values()
        if record.get("termination_reason") is not None
    )
    return {
        "requested_episodes": len(expected_episode_ids),
        "completed_episodes": len(episode_metrics),
        "missing_episodes": len(missing_episode_ids),
        "missing_episode_ids": missing_episode_ids,
        "is_complete": not missing_episode_ids,
        "termination_reason_counts": {
            reason: termination_reasons.count(reason)
            for reason in sorted(set(termination_reasons))
        },
    }
