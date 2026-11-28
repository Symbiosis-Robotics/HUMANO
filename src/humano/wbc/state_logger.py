"""
Ring-buffer state logger shared by all wbc/ implementations.

Mirrors the C++ StateLogger API (log_full_state → log_post_state → get_latest)
with the same joint-ordering convention used by GatherRobotStateToLogger().
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np


@dataclass
class StateLoggerEntry:
    """Single control-tick snapshot — matches the relevant fields of C++ StateLogger::Entry."""

    base_quat: np.ndarray = field(
        default_factory=lambda: np.zeros(4, dtype=np.float64)
    )  # [qw, qx, qy, qz]
    base_ang_vel: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64)
    )  # [wx, wy, wz]
    body_q: np.ndarray = field(
        default_factory=lambda: np.zeros(29, dtype=np.float64)
    )  # [29] in hardware order, default_angles already subtracted
    body_dq: np.ndarray = field(
        default_factory=lambda: np.zeros(29, dtype=np.float64)
    )  # [29] in hardware order
    last_action: np.ndarray = field(
        default_factory=lambda: np.zeros(29, dtype=np.float64)
    )  # [29] previous tick's policy action (isaaclab order)
    token_state: np.ndarray | None = None  # set by log_post_state()


class StateLogger:
    """Thread-safe-free ring buffer for robot state history.

    Parameters
    ----------
    capacity:
        Maximum number of entries kept in the ring.  Default 200 covers
        ~4 s of history at 50 Hz.
    """

    def __init__(self, capacity: int = 200) -> None:
        self._ring: deque[StateLoggerEntry] = deque(maxlen=capacity)

    # ------------------------------------------------------------------
    # Write API
    # ------------------------------------------------------------------
    def log_full_state(
        self,
        base_quat: np.ndarray,
        base_ang_vel: np.ndarray,
        body_q: np.ndarray,
        body_dq: np.ndarray,
        last_action: np.ndarray,
    ) -> None:
        """Push a new entry to the ring buffer.

        ``body_q`` and ``body_dq`` are expected to already be in the
        C++-compatible scrambled order produced by
        :meth:`SonicController._remap_joints`.
        """
        entry = StateLoggerEntry(
            base_quat=base_quat.copy(),
            base_ang_vel=base_ang_vel.copy(),
            body_q=body_q.copy(),
            body_dq=body_dq.copy(),
            last_action=last_action.copy(),
        )
        self._ring.append(entry)

    def log_post_state(self, token_state: np.ndarray) -> None:
        """Append token data to the *latest* entry (C++ LogPostState)."""
        if self._ring:
            self._ring[-1].token_state = token_state.copy()

    # ------------------------------------------------------------------
    # Read API
    # ------------------------------------------------------------------
    def get_latest(
        self,
        n: int,
        sample_dt_seconds: float,
        newest_first: bool = False,
    ) -> list[StateLoggerEntry]:
        """Return exactly *n* historical entries sampled at *sample_dt* intervals.

        Parameters
        ----------
        n:
            Number of entries to return.
        sample_dt_seconds:
            Sampling stride in seconds.  The actual stride is
            ``round(sample_dt / 0.02)`` ticks (assuming 50 Hz logging).
        newest_first:
            If ``False`` (default) returns ``[oldest, ..., newest]``,
            matching the C++ ``GetLatest(n, dt, newest_first=false)``
            convention used by all ``his_*`` observation gatherers.

        Returns
        -------
        list[StateLoggerEntry]
            Always returns exactly *n* entries.  Missing frames are filled
            with zero-initialised ``StateLoggerEntry`` objects, matching the
            C++ ``StateLogger::GetLatest`` + ``makeZeroEntry_()`` behaviour.
        """
        log_dt = 0.02  # fixed 50 Hz logging period
        step = max(1, int(round(sample_dt_seconds / log_dt)))
        # print(f"[StateLogger] newest_first: {newest_first}, sample_dt_seconds: {sample_dt_seconds}")

        # Walk backwards from newest, taking every step-th entry (newest-first).
        indices = list(range(len(self._ring) - 1, -1, -step))[:n]
        result = [self._ring[i] for i in indices]

        # Zero-pad to exactly n entries (matches C++ GetLatest + makeZeroEntry_).
        while len(result) < n:
            result.append(StateLoggerEntry())

        if newest_first:
            return result

        # Default: oldest-first.
        return list(reversed(result))

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def size(self) -> int:
        """Current number of entries in the buffer."""
        return len(self._ring)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def clear(self) -> None:
        """Discard all history (useful between episodes)."""
        self._ring.clear()
