"""
HUMANO: Humanoid Benchmark and Learning Framework

Originally derived from the SIMPLE project.
Copyright (c) 2025 Songlin Wei and Contributors
Copyright (c) 2026 HUMANO Contributors
Licensed under the terms in LICENSE file.
"""

# from __future__ import annotations
from typing import List, Protocol, runtime_checkable

import numpy as np

# class Point:


class Box:
    low: float | list[float]
    high: float | list[float]

    def __init__(self, low, high) -> None:
        self.low = low
        self.high = high

    def uniform(self) -> float | list[float]:
        sampled = np.random.uniform(self.low, self.high)
        if (
            isinstance(self.low, list)
            or isinstance(self.low, tuple)
            or isinstance(self.low, np.ndarray)
        ):
            return sampled.tolist()
        else:
            return float(sampled)

    def sample(self) -> float | list[float]:
        return self.uniform()

    def middle(self) -> float | list[float]:
        if (
            isinstance(self.low, list)
            or isinstance(self.low, tuple)
            or isinstance(self.low, np.ndarray)
        ):
            return [(l + h) / 2.0 for l, h in zip(self.low, self.high)]
        else:
            return (self.low + self.high) / 2.0
