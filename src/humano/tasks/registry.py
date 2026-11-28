"""
HUMANO: Humanoid Benchmark and Learning Framework

Originally derived from the SIMPLE project.
Copyright (c) 2025 Songlin Wei and Contributors
Copyright (c) 2026 HUMANO Contributors
Licensed under the terms in LICENSE file.
"""

from typing import Type

from humano.core.registry import RegistryMixin
from humano.core.task import Task


class TaskRegistry(RegistryMixin[Task]):
    @classmethod
    def _base_type(cls) -> type:
        return Task

    @classmethod
    def make(cls, uid: str, *args, **kwargs) -> Task:
        if uid not in cls._registry:
            raise ValueError(f"No task registered under uid '{uid}'")
        return cls._registry[uid](*args, **kwargs)
