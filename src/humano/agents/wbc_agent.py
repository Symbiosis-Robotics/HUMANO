"""
HUMANO: Humanoid Benchmark and Learning Framework

Originally derived from the SIMPLE project.
Copyright (c) 2025 Songlin Wei and Contributors
Copyright (c) 2026 HUMANO Contributors
Licensed under the terms in LICENSE file.
"""

from abc import ABC, abstractmethod


class WholeBodyControlAgent(ABC):
    ...

    @abstractmethod
    def get_action(self, observation, instruction=None, **kwargs):
        raise NotImplementedError

    def reset(self, **kwargs): ...
