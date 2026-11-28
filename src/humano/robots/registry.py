"""
HUMANO: Humanoid Benchmark and Learning Framework

Originally derived from the SIMPLE project.
Copyright (c) 2025 Songlin Wei and Contributors
Copyright (c) 2026 HUMANO Contributors
Licensed under the terms in LICENSE file.
"""

from typing import TYPE_CHECKING, ClassVar, Type

from humano.core.registry import RegistryMixin
from humano.core.robot import Robot

if TYPE_CHECKING:
    from humano.robots.hands import HandSpec


class RobotRegistry(RegistryMixin[Robot]):
    @classmethod
    def _base_type(cls) -> type:
        return Robot

    @classmethod
    def make(cls, uid: str, *args, hand_spec: "HandSpec | None" = None, **kwargs) -> Robot:
        if hand_spec is None:
            return super().make(uid, *args, **kwargs)
        # Use (uid, hand_spec.name) as cache key so different hand types
        # get different instances even with the same robot uid.
        instance_key = (uid, hand_spec.name)
        if instance_key not in cls._instances:
            cls._instances[instance_key] = cls._registry[uid](
                *args, hand_spec=hand_spec, **kwargs
            )
        return cls._instances[instance_key]

    # _registry: ClassVar[dict[str, type[Robot]]] = {}
    # _instances: ClassVar[dict[str, Robot]] = {}

    # @classmethod
    # def register(cls, name: str):
    #     def wrapper(subclass: type[Robot]):
    #         if not issubclass(subclass, Robot):
    #             raise TypeError(f"{subclass.__name__} must inherit from {Robot.__name__}")
    #         cls._registry[name] = subclass
    #         return subclass
    #     return wrapper

    # @classmethod
    # def make(cls, res_id: str, *args, **kwargs) -> 'Robot':
    #     """Get the Robot instance for a specific Robot ID."""

    #     if res_id not in cls._registry:
    #         raise ValueError(f"No Robot registered for Robot UID '{res_id}'")

    #     if res_id not in cls._instances:
    #         # Create and cache the singleton instance
    #         cls._instances[res_id] = cls._registry[res_id](*args, **kwargs)

    #     return cls._instances[res_id]
