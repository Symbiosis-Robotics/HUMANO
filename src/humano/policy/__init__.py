"""Policy adapters used by HUMANO evaluation."""

from typing import Any

__all__ = ["Gr00tN17Agent", "SonicPolicyAgent"]


def __getattr__(name: str) -> Any:
    if name == "Gr00tN17Agent":
        from .gr00t_n17 import Gr00tN17Agent

        return Gr00tN17Agent
    if name == "SonicPolicyAgent":
        from .sonic_policy import SonicPolicyAgent

        return SonicPolicyAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
