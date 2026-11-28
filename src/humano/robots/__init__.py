"""HUMANO robot registrations."""

from .g1_sonic import G1Sonic
from .protocols import HeadCamMountable, Humanoid, WristCamMountable

__all__ = ["G1Sonic", "HeadCamMountable", "Humanoid", "WristCamMountable"]
