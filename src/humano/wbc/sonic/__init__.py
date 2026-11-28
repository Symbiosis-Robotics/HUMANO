"""SONIC latent-token → motor-command controller (wbc/ implementation)."""

from humano.wbc.sonic.controller import SonicController
from humano.wbc.sonic.policy_parameters import (
    SONIC_STAND_PD_KD,
    SONIC_STAND_PD_KP,
)

SONIC_HANDOFF_BIRTH_ROOT_Z = 0.79275475
WBC_SONIC_WARMUP_WBC_S = 1.5
WBC_SONIC_WARMUP_READY_S = 0.5


__all__ = [
    "SONIC_STAND_PD_KD",
    "SONIC_STAND_PD_KP",
    "SonicController",
    "SONIC_HANDOFF_BIRTH_ROOT_Z",
    "WBC_SONIC_WARMUP_WBC_S",
    "WBC_SONIC_WARMUP_READY_S",
]
