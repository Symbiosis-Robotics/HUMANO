"""
HUMANO: Humanoid Benchmark and Learning Framework

Originally derived from the SIMPLE project.
Copyright (c) 2025 Songlin Wei and Contributors
Copyright (c) 2026 HUMANO Contributors
Licensed under the terms in LICENSE file.
"""

from importlib import import_module
from typing import Any

_EXPORTS = {
    "ArticulatedObjectDr": "humano.dr.articulate_object",
    "ArticulatedObjectDrCfg": "humano.dr.articulate_object",
    "CameraDR": "humano.dr.camera",
    "CameraDRCfg": "humano.dr.camera",
    "DistractorDR": "humano.dr.distractor",
    "DistractorDRCfg": "humano.dr.distractor",
    "LanguageDR": "humano.dr.language",
    "LanguageDRCfg": "humano.dr.language",
    "LightingDR": "humano.dr.lighting",
    "LightingDRCfg": "humano.dr.lighting",
    "MaterialDR": "humano.dr.material",
    "MaterialDRCfg": "humano.dr.material",
    "SceneDR": "humano.dr.scene",
    "SpatialDR": "humano.dr.spatial",
    "SpatialDRCfg": "humano.dr.spatial",
    "TabletopSceneDR": "humano.dr.scene",
    "TabletopSceneDRCfg": "humano.dr.scene",
    "TargetDR": "humano.dr.target",
    "TargetDRCfg": "humano.dr.target",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
