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
    "HssdSuite": "humano.scenes.hssd",
    "IndexedBackgroundScene": "humano.scenes.indexed",
    "SceneManager": "humano.scenes.scene_manager",
    "ShowHouse": "humano.scenes.show_house",
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
