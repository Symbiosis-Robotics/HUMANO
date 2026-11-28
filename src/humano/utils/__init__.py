from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "NumpyArrayEncoder": "humano.utils.serialization",
    "Timer": "humano.utils.timing",
    "class_to_str": "humano.utils.naming",
    "configure_logging": "humano.utils.config",
    "dump_json": "humano.utils.serialization",
    "env_flag": "humano.utils.config",
    "get_data_dir": "humano.utils.paths",
    "get_resources_dir": "humano.utils.paths",
    "get_res_dir": "humano.utils.paths",
    "is_ffmpeg_installed": "humano.utils.system",
    "is_valid_rotation_matrix": "humano.utils.geometry",
    "load_yaml": "humano.utils.config",
    "resolve_data_path": "humano.utils.paths",
    "resolve_res_path": "humano.utils.paths",
    "snake_to_pascal": "humano.utils.naming",
    "timestamp_str": "humano.utils.timing",
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
