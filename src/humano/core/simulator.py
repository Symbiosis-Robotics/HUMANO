"""
HUMANO: Humanoid Benchmark and Learning Framework

Originally derived from the SIMPLE project.
Copyright (c) 2025 Songlin Wei and Contributors
Copyright (c) 2026 HUMANO Contributors
Licensed under the terms in LICENSE file.
"""

from abc import ABC
from typing import Any, Dict, List, Protocol, Union

import numpy as np

from humano.utils.paths import (
    get_data_dir as _get_data_dir,
)
from humano.utils.paths import (
    resolve_data_path as _resolve_data_path,
)
from humano.utils.paths import (
    resolve_res_path as _resolve_res_path,
)


class Simulator(ABC):
    def update_layout(self) -> None: ...

    def set_states(self, states: dict[str, Any]) -> None: ...

    def get_states(self) -> dict[str, Any]: ...

    def step(self) -> dict[str, np.ndarray] | None: ...

    def render(self, *args, **kwargs) -> dict[str, np.ndarray]: ...

    def resolve_res_path(self, rel_path=None) -> str:
        return _resolve_res_path(rel_path)

    def get_res_dir(self) -> str:
        return str(_resolve_res_path())

    def get_data_dir(self) -> str:
        return str(_get_data_dir())

    def resolve_data_path(self, rel_path=None, create_if_not_exist=False) -> str:
        return _resolve_data_path(rel_path, create_if_not_exist=create_if_not_exist)
