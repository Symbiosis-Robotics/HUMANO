from __future__ import annotations

import logging
from typing import Any


def initialize_overwatch(name: str) -> logging.Logger:
    return logging.getLogger(name)


def count_parameters(module: Any) -> int:
    return sum(parameter.numel() for parameter in module.parameters())
