from __future__ import annotations

import re


def class_to_str(cls: type) -> str:
    return f"{cls.__module__}.{cls.__qualname__}"


def snake_to_pascal(snake_str: str) -> str:
    parts = re.split(r"[-_]", snake_str)
    return "".join([parts[0].capitalize()] + [p.capitalize() for p in parts[1:]])
