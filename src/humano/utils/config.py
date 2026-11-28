from __future__ import annotations

import builtins
import logging
import os
import re
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

_HANDLER_TAG = "_humano_logging_handler"
_FORMAT = "[%(asctime)s] %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def _message_from_print_args(*args: Any, **kwargs: Any) -> str:
    sep = kwargs.get("sep", " ")
    end = kwargs.get("end", "\n")
    return (sep.join(str(arg) for arg in args) + end).rstrip("\n")


def env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def load_yaml(file_path: Any) -> dict:
    """Load yaml file and return as dictionary. If file_path is a dictionary, return as is."""

    import yaml
    from yaml import SafeLoader as Loader

    Loader.add_implicit_resolver(
        "tag:yaml.org,2002:float",
        re.compile(
            """^(?:
        [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
        |[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$""",
            re.X,
        ),
        list("-+0123456789."),
    )
    if isinstance(file_path, str):
        with open(file_path) as file_p:
            yaml_params = yaml.load(file_p, Loader=Loader)
    else:
        yaml_params = file_path
    return yaml_params


def _parse_log_level(level: str | int | None) -> int:
    if isinstance(level, int):
        return level
    if level is None:
        level = os.environ.get("HUMANO_LOG_LEVEL", "INFO")
    if isinstance(level, str):
        parsed = logging.getLevelName(level.upper())
        if isinstance(parsed, int):
            return parsed
    return logging.INFO


def configure_logging(
    *,
    level: str | int | None = None,
    log_file: str | os.PathLike[str] | None = None,
    console: bool = True,
) -> logging.Logger:
    resolved_level = _parse_log_level(level)
    resolved_log_file = log_file
    if resolved_log_file is None:
        resolved_log_file = os.environ.get("HUMANO_LOG_FILE")

    root_logger = logging.getLogger()
    root_logger.setLevel(resolved_level)
    formatter = logging.Formatter(_FORMAT, datefmt=_DATEFMT)

    for handler in list(root_logger.handlers):
        if getattr(handler, _HANDLER_TAG, False):
            root_logger.removeHandler(handler)
            handler.close()

    if console:
        stream = sys.__stdout__ or sys.stdout
        stream_handler = logging.StreamHandler(stream)
        stream_handler.setFormatter(formatter)
        stream_handler.setLevel(resolved_level)
        setattr(stream_handler, _HANDLER_TAG, True)
        root_logger.addHandler(stream_handler)

    if resolved_log_file:
        path = Path(resolved_log_file).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, mode="a")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(resolved_level)
        setattr(file_handler, _HANDLER_TAG, True)
        root_logger.addHandler(file_handler)

    return logging.getLogger("humano")


@contextmanager
def route_prints_to_logger(logger: logging.Logger) -> Iterator[None]:
    previous_print = builtins.print

    def log_print(*args: Any, **kwargs: Any) -> None:
        file = kwargs.get("file")
        if file not in (None, sys.stdout, sys.stderr):
            previous_print(*args, **kwargs)
            return
        logger.info(_message_from_print_args(*args, **kwargs))

    builtins.print = log_print
    try:
        yield
    finally:
        builtins.print = previous_print


@contextmanager
def route_stderr_to_logger(logger: logging.Logger, level: int = logging.WARNING) -> Iterator[None]:
    saved_stderr_fd = os.dup(2)
    capture_fd: int | None = None
    for handler in logger.root.handlers:
        if getattr(handler, _HANDLER_TAG, False) and isinstance(handler, logging.FileHandler):
            capture_fd = os.open(
                handler.baseFilename,
                os.O_APPEND | os.O_CREAT | os.O_WRONLY,
                0o666,
            )
            break
    try:
        if capture_fd is not None:
            os.dup2(capture_fd, 2)
        yield
    finally:
        try:
            sys.stderr.flush()
        except Exception:
            pass
        os.dup2(saved_stderr_fd, 2)
        if capture_fd is not None:
            os.close(capture_fd)
        os.close(saved_stderr_fd)
