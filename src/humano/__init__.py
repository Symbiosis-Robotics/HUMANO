"""
HUMANO runtime package.
"""

__version__ = "0.1.0"
__all__ = [
    "__version__",
    "EvalConfig",
    "EvalResult",
    "EvalRunner",
]


def __getattr__(name: str):
    if name in {"EvalConfig", "EvalResult", "EvalRunner"}:
        from humano.evals.api import EvalConfig, EvalResult, EvalRunner

        exports = {
            "EvalConfig": EvalConfig,
            "EvalResult": EvalResult,
            "EvalRunner": EvalRunner,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
