"""LeRobot X-VLA v1 adapter for the RoboDojo policy server."""

from __future__ import annotations

from typing import Any

__all__ = ["Model", "get_model"]


def __getattr__(name: str) -> Any:
    # The simulator only imports deploy.py and must not need the policy's
    # torch/transformers dependencies.
    if name in __all__:
        from .model import Model, get_model

        return {"Model": Model, "get_model": get_model}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
