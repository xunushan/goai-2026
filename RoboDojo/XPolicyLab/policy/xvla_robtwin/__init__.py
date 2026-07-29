"""X-VLA RoboTwin2 adapter without eager policy-model imports.

The Isaac/RoboDojo eval client imports ``xvla_robtwin.deploy`` only for the
episode-driving functions. Importing the policy model here would incorrectly
force the simulation environment to install torch/transformers/timm. Resolve
the model exports lazily so only the policy-server process needs those
dependencies.
"""

from __future__ import annotations

from typing import Any

__all__ = ["Model", "get_model"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from .model import Model, get_model

        return {"Model": Model, "get_model": get_model}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
