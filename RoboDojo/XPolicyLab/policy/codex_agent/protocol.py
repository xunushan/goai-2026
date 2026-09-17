"""Convert the validated model reply into arm commands."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .motion import ArmCommand, normalize_quat_wxyz


class ParseError(ValueError):
    pass


@dataclass(frozen=True)
class ParsedDecision:
    left: ArmCommand
    right: ArmCommand


def _vector(value: Any, length: int, name: str) -> np.ndarray | None:
    if value == "keep":
        return None
    try:
        vector = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ParseError(f"{name} must contain {length} finite numbers") from exc
    if vector.shape != (length,) or not np.isfinite(vector).all():
        raise ParseError(f"{name} must contain {length} finite numbers")
    return vector


def _arm(value: Any, side: str, gripper_open: float, gripper_close: float) -> ArmCommand:
    if not isinstance(value, dict) or set(value) != {"position", "orientation", "gripper"}:
        raise ParseError(f"{side} must contain position, orientation and gripper")
    position = _vector(value["position"], 3, f"{side}.position")
    orientation = _vector(value["orientation"], 4, f"{side}.orientation")
    if orientation is not None:
        try:
            orientation = normalize_quat_wxyz(orientation)
        except ValueError as exc:
            raise ParseError(f"{side}.orientation is not a valid quaternion") from exc
    gripper = value["gripper"]
    if gripper == "keep":
        target_gripper = None
    elif gripper == "open":
        target_gripper = float(gripper_open)
    elif gripper == "close":
        target_gripper = float(gripper_close)
    else:
        raise ParseError(f'{side}.gripper must be "keep", "open" or "close"')
    return ArmCommand(position=position, quat=orientation, gripper=target_gripper)


def parse_decision(
    payload: Any,
    *,
    gripper_open: float = 1.0,
    gripper_close: float = 0.0,
) -> ParsedDecision:
    if not isinstance(payload, dict):
        raise ParseError("decision must be an object")
    return ParsedDecision(
        left=_arm(payload.get("left"), "left", gripper_open, gripper_close),
        right=_arm(payload.get("right"), "right", gripper_open, gripper_close),
    )
