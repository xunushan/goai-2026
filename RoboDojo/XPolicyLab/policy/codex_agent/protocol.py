"""Parsing and validation of the Codex decision payload.

The model is asked for a small JSON object (see :mod:`prompt`). Everything that
can go wrong with an LLM-produced payload is handled here, tolerantly but
loudly: every accepted field is validated, every rejected field carries a reason
that gets fed back to the model on the next turn.

Accepted shape::

    {"left":  {"position": "keep" | [x, y, z],
               "orientation": "keep" | [yaw, pitch, roll] | {"quat": [w, x, y, z]},
               "gripper": "keep" | "open" | "close" | <float>},
     "right": {...},
     "note": "...",
     "phase": "..."}

``orientation`` as a 3-list is **relative to the HOME orientation** (the pose the
arms start every episode in) and is resolved to an absolute quaternion here;
as a 4-list or ``{"quat": [...]}`` it is an absolute ``wxyz`` quaternion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .motion import (
    ArmCommand,
    KEEP,
    absolute_quat_from_relative_rpy,
    normalize_quat_wxyz,
)

KEEP_TOKENS = {"keep", "hold", "same", "unchanged", "none", "null", ""}
OPEN_TOKENS = {"open", "release", "released", "opened"}
CLOSE_TOKENS = {"close", "closed", "grasp", "grip", "grab"}


class ParseError(ValueError):
    """The payload could not be understood at all."""


@dataclass
class ParsedDecision:
    left: ArmCommand = KEEP
    right: ArmCommand = KEEP
    note: str = ""
    phase: str = ""
    problems: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def is_noop(self) -> bool:
        """Both arms asked to keep everything: legal, but a wasted decision."""
        return self.left.is_keep and self.right.is_keep


def _as_keep(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in KEEP_TOKENS
    return False


def _flat_floats(value: Any) -> list[float] | None:
    """Flatten a JSON value into finite floats, or ``None`` if it isn't numeric."""
    if isinstance(value, bool) or not isinstance(value, (list, tuple, np.ndarray)):
        return None
    try:
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(arr).all():
        return None
    return [float(v) for v in arr]


def _to_float_list(value: Any, length: int, what: str) -> list[float]:
    values = _flat_floats(value)
    if values is None:
        raise ParseError(f"{what} is not a list of {length} finite numbers: {value!r}")
    if len(values) != length:
        raise ParseError(f"{what} must have {length} numbers, got {len(values)}")
    return values


def _parse_position(value: Any, arm: str, problems: list[str]) -> np.ndarray | None:
    if _as_keep(value):
        return None
    if isinstance(value, dict):
        for key in ("xyz", "position", "pos", "value"):
            if key in value:
                value = value[key]
                break
        else:
            problems.append(f"{arm}.position: unrecognised mapping {sorted(value)}")
            return None
    try:
        return np.asarray(_to_float_list(value, 3, f"{arm}.position"), dtype=np.float64)
    except ParseError as exc:
        problems.append(str(exc))
        return None


def _parse_orientation(
    value: Any, arm: str, home_quat: np.ndarray, problems: list[str]
) -> np.ndarray | None:
    """``keep`` | relative ``[r,p,y]`` | absolute ``[w,x,y,z]`` / ``{"quat": [...]}``."""
    if _as_keep(value):
        return None
    if isinstance(value, dict):
        for key in ("quat", "quaternion", "absolute", "value"):
            if key in value:
                value = value[key]
                break
        else:
            problems.append(f"{arm}.orientation: unrecognised mapping {sorted(value)}")
            return None
    values = _flat_floats(value) or []
    if len(values) == 4:
        try:
            return normalize_quat_wxyz(values)
        except ValueError as exc:
            problems.append(f"{arm}.orientation: invalid quaternion ({exc})")
            return None
    if len(values) == 3:
        try:
            return absolute_quat_from_relative_rpy(values, home_quat)
        except ValueError as exc:
            problems.append(f"{arm}.orientation: invalid rpy ({exc})")
            return None
    problems.append(
        f"{arm}.orientation must be \"keep\", 3 relative yaw/pitch/roll numbers, "
        f"or 4 absolute quaternion numbers; got {value!r}"
    )
    return None


def _parse_gripper(
    value: Any, arm: str, gripper_open: float, gripper_close: float, problems: list[str]
) -> float | None:
    if _as_keep(value):
        return None
    if isinstance(value, str):
        token = value.strip().lower()
        if token in OPEN_TOKENS:
            return float(gripper_open)
        if token in CLOSE_TOKENS:
            return float(gripper_close)
        problems.append(f'{arm}.gripper must be "keep", "open", "close" or a number; got {value!r}')
        return None
    if isinstance(value, bool):
        problems.append(f"{arm}.gripper must not be a boolean")
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        problems.append(f"{arm}.gripper is not a number: {value!r}")
        return None
    if not np.isfinite(number):
        problems.append(f"{arm}.gripper is not finite")
        return None
    return float(np.clip(number, 0.0, 1.0))


def _parse_arm(
    value: Any,
    arm: str,
    home_quat: np.ndarray,
    gripper_open: float,
    gripper_close: float,
    problems: list[str],
) -> ArmCommand:
    if _as_keep(value):
        return KEEP
    if not isinstance(value, dict):
        problems.append(f"{arm} must be an object or \"keep\", got {type(value).__name__}")
        return KEEP
    unknown = set(value) - {"position", "orientation", "rotation", "quat", "gripper"}
    if unknown:
        problems.append(f"{arm} has unrecognised keys {sorted(unknown)}")
    orientation = value.get("orientation", value.get("rotation", value.get("quat", "keep")))
    return ArmCommand(
        position=_parse_position(value.get("position", "keep"), arm, problems),
        quat=_parse_orientation(orientation, arm, home_quat, problems),
        gripper=_parse_gripper(
            value.get("gripper", "keep"), arm, gripper_open, gripper_close, problems
        ),
    )


def parse_decision(
    payload: Any,
    *,
    home_quat_left: np.ndarray,
    home_quat_right: np.ndarray,
    gripper_open: float = 1.0,
    gripper_close: float = 0.0,
) -> ParsedDecision:
    """Validate a decoded decision payload.

    Raises :class:`ParseError` only when the payload is not an object at all;
    per-field problems are collected in ``problems`` and the offending field
    falls back to *keep* so the episode can continue.
    """
    if not isinstance(payload, dict):
        raise ParseError(f"expected a JSON object, got {type(payload).__name__}")
    problems: list[str] = []
    left = _parse_arm(
        payload.get("left", "keep"), "left", home_quat_left, gripper_open, gripper_close, problems
    )
    right = _parse_arm(
        payload.get("right", "keep"), "right", home_quat_right, gripper_open, gripper_close,
        problems,
    )
    note = payload.get("note", "")
    phase = payload.get("phase", "")
    return ParsedDecision(
        left=left,
        right=right,
        note=str(note)[:1000],
        phase=str(phase)[:120],
        problems=problems,
        raw=payload,
    )
