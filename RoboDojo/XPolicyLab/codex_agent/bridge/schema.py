"""Strict wire validation for the observation packet and the model's reply.

Two halves of the same contract:

* ``Observation.parse`` refuses a request that does not say exactly what the turn
  text needs. The adapter is the only producer, so a malformed packet is a bug on
  the GPU side, and finding out here -- with a 400 and a sentence naming the field
  -- is much cheaper than finding out from a Codex turn that was spent on a
  half-empty prompt.
* ``output_schema`` / ``validate_response`` are the other direction. The schema is
  what the App Server constrains the model to; the validator is what we check
  before believing it. Both exist because a schema is a hint and not a guarantee.

The shape the model must return is the one the adapter's ``protocol.parse_decision``
already understands -- ``{"left": .., "right": .., "note": .., "phase": ..}`` with
``keep`` / list / ``open``|``close`` spellings -- so validating here does not
narrow what the adapter can accept.
"""

from __future__ import annotations

import base64
import binascii
import math
from dataclasses import dataclass
from typing import Any

ARMS = ("left", "right")

# Bounds on what one request may carry. Generous on purpose -- the images ride a
# tunnel at their original resolution, so these are there to stop a runaway
# payload, not to shape a normal one.
MAX_IMAGE_BYTES = 4 * 1024 * 1024
MAX_TOTAL_IMAGE_BYTES = 12 * 1024 * 1024

# What each arm's current measured EEF pose carries.
_ARM_FIELDS = ("position", "orientation", "gripper")
_TASK_FIELDS = ("name", "instruction", "guidance")
_BUDGET_FIELDS = ("max_decisions", "max_sim_steps", "remaining_decisions", "remaining_steps")
_REQUIRED = ("episode_id", "request_id", "step_id", "turn_index", "task", "budget", "observation", "images")


class PolicyValidationError(ValueError):
    """A request or a model reply violates the public policy protocol."""


def _number(value: Any, name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise PolicyValidationError(f"{name} must be a finite number")
    return float(value)


def _vector(value: Any, length: int, name: str) -> list[float]:
    if not isinstance(value, list) or len(value) != length:
        raise PolicyValidationError(f"{name} must be a {length}-element array")
    return [_number(item, f"{name}[{index}]") for index, item in enumerate(value)]


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PolicyValidationError(f"{name} must be a non-empty string")
    return value


def _count(value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise PolicyValidationError(f"{name} must be a non-negative integer")
    return value


@dataclass(frozen=True)
class ImageInput:
    name: str
    mime: str
    data: bytes


@dataclass(frozen=True)
class ArmObservation:
    position: tuple[float, float, float]
    orientation: tuple[float, float, float, float]
    gripper: float


@dataclass(frozen=True)
class Observation:
    episode_id: str
    request_id: str
    step_id: int
    turn_index: int
    task: dict[str, Any]
    budget: dict[str, int]
    arms: dict[str, ArmObservation]
    feedback: tuple[str, ...]
    images: tuple[ImageInput, ...]
    vla_review: dict[str, Any] | None
    continuation: dict[str, Any] | None

    @classmethod
    def parse(cls, value: Any) -> "Observation":
        if not isinstance(value, dict):
            raise PolicyValidationError("request body must be an object")
        missing = [key for key in _REQUIRED if key not in value]
        if missing:
            raise PolicyValidationError(f"request missing fields: {sorted(missing)}")
        episode_id = _text(value["episode_id"], "episode_id")
        request_id = _text(value["request_id"], "request_id")

        task = value["task"]
        if not isinstance(task, dict):
            raise PolicyValidationError("task must be an object")
        for key in _TASK_FIELDS:
            if key not in task:
                raise PolicyValidationError(f"task must define {key!r}")
        if set(task) != set(_TASK_FIELDS):
            raise PolicyValidationError("task must contain exactly 'name', 'instruction' and 'guidance'")
        guidance = task["guidance"]
        if not isinstance(guidance, list) or not all(isinstance(line, str) and line.strip() for line in guidance):
            raise PolicyValidationError("task.guidance must be an array of non-empty strings")
        parsed_task = {
            "name": _text(task["name"], "task.name"),
            "instruction": _text(task["instruction"], "task.instruction"),
            "guidance": list(guidance),
        }

        budget = value["budget"]
        if not isinstance(budget, dict):
            raise PolicyValidationError("budget must be an object")
        for key in _BUDGET_FIELDS:
            if key not in budget:
                raise PolicyValidationError(f"budget must define {key!r}")
        parsed_budget = {key: _count(budget[key], f"budget.{key}") for key in _BUDGET_FIELDS}
        if parsed_budget["max_decisions"] < 1 or parsed_budget["max_sim_steps"] < 1:
            raise PolicyValidationError("budget limits must be at least 1")

        observation = value["observation"]
        if not isinstance(observation, dict) or set(observation) != set(ARMS):
            raise PolicyValidationError("observation must contain exactly left and right")

        feedback = value.get("feedback") or []
        if not isinstance(feedback, list) or not all(isinstance(line, str) for line in feedback):
            raise PolicyValidationError("feedback must be an array of strings")

        images = value["images"]
        if not isinstance(images, list) or not images:
            raise PolicyValidationError("images must be a non-empty array")
        parsed_images = tuple(_parse_image(entry, index) for index, entry in enumerate(images))
        if len({image.name for image in parsed_images}) != len(parsed_images):
            raise PolicyValidationError("images[].name values must be unique")
        total = sum(len(image.data) for image in parsed_images)
        if total > MAX_TOTAL_IMAGE_BYTES:
            raise PolicyValidationError(f"images total {total} bytes, over the {MAX_TOTAL_IMAGE_BYTES} limit")

        return cls(
            episode_id=episode_id,
            request_id=request_id,
            step_id=_count(value["step_id"], "step_id"),
            turn_index=_count(value["turn_index"], "turn_index"),
            task=parsed_task,
            budget=parsed_budget,
            arms={arm: _parse_arm(observation[arm], arm) for arm in ARMS},
            feedback=tuple(feedback),
            images=parsed_images,
            vla_review=_parse_vla_review(value.get("vla_review")),
            continuation=_parse_continuation(value.get("continuation")),
        )


def _parse_continuation(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != {"previous_request_id", "verify_previous"}:
        raise PolicyValidationError("continuation must contain previous_request_id and verify_previous")
    if type(value["verify_previous"]) is not bool:
        raise PolicyValidationError("continuation.verify_previous must be boolean")
    return {"previous_request_id": _text(value["previous_request_id"], "continuation.previous_request_id"),
            "verify_previous": value["verify_previous"]}


def _parse_vla_review(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != {"chunk", "summary", "gripper_change_threshold"}:
        raise PolicyValidationError("vla_review must contain chunk, summary and gripper_change_threshold")
    threshold = _number(value["gripper_change_threshold"], "vla_review.gripper_change_threshold")
    if not 0 <= threshold <= 1:
        raise PolicyValidationError("vla_review.gripper_change_threshold must be within [0,1]")
    chunk = value["chunk"]
    if not isinstance(chunk, dict) or set(chunk) != {"horizon", "left", "right"}:
        raise PolicyValidationError("vla_review.chunk must contain horizon, left and right")
    horizon = _count(chunk["horizon"], "vla_review.chunk.horizon")
    if horizon < 1:
        raise PolicyValidationError("vla_review.chunk.horizon must be positive")
    parsed_chunk: dict[str, Any] = {"horizon": horizon}
    for side in ARMS:
        arm = chunk[side]
        if not isinstance(arm, dict) or set(arm) != {"position", "orientation", "gripper"}:
            raise PolicyValidationError(f"vla_review.chunk.{side} has invalid fields")
        if not all(isinstance(arm[key], list) and len(arm[key]) == horizon for key in arm):
            raise PolicyValidationError(f"vla_review.chunk.{side} arrays must match horizon")
        parsed_chunk[side] = {
            "position": [_vector(row, 3, f"vla_review.chunk.{side}.position") for row in arm["position"]],
            "orientation": [_vector(row, 4, f"vla_review.chunk.{side}.orientation") for row in arm["orientation"]],
            "gripper": [_number(item, f"vla_review.chunk.{side}.gripper") for item in arm["gripper"]],
        }
    if not isinstance(value["summary"], dict):
        raise PolicyValidationError("vla_review.summary must be an object")
    return {"chunk": parsed_chunk, "summary": value["summary"], "gripper_change_threshold": threshold}


def _parse_arm(value: Any, arm: str) -> ArmObservation:
    if not isinstance(value, dict) or set(value) != set(_ARM_FIELDS):
        raise PolicyValidationError(f"observation.{arm} must contain exactly {sorted(_ARM_FIELDS)}")
    position = _vector(value["position"], 3, f"observation.{arm}.position")
    orientation = _vector(value["orientation"], 4, f"observation.{arm}.orientation")
    gripper = _number(value["gripper"], f"observation.{arm}.gripper")
    return ArmObservation(tuple(position), tuple(orientation), gripper)


def _parse_image(value: Any, index: int) -> ImageInput:
    if not isinstance(value, dict):
        raise PolicyValidationError(f"images[{index}] must be an object")
    name = value.get("name")
    if not isinstance(name, str) or not name.strip():
        raise PolicyValidationError(f"images[{index}].name must be a non-empty string")
    mime = value.get("mime")
    if mime not in ("image/jpeg", "image/png"):
        raise PolicyValidationError(f"images[{index}].mime must be image/jpeg or image/png")
    try:
        data = base64.b64decode(value.get("b64", ""), validate=True)
    except (binascii.Error, TypeError) as error:
        raise PolicyValidationError(f"images[{index}].b64 is invalid") from error
    if not data:
        raise PolicyValidationError(f"images[{index}].b64 is empty")
    if len(data) > MAX_IMAGE_BYTES:
        raise PolicyValidationError(f"images[{index}] is {len(data)} bytes, over the {MAX_IMAGE_BYTES} limit")
    return ImageInput(name=name, mime=mime, data=data)


# --------------------------------------------------------------------------- #
# reply schema
# --------------------------------------------------------------------------- #


def _arm_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": list(_ARM_FIELDS),
        "properties": {
            "position": {
                "anyOf": [
                    {"type": "string", "enum": ["keep"]},
                    {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3},
                ]
            },
            "orientation": {
                "anyOf": [
                    {"type": "string", "enum": ["keep"]},
                    {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4},
                ]
            },
            "gripper": {"type": "string", "enum": ["keep", "open", "close"]},
        },
    }


def output_schema(vla_review: bool = False) -> dict[str, Any]:
    """JSON Schema handed to the App Server's ``turn/start``.

    Inlined rather than using ``$defs``/``$ref`` so the schema stays within the
    subset structured-output backends reliably accept. ``left`` and ``right`` are
    separate objects on purpose: a shared dict edited by one consumer would
    silently retype the other arm's schema.

    The adapter validates the same exact shape again before execution.
    """
    eef = {
        "type": "object",
        "additionalProperties": False,
        "required": ["left", "right", "note", "phase"],
        "properties": {
            "left": _arm_schema(),
            "right": _arm_schema(),
            # Brevity is a prompt preference, not grounds for discarding an
            # otherwise executable pose decision.
            "note": {"type": "string", "minLength": 1},
            "phase": {"type": "string", "minLength": 1, "maxLength": 10},
        },
    }
    if not vla_review:
        return eef
    eef["required"] = ["mode", "left", "right", "note", "phase"]
    eef["properties"]["mode"] = {"type": "string", "enum": ["eef"]}
    return {
        "oneOf": [
            {
                "type": "object", "additionalProperties": False,
                "required": ["mode", "vla_steps", "verify_next", "note", "phase"],
                "properties": {
                    "mode": {"type": "string", "enum": ["vla"]},
                    "vla_steps": {"type": "integer", "minimum": 1},
                    "verify_next": {"type": "boolean"},
                    "note": {"type": "string", "minLength": 1},
                    "phase": {"type": "string", "minLength": 1, "maxLength": 10},
                },
            },
            eef,
        ]
    }


def validate_response(value: Any, *, vla_horizon: int | None = None) -> dict[str, Any]:
    """Check the model's object before it is believed, and hand it back unchanged.

    Only the structure is checked. Whether a target is reachable, whether it is
    into an action chunk, and what it means for the arms are the adapter's
    business -- the bridge must not start having opinions about the motion it is
    relaying.
    """
    if not isinstance(value, dict):
        raise PolicyValidationError("the reply must be a JSON object")
    if vla_horizon is not None and value.get("mode") == "vla":
        wanted = {"mode", "vla_steps", "verify_next", "note", "phase"}
        if set(value) != wanted:
            raise PolicyValidationError(f"the VLA reply must contain exactly {sorted(wanted)}")
        if type(value["vla_steps"]) is not int or not 1 <= value["vla_steps"] <= vla_horizon:
            raise PolicyValidationError("vla_steps must be within the proposed horizon")
        if type(value["verify_next"]) is not bool:
            raise PolicyValidationError("verify_next must be boolean")
        _validate_note_phase(value)
        return value
    wanted = {"left", "right", "note", "phase"}
    if vla_horizon is not None:
        wanted.add("mode")
        if value.get("mode") != "eef":
            raise PolicyValidationError("a VLA-review EEF reply must use mode='eef'")
    if set(value) != wanted:
        raise PolicyValidationError(f"the reply must contain exactly {sorted(wanted)}")
    _validate_note_phase(value)
    for arm in ARMS:
        _validate_arm_reply(value[arm], f"the reply's {arm} arm")
    return value


def _validate_note_phase(value: dict[str, Any]) -> None:
    if not isinstance(value["note"], str) or not value["note"].strip():
        raise PolicyValidationError("the reply's note must be a non-empty string")
    if not isinstance(value["phase"], str) or not value["phase"].strip():
        raise PolicyValidationError("the reply's phase must be a non-empty string")
    if len(value["phase"]) > 10:
        raise PolicyValidationError("the reply's phase must be at most 10 characters")


def _validate_arm_reply(value: Any, name: str) -> None:
    if not isinstance(value, dict) or set(value) != set(_ARM_FIELDS):
        raise PolicyValidationError(f"{name} must contain exactly {sorted(_ARM_FIELDS)}")
    for field, length in (("position", 3), ("orientation", 4)):
        item = value[field]
        if item == "keep":
            continue
        _vector(item, length, f"{name}.{field}")
    if value["gripper"] not in ("keep", "open", "close"):
        raise PolicyValidationError(f"{name}.gripper must be 'keep', 'open' or 'close'")
