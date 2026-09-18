"""Load one task-matched successful demonstration for a new Codex thread."""

from __future__ import annotations

import base64
import json
import math
import mimetypes
from pathlib import Path
from typing import Any


GRIPPER_STAGES = frozenset({"grasp", "place"})
CAMERA_KEYS = (
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)


class ExperienceError(ValueError):
    """The host-maintained experience library is malformed."""


class ExperienceLibrary:
    """An exact task-name index outside the agent-readable workspace."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root).expanduser().resolve()
        index_path = self.root / "index.json"
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ExperienceError(f"cannot load experience index {index_path}: {error}") from error
        if not isinstance(index, dict) or not all(
            isinstance(name, str) and name and isinstance(path, str) and path
            for name, path in index.items()
        ):
            raise ExperienceError("experience index must map non-empty task names to demo paths")
        self.index: dict[str, str] = index
        self._cache: dict[str, tuple[dict[str, Any], ...]] = {}

    def items(self, task_name: str) -> list[dict[str, Any]]:
        """Return App Server input items, or no items for an unmapped task."""
        relative = self.index.get(task_name)
        if relative is None:
            return []
        cached = self._cache.get(task_name)
        if cached is None:
            demo_path = (self.root / relative).resolve()
            if self.root not in demo_path.parents:
                raise ExperienceError(f"experience path escapes the library: {relative}")
            cached = tuple(_compile_demo(demo_path, task_name))
            self._cache[task_name] = cached
        return [dict(item) for item in cached]


def render_demo_text(demo: dict[str, Any], task_name: str) -> list[str]:
    """Render the concise text blocks paired with a demonstration's images."""
    frames = _validated_frames(demo, task_name)
    blocks = [
        "\n".join([
            "HISTORICAL SUCCESSFUL DEMONSTRATION",
            f"Task name: {task_name}",
            f"Goal: {demo['task']}",
            "Reference only: reuse stage order, arm roles, grasp orientation and gripper timing; "
            "adapt positions to the current images and measured state.",
            f"State layout: {demo.get('state_layout', 'position xyz, quaternion wxyz, gripper')}",
            "State is the measured keyframe state. Decision is the recorded command associated "
            "with that keyframe; it is historical evidence, not a pending command.",
        ])
    ]
    for index, frame in enumerate(frames, 1):
        roles = frame["roles"]
        state = frame["state"]
        decision = frame["decision"]
        state_text = " | ".join(
            f"{side[0].upper()} {_arm_state(state[side])}"
            if roles[side] != "idle" or index in (1, len(frames))
            else f"{side[0].upper()} idle"
            for side in ("left", "right")
        )
        decision_text = " | ".join(
            f"{side[0].upper()} keep" if roles[side] == "idle" else f"{side[0].upper()} {_arm_decision(decision[side])}"
            for side in ("left", "right")
        )
        blocks.append("\n".join([
            f"[EXAMPLE {index}/{len(frames)} | {frame['stage']} | frame {frame['frame_index']}]",
            f"Observed: {frame['observation']}",
            f"Roles: left={roles['left']}, right={roles['right']}",
            f"State: {state_text}",
            f"Decision: {decision_text}",
            f"Outcome: {frame['result']}",
        ]))
    blocks.append("END HISTORICAL DEMONSTRATION. Decide from the current observation below.")
    return blocks


def _compile_demo(demo_path: Path, task_name: str) -> list[dict[str, Any]]:
    try:
        demo = json.loads(demo_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ExperienceError(f"cannot load demonstration {demo_path}: {error}") from error
    frames = _validated_frames(demo, task_name)
    text_blocks = render_demo_text(demo, task_name)
    items: list[dict[str, Any]] = [{"type": "text", "text": text_blocks[0]}]
    for index, (frame, text_block) in enumerate(zip(frames, text_blocks[1:-1]), 1):
        items.append({"type": "text", "text": text_block})
        images = frame["images"]
        selected_keys = CAMERA_KEYS if frame["stage"] in GRIPPER_STAGES else CAMERA_KEYS[:1]
        selected = [(camera, images[camera]) for camera in selected_keys]
        for camera, relative in selected:
            image_path = (demo_path.parent / relative).resolve()
            if demo_path.parent not in image_path.parents or not image_path.is_file():
                raise ExperienceError(f"missing or unsafe experience image: {relative}")
            mime, _ = mimetypes.guess_type(image_path.name)
            if mime not in ("image/jpeg", "image/png"):
                raise ExperienceError(f"unsupported experience image type: {image_path}")
            try:
                encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
            except OSError as error:
                raise ExperienceError(f"cannot read experience image {image_path}: {error}") from error
            label = camera.rsplit(".", 1)[-1]
            items.append({"type": "text", "text": f"Historical image {index}: {label}"})
            items.append({"type": "image", "url": f"data:{mime};base64,{encoded}"})
    items.append({"type": "text", "text": text_blocks[-1]})
    return items


def _validated_frames(demo: Any, task_name: str) -> list[dict[str, Any]]:
    if not isinstance(demo, dict) or demo.get("task_slug") != task_name:
        raise ExperienceError(f"demonstration task_slug must equal {task_name!r}")
    if not isinstance(demo.get("task"), str) or not demo["task"].strip():
        raise ExperienceError("demonstration task must be non-empty text")
    frames = demo.get("keyframes")
    if not isinstance(frames, list) or not frames:
        raise ExperienceError("demonstration requires keyframes")
    required = {"frame_index", "stage", "observation", "roles", "result", "state", "decision", "images"}
    for index, frame in enumerate(frames):
        if not isinstance(frame, dict) or not required <= set(frame):
            raise ExperienceError(f"keyframe {index} is missing required fields")
        if type(frame["frame_index"]) is not int or frame["frame_index"] < 0:
            raise ExperienceError(f"keyframe {index} frame_index must be a non-negative integer")
        for field in ("stage", "observation", "result"):
            if not isinstance(frame[field], str) or not frame[field].strip():
                raise ExperienceError(f"keyframe {index} {field} must be non-empty text")
        roles = frame["roles"]
        state = frame["state"]
        decision = frame["decision"]
        images = frame["images"]
        if not isinstance(roles, dict) or set(roles) != {"left", "right"}:
            raise ExperienceError(f"keyframe {index} must define both arm roles")
        if not all(isinstance(role, str) and role for role in roles.values()):
            raise ExperienceError(f"keyframe {index} arm roles must be non-empty text")
        if not isinstance(state, dict) or set(state) != {"left", "right"}:
            raise ExperienceError(f"keyframe {index} must describe both arms")
        if not isinstance(decision, dict) or set(decision) != {"left", "right"}:
            raise ExperienceError(f"keyframe {index} must define both arm decisions")
        for side in ("left", "right"):
            _validate_arm(state[side], f"keyframe {index} state.{side}", allow_keep=False)
            _validate_arm(decision[side], f"keyframe {index} decision.{side}", allow_keep=True)
        if not isinstance(images, dict) or tuple(images) != CAMERA_KEYS:
            raise ExperienceError(
                f"keyframe {index} images must contain the three cameras in canonical order"
            )
        if not all(isinstance(path, str) and path.strip() for path in images.values()):
            raise ExperienceError(f"keyframe {index} image paths must be non-empty text")
    return frames


def _validate_arm(value: Any, name: str, *, allow_keep: bool) -> None:
    if not isinstance(value, dict) or set(value) != {"position", "orientation", "gripper"}:
        raise ExperienceError(f"{name} must contain position, orientation and gripper")
    for field, length in (("position", 3), ("orientation", 4)):
        vector = value[field]
        if allow_keep and vector == "keep":
            continue
        if not isinstance(vector, list) or len(vector) != length or not all(
            type(item) in (int, float) and math.isfinite(float(item)) for item in vector
        ):
            raise ExperienceError(f"{name}.{field} must be keep or a finite {length}-vector")
    gripper = value["gripper"]
    if not isinstance(gripper, (str, int, float)) or isinstance(gripper, bool):
        raise ExperienceError(f"{name}.gripper must be text or a number")


def _numbers(values: Any) -> str:
    return "[" + ",".join(f"{float(value):.4f}" for value in values) + "]"


def _arm_state(arm: dict[str, Any]) -> str:
    return f"xyz={_numbers(arm['position'])} q={_numbers(arm['orientation'])} grip={arm['gripper']}"


def _arm_decision(arm: dict[str, Any]) -> str:
    position = arm["position"] if arm["position"] == "keep" else _numbers(arm["position"])
    orientation = arm["orientation"] if arm["orientation"] == "keep" else _numbers(arm["orientation"])
    return f"xyz={position} q={orientation} grip={arm['gripper']}"
