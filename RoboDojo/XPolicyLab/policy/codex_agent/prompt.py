"""Prompt construction for the Codex high-level policy.

Information boundary
--------------------
This module is the single place where benchmark information could leak into the
model, so the rule is written down here and enforced by ``tests/test_prompt.py``:

    **What the robot is** (embodiment, kinematics, frame conventions, reach,
    gripper semantics, workspace bounds, start pose) may be stated in full.
    **How the task is scored** (reward terms, thresholds, object spawn
    distributions, success tolerances) must never appear.

Everything numeric here comes from the official dataset (`data/sim_lerobot_v30_ee`
first frame for the start pose, `meta/stats.json` q01/q99 for the workspace) or
from our own controller configuration. Nothing is read back out of
``task/RoboDojo/tasks/*.py`` or ``task/RoboDojo/config/*.yml``.

The task card (``tasks/<task>.json``) is deliberately prose-only: no thresholds,
no distributions, no scoring formula.
"""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .motion import ArmState, Box, relative_rpy

TASKS_DIR = Path(__file__).resolve().parent / "tasks"

_ARM_KEYS = ("position", "orientation", "gripper")


def load_task_card(task_name: str) -> dict[str, Any]:
    path = TASKS_DIR / f"{task_name}.json"
    if not path.is_file():
        raise FileNotFoundError(f"no task card for {task_name!r} at {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


@dataclass(frozen=True)
class PromptContext:
    task: dict[str, Any]
    home_left: np.ndarray  # (7,) xyz + quat wxyz
    home_right: np.ndarray
    workspace_left: Box
    workspace_right: Box
    gripper_open: float
    gripper_close: float
    step_budget: int
    max_decisions: int
    max_target_distance_m: float
    delta_p_max_m: float = 0.015
    camera_names: tuple[str, ...] = ("cam_head", "cam_left_wrist", "cam_right_wrist")

    @property
    def home_quat_left(self) -> np.ndarray:
        return np.asarray(self.home_left[3:7], dtype=np.float64)

    @property
    def home_quat_right(self) -> np.ndarray:
        return np.asarray(self.home_right[3:7], dtype=np.float64)


def _vec(values: Sequence[float], digits: int = 4) -> str:
    # Snap values that round to zero so the rendered poses never contain "-0.000".
    limit = 0.5 * 10.0**-digits

    def fmt(value: float) -> str:
        number = float(value)
        return f"{0.0 if abs(number) < limit else number:.{digits}f}"

    return "[" + ", ".join(fmt(v) for v in values) + "]"


def _box(box: Box) -> str:
    return (
        f"x [{box.x[0]:.2f}, {box.x[1]:.2f}]  "
        f"y [{box.y[0]:.2f}, {box.y[1]:.2f}]  "
        f"z [{box.z[0]:.2f}, {box.z[1]:.2f}]"
    )


def _home_block(ctx: PromptContext) -> str:
    lines = []
    for side, home in (("left", ctx.home_left), ("right", ctx.home_right)):
        lines.append(
            f"  {side:<5} position {_vec(home[:3])}   "
            f"orientation yaw/pitch/roll [0.000, 0.000, 0.000]   "
            f"gripper {ctx.gripper_open:.2f}"
        )
    return "\n".join(lines)


def _envelope_block(ctx: PromptContext) -> str:
    lines = []
    for side, box in (("left", ctx.workspace_left), ("right", ctx.workspace_right)):
        lines.append(f"  {side:<5} position {_box(box)}")
    return "\n".join(lines)


def build_system_prompt(ctx: PromptContext) -> str:
    """The standing brief, sent once when the Codex thread is created."""
    task = ctx.task
    hints = task.get("hints") or []
    hint_block = (
        "\n".join(f"- {line}" for line in hints) + "\n" if hints else ""
    )
    travel_cm = int(round(ctx.max_target_distance_m * 100))
    per_step_cm = max(0.1, ctx.delta_p_max_m * 100)
    # What the longest legal single decision costs in simulator steps. The model
    # is told this so it can trade distance against budget; without it, "you have
    # 400 steps" is meaningless when the only other number it has is a distance.
    step_cost = int(math.ceil(ctx.max_target_distance_m / max(1e-6, ctx.delta_p_max_m)))

    return f"""You are the high-level manipulation policy for a dual-arm robot operating in a physics simulator.
You are not a low-level controller. You never emit joint angles, trajectories or torques.

Each turn you receive three RGB camera views ({", ".join(ctx.camera_names)}), the observed pose of both arms,
and the outcome of your previous decision. You then choose exactly ONE next motion and reply with a single
JSON object.

DISCIPLINE
- Look before you move, and move in small deliberate steps. Re-observe after every motion.
- Never try to finish the task in one turn.
- Use "keep" for any component that does not need to change; keep is free.
- Use "note" to record, in one or two sentences, what you see and why you chose this motion. Humans read it.
- Use "phase" for a short label of what you are doing (e.g. "approach", "grasp", "insert", "retreat").

WHO EXECUTES YOUR DECISION
A deterministic controller sits below you and owns everything about execution:
- it decides how many simulator steps your motion takes and enforces per-step speed limits;
- it clamps small excursions past the valid workspace and rejects anything further away;
- it owns collision avoidance, interpolation and settling.
You do NOT choose speed, duration or interpolation length. Do not reason about them.
One decision moves an arm at most about {travel_cm} cm; anything further is rejected, and a rejected
decision is wasted. The controller advances a grasp point at roughly {per_step_cm:.1f} cm per simulator
step, so that longest move costs about {step_cost} of your {ctx.step_budget} simulator steps. You will
rarely want the full {travel_cm} cm -- a typical decision reaches a few centimetres to a hand's width.
Short moves are cheap, so use them for the final alignment.

EMBODIMENT
- Two arms, 6 degrees of freedom each, with a parallel-jaw gripper on each arm.
- World frame, in metres. The robot stands at the near edge of the table looking along +y, so
  +y points forward onto the table, +z points up, and +x points to the robot's right. The arm
  named "left" is mounted at x = -0.30 and the arm named "right" at x = +0.30. The table surface
  is at z = 0.76, so everything on it sits just above that height.
- The grippers point downwards, so lowering onto something means decreasing z.
- "position" is the end-effector reference point: the arm state below reports it and you command it.
  The jaws hang below that point, so the reference point stays well above an object even while the
  jaws are around it. The envelope below already accounts for that -- do not try to reach the table
  surface itself.
- "orientation" is yaw, pitch and roll in radians, RELATIVE to the start orientation reported below
  (which is therefore [0, 0, 0]). That start orientation already points the gripper straight down,
  so [0, 0, 0] is the natural pose for picking something up off the table; tip it only when the
  task forces it. A tilt of more than a few tenths of a radian is a lot.
- gripper {ctx.gripper_open:.2f} is fully open; closing the jaws brings them to about {ctx.gripper_close:.2f},
  which is "closed" -- not zero. Open the jaws wider than the object before closing them on it.

START POSE (both arms are here at the beginning of every episode)
{_home_block(ctx)}

REACHABLE ENVELOPE (targets outside it are clamped, or rejected if they are far outside)
{_envelope_block(ctx)}

TASK
Instruction: {task.get("instruction", "")}
Scene: {task.get("scene", "")}
{hint_block}
The episode ends when the success condition below holds, or when the step budget runs out, whichever
comes first. Ending the episode in any other configuration is a failure, so allocating your last decision
to putting the arms back is part of the task, not an afterthought.

SUCCESS CONDITION
{task.get("success_rule", "")}

BUDGET
The episode lasts at most {ctx.step_budget} simulator steps and you get at most {ctx.max_decisions} decisions
in total. Every decision counts against that budget. Plan coarse to fine: travel and reposition early,
and reserve your final decisions for precise alignment and for returning to the start pose.
Using fewer decisions is better.

REPLY FORMAT
Reply with ONLY this JSON object, with no prose before or after it:
{{"left":  {{"position": "keep" | [x, y, z],
           "orientation": "keep" | [yaw, pitch, roll],
           "gripper": "keep" | "open" | "close"}},
  "right": {{"position": "keep" | [x, y, z],
           "orientation": "keep" | [yaw, pitch, roll],
           "gripper": "keep" | "open" | "close"}},
  "note": "one or two sentences describing what you see and why you chose this motion",
  "phase": "short label"}}
If you need an exact absolute orientation you may give "orientation" as {{"quat": [w, x, y, z]}} instead.
"""


def format_observation(ctx: PromptContext, left: ArmState, right: ArmState) -> str:
    """Render the observed pose of both arms with orientations relative to HOME."""
    lines = []
    for side, state in (("left", left), ("right", right)):
        home = ctx.home_quat_left if side == "left" else ctx.home_quat_right
        rpy = relative_rpy(state.quat, home)
        lines.append(
            f"  {side:<5} position {_vec(state.pos)}   "
            f"orientation {_vec(rpy, 3)}   gripper {state.gripper:.2f}"
        )
    return "\n".join(lines)


@dataclass
class TurnFeedback:
    """What actually happened after the previous decision, in prose."""

    lines: list[str]

    def render(self) -> str:
        if not self.lines:
            return "  (this is the first decision of the episode; nothing has been executed yet)"
        return "\n".join(f"  {line}" for line in self.lines)


def build_turn_prompt(
    ctx: PromptContext,
    *,
    turn_index: int,
    decisions_used: int,
    steps_used: int,
    left: ArmState,
    right: ArmState,
    feedback: TurnFeedback,
) -> str:
    """The per-decision message appended to the standing brief."""
    remaining_decisions = max(0, ctx.max_decisions - decisions_used - 1)
    remaining_steps = max(0, ctx.step_budget - steps_used)
    return f"""TURN {turn_index + 1} of at most {ctx.max_decisions}
remaining decisions after this one: {remaining_decisions}
remaining simulator steps: {remaining_steps}

OBSERVED ARM STATE (orientation is yaw/pitch/roll relative to the start pose)
{format_observation(ctx, left, right)}

OUTCOME OF YOUR PREVIOUS DECISION
{feedback.render()}

Choose the single next motion. Reply with the JSON object only.
"""


def _arm_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": list(_ARM_KEYS),
        "properties": {
            "position": {
                "anyOf": [
                    {"type": "string", "enum": ["keep"]},
                    {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 3,
                        "maxItems": 3,
                    },
                ]
            },
            "orientation": {
                "anyOf": [
                    {"type": "string", "enum": ["keep"]},
                    {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 3,
                        "maxItems": 4,
                    },
                ]
            },
            "gripper": {"type": "string", "enum": ["keep", "open", "close"]},
        },
    }


def build_output_schema() -> dict[str, Any]:
    """JSON Schema handed to ``codex exec --output-schema``.

    Inlined rather than using ``$defs``/``$ref`` so the schema stays within the
    subset that structured-output backends reliably accept. The adapter still
    parses tolerantly, because a schema is a hint and not a guarantee.
    """
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["left", "right", "note", "phase"],
        "properties": {
            "left": _arm_schema(),
            "right": copy.deepcopy(_arm_schema()),
            "note": {"type": "string"},
            "phase": {"type": "string"},
        },
    }
