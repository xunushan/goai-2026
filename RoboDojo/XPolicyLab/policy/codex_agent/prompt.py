"""Prompt construction for the Codex high-level policy.

Information boundary
--------------------
This module is the single place where benchmark information could leak into the
model, so the rule is written down here and enforced by ``tests/test_prompt.py``:

    **What the robot is** (embodiment, kinematics, frame conventions, gripper
    semantics, the pose it starts from) may be stated in full.
    **How the task is scored** (reward terms, thresholds, object spawn
    distributions, success tolerances) must never appear.

The stronger rule, and the one this file is written around: **do not state a
number we cannot stand behind.** A plausible-looking bound that no observation
supports is worse than saying nothing, because the model believes it and steers
by it. Everything that survives here is either an operator budget (step and
decision counts), a limit our own controller actually enforces, or a fact the
robot itself reports -- and the start pose is read off the robot's first
observation rather than baked in as a constant.

Gone, after they turned out to cost more than they were worth: the workspace box
and per-decision travel figures came from q01/q99 of the demonstrations, but the
model read them as "this is where I am allowed to be" and avoided targets the arm
could physically reach; and a hard-coded start pose is just a constant that can
be wrong, when the robot reports the real one on every first frame. The limits
themselves still live in ``deploy.yml`` and ``motion.py`` -- they are simply not
advertised as the geometry of the world.

Nothing here is read out of ``task/RoboDojo/tasks/*.py`` or
``task/RoboDojo/config/*.yml``. The task card (``tasks/<task>.json``) is
deliberately prose-only: no thresholds, no distributions, no scoring formula.
"""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .motion import ArmState, relative_rpy

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
    """Everything the prompt may state, resolved once per episode.

    ``home_left``/``home_right`` are the poses the robot itself reported on this
    episode's first decision -- the frame that "orientation" is relative to, and
    the pose the task requires the arms to be back at when it ends. They are
    observed, not configured, so there is no constant here that can be wrong.
    """

    task: dict[str, Any]
    home_left: ArmState
    home_right: ArmState
    gripper_open: float
    step_budget: int
    max_decisions: int
    camera_names: tuple[str, ...] = ("cam_head", "cam_left_wrist", "cam_right_wrist")

    @property
    def home_quat_left(self) -> np.ndarray:
        return np.asarray(self.home_left.quat, dtype=np.float64)

    @property
    def home_quat_right(self) -> np.ndarray:
        return np.asarray(self.home_right.quat, dtype=np.float64)


def _vec(values: Sequence[float], digits: int = 4) -> str:
    # Snap values that round to zero so the rendered poses never contain "-0.000".
    limit = 0.5 * 10.0**-digits

    def fmt(value: float) -> str:
        number = float(value)
        return f"{0.0 if abs(number) < limit else number:.{digits}f}"

    return "[" + ", ".join(fmt(v) for v in values) + "]"


# What each view is, so the three attached images can be told apart. Without
# this the images arrive unlabelled and an arm has to work out from the pixels
# alone which wrist view is its own -- which it cannot, and the recorded run
# shows it guessing wrong ("the wrist views show both objects behind the
# current grasp centers").
_CAMERA_ROLES = {
    "cam_head": "a fixed camera above the table, looking down at the whole scene",
    "cam_left_wrist": "mounted on the LEFT arm, beside its jaws",
    "cam_right_wrist": "mounted on the RIGHT arm, beside its jaws",
}


def _camera_block(ctx: PromptContext) -> str:
    """Name every view in attachment order, and say what a wrist view shows."""
    lines = [
        f"  {index}. {name:<16} {_CAMERA_ROLES.get(name, 'an additional view')}"
        for index, name in enumerate(ctx.camera_names, start=1)
    ]
    lines.append(
        "A view labelled *_wrist is a close-up of that one gripper: its own jaws, the\n"
        "table immediately around them, and whatever sits close to that gripper -- and\n"
        "nothing else. Use cam_head as the primary view for scene understanding and\n"
        "approach; once close, use the active arm's wrist view for final alignment."
    )
    return "\n".join(lines)


def _camera_order_line(ctx: PromptContext) -> str:
    """The one-line reminder repeated each turn, where the images actually ride.

    Kept to a single line on purpose: the standing brief already explains what a
    wrist view is, and every repeated line is paid for out of a ten-call budget.
    """
    order = ", ".join(
        f"{index}. {name}" for index, name in enumerate(ctx.camera_names, start=1)
    )
    return f"ATTACHED VIEWS, in this order: {order}"


def build_system_prompt(ctx: PromptContext) -> str:
    """The standing brief, sent once when the Codex thread is created.

    Everything stated here is either a fact the robot reports, an operator
    budget, or a limit our own controller enforces. Nothing is a statistic
    inferred from the demonstrations -- see the module docstring for why those
    were removed.
    """
    task = ctx.task
    hints = task.get("hints") or []
    hint_block = (
        "\n".join(f"- {line}" for line in hints) + "\n" if hints else ""
    )

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
A deterministic controller sits below you and owns execution: it works out how many simulator steps your
motion takes, enforces per-step speed limits, and interpolates between where an arm is now and the target
you name. You do NOT choose speed, duration or interpolation length; do not reason about them.
If the controller cannot carry a target out exactly as you asked, the next turn's feedback says so, so
read it before repeating the same request. Short moves are cheap: ask for a small step, look, then ask
for the next one.

EMBODIMENT
- Two arms, 6 degrees of freedom each, with a parallel-jaw gripper on each arm.
- World frame, in metres. The robot stands at the near edge of the table looking along +y, so
  +y points forward onto the table, +z points up, and +x points to the robot's right. The arm named
  "left" works on the -x side of the table and the arm named "right" on the +x side. The table
  surface is at z = 0.76, so everything on it sits just above that height.
- The grippers point downwards, so lowering onto something means decreasing z.
- "position" is the end-effector reference point: the arm state below reports it and you command it.
  The jaws are below that point, so the reference point stays above an object even while the jaws are
  around it. Do not treat the reference point as the part of the arm that meets the table.
- "orientation" is yaw, pitch and roll in radians, RELATIVE to the pose the arms held at this episode's
  first decision. That start pose already points the gripper straight down, so [0, 0, 0] is the natural
  pose for picking something up off the table; tip it only when the task forces it. A tilt of more than
  a few tenths of a radian is a lot.
- The gripper is a single number. {ctx.gripper_open:.2f} is fully open; "close" drives the jaws together
  until they meet whatever is between them, so after a close the number you observe tells you how wide
  that object is. Open the jaws wider than the object before closing them on it.

CAMERA VIEWS (three per turn, attached in this order)
{_camera_block(ctx)}

START POSE
Both arms begin every episode in the same pose, and the success condition below ends with them back there.
That pose is what the arm state reports on the first decision of the episode: read it then, and note it,
because it is also the reference that "orientation" is measured from.

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
    """Render the observed pose of both arms, orientations relative to the start pose."""
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
{_camera_order_line(ctx)}
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
