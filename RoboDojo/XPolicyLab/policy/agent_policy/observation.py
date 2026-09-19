"""The observation packet: what this episode is, and what the robot reports now.

This module carries the structured facts for one policy decision. Standing
instructions live in the Codex workspace; the bridge renders only the changing
observation into the turn sent to Codex.

Two things live here, and they have different lifetimes.

* :class:`EpisodeContext` is fixed for the whole episode: task and budgets.
* :func:`build_request` is the per-decision half: the measured arm state, the
  feedback for the previous decision, the images, and where this turn sits in the
  budget.

Information boundary
--------------------
The text no longer passes through here, but the *data* does, and the rule is the
same one the workspace and ``bridge/schema.py`` are held to:

    **What the robot is** may be stated in full. **How the task is scored** must
    never leave the GPU side: reward terms, thresholds, spawn distributions and
    the benchmark's own identity are read out of the task and reward code by the
    adapter and must not be forwarded.

The stronger rule, and the one this module is written around: **do not state a
number we cannot stand behind.** In a request, every number is either an operator
budget or something measured on this very turn. There is no constant here that
could be wrong about the world, and no statistic from the demonstrations.

``orientation`` is the measured absolute quaternion in ``wxyz`` order.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from XPolicyLab.codex_agent.bridge.motion import ArmState

TASKS_FILE = Path(__file__).resolve().parent / "tasks.json"

DEFAULT_CAMERA_NAMES = ("cam_head", "cam_left_wrist", "cam_right_wrist")


def load_task_card(task_name: str) -> dict[str, Any]:
    with TASKS_FILE.open("r", encoding="utf-8") as handle:
        tasks = json.load(handle)
    if task_name not in tasks:
        raise KeyError(f"no task card for {task_name!r} in {TASKS_FILE}")
    return {"task_name": task_name, **tasks[task_name]}


@dataclass(frozen=True)
class EpisodeContext:
    """Facts that stay constant for one episode."""

    task_name: str
    task: dict[str, Any]
    step_budget: int
    max_decisions: int

def arm_observation(observed: ArmState) -> dict[str, Any]:
    """Current measured EEF pose in world xyz and absolute wxyz quaternion."""
    return {
        "position": [float(value) for value in np.asarray(observed.pos).reshape(3)],
        "orientation": [float(value) for value in np.asarray(observed.quat).reshape(4)],
        "gripper": float(observed.gripper),
    }


def build_request(
    context: EpisodeContext,
    *,
    episode_id: str,
    request_id: str,
    turn_index: int,
    calls_used: int,
    steps_used: int,
    left: ArmState,
    right: ArmState,
    feedback: Sequence[str],
    images: Sequence[dict[str, str]],
    continuation: dict[str, Any] | None = None,
    vla_review: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """One decision's observation packet, ready for ``POST /v1/decide``.

    Every counter in here is resolved from a single number rather than passed in
    separately, so nothing can drift out of agreement: ``step_id`` and
    ``remaining_steps`` are derived from ``steps_used``; remaining decisions are
    derived from ``calls_used``. The adapter increments ``calls_used`` immediately
    before every bridge request, including requests that fail, and increments
    ``steps_used`` by the returned chunk length after interpolation.

    Images are attached in the order supplied by the adapter.
    """
    request = {
        "episode_id": str(episode_id),
        "request_id": str(request_id),
        "step_id": int(steps_used),
        "turn_index": int(turn_index),
        "task": {
            "name": context.task_name,
            "instruction": str(context.task.get("instruction", "")),
            "guidance": [str(line) for line in context.task.get("guidance", [])],
        },
        "budget": {
            "max_decisions": int(context.max_decisions),
            "max_sim_steps": int(context.step_budget),
            "remaining_decisions": max(0, int(context.max_decisions) - int(calls_used)),
            "remaining_steps": max(0, int(context.step_budget) - int(steps_used)),
        },
        "observation": {
            "left": arm_observation(left),
            "right": arm_observation(right),
        },
        "feedback": [str(line) for line in feedback],
        "images": [
            {
                "name": str(image["name"]),
                "mime": str(image["mime"]),
                "b64": str(image["b64"]),
            }
            for image in images
        ],
    }
    if continuation is not None:
        request["continuation"] = continuation
    if vla_review is not None:
        request["vla_review"] = vla_review
    return request
