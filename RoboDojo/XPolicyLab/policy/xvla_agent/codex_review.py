"""Thin X-VLA client for the standalone Codex robot harness."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .bridge_client import BridgeClient, build_image_payload
except ImportError:  # model.py is also loaded directly by X_VLA's offline tests
    from bridge_client import BridgeClient, build_image_payload

CAMERAS = ("cam_head", "cam_left_wrist", "cam_right_wrist")
TASKS_FILE = Path(__file__).resolve().parent / "tasks.json"


def _image(observation: dict[str, Any], name: str) -> np.ndarray:
    value = observation["vision"][name]
    if isinstance(value, dict):
        value = value.get("color", value.get("rgb"))
    return np.asarray(value)


def _arm_major(actions: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"horizon": len(actions)}
    for side in ("left", "right"):
        poses = [np.asarray(action[f"{side}_ee_pose"], dtype=float).reshape(7) for action in actions]
        result[side] = {
            "position": [pose[:3].tolist() for pose in poses],
            "orientation": [pose[3:].tolist() for pose in poses],
            "gripper": [float(np.asarray(action[f"{side}_ee_joint_state"]).reshape(-1)[-1]) for action in actions],
        }
    return result


def _summary(chunk: dict[str, Any]) -> dict[str, Any]:
    result = {}
    for side in ("left", "right"):
        position = np.asarray(chunk[side]["position"], dtype=float)
        gripper = np.asarray(chunk[side]["gripper"], dtype=float)
        q0 = np.asarray(chunk[side]["orientation"][0], dtype=float)
        q1 = np.asarray(chunk[side]["orientation"][-1], dtype=float)
        q0 /= np.linalg.norm(q0)
        q1 /= np.linalg.norm(q1)
        result[side] = {
            "endpoint_translation_m": float(np.linalg.norm(position[-1] - position[0])),
            "endpoint_rotation_rad": float(2 * np.arccos(np.clip(abs(np.dot(q0, q1)), 0, 1))),
            "gripper_start": float(gripper[0]), "gripper_end": float(gripper[-1]),
            "gripper_min": float(gripper.min()), "gripper_max": float(gripper.max()),
        }
    return result


class CodexReviewer:
    def __init__(self, config: dict[str, Any]) -> None:
        self.task_name = str(config["task_name"])
        tasks = json.loads(TASKS_FILE.read_text(encoding="utf-8"))
        if self.task_name not in tasks:
            raise KeyError(f"no task card for {self.task_name!r} in {TASKS_FILE}")
        self.task = tasks[self.task_name]
        self.bridge = BridgeClient(str(os.environ.get("CODEX_BRIDGE_URL") or config.get("bridge_url") or "http://localhost:8765"), token=os.environ.get("CODEX_BRIDGE_TOKEN"))
        self.timeout_s = float(config.get("bridge_timeout_s", 105.0))
        self.jpeg_quality = int(config.get("jpeg_quality", 88))
        self.gripper_change_threshold = float(config.get("gripper_change_threshold", 0.1))
        if not 0 <= self.gripper_change_threshold <= 1:
            raise ValueError("gripper_change_threshold must be within [0,1]")
        self.control = {
            "delta_p_max_m": 0.005, "delta_theta_max_rad": 0.035,
            "max_target_translation_m": 0.05, "max_target_rotation_rad": 0.35,
            "settle_steps": 3, "gripper_open": 1.0, "gripper_close": 0.0,
            **dict(config.get("codex_control") or {}),
        }
        self.episode_number = 0
        self.calls = 0
        self.steps = 0
        self.continuation: dict[str, Any] | None = None

    def reset(self) -> None:
        self.episode_number += 1
        self.calls = self.steps = 0
        self.continuation = None

    def review(self, observation: dict[str, Any], actions: list[dict[str, Any]]) -> list[dict[str, np.ndarray]]:
        self.calls += 1
        episode_id = f"ep{self.episode_number:04d}"
        chunk = _arm_major(actions)
        state = observation["state"]
        request = {
            "episode_id": episode_id, "request_id": f"{episode_id}-{self.calls:06d}",
            "step_id": self.steps, "turn_index": self.calls - 1,
            "task": {"name": self.task_name, "instruction": self.task["instruction"], "guidance": self.task.get("guidance", [])},
            "budget": {"max_decisions": self.task["max_decisions"], "max_sim_steps": self.task["step_budget"],
                       "remaining_decisions": max(0, self.task["max_decisions"] - self.calls),
                       "remaining_steps": max(0, self.task["step_budget"] - self.steps)},
            "control": self.control,
            "observation": {
                side: {"position": list(map(float, state[f"{side}_ee_pose"][:3])),
                       "orientation": list(map(float, state[f"{side}_ee_pose"][3:])),
                       "gripper": float(state[f"{side}_ee_joint_state"][-1])}
                for side in ("left", "right")
            },
            "feedback": [],
            "images": [build_image_payload(name, _image(observation, name), quality=self.jpeg_quality) for name in CAMERAS],
            "vla_review": {
                "chunk": chunk,
                "summary": _summary(chunk),
                "gripper_change_threshold": self.gripper_change_threshold,
            },
        }
        if self.continuation is not None:
            request["continuation"] = self.continuation
        result = self.bridge.decide(request, timeout_s=self.timeout_s)
        if not result.ok or not result.action_chunk:
            raise RuntimeError(f"Codex Bridge failed: {result.error_kind}: {result.error}")
        self.continuation = result.continuation
        self.steps += len(result.action_chunk)
        return [{key: np.asarray(value, dtype=np.float32) for key, value in action.items()} for action in result.action_chunk]
