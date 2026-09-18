"""RoboDojo adapter: observation -> Codex EEF target -> action chunk."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

_PACKAGE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _PACKAGE_DIR.parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from XPolicyLab.model_template import ModelTemplate

from .bridge_client import BridgeClient, build_image_payload
from .motion import ArmState, MotionConfig, hold_chunk, interpolate_chunk, target_error
from .observation import DEFAULT_CAMERA_NAMES, EpisodeContext, build_request, load_task_card
from .protocol import ParseError, parse_decision

TASK_CARD_KEYS = {
    "task_name", "instruction", "step_budget", "max_decisions",
}


@dataclass
class EpisodeState:
    episode_id: str
    calls_used: int = 0
    steps_used: int = 0
    feedback: list[str] = field(default_factory=list)


def _section(config: dict[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name)
    return dict(value) if isinstance(value, dict) else {}


def _image(observation: dict[str, Any], name: str) -> np.ndarray:
    entry = observation["vision"][name]
    if isinstance(entry, dict):
        entry = entry["color"] if "color" in entry else entry["rgb"]
    array = np.asarray(entry)
    if array.ndim != 3:
        raise ValueError(f"{name} must be a 3-D image, got {array.shape}")
    if array.shape[-1] not in (1, 3, 4) and array.shape[0] in (1, 3, 4):
        array = np.transpose(array, (1, 2, 0))
    if np.issubdtype(array.dtype, np.floating):
        array = (np.clip(array, 0.0, 1.0) * 255).astype(np.uint8)
    elif array.dtype != np.uint8:
        array = array.astype(np.uint8)
    return array


def _arm(observation: dict[str, Any], side: str) -> ArmState:
    state = observation["state"]
    pose = np.asarray(state[f"{side}_ee_pose"], dtype=np.float64).reshape(-1)
    gripper = np.asarray(state[f"{side}_ee_joint_state"], dtype=np.float64).reshape(-1)
    if pose.shape != (7,) or not gripper.size:
        raise ValueError(f"invalid {side} arm state")
    return ArmState.from_pose7(pose, gripper[-1])


class Model(ModelTemplate):
    def __init__(self, model_cfg: dict[str, Any]):
        config = dict(model_cfg)
        if config.get("action_type", "ee") != "ee":
            raise ValueError("codex_agent supports only action_type='ee'")

        bridge = _section(config, "bridge")
        motion = _section(config, "motion")
        images = _section(config, "images")
        self.motion = MotionConfig(
            delta_p_max_m=float(motion.get("delta_p_max_m", 0.005)),
            delta_theta_max_rad=float(motion.get("delta_theta_max_rad", 0.035)),
            max_target_translation_m=float(motion.get("max_target_translation_m", 0.05)),
            max_target_rotation_rad=float(motion.get("max_target_rotation_rad", 0.35)),
            settle_steps=int(motion.get("settle_steps", 3)),
            gripper_open=float(motion.get("gripper_open", 1.0)),
            gripper_close=float(motion.get("gripper_close", 0.0)),
        )
        self.jpeg_quality = int(images.get("jpeg_quality", 88))
        self.request_timeout_s = float(bridge.get("request_timeout_s", 105.0))
        self.bridge = BridgeClient(
            str(os.environ.get("CODEX_BRIDGE_URL") or config.get("bridge_url")
                or bridge.get("base_url") or "http://localhost:8765"),
            token=os.environ.get("CODEX_BRIDGE_TOKEN") or bridge.get("token"),
        )

        task_name = config.get("task_name")
        if not task_name:
            raise ValueError("task_name is required")
        self.task = load_task_card(str(task_name))
        unknown = set(self.task) - TASK_CARD_KEYS
        if unknown:
            raise ValueError(f"unexpected task-card fields: {sorted(unknown)}")
        self.step_budget = int(self.task["step_budget"])
        self.call_budget = int(self.task["max_decisions"])
        if self.step_budget < 1 or self.call_budget < 1:
            raise ValueError("task budgets must be positive")
        self.context = EpisodeContext(
            task_name=str(task_name),
            task=self.task,
            step_budget=self.step_budget,
            max_decisions=self.call_budget,
        )

        self.observation: dict[str, Any] | None = None
        self.episode: EpisodeState | None = None
        self.episode_number = 0
        self.last_arms: tuple[ArmState, ArmState] | None = None

    def update_obs(self, obs):
        self.observation = obs
        try:
            self.last_arms = (_arm(obs, "left"), _arm(obs, "right"))
        except (KeyError, TypeError, ValueError, IndexError):
            pass

    def get_action(self, **kwargs):
        try:
            return self._decide()
        except Exception as exc:
            if self.last_arms is None:
                raise
            print(f"[codex_agent] decision failed; holding position: {exc}", flush=True)
            if self.episode is not None:
                # hold_chunk contains one executable action frame. It commands
                # the measured pose/gripper unchanged, but the simulator still
                # advances by one step when it consumes that frame.
                self.episode.steps_used += 1
            return hold_chunk(*self.last_arms)

    def reset(self):
        self.episode_number += 1
        self.episode = None
        self.observation = None
        self.last_arms = None

    def _decide(self) -> list[dict[str, np.ndarray]]:
        if self.observation is None:
            raise RuntimeError("update_obs must be called before get_action")
        if self.episode is None:
            self.episode = EpisodeState(f"ep{self.episode_number:04d}")
        state = self.episode

        left = _arm(self.observation, "left")
        right = _arm(self.observation, "right")
        self.last_arms = (left, right)
        if state.calls_used >= self.call_budget or state.steps_used >= self.step_budget:
            state.steps_used += 1
            return hold_chunk(left, right)

        images = [
            build_image_payload(name, _image(self.observation, name), quality=self.jpeg_quality)
            for name in DEFAULT_CAMERA_NAMES
        ]
        state.calls_used += 1
        result = self.bridge.decide(
            build_request(
                self.context,
                episode_id=state.episode_id,
                request_id=f"{state.episode_id}-{state.calls_used:06d}",
                turn_index=state.calls_used - 1,
                calls_used=state.calls_used,
                steps_used=state.steps_used,
                left=left,
                right=right,
                feedback=state.feedback,
                images=images,
            ),
            timeout_s=self.request_timeout_s,
        )
        if not result.ok:
            state.feedback = [f"Policy call failed ({result.error_kind}); the robot held position."]
            # The one-frame hold is motionless, not free: executing it advances
            # the simulator and therefore consumes one simulator-step budget.
            state.steps_used += 1
            return hold_chunk(left, right)

        try:
            decision = parse_decision(
                result.decision,
                gripper_open=self.motion.gripper_open,
                gripper_close=self.motion.gripper_close,
            )
        except ParseError as exc:
            state.feedback = [f"The previous reply was invalid ({exc}); the robot held position."]
            state.steps_used += 1
            return hold_chunk(left, right)

        for side, current, command in (
            ("left", left, decision.left),
            ("right", right, decision.right),
        ):
            translation, rotation = target_error(current, command)
            if (translation > self.motion.max_target_translation_m
                    or rotation > self.motion.max_target_rotation_rad):
                state.feedback = [
                    f"The {side} target exceeded the per-decision limit "
                    f"({translation:.3f} m, {rotation:.3f} rad); the robot held position."
                ]
                state.steps_used += 1
                return hold_chunk(left, right)

        chunk = interpolate_chunk(left, decision.left, right, decision.right, self.motion)
        state.steps_used += len(chunk)
        state.feedback = [
            f"Executed the requested target for {len(chunk)} simulator steps. "
            "Use the fresh measured pose and images to judge the result."
        ]
        return chunk
