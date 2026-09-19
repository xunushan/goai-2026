"""RoboDojo adapter: observation -> Codex EEF target -> action chunk."""

from __future__ import annotations

import json
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
from XPolicyLab.codex_agent.bridge.motion import ArmState, hold_chunk
from XPolicyLab.utils.episode_index import EpisodeIndexResolver
from XPolicyLab.utils.task_name_resolver import TaskNameResolver, load_real_task_name_map
from .observation import DEFAULT_CAMERA_NAMES, EpisodeContext, build_request, load_task_card

TASK_CARD_KEYS = {
    "task_name", "instruction", "guidance", "step_budget", "max_decisions",
}


@dataclass
class EpisodeState:
    episode_id: str
    requests_seen: int = 0
    calls_used: int = 0
    steps_used: int = 0
    feedback: list[str] = field(default_factory=list)
    continuation: dict[str, Any] | None = None


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


def _action_chunk(value: Any) -> list[dict[str, np.ndarray]]:
    required = {"left_ee_pose", "left_ee_joint_state", "right_ee_pose", "right_ee_joint_state"}
    if not isinstance(value, list) or not value:
        raise ValueError("bridge action_chunk must be non-empty")
    result = []
    for index, action in enumerate(value):
        if not isinstance(action, dict) or set(action) != required:
            raise ValueError(f"bridge action_chunk[{index}] has invalid fields")
        converted = {key: np.asarray(item, dtype=np.float32).reshape(-1) for key, item in action.items()}
        if converted["left_ee_pose"].shape != (7,) or converted["right_ee_pose"].shape != (7,):
            raise ValueError(f"bridge action_chunk[{index}] has invalid EEF pose")
        if converted["left_ee_joint_state"].shape != (1,) or converted["right_ee_joint_state"].shape != (1,):
            raise ValueError(f"bridge action_chunk[{index}] has invalid gripper state")
        if not all(np.isfinite(item).all() for item in converted.values()):
            raise ValueError(f"bridge action_chunk[{index}] contains non-finite values")
        result.append(converted)
    return result


def _finite_list(value: Any) -> list[float | None]:
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    return [float(item) if np.isfinite(item) else None for item in array]


def _actions_16d(actions: list[dict[str, np.ndarray]]) -> list[list[float]]:
    return [
        np.concatenate(
            [
                action["left_ee_pose"],
                action["left_ee_joint_state"],
                action["right_ee_pose"],
                action["right_ee_joint_state"],
            ]
        ).astype(np.float32).tolist()
        for action in actions
    ]


class Model(ModelTemplate):
    def __init__(self, model_cfg: dict[str, Any]):
        config = dict(model_cfg)
        if config.get("action_type", "ee") != "ee":
            raise ValueError("agent_policy supports only action_type='ee'")

        bridge = _section(config, "bridge")
        images = _section(config, "images")
        experience = _section(config, "experience")
        self.use_experience = bool(experience.get("enabled", True))
        self.jpeg_quality = int(images.get("jpeg_quality", 88))
        self.request_timeout_s = float(bridge.get("request_timeout_s", 105.0))
        self.bridge = BridgeClient(
            str(os.environ.get("CODEX_BRIDGE_URL") or config.get("bridge_url")
                or bridge.get("base_url") or "http://localhost:8765"),
            token=os.environ.get("CODEX_BRIDGE_TOKEN") or bridge.get("token"),
        )

        self.task_resolver = TaskNameResolver(
            config.get("task_name"),
            load_real_task_name_map(config.get("task_instruction_json")),
        )
        self.task: dict[str, Any] | None = None
        self.step_budget = 0
        self.call_budget = 0
        self.context: EpisodeContext | None = None
        if config.get("task_name"):
            self._bind_task(str(config["task_name"]))

        self.observation: dict[str, Any] | None = None
        self.episode: EpisodeState | None = None
        self.episode_index = EpisodeIndexResolver()
        self.log_io = bool(config.get("log_io", True))
        self.last_arms: tuple[ArmState, ArmState] | None = None

    def _bind_task(self, task_name: str) -> None:
        self.task = load_task_card(task_name)
        unknown = set(self.task) - TASK_CARD_KEYS
        if unknown:
            raise ValueError(f"unexpected task-card fields: {sorted(unknown)}")
        self.step_budget = int(self.task["step_budget"])
        self.call_budget = int(self.task["max_decisions"])
        if self.step_budget < 1 or self.call_budget < 1:
            raise ValueError("task budgets must be positive")
        self.context = EpisodeContext(
            task_name=task_name,
            task=self.task,
            step_budget=self.step_budget,
            max_decisions=self.call_budget,
        )

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
            print(f"[agent_policy] decision failed; holding position: {exc}", flush=True)
            if self.episode is not None:
                # hold_chunk contains one executable action frame. It commands
                # the measured pose/gripper unchanged, but the simulator still
                # advances by one step when it consumes that frame.
                self.episode.steps_used += 1
            chunk = hold_chunk(*self.last_arms)
            self._log_actions(chunk, source="error_hold", error=str(exc))
            return chunk

    def reset(self):
        self.episode = None
        self.observation = None
        self.last_arms = None
        self.task_resolver.reset()
        self.episode_index.reset()

    def _log_observation(
        self, state: EpisodeState, task_name: str, request_index: int
    ) -> None:
        if not self.log_io or self.observation is None:
            return
        observed_state = self.observation.get("state", {})
        state_summary = {
            key: _finite_list(observed_state[key])
            for key in (
                "left_ee_pose",
                "left_ee_joint_state",
                "right_ee_pose",
                "right_ee_joint_state",
            )
            if key in observed_state
        }
        images = {}
        for name in DEFAULT_CAMERA_NAMES:
            try:
                image = _image(self.observation, name)
            except (KeyError, TypeError, ValueError):
                continue
            images[name] = {
                "shape": list(image.shape),
                "dtype": str(image.dtype),
                "min": float(np.min(image)),
                "max": float(np.max(image)),
                "mean": float(np.mean(image)),
            }
        summary = {
            "event": "client_observation",
            "request": request_index,
            "env_idx": int(self.observation.get("env_idx") or 0),
            "episode_idx": state.episode_id,
            "task_name": task_name,
            "instruction": str(self.observation.get("instruction", ""))[:200],
            "state": state_summary,
            "images": images,
        }
        print("[agent_policy][io] " + json.dumps(summary, ensure_ascii=False), flush=True)

    def _log_actions(
        self,
        chunk: list[dict[str, np.ndarray]],
        *,
        source: str,
        error: str | None = None,
    ) -> None:
        if not self.log_io:
            return
        summary = {
            "event": "server_actions",
            "request": max(0, self.episode.requests_seen - 1) if self.episode else 0,
            "env_idx": int((self.observation or {}).get("env_idx") or 0),
            "episode_idx": self.episode.episode_id if self.episode else None,
            "task_name": self.context.task_name if self.context else None,
            "source": source,
            "execute_steps": len(chunk),
            "error": error,
            "actions_16d": _actions_16d(chunk),
        }
        print("[agent_policy][io] " + json.dumps(summary, ensure_ascii=False), flush=True)

    def _decide(self) -> list[dict[str, np.ndarray]]:
        if self.observation is None:
            raise RuntimeError("update_obs must be called before get_action")
        task_name = self.task_resolver.resolve(self.observation)
        if self.context is None or self.context.task_name != task_name:
            self._bind_task(task_name)
        assert self.context is not None
        if self.episode is None:
            self.episode = EpisodeState(
                self.episode_index.resolve(self.observation.get("episode_idx"))
            )
        state = self.episode
        request_index = state.requests_seen
        state.requests_seen += 1

        left = _arm(self.observation, "left")
        right = _arm(self.observation, "right")
        self.last_arms = (left, right)
        self._log_observation(state, task_name, request_index)
        if state.calls_used >= self.call_budget or state.steps_used >= self.step_budget:
            state.steps_used += 1
            chunk = hold_chunk(left, right)
            self._log_actions(chunk, source="budget_hold")
            return chunk

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
                use_experience=self.use_experience,
                continuation=state.continuation,
            ),
            timeout_s=self.request_timeout_s,
        )
        if not result.ok:
            state.feedback = [f"Policy call failed ({result.error_kind}); the robot held position."]
            # The one-frame hold is motionless, not free: executing it advances
            # the simulator and therefore consumes one simulator-step budget.
            state.steps_used += 1
            chunk = hold_chunk(left, right)
            self._log_actions(
                chunk,
                source="bridge_hold",
                error=f"{result.error_kind}: {result.error}",
            )
            return chunk

        chunk = _action_chunk(result.action_chunk)
        state.continuation = result.continuation
        state.steps_used += len(chunk)
        state.feedback = [
            f"Executed the requested target for {len(chunk)} simulator steps. "
            "Use the fresh measured pose and images to judge the result."
        ]
        mode = (result.decision or {}).get("mode")
        self._log_actions(chunk, source=str(mode or "eef"))
        return chunk
