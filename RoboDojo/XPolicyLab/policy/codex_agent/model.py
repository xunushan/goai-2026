"""``codex_agent``: a stateful Codex thread as the high-level policy for RoboDojo.

One episode is one Codex thread. At every decision point the adapter sends the
current camera views plus the observed arm state as a new turn on that thread,
receives a **target end-effector pose** (never a trajectory), and hands it to the
deterministic :mod:`motion` interpolator to produce an ActionChunk for the
simulator to execute open-loop. When the chunk is exhausted the loop starts
again from a freshly observed pose.

Two invariants dominate every line of this file, because violating either one
destroys an evaluation run rather than merely degrading it:

1. **``len(chunk) >= 1`` always.** ``deploy.py`` loops ``while not
   is_episode_end()``; an empty chunk advances no simulator step and the episode
   hangs forever.
2. **``get_action`` never raises.** An exception propagates to the eval client as
   a websocket error, which ``main.py`` swallows: the episode is dropped
   silently with ``eval_time = 0``.

The third constraint is the operator's: **at most ``max_codex_calls`` decisions
per episode**. Calls are counted whether or not they succeed, and running out
degrades to the task card's closing rule (both arms back at the start pose)
rather than to nothing.
"""

from __future__ import annotations

import json
import math
import os
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

_CUR_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _CUR_DIR.parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from XPolicyLab.model_template import ModelTemplate

from .bridge_client import BridgeClient, BridgeResult, build_image_payload
from .motion import (
    ArmCommand,
    ArmState,
    Box,
    GuardrailConfig,
    MotionConfig,
    check_arm_command,
    hold_chunk,
    interpolate_chunk,
    normalize_quat_wxyz,
    quat_angle_between,
)
from .prompt import (
    PromptContext,
    TurnFeedback,
    build_output_schema,
    build_system_prompt,
    build_turn_prompt,
    load_task_card,
)
from .protocol import ParseError, parse_decision

DEFAULT_CAMERA_NAMES = ("cam_head", "cam_left_wrist", "cam_right_wrist")
CAMERA_CANDIDATES = {
    "cam_head": ("cam_head", "cam_high", "head_camera", "top_camera"),
    "cam_left_wrist": ("cam_left_wrist", "left_camera", "left_wrist"),
    "cam_right_wrist": ("cam_right_wrist", "right_camera", "right_wrist"),
}

# Keys allowed inside a task card. Guards against someone pasting scoring
# thresholds into tasks/<task>.json later: the prompt only ever consumes prose.
TASK_CARD_KEYS = {
    "task_name",
    "instruction",
    "scene",
    "success_rule",
    "hints",
    "step_budget",
    "max_decisions",
}

LOG_PREFIX = "[codex_agent]"


# --------------------------------------------------------------------------- #
# observation helpers (kept local so this module never imports a heavy policy)
# --------------------------------------------------------------------------- #


def extract_image(observation: dict[str, Any], camera_name: str) -> np.ndarray:
    """Extract one camera view by exact name, then by fallback candidates."""
    vision = observation.get("vision", {})
    if not isinstance(vision, dict):
        raise KeyError("observation must contain a 'vision' mapping")

    def value(entry: Any) -> np.ndarray:
        if isinstance(entry, dict):
            for key in ("color", "rgb"):
                if key in entry:
                    return np.asarray(entry[key])
            raise KeyError(f"camera entry has no color/rgb field: {sorted(entry)}")
        return np.asarray(entry)

    for name in (camera_name, *CAMERA_CANDIDATES.get(camera_name, ())):
        if name in vision:
            return value(vision[name])
    raise KeyError(f"missing camera {camera_name!r}; available={sorted(vision)}")


def ensure_hwc_uint8(image: Any) -> np.ndarray:
    """Normalize a camera image to an HWC uint8 array."""
    array = np.asarray(image)
    if array.ndim != 3:
        raise ValueError(f"expected image ndim=3, got shape {array.shape}")
    if np.issubdtype(array.dtype, np.floating):
        array = (np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8)
    elif array.dtype != np.uint8:
        array = array.astype(np.uint8)
    if array.shape[-1] in (1, 3, 4):
        return array
    if array.shape[0] in (1, 3, 4):
        return np.transpose(array, (1, 2, 0))
    raise ValueError(f"unsupported image shape: {array.shape}")


def _section(cfg: dict[str, Any], name: str) -> dict[str, Any]:
    value = cfg.get(name)
    return dict(value) if isinstance(value, dict) else {}


def _as_pose(value: Any, what: str) -> np.ndarray:
    pose = np.asarray(value, dtype=np.float64).reshape(-1)
    if pose.shape != (7,):
        raise ValueError(f"{what} must be 7 numbers (xyz + quaternion wxyz), got {pose.shape}")
    if not np.isfinite(pose).all():
        raise ValueError(f"{what} contains NaN or Inf")
    return pose


def _home_pose(cfg: Any, what: str) -> np.ndarray:
    """Accept either ``[x,y,z,qw,qx,qy,qz]`` or ``{pos: [...], quat: [...]}``."""
    if not isinstance(cfg, dict):
        return _as_pose(cfg, what)
    if "pos" in cfg and "quat" in cfg:
        pos = np.asarray(cfg["pos"], dtype=np.float64).reshape(-1)
        if pos.shape != (3,) or not np.isfinite(pos).all():
            raise ValueError(f"{what}.pos must be 3 finite numbers, got {cfg['pos']!r}")
        return np.concatenate([pos, normalize_quat_wxyz(cfg["quat"])])
    if "pose" in cfg:
        return _as_pose(cfg["pose"], what)
    raise ValueError(f"{what} must be [x,y,z,qw,qx,qy,qz] or {{pos: [...], quat: [...]}}")


# --------------------------------------------------------------------------- #
# per-episode bookkeeping
# --------------------------------------------------------------------------- #


@dataclass
class EpisodeState:
    episode_id: str
    calls_used: int = 0
    sim_steps_used: int = 0
    turn_index: int = 0
    thread_id: str | None = None
    invalid_streak: int = 0
    bridge_failures: int = 0
    bridge_degraded: bool = False
    obs_failures: int = 0
    feedback: TurnFeedback = field(default_factory=lambda: TurnFeedback([]))
    prev_start_left: ArmState | None = None
    prev_start_right: ArmState | None = None
    prev_target: dict[str, Any] = field(default_factory=dict)
    flags: dict[str, Any] = field(default_factory=dict)
    decisions: list[dict[str, Any]] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# adapter
# --------------------------------------------------------------------------- #


class Model(ModelTemplate):
    def __init__(self, model_cfg: dict[str, Any]):
        self.model_cfg = dict(model_cfg)
        if str(self.model_cfg.get("action_type", "ee")) != "ee":
            raise ValueError("codex_agent supports only action_type='ee'.")

        bridge_cfg = _section(self.model_cfg, "bridge")
        episode_cfg = _section(self.model_cfg, "episode")
        motion_cfg = _section(self.model_cfg, "motion")
        guard_cfg = _section(self.model_cfg, "guardrail")
        codex_cfg = _section(self.model_cfg, "codex")
        image_cfg = _section(self.model_cfg, "images")
        home_cfg = _section(self.model_cfg, "home")

        # -- budget ------------------------------------------------------ #
        self.step_budget = int(episode_cfg.get("max_sim_steps", 400))
        self.max_codex_calls = int(episode_cfg.get("max_codex_calls", 10))
        self.min_sim_steps_per_call = int(episode_cfg.get("min_sim_steps_per_call", 5))
        self.count_failed_calls = bool(episode_cfg.get("count_failed_calls", True))
        if self.max_codex_calls < 1:
            raise ValueError("episode.max_codex_calls must be at least 1")
        if self.step_budget < 1:
            raise ValueError("episode.max_sim_steps must be at least 1")

        # -- motion ------------------------------------------------------ #
        self.motion = MotionConfig(
            delta_p_max_m=float(motion_cfg.get("delta_p_max_m", 0.015)),
            delta_theta_max_deg=float(motion_cfg.get("delta_theta_max_deg", 5.0)),
            settle_steps=int(motion_cfg.get("settle_steps", 3)),
            gripper_open=float(motion_cfg.get("gripper_open", 1.0)),
            gripper_close=float(motion_cfg.get("gripper_close", 0.0)),
        )
        self.min_chunk_steps = max(1, int(motion_cfg.get("min_chunk_steps", 2)))
        self.max_chunk_steps = max(self.min_chunk_steps, int(motion_cfg.get("max_chunk_steps", 45)))

        # -- workspace / home -------------------------------------------- #
        self.workspace_left = Box.from_cfg(guard_cfg.get("workspace", {}).get("left"))
        self.workspace_right = Box.from_cfg(guard_cfg.get("workspace", {}).get("right"))
        self.home_left = _home_pose(home_cfg.get("left"), "home.left")
        self.home_right = _home_pose(home_cfg.get("right"), "home.right")
        self.guardrail = GuardrailConfig(
            workspace=self.workspace_left,
            reject_margin_m=float(guard_cfg.get("reject_margin_m", 0.03)),
            max_target_distance_m=float(motion_cfg.get("max_target_distance_m", 0.45)),
        )
        self.guardrail_right = GuardrailConfig(
            workspace=self.workspace_right,
            reject_margin_m=self.guardrail.reject_margin_m,
            max_target_distance_m=self.guardrail.max_target_distance_m,
        )
        self.max_validation_retries = int(guard_cfg.get("max_validation_retries", 1))
        self.invalid_streak_limit = int(guard_cfg.get("invalid_streak_limit", 3))

        # -- bridge ------------------------------------------------------ #
        # Precedence: environment -> flat `bridge_url` override (what
        # setup_policy_server.py's --overrides can actually reach, since it only
        # sets top-level keys) -> the nested `bridge.base_url` in deploy.yml.
        bridge_url = (
            os.environ.get("CODEX_BRIDGE_URL")
            or self.model_cfg.get("bridge_url")
            or bridge_cfg.get("base_url")
            or "http://localhost:8765"
        )
        self.bridge = BridgeClient(
            str(bridge_url),
            token=os.environ.get("CODEX_BRIDGE_TOKEN") or bridge_cfg.get("token"),
            connect_timeout_s=float(bridge_cfg.get("connect_timeout_s", 5.0)),
        )
        self.request_timeout_s = float(bridge_cfg.get("request_timeout_s", 95.0))
        self.request_timeout_first_turn_s = float(
            bridge_cfg.get("request_timeout_first_turn_s", 105.0)
        )
        self.decision_wall_budget_s = float(bridge_cfg.get("decision_wall_budget_s", 110.0))
        self.use_output_schema = bool(bridge_cfg.get("use_output_schema", True))
        # The eval client kills the request at 120 s; a decision that takes longer
        # than the wall budget is abandoned here instead of losing the episode.
        self.request_timeout_s = min(self.request_timeout_s, self.decision_wall_budget_s)

        # -- images ------------------------------------------------------ #
        self.camera_names = tuple(
            str(name).strip()
            for name in (image_cfg.get("camera_names") or DEFAULT_CAMERA_NAMES)
            if str(name).strip()
        )
        self.jpeg_quality = int(image_cfg.get("jpeg_quality", 88))
        self.max_image_width = int(image_cfg.get("max_width", 640))

        # -- codex thread ------------------------------------------------ #
        self.codex_model = codex_cfg.get("model") or None
        self.codex_bin = str(codex_cfg.get("bin", ""))

        # -- task card --------------------------------------------------- #
        self.configured_task_name = self.model_cfg.get("task_name") or None
        self.prompt_context: PromptContext | None = None
        self._task_card: dict[str, Any] = {}

        # -- logging ----------------------------------------------------- #
        self.log_io = bool(self.model_cfg.get("log_io", True))
        output_dir = self.model_cfg.get("output_dir")
        self.output_dir = Path(str(output_dir)).expanduser() if output_dir else None
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self._decisions_path = self.output_dir / "decisions.jsonl"
        else:
            self._decisions_path = None
        self._actions_since_last_flush: list[np.ndarray] = []

        # -- runtime state ----------------------------------------------- #
        self._episode: EpisodeState | None = None
        self._episode_counter = 0
        self._request_index = 0
        self._latest_obs: dict[str, Any] | None = None
        self._latest_env_idx = 0
        self._last_left = ArmState.from_pose7(self.home_left, self.motion.gripper_open)
        self._last_right = ArmState.from_pose7(self.home_right, self.motion.gripper_open)
        self._hold_length = self.min_chunk_steps

        self._ensure_prompt_context()

        print(
            f"{LOG_PREFIX} ready bridge={self.bridge.base_url} "
            f"task={self.configured_task_name or '(from observation)'} "
            f"step_budget={self.step_budget} max_codex_calls={self.max_codex_calls} "
            f"max_chunk_steps={self.max_chunk_steps} settle={self.motion.settle_steps} "
            f"delta_p_max={self.motion.delta_p_max_m} "
            f"delta_theta_max_deg={self.motion.delta_theta_max_deg} "
            f"cameras={list(self.camera_names)}",
            flush=True,
        )

    # ------------------------------------------------------------------ #
    # prompt context
    # ------------------------------------------------------------------ #
    def _ensure_prompt_context(self) -> PromptContext:
        if self.prompt_context is not None:
            return self.prompt_context
        task_name = self.configured_task_name or "plug_in_charger"
        card = load_task_card(task_name)
        unknown = set(card) - TASK_CARD_KEYS
        if unknown:
            raise ValueError(
                f"task card {task_name!r} has unexpected keys {sorted(unknown)}; task cards "
                "must stay prose-only (see prompt.py's information boundary)"
            )
        self._task_card = card
        # The config, not the card, owns the budget: it is the operator's lever and
        # the prompt must describe the budget that is actually enforced.
        for key, actual in (
            ("step_budget", self.step_budget),
            ("max_decisions", self.max_codex_calls),
        ):
            if key in card and int(card[key]) != actual:
                print(
                    f"{LOG_PREFIX} note: task card {task_name!r} says {key}={card[key]} but "
                    f"deploy config enforces {actual}; the prompt will state {actual}.",
                    flush=True,
                )
        self.prompt_context = PromptContext(
            task=card,
            home_left=self.home_left,
            home_right=self.home_right,
            workspace_left=self.workspace_left,
            workspace_right=self.workspace_right,
            gripper_open=self.motion.gripper_open,
            gripper_close=self.motion.gripper_close,
            step_budget=self.step_budget,
            max_decisions=self.max_codex_calls,
            max_target_distance_m=self.guardrail.max_target_distance_m,
            delta_p_max_m=self.motion.delta_p_max_m,
            camera_names=self.camera_names,
        )
        return self.prompt_context

    # ------------------------------------------------------------------ #
    # ModelTemplate interface
    # ------------------------------------------------------------------ #
    def update_obs(self, obs):
        try:
            self._latest_obs = obs
            self._latest_env_idx = int(obs.get("env_idx", 0)) if isinstance(obs, dict) else 0
        except BaseException:  # noqa: BLE001 - update_obs must not raise either
            traceback.print_exc()

    def update_obs_batch(self, obs_list):
        try:
            if not obs_list:
                return
            self.update_obs(obs_list[0])
        except BaseException:  # noqa: BLE001
            traceback.print_exc()

    def get_action(self, **kwargs):
        try:
            return self._decide()
        except BaseException:  # noqa: BLE001 - see module docstring, invariant 2
            print(f"{LOG_PREFIX} unexpected error while deciding:", flush=True)
            traceback.print_exc()
            self._hold_length = self.max_chunk_steps
            return hold_chunk(self._last_left, self._last_right, self.max_chunk_steps)

    def get_action_batch(self, env_idx_list=None, **kwargs):
        return [self.get_action(**kwargs)]

    def reset(self):
        """Start a new episode. Thread creation is deferred to the first decision."""
        self._episode = None
        self._episode_counter += 1
        self._latest_obs = None
        self._request_index = 0
        self._hold_length = self.min_chunk_steps
        self._last_left = ArmState.from_pose7(self.home_left, self.motion.gripper_open)
        self._last_right = ArmState.from_pose7(self.home_right, self.motion.gripper_open)

    def prepare_case(self, case_meta=None):
        self._ensure_prompt_context()

    def on_trial_end(self, result=None):
        state = self._episode
        if state is None:
            return
        summary = {
            "event": "episode_end",
            "episode_id": state.episode_id,
            "calls_used": state.calls_used,
            "sim_steps_used": state.sim_steps_used,
            "decisions": state.turn_index,
            "thread_id": state.thread_id,
            "invalid_streak": state.invalid_streak,
            "bridge_failures": state.bridge_failures,
            "obs_failures": state.obs_failures,
            "flags": state.flags,
            "result": str(result)[:500] if result is not None else None,
        }
        print(f"{LOG_PREFIX}[episode] " + json.dumps(summary, ensure_ascii=False), flush=True)
        self._flush_actions()

    # ------------------------------------------------------------------ #
    # observation -> ArmState
    # ------------------------------------------------------------------ #
    def _read_arm_states(
        self, observation: dict[str, Any], state: EpisodeState
    ) -> tuple[ArmState, ArmState, list[str]]:
        notes: list[str] = []
        parsed: list[ArmState] = []
        for side in ("left", "right"):
            key = f"{side}_ee_pose"
            grip_key = f"{side}_ee_joint_state"
            try:
                pose = _as_pose(observation["state"][key], key)
                norm = float(np.linalg.norm(pose[3:7]))
                if not 0.5 <= norm <= 1.5:
                    notes.append(f"obs_quat_repaired_{side}")
                gripper = np.asarray(
                    observation["state"][grip_key], dtype=np.float64
                ).reshape(-1)[-1]
                parsed.append(ArmState.from_pose7(pose, gripper))
            except (KeyError, ValueError, IndexError, TypeError) as exc:
                notes.append(f"obs_pose_unusable_{side}: {exc}")
                fallback = self._last_left if side == "left" else self._last_right
                parsed.append(fallback)
        return parsed[0], parsed[1], notes

    # ------------------------------------------------------------------ #
    # images
    # ------------------------------------------------------------------ #
    def _build_images(
        self, observation: dict[str, Any], state: EpisodeState
    ) -> tuple[list[dict[str, str]], list[str]]:
        images: list[dict[str, str]] = []
        notes: list[str] = []
        for name in self.camera_names:
            try:
                rgb = ensure_hwc_uint8(extract_image(observation, name))
                images.append(
                    build_image_payload(
                        name,
                        rgb,
                        quality=self.jpeg_quality,
                        max_width=self.max_image_width,
                    )
                )
            except (KeyError, ValueError, TypeError, OSError) as exc:
                notes.append(f"camera_{name}_unavailable: {exc}")
                state.flags[f"missing_{name}"] = True
        if not images:
            notes.append("no_images")
            state.flags["no_images"] = True
        return images, notes

    # ------------------------------------------------------------------ #
    # budget
    # ------------------------------------------------------------------ #
    def _action_cap(self, state: EpisodeState) -> int:
        """Spread the remaining simulator steps across the remaining decisions."""
        remaining_calls = max(1, self.max_codex_calls - state.calls_used)
        remaining_steps = max(0, self.step_budget - state.sim_steps_used)
        reserve = self.min_sim_steps_per_call * max(0, remaining_calls - 1)
        cap = remaining_steps - reserve
        return int(max(self.min_chunk_steps, min(self.max_chunk_steps, cap)))

    def _budget_exhausted(self, state: EpisodeState) -> str:
        if state.calls_used >= self.max_codex_calls:
            return "decision_budget_reached"
        if state.sim_steps_used >= self.step_budget:
            return "step_budget_reached"
        return ""

    # ------------------------------------------------------------------ #
    # the decision
    # ------------------------------------------------------------------ #
    def _decide(self) -> list[dict[str, np.ndarray]]:
        observation = self._latest_obs or {}
        state = self._ensure_episode(observation)

        left, right, obs_notes = self._read_arm_states(observation, state)
        self._last_left, self._last_right = left, right
        state.obs_failures += sum(1 for note in obs_notes if note.startswith("obs_pose_unusable"))

        exhausted = self._budget_exhausted(state)
        if exhausted:
            state.flags[exhausted] = True
            return self._emit_synthetic(state, left, right, mode=exhausted,
                                        reason="the episode budget is spent; returning to the start pose")

        if state.bridge_degraded:
            return self._emit_synthetic(
                state, left, right, mode="bridge_degraded",
                reason="the Codex bridge is unavailable; returning to the start pose",
            )

        if state.invalid_streak >= self.invalid_streak_limit:
            state.invalid_streak = 0
            state.flags["force_home"] = True
            return self._emit_synthetic(
                state, left, right, mode="force_home_recovery",
                reason=f"{self.invalid_streak_limit} targets in a row were rejected",
            )

        return self._decide_with_codex(state, observation, left, right, obs_notes)

    def _decide_with_codex(
        self,
        state: EpisodeState,
        observation: dict[str, Any],
        left: ArmState,
        right: ArmState,
        obs_notes: list[str],
    ) -> list[dict[str, np.ndarray]]:
        ctx = self._ensure_prompt_context()
        cap = self._action_cap(state)
        prompt = build_turn_prompt(
            ctx,
            turn_index=state.turn_index,
            decisions_used=state.calls_used,
            steps_used=state.sim_steps_used,
            left=left,
            right=right,
            feedback=state.feedback,
        )
        images, image_notes = self._build_images(observation, state)
        notes = obs_notes + image_notes

        timeout_s = self.request_timeout_s if state.thread_id else self.request_timeout_first_turn_s
        timeout_s = min(timeout_s, self.decision_wall_budget_s)

        alive, _ = self.bridge.healthz() if state.thread_id is None else (True, "")
        if not alive:
            state.bridge_failures += 1
            state.flags["bridge_unreachable"] = True
            if state.bridge_failures >= 2:
                state.bridge_degraded = True
            return self._emit_synthetic(
                state, left, right, mode="bridge_unreachable",
                reason=f"the Codex bridge at {self.bridge.base_url} is not reachable; "
                       "held position instead of spending a decision",
                notes=notes,
            )

        self._record_call(state)
        result = self.bridge.decide(
            episode_id=state.episode_id,
            thread_id=state.thread_id,
            turn_index=state.turn_index,
            prompt=prompt,
            schema=build_output_schema() if self.use_output_schema else None,
            images=images,
            timeout_s=timeout_s,
        )
        if result.thread_id:
            state.thread_id = result.thread_id

        if not result.ok:
            state.bridge_failures += 1
            state.flags[f"bridge_error_{result.error_kind}"] = True
            if result.error_kind in ("bridge_unreachable", "bridge_bad_response") and state.bridge_failures >= 2:
                state.bridge_degraded = True
            state.feedback = TurnFeedback(
                [
                    f"Your previous decision could not be obtained ({result.error_kind}: "
                    f"{result.error}). The robot held still and the decision was still charged "
                    "to your budget. Continue from the current state."
                ]
            )
            return self._emit_synthetic(
                state, left, right, mode=f"codex_{result.error_kind or 'failure'}",
                reason=result.error or "codex call failed",
                notes=notes, result=result,
            )

        state.bridge_failures = 0
        try:
            decision = parse_decision(
                result.parsed,
                home_quat_left=ctx.home_quat_left,
                home_quat_right=ctx.home_quat_right,
                gripper_open=self.motion.gripper_open,
                gripper_close=self.motion.gripper_close,
            )
        except ParseError as exc:
            state.feedback = TurnFeedback(
                [
                    f"Your previous reply could not be parsed ({exc}). It is shown below so you "
                    "can fix the format. The decision was still charged to your budget.",
                    f"Your reply was: {result.text[:600]}",
                ]
            )
            return self._emit_synthetic(
                state, left, right, mode="parse_failure",
                reason=str(exc), notes=notes, result=result,
            )

        left_outcome = check_arm_command(decision.left, left, self.guardrail)
        right_outcome = check_arm_command(decision.right, right, self.guardrail_right)
        rejected = [o for o in (left_outcome, right_outcome) if not o.ok]
        clamped = [o for o in (left_outcome, right_outcome) if o.ok and o.clamped]

        if rejected:
            state.invalid_streak += 1
            state.flags["target_rejected"] = True
            feedback_lines = [o.feedback("left" if o is left_outcome else "right") for o in rejected]
            feedback_lines += [o.feedback("left" if o is left_outcome else "right") for o in clamped]
            if decision.problems:
                feedback_lines += [f"Format problem: {p}" for p in decision.problems]
            feedback_lines.append("Choose a target inside the valid workspace and try again.")
            state.feedback = TurnFeedback(feedback_lines)
            return self._emit_synthetic(
                state, left, right, mode="target_rejected",
                reason="; ".join(o.reason for o in rejected),
                notes=notes, result=result,
            )

        state.invalid_streak = 0

        if decision.is_noop:
            state.flags["noop_target"] = True
            state.feedback = TurnFeedback(
                [
                    "Your previous decision changed nothing (everything was \"keep\"), so the "
                    "robot held still and the decision was charged to your budget.",
                    "Issue a concrete motion next time.",
                ]
            )
            return self._emit_synthetic(
                state, left, right, mode="noop_target",
                reason="decision kept every component", notes=notes, result=result,
            )

        chunk, info = interpolate_chunk(
            left, left_outcome.command, right, right_outcome.command, cap, self.motion
        )
        self._record_decision(
            state, mode="codex", capacity=cap, chunk_len=len(chunk), decision=decision,
            left=left, right=right, result=result, info=info,
            clamped=bool(clamped), notes=notes,
        )
        state.prev_start_left, state.prev_start_right = left, right
        state.prev_target = {
            "left": left_outcome.command,
            "right": right_outcome.command,
        }
        state.feedback = self._build_feedback(left_outcome, right_outcome, info, left, right)
        state.sim_steps_used += len(chunk)
        self._remember_actions(chunk)
        return chunk

    def _build_feedback(
        self,
        left_outcome,
        right_outcome,
        info,
        left: ArmState,
        right: ArmState,
    ) -> TurnFeedback:
        """Describe the executed motion and, on the next turn, how far it actually got."""
        lines: list[str] = []
        for arm, outcome, start in (
            ("left", left_outcome, left),
            ("right", right_outcome, right),
        ):
            command: ArmCommand = outcome.command
            if command.is_keep:
                lines.append(f"You left the {arm} arm unchanged.")
                continue
            parts = []
            if command.position is not None:
                parts.append(f"position {np.round(command.position, 4).tolist()}")
            if command.quat is not None:
                parts.append(
                    f"orientation {round(math.degrees(quat_angle_between(start.quat, command.quat)), 1)} deg away"
                )
            if command.gripper is not None:
                parts.append(f"gripper {command.gripper:.2f}")
            lines.append(f"You commanded the {arm} arm: " + ", ".join(parts) + ".")
            if outcome.clamped:
                lines.append(f"The {arm} target was clamped onto the workspace boundary.")
        if info.truncated:
            lines.append(
                "The motion was cut short by the remaining step budget, so the arms stopped "
                f"part of the way ({', '.join(info.truncated_arms) or 'both arms'} did not arrive)."
            )
        if info.gripper_deferred:
            lines.append(
                "Your gripper command was held over because the arm had not arrived; it will "
                "take effect once the arm reaches the target."
            )
        lines.append(
            "The arm state reported above is measured after the motion completed, so compare it "
            "against the target you commanded."
        )
        return TurnFeedback(lines)

    # ------------------------------------------------------------------ #
    # synthetic (no Codex) decisions
    # ------------------------------------------------------------------ #
    def _emit_synthetic(
        self,
        state: EpisodeState,
        left: ArmState,
        right: ArmState,
        *,
        mode: str,
        reason: str,
        notes: list[str] | None = None,
        result: BridgeResult | None = None,
    ) -> list[dict[str, np.ndarray]]:
        """Return a chunk that was not produced by Codex.

        Every failure path lands here. Hold at the observed pose when the model
        might still recover, and drive back to the start pose once the episode is
        over budget -- the task card makes returning home part of the success
        condition, so the last steps of every episode spend themselves there.
        """
        ctx = self._ensure_prompt_context()
        go_home = mode in (
            "decision_budget_reached",
            "step_budget_reached",
            "bridge_degraded",
            "force_home_recovery",
        )
        if go_home:
            left_command = ArmCommand(
                position=self.home_left[:3],
                quat=ctx.home_quat_left,
                gripper=self.motion.gripper_open,
            )
            right_command = ArmCommand(
                position=self.home_right[:3],
                quat=ctx.home_quat_right,
                gripper=self.motion.gripper_open,
            )
            cap = max(
                self.min_chunk_steps,
                min(self.max_chunk_steps, self.step_budget - state.sim_steps_used),
            )
        else:
            left_command = right_command = ArmCommand()
            cap = max(1, self._hold_length)

        chunk, info = interpolate_chunk(left, left_command, right, right_command, cap, self.motion)
        state.sim_steps_used += len(chunk)
        self._record_decision(
            state, mode=mode, capacity=cap, chunk_len=len(chunk), decision=None,
            left=left, right=right, result=result, info=info, clamped=False,
            notes=notes or [], extra={"reason": reason},
        )
        self._remember_actions(chunk)
        return chunk

    # ------------------------------------------------------------------ #
    # episode plumbing
    # ------------------------------------------------------------------ #
    def _ensure_episode(self, observation: dict[str, Any]) -> EpisodeState:
        episode_idx = observation.get("episode_idx")
        episode_id = f"ep{self._episode_counter:04d}"
        if episode_idx is not None:
            episode_id = f"{episode_id}_idx{episode_idx}"
        if self._episode is None:
            self._episode = EpisodeState(episode_id=episode_id)
            self._episode.feedback = TurnFeedback([])
        elif observation.get("episode_idx") != getattr(self, "_last_episode_idx", None):
            # episode_idx changed without a reset(): the previous episode is over.
            previous = self._episode
            if previous.turn_index or previous.calls_used:
                print(
                    f"{LOG_PREFIX}[episode] "
                    + json.dumps(
                        {
                            "event": "episode_rotated",
                            "episode_id": previous.episode_id,
                            "calls_used": previous.calls_used,
                            "decisions": previous.turn_index,
                        }
                    ),
                    flush=True,
                )
            self._episode = EpisodeState(episode_id=episode_id)
        self._last_episode_idx = observation.get("episode_idx")
        return self._episode

    def _record_call(self, state: EpisodeState) -> None:
        state.calls_used += 1

    def _remember_actions(self, chunk: list[dict[str, np.ndarray]]) -> None:
        if self.output_dir is None:
            return
        for action in chunk:
            self._actions_since_last_flush.append(
                np.concatenate(
                    [
                        action["left_ee_pose"],
                        action["left_ee_joint_state"],
                        action["right_ee_pose"],
                        action["right_ee_joint_state"],
                    ]
                ).astype(np.float32)
            )

    def _flush_actions(self) -> None:
        if self.output_dir is None or not self._actions_since_last_flush:
            return
        try:
            array = np.stack(self._actions_since_last_flush)
            np.save(self.output_dir / "low_level_actions.npy", array)
        except (OSError, ValueError):
            traceback.print_exc()
        self._actions_since_last_flush = []

    def _record_decision(
        self,
        state: EpisodeState,
        *,
        mode: str,
        capacity: int,
        chunk_len: int,
        decision,
        left: ArmState,
        right: ArmState,
        result: BridgeResult | None,
        info,
        clamped: bool,
        notes: list[str],
        extra: dict[str, Any] | None = None,
    ) -> None:
        record: dict[str, Any] = {
            "event": "decision",
            "request": self._request_index,
            "episode_id": state.episode_id,
            "turn": state.turn_index,
            "mode": mode,
            "calls_used": state.calls_used,
            "sim_steps_used": state.sim_steps_used,
            "remaining_calls": max(0, self.max_codex_calls - state.calls_used),
            "remaining_steps": max(0, self.step_budget - state.sim_steps_used),
            "cap": capacity,
            "chunk_len": chunk_len,
            "clamped": bool(clamped),
            "thread_id": state.thread_id,
            "notes": notes,
        }
        if decision is not None:
            record.update(
                {
                    "note": decision.note,
                    "phase": decision.phase,
                    "problems": decision.problems,
                    "left_target": _command_summary(decision.left),
                    "right_target": _command_summary(decision.right),
                }
            )
        if info is not None:
            record.update(
                {
                    "n_raw_left": info.n_raw_left,
                    "n_raw_right": info.n_raw_right,
                    "truncated": info.truncated,
                    "gripper_deferred": info.gripper_deferred,
                    "settle_steps": info.settle_steps,
                }
            )
        if result is not None:
            record["codex"] = {
                "ok": result.ok,
                "error_kind": result.error_kind,
                "error": result.error[:400],
                "latency_ms": result.latency_ms,
                "runs_dir": result.runs_dir,
                "usage": result.usage,
            }
        if extra:
            record.update(extra)
        self._request_index += 1
        state.turn_index += 1
        state.decisions.append(record)

        line = LOG_PREFIX + "[decision] " + json.dumps(record, ensure_ascii=False, default=str)
        print(line, flush=True)
        if self._decisions_path is not None:
            try:
                with self._decisions_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            except OSError:
                traceback.print_exc()

    # ------------------------------------------------------------------ #
    def warmup_prompt(self) -> str:
        """The standing brief, exposed for offline prompt review and tests."""
        return build_system_prompt(self._ensure_prompt_context())


def _command_summary(command: ArmCommand) -> dict[str, Any]:
    return {
        "position": None if command.position is None else np.round(command.position, 5).tolist(),
        "quat": None if command.quat is None else np.round(command.quat, 5).tolist(),
        "gripper": command.gripper,
    }
