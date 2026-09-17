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
from typing import Any, Sequence

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
from .protocol import ParseError, ParsedDecision, parse_decision

DEFAULT_CAMERA_NAMES = ("cam_head", "cam_left_wrist", "cam_right_wrist")
# An axis with both ends open: the guardrail then enforces nothing along it.
_OPEN_BOX = {"x": [None, None], "y": [None, None], "z": [None, None]}
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
    """Accept either ``[x,y,z,qw,qx,qy,qz]`` or ``{pos: [...], quat: [...]}``.

    Retained for tests and for anyone who still wants to pin a pose by hand. The
    adapter itself no longer reads ``home:`` from deploy.yml -- see
    :class:`EpisodeState`.
    """
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
    # The pose the arms were in at this episode's first decision, read off the
    # robot's own observation rather than from a constant in deploy.yml: it is the
    # frame "orientation" is measured from and the pose the episode has to end at,
    # so it has to be the real one. Set once, by _decide.
    home_left: ArmState | None = None
    home_right: ArmState | None = None
    prompt_context: PromptContext | None = None
    prev_start_left: ArmState | None = None
    prev_start_right: ArmState | None = None
    prev_target: dict[str, Any] = field(default_factory=dict)
    flags: dict[str, Any] = field(default_factory=dict)
    decisions: list[dict[str, Any]] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# adapter
# --------------------------------------------------------------------------- #


def _problem_lines(problems: Sequence[str]) -> list[str]:
    """Explain the fields of the model's reply that could not be used.

    ``protocol.py`` is deliberately tolerant: a field it cannot read falls back
    to ``keep`` while the rest of the decision still executes. That is the right
    behaviour -- the arm should not freeze because one number was malformed --
    but it makes a partial failure look like a success. A model that asked for a
    tilt, had its ``orientation`` silently dropped, and is then told only "you
    commanded the left arm: position [...]" will ask for the same tilt again on
    every remaining turn and never learn why nothing tilts.

    So these lines go out on *every* path that follows a successful parse --
    a rejected target, a no-op, and a normal motion -- not just on rejection.
    Without them the only record of the problem is the decision log, which the
    model cannot read.
    """
    if not problems:
        return []
    lines = ["Part of your last reply could not be used; those fields were left unchanged:"]
    lines += [f"  - {p}" for p in problems]
    lines.append('Resend the affected field in the documented format, or say "keep" for it.')
    return lines


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

        # -- workspace ---------------------------------------------------- #
        # No bounds in deploy.yml means an open box, and an open box means the
        # guardrail clamps and rejects nothing: the targets the model asks for are
        # the targets the arm is given. The machinery is still here and still
        # tested, so a bound can be set again once something real pins one down --
        # but inventing one is worse than leaving it open, which is why nothing is
        # defaulted to a plausible-looking number.
        self.workspace_left = Box.from_cfg(guard_cfg.get("workspace", {}).get("left") or _OPEN_BOX)
        self.workspace_right = Box.from_cfg(guard_cfg.get("workspace", {}).get("right") or _OPEN_BOX)
        self.guardrail = GuardrailConfig(
            workspace=self.workspace_left,
            reject_margin_m=float(guard_cfg.get("reject_margin_m", 0.0)),
            max_target_distance_m=float(motion_cfg.get("max_target_distance_m", math.inf)),
            max_target_rotation_deg=float(motion_cfg.get("max_target_rotation_deg", math.inf)),
        )
        self.guardrail_right = GuardrailConfig(
            workspace=self.workspace_right,
            reject_margin_m=self.guardrail.reject_margin_m,
            max_target_distance_m=self.guardrail.max_target_distance_m,
            max_target_rotation_deg=self.guardrail.max_target_rotation_deg,
        )
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
        # The last pose the robot reported. None only before the very first
        # observation, which is the one case where we have nothing to hold at --
        # see _hold_state.
        self._last_left: ArmState | None = None
        self._last_right: ArmState | None = None
        self._hold_length = self.min_chunk_steps

        # Fail fast on a missing or malformed task card, without needing a pose.
        self._task_card = self._load_task_card()

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
    def _load_task_card(self) -> dict[str, Any]:
        """Read and validate the configured task card (prose-only, known keys)."""
        task_name = self.configured_task_name or "plug_in_charger"
        card = load_task_card(task_name)
        unknown = set(card) - TASK_CARD_KEYS
        if unknown:
            raise ValueError(
                f"task card {task_name!r} has unexpected keys {sorted(unknown)}; task cards "
                "must stay prose-only (see prompt.py's information boundary)"
            )
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
        return card

    def _build_prompt_context(self, card: dict[str, Any], left: ArmState,
                              right: ArmState) -> PromptContext:
        return PromptContext(
            task=card,
            home_left=left,
            home_right=right,
            gripper_open=self.motion.gripper_open,
            step_budget=self.step_budget,
            max_decisions=self.max_codex_calls,
            camera_names=self.camera_names,
        )

    def _ensure_prompt_context(self, state: EpisodeState | None = None) -> PromptContext:
        """The prompt context for an episode, built from that episode's own start pose.

        Cached per episode, not on the adapter: two episodes can begin in
        different poses, and the whole point of deriving the pose from the first
        observation is that the prompt describes the run that is actually
        happening.
        """
        state = self._episode if state is None else state
        if state is None:
            raise RuntimeError(
                "no episode is open, so there is no observed start pose to build the prompt "
                "from; the first decision of an episode is what establishes it"
            )
        if state.prompt_context is not None:
            return state.prompt_context
        if state.home_left is None or state.home_right is None:
            raise RuntimeError(
                "the arms' pose at this episode's first decision has not been read yet, and "
                "the prompt is defined relative to it"
            )
        card = self._task_card or self._load_task_card()
        state.prompt_context = self._build_prompt_context(card, state.home_left, state.home_right)
        return state.prompt_context

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
            return hold_chunk(
                self._hold_state("left"), self._hold_state("right"), self.max_chunk_steps
            )

    def get_action_batch(self, env_idx_list=None, **kwargs):
        return [self.get_action(**kwargs)]

    def reset(self):
        """Start a new episode. Thread creation is deferred to the first decision."""
        self._episode = None
        self._episode_counter += 1
        self._latest_obs = None
        self._request_index = 0
        self._hold_length = self.min_chunk_steps
        self._last_left = None
        self._last_right = None

    def prepare_case(self, case_meta=None):
        self._task_card = self._load_task_card()

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
                parsed.append(self._hold_state(side))
        return parsed[0], parsed[1], notes

    def _hold_state(self, side: str) -> ArmState:
        """The pose to hold when an observation carries no usable arm state.

        Normally this is simply the last pose the robot reported. Before the
        first observation there is no such pose, and rather than fill the gap
        with a plausible-looking guess -- the exact habit this module was
        rewritten to remove -- it commands the arms to the world origin and says
        so on stdout. A client that sends no arm state on its first frame has
        already lost the episode; a loud, obviously wrong hold beats a quiet one
        that looks like a real pose.
        """
        known = self._last_left if side == "left" else self._last_right
        if known is not None:
            return known
        print(
            f"{LOG_PREFIX} no arm state has ever been observed; holding {side} at the world "
            "origin (this episode cannot be salvaged)",
            flush=True,
        )
        return ArmState(
            pos=np.zeros(3),
            quat=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
            gripper=self.motion.gripper_open,
        )

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
        if remaining_steps == 0:
            return 0
        reserve = self.min_sim_steps_per_call * max(0, remaining_calls - 1)
        cap = remaining_steps - reserve
        # A final one-step remainder is valid.  The minimum chunk length is a
        # preferred size, never permission to run past the episode budget.
        preferred = max(self.min_chunk_steps, min(self.max_chunk_steps, cap))
        return int(min(remaining_steps, preferred))

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
        if state.home_left is None:
            # The first decision of the episode fixes the pose everything is
            # measured against: where the arms began, and where the task requires
            # them to be at the end.
            state.home_left, state.home_right = left, right

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
        ctx = self._ensure_prompt_context(state)
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
        if state.thread_id is None:
            # This call is the one that creates the Codex thread, so the standing
            # brief has to ride on it: `codex exec resume` only ever replays what
            # the thread was given, and there is no second chance to introduce
            # the embodiment, the task or the reply format. Later turns must NOT
            # repeat it -- the thread already holds it, and every repeated copy
            # is paid for out of a budget of ten calls.
            prompt = f"{build_system_prompt(ctx)}\n\n{prompt}"
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
            feedback_lines += _problem_lines(decision.problems)
            feedback_lines.append("Choose a target inside the valid workspace and try again.")
            state.feedback = TurnFeedback(feedback_lines)
            return self._emit_synthetic(
                state, left, right, mode="target_rejected",
                reason="; ".join(o.reason for o in rejected),
                notes=notes, result=result, decision=decision,
            )

        state.invalid_streak = 0

        if decision.is_noop:
            state.flags["noop_target"] = True
            noop_lines = _problem_lines(decision.problems)
            if decision.problems:
                # The unreadable fields are why everything collapsed to "keep",
                # so the usual "issue a concrete motion" advice would mislead.
                noop_lines.append(
                    "Those fields fell back to \"keep\", so the decision changed nothing: the "
                    "robot held still and the decision was charged to your budget."
                )
            else:
                noop_lines.append(
                    "Your previous decision changed nothing (everything was \"keep\"), so the "
                    "robot held still and the decision was charged to your budget."
                )
                noop_lines.append("Issue a concrete motion next time.")
            state.feedback = TurnFeedback(noop_lines)
            return self._emit_synthetic(
                state, left, right, mode="noop_target",
                reason="decision kept every component", notes=notes, result=result,
                decision=decision,
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
        state.feedback = self._build_feedback(
            left_outcome, right_outcome, info, left, right, problems=decision.problems
        )
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
        problems: Sequence[str] = (),
    ) -> TurnFeedback:
        """Describe the executed motion and, on the next turn, how far it actually got."""
        lines: list[str] = _problem_lines(problems)
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
        decision: ParsedDecision | None = None,
    ) -> list[dict[str, np.ndarray]]:
        """Return a chunk that was not produced by Codex.

        Every failure path lands here. Hold at the observed pose when the model
        might still recover, and drive back to the start pose once the episode is
        over budget -- the task card makes returning home part of the success
        condition, so the last steps of every episode spend themselves there.
        """
        ctx = self._ensure_prompt_context(state)
        go_home = mode in (
            "decision_budget_reached",
            "step_budget_reached",
            "bridge_degraded",
            "force_home_recovery",
        )
        if go_home:
            # Back to where the arms were at the first decision of *this* episode
            # -- the pose the robot reported, not a constant in deploy.yml.
            left_command = ArmCommand(
                position=ctx.home_left.pos,
                quat=ctx.home_quat_left,
                gripper=self.motion.gripper_open,
            )
            right_command = ArmCommand(
                position=ctx.home_right.pos,
                quat=ctx.home_quat_right,
                gripper=self.motion.gripper_open,
            )
            remaining_steps = max(0, self.step_budget - state.sim_steps_used)
            # This path is entered only after the normal budget check, so retain
            # a non-empty chunk for the simulator while never exceeding a
            # positive remainder.
            cap = max(1, min(self.max_chunk_steps, remaining_steps))
        else:
            left_command = right_command = ArmCommand()
            cap = max(1, self._hold_length)

        chunk, info = interpolate_chunk(left, left_command, right, right_command, cap, self.motion)
        state.sim_steps_used += len(chunk)
        self._record_decision(
            state, mode=mode, capacity=cap, chunk_len=len(chunk), decision=decision,
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
    def warmup_prompt(
        self, start_left: ArmState | None = None, start_right: ArmState | None = None
    ) -> str:
        """The standing brief, exposed for offline prompt review and tests.

        Nothing in the text depends on the pose any more, so any plausible pair
        renders the same brief -- the argument exists so the caller has to name
        the frame the run will be measured in, rather than let the adapter
        quietly supply one. When an episode is open, its observed start pose is
        used by default.
        """
        state = self._episode
        left = start_left or (state.home_left if state else None) or self._last_left
        right = start_right or (state.home_right if state else None) or self._last_right
        if left is None or right is None:
            raise RuntimeError(
                "warmup_prompt needs the pose the arms start the episode in; pass "
                "start_left/start_right, or run a decision first"
            )
        card = self._task_card or self._load_task_card()
        return build_system_prompt(self._build_prompt_context(card, left, right))


def _command_summary(command: ArmCommand) -> dict[str, Any]:
    return {
        "position": None if command.position is None else np.round(command.position, 5).tolist(),
        "quat": None if command.quat is None else np.round(command.quat, 5).tolist(),
        "gripper": command.gripper,
    }
