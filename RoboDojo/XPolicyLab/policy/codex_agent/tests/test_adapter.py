"""End-to-end tests for the ``codex_agent`` adapter, with no Codex involved.

``tests/fake_codex`` tests the real bridge by faking the CLI under it. This tests
``model.py`` against :mod:`tests.mock_bridge`, which replaces the bridge
altogether, so the whole adapter runs over real HTTP with no Codex, no
simulator and no Isaac.

The two invariants from the module docstring are the point of the exercise:

1. ``len(chunk) >= 1`` on **every** path -- an empty chunk hangs the evaluator.
2. ``get_action`` never raises -- an exception silently drops the episode with
   ``eval_time = 0``.

The third thing under test is the operator's budget: **at most 10 Codex calls per
episode**, counted whether or not they succeed.

    python tests/test_adapter.py
"""

from __future__ import annotations

import contextlib
import io
import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import yaml

_PKG = Path(__file__).resolve().parent.parent
if str(_PKG.parent) not in sys.path:
    sys.path.insert(0, str(_PKG.parent))

from codex_agent.model import Model  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mock_bridge import (  # noqa: E402
    FAILURE_MODES,
    MockBridge,
    legal_reply,
    unused_port,
)

CONFIG_PATH = _PKG / "deploy.yml"

HOME_LEFT_POS = np.array([-0.2995, -0.3523, 0.9215])
HOME_RIGHT_POS = np.array([0.3005, -0.3523, 0.9215])
HOME_QUAT = np.array([0.7070, 0.0, 0.0, 0.7072])

PROTOCOL_KEYS = {"left_ee_pose", "left_ee_joint_state", "right_ee_pose", "right_ee_joint_state"}

FAILURES: list[str] = []
CHECKS = 0


def check(condition: bool, label: str) -> None:
    global CHECKS
    CHECKS += 1
    if not condition:
        FAILURES.append(label)
        print(f"  FAIL  {label}")


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #


def load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def deep_update(base: dict, patch: dict) -> dict:
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def make_model(overrides: dict | None = None, **kwargs) -> Model:
    """A Model built from the shipped deploy.yml, with a few keys overridden."""
    config = load_config()
    deep_update(config, overrides or {})
    with contextlib.redirect_stdout(io.StringIO()):
        return Model(config, **kwargs)


def make_obs(
    *,
    episode_idx: int = 0,
    left_pos=None,
    right_pos=None,
    left_quat=None,
    right_quat=None,
    left_gripper: float = 1.0,
    right_gripper: float = 1.0,
    cameras=("cam_head", "cam_left_wrist", "cam_right_wrist"),
    camera_shape: tuple[int, int, int] = (480, 640, 3),
    include_state: bool = True,
) -> dict:
    obs: dict = {"episode_idx": episode_idx, "env_idx": 0, "task_name": "plug_in_charger"}
    if include_state:
        obs["state"] = {
            "left_ee_pose": np.concatenate(
                [HOME_LEFT_POS if left_pos is None else left_pos,
                 HOME_QUAT if left_quat is None else left_quat]
            ).astype(np.float32),
            "left_ee_joint_state": np.array([left_gripper], dtype=np.float32),
            "right_ee_pose": np.concatenate(
                [HOME_RIGHT_POS if right_pos is None else right_pos,
                 HOME_QUAT if right_quat is None else right_quat]
            ).astype(np.float32),
            "right_ee_joint_state": np.array([right_gripper], dtype=np.float32),
        }
    obs["vision"] = {
        name: {"color": np.zeros(camera_shape, dtype=np.uint8)} for name in cameras
    }
    return obs


def step(model: Model, obs: dict) -> tuple[list[dict], str]:
    """One decision, with the adapter's stdout captured."""
    model.update_obs(obs)
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        chunk = model.get_action()
    return chunk, buffer.getvalue()


def decisions_in(text: str) -> list[dict]:
    prefix = "[codex_agent][decision] "
    return [
        json.loads(line[len(prefix):])
        for line in text.splitlines()
        if line.startswith(prefix)
    ]


def check_chunk_shape(chunk, label: str) -> None:
    """Every chunk, on every path, must be a legal ActionChunk."""
    check(isinstance(chunk, list), f"{label}: the chunk is a list")
    check(len(chunk) >= 1, f"{label}: the chunk is never empty (len={len(chunk) if chunk else 0})")
    if not chunk:
        return
    for index, action in enumerate(chunk):
        check(isinstance(action, dict), f"{label}: step {index} is a dict")
        check(
            set(action) == PROTOCOL_KEYS,
            f"{label}: step {index} has exactly the protocol keys, got {sorted(action)}",
        )
        if set(action) != PROTOCOL_KEYS:
            return
        for key, width in (
            ("left_ee_pose", 7),
            ("left_ee_joint_state", 1),
            ("right_ee_pose", 7),
            ("right_ee_joint_state", 1),
        ):
            array = np.asarray(action[key])
            check(array.shape == (width,), f"{label}: {key} has shape ({width},), got {array.shape}")
            check(array.dtype == np.float32, f"{label}: {key} is float32, got {array.dtype}")
            check(np.isfinite(array).all(), f"{label}: {key} is finite")
        for key in ("left_ee_pose", "right_ee_pose"):
            norm = float(np.linalg.norm(np.asarray(action[key])[3:]))
            check(abs(norm - 1.0) < 1e-5, f"{label}: {key} quaternion is unit norm, got {norm:.6f}")


# --------------------------------------------------------------------------- #
# the happy path
# --------------------------------------------------------------------------- #


def test_healthy_episode() -> None:
    print("a healthy episode")
    with MockBridge("legal") as bridge:
        bridge.wait_until_responsive()
        model = make_model({"bridge_url": bridge.url})
        chunk, text = step(model, make_obs())

        check_chunk_shape(chunk, "healthy")
        records = decisions_in(text)
        check(len(records) == 1, f"one decision was logged, got {len(records)}")
        record = records[0]
        check(record["mode"] == "codex", f"the decision came from codex, mode={record['mode']}")
        check(record["calls_used"] == 1, "the call was charged to the budget")
        check(record["remaining_calls"] == model.max_codex_calls - 1, "the remaining budget is reported")
        check(record["phase"] == "approach", "the model's phase reached the log")
        check(record["note"].startswith("moving the left arm"), "the model's note reached the log")
        check(record["codex"]["ok"] is True, "the codex result is recorded")
        check(record["codex"]["latency_ms"] == 24000, "the bridge latency is carried through")
        check(record["clamped"] is False, "nothing was clamped")
        check(record["truncated"] is False, "nothing was truncated")

        # 0.222 m of travel at 1.5 cm per simulator step, plus 3 settle steps.
        travel = float(np.linalg.norm(np.array(legal_reply()["left"]["position"]) - HOME_LEFT_POS))
        check(abs(travel - 0.2221) < 2e-3, f"the test target is 0.222 m away, got {travel:.4f}")
        expected_motion = math.ceil(travel / model.motion.delta_p_max_m - 1e-9)
        check(
            len(chunk) == expected_motion + model.motion.settle_steps,
            f"the chunk is {expected_motion}+{model.motion.settle_steps} steps, got {len(chunk)}",
        )

        # the moving arm arrives at the target; the other one is untouched
        target = np.array(legal_reply()["left"]["position"], dtype=np.float32)
        check(
            np.allclose(chunk[-1]["left_ee_pose"][:3], target, atol=1e-6),
            f"the left arm arrived at the target, got {chunk[-1]['left_ee_pose'][:3]}",
        )
        check(
            np.allclose(chunk[-1]["right_ee_pose"][:3], HOME_RIGHT_POS, atol=1e-6),
            "the right arm stayed where it was",
        )

        # the trailing settle steps repeat the final pose exactly
        for key in PROTOCOL_KEYS:
            check(
                np.array_equal(chunk[-1][key], chunk[-2][key]),
                f"the settle steps repeat the final pose exactly ({key})",
            )

        # the thread is created once and reused
        check(
            bridge.last_payload()["thread_id"] is None,
            "the first turn starts a fresh thread",
        )
        step(model, make_obs())
        delivered = bridge.last_payload()["thread_id"]
        check(delivered is not None, "the second turn resumes a thread")
        check(
            delivered == model._episode.thread_id,
            f"the adapter replays the thread id the bridge handed back: {delivered}",
        )
        check(bridge.decide_count == 2, "two decisions reached the bridge")
        check(bridge.healthz_count == 1, "healthz is probed once, not on every turn")


# --------------------------------------------------------------------------- #
# the standing brief
# --------------------------------------------------------------------------- #


def test_the_brief_rides_on_the_thread_creating_turn_only() -> None:
    """The system prompt must reach Codex, and reach it exactly once.

    ``build_system_prompt`` and ``build_turn_prompt`` describe themselves as two
    halves of one message ("sent once when the thread is created" / "appended to
    the standing brief"), but nothing joined them: the adapter sent the turn
    prompt alone, so the model never learned the embodiment, the task or the
    reply format. Individually rendered, both prompts look complete, which is
    why the offline prompt review did not catch it. Hence a request-level
    assertion on the payload that actually goes out.
    """
    print("the standing brief is sent once, on the thread-creating turn")
    with MockBridge("legal") as bridge:
        bridge.wait_until_responsive()
        model = make_model({"bridge_url": bridge.url})
        step(model, make_obs())
        step(model, make_obs())

        check(bridge.decide_count == 2, "two decisions reached the bridge")
        first, second = bridge.decide_payloads[0], bridge.decide_payloads[1]

        # Sentinels: if the turn ordering ever changes, these fire first, so the
        # two assertions below cannot pass vacuously on the wrong turns.
        check(first["thread_id"] is None, "turn 1 is the thread-creating turn")
        check(second["thread_id"] is not None, "turn 2 resumes that thread")
        check(first["turn_index"] == 0 and second["turn_index"] == 1, "the turns are ordered")

        for label, needle in (
            ("the opening line", "high-level manipulation policy"),
            ("the embodiment", "EMBODIMENT"),
            ("the start pose", "START POSE"),
            ("the task", "Plug the charger into the power strip."),
            ("the reply format", "REPLY FORMAT"),
        ):
            check(needle in first["prompt"], f"the first prompt carries {label}")

        check(
            "high-level manipulation policy" not in second["prompt"],
            "the brief is not repeated once the thread holds it",
        )
        check(
            "TURN 2 of at most" in second["prompt"],
            "the second prompt is still the per-decision one",
        )
        check(
            len(second["prompt"]) < len(first["prompt"]),
            "a resumed turn is shorter than the one that introduced the thread",
        )


def test_every_turn_names_the_views_it_attaches() -> None:
    """Three images ride on every turn, so their order has to be stated.

    The images go out as attachments in ``camera_names`` order; nothing else in
    the payload identifies them. Without a manifest naming that order and saying
    which arm each wrist view belongs to, an arm has only the pixels to go on --
    and the recorded run shows it guessing wrong ("the wrist views show both
    objects behind the current grasp centers").
    """
    print("each turn states which attached view is which")
    with MockBridge("legal") as bridge:
        bridge.wait_until_responsive()
        model = make_model({"bridge_url": bridge.url})
        step(model, make_obs())
        step(model, make_obs())

        first, second = bridge.decide_payloads[0], bridge.decide_payloads[1]

        # The order line rides on every turn: it is what disambiguates the
        # attachments, so it cannot live only in the standing brief.
        order = "ATTACHED VIEWS, in this order: 1. cam_head, 2. cam_left_wrist, 3. cam_right_wrist"
        check(order in first["prompt"], "turn 1 states the attachment order")
        check(order in second["prompt"], "turn 2 states it too")

        # The per-arm reading guidance is part of the standing brief only.
        for needle in (
            "mounted on the LEFT arm",
            "mounted on the RIGHT arm",
            "a close-up of that one gripper",
        ):
            check(needle in first["prompt"], f"the brief explains the wrist views: {needle!r}")
        check(
            "mounted on the LEFT arm" not in second["prompt"],
            "the brief is not repeated, so the resumed turn stays cheap",
        )


# --------------------------------------------------------------------------- #
# the operator's budget
# --------------------------------------------------------------------------- #


def test_call_cap_is_enforced() -> None:
    print("the <=10 Codex call cap")
    with MockBridge("legal") as bridge:
        bridge.wait_until_responsive()
        model = make_model({"bridge_url": bridge.url})
        obs = make_obs()

        modes = []
        lengths = []
        for _ in range(model.max_codex_calls + 5):
            chunk, text = step(model, obs)
            check_chunk_shape(chunk, "cap")
            lengths.append(len(chunk))
            records = decisions_in(text)
            modes.append(records[-1]["mode"] if records else "?")

        check(
            bridge.decide_count == model.max_codex_calls,
            f"exactly {model.max_codex_calls} calls reached the bridge, got {bridge.decide_count}",
        )
        codex_modes = [mode for mode in modes if mode == "codex"]
        check(
            len(codex_modes) == model.max_codex_calls,
            f"{model.max_codex_calls} decisions were served by codex, got {len(codex_modes)}",
        )
        check(
            modes[model.max_codex_calls:] == ["decision_budget_reached"] * 5,
            f"every later decision is reported as out of budget, got {modes[model.max_codex_calls:]}",
        )
        check(all(length >= 1 for length in lengths), "no chunk was ever empty")
        check(model._episode.flags.get("decision_budget_reached") is True, "the flag is set")
        check(
            model._episode.sim_steps_used <= model.step_budget,
            f"the step budget is respected: {model._episode.sim_steps_used} <= {model.step_budget}",
        )


def test_exhausted_budget_drives_home() -> None:
    print("an exhausted budget returns the arms home, not to a standstill")
    with MockBridge("legal") as bridge:
        bridge.wait_until_responsive()
        model = make_model({"bridge_url": bridge.url})
        obs = make_obs()

        for _ in range(model.max_codex_calls):
            step(model, obs)

        # Feed an observation in which the arms are somewhere else entirely, so a
        # hold and a return-home are clearly distinguishable.
        away = np.array([-0.15, -0.25, 1.05])
        obs_away = make_obs(left_pos=away, right_pos=np.array([0.15, -0.25, 1.05]))
        chunk, text = step(model, obs_away)
        check_chunk_shape(chunk, "go_home")
        record = decisions_in(text)[-1]
        check(record["mode"] == "decision_budget_reached", f"mode={record['mode']}")
        check(bridge.decide_count == model.max_codex_calls, "no extra call was made")

        # The chunk must move the left arm back towards HOME, not hold it at
        # `away`. Only the first few steps run here, so check the direction.
        start = float(np.linalg.norm(away - HOME_LEFT_POS))
        end = float(np.linalg.norm(np.asarray(chunk[-1]["left_ee_pose"][:3]) - HOME_LEFT_POS))
        check(end < start, f"the left arm moved towards home: {start:.4f} -> {end:.4f}")
        check(
            np.allclose(chunk[-1]["left_ee_joint_state"], [1.0], atol=1e-6),
            "the gripper is opened on the way home",
        )


def test_failed_calls_still_cost_budget() -> None:
    print("a failed call is still charged")
    with MockBridge("unparseable") as bridge:
        bridge.wait_until_responsive()
        model = make_model({"bridge_url": bridge.url})
        obs = make_obs()

        for _ in range(3):
            step(model, obs)
        check(model._episode.calls_used == 3, f"three failed calls were charged, got {model._episode.calls_used}")
        check(bridge.decide_count == 3, "each failure was a real request")


# --------------------------------------------------------------------------- #
# failure modes
# --------------------------------------------------------------------------- #


def test_every_failure_mode_still_yields_a_chunk() -> None:
    print("no failure mode escapes as an exception")
    for mode in ("all_keep", "out_of_box", "far_away", *FAILURE_MODES):
        delay = 0.6 if mode == "hung" else 0.0
        with MockBridge(mode, delay_s=delay) as bridge:
            bridge.wait_until_responsive()
            overrides = {
                "bridge_url": bridge.url,
                "bridge": {
                    "request_timeout_s": 0.4,
                    "request_timeout_first_turn_s": 0.4,
                    "decision_wall_budget_s": 1.5,
                    "connect_timeout_s": 0.3,
                },
            }
            model = make_model(overrides)
            try:
                chunk, text = step(model, make_obs())
            except BaseException as exc:  # noqa: BLE001 - that is exactly what we test
                check(False, f"{mode}: get_action raised {type(exc).__name__}: {exc}")
                continue
            check_chunk_shape(chunk, mode)
            records = decisions_in(text)
            check(len(records) == 1, f"{mode}: one decision was logged")
            if not records:
                continue
            record = records[-1]
            check(record["mode"] != "codex", f"{mode}: the decision was not served by codex")
            if mode in FAILURE_MODES:
                expected = FAILURE_MODES[mode]
                check(
                    record["mode"] == f"codex_{expected}",
                    f"{mode}: mode is codex_{expected}, got {record['mode']}",
                )
                check(
                    record["codex"]["error_kind"] == expected,
                    f"{mode}: error_kind is {expected}, got {record['codex']['error_kind']}",
                )
            # the episode survived
            check(model._episode is not None, f"{mode}: the episode state survived")


def test_all_keep_is_reported_as_a_noop() -> None:
    print("an all-keep reply is a charged no-op, not a hang")
    with MockBridge("all_keep") as bridge:
        bridge.wait_until_responsive()
        model = make_model({"bridge_url": bridge.url})
        chunk, text = step(model, make_obs())
        check_chunk_shape(chunk, "all_keep")
        record = decisions_in(text)[-1]
        check(record["mode"] == "noop_target", f"mode={record['mode']}")
        check(model._episode.flags.get("noop_target") is True, "the flag is set")
        check(model._episode.calls_used == 1, "the no-op still cost a call")
        check(
            np.allclose(chunk[-1]["left_ee_pose"][:3], HOME_LEFT_POS, atol=1e-6),
            "the arm did not move",
        )

        # the next turn's feedback must tell the model its reply was wasted
        step(model, make_obs())
        payload = bridge.last_payload()
        check("charged to your budget" in payload["prompt"], "the feedback explains the waste")
        # Nothing was malformed here, so the format-problem lines must stay away:
        # they are a diagnostic, not boilerplate in every prompt.
        check(
            "could not be used" not in payload["prompt"],
            "a clean all-keep reply does not get the format-problem lines",
        )


def test_a_noop_caused_by_bad_fields_says_so() -> None:
    print("a no-op caused by unreadable fields is explained, not blamed on the model")
    with MockBridge("broken_fields") as bridge:
        bridge.wait_until_responsive()
        model = make_model({"bridge_url": bridge.url})
        chunk, text = step(model, make_obs())
        check_chunk_shape(chunk, "broken_fields")
        record = decisions_in(text)[-1]
        check(record["mode"] == "noop_target", f"every field fell back to keep: {record['mode']}")
        check(len(record["problems"]) >= 4, f"all four fields are reported: {record['problems']}")
        check(
            np.allclose(chunk[-1]["left_ee_pose"][:3], HOME_LEFT_POS, atol=1e-6),
            "the arm did not move",
        )

        step(model, make_obs())
        prompt = bridge.last_payload()["prompt"]
        check("could not be used" in prompt, "the model is told the fields were unreadable")
        check("left.position" in prompt, "and which fields they were")
        check(
            "Those fields fell back to \"keep\"" in prompt,
            "the no-op is attributed to the format problem, not to indecision",
        )
        # The wrong diagnosis would send the model off to pick a target, which
        # is not the fix when the target it already picked was unreadable.
        check(
            "Issue a concrete motion next time" not in prompt,
            "the misleading advice is suppressed on this path",
        )


def test_out_of_workspace_targets_are_rejected() -> None:
    print("targets outside the workspace are rejected, and explained")
    for mode in ("out_of_box", "far_away"):
        with MockBridge(mode) as bridge:
            bridge.wait_until_responsive()
            model = make_model({"bridge_url": bridge.url})
            chunk, text = step(model, make_obs())
            check_chunk_shape(chunk, mode)
            record = decisions_in(text)[-1]
            check(record["mode"] == "target_rejected", f"{mode}: mode={record['mode']}")
            check(model._episode.invalid_streak == 1, f"{mode}: the invalid streak advanced")
            # The audit log has to show what was asked for, not just that it was
            # refused -- otherwise "why was this rejected?" is unanswerable from
            # the decision lines alone.
            check(
                record["left_target"]["position"] is not None,
                f"{mode}: the rejected target is recorded ({record['left_target']})",
            )
            check(
                np.allclose(chunk[-1]["left_ee_pose"][:3], HOME_LEFT_POS, atol=1e-6),
                f"{mode}: the arm was not moved to the illegal target",
            )

            # the feedback for the next turn names the workspace
            step(model, make_obs())
            prompt = bridge.last_payload()["prompt"]
            check("rejected and NOT executed" in prompt, f"{mode}: the rejection is explained")

    # three rejections in a row trigger a free return home
    with MockBridge("out_of_box") as bridge:
        bridge.wait_until_responsive()
        model = make_model({"bridge_url": bridge.url})
        modes = []
        for _ in range(model.invalid_streak_limit + 1):
            _, text = step(model, make_obs())
            modes.append(decisions_in(text)[-1]["mode"])
        check(
            modes[-1] == "force_home_recovery",
            f"the third rejection forces a return home, got {modes}",
        )


def test_degenerate_orientation_falls_back_to_keep() -> None:
    print("an unusable orientation is repaired into a keep, not a crash")
    with MockBridge("absurd_quat") as bridge:
        bridge.wait_until_responsive()
        model = make_model({"bridge_url": bridge.url})
        chunk, text = step(model, make_obs())
        check_chunk_shape(chunk, "absurd_quat")
        record = decisions_in(text)[-1]
        # the position is legal, so the motion still happens; only the
        # orientation falls back to keep -- and the model is told why.
        check(record["mode"] == "codex", f"the motion still happened, mode={record['mode']}")
        check(
            any("invalid quaternion" in problem for problem in record["problems"]),
            f"the bad quaternion is reported: {record['problems']}",
        )
        check(record["left_target"]["quat"] is None, "the orientation fell back to keep")
        check(record["left_target"]["position"] is not None, "the position survived")
        check(model._episode.invalid_streak == 0, "a repaired field is not an invalid target")

        # A repaired field is only recoverable if the model hears about it: the
        # motion still happened, so nothing else in the feedback gives it away.
        # Without this, the model re-sends the same bad orientation every turn.
        check(
            model._episode.feedback is not None
            and any("invalid quaternion" in line for line in model._episode.feedback.lines),
            f"the next turn's feedback carries the problem: {model._episode.feedback.lines}",
        )
        step(model, make_obs())
        prompt = bridge.last_payload()["prompt"]
        check("could not be used" in prompt, "the problem reached the model's next prompt")
        check("invalid quaternion" in prompt, "and it names the offending field")


def test_hung_bridge_returns_within_the_wall_budget() -> None:
    print("a hung bridge is abandoned inside the wall budget")
    with MockBridge("hung", delay_s=10.0) as bridge:
        bridge.wait_until_responsive()
        model = make_model(
            {
                "bridge_url": bridge.url,
                "bridge": {
                    "request_timeout_s": 0.5,
                    "request_timeout_first_turn_s": 0.5,
                    "decision_wall_budget_s": 1.2,
                    "connect_timeout_s": 0.4,
                },
            }
        )
        started = time.monotonic()
        chunk, text = step(model, make_obs())
        elapsed = time.monotonic() - started
        check_chunk_shape(chunk, "hung")
        check(elapsed < 3.0, f"the decision returned in {elapsed:.2f}s, well under 120 s")
        record = decisions_in(text)[-1]
        check(record["mode"] == "codex_bridge_unreachable", f"mode={record['mode']}")
        check(
            model.request_timeout_s <= model.decision_wall_budget_s,
            "the effective per-turn timeout never exceeds the wall budget",
        )


def test_unreachable_bridge_degrades_after_two_failures() -> None:
    print("a dead bridge is retried once, then abandoned for the episode")
    model = make_model({"bridge_url": f"http://127.0.0.1:{unused_port()}"})
    obs = make_obs()

    chunk, text = step(model, obs)
    check_chunk_shape(chunk, "unreachable")
    check(
        decisions_in(text)[-1]["mode"] == "bridge_unreachable",
        f"mode={decisions_in(text)[-1]['mode']}",
    )
    check(model._episode.bridge_failures == 1, "one failure recorded")
    check(model._episode.bridge_degraded is False, "a single failure is not yet fatal")

    chunk, text = step(model, obs)
    check_chunk_shape(chunk, "unreachable_2")
    check(model._episode.bridge_degraded is True, "the bridge is marked degraded after two")
    check(model._episode.calls_used == 0, "an unreachable bridge costs no Codex quota")

    chunk, text = step(model, obs)
    check_chunk_shape(chunk, "degraded")
    check(
        decisions_in(text)[-1]["mode"] == "bridge_degraded",
        "the episode now runs without Codex",
    )
    check(model._episode.calls_used == 0, "and still costs no quota")


# --------------------------------------------------------------------------- #
# observation handling
# --------------------------------------------------------------------------- #


def test_images_are_encoded_and_named() -> None:
    print("the camera views are encoded and labelled")
    with MockBridge("legal") as bridge:
        bridge.wait_until_responsive()
        model = make_model({"bridge_url": bridge.url})
        step(model, make_obs())

        images = bridge.last_payload()["images"]
        check(len(images) == 3, f"three views were sent, got {len(images)}")
        check(
            [entry["name"] for entry in images] == list(model.camera_names),
            f"the views are named after the cameras: {[e['name'] for e in images]}",
        )
        for entry in images:
            check(entry["format"] == "image/jpeg", "the payload declares JPEG")
            check(entry["mime"] == "image/jpeg", "the mime type is JPEG")
            import base64

            payload = base64.b64decode(entry["b64"], validate=True)
            check(payload.startswith(b"\xff\xd8\xff"), "the decoded payload really is a JPEG")
            check(len(payload) < 4 * 1024 * 1024, "each view fits the bridge's per-image limit")

        # the schema is attached, and is the same object the prompt promises
        schema = bridge.last_payload()["schema"]
        check(isinstance(schema, dict) and schema.get("type") == "object", "the schema was sent")
        check(set(schema["required"]) == {"left", "right", "note", "phase"}, "the schema is the contract")

        # the prompt and the schema agree about the budget
        prompt = bridge.last_payload()["prompt"]
        check(f"{model.max_codex_calls}" in prompt, "the prompt states the decision budget")
        check(f"{model.step_budget}" in prompt, "the prompt states the step budget")


def test_a_missing_camera_is_survivable() -> None:
    print("a missing camera drops that view instead of the decision")
    with MockBridge("legal") as bridge:
        bridge.wait_until_responsive()
        model = make_model({"bridge_url": bridge.url})
        chunk, text = step(model, make_obs(cameras=("cam_head",)))
        check_chunk_shape(chunk, "one_camera")
        images = bridge.last_payload()["images"]
        check(len(images) == 1, f"only the available view was sent, got {len(images)}")
        check(model._episode.flags.get("missing_cam_left_wrist") is True, "the gap is flagged")
        check(decisions_in(text)[-1]["mode"] == "codex", "the decision still happened")


def test_no_cameras_at_all_is_survivable() -> None:
    print("no cameras at all still produces a decision")
    with MockBridge("legal") as bridge:
        bridge.wait_until_responsive()
        model = make_model({"bridge_url": bridge.url})
        chunk, text = step(model, make_obs(cameras=()))
        check_chunk_shape(chunk, "no_cameras")
        check(bridge.last_payload()["images"] == [], "no images were sent")
        check(model._episode.flags.get("no_images") is True, "the gap is flagged")
        record = decisions_in(text)[-1]
        check(record["mode"] == "codex", "the model reasoned from the state alone")
        check("no_images" in record["notes"], f"the note explains it: {record['notes']}")


def test_unusable_observation_falls_back_to_the_last_pose() -> None:
    print("a broken observation does not break the decision")
    with MockBridge("legal") as bridge:
        bridge.wait_until_responsive()
        model = make_model({"bridge_url": bridge.url})

        chunk, text = step(model, make_obs(include_state=False))
        check_chunk_shape(chunk, "no_state")
        record = decisions_in(text)[-1]
        check(record["mode"] == "codex", "the decision still happened")
        check(
            any("obs_pose_unusable" in note for note in record["notes"]),
            f"the failure is noted: {record['notes']}",
        )
        check(model._episode.obs_failures == 2, f"both arms failed, got {model._episode.obs_failures}")

        # a quaternion that is not unit norm is repaired, not rejected
        chunk, text = step(model, make_obs(left_quat=np.array([2.0, 0.0, 0.0, 0.0])))
        check_chunk_shape(chunk, "bad_quat")
        check(
            any("obs_quat_repaired_left" in note for note in decisions_in(text)[-1]["notes"]),
            "the repair is noted",
        )
        check(model._episode.obs_failures == 2, "a repaired quaternion is not a failure")

        # a degenerate observation (the debug client sends ones) must survive too
        ones = make_obs()
        ones["state"]["left_ee_pose"] = np.ones(7, dtype=np.float32)
        ones["state"]["right_ee_pose"] = np.ones(7, dtype=np.float32)
        chunk, text = step(model, ones)
        check_chunk_shape(chunk, "ones")


# --------------------------------------------------------------------------- #
# episode plumbing
# --------------------------------------------------------------------------- #


def test_reset_twice_is_harmless() -> None:
    print("reset() is idempotent and creates no thread")
    with MockBridge("legal") as bridge:
        bridge.wait_until_responsive()
        model = make_model({"bridge_url": bridge.url})

        model.reset()
        model.reset()
        check(model._episode is None, "no episode exists until the first decision")
        check(bridge.decide_count == 0, "reset() contacted no bridge")
        check(bridge.healthz_count == 0, "reset() probed nothing")

        step(model, make_obs())
        check(bridge.decide_count == 1, "the first decision is the first contact")

        # reset between episodes: the counter moves on, the chunk stays legal
        model.reset()
        model.reset()
        chunk, _ = step(model, make_obs())
        check_chunk_shape(chunk, "after_reset")
        check(model._episode.calls_used == 1, "the budget is reset with the episode")

    # resetting mid-episode must clear the thread, not resume it
    with MockBridge("legal") as bridge:
        bridge.wait_until_responsive()
        model = make_model({"bridge_url": bridge.url})
        step(model, make_obs())
        check(model._episode.thread_id is not None, "a thread was established")
        model.reset()
        step(model, make_obs())
        check(model._episode.thread_id is not None, "a fresh thread was established")
        check(model._episode.calls_used == 1, "the budget restarted")


def test_episode_idx_change_rotates_the_episode() -> None:
    print("a new episode_idx starts a new episode without a reset()")
    with MockBridge("legal") as bridge:
        bridge.wait_until_responsive()
        model = make_model({"bridge_url": bridge.url})

        step(model, make_obs(episode_idx=0))
        step(model, make_obs(episode_idx=0))
        check(model._episode.calls_used == 2, "two calls in the first episode")
        first_id = model._episode.episode_id
        check("_idx0" in first_id, f"the episode id carries the index: {first_id}")

        _, text = step(model, make_obs(episode_idx=1))
        check(model._episode.episode_id != first_id, "the episode rotated")
        check("_idx1" in model._episode.episode_id, f"the new id: {model._episode.episode_id}")
        check(model._episode.calls_used == 1, "the new episode has a fresh budget")
        check(model._episode.turn_index == 1, "the turn counter restarted")
        check("episode_rotated" in text, "the rotation was logged for the audit trail")


def test_on_trial_end_summarises_the_episode() -> None:
    print("on_trial_end() summarises and flushes the episode")
    with tempfile.TemporaryDirectory() as workdir:
        with MockBridge("legal") as bridge:
            bridge.wait_until_responsive()
            model = make_model({"bridge_url": bridge.url, "output_dir": workdir})
            obs = make_obs()
            for _ in range(2):
                step(model, obs)

            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                model.on_trial_end(result="success")
            text = buffer.getvalue()
            check("[codex_agent][episode]" in text, "the episode summary was logged")
            summary = json.loads(text.split("[codex_agent][episode] ", 1)[1].splitlines()[0])
            check(summary["calls_used"] == 2, "the summary counts the calls")
            check(summary["decisions"] == 2, "the summary counts the decisions")
            check(summary["result"] == "success", "the result is carried through")
            check(summary["thread_id"] is not None, "the thread id is recorded")

            # the audit artifacts
            decisions_path = Path(workdir) / "decisions.jsonl"
            check(decisions_path.is_file(), "decisions.jsonl was written")
            lines = [json.loads(line) for line in decisions_path.read_text().splitlines()]
            check(len(lines) == 2, f"one line per decision, got {len(lines)}")
            check(all(line["event"] == "decision" for line in lines), "each line is a decision")

            actions_path = Path(workdir) / "low_level_actions.npy"
            check(actions_path.is_file(), "low_level_actions.npy was written")
            actions = np.load(actions_path)
            check(actions.ndim == 2 and actions.shape[1] == 16, f"shape {actions.shape}")
            check(actions.dtype == np.float32, f"dtype {actions.dtype}")
            check(
                actions.shape[0] == model._episode.sim_steps_used,
                f"one row per simulator step: {actions.shape[0]} vs {model._episode.sim_steps_used}",
            )


def test_on_trial_end_without_an_episode_is_harmless() -> None:
    print("on_trial_end() before any decision is harmless")
    model = make_model()
    model.on_trial_end()
    check(True, "no exception")


# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #


def test_bridge_url_precedence() -> None:
    print("the bridge URL precedence")
    previous = os.environ.get("CODEX_BRIDGE_URL")
    os.environ["CODEX_BRIDGE_URL"] = "http://env-host:9999"
    try:
        model = make_model({"bridge_url": "http://flat-host:1111"})
        check(model.bridge.base_url == "http://env-host:9999", "the environment wins")
    finally:
        os.environ.pop("CODEX_BRIDGE_URL", None)
        if previous is not None:
            os.environ["CODEX_BRIDGE_URL"] = previous

    model = make_model({"bridge_url": "http://flat-host:1111"})
    check(model.bridge.base_url == "http://flat-host:1111", "the flat key beats the nested one")
    model = make_model()
    check(model.bridge.base_url == "http://localhost:8765", "the shipped default is loopback")


def test_config_is_validated() -> None:
    print("the config is validated at construction")
    try:
        make_model({"action_type": "joint"})
    except ValueError as exc:
        check("action_type" in str(exc), "a non-ee action type is refused")
    else:
        check(False, "a non-ee action type must be refused")

    try:
        make_model({"episode": {"max_codex_calls": 0}})
    except ValueError as exc:
        check("max_codex_calls" in str(exc), "a zero call budget is refused")
    else:
        check(False, "a zero call budget must be refused")

    model = make_model({"motion": {"min_chunk_steps": 50, "max_chunk_steps": 10}})
    check(
        model.max_chunk_steps >= model.min_chunk_steps,
        "max_chunk_steps is repaired to be at least min_chunk_steps",
    )


def test_budget_bookkeeping_is_internally_consistent() -> None:
    print("the chunk cap spreads the remaining budget")
    with MockBridge("legal") as bridge:
        bridge.wait_until_responsive()
        model = make_model({"bridge_url": bridge.url})
        obs = make_obs()

        caps = []
        for _ in range(model.max_codex_calls):
            chunk, text = step(model, obs)
            record = decisions_in(text)[-1]
            caps.append(record["cap"])
            check(
                len(chunk) <= record["cap"],
                f"the chunk fits its cap: {len(chunk)} <= {record['cap']}",
            )
            check(
                record["remaining_calls"] + record["calls_used"] == model.max_codex_calls,
                "the remaining/used calls add up",
            )
            check(
                record["remaining_steps"] + record["sim_steps_used"] <= model.step_budget,
                "the remaining/used steps add up",
            )
        check(len(caps) == model.max_codex_calls, "every call was made")
        check(
            all(cap >= model.min_chunk_steps for cap in caps),
            f"the cap never drops below the minimum: {caps}",
        )


def test_feedback_describes_the_executed_motion() -> None:
    print("the feedback describes the motion that was actually executed")
    with MockBridge("legal") as bridge:
        bridge.wait_until_responsive()
        model = make_model({"bridge_url": bridge.url})
        step(model, make_obs())
        prompt = bridge.last_payload()["prompt"]
        check(
            "this is the first decision of the episode" in prompt,
            "the first turn says so instead of inventing feedback",
        )
        # the feedback arrives on the turn AFTER the motion it describes
        step(model, make_obs())
        prompt = bridge.last_payload()["prompt"]
        check("You commanded the left arm" in prompt, "the command is echoed back")
        check("You left the right arm unchanged" in prompt, "the untouched arm is named")
        check(
            "measured after the motion completed" in prompt,
            "the model is told the state is post-motion",
        )
        check("-0.000" not in prompt, "no negative zero leaked into the prompt")


def main() -> int:
    # The adapter reads these from the environment first, which would silently
    # override every test's bridge_url.
    saved = {key: os.environ.pop(key, None) for key in ("CODEX_BRIDGE_URL", "CODEX_BRIDGE_TOKEN")}

    for test in (
        test_healthy_episode,
        test_the_brief_rides_on_the_thread_creating_turn_only,
        test_every_turn_names_the_views_it_attaches,
        test_call_cap_is_enforced,
        test_exhausted_budget_drives_home,
        test_failed_calls_still_cost_budget,
        test_every_failure_mode_still_yields_a_chunk,
        test_all_keep_is_reported_as_a_noop,
        test_a_noop_caused_by_bad_fields_says_so,
        test_out_of_workspace_targets_are_rejected,
        test_degenerate_orientation_falls_back_to_keep,
        test_hung_bridge_returns_within_the_wall_budget,
        test_unreachable_bridge_degrades_after_two_failures,
        test_images_are_encoded_and_named,
        test_a_missing_camera_is_survivable,
        test_no_cameras_at_all_is_survivable,
        test_unusable_observation_falls_back_to_the_last_pose,
        test_reset_twice_is_harmless,
        test_episode_idx_change_rotates_the_episode,
        test_on_trial_end_summarises_the_episode,
        test_on_trial_end_without_an_episode_is_harmless,
        test_bridge_url_precedence,
        test_config_is_validated,
        test_budget_bookkeeping_is_internally_consistent,
        test_feedback_describes_the_executed_motion,
    ):
        test()
        sys.stdout.flush()

    for key, value in saved.items():
        if value is not None:
            os.environ[key] = value

    print()
    if FAILURES:
        print(f"FAILED {len(FAILURES)} of {CHECKS} checks:")
        seen = set()
        for label in FAILURES:
            if label in seen:
                continue
            seen.add(label)
            print(f"  - {label}")
        return 1
    print(f"ok: {CHECKS} checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
