"""Offline contract test for the X-VLA → Codex Bridge adapter."""

from __future__ import annotations

import numpy as np

from bridge_client import BridgeResult
from codex_review import CodexReviewer


def action(x: float, grip: float) -> dict:
    return {
        "left_ee_pose": np.asarray([x, 0, 0, 1, 0, 0, 0], dtype=np.float32),
        "left_ee_joint_state": np.asarray([grip], dtype=np.float32),
        "right_ee_pose": np.asarray([0, 0, 0, 1, 0, 0, 0], dtype=np.float32),
        "right_ee_joint_state": np.asarray([1], dtype=np.float32),
    }


def test_review_packet_and_response() -> None:
    reviewer = CodexReviewer({"task_name": "stack_blocks"})
    captured = {}

    def decide(packet, timeout_s):
        captured.update(packet)
        return BridgeResult(ok=True, action_chunk=[action(0.01, 1)], continuation={"previous_request_id": packet["request_id"], "verify_previous": True})

    reviewer.bridge.decide = decide
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    observation = {
        "episode_idx": 12,
        "state": {"left_ee_pose": [0, 0, 0, 1, 0, 0, 0], "left_ee_joint_state": [1],
                  "right_ee_pose": [0, 0, 0, 1, 0, 0, 0], "right_ee_joint_state": [1]},
        "vision": {name: image for name in ("cam_head", "cam_left_wrist", "cam_right_wrist")},
    }
    result = reviewer.review(observation, [action(0, 1), action(0.01, 0)])
    assert len(result) == 1
    assert captured["vla_review"]["chunk"]["horizon"] == 2
    assert captured["vla_review"]["gripper_change_threshold"] == 0.1
    assert captured["budget"]["max_sim_steps"] == 550
    assert captured["episode_id"] == "12"
    assert captured["use_experience"] is True
    reviewer.review(observation, [action(0, 1)])
    assert captured["continuation"]["verify_previous"] is True


def test_real_instruction_resolves_task() -> None:
    reviewer = CodexReviewer({"task_name": None, "experience": {"enabled": False}})
    captured = {}
    reviewer.bridge.decide = lambda packet, timeout_s: (
        captured.update(packet)
        or BridgeResult(ok=True, action_chunk=[action(0.01, 1)])
    )
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    observation = {
        "instruction": "Stand the bottle upright.",
        "state": {"left_ee_pose": [0, 0, 0, 1, 0, 0, 0], "left_ee_joint_state": [1],
                  "right_ee_pose": [0, 0, 0, 1, 0, 0, 0], "right_ee_joint_state": [1]},
        "vision": {name: image for name in ("cam_head", "cam_left_wrist", "cam_right_wrist")},
    }
    reviewer.review(observation, [action(0, 1)])
    assert captured["task"]["name"] == "stand_up_bottles"
    assert captured["budget"]["max_sim_steps"] == 1190
    assert captured["episode_id"].startswith("ep001_")
    assert captured["use_experience"] is False


def test_bridge_failure_returns_one_measured_hold() -> None:
    reviewer = CodexReviewer({"task_name": "stack_blocks"})
    reviewer.bridge.decide = lambda packet, timeout_s: BridgeResult(
        ok=False,
        error_kind="policy_timeout",
        error="timed out",
    )
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    observation = {
        "episode_idx": 33,
        "state": {"left_ee_pose": [0.1, 0.2, 0.3, 1, 0, 0, 0], "left_ee_joint_state": [0.7],
                  "right_ee_pose": [-0.1, 0.4, 0.5, 1, 0, 0, 0], "right_ee_joint_state": [0.2]},
        "vision": {name: image for name in ("cam_head", "cam_left_wrist", "cam_right_wrist")},
    }
    held = reviewer.review(observation, [action(0, 1)])
    assert len(held) == 1
    assert np.allclose(held[0]["left_ee_pose"], observation["state"]["left_ee_pose"])
    assert np.allclose(held[0]["right_ee_pose"], observation["state"]["right_ee_pose"])
    assert np.allclose(held[0]["left_ee_joint_state"], [0.7])
    assert np.allclose(held[0]["right_ee_joint_state"], [0.2])
    assert reviewer.steps == 1 and reviewer.continuation is None


if __name__ == "__main__":
    test_review_packet_and_response()
    test_real_instruction_resolves_task()
    test_bridge_failure_returns_one_measured_hold()
    print("xvla-agent Codex review tests passed")
