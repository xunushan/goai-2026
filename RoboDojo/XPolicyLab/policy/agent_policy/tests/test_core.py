#!/usr/bin/env python3
"""Main policy path and its four failure boundaries."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE.parents[2]))

from XPolicyLab.codex_agent.bridge.schema import PolicyValidationError, validate_response
from XPolicyLab.policy.agent_policy.bridge_client import BridgeResult
from XPolicyLab.policy.agent_policy.model import Model
from XPolicyLab.codex_agent.bridge.protocol import ParseError, parse_decision
from XPolicyLab.utils.episode_index import EpisodeIndexResolver

CAMERAS = ("cam_head", "cam_left_wrist", "cam_right_wrist")


def arm(position):
    return {"position": position, "orientation": [1, 0, 0, 0], "gripper": "keep"}


def decision(position=0.01):
    return {"left": arm([position, 0, 0]), "right": arm([0, 0, 0]), "note": "test", "phase": "move"}


def action(position=0.01):
    return {
        "left_ee_pose": [position, 0, 0, 1, 0, 0, 0], "left_ee_joint_state": [1],
        "right_ee_pose": [0, 0, 0, 1, 0, 0, 0], "right_ee_joint_state": [1],
    }


def observation(with_images=True):
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    return {
        "state": {
            "left_ee_pose": [0, 0, 0, 1, 0, 0, 0],
            "left_ee_joint_state": [1],
            "right_ee_pose": [0, 0, 0, 1, 0, 0, 0],
            "right_ee_joint_state": [1],
        },
        "vision": {name: image for name in CAMERAS} if with_images else {},
    }


def model(result):
    instance = Model({"task_name": "stack_blocks", "bridge_url": "http://unused"})
    instance.bridge.decide = lambda packet, timeout_s: result
    instance.update_obs(observation())
    return instance


def main() -> int:
    episode_index = EpisodeIndexResolver()
    assert episode_index.resolve(17) == "17"
    episode_index.reset()
    generated = episode_index.resolve(None)
    assert generated.startswith("ep001_")
    assert episode_index.resolve(None) == generated

    validate_response(decision())
    validate_response({**decision(), "note": "x" * 31})
    try:
        validate_response({**decision(), "phase": "x" * 11})
    except PolicyValidationError:
        pass
    else:
        raise AssertionError("schema accepted an overly long phase")
    gripper_while_moving = decision()
    gripper_while_moving["left"]["gripper"] = "close"
    validate_response(gripper_while_moving)
    try:
        validate_response({**decision(), "left": {**arm([0, 0, 0]), "orientation": [0, 0, 0]}})
    except PolicyValidationError:
        pass
    else:
        raise AssertionError("schema accepted a three-number orientation")

    try:
        parse_decision({"left": {"position": {"xyz": [0, 0, 0]}}, "right": arm([0, 0, 0])})
    except ParseError:
        pass
    else:
        raise AssertionError("parser silently accepted a compatibility shape")

    instance = model(BridgeResult(ok=True, decision=decision(), action_chunk=[action(0.005), action()]))
    chunk = instance.get_action()
    assert len(chunk) == 2
    assert np.allclose(chunk[-1]["left_ee_pose"][:3], [0.01, 0, 0])
    assert instance.episode.calls_used == 1 and instance.episode.steps_used == 2

    for result in (
        BridgeResult(ok=False, error_kind="timeout"),
        BridgeResult(ok=True, decision={"bad": True}, action_chunk=[{"bad": []}]),
    ):
        instance = model(result)
        chunk = instance.get_action()
        assert len(chunk) == 1
        assert np.allclose(chunk[0]["left_ee_pose"][:3], [0, 0, 0])
        assert instance.episode.steps_used == 1

    instance = model(BridgeResult(ok=True, decision=decision(), action_chunk=[action()]))
    instance.update_obs(observation(with_images=False))
    assert len(instance.get_action()) == 1

    instance = Model({"task_name": None, "bridge_url": "http://unused"})
    instance.bridge.decide = lambda packet, timeout_s: BridgeResult(
        ok=True, decision=decision(), action_chunk=[action()]
    )
    real_observation = observation()
    real_observation["instruction"] = "Stack the bowls on the table."
    instance.update_obs(real_observation)
    assert len(instance.get_action()) == 1
    assert instance.context is not None
    assert instance.context.task_name == "stack_bowls"
    assert instance.step_budget == 1220

    captured = {}
    instance = Model({"task_name": "stack_blocks", "bridge_url": "http://unused"})
    instance.bridge.decide = lambda packet, timeout_s: (
        captured.update(packet)
        or BridgeResult(ok=True, decision=decision(), action_chunk=[action()])
    )
    sim_observation = observation()
    sim_observation["episode_idx"] = 23
    instance.update_obs(sim_observation)
    instance.get_action()
    assert captured["episode_id"] == "23"
    print("core policy tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
