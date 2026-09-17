#!/usr/bin/env python3
"""Main policy path and its four failure boundaries."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE.parent))

from codex_agent.bridge.schema import PolicyValidationError, validate_response
from codex_agent.bridge_client import BridgeResult
from codex_agent.model import Model
from codex_agent.protocol import ParseError, parse_decision

CAMERAS = ("cam_head", "cam_left_wrist", "cam_right_wrist")


def arm(position):
    return {"position": position, "orientation": [1, 0, 0, 0], "gripper": "keep"}


def decision(position=0.01):
    return {"left": arm([position, 0, 0]), "right": arm([0, 0, 0]), "note": "test", "phase": "move"}


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
    instance = Model({"task_name": "stack_blocks", "bridge_url": "http://unused", "motion": {"settle_steps": 0}})
    instance.bridge.decide = lambda packet, timeout_s: result
    instance.update_obs(observation())
    return instance


def main() -> int:
    validate_response(decision())
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

    instance = model(BridgeResult(ok=True, decision=decision()))
    chunk = instance.get_action()
    assert len(chunk) == 2
    assert np.allclose(chunk[-1]["left_ee_pose"][:3], [0.01, 0, 0])
    assert instance.episode.calls_used == 1 and instance.episode.steps_used == 2

    for result in (
        BridgeResult(ok=False, error_kind="timeout"),
        BridgeResult(ok=True, decision={"bad": True}),
        BridgeResult(ok=True, decision=decision(1.0)),
    ):
        instance = model(result)
        chunk = instance.get_action()
        assert len(chunk) == 1
        assert np.allclose(chunk[0]["left_ee_pose"][:3], [0, 0, 0])
        assert instance.episode.steps_used == 1

    instance = model(BridgeResult(ok=True, decision=decision()))
    instance.update_obs(observation(with_images=False))
    assert len(instance.get_action()) == 1
    print("core policy tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
