#!/usr/bin/env python3
"""VLA-review wire schema, routing and action synthesis checks."""

from __future__ import annotations

import base64
import sys
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE.parent))

from codex_agent.bridge.bridge import _has_gripper_change, _synthesise  # noqa: E402
from codex_agent.bridge.motion import MotionConfig  # noqa: E402
from codex_agent.bridge.schema import Observation, output_schema, validate_response  # noqa: E402


def packet() -> dict:
    arm = {"position": [0, 0, 0], "orientation": [1, 0, 0, 0], "gripper": 1.0}
    chunk_arm = {
        "position": [[0, 0, 0], [0.01, 0, 0]],
        "orientation": [[1, 0, 0, 0], [1, 0, 0, 0]],
        "gripper": [1.0, 0.0],
    }
    return {
        "episode_id": "ep", "request_id": "req", "step_id": 0, "turn_index": 0,
        "task": {"name": "task", "instruction": "do task", "guidance": []},
        "budget": {"max_decisions": 10, "max_sim_steps": 20, "remaining_decisions": 9, "remaining_steps": 20},
        "use_experience": True,
        "observation": {"left": arm, "right": arm}, "feedback": [],
        "images": [{"name": "cam_head", "mime": "image/jpeg", "b64": base64.b64encode(b"\xff\xd8\xffx").decode()}],
        "vla_review": {
            "chunk": {"horizon": 2, "left": chunk_arm, "right": chunk_arm},
            "summary": {},
            "gripper_change_threshold": 0.1,
        },
    }


def main() -> int:
    schema = output_schema(vla_review=True)
    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert schema["required"] == ["decision"]
    assert "anyOf" not in schema
    assert len(schema["properties"]["decision"]["anyOf"]) == 2
    observation = Observation.parse(packet())
    assert _has_gripper_change(observation.vla_review)
    quiet_packet = packet()
    quiet_packet["vla_review"]["chunk"]["left"]["gripper"] = [1.0, 0.95]
    quiet_packet["vla_review"]["chunk"]["right"]["gripper"] = [1.0, 0.95]
    assert not _has_gripper_change(Observation.parse(quiet_packet).vla_review)
    vla = validate_response({"decision": {"mode": "vla", "vla_steps": 2, "verify_next": True, "note": "ok", "phase": "grasp"}}, vla_horizon=2)
    motion = MotionConfig(settle_steps=0)
    chunk, continuation = _synthesise(observation, vla, motion)
    assert len(chunk) == 2 and continuation["verify_previous"] is True
    before_event = validate_response({"decision": {"mode": "vla", "vla_steps": 1, "verify_next": True, "note": "approach only", "phase": "approach"}}, vla_horizon=2)
    chunk, continuation = _synthesise(observation, before_event, motion)
    assert len(chunk) == 1 and continuation["verify_previous"] is False
    no_verify = validate_response({"decision": {"mode": "vla", "vla_steps": 2, "verify_next": False, "note": "no follow-up", "phase": "grasp"}}, vla_horizon=2)
    _, continuation = _synthesise(observation, no_verify, motion)
    assert continuation["verify_previous"] is False
    eef = validate_response({"decision": {"mode": "eef", "left": {"position": "keep", "orientation": "keep", "gripper": "keep"}, "right": {"position": [0.01, 0, 0], "orientation": "keep", "gripper": "keep"}, "note": "align", "phase": "align"}}, vla_horizon=2)
    chunk, continuation = _synthesise(observation, eef, motion)
    assert len(chunk) == 2 and continuation["verify_previous"] is False
    keep = validate_response({"decision": {"mode": "eef", "left": {"position": "keep", "orientation": "keep", "gripper": "keep"}, "right": {"position": "keep", "orientation": "keep", "gripper": "keep"}, "note": "hold", "phase": "wait"}}, vla_horizon=2)
    chunk, _ = _synthesise(observation, keep, MotionConfig(settle_steps=3))
    assert len(chunk) == 1
    print("vla-review tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
