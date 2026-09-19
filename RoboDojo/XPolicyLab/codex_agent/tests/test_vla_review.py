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
from codex_agent.bridge.schema import Observation, validate_response  # noqa: E402


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
        "observation": {"left": arm, "right": arm}, "feedback": [],
        "images": [{"name": "cam_head", "mime": "image/jpeg", "b64": base64.b64encode(b"\xff\xd8\xffx").decode()}],
        "vla_review": {
            "chunk": {"horizon": 2, "left": chunk_arm, "right": chunk_arm},
            "summary": {},
            "gripper_change_threshold": 0.1,
        },
    }


def main() -> int:
    observation = Observation.parse(packet())
    assert _has_gripper_change(observation.vla_review)
    quiet_packet = packet()
    quiet_packet["vla_review"]["chunk"]["left"]["gripper"] = [1.0, 0.95]
    quiet_packet["vla_review"]["chunk"]["right"]["gripper"] = [1.0, 0.95]
    assert not _has_gripper_change(Observation.parse(quiet_packet).vla_review)
    vla = validate_response({"mode": "vla", "vla_steps": 1, "verify_next": True, "note": "ok", "phase": "grasp"}, vla_horizon=2)
    motion = MotionConfig(settle_steps=0)
    chunk, continuation = _synthesise(observation, vla, motion)
    assert len(chunk) == 1 and continuation["verify_previous"] is True
    eef = validate_response({"mode": "eef", "left": {"position": "keep", "orientation": "keep", "gripper": "keep"}, "right": {"position": [0.01, 0, 0], "orientation": "keep", "gripper": "keep"}, "note": "align", "phase": "align"}, vla_horizon=2)
    chunk, continuation = _synthesise(observation, eef, motion)
    assert len(chunk) == 2 and continuation["verify_previous"] is False
    print("vla-review tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
