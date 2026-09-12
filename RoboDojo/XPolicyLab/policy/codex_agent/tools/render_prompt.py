#!/usr/bin/env python3
"""Render the prompts ``codex_agent`` would send, without touching Codex.

The system prompt is built once per episode; turn prompts once per decision.
Both are pure functions of the deploy config, the task card and the observed
poses, so they can be inspected offline -- which is the point: the operator's
Codex quota is reserved for real runs.

Usage::

    python tools/render_prompt.py                     # system + one turn
    python tools/render_prompt.py --turn 3            # a later turn
    python tools/render_prompt.py --schema out.json   # also dump the JSON schema
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

_HERE = Path(__file__).resolve().parent
_PKG = _HERE.parent
if str(_PKG.parent) not in sys.path:
    sys.path.insert(0, str(_PKG.parent))

from codex_agent.model import Model  # noqa: E402
from codex_agent.motion import ArmState  # noqa: E402
from codex_agent.prompt import (  # noqa: E402
    TurnFeedback,
    build_output_schema,
    build_system_prompt,
    build_turn_prompt,
)

DEFAULT_CONFIG = _PKG / "deploy.yml"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument(
        "--turn",
        type=int,
        default=0,
        help="0 renders the system prompt plus the first turn; N>0 renders turn N+1",
    )
    parser.add_argument(
        "--drift",
        type=float,
        default=0.0,
        help="offset the reported pose by this many metres, to fake a moved arm",
    )
    parser.add_argument("--schema", default=None, help="write the output schema here")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    model = Model(cfg)
    ctx = model._ensure_prompt_context()

    # A plausible mid-episode observation: start from HOME and nudge the left
    # arm forward/down, as if it were reaching over the table.
    left = ArmState.from_pose7(model.home_left, model.motion.gripper_open)
    right = ArmState.from_pose7(model.home_right, model.motion.gripper_open)
    left = ArmState(
        pos=np.asarray(left.pos) + np.array([0.0, 0.15 + args.drift, -0.05]),
        quat=np.asarray(left.quat),
        gripper=model.motion.gripper_close,
    )

    print("=" * 78)
    print("SYSTEM PROMPT")
    print("=" * 78)
    print(build_system_prompt(ctx))

    if args.turn > 0:
        print()
        print("=" * 78)
        print(f"TURN PROMPT (turn_index={args.turn})")
        print("=" * 78)
        print(
            build_turn_prompt(
                ctx,
                turn_index=args.turn,
                decisions_used=args.turn,
                steps_used=args.turn * 30,
                left=left,
                right=right,
                feedback=TurnFeedback(
                    [
                        "your previous decision asked left to move to [x, y, z] and right to keep",
                        "left reached [-0.30, -0.20, 0.87] (remaining error 0.021 m, 0.03 rad)",
                        "left gripper closed to 0.36",
                        "no component was clamped and no target was rejected",
                    ]
                ),
            )
        )
        print()
        print("OUTPUT SCHEMA")
        print(json.dumps(build_output_schema(), indent=2))

    if args.schema:
        Path(args.schema).write_text(
            json.dumps(build_output_schema(), indent=2), encoding="utf-8"
        )
        print(f"\nwrote {args.schema}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
