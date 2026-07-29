#!/usr/bin/env python3
"""Validate an xvla_robtwin WebSocket server before Isaac Sim evaluation."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from client_server.ws.protocol.client import PolicyEvalClient, PolicyEvalClientConfig


ACTION_NAMES = (
    "l_x", "l_y", "l_z", "l_w", "l_wx", "l_wy", "l_wz", "l_g",
    "r_x", "r_y", "r_z", "r_w", "r_wx", "r_wy", "r_wz", "r_g",
)
DEFAULT_STATE16 = np.asarray(
    [
        -0.29952908, -0.35229987, 0.92150003,
        0.70699996, 0.0, 0.0, 0.70721352, 1.0,
        0.30047095, -0.35229987, 0.92150003,
        0.70699996, 0.0, 0.0, 0.70721352, 1.0,
    ],
    dtype=np.float32,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="ws://127.0.0.1:6000")
    parser.add_argument(
        "--instruction",
        default="Stack the three blocks with different textures.",
    )
    parser.add_argument("--requests", type=int, default=2)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/xvla_robtwin_mock.json"),
    )
    return parser.parse_args()


def synthetic_rgb(height: int, width: int, phase: int) -> np.ndarray:
    y, x = np.indices((height, width), dtype=np.uint16)
    return np.stack(
        [
            (x + phase * 17) % 256,
            (y + phase * 29) % 256,
            ((x // 2 + y // 2) + phase * 11) % 256,
        ],
        axis=-1,
    ).astype(np.uint8)


def make_observation(
    state16: np.ndarray,
    instruction: str,
    *,
    height: int,
    width: int,
    phase: int,
) -> dict[str, Any]:
    images = {
        "cam_head": synthetic_rgb(height, width, phase),
        "cam_left_wrist": synthetic_rgb(height, width, phase + 1),
        "cam_right_wrist": synthetic_rgb(height, width, phase + 2),
    }
    return {
        "vision": {
            name: {"color": image, "shape": image.shape}
            for name, image in images.items()
        },
        "state": {
            "left_ee_pose": state16[0:7],
            "left_ee_joint_state": state16[7:8],
            "right_ee_pose": state16[8:15],
            "right_ee_joint_state": state16[15:16],
        },
        "action": {},
        "instruction": instruction,
        "additional_info": {"frequency": 30},
        "data_format_version": "v1.0",
        "env_idx": 0,
    }


def flatten_actions(actions: Any) -> np.ndarray:
    if not isinstance(actions, list) or not actions:
        raise ValueError(f"Server must return a non-empty action list, got {type(actions)}")
    rows = []
    for index, action in enumerate(actions):
        required = {
            "left_ee_pose",
            "left_ee_joint_state",
            "right_ee_pose",
            "right_ee_joint_state",
        }
        if not isinstance(action, dict) or not required.issubset(action):
            raise ValueError(f"Action {index} has invalid fields: {action}")
        rows.append(
            np.concatenate(
                [
                    np.asarray(action["left_ee_pose"], dtype=np.float32),
                    np.asarray(action["left_ee_joint_state"], dtype=np.float32),
                    np.asarray(action["right_ee_pose"], dtype=np.float32),
                    np.asarray(action["right_ee_joint_state"], dtype=np.float32),
                ]
            )
        )
    chunk = np.stack(rows)
    if chunk.ndim != 2 or chunk.shape[1] != 16:
        raise ValueError(f"Expected action chunk [T,16], got {chunk.shape}")
    if not np.isfinite(chunk).all():
        raise ValueError("Server returned NaN or infinity.")
    return chunk


def validate_chunk(chunk: np.ndarray) -> dict[str, Any]:
    left_norm = np.linalg.norm(chunk[:, 3:7], axis=1)
    right_norm = np.linalg.norm(chunk[:, 11:15], axis=1)
    grippers = chunk[:, [7, 15]]
    if not np.allclose(left_norm, 1.0, atol=1e-4):
        raise ValueError(f"Invalid left quaternion norms: {left_norm}")
    if not np.allclose(right_norm, 1.0, atol=1e-4):
        raise ValueError(f"Invalid right quaternion norms: {right_norm}")
    if not np.isin(grippers, [0.0, 1.0]).all():
        raise ValueError(f"Grippers must be binary 1=open/0=closed: {grippers}")
    return {
        "shape": list(chunk.shape),
        "all_finite": True,
        "left_quaternion_norm": left_norm.tolist(),
        "right_quaternion_norm": right_norm.tolist(),
        "grippers": grippers.tolist(),
        "min": chunk.min(axis=0).tolist(),
        "max": chunk.max(axis=0).tolist(),
    }


async def run(args: argparse.Namespace) -> None:
    if args.requests <= 0 or args.height <= 0 or args.width <= 0:
        raise ValueError("requests, height and width must be positive")

    client = PolicyEvalClient(
        PolicyEvalClientConfig(
            url=args.url,
            evaluation_id="xvla-robtwin-mock",
        )
    )
    await client.connect(handshake=True)
    await client.reset(trial_id="xvla-robtwin-mock")

    records = []
    state = DEFAULT_STATE16.copy()
    try:
        for request_index in range(args.requests):
            observation = make_observation(
                state,
                args.instruction,
                height=args.height,
                width=args.width,
                phase=request_index,
            )
            start = time.perf_counter()
            response = await client.infer(
                observation,
                trial_id="xvla-robtwin-mock",
                action_case_id=args.instruction,
                step=request_index,
            )
            round_trip_ms = (time.perf_counter() - start) * 1000.0
            chunk = flatten_actions(response.payload.get("actions"))
            diagnostics = validate_chunk(chunk)
            records.append(
                {
                    "request": request_index,
                    "instruction": args.instruction,
                    "input_state16": state.tolist(),
                    "action_names": list(ACTION_NAMES),
                    "actions": chunk.tolist(),
                    "diagnostics": diagnostics,
                    "server_latency_ms": float(
                        response.payload.get("latency_ms", 0.0)
                    ),
                    "round_trip_ms": round_trip_ms,
                }
            )
            print(
                f"request={request_index} shape={chunk.shape} "
                f"grippers={diagnostics['grippers']} "
                f"server={records[-1]['server_latency_ms']:.1f}ms "
                f"roundtrip={round_trip_ms:.1f}ms",
                flush=True,
            )
            # Emulate absolute-EE execution with the first predicted action.
            state = chunk[0].copy()
    finally:
        await client.close()

    result = {
        "url": args.url,
        "request_count": len(records),
        "status": "passed",
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"PASSED: saved {args.output.resolve()}")


def main() -> None:
    asyncio.run(run(parse_args()))


if __name__ == "__main__":
    main()
