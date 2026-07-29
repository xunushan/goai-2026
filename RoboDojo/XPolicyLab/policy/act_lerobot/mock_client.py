#!/usr/bin/env python3
"""Replay real LeRobot dataset frames through the WebSocket policy server.

Artifacts are written for later curve plotting:

* ``requests.jsonl``: one compact record per model request;
* ``curves.csv``: predicted/expert/error value for every horizon and dimension;
* ``summary.json``: aggregate MAE and latency;
* ``images/``: the three input frames for every sampled request.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import time
from pathlib import Path
from typing import Any

import av
import cv2
import numpy as np
import pyarrow.parquet as pq

from client_server.ws.protocol.client import PolicyEvalClient, PolicyEvalClientConfig


ACTION_NAMES = (
    "l_x", "l_y", "l_z", "l_w", "l_wx", "l_wy", "l_wz", "l_g",
    "r_x", "r_y", "r_z", "r_w", "r_wx", "r_wy", "r_wz", "r_g",
)
CAMERAS = {
    "cam_head": "observation.images.cam_high",
    "cam_left_wrist": "observation.images.cam_left_wrist",
    "cam_right_wrist": "observation.images.cam_right_wrist",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="ws://127.0.0.1:19999")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--stride", type=int, default=25)
    parser.add_argument("--max-samples", type=int, default=5)
    parser.add_argument("--action-steps", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/act_lerobot_mock"))
    return parser.parse_args()


def load_episode_metadata(dataset: Path, episode: int) -> dict[str, Any]:
    paths = sorted((dataset / "meta" / "episodes").rglob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No episode metadata parquet under {dataset}")
    table = pq.read_table(paths, filters=[("episode_index", "=", episode)])
    rows = table.to_pylist()
    if len(rows) != 1:
        raise ValueError(f"Expected one metadata row for episode {episode}, got {len(rows)}")
    return rows[0]


def load_episode_data(dataset: Path, metadata: dict[str, Any]) -> list[dict[str, Any]]:
    chunk = int(metadata["data/chunk_index"])
    file_index = int(metadata["data/file_index"])
    path = dataset / "data" / f"chunk-{chunk:03d}" / f"file-{file_index:03d}.parquet"
    table = pq.read_table(
        path,
        columns=["observation.state", "action", "timestamp", "frame_index", "episode_index", "index"],
        filters=[("episode_index", "=", int(metadata["episode_index"]))],
    )
    rows = table.to_pylist()
    if len(rows) != int(metadata["length"]):
        raise ValueError(f"Episode length mismatch: meta={metadata['length']}, data={len(rows)}")
    return rows


def decode_rgb_at(video_path: Path, timestamp: float) -> np.ndarray:
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        target = max(0.0, float(timestamp))
        container.seek(
            int(target / float(stream.time_base)),
            stream=stream,
            backward=True,
            any_frame=False,
        )
        selected = None
        for frame in container.decode(stream):
            selected = frame
            frame_time = float(frame.pts * stream.time_base) if frame.pts is not None else target
            if frame_time + 1e-6 >= target:
                break
        if selected is None:
            raise RuntimeError(f"Could not decode {video_path} at {timestamp:.6f}s")
        return selected.to_ndarray(format="rgb24")


def load_images(
    dataset: Path,
    metadata: dict[str, Any],
    local_frame_index: int,
    fps: float,
) -> dict[str, np.ndarray]:
    images = {}
    for client_name, dataset_key in CAMERAS.items():
        prefix = f"videos/{dataset_key}"
        chunk = int(metadata[f"{prefix}/chunk_index"])
        file_index = int(metadata[f"{prefix}/file_index"])
        start = float(metadata[f"{prefix}/from_timestamp"])
        timestamp = start + local_frame_index / fps
        path = (
            dataset / "videos" / dataset_key
            / f"chunk-{chunk:03d}" / f"file-{file_index:03d}.mp4"
        )
        images[client_name] = decode_rgb_at(path, timestamp)
    return images


def state_to_observation(
    state16: np.ndarray,
    images: dict[str, np.ndarray],
    instruction: str,
) -> dict[str, Any]:
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
        "additional_info": {"frequency": 25},
        "data_format_version": "v1.0",
        "env_idx": 0,
    }


def flatten_actions(actions: list[dict[str, Any]]) -> np.ndarray:
    rows = []
    for action in actions:
        rows.append(np.concatenate([
            np.asarray(action["left_ee_pose"], dtype=np.float32),
            np.asarray(action["left_ee_joint_state"], dtype=np.float32),
            np.asarray(action["right_ee_pose"], dtype=np.float32),
            np.asarray(action["right_ee_joint_state"], dtype=np.float32),
        ]))
    result = np.stack(rows)
    if result.ndim != 2 or result.shape[1] != 16 or not np.isfinite(result).all():
        raise ValueError(f"Invalid server action chunk: {result.shape}")
    return result


def image_summary(images: dict[str, np.ndarray]) -> dict[str, Any]:
    return {
        name: {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "min": int(value.min()),
            "max": int(value.max()),
            "mean": float(value.mean()),
        }
        for name, value in images.items()
    }


async def run(args: argparse.Namespace) -> None:
    dataset = args.dataset.resolve()
    output = args.output_dir.resolve()
    image_dir = output / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    info = json.loads((dataset / "meta" / "info.json").read_text(encoding="utf-8"))
    fps = float(info["fps"])
    metadata = load_episode_metadata(dataset, args.episode)
    rows = load_episode_data(dataset, metadata)
    instruction = str((metadata.get("tasks") or [""])[0])

    sample_indices = list(range(0, len(rows), args.stride))
    if args.max_samples > 0:
        sample_indices = sample_indices[: args.max_samples]

    client = PolicyEvalClient(
        PolicyEvalClientConfig(url=args.url, evaluation_id="act-lerobot-dataset-mock")
    )
    await client.connect(handshake=True)
    await client.reset(trial_id=f"episode-{args.episode}")

    request_records = []
    curve_rows = []
    all_errors = []
    try:
        for request_index, local_index in enumerate(sample_indices):
            row = rows[local_index]
            state = np.asarray(row["observation.state"], dtype=np.float32)
            images = load_images(dataset, metadata, local_index, fps)
            observation = state_to_observation(state, images, instruction)

            start = time.perf_counter()
            response = await client.infer(
                observation,
                trial_id=f"episode-{args.episode}",
                action_case_id=instruction,
                step=request_index,
            )
            round_trip_ms = (time.perf_counter() - start) * 1000.0
            predicted = flatten_actions(response.payload["actions"])

            valid_steps = min(
                args.action_steps,
                len(predicted),
                len(rows) - local_index,
            )
            expert = np.asarray(
                [rows[local_index + offset]["action"] for offset in range(valid_steps)],
                dtype=np.float32,
            )
            predicted = predicted[:valid_steps]
            error = np.abs(predicted - expert)
            all_errors.append(error)

            for camera_name, image in images.items():
                path = image_dir / f"request_{request_index:04d}_{camera_name}.jpg"
                cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

            record = {
                "request_index": request_index,
                "episode_index": args.episode,
                "local_frame_index": local_index,
                "frame_index": int(row["frame_index"]),
                "dataset_index": int(row["index"]),
                "instruction": instruction,
                "input_state": state.tolist(),
                "image_summary": image_summary(images),
                "predicted_actions": predicted.tolist(),
                "expert_actions": expert.tolist(),
                "mae": float(error.mean()),
                "server_latency_ms": float(response.payload.get("latency_ms", 0.0)),
                "round_trip_ms": round_trip_ms,
            }
            request_records.append(record)
            print(
                f"request={request_index} frame={row['frame_index']} "
                f"steps={valid_steps} mae={record['mae']:.6f} "
                f"server={record['server_latency_ms']:.1f}ms "
                f"roundtrip={round_trip_ms:.1f}ms",
                flush=True,
            )

            for horizon in range(valid_steps):
                for dimension, name in enumerate(ACTION_NAMES):
                    curve_rows.append({
                        "request_index": request_index,
                        "frame_index": int(row["frame_index"]),
                        "horizon": horizon,
                        "dimension": dimension,
                        "name": name,
                        "state": float(state[dimension]),
                        "expert": float(expert[horizon, dimension]),
                        "predicted": float(predicted[horizon, dimension]),
                        "abs_error": float(error[horizon, dimension]),
                    })
    finally:
        await client.close()

    with (output / "requests.jsonl").open("w", encoding="utf-8") as file:
        for record in request_records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    with (output / "curves.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(curve_rows[0]))
        writer.writeheader()
        writer.writerows(curve_rows)

    errors = np.concatenate(all_errors, axis=0)
    predicted_all = np.concatenate([
        np.asarray(record["predicted_actions"], dtype=np.float32)
        for record in request_records
    ], axis=0)
    left_quaternion_norm = np.linalg.norm(predicted_all[:, 3:7], axis=1)
    right_quaternion_norm = np.linalg.norm(predicted_all[:, 11:15], axis=1)
    grippers = predicted_all[:, [7, 15]]
    gripper_out_of_range = int(np.count_nonzero((grippers < 0.0) | (grippers > 1.0)))
    summary = {
        "url": args.url,
        "dataset": str(dataset),
        "episode": args.episode,
        "instruction": instruction,
        "request_count": len(request_records),
        "action_steps": args.action_steps,
        "mae": float(errors.mean()),
        "per_dimension_mae": {
            name: float(value)
            for name, value in zip(ACTION_NAMES, errors.mean(axis=0), strict=True)
        },
        "output_diagnostics": {
            "all_finite": bool(np.isfinite(predicted_all).all()),
            "left_quaternion_norm_min": float(left_quaternion_norm.min()),
            "left_quaternion_norm_max": float(left_quaternion_norm.max()),
            "right_quaternion_norm_min": float(right_quaternion_norm.min()),
            "right_quaternion_norm_max": float(right_quaternion_norm.max()),
            "gripper_min": float(grippers.min()),
            "gripper_max": float(grippers.max()),
            "gripper_out_of_range_values": gripper_out_of_range,
        },
        "mean_server_latency_ms": float(np.mean([
            record["server_latency_ms"] for record in request_records
        ])),
        "mean_round_trip_ms": float(np.mean([
            record["round_trip_ms"] for record in request_records
        ])),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if gripper_out_of_range:
        print(
            f"WARNING: {gripper_out_of_range} predicted gripper values are outside [0,1]; "
            "RoboDojo clips them during execution.",
            flush=True,
        )
    print(f"Saved mock artifacts: {output}")


def main() -> None:
    args = parse_args()
    if args.stride <= 0 or args.action_steps <= 0:
        raise ValueError("stride and action-steps must be positive")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
