#!/usr/bin/env python3
"""Replay our 20d (arx_ee6d) training dataset through the xvla_2 WebSocket policy server.

Each 20d absolute state is converted to the 16d end-effector layout the simulator
presents (gripper inverted, `xvla20_to_ee16(..., invert_gripper=True)`), the three
camera frames are decoded, and one `infer` request is issued per sampled frame.
The server must return a full action chunk (default 30 x 16d) with finite values,
unit quaternions and gripper in [0,1].

For every sampled frame the prediction is also compared against the expert ground
truth. Ground truth is built exactly like training (LeRobotV3RoboDojoHandler.
iter_episode): the episode's 20d absolute state trajectory is interpolated on the
action grid (step = qdur/num_actions, decoupled from recording fps), and the chunk
for start index ``idx`` is the next ``num_actions`` absolute targets
``interp1d(linspace(lt[idx], lt[idx]+qdur, num_actions+1))[1:]``, converted to 16d
with ``xvla20_to_ee16(invert_gripper=True)`` so it shares the server output space
(gripper 1=open). Per-dimension MAE, group MAE (position/quaternion/gripper) and a
"stay still" (zero-motion) baseline MAE are reported in summary.json.

Artifacts written under --output-dir:
* ``requests.jsonl`` — one record per request (input state16/20, predicted chunk,
  ground truth chunk, per-step absolute error);
* ``summary.json``   — shape/finiteness/quaternion/gripper diagnostics + latency
  + prediction-vs-truth error;
* ``images/``        — the three input frames per sampled request.

Run from the policy dir with XPolicyLab on PYTHONPATH, e.g.:
  PYTHONPATH=$PWD/../../.. python mock_client.py \
      --url ws://127.0.0.1:PORT --dataset /data/data/lerobot_v30_ee_6d \
      --episode 0 --stride 25 --max-samples 5 --action-steps 30
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
from scipy.interpolate import interp1d

from client_server.ws.protocol.client import PolicyEvalClient, PolicyEvalClientConfig
from xvla_datasets.utils import ee16_to_xvla20, xvla20_to_ee16

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
    parser.add_argument("--action-steps", type=int, default=30)
    parser.add_argument("--qdur", type=float, default=1.0, help="训练动作窗口时长（秒），ground truth 网格用")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/xvla_2_mock"))
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
        columns=["observation.state", "timestamp", "frame_index", "episode_index", "index"],
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


def load_images(dataset, metadata, local_frame_index, fps) -> dict[str, np.ndarray]:
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


def state20_to_observation(
    state20: np.ndarray, images: dict[str, np.ndarray], instruction: str
) -> dict[str, Any]:
    """20d (arx_ee6d) 数据集状态 -> 仿真 16d 观测（gripper 反转，与仿真一致）。"""
    state16 = xvla20_to_ee16(state20, invert_gripper=True, clip_gripper=True)
    return {
        "vision": {
            name: {"color": image, "shape": image.shape} for name, image in images.items()
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


def compute_prediction_error(request_records: list[dict[str, Any]]) -> dict[str, Any]:
    """预测 vs 真实误差汇总（16d 空间，gripper 1=开）。

    每样本 ground truth = 训练一致网格插值 chunk（xvla20_to_ee16 invert_gripper=True）。
    group_mae 分组：position=xyz(0:3,8:11)，quaternion=wxyz(3:7,11:15)，gripper=1 维(7,15)。
    zero-motion 基线 = 始终预测当前状态时的误差（|gt - current_state|），量化"不做比做差多少"。
    """
    pos_dims = [0, 1, 2, 8, 9, 10]
    quat_dims = [3, 4, 5, 6, 11, 12, 13, 14]
    gripper_dims = [7, 15]

    errs: list[np.ndarray] = []   # [H,16] per request
    chunks: list[np.ndarray] = []  # [H,16] per request (gt)
    states: list[np.ndarray] = []  # [16] per request (input state)
    for rec in request_records:
        e = rec.get("error_abs")
        if not e:
            continue
        errs.append(np.asarray(e, dtype=np.float32))
        chunks.append(np.asarray(rec["ground_truth_actions"], dtype=np.float32))
        states.append(np.asarray(rec["input_state16"], dtype=np.float32))
    if not errs:
        return {"n_samples": 0, "note": "no ground-truth samples"}

    E = np.concatenate(errs, axis=0)  # [N*H,16]
    per_dim_mae = E.mean(axis=0).tolist()
    group_mae = lambda dims: float(E[:, dims].mean())
    # 逐 horizon 位置 MAE（观察误差沿 chunk 的演化）
    E_chunk = np.stack(errs, axis=0)  # [N,H,16]
    per_horizon_position_mae = E_chunk[:, :, pos_dims].mean(axis=(0, 2)).tolist()

    # zero-motion 基线：预测 = 当前状态（恒等复制）
    B = np.concatenate(
        [np.abs(gt - s[None, :]) for gt, s in zip(chunks, states)], axis=0
    )  # [N*H,16]
    return {
        "n_samples": int(len(errs)),
        "n_horizons": int(E_chunk.shape[1]),
        "per_dim_mae_16d": per_dim_mae,
        "dim_names": list(ACTION_NAMES),
        "position_mae": group_mae(pos_dims),
        "quaternion_mae": group_mae(quat_dims),
        "gripper_mae": group_mae(gripper_dims),
        "per_horizon_position_mae": per_horizon_position_mae,
        "zero_motion_position_mae": float(B[:, pos_dims].mean()),
        "zero_motion_gripper_mae": float(B[:, gripper_dims].mean()),
        "gt_convention": (
            "training grid interp1d(qdur, step=qdur/num_actions); "
            "16d gripper 1=open; units: m / unit-quat / dimensionless"
        ),
    }


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


def build_state_interpolant(state20: np.ndarray, qdur: float, num_actions: int):
    """训练一致动作网格插值器（与 LeRobotV3RoboDojoHandler.iter_episode 完全一致）。

    网格步长 = qdur/num_actions（与录制帧率解耦），lt = arange(T) * step。
    """
    T = state20.shape[0]
    lt = np.arange(T, dtype=np.float64) * (qdur / num_actions)
    L = interp1d(
        lt, state20, axis=0, bounds_error=False, fill_value=(state20[0], state20[-1])
    )
    return L, lt


def ground_truth_chunk(L, lt, idx: int, qdur: float, num_actions: int) -> np.ndarray | None:
    """训练一致 ground truth：start=idx 处起未来 num_actions 个绝对 20d 目标。

    双臂完全静止段返回 None（handler 对 (seq[1]-seq[0]).abs().max() < 1e-5 的样本同样跳过）。
    """
    q = np.linspace(lt[idx], lt[idx] + qdur, num_actions + 1, dtype=np.float32)
    seq = np.asarray(L(q), dtype=np.float32)  # [num_actions+1, 20]
    if float(np.abs(seq[1] - seq[0]).max()) < 1e-5:
        return None
    return seq[1:]  # [num_actions, 20]


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

    # 完整 20d 状态轨迹 + 训练一致网格插值器（ground truth 用；训练仅对
    # lt[i] <= lt[-1]-qdur 的帧采样，即 i < T-num_actions）
    state20_full = np.stack([
        np.asarray(r["observation.state"], dtype=np.float32) for r in rows
    ])  # [T, 20]
    L, lt = build_state_interpolant(state20_full, args.qdur, args.action_steps)
    max_valid_start = int(state20_full.shape[0]) - args.action_steps

    sample_indices = list(range(0, len(rows), args.stride))
    if args.max_samples > 0:
        sample_indices = sample_indices[: args.max_samples]

    client = PolicyEvalClient(
        PolicyEvalClientConfig(url=args.url, evaluation_id="xvla-2-dataset-mock")
    )
    await client.connect(handshake=True)
    await client.reset(trial_id=f"episode-{args.episode}")

    request_records = []
    curve_rows = []
    gripper_roundtrip_ok = True
    try:
        for request_index, local_index in enumerate(sample_indices):
            row = rows[local_index]
            state20 = np.asarray(row["observation.state"], dtype=np.float32)
            if state20.shape != (20,):
                raise ValueError(f"Expected 20d observation.state, got {state20.shape}")
            # 转换自检：16d -> 20d 再反转应回到原 gripper（转换本身无信息损失）
            state16 = xvla20_to_ee16(state20, invert_gripper=True, clip_gripper=True)
            back20 = ee16_to_xvla20(state16, invert_gripper=True)
            if not np.allclose(back20[[9, 19]], state20[[9, 19]], atol=1e-5):
                gripper_roundtrip_ok = False

            images = load_images(dataset, metadata, local_index, fps)
            observation = state20_to_observation(state20, images, instruction)

            start = time.perf_counter()
            response = await client.infer(
                observation,
                trial_id=f"episode-{args.episode}",
                action_case_id=instruction,
                step=request_index,
            )
            round_trip_ms = (time.perf_counter() - start) * 1000.0
            predicted = flatten_actions(response.payload["actions"])
            if len(predicted) != args.action_steps:
                print(
                    f"WARNING: expected {args.action_steps} actions, got {len(predicted)}",
                    flush=True,
                )

            # 预测 vs 真实误差（训练一致 ground truth，16d gripper 1=开，与服务端输出同空间）
            gt16 = None
            error_abs = None
            if local_index < max_valid_start:
                gt20 = ground_truth_chunk(L, lt, local_index, args.qdur, args.action_steps)
                if gt20 is not None:
                    gt16 = xvla20_to_ee16(gt20, invert_gripper=True)  # [H,16]
                    if predicted.shape == gt16.shape:
                        error_abs = np.abs(predicted - gt16)  # [H,16]
                    else:
                        print(
                            f"WARNING: shape mismatch predicted {predicted.shape} vs "
                            f"ground truth {gt16.shape}; skip error for request {request_index}",
                            flush=True,
                        )

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
                "input_state16": state16.tolist(),
                "input_state20": state20.tolist(),
                "image_summary": image_summary(images),
                "predicted_actions": predicted.tolist(),
                "ground_truth_actions": gt16.tolist() if gt16 is not None else None,
                "error_abs": error_abs.tolist() if error_abs is not None else None,
                "server_latency_ms": float(response.payload.get("latency_ms", 0.0)),
                "round_trip_ms": round_trip_ms,
            }
            request_records.append(record)
            print(
                f"request={request_index} frame={row['frame_index']} "
                f"steps={len(predicted)} server={record['server_latency_ms']:.1f}ms "
                f"roundtrip={round_trip_ms:.1f}ms",
                flush=True,
            )

            for horizon in range(len(predicted)):
                for dimension, name in enumerate(ACTION_NAMES):
                    curve_rows.append({
                        "request_index": request_index,
                        "frame_index": int(row["frame_index"]),
                        "horizon": horizon,
                        "dimension": dimension,
                        "name": name,
                        "state16": float(state16[dimension]),
                        "predicted": float(predicted[horizon, dimension]),
                        "ground_truth": float(gt16[horizon, dimension]) if gt16 is not None else None,
                        "error": float(error_abs[horizon, dimension]) if error_abs is not None else None,
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
        "expected_action_steps": args.action_steps,
        "actual_action_steps": [len(record["predicted_actions"]) for record in request_records],
        "gripper_inversion_roundtrip_ok": bool(gripper_roundtrip_ok),
        "output_diagnostics": {
            "all_finite": bool(np.isfinite(predicted_all).all()),
            "chunk_shape": list(predicted_all.shape),
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
        "prediction_error": compute_prediction_error(request_records),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    err_summary = summary.get("prediction_error") or {}
    if err_summary.get("n_samples"):
        print(
            f"prediction error: n={err_summary['n_samples']} "
            f"pos_mae={err_summary['position_mae']:.4f} "
            f"quat_mae={err_summary['quaternion_mae']:.4f} "
            f"gripper_mae={err_summary['gripper_mae']:.4f} "
            f"zero_motion_pos_mae={err_summary['zero_motion_position_mae']:.4f}",
            flush=True,
        )
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
