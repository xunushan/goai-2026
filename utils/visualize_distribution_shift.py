#!/usr/bin/env python3
"""Visualize ACT closed-loop distribution shift against one training task.

Example:
    python act/visualize_distribution_shift.py \
        --dataset-root data/lerobot_v30_ee \
        --task "Stack the three blocks with different textures." \
        --run rot5=eval_results/stack_blocks_rot5 \
        --run rot6=eval_results/stack_blocks_limiter_test \
        --output-dir outputs/stack_blocks_distribution_shift

Each run directory must contain one policy-server ``*.log`` and may contain an
``ik_failures.jsonl``. Large logs are scanned line by line.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq


IO_MARKER = "[act_lerobot][io] "
RIGHT_POSE = slice(8, 15)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--task", required=True, help="Exact dataset task text.")
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="LABEL=RESULT_DIR",
        help="Repeat for every rollout result directory.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-pca-training-points", type=int, default=20000)
    return parser.parse_args()


def parse_runs(values: list[str]) -> dict[str, Path]:
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--run must be LABEL=RESULT_DIR, got {value!r}")
        label, path = value.split("=", 1)
        if not label or label in result:
            raise ValueError(f"Invalid or duplicate run label: {label!r}")
        directory = Path(path).resolve()
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        result[label] = directory
    return result


def task_index_for_name(dataset_root: Path, task: str) -> int:
    path = dataset_root / "meta" / "tasks.parquet"
    rows = pq.read_table(path).to_pylist()
    mapping = {
        str(row.get("task") or row.get("__index_level_0__")): int(row["task_index"])
        for row in rows
    }
    if task not in mapping:
        available = "\n  ".join(sorted(mapping))
        raise ValueError(f"Unknown task {task!r}. Available tasks:\n  {available}")
    return mapping[task]


def load_task_training_data(
    dataset_root: Path,
    task_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    states = []
    actions = []
    paths = sorted((dataset_root / "data").rglob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No parquet files under {dataset_root / 'data'}")
    for path in paths:
        table = pq.read_table(
            path,
            columns=["observation.state", "action", "task_index"],
            filters=[("task_index", "=", task_index)],
        )
        if table.num_rows == 0:
            continue
        data = table.to_pydict()
        states.append(np.asarray(data["observation.state"], dtype=np.float64))
        actions.append(np.asarray(data["action"], dtype=np.float64))
    if not states:
        raise ValueError(f"Task index {task_index} has no training frames")
    return np.concatenate(states), np.concatenate(actions)


def find_server_log(directory: Path) -> Path:
    candidates = sorted(directory.glob("*.log"))
    policy_logs = [path for path in candidates if "server" in path.name]
    if len(policy_logs) == 1:
        return policy_logs[0]
    if len(candidates) == 1:
        return candidates[0]
    raise ValueError(
        f"Expected exactly one policy server log under {directory}, "
        f"found {[path.name for path in candidates]}"
    )


def load_server_events(path: Path) -> list[dict[str, Any]]:
    events = []
    with path.open(encoding="utf-8", errors="replace") as file:
        for line in file:
            marker_index = line.find(IO_MARKER)
            if marker_index < 0:
                continue
            payload = line[marker_index + len(IO_MARKER) :].strip()
            try:
                event = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if event.get("event") in {"client_observation", "server_actions"}:
                events.append(event)
    return events


def first_ik_failure(directory: Path) -> int | None:
    path = directory / "ik_failures.jsonl"
    if not path.is_file():
        return None
    with path.open(encoding="utf-8", errors="replace") as file:
        for line in file:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("event") == "failure_start" and event.get("arm") == "right_arm":
                return int(event["step"])
    return None


def canonical_quaternion(quaternion: np.ndarray) -> np.ndarray:
    result = np.asarray(quaternion, dtype=np.float64).copy()
    norm = np.linalg.norm(result, axis=-1, keepdims=True)
    result = result / np.maximum(norm, 1e-12)
    flip = result[..., :1] < 0.0
    return np.where(flip, -result, result)


def quaternion_angle_deg(q0: np.ndarray, q1: np.ndarray) -> np.ndarray:
    q0 = canonical_quaternion(q0)
    q1 = canonical_quaternion(q1)
    dots = np.sum(q0 * q1, axis=-1)
    return np.rad2deg(2.0 * np.arccos(np.clip(np.abs(dots), 0.0, 1.0)))


def pose_delta(reference: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    translation = np.linalg.norm(target[..., :3] - reference[..., :3], axis=-1)
    rotation = quaternion_angle_deg(reference[..., 3:7], target[..., 3:7])
    return translation, rotation


def percentile_summary(values: np.ndarray) -> dict[str, float]:
    return {
        "p95": float(np.percentile(values, 95.0)),
        "p99": float(np.percentile(values, 99.0)),
        "p99.9": float(np.percentile(values, 99.9)),
        "max": float(np.max(values)),
    }


def rollout_from_events(events: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    observations = {}
    actions = {}
    limiter = {}
    for event in events:
        request = int(event["request"])
        if event["event"] == "client_observation":
            observations[request] = np.asarray(event["state16"], dtype=np.float64)[RIGHT_POSE]
        elif event["event"] == "server_actions":
            actions[request] = np.asarray(event["actions"][0], dtype=np.float64)[RIGHT_POSE]
            details = event.get("action_limiter", {}).get("steps", [])
            if details:
                limiter[request] = details[0].get("arms", {}).get("right", {})

    requests = sorted(set(observations) & set(actions))
    if not requests:
        raise ValueError("Server log contains no paired observations/actions")
    state = np.stack([observations[index] for index in requests])
    action = np.stack([actions[index] for index in requests])
    output_translation, output_rotation = pose_delta(state, action)
    raw_translation = np.asarray(
        [limiter.get(index, {}).get("raw_translation_m", output_translation[i])
         for i, index in enumerate(requests)]
    )
    raw_rotation = np.asarray(
        [limiter.get(index, {}).get("raw_rotation_deg", output_rotation[i])
         for i, index in enumerate(requests)]
    )
    return {
        "request": np.asarray(requests),
        "state": state,
        "action": action,
        "raw_translation": raw_translation,
        "raw_rotation": raw_rotation,
        "output_translation": output_translation,
        "output_rotation": output_rotation,
    }


def nearest_training_pose(
    training_pose: np.ndarray,
    rollout_pose: np.ndarray,
    translation_scale: float,
    rotation_scale: float,
) -> dict[str, np.ndarray]:
    nearest_score = []
    nearest_translation = []
    nearest_rotation = []
    # Chunk over rollout queries; the selected task is small enough to keep its
    # pose matrix resident, while the full source dataset is never loaded.
    for query in rollout_pose:
        translation = np.linalg.norm(training_pose[:, :3] - query[:3], axis=1)
        rotation = quaternion_angle_deg(
            training_pose[:, 3:7],
            np.broadcast_to(query[3:7], training_pose[:, 3:7].shape),
        )
        score = np.sqrt(
            (translation / max(translation_scale, 1e-12)) ** 2
            + (rotation / max(rotation_scale, 1e-12)) ** 2
        )
        index = int(np.argmin(score))
        nearest_score.append(score[index])
        nearest_translation.append(translation[index])
        nearest_rotation.append(rotation[index])
    return {
        "score": np.asarray(nearest_score),
        "translation_m": np.asarray(nearest_translation),
        "rotation_deg": np.asarray(nearest_rotation),
    }


def pca_projection(
    training_pose: np.ndarray,
    rollout_poses: dict[str, np.ndarray],
    max_training_points: int,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, Any]]:
    training = training_pose.copy()
    training[:, 3:7] = canonical_quaternion(training[:, 3:7])
    if len(training) > max_training_points:
        indices = np.linspace(0, len(training) - 1, max_training_points).astype(int)
        training = training[indices]
    mean = training.mean(axis=0)
    std = training.std(axis=0)
    std[std < 1e-9] = 1.0
    normalized = (training - mean) / std
    covariance = normalized.T @ normalized / max(len(normalized) - 1, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    components = eigenvectors[:, order[:2]]
    projected_training = normalized @ components
    projected_rollouts = {}
    for label, pose in rollout_poses.items():
        value = pose.copy()
        value[:, 3:7] = canonical_quaternion(value[:, 3:7])
        projected_rollouts[label] = ((value - mean) / std) @ components
    explained = eigenvalues[order[:2]] / np.maximum(eigenvalues.sum(), 1e-12)
    return projected_training, projected_rollouts, {
        "explained_variance_ratio": explained.tolist()
    }


def add_failure_line(axis, failure_step: int | None) -> None:
    if failure_step is not None:
        axis.axvline(failure_step, color="red", linestyle="--", alpha=0.8, label="first IK failure")


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dirs = parse_runs(args.run)

    task_index = task_index_for_name(dataset_root, args.task)
    training_state, training_action = load_task_training_data(dataset_root, task_index)
    training_pose = training_state[:, RIGHT_POSE]
    expert_translation, expert_rotation = pose_delta(
        training_state[:, RIGHT_POSE],
        training_action[:, RIGHT_POSE],
    )
    training_stats = {
        "translation_m": percentile_summary(expert_translation),
        "rotation_deg": percentile_summary(expert_rotation),
    }

    rollouts = {}
    failures = {}
    for label, directory in run_dirs.items():
        rollouts[label] = rollout_from_events(load_server_events(find_server_log(directory)))
        failures[label] = first_ik_failure(directory)

    # 1) Raw/output action deltas against training percentiles.
    figure, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    for label, rollout in rollouts.items():
        axes[0].plot(rollout["request"], rollout["raw_translation"] * 1000.0, alpha=0.65, label=f"{label} raw")
        axes[0].plot(rollout["request"], rollout["output_translation"] * 1000.0, label=f"{label} output")
        axes[1].plot(rollout["request"], rollout["raw_rotation"], alpha=0.65, label=f"{label} raw")
        axes[1].plot(rollout["request"], rollout["output_rotation"], label=f"{label} output")
        add_failure_line(axes[0], failures[label])
        add_failure_line(axes[1], failures[label])
    for name, style in (("p95", ":"), ("p99", "-."), ("max", "--")):
        axes[0].axhline(training_stats["translation_m"][name] * 1000.0, color="black", linestyle=style, alpha=0.6, label=f"train {name}")
        axes[1].axhline(training_stats["rotation_deg"][name], color="black", linestyle=style, alpha=0.6, label=f"train {name}")
    axes[0].set_ylabel("Right EE translation delta (mm)")
    axes[1].set_ylabel("Right EE rotation delta (deg)")
    axes[1].set_xlabel("Policy request / executed step")
    for axis in axes:
        axis.grid(alpha=0.25)
        handles, labels = axis.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        axis.legend(unique.values(), unique.keys(), ncol=3, fontsize=8)
    figure.suptitle(f"Action deltas vs training distribution\n{args.task}")
    figure.tight_layout()
    figure.savefig(output_dir / "action_delta_timeline.png", dpi=180)
    plt.close(figure)

    # 2) Geometric kNN distance to the selected task's training pose manifold.
    knn = {}
    figure, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True)
    for label, rollout in rollouts.items():
        knn[label] = nearest_training_pose(
            training_pose,
            rollout["state"],
            training_stats["translation_m"]["p95"],
            training_stats["rotation_deg"]["p95"],
        )
        axes[0].plot(rollout["request"], knn[label]["score"], label=label)
        axes[1].plot(rollout["request"], knn[label]["translation_m"] * 1000.0, label=label)
        axes[2].plot(rollout["request"], knn[label]["rotation_deg"], label=label)
        for axis in axes:
            add_failure_line(axis, failures[label])
    axes[0].set_ylabel("Normalized combined kNN distance")
    axes[1].set_ylabel("Nearest pose translation (mm)")
    axes[2].set_ylabel("Nearest pose rotation (deg)")
    axes[2].set_xlabel("Policy request / executed step")
    for axis in axes:
        axis.grid(alpha=0.25)
        handles, labels = axis.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        axis.legend(unique.values(), unique.keys(), fontsize=8)
    figure.suptitle(f"Rollout distance to training EE-pose manifold\n{args.task}")
    figure.tight_layout()
    figure.savefig(output_dir / "rollout_knn_distance.png", dpi=180)
    plt.close(figure)

    # 3) PCA overview of training and rollout right-EE poses.
    projected_training, projected_rollouts, pca_info = pca_projection(
        training_pose,
        {label: rollout["state"] for label, rollout in rollouts.items()},
        args.max_pca_training_points,
    )
    figure, axis = plt.subplots(figsize=(11, 9))
    axis.scatter(
        projected_training[:, 0],
        projected_training[:, 1],
        s=4,
        alpha=0.12,
        color="gray",
        label="training poses",
    )
    for label, projected in projected_rollouts.items():
        axis.plot(projected[:, 0], projected[:, 1], linewidth=1.5, label=f"{label} rollout")
        axis.scatter(projected[0, 0], projected[0, 1], marker="o", s=45)
        failure = failures[label]
        if failure is not None and failure - 1 < len(projected):
            point = projected[failure - 1]
            axis.scatter(point[0], point[1], marker="x", s=90, linewidth=2, label=f"{label} IK failure")
    axis.set_xlabel("Pose PCA 1")
    axis.set_ylabel("Pose PCA 2")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    axis.set_title(f"Right EE pose manifold\n{args.task}")
    figure.tight_layout()
    figure.savefig(output_dir / "right_ee_pose_pca.png", dpi=180)
    plt.close(figure)

    summary = {
        "dataset_root": str(dataset_root),
        "task": args.task,
        "task_index": task_index,
        "training_frames": len(training_state),
        "training_state_to_action": training_stats,
        "pca": pca_info,
        "runs": {},
    }
    for label, rollout in rollouts.items():
        failure = failures[label]
        failure_index = None
        if failure is not None:
            matches = np.flatnonzero(rollout["request"] == failure)
            if len(matches):
                failure_index = int(matches[0])
        summary["runs"][label] = {
            "result_dir": str(run_dirs[label]),
            "requests": int(len(rollout["request"])),
            "first_ik_failure_step": failure,
            "max_raw_translation_m": float(np.max(rollout["raw_translation"])),
            "max_raw_rotation_deg": float(np.max(rollout["raw_rotation"])),
            "max_knn_score": float(np.max(knn[label]["score"])),
            "at_first_ik_failure": None if failure_index is None else {
                "knn_score": float(knn[label]["score"][failure_index]),
                "nearest_training_translation_m": float(
                    knn[label]["translation_m"][failure_index]
                ),
                "nearest_training_rotation_deg": float(
                    knn[label]["rotation_deg"][failure_index]
                ),
            },
        }
    (output_dir / "distribution_shift_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved visualizations to {output_dir}")


if __name__ == "__main__":
    main()
