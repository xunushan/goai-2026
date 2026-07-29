#!/usr/bin/env python3
"""Compare training gripper labels with a policy-server rollout.

The parquet dataset is read one row group at a time and only the state/action
columns of selected training episodes are loaded.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq

IO_MARKER = "[act_lerobot][io] "
LEFT_GRIPPER = 7
RIGHT_GRIPPER = 15


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--train-config", type=Path, required=True)
    parser.add_argument("--server-log", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_training_grippers(
    dataset_root: Path, train_episodes: set[int]
) -> tuple[np.ndarray, np.ndarray, list[dict[str, np.ndarray]]]:
    actions: list[np.ndarray] = []
    states: list[np.ndarray] = []
    episodes: list[dict[str, np.ndarray]] = []
    for path in sorted((dataset_root / "data").glob("**/*.parquet")):
        parquet = pq.ParquetFile(path)
        for row_group in range(parquet.num_row_groups):
            column = parquet.metadata.row_group(row_group).column(
                parquet.schema_arrow.get_field_index("episode_index")
            )
            stats = column.statistics
            if stats is not None and (
                int(stats.max) < min(train_episodes)
                or int(stats.min) > max(train_episodes)
            ):
                continue
            table = parquet.read_row_group(
                row_group,
                columns=["observation.state", "action", "episode_index", "frame_index"],
            )
            episode_index = np.asarray(table["episode_index"])
            mask = np.isin(episode_index, list(train_episodes))
            if not np.any(mask):
                continue
            action = np.asarray(table["action"].to_pylist(), dtype=np.float64)[mask]
            state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float64)[mask]
            frame = np.asarray(table["frame_index"])[mask]
            selected_episode = episode_index[mask]
            actions.append(action[:, [LEFT_GRIPPER, RIGHT_GRIPPER]])
            states.append(state[:, [LEFT_GRIPPER, RIGHT_GRIPPER]])
            for index in np.unique(selected_episode):
                episode_mask = selected_episode == index
                order = np.argsort(frame[episode_mask])
                episodes.append(
                    {
                        "episode_index": np.asarray([int(index)]),
                        "action": action[episode_mask][order][:, [LEFT_GRIPPER, RIGHT_GRIPPER]],
                    }
                )
    if not actions:
        raise ValueError("No selected training episodes found in parquet data")
    return np.concatenate(actions), np.concatenate(states), episodes


def load_rollout(server_log: Path) -> dict[str, np.ndarray]:
    observations: dict[int, np.ndarray] = {}
    actions: dict[int, np.ndarray] = {}
    instructions: dict[int, str] = {}
    with server_log.open(encoding="utf-8", errors="replace") as file:
        for line in file:
            if IO_MARKER not in line:
                continue
            try:
                event = json.loads(line.split(IO_MARKER, 1)[1])
            except json.JSONDecodeError:
                continue
            request = int(event["request"])
            if event.get("event") == "client_observation":
                observations[request] = np.asarray(event["state16"], dtype=np.float64)
                instructions[request] = event.get("instruction", "")
            elif event.get("event") == "server_actions":
                actions[request] = np.asarray(event["actions"][0], dtype=np.float64)
    requests = sorted(set(observations) & set(actions))
    if not requests:
        raise ValueError("No paired observation/action events found")
    return {
        "request": np.asarray(requests),
        "state": np.stack([observations[index] for index in requests]),
        "action": np.stack([actions[index] for index in requests]),
        "instructions": np.asarray([instructions[index] for index in requests]),
    }


def stats(values: np.ndarray) -> dict[str, Any]:
    return {
        "count": int(values.size),
        "min": float(np.min(values)),
        "p01": float(np.percentile(values, 1)),
        "p10": float(np.percentile(values, 10)),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "p99": float(np.percentile(values, 99)),
        "max": float(np.max(values)),
        "fraction_le_0.5": float(np.mean(values <= 0.5)),
        "fraction_ge_0.9": float(np.mean(values >= 0.9)),
    }


def normalized_episode_curves(
    episodes: list[dict[str, np.ndarray]], arm: int, points: int = 101
) -> np.ndarray:
    destination = np.linspace(0.0, 1.0, points)
    curves = []
    for episode in episodes:
        values = episode["action"][:, arm]
        source = np.linspace(0.0, 1.0, len(values))
        curves.append(np.interp(destination, source, values))
    return np.stack(curves)


def main() -> None:
    args = parse_args()
    config = json.loads(args.train_config.read_text(encoding="utf-8"))
    train_episodes = {int(index) for index in config["dataset"]["episodes"]}
    train_action, train_state, episodes = load_training_grippers(
        args.dataset_root, train_episodes
    )
    rollout = load_rollout(args.server_log)
    rollout_action = rollout["action"][:, [LEFT_GRIPPER, RIGHT_GRIPPER]]
    rollout_state = rollout["state"][:, [LEFT_GRIPPER, RIGHT_GRIPPER]]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 2, figsize=(15, 10))
    colors = ("tab:blue", "tab:orange")
    names = ("left", "right")

    for arm, (name, color) in enumerate(zip(names, colors)):
        axes[0, 0].hist(
            train_action[:, arm],
            bins=np.linspace(0, 1, 51),
            alpha=0.45,
            color=color,
            label=f"training {name}",
        )
        axes[0, 0].axvline(
            np.median(rollout_action[:, arm]),
            color=color,
            linestyle="--",
            label=f"rollout {name} median",
        )
        axes[0, 1].plot(
            rollout["request"],
            rollout_action[:, arm],
            color=color,
            label=f"{name} action",
        )
        axes[0, 1].plot(
            rollout["request"],
            rollout_state[:, arm],
            color=color,
            linestyle=":",
            alpha=0.7,
            label=f"{name} state",
        )
        curves = normalized_episode_curves(episodes, arm)
        x = np.linspace(0, 100, curves.shape[1])
        median = np.median(curves, axis=0)
        low, high = np.percentile(curves, [10, 90], axis=0)
        axes[1, 0].plot(x, median, color=color, label=f"{name} median")
        axes[1, 0].fill_between(x, low, high, color=color, alpha=0.18, label=f"{name} P10–P90")
        axes[1, 1].hist(
            rollout_action[:, arm],
            bins=np.linspace(0.9, 1.0, 51),
            alpha=0.45,
            color=color,
            label=f"rollout {name}",
        )

    axes[0, 0].set_title("Training action gripper distribution")
    axes[0, 0].set_xlabel("Gripper action")
    axes[0, 0].set_ylabel("Frames")
    axes[0, 0].set_yscale("log")
    axes[0, 1].set_title("Rollout gripper timeline")
    axes[0, 1].set_xlabel("Policy request / executed step")
    axes[0, 1].set_ylabel("Gripper value")
    axes[1, 0].set_title("Training gripper action over normalized episode progress")
    axes[1, 0].set_xlabel("Episode progress (%)")
    axes[1, 0].set_ylabel("Gripper action")
    axes[1, 1].set_title("Rollout action detail near open state")
    axes[1, 1].set_xlabel("Gripper action")
    axes[1, 1].set_ylabel("Frames")
    for axis in axes.flat:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(args.output_dir / "gripper_training_vs_rollout.png", dpi=180)
    plt.close(figure)

    per_episode = {}
    for arm, name in enumerate(names):
        closed = []
        first_closed_progress = []
        for episode in episodes:
            values = episode["action"][:, arm]
            indices = np.flatnonzero(values <= 0.5)
            closed.append(bool(indices.size))
            if indices.size:
                first_closed_progress.append(float(indices[0] / max(len(values) - 1, 1)))
        per_episode[name] = {
            "episodes_with_action_le_0.5": int(sum(closed)),
            "total_episodes": len(closed),
            "median_first_action_le_0.5_progress": (
                float(np.median(first_closed_progress))
                if first_closed_progress
                else None
            ),
        }

    report = {
        "train_config": str(args.train_config.resolve()),
        "dataset_root": str(args.dataset_root.resolve()),
        "training_episode_count": len(train_episodes),
        "training_frame_count": int(len(train_action)),
        "training": {
            "action": {
                "left": stats(train_action[:, 0]),
                "right": stats(train_action[:, 1]),
            },
            "state": {
                "left": stats(train_state[:, 0]),
                "right": stats(train_state[:, 1]),
            },
            "per_episode": per_episode,
        },
        "rollout": {
            "request_count": int(len(rollout["request"])),
            "instructions": dict(Counter(rollout["instructions"].tolist())),
            "action": {
                "left": stats(rollout_action[:, 0]),
                "right": stats(rollout_action[:, 1]),
            },
            "state": {
                "left": stats(rollout_state[:, 0]),
                "right": stats(rollout_state[:, 1]),
            },
        },
    }
    (args.output_dir / "gripper_analysis.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
