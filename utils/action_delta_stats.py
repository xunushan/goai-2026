#!/usr/bin/env python3
"""Compute Cartesian action-delta statistics from a LeRobot EE dataset.

Two distributions are reported for each arm:

* state_to_action: expert action relative to the same frame's observation;
* consecutive_action: expert action relative to the preceding expert action
  in the same episode.

The first distribution directly matches the policy-server limiter reference.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq


PERCENTILES = (50.0, 90.0, 95.0, 99.0, 99.5, 99.9, 100.0)
ARM_SLICES = {
    "left": slice(0, 7),
    "right": slice(8, 15),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/act_lerobot_action_delta_stats.json"),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=65536,
        help="Parquet rows processed at once; does not load the dataset at once.",
    )
    return parser.parse_args()


def quaternion_angle_deg(q0: np.ndarray, q1: np.ndarray) -> np.ndarray:
    q0_norm = np.linalg.norm(q0, axis=1, keepdims=True)
    q1_norm = np.linalg.norm(q1, axis=1, keepdims=True)
    valid = (q0_norm[:, 0] > 1e-8) & (q1_norm[:, 0] > 1e-8)
    result = np.full(q0.shape[0], np.nan, dtype=np.float64)
    dots = np.sum(
        q0[valid] / q0_norm[valid] * (q1[valid] / q1_norm[valid]),
        axis=1,
    )
    result[valid] = np.rad2deg(
        2.0 * np.arccos(np.clip(np.abs(dots), 0.0, 1.0))
    )
    return result


def append_delta(
    store: dict[tuple[str, str, str], list[np.ndarray]],
    *,
    scope: str,
    relation: str,
    arm: str,
    reference: np.ndarray,
    target: np.ndarray,
) -> None:
    translation = np.linalg.norm(target[:, :3] - reference[:, :3], axis=1)
    rotation = quaternion_angle_deg(reference[:, 3:7], target[:, 3:7])
    store[(scope, relation, f"{arm}_translation_m")].append(translation)
    store[(scope, relation, f"{arm}_rotation_deg")].append(rotation)


def describe(parts: list[np.ndarray]) -> dict[str, Any]:
    if not parts:
        return {"count": 0}
    values = np.concatenate(parts)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"count": 0}
    quantiles = np.percentile(values, PERCENTILES)
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "percentiles": {
            f"p{percentile:g}": float(value)
            for percentile, value in zip(PERCENTILES, quantiles, strict=True)
        },
    }


def load_task_names(dataset: Path) -> dict[int, str]:
    path = dataset / "meta" / "tasks.parquet"
    if not path.is_file():
        return {}
    rows = pq.read_table(path).to_pylist()
    return {
        int(row["task_index"]): str(
            row.get("task")
            or row.get("__index_level_0__")
            or f"task_{row['task_index']}"
        )
        for row in rows
    }


def run(args: argparse.Namespace) -> None:
    dataset = args.dataset.resolve()
    paths = sorted((dataset / "data").rglob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No parquet files under {dataset / 'data'}")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    task_names = load_task_names(dataset)
    values: dict[tuple[str, str, str], list[np.ndarray]] = defaultdict(list)
    previous_action: dict[int, np.ndarray] = {}
    row_count = 0

    for path in paths:
        parquet = pq.ParquetFile(path)
        columns = [
            "observation.state",
            "action",
            "episode_index",
            "task_index",
        ]
        for batch in parquet.iter_batches(
            batch_size=args.batch_size,
            columns=columns,
        ):
            data = batch.to_pydict()
            state = np.asarray(data["observation.state"], dtype=np.float64)
            action = np.asarray(data["action"], dtype=np.float64)
            episode = np.asarray(data["episode_index"], dtype=np.int64)
            task = np.asarray(data["task_index"], dtype=np.int64)
            row_count += len(action)

            scopes: list[tuple[str, np.ndarray]] = [
                ("all", np.ones(len(action), dtype=bool))
            ]
            scopes.extend(
                (
                    f"task:{int(task_index)}:{task_names.get(int(task_index), 'unknown')}",
                    task == task_index,
                )
                for task_index in np.unique(task)
            )

            for scope, mask in scopes:
                for arm, pose_slice in ARM_SLICES.items():
                    append_delta(
                        values,
                        scope=scope,
                        relation="state_to_action",
                        arm=arm,
                        reference=state[mask, pose_slice],
                        target=action[mask, pose_slice],
                    )

            # Preserve episode boundaries, including across parquet batches.
            consecutive_reference = []
            consecutive_target = []
            consecutive_task = []
            for index in range(len(action)):
                episode_index = int(episode[index])
                prior = previous_action.get(episode_index)
                if prior is not None:
                    consecutive_reference.append(prior)
                    consecutive_target.append(action[index])
                    consecutive_task.append(int(task[index]))
                previous_action[episode_index] = action[index].copy()

            if consecutive_target:
                reference = np.asarray(consecutive_reference)
                target = np.asarray(consecutive_target)
                task_for_delta = np.asarray(consecutive_task)
                delta_scopes: list[tuple[str, np.ndarray]] = [
                    ("all", np.ones(len(target), dtype=bool))
                ]
                delta_scopes.extend(
                    (
                        f"task:{int(task_index)}:{task_names.get(int(task_index), 'unknown')}",
                        task_for_delta == task_index,
                    )
                    for task_index in np.unique(task_for_delta)
                )
                for scope, mask in delta_scopes:
                    for arm, pose_slice in ARM_SLICES.items():
                        append_delta(
                            values,
                            scope=scope,
                            relation="consecutive_action",
                            arm=arm,
                            reference=reference[mask, pose_slice],
                            target=target[mask, pose_slice],
                        )

        print(f"processed_rows={row_count} file={path}", flush=True)

    result: dict[str, Any] = {
        "dataset": str(dataset),
        "row_count": row_count,
        "units": {
            "translation": "meter",
            "rotation": "degree",
        },
        "percentiles": list(PERCENTILES),
        "scopes": {},
    }
    for (scope, relation, metric), parts in sorted(values.items()):
        result["scopes"].setdefault(scope, {}).setdefault(relation, {})[
            metric
        ] = describe(parts)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"saved={args.output.resolve()}", flush=True)
    print(
        json.dumps(result["scopes"].get("all", {}), ensure_ascii=False, indent=2),
        flush=True,
    )


if __name__ == "__main__":
    run(parse_args())
