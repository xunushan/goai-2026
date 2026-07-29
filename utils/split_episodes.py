#!/usr/bin/env python3
"""Create a reproducible task-stratified train/validation episode split.

This script does not modify the LeRobot dataset.  It reads the episode metadata
from ``meta/episodes/**/*.parquet`` and writes episode indices to a separate
JSON file.

Example:
    python act/split_episodes.py \
        --dataset-root /workspace/data/lerobot_v30_ee \
        --output /workspace/splits/lerobot_v30_ee_seed42.json \
        --train-ratio 0.9 \
        --tasks "Stack the three blocks with different textures." \
        --seed 42
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split LeRobot v3 episodes into task-stratified train/val sets."
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--tasks",
        nargs="+",
        help=(
            "Exact task strings to retain. If omitted, retain every task. "
            "Each task containing spaces must be shell-quoted."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing split file. By default an existing file is rejected.",
    )
    parser.add_argument(
        "--print-split",
        choices=("train", "val"),
        help="After validation, print one split as compact JSON for CLI use.",
    )
    return parser.parse_args()


def load_info(dataset_root: Path) -> dict[str, Any]:
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"Missing LeRobot metadata: {info_path}")
    with info_path.open(encoding="utf-8") as file:
        return json.load(file)


def load_episode_rows(dataset_root: Path) -> list[dict[str, Any]]:
    episode_dir = dataset_root / "meta" / "episodes"
    parquet_files = sorted(episode_dir.glob("chunk-*/*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No episode metadata parquet files found under {episode_dir}")

    rows: list[dict[str, Any]] = []
    for parquet_path in parquet_files:
        table = pq.read_table(parquet_path, columns=["episode_index", "tasks", "length"])
        rows.extend(table.to_pylist())
    rows.sort(key=lambda row: int(row["episode_index"]))
    return rows


def task_key(row: dict[str, Any]) -> str:
    """Return the episode's task used for stratification.

    The current GOAI dataset has exactly one task string per episode.  Reject
    empty or multi-task episodes instead of silently placing them in a wrong
    group.
    """

    tasks = row.get("tasks") or []
    if len(tasks) != 1:
        raise ValueError(
            f"episode {row.get('episode_index')} must contain exactly one task, got {tasks!r}"
        )
    return str(tasks[0])


def filter_rows_by_tasks(
    rows: list[dict[str, Any]],
    selected_tasks: list[str] | None,
) -> tuple[list[dict[str, Any]], list[str]]:
    available = sorted({task_key(row) for row in rows})
    if selected_tasks is None:
        return rows, available

    requested = list(dict.fromkeys(str(task) for task in selected_tasks))
    empty = [task for task in requested if not task]
    if empty:
        raise ValueError("Task names must not be empty")
    unknown = sorted(set(requested) - set(available))
    if unknown:
        rendered_available = "\n  ".join(available)
        raise ValueError(
            f"Unknown requested tasks: {unknown!r}\n"
            f"Available tasks:\n  {rendered_available}"
        )
    selected = set(requested)
    filtered = [row for row in rows if task_key(row) in selected]
    if not filtered:
        raise ValueError("Task filter selected zero episodes")
    return filtered, sorted(selected)


def split_rows(
    rows: list[dict[str, Any]], train_ratio: float, seed: int
) -> tuple[list[int], list[int], list[dict[str, Any]]]:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError(f"train_ratio must be between 0 and 1, got {train_ratio}")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[task_key(row)].append(row)

    train_indices: list[int] = []
    val_indices: list[int] = []
    per_task: list[dict[str, Any]] = []

    # A task-specific RNG makes the result stable even if metadata task groups
    # are encountered in a different order.
    for task_index, task in enumerate(sorted(grouped)):
        task_rows = sorted(grouped[task], key=lambda row: int(row["episode_index"]))
        indices = [int(row["episode_index"]) for row in task_rows]
        task_rng = random.Random(f"{seed}:{task_index}:{task}")
        task_rng.shuffle(indices)

        # Keep both sets non-empty. For the current dataset (100 episodes/task),
        # train_ratio=0.8 gives exactly 80 train and 20 validation episodes.
        n_train = math.floor(len(indices) * train_ratio)
        n_train = min(max(n_train, 1), len(indices) - 1)
        task_train = sorted(indices[:n_train])
        task_val = sorted(indices[n_train:])

        lengths = {int(row["episode_index"]): int(row["length"]) for row in task_rows}
        train_indices.extend(task_train)
        val_indices.extend(task_val)
        per_task.append(
            {
                "task": task,
                "total_episodes": len(indices),
                "train_episodes": len(task_train),
                "val_episodes": len(task_val),
                "train_frames": sum(lengths[index] for index in task_train),
                "val_frames": sum(lengths[index] for index in task_val),
            }
        )

    return sorted(train_indices), sorted(val_indices), per_task


def validate_split(
    split: dict[str, Any],
    rows: list[dict[str, Any]],
    expected_total: int,
    *,
    expected_seed: int | None = None,
    expected_train_ratio: float | None = None,
    expected_tasks: list[str] | None = None,
) -> None:
    train = [int(index) for index in split["train"]]
    val = [int(index) for index in split["val"]]
    all_episode_indices = {int(row["episode_index"]) for row in rows}

    if len(all_episode_indices) != len(rows):
        raise ValueError("Duplicate episode_index values found in episode metadata")
    if expected_total != len(rows):
        raise ValueError(
            f"info.json total_episodes={expected_total}, but metadata contains {len(rows)} rows"
        )
    if len(train) != len(set(train)) or len(val) != len(set(val)):
        raise ValueError("Duplicate episode indices found inside a split")
    overlap = set(train) & set(val)
    if overlap:
        raise ValueError(f"Train/val overlap detected: {sorted(overlap)[:20]}")
    if set(train) | set(val) != all_episode_indices:
        missing = all_episode_indices - (set(train) | set(val))
        unknown = (set(train) | set(val)) - all_episode_indices
        raise ValueError(
            f"Split does not cover the source dataset exactly; "
            f"missing={sorted(missing)[:20]}, unknown={sorted(unknown)[:20]}"
        )
    if expected_seed is not None and int(split.get("seed", -1)) != expected_seed:
        raise ValueError(
            f"Existing split seed={split.get('seed')} does not match requested seed={expected_seed}. "
            "Use a different output path or pass --overwrite."
        )
    if expected_train_ratio is not None and not math.isclose(
        float(split.get("train_ratio", -1.0)), expected_train_ratio
    ):
        raise ValueError(
            f"Existing split train_ratio={split.get('train_ratio')} does not match requested "
            f"train_ratio={expected_train_ratio}. Use a different output path or pass --overwrite."
        )
    if expected_tasks is not None and sorted(split.get("selected_tasks", [])) != sorted(
        expected_tasks
    ):
        raise ValueError(
            f"Existing split selected_tasks={split.get('selected_tasks')} does not match "
            f"requested tasks={expected_tasks}. Use a different output path or pass --overwrite."
        )


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    output = args.output.resolve()
    info = load_info(dataset_root)
    all_rows = load_episode_rows(dataset_root)
    rows, selected_tasks = filter_rows_by_tasks(all_rows, args.tasks)

    if output.exists() and not args.overwrite:
        with output.open(encoding="utf-8") as file:
            split = json.load(file)
        validate_split(
            split,
            rows,
            len(rows),
            expected_seed=args.seed,
            expected_train_ratio=args.train_ratio,
            expected_tasks=selected_tasks,
        )
    else:
        train, val, per_task = split_rows(rows, args.train_ratio, args.seed)
        split = {
            "version": 2,
            "strategy": "task_filtered_stratified_random_episode_split",
            "seed": args.seed,
            "train_ratio": args.train_ratio,
            "val_ratio": 1.0 - args.train_ratio,
            "selected_tasks": selected_tasks,
            "source_dataset": {
                "root": str(dataset_root),
                "codebase_version": info.get("codebase_version"),
                "total_episodes": int(info["total_episodes"]),
                "total_frames": int(info["total_frames"]),
                "total_tasks": int(info["total_tasks"]),
                "selected_episodes": len(rows),
                "selected_tasks": len(selected_tasks),
            },
            "train": train,
            "val": val,
            "summary": {
                "train_episodes": len(train),
                "val_episodes": len(val),
                "per_task": per_task,
            },
        }
        validate_split(
            split,
            rows,
            len(rows),
            expected_seed=args.seed,
            expected_train_ratio=args.train_ratio,
            expected_tasks=selected_tasks,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as file:
            json.dump(split, file, ensure_ascii=False, indent=2)
            file.write("\n")

    if args.print_split:
        print(json.dumps(split[args.print_split], separators=(",", ":")))
    else:
        summary = split["summary"]
        print(f"Split file: {output}")
        print(
            f"Episodes: train={summary['train_episodes']}, "
            f"val={summary['val_episodes']}"
        )
        for item in summary["per_task"]:
            print(
                f"- {item['task']}: "
                f"{item['train_episodes']} train / {item['val_episodes']} val"
            )


if __name__ == "__main__":
    main()
