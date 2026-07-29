#!/usr/bin/env python3
"""Report LeRobot episode/task distribution and on-disk ordering."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("data/lerobot_v30_ee"))
    args = parser.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

    root = args.dataset.resolve()
    metadata = LeRobotDatasetMetadata(root.name, root=root)
    episodes = metadata.episodes
    tasks = [tuple(value) for value in episodes["tasks"]]
    lengths = list(episodes["length"])

    episode_counts = Counter(tasks)
    frame_counts: Counter[tuple[str, ...]] = Counter()
    for task, length in zip(tasks, lengths, strict=True):
        frame_counts[task] += int(length)

    print(f"Episodes: {len(tasks)}")
    print(f"Unique task descriptions: {len(episode_counts)}")
    print("\nPer-task distribution:")
    for task, count in episode_counts.items():
        print(f"  episodes={count:4d} frames={frame_counts[task]:7d} task={task}")

    runs: list[tuple[int, int, tuple[str, ...]]] = []
    run_start = 0
    previous = tasks[0]
    for index, task in enumerate(tasks[1:], start=1):
        if task != previous:
            runs.append((run_start, index - 1, previous))
            run_start = index
            previous = task
    runs.append((run_start, len(tasks) - 1, previous))

    print(f"\nContiguous same-task runs: {len(runs)}")
    for start, end, task in runs:
        print(f"  episodes={start:4d}..{end:4d} count={end-start+1:4d} task={task}")


if __name__ == "__main__":
    main()
