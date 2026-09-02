#!/usr/bin/env python
"""Break down where a single __getitem__ spends its time for lerobot 0.6.0.

Stages (mirroring dataset_reader.py get_item):
  1. base parquet row read     : reader.hf_dataset[idx]  (state/action/timestamp...)
  2. delta parquet read        : _get_query_indices + _query_hf_dataset (action horizon rows)
  3. video decode              : _get_query_timestamps + _query_videos (3 cams)
  4. task lookup               : meta.tasks.iloc[task_idx].name
"""
import json
import statistics
import time

import numpy as np

# Local-only subset: skip hub revision sync.
import lerobot.datasets.dataset_metadata as _dmeta
import lerobot.datasets.lerobot_dataset as _lds


def _sv(repo_id, version):
    return version


_dmeta.get_safe_version = _sv
_lds.get_safe_version = _sv

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402

fps = 25
delta = {
    "action": [t / fps for t in range(50)],
    "observation.images.cam_high": [0.0],
    "observation.images.cam_left_wrist": [0.0],
    "observation.images.cam_right_wrist": [0.0],
}
ds = LeRobotDataset(
    repo_id="real_lerobot_v30_joint_224p3",
    root="/cloud/cloud-ssd1/lerobot_bench/ds224/real_lerobot_v30_joint_224p3",
    episodes=list(range(46, 63)),
    delta_timestamps=delta,
    video_backend="torchcodec",
    return_uint8=True,
)
reader = ds.reader
print(f"len={len(ds)} video_keys={len(reader._meta.video_keys)} "
      f"delta_indices keys={list(reader.delta_indices)}")

rng = np.random.RandomState(0)
idxs = rng.randint(0, len(ds), size=300)

# warmup (lazy decoder cache fill etc.)
reader.get_item(int(idxs[0]))

t_base, t_delta, t_vid, t_task, t_ts = [], [], [], [], []
for idx in idxs:
    t0 = time.perf_counter()
    item = reader.hf_dataset[idx]
    ep_idx = item["episode_index"].item()
    abs_idx = item["index"].item()
    t1 = time.perf_counter()
    qi, padding = reader._get_query_indices(abs_idx, ep_idx)
    qr = reader._query_hf_dataset(qi)
    t2 = time.perf_counter()
    current_ts = item["timestamp"].item()
    qt = reader._get_query_timestamps(current_ts, qi)
    t3 = time.perf_counter()
    vf = reader._query_videos(qt, ep_idx)
    t4 = time.perf_counter()
    task_idx = item["task_index"].item()
    _ = reader._meta.tasks.iloc[task_idx].name
    t5 = time.perf_counter()
    t_base.append(t1 - t0)
    t_delta.append(t2 - t1)
    t_ts.append(t3 - t2)
    t_vid.append(t4 - t3)
    t_task.append(t5 - t4)

rows = [
    ("base parquet row", t_base),
    ("action parquet", t_delta),
    ("query timestamps", t_ts),
    ("video decode", t_vid),
    ("task lookup", t_task),
]
tot = sum(statistics.mean(a) for _, a in rows)
print(f"\nsum of stages (per sample) mean={tot*1e3:.2f}ms")
for name, arr in rows:
    m = statistics.mean(arr)
    p50 = sorted(arr)[len(arr) // 2]
    print(f"  {name:18s} mean={m*1e3:7.2f}ms p50={p50*1e3:7.2f}ms  share={m / tot * 100:5.1f}%")
