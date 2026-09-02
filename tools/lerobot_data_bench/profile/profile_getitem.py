#!/usr/bin/env python
"""Break down where a single __getitem__ spends its time for lerobot 0.4.4.

Stages (mirroring lerobot/datasets/lerobot_dataset.py __getitem__):
  1. base parquet row read       : ds.hf_dataset[idx]
  2. delta timestamp parquet read: _get_query_indices + _query_hf_dataset
  3. video decode                : _query_videos (3 cams x 1 frame, pyav)
"""
from __future__ import annotations

import json
import pathlib
import statistics
import time

import numpy as np

ROOT = pathlib.Path("/cloud/cloud-ssd1/lerobot_data")
REPO = "real_lerobot_v30_joint"
N_SAMPLES = 200

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402

fps = json.loads((ROOT / REPO / "meta" / "info.json").read_text())["fps"]
delta = {"action": [t / fps for t in range(50)]}
for k in [
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
]:
    delta[k] = [0.0]

ds = LeRobotDataset(REPO, root=str(ROOT / REPO), delta_timestamps=delta, video_backend="pyav")
ds._ensure_hf_dataset_loaded()
print(f"len(ds)={len(ds)} fps={fps}")

rng = np.random.RandomState(0)
idxs = rng.randint(0, len(ds), size=N_SAMPLES)

# warmup (lazy video metadata load etc.)
i = int(idxs[0])
item = ds.hf_dataset[i]
ep_idx = item["episode_index"].item()
abs_idx = item["index"].item()
qs, padding = ds._get_query_indices(abs_idx, ep_idx)
ds._query_hf_dataset(qs)
current_ts = item["timestamp"].item()
qt = ds._get_query_timestamps(current_ts, qs)
ds._query_videos(qt, ep_idx)

t_hf, t_q, t_vid = [], [], []
for idx in idxs:
    t0 = time.perf_counter()
    item = ds.hf_dataset[idx]
    t1 = time.perf_counter()
    ep_idx = item["episode_index"].item()
    abs_idx = item["index"].item()
    qs, padding = ds._get_query_indices(abs_idx, ep_idx)
    qr = ds._query_hf_dataset(qs)
    t2 = time.perf_counter()
    current_ts = item["timestamp"].item()
    qt = ds._get_query_timestamps(current_ts, qs)
    vf = ds._query_videos(qt, ep_idx)
    t3 = time.perf_counter()
    t_hf.append(t1 - t0)
    t_q.append(t2 - t1)
    t_vid.append(t3 - t2)

tot_mean = sum(statistics.mean(a) for a in (t_hf, t_q, t_vid))
print(f"\nper-sample total(3 stages) mean={tot_mean*1e3:.2f}ms")
for name, arr in (
    ("parquet row read", t_hf),
    ("delta parquet read", t_q),
    ("video decode", t_vid),
):
    m = statistics.mean(arr)
    p50 = sorted(arr)[len(arr) // 2]
    print(f"  {name:20s} mean={m*1e3:8.2f}ms p50={p50*1e3:8.2f}ms  share={m/tot_mean*100:5.1f}%")
