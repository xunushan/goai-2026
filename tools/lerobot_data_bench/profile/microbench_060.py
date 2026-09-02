#!/usr/bin/env python
"""Micro-bench: two suspected overheads in lerobot 0.6.0 get_item.
1) action 50-row read: list-of-indices (current) vs contiguous slice.
2) ThreadPoolExecutor created per _query_videos call vs reused pool.
"""
import statistics
import time

import numpy as np

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
hfd = reader.hf_dataset
N = 200
rng = np.random.RandomState(0)
starts = rng.randint(0, len(ds) - 50, size=N)

# --- 1) action read: list-of-50 vs slice ---
t_list, t_slice = [], []
for a in starts:
    idxs = list(range(a, a + 50))
    t0 = time.perf_counter()
    v1 = hfd["action"][idxs]
    t1 = time.perf_counter()
    v2 = hfd["action"][a:a + 50]
    t2 = time.perf_counter()
    t_list.append(t1 - t0)
    t_slice.append(t2 - t1)
print(f"action read: list50 mean={statistics.mean(t_list)*1e3:.2f}ms  slice mean={statistics.mean(t_slice)*1e3:.2f}ms  ratio={statistics.mean(t_list)/statistics.mean(t_slice):.1f}x")

# --- 2) ThreadPoolExecutor creation overhead ---
from concurrent.futures import ThreadPoolExecutor

t_new, t_reuse = [], []
with ThreadPoolExecutor(max_workers=3) as pool:
    for _ in range(N):
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=3) as p2:
            pass
        t1 = time.perf_counter()
        fut = pool.submit(lambda: None)
        fut.result()
        t2 = time.perf_counter()
        t_new.append(t1 - t0)
        t_reuse.append(t2 - t1)
print(f"TPE: new-per-call mean={statistics.mean(t_new)*1e3:.3f}ms  reuse-submit mean={statistics.mean(t_reuse)*1e3:.3f}ms")
