#!/usr/bin/env python
"""Break down where LanceDBDataset random-access __getitem__ spends time."""
from __future__ import annotations

import json
import pathlib
import statistics
import time

import numpy as np

ROOT = pathlib.Path("/cloud/cloud-ssd1/lerobot_data")
REPO = "real_lerobot_v30_joint"
LANCE = "/cloud/cloud-ssd1/lerobot_bench/dataset-lance"
N_SAMPLES = 50

fps = json.loads((ROOT / REPO / "meta" / "info.json").read_text())["fps"]
delta = {"action": [t / fps for t in range(50)]}
for k in ["observation.images.cam_high", "observation.images.cam_left_wrist", "observation.images.cam_right_wrist"]:
    delta[k] = [0.0]

from lerobot_lancedb import LanceDBDataset  # noqa: E402

ds = LanceDBDataset(root=LANCE, delta_timestamps=delta)
print(f"len(ds)={len(ds)}")

rng = np.random.RandomState(0)
idxs = rng.randint(0, len(ds), size=N_SAMPLES)

# warmup
ds[int(idxs[0])]

t_fetch, t_video, t_total = [], [], []
for idx in idxs:
    t0 = time.perf_counter()
    # stage A: frames-table tabular fetch (simulate what __getitems__ does pre-decode)
    plan = ds._plan_batch([int(idx)])[0]
    rows, row_pos = ds._batch_rows([plan])
    cols = ds._fetch_rows(rows)
    t1 = time.perf_counter()
    # stage B: video decode (call _decode_videos with the prepared pieces)
    windows = ds._plan_file_windows([plan])
    prepared = ds._prepare_files(sorted(windows), windows)
    decoded = ds._decode_videos([plan], cols, row_pos, prepared)
    t2 = time.perf_counter()
    # stage C: full __getitem__
    t3 = time.perf_counter()
    ds[int(idx)]
    t4 = time.perf_counter()
    t_fetch.append(t1 - t0)
    t_video.append(t2 - t1)
    t_total.append(t4 - t3)

tot = sum(statistics.mean(a) for a in (t_fetch, t_video))
print(f"\nper-sample random access: fetch={statistics.mean(t_fetch)*1e3:.1f}ms "
      f"video={statistics.mean(t_video)*1e3:.1f}ms | full __getitem__ mean={statistics.mean(t_total)*1e3:.1f}ms")
for name, arr in (("frames-table fetch", t_fetch), ("video fetch+decode", t_video)):
    print(f"  {name:22s} mean={statistics.mean(arr)*1e3:9.1f}ms p50={sorted(arr)[len(arr)//2]*1e3:9.1f}ms")
