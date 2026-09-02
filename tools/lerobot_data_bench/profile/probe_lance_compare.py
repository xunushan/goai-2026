#!/usr/bin/env python
"""Ground truth: pure ds[idx] random access vs torch DataLoader (shuffle=True) for LanceDBDataset."""
from __future__ import annotations

import json
import statistics
import time

import numpy as np
import torch

LANCE = "/cloud/cloud-ssd1/lerobot_bench/dataset-lance"

from lerobot_lancedb import LanceDBDataset  # noqa: E402

fps = 25
delta = {"action": [t / fps for t in range(50)]}
for k in ["observation.images.cam_high", "observation.images.cam_left_wrist", "observation.images.cam_right_wrist"]:
    delta[k] = [0.0]

ds = LanceDBDataset(root=LANCE, delta_timestamps=delta)
print(f"len={len(ds)}", flush=True)

# --- A. pure __getitem__ random loop ---
ds[0]  # warmup (opens tables/decoders)
idxs = np.random.RandomState(0).randint(0, len(ds), size=100)
ts = []
for idx in idxs:
    t0 = time.perf_counter()
    ds[int(idx)]
    ts.append(time.perf_counter() - t0)
print(f"A) pure ds[idx]: mean={statistics.mean(ts)*1e3:.1f}ms p50={sorted(ts)[len(ts)//2]*1e3:.1f}ms "
      f"p95={sorted(ts)[int(len(ts)*0.95)]*1e3:.1f}ms", flush=True)

# --- B. one DataLoader batch of 32 (shuffle=True) ---
def collate(items):
    def m(f, *v):
        if isinstance(v[0], dict):
            return {k: m(f, *[x[k] for x in v]) for k in v[0]}
        return f(*v)
    return m(lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0), *items)

g = torch.Generator()
g.manual_seed(0)
loader = torch.utils.data.DataLoader(ds, batch_size=32, num_workers=0, shuffle=True, drop_last=True,
                                     collate_fn=collate, generator=g)
it = iter(loader)
next(it)  # warmup batch
ts_b = []
for _ in range(5):
    t0 = time.perf_counter()
    next(it)
    ts_b.append(time.perf_counter() - t0)
print(f"B) DataLoader batch: mean={statistics.mean(ts_b)*1e3:.0f}ms per batch "
      f"({statistics.mean(ts_b)*1e3/32:.1f}ms/sample)", flush=True)
