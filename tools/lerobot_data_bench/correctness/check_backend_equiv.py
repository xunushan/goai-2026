#!/usr/bin/env python
"""Plan section 7 correctness: pyav vs torchcodec equivalence on LeRobotDataset 0.4.4.

Loads the same dataset with video_backend=pyav and video_backend=torchcodec,
fetches the same fixed indices (incl. episode boundaries), and compares
key sets / shapes / dtypes / image pixels / action arrays.
Pass = max |pyav - torchcodec| <= 1/255 (one uint8 quantization step).
"""
from __future__ import annotations

import numpy as np

ROOT = "/cloud/cloud-ssd1/lerobot_data/real_lerobot_v30_joint"
REPO = "real_lerobot_v30_joint"
CAMERAS = ["observation.images.cam_high", "observation.images.cam_left_wrist", "observation.images.cam_right_wrist"]

import sys  # noqa: E402
sys.path.insert(0, "/cloud/cloud-ssd1/lerobot_bench")
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402

fps = 25
delta = {"action": [t / fps for t in range(50)]}
for k in CAMERAS:
    delta[k] = [0.0]

ds_pyav = LeRobotDataset(repo_id=REPO, root=ROOT, delta_timestamps=delta, video_backend="pyav")
ds_tc = LeRobotDataset(repo_id=REPO, root=ROOT, delta_timestamps=delta, video_backend="torchcodec")
print(f"len={len(ds_pyav)} (pyav) vs {len(ds_tc)} (torchcodec)")

idxs = [0, 1, 31, 100, 100000, 400000, len(ds_pyav) - 2, len(ds_pyav) - 1]
print(f"idxs={idxs}")

ds_pyav[0]; ds_tc[0]  # warmup

ok_all = True
for i in idxs:
    a = ds_pyav[i]
    b = ds_tc[i]
    keys_a, keys_b = set(a.keys()), set(b.keys())
    if keys_a != keys_b:
        print(f"[{i}] KEY MISMATCH: {keys_a ^ keys_b}")
        ok_all = False
    for cam in CAMERAS:
        im_a = np.asarray(a[cam], dtype=np.float32)
        im_b = np.asarray(b[cam], dtype=np.float32)
        shape_ok = im_a.shape == im_b.shape and im_a.dtype == im_b.dtype
        if not shape_ok:
            print(f"[{i}] {cam} SHAPE/DTYPE MISMATCH: {im_a.shape}/{im_a.dtype} vs {im_b.shape}/{im_b.dtype}")
            ok_all = False
            continue
        diff = np.abs(im_a - im_b)
        # source is uint8; both should be exactly u/255. Allow 1/255 tolerance for
        # decoder color-conversion differences (different FFmpeg builds / chroma siting).
        tol = 1.0 / 255.0
        ok = float(diff.max()) <= tol
        ok_all &= ok
        print(f"[{i}] {cam:32s} shape={im_a.shape} dtype={im_a.dtype} | max|pyav-tc|={float(diff.max()):.5f} "
              f"mean={float(diff.mean()):.5f} -> {'OK' if ok else 'FAIL'}")
    act_eq = np.array_equal(np.asarray(a["action"]), np.asarray(b["action"]))
    ts_a = a.get("timestamp"); ts_b = b.get("timestamp")
    ts_eq = (ts_a is None and ts_b is None) or np.array_equal(np.asarray(ts_a), np.asarray(ts_b))
    ok_all &= act_eq and ts_eq
    print(f"[{i}] action identical={act_eq} timestamp identical={ts_eq}")

print(f"\n{'ALL OK' if ok_all else 'MISMATCHES FOUND'}")
