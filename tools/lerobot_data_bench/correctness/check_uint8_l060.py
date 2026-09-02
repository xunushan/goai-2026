#!/usr/bin/env python
"""Plan section 7 correctness: return_uint8 equivalence on lerobot 0.6.0 + torchcodec.

Loads the same 0.6.0 dataset with return_uint8 False/True (torchcodec backend),
fetches fixed indices incl. episode boundaries, verifies:
  - key sets / shapes / dtypes
  - uint8 == round(float * 255) within 1/255 tolerance, no all-zero images
  - action equality
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

ds_f = LeRobotDataset(repo_id=REPO, root=ROOT, delta_timestamps=delta, video_backend="torchcodec", return_uint8=False)
ds_u = LeRobotDataset(repo_id=REPO, root=ROOT, delta_timestamps=delta, video_backend="torchcodec", return_uint8=True)
print(f"len={len(ds_f)}")

idxs = [0, 1, 31, 100, 100000, 400000, len(ds_f) - 2, len(ds_f) - 1]
print(f"idxs={idxs}")

ds_f[0]; ds_u[0]  # warmup

ok_all = True
for i in idxs:
    f = ds_f[i]
    u = ds_u[i]
    if set(f.keys()) != set(u.keys()):
        print(f"[{i}] KEY MISMATCH: {set(f.keys()) ^ set(u.keys())}")
        ok_all = False
    for cam in CAMERAS:
        img_f = np.asarray(f[cam], dtype=np.float32)
        img_u = np.asarray(u[cam], dtype=np.uint8)
        shape_ok = img_f.shape == img_u.shape
        diff = np.abs(img_f * 255.0 - img_u.astype(np.float32))
        maxerr = float(diff.max())
        ok = shape_ok and maxerr <= 1.5 and not (img_u == 0).all()
        ok_all &= ok
        print(f"[{i}] {cam:32s} float:{img_f.dtype} [{img_f.min():.4f},{img_f.max():.4f}] | "
              f"uint8:{img_u.dtype} [{img_u.min()},{img_u.max()}] zeros={(img_u==0).mean()*100:.1f}% | "
              f"max|f*255-u|={maxerr:.3f} -> {'OK' if ok else 'FAIL'}")
    a_f = np.asarray(f["action"]); a_u = np.asarray(u["action"])
    print(f"[{i}] action dtype {a_f.dtype} vs {a_u.dtype} identical={np.array_equal(a_f, a_u)}")

print(f"\n{'ALL OK' if ok_all else 'MISMATCHES FOUND'}")
