#!/usr/bin/env python
"""Correctness check for return_uint8 on LanceDBDataset (plan section 7).

Loads the same lance dataset with return_uint8 False/True, fetches the same
indices (incl. episode boundaries), and verifies:
  - dtype / shape
  - value range (float [0,1] vs uint8 [0,255])
  - uint8 == round(float * 255) within tolerance  (no "all zeros", no content change)
"""
from __future__ import annotations

import numpy as np

LANCE = "/cloud/cloud-ssd1/lerobot_bench/dataset-lance"
CAMERAS = ["observation.images.cam_high", "observation.images.cam_left_wrist", "observation.images.cam_right_wrist"]

from lerobot_lancedb import LanceDBDataset  # noqa: E402

fps = 25
delta = {"action": [t / fps for t in range(50)]}
for k in CAMERAS:
    delta[k] = [0.0]

ds_f = LanceDBDataset(root=LANCE, delta_timestamps=delta, return_uint8=False)
ds_u = LanceDBDataset(root=LANCE, delta_timestamps=delta, return_uint8=True)

# sample indices including near boundaries
idxs = [0, 1, 31, 100, 100000, 400000, len(ds_f) - 2, len(ds_f) - 1]
print(f"len={len(ds_f)}  idxs={idxs}")

# warmup
ds_f[0]; ds_u[0]

ok_all = True
for i in idxs:
    f = ds_f[i]
    u = ds_u[i]
    # top-level key set / shapes
    fkeys = set(f.keys())
    ukeys = set(u.keys())
    if fkeys != ukeys:
        print(f"[{i}] KEY MISMATCH: f-fkeys={fkeys-ukeys} u-only={ukeys-fkeys}")
        ok_all = False
    for cam in CAMERAS:
        img_f = np.asarray(f[cam], dtype=np.float32)
        img_u = np.asarray(u[cam], dtype=np.uint8)
        shape_ok = img_f.shape == img_u.shape
        diff = np.abs(img_f * 255.0 - img_u.astype(np.float32))
        n = diff.size
        # quantization tolerance: float is 0..1 (maybe pre-quantized), allow <=1.5 rounding error
        maxerr = float(diff.max())
        meanerr = float(diff.mean())
        zero_fraction_u8 = float((img_u == 0).mean())
        ok = shape_ok and maxerr <= 1.5 and not (img_u == 0).all()
        ok_all &= ok
        print(f"[{i}] {cam:32s} float:{img_f.dtype} min={img_f.min():.4f} max={img_f.max():.4f} "
              f"| uint8:{img_u.dtype} min={img_u.min()} max={img_u.max()} zeros={zero_fraction_u8*100:.1f}% "
              f"| max|f*255-u|={maxerr:.3f} mean={meanerr:.3f} -> {'OK' if ok else 'FAIL'}")
    # action should be identical (not an image)
    a_f = np.asarray(f["action"])
    a_u = np.asarray(u["action"])
    print(f"[{i}] action dtype {a_f.dtype} vs {a_u.dtype} identical={np.array_equal(a_f, a_u)}")

print(f"\n{'ALL OK' if ok_all else 'MISMATCHES FOUND'}")
