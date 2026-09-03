#!/usr/bin/env python
"""Compare two check_cross_version.py npz dumps (0.4.4 vs 0.6.0).

Report per-index per-camera max |a-b| on uint8 frames, action max abs diff /
exactness, scalar field equality, dataset len, and index sets.
"""
from __future__ import annotations

import argparse
import numpy as np

CAMERAS = ["observation.images.cam_high", "observation.images.cam_left_wrist", "observation.images.cam_right_wrist"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("a")  # 0.4.4 npz
    p.add_argument("b")  # 0.6.0 npz
    p.add_argument("--tol", type=float, default=1.0, help="uint8 gray-level tolerance")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    A = np.load(args.a)
    B = np.load(args.b)
    print(f"A={A['lerobot_version']} len={A['dataset_len']} idxs={len(A['idxs'])}")
    print(f"B={B['lerobot_version']} len={B['dataset_len']} idxs={len(B['idxs'])}")
    len_ok = int(A["dataset_len"]) == int(B["dataset_len"])
    idxs_a = list(A["idxs"]); idxs_b = list(B["idxs"])
    idx_ok = idxs_a == idxs_b
    print(f"dataset len equal: {len_ok} | index set equal: {idx_ok}")

    ok_all = True
    for i in A["idxs"]:
        row_ok = True
        notes = []
        for c in CAMERAS:
            ka, kb = f"i{i}_{c}", f"i{i}_{c}"
            if ka not in A or kb not in B:
                notes.append(f"{c}: MISSING")
                row_ok = False
                continue
            a = A[ka]; b = B[kb]
            if a.shape != b.shape:
                notes.append(f"{c}: SHAPE {a.shape} vs {b.shape}")
                row_ok = False
                continue
            diff = np.abs(a.astype(np.int16) - b.astype(np.int16))
            md = int(diff.max())
            if md > args.tol:
                row_ok = False
                notes.append(f"{c}: max|diff|={md} > tol")
            else:
                notes.append(f"{c}: max|diff|={md}")
        a_act, b_act = A[f"i{i}_action"], B[f"i{i}_action"]
        if a_act.shape != b_act.shape:
            notes.append(f"action SHAPE {a_act.shape} vs {b_act.shape}")
            row_ok = False
        else:
            ad = float(np.abs(a_act - b_act).max())
            if not np.array_equal(a_act, b_act):
                row_ok = False
                notes.append(f"action max|diff|={ad:.3e} (NOT bit-equal)")
            else:
                notes.append("action bit-equal")
        # scalar fields
        for sf in ("episode_index", "frame_index", "timestamp"):
            a = np.asarray(A[f"i{i}_{sf}"]); b = np.asarray(B[f"i{i}_{sf}"])
            eq = np.array_equal(a, b)
            if not eq:
                row_ok = False
                notes.append(f"{sf}: {a} vs {b}")
        ok_all &= row_ok
        print(f"i={i:6d} {'OK ' if row_ok else 'FAIL'} | " + " | ".join(notes))

    print(f"\n{'ALL OK (0.4.4 == 0.6.0 within tol)' if (ok_all and len_ok and idx_ok) else 'MISMATCHES FOUND'}")


if __name__ == "__main__":
    main()
