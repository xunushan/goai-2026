#!/usr/bin/env python
"""Cross-version data correctness: lerobot 0.4.4 (py3.11) vs 0.6.0 (py3.12).

Dumps, for a fixed set of global indices (incl. episode boundaries read
version-independently from the frames parquet), each camera frame normalized to
uint8, the action block, and per-sample scalar fields (episode_index,
frame_index, timestamp). Run once under each lerobot version and diff the two
npz outputs with check_cross_version_compare.py.

Bounds the question "does switching lerobot version change the data?" while
keeping the decode backend constant (torchcodec + same FFmpeg libs), so any
difference is attributable to the lerobot version switch.
"""
from __future__ import annotations

import argparse
import glob
import json
import numpy as np
import torchcodec  # noqa: F401  import before lerobot/PIL (libjpeg soname)


CAMERAS = ["observation.images.cam_high", "observation.images.cam_left_wrist", "observation.images.cam_right_wrist"]
# Scattered "probe episodes" spread over the dataset for boundary coverage.
PROBE_EPISODES = [0, 1, 2, 10, 50, 100, 200, 299]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root", required=True)  # parent dir of repo
    p.add_argument("--repo-id", required=True)
    p.add_argument("--out", required=True)  # .npz
    return p.parse_args()


def episode_boundaries(dataset_root: str, repo: str, total_frames: int) -> list[int]:
    """Global index where each episode starts, read straight from the frames
    parquet (no lerobot internals), so both versions agree by construction."""
    import pyarrow.parquet as pq

    parquet_files = sorted(
        glob.glob(f"{dataset_root}/{repo}/data/chunk-*/file-*.parquet")
    )
    assert parquet_files, "no frames parquet found"
    ep_col = []
    for pf in parquet_files:
        t = pq.read_table(pf, columns=["episode_index"])
        ep_col.extend(t.column("episode_index").to_pylist())
    ep = np.asarray(ep_col)
    assert len(ep) == total_frames, f"frames parquet {len(ep)} != dataset len {total_frames}"
    starts = [0]
    starts.extend(int(i) for i in np.flatnonzero(np.diff(ep) != 0) + 1)
    return starts


def to_uint8(img) -> np.ndarray:
    a = np.asarray(img)
    if a.dtype == np.uint8:
        return a.copy()
    # normalized float in [0,1] (lerobot default non-uint8 output)
    return np.clip(np.round(a.astype(np.float32) * 255.0), 0, 255).astype(np.uint8)


def main() -> None:
    args = parse_args()
    import lerobot
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    repo = args.repo_id
    root = f"{args.dataset_root}/{repo}"
    fps = 25
    meta_path = f"{root}/meta/info.json"
    try:
        fps = int(json.load(open(meta_path)).get("fps", 25))
    except Exception:
        pass

    delta = {"action": [t / fps for t in range(50)]}
    for k in CAMERAS:
        delta[k] = [0.0]

    kwargs = dict(repo_id=repo, root=root, delta_timestamps=delta, video_backend="torchcodec")
    import inspect
    if "return_uint8" in inspect.signature(LeRobotDataset.__init__).parameters:
        kwargs["return_uint8"] = True

    ds = LeRobotDataset(**kwargs)
    n = len(ds)
    print(f"[{lerobot.__version__}] len={n}")

    starts = episode_boundaries(args.dataset_root, repo, n)
    num_ep = len(starts)
    # indices: global 0 & last, each probe episode first/last frame, mid scattered
    idxs = set()
    idxs.add(0)
    idxs.add(n - 1)
    for e in PROBE_EPISODES:
        if e >= num_ep:
            continue
        start = starts[e]
        end = starts[e + 1] if e + 1 < num_ep else n
        idxs.add(start)          # episode first frame
        idxs.add(end - 1)        # episode last frame
    for m in (1, 7, 23):
        idxs.add(int(n * m / 60))
    idxs = sorted(idxs)
    print(f"[{lerobot.__version__}] episodes={num_ep} idxs({len(idxs)})={idxs[:8]}...{idxs[-4:]}")

    data = {"lerobot_version": lerobot.__version__, "dataset_len": n, "idxs": idxs}
    for i in idxs:
        item = ds[i]
        for c in CAMERAS:
            data[f"i{i}_{c}"] = to_uint8(item[c])
        data[f"i{i}_action"] = np.asarray(item["action"], dtype=np.float32)
        for sf in ("episode_index", "frame_index", "timestamp"):
            v = np.asarray(item[sf]) if sf in item else np.asarray(-1)
            data[f"i{i}_{sf}"] = np.atleast_1d(v)
    np.savez(args.out, **data)
    print(f"[{lerobot.__version__}] wrote -> {args.out}")


if __name__ == "__main__":
    main()
