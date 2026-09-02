#!/usr/bin/env python
"""Benchmark LeRobot dataset batch loading time.

Replicates the data access pattern of
  RoboDojo/XPolicyLab/policy/Pi_05/openpi/src/openpi/training/data_loader.py

- LeRobotDataset(repo_id, delta_timestamps={key: [t/fps for t in range(action_horizon)]},
                 video_backend="pyav")   # action_sequence_keys only -> images decode 1 frame/cam
- torch.utils.data.DataLoader(batch_size=32, collate_fn=np-stack)

Usage:
  python bench_data_loader.py \
      --dataset-root /cloud/cloud-ssd1/lerobot_data \
      --repo-id real_lerobot_v30_joint \
      --backend lerobot|lance \
      --action-horizon 50 --batch-size 32 --num-workers 0 --num-batches 50 \
      [--image-horizon 0] [--video-backend pyav] [--lance-path PATH] \
      [--single-sample] [--out out.json]
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root", required=True, help="HF_LEROBOT_HOME (parent of dataset dir)")
    p.add_argument("--repo-id", required=True)
    p.add_argument("--backend", choices=["lerobot", "lance"], default="lerobot")
    p.add_argument("--lance-path", default=None, help="Path to converted lance dataset (backend=lance)")
    p.add_argument("--action-key", default="action", help="key holding the action feature")
    p.add_argument("--action-horizon", type=int, default=50)
    p.add_argument("--image-keys", default="observation.images.cam_high,observation.images.cam_left_wrist,observation.images.cam_right_wrist",
                   help="comma-separated; empty string = don't decode image sequences (data_loader.py default)")
    p.add_argument("--image-horizon", type=int, default=0, help="delta horizon for image keys; 0 = only current frame")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--num-batches", type=int, default=50)
    p.add_argument("--warmup-batches", type=int, default=1)
    p.add_argument("--video-backend", default="pyav")
    p.add_argument("--single-sample", action="store_true", help="also time N raw __getitem__ calls")
    p.add_argument("--num-samples", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="path to write JSON results")
    return p.parse_args()


def get_lerobot_classes():
    """Import LeRobotDataset handling the path change across versions (mirrors data_loader.py)."""
    try:
        import lerobot.datasets.lerobot_dataset as _m
    except ModuleNotFoundError as exc:
        if exc.name != "lerobot.datasets":
            raise
        import lerobot.common.datasets.lerobot_dataset as _m
    return _m.LeRobotDataset


def get_lance_classes():
    """Import the Lance loader (new LanceDBDataset; old LeRobotLanceDataset is pre-0.3)."""
    from lerobot_lancedb import LanceDBDataset  # noqa: N813

    return LanceDBDataset


def build_dataset(args) -> tuple[object, dict]:
    fps = 25
    # Read fps from meta/info.json when available (any backend).
    import pathlib

    meta_path = pathlib.Path(args.dataset_root) / args.repo_id / "meta" / "info.json"
    if meta_path.exists():
        import json as _json

        meta = _json.loads(meta_path.read_text())
        fps = meta.get("fps", 25)

    action_key = args.action_key
    delta = {action_key: [t / fps for t in range(args.action_horizon)]}

    img_keys = [k for k in args.image_keys.split(",") if k.strip()]
    if img_keys:
        img_h = args.image_horizon if args.image_horizon > 0 else 1
        for k in img_keys:
            delta[k] = [t / fps for t in range(img_h)]

    import logging

    logging.basicConfig(level=logging.INFO)

    # lerobot treats `root` as the FULL path to the dataset dir, not its parent.
    full_root = str(pathlib.Path(args.dataset_root) / args.repo_id)

    if args.backend == "lerobot":
        cls = get_lerobot_classes()
        t0 = time.perf_counter()
        ds = cls(
            args.repo_id,
            root=full_root,
            delta_timestamps=delta,
            video_backend=args.video_backend,
        )
        init_s = time.perf_counter() - t0
        kind = f"LeRobotDataset(v{getattr(ds, 'version', '?')})"
        return ds, {"init_s": init_s, "kind": kind, "fps": fps, "delta": delta}

    else:  # lance
        cls = get_lance_classes()
        if args.lance_path is None:
            raise ValueError("--lance-path is required for backend=lance")
        t0 = time.perf_counter()
        ds = cls(root=args.lance_path, delta_timestamps=delta)
        init_s = time.perf_counter() - t0
        return ds, {"init_s": init_s, "kind": f"{cls.__name__}", "fps": fps, "delta": delta}


def numpy_collate(items):
    return {k: np.stack([np.asarray(x[k]) for x in items], axis=0) for k in items[0]}


def main() -> None:
    args = parse_args()

    ds, meta = build_dataset(args)

    # Pre-load the HF dataset in the main process so forked DataLoader workers
    # share it via copy-on-write instead of each allocating its own copy (avoids OOM).
    if hasattr(ds, "_ensure_hf_dataset_loaded"):
        t0 = time.perf_counter()
        ds._ensure_hf_dataset_loaded()
        print(f"  hf_dataset preload: {time.perf_counter()-t0:.1f}s (n_rows={len(ds.hf_dataset)})")

    print(f"=== backend={args.backend} kind={meta['kind']} len(dataset)={len(ds)} fps={meta['fps']}")
    print(f"=== delta_timestamps keys: {list(meta['delta'])} (horizon action={args.action_horizon}, img={args.image_horizon})")

    # Optional per-sample timing (raw __getitem__, no batching).
    if args.single_sample:
        idxs = np.random.RandomState(args.seed).randint(0, len(ds), size=args.num_samples)
        ts = []
        t0 = time.perf_counter()
        sample = ds[int(idxs[0])]  # warmup incl. lazy hf_dataset load
        warm_s = time.perf_counter() - t0
        for i in idxs:
            t0 = time.perf_counter()
            ds[int(i)]
            ts.append(time.perf_counter() - t0)
        print(f"  single-sample: warmup={warm_s:.3f}s mean={statistics.mean(ts)*1e3:.2f}ms "
              f"p50={sorted(ts)[len(ts)//2]*1e3:.2f}ms p95={sorted(ts)[int(len(ts)*0.95)]*1e3:.2f}ms")
        meta["single_sample_ms"] = {"mean": statistics.mean(ts) * 1e3,
                                    "p50": sorted(ts)[len(ts) // 2] * 1e3,
                                    "p95": sorted(ts)[int(len(ts) * 0.95)] * 1e3,
                                    "warmup_s": warm_s}

    g = torch.Generator()
    g.manual_seed(args.seed)
    mp_context = None
    if args.backend == "lance" and args.num_workers > 0:
        from lerobot_lancedb import lance_mp_context

        mp_context = lance_mp_context()
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        drop_last=True,
        collate_fn=numpy_collate,
        generator=g,
        persistent_workers=(args.num_workers > 0),
        multiprocessing_context=mp_context,
    )

    it = iter(loader)
    # Warmup (skips lazy-load / first-file effects).
    for _ in range(args.warmup_batches):
        next(it)
    batches = []
    t0_all = time.perf_counter()
    for _ in range(args.num_batches):
        t0 = time.perf_counter()
        b = next(it)
        batches.append(time.perf_counter() - t0)
    total_s = time.perf_counter() - t0_all

    arr = np.array(batches)
    n = len(arr)
    stats = {
        "num_batches": n,
        "batch_size": args.batch_size,
        "mean_s": float(arr.mean()),
        "std_s": float(arr.std(ddof=1)) if n > 1 else 0.0,
        "p50_s": float(np.median(arr)),
        "p90_s": float(np.percentile(arr, 90)),
        "p95_s": float(np.percentile(arr, 95)),
        "min_s": float(arr.min()),
        "max_s": float(arr.max()),
        "total_s": total_s,
        "batches_per_s": n / total_s,
        "samples_per_s": n * args.batch_size / total_s,
        "frames_per_s": n * args.batch_size / total_s,  # one 'frame'/sample in this access pattern
    }
    print(f"=== batch_size={args.batch_size} num_workers={args.num_workers} num_batches={n}")
    print(f"  mean={stats['mean_s']*1e3:.1f}ms  std={stats['std_s']*1e3:.1f}ms  "
          f"p50={stats['p50_s']*1e3:.1f}ms  p95={stats['p95_s']*1e3:.1f}ms  "
          f"min={stats['min_s']*1e3:.1f}ms  max={stats['max_s']*1e3:.1f}ms")
    print(f"  throughput: {stats['samples_per_s']:.0f} samples/s = {stats['batches_per_s']:.2f} batches/s")

    meta.update({"batch_stats": stats, "args": vars(args)})
    if args.out:
        with open(args.out, "w") as f:
            json.dump(meta, f, indent=2, default=str)
        print(f"wrote -> {args.out}")


if __name__ == "__main__":
    main()
