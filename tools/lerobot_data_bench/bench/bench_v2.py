#!/usr/bin/env python
"""OpenPI-aligned LeRobot data loading benchmark (v2).

Follows docs/LeRobot_v3数据加载性能后续测试方案.md:

- real sampler : torch DataLoader shuffle=True (RandomSampler), seeded
                 (openpi training calls create_data_loader(config, shuffle=True))
- real collate  : openpi.training.data_loader._collate_fn semantics
                 jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0), *items)
- warmup 10 batches, timed 100 batches
- records peak RSS (self + DataLoader workers) via psutil
- camera_frames_per_s = samples_per_s * 3
- --thread-limit : OMP/MKL/OPENBLAS/RAYON=1 + torch.set_num_threads(1) (Lance oversubscription check)
- --return-uint8 : pass return_uint8=True when the dataset class supports it
- --preload-action : preload the whole action column into RAM and serve the delta
                     (action_horizon) window from it instead of per-sample parquet
                     row gathers; bit-identical (light sentinel verifies). num_workers
                     must be 0 (in-process reader cache; item4 seam A prototype).

One process = one independent run. The shell driver loops for repeats and aggregates medians.
"""
from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
import os
import pathlib
import threading
import time
import types

import numpy as np
import torch

# Import torchcodec before lerobot/PIL/cv2 so the correct libjpeg/libav sonames
# win (see torchcodec-libjpeg-conflict). The launching shell must already have
# set LD_LIBRARY_PATH to the env's ffmpeg libs (or an fflib of soname symlinks).
import torchcodec  # noqa: E402,F401


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--repo-id", required=True)
    p.add_argument("--backend", choices=["lerobot", "lance"], default="lerobot")
    p.add_argument("--lance-path", default=None)
    p.add_argument("--video-backend", default="torchcodec")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--return-uint8", action="store_true")
    p.add_argument("--thread-limit", action="store_true")
    p.add_argument("--action-horizon", type=int, default=50)
    p.add_argument("--image-horizon", type=int, default=1)
    p.add_argument("--image-keys", default="observation.images.cam_high,observation.images.cam_left_wrist,observation.images.cam_right_wrist")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--warmup-batches", type=int, default=10)
    p.add_argument("--num-batches", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--episodes", default=None,
                   help="Comma list or 'A-B' range of episode indices to load (lerobot 0.6.0 'episodes' arg). "
                        "Default = all.")
    p.add_argument("--decoder-cache", type=int, default=None,
                   help="Bound lerobot's torchcodec VideoDecoderCache to N entries with LRU eviction "
                        "(closes evicted file handles). None = current unbounded cache.")
    p.add_argument("--preload-action", action="store_true",
                   help="Preload the whole 'action' column and serve delta windows from RAM "
                        "(item4 seam A prototype; requires --num-workers 0).")
    p.add_argument("--out", default=None)
    return p.parse_args()


# --- openpi collate: jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0), *items) ---
def _map_struct(func, *vals):
    v0 = vals[0]
    if isinstance(v0, dict):
        return {k: _map_struct(func, *[v[k] for v in vals]) for k in v0}
    if isinstance(v0, (list, tuple)):
        return type(v0)(_map_struct(func, *vs) for vs in zip(*vals))
    return func(*vals)


def openpi_collate(items):
    return _map_struct(lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0), *items)


def worker_init_fn(worker_id: int) -> None:
    torch.set_num_threads(1)


class LruVideoDecoderCache:
    """Bounded-LRU decoder cache for lerobot's torchcodec path.

    Mirrors lerobot.datasets.video_utils.VideoDecoderCache (video_path -> (decoder, fh))
    but keeps at most ``maxsize`` decoders, evicting least-recently-used entries and
    closing their file handles. Replaces the module-level ``_default_decoder_cache``
    so no installed package is modified.
    """

    def __init__(self, maxsize: int):
        from collections import OrderedDict
        self._cache: "OrderedDict[str, tuple]" = OrderedDict()
        self._lock = threading.Lock()
        self._maxsize = maxsize

    def get_decoder(self, video_path: str):
        import importlib.util
        if importlib.util.find_spec("torchcodec"):
            from torchcodec.decoders import VideoDecoder
        else:
            raise ImportError("torchcodec is required but not available.")
        video_path = str(video_path)
        with self._lock:
            item = self._cache.pop(video_path, None)
            if item is not None:
                self._cache[video_path] = item  # mark most-recently-used
                return item[0]
            import fsspec
            fh = fsspec.open(video_path).__enter__()
            decoder = VideoDecoder(fh, seek_mode="approximate")
            self._cache[video_path] = (decoder, fh)
            if len(self._cache) > self._maxsize:
                _, (_, evicted_fh) = self._cache.popitem(last=False)
                evicted_fh.close()
            return decoder

    def clear(self):
        with self._lock:
            for _, fh in self._cache.values():
                fh.close()
            self._cache.clear()

    def size(self) -> int:
        with self._lock:
            return len(self._cache)


def build_dataset(args) -> tuple[object, dict]:
    import inspect

    fps = 25
    meta_path = pathlib.Path(args.dataset_root) / args.repo_id / "meta" / "info.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        fps = meta.get("fps", 25)

    delta = {"action": [t / fps for t in range(args.action_horizon)]}
    img_keys = [k for k in args.image_keys.split(",") if k.strip()]
    if img_keys:
        img_h = args.image_horizon if args.image_horizon > 0 else 1
        for k in img_keys:
            delta[k] = [t / fps for t in range(img_h)]

    full_root = str(pathlib.Path(args.dataset_root) / args.repo_id)

    episodes = None
    meta_extra = ""
    if args.episodes:
        if "-" in args.episodes and "," not in args.episodes:
            a, b = (int(x) for x in args.episodes.split("-", 1))
            episodes = list(range(a, b + 1))
        else:
            episodes = [int(x) for x in args.episodes.split(",")]

    if args.backend == "lerobot":
        # Local-only loading: skip hub revision sync so local datasets load as-is.
        # Patch get_safe_version -> identity on whichever lerobot layout is present
        # (0.6.0: lerobot.datasets.*; 0.4.4 pi05_openpi: direct lerobot/datasets/*,
        # no dataset_metadata module at all, so patching is skipped there).
        import importlib as _ilib

        def _identity_version(repo_id, version):
            return version

        for _cand in (
            "lerobot.datasets.dataset_metadata",
            "lerobot.datasets.lerobot_dataset",
            "lerobot.common.datasets.dataset_metadata",
            "lerobot.common.datasets.lerobot_dataset",
        ):
            try:
                _mod = _ilib.import_module(_cand)
            except ModuleNotFoundError:
                continue
            if hasattr(_mod, "get_safe_version"):
                _mod.get_safe_version = _identity_version

        try:
            import lerobot.datasets.lerobot_dataset as _m
        except ModuleNotFoundError as exc:
            if exc.name != "lerobot.datasets":
                raise
            import lerobot.common.datasets.lerobot_dataset as _m
        cls = _m.LeRobotDataset
        kwargs = dict(
            repo_id=args.repo_id,
            root=full_root,
            delta_timestamps=delta,
            video_backend=args.video_backend,
        )
        if episodes is not None and "episodes" in inspect.signature(cls.__init__).parameters:
            kwargs["episodes"] = episodes
            meta_extra = f" episodes={len(episodes)}[{episodes[0]}-{episodes[-1]}]"
        else:
            meta_extra = ""
        if "return_uint8" in inspect.signature(cls.__init__).parameters:
            kwargs["return_uint8"] = args.return_uint8
        t0 = time.perf_counter()
        ds = cls(**kwargs)
        init_s = time.perf_counter() - t0
        kind = f"LeRobotDataset(v{getattr(ds, 'version', '?')}, {args.video_backend}){meta_extra}"
    else:
        from lerobot_lancedb import LanceDBDataset
        cls = LanceDBDataset
        kwargs = dict(root=args.lance_path, delta_timestamps=delta)
        if "return_uint8" in inspect.signature(cls.__init__).parameters:
            kwargs["return_uint8"] = args.return_uint8
        t0 = time.perf_counter()
        ds = cls(**kwargs)
        init_s = time.perf_counter() - t0
        kind = f"{cls.__name__}"

    return ds, {"init_s": init_s, "kind": kind, "fps": fps, "delta_keys": list(delta)}


def _eq(a, b) -> bool:
    if isinstance(a, torch.Tensor):
        return bool(torch.equal(a, b))
    return bool(np.array_equal(a, b))


def enable_action_preload(ds) -> dict:
    """Seam A (item4): preload the whole ``action`` column and serve delta windows from RAM.

    The dominant non-decode cost in the openpi-aligned path is the per-sample
    ``action_horizon``-row parquet gather (~3.26 ms/sample of ~9.4 ms; see
    profile_060_getitem / 汇总文档 §6). We monkeypatch the lerobot DatasetReader
    instance method ``_query_hf_dataset`` so the ``action`` key is read from a
    column-level cache (row order == relative HF row index for the unfiltered
    dataset this bench uses). Everything else stays on the original code path:
    window clamp/pad semantics live in ``_get_query_indices`` (untouched), video
    keys are skipped the same way, and non-action non-video delta keys defer to
    the original gather => outputs are bit-identical to the default path.

    A light correctness sentinel runs before the timed region: default vs cached
    action window on fixed rows (incl. episode-tail rows whose window clamps to
    the episode's last frame). Raises on any mismatch.

    Returns meta fields (cache build seconds, sentinel result). Requires
    num_workers=0 (the MethodType override is not spawn-picklable).
    """
    reader = None
    for attr in ("reader", "_reader", "_dataset_reader"):
        if getattr(ds, attr, None) is not None:
            reader = getattr(ds, attr)
            break
    if reader is None:
        raise RuntimeError("preload-action: no lerobot reader found on dataset ('reader'/'_reader')")
    hf = reader.hf_dataset
    if hf is None:
        eager = getattr(ds, "_ensure_hf_dataset_loaded", None)
        if eager is not None:
            eager()
            hf = reader.hf_dataset
    if hf is None:
        lazy = getattr(reader, "load_and_activate", None)
        if lazy is None:
            raise RuntimeError("preload-action: cannot load hf_dataset "
                               "(no _ensure_hf_dataset_loaded / load_and_activate)")
        lazy()
        hf = reader.hf_dataset

    orig_query_hf = reader._query_hf_dataset

    # --- sentinel rows: default path vs cached path must agree ---
    n = len(hf)
    cand = {0, min(1, n - 1), n // 4, n // 2, (3 * n) // 4, n - 2, n - 1}
    if n > 50:
        cand.add(n - 1 - 49)  # episode-tail window: clamps & repeats last frame
        cand.add(n - 1 - 24)
    rows = sorted(r for r in cand if 0 <= r < n)
    refs = []
    for r in rows:
        base = hf[r]  # raw single row (cheap, no video decode)
        qidx, _pad = reader._get_query_indices(int(base["index"]), int(base["episode_index"]))
        refs.append((qidx, orig_query_hf(qidx)))

    # --- build full action column cache (one columnar read) ---
    t0 = time.perf_counter()
    colvals = hf["action"][list(range(n))] if n else []
    if isinstance(colvals, torch.Tensor):
        cache = colvals
    elif isinstance(colvals, np.ndarray):
        cache = torch.from_numpy(np.asarray(colvals))
    elif len(colvals) and isinstance(colvals[0], torch.Tensor):
        cache = torch.stack(list(colvals))
    else:
        cache = torch.from_numpy(np.stack([np.asarray(v) for v in colvals]))
    build_s = time.perf_counter() - t0

    rel_map = reader._absolute_to_relative_idx
    vid_keys = set(reader._meta.video_keys) if getattr(reader, "_meta", None) is not None else set()

    def _query_hf_with_cache(self, query_indices):
        out = {}
        for key, q_idx in query_indices.items():
            if key == "action":
                rel = q_idx if rel_map is None else [rel_map[i] for i in q_idx]
                out[key] = cache[list(rel)]
            elif key in vid_keys:
                continue  # mirror original: video keys are handled by _query_videos
            else:
                out[key] = orig_query_hf({key: q_idx})[key]
        return out

    reader._query_hf_dataset = types.MethodType(_query_hf_with_cache, reader)

    # --- verify cached path == default path on sentinel rows ---
    ok = True
    first_bad = None
    for qidx, ref in refs:
        pre = reader._query_hf_dataset(qidx)
        for k, v in ref.items():
            if not _eq(v, pre[k]):
                ok = False
                first_bad = first_bad or f"{k}@{qidx}"
    if not ok:
        raise RuntimeError(f"preload-action sentinel MISMATCH (first: {first_bad}); aborting")
    print(f"  preload-action: cached action == default on sentinel rows ({len(refs)} rows, "
          f"build={build_s:.2f}s, cache={tuple(cache.shape)} {cache.dtype})")
    return {
        "action_preload": True,
        "action_preload_build_s": round(build_s, 4),
        "action_preload_sentinel_n": len(refs),
        "action_preload_sentinel_ok": True,
    }


class RssMonitor:
    """Samples RSS of self + all children (DataLoader workers) every 0.1s, tracks peak."""

    def __init__(self):
        self._stop = False
        self.peak_self_gb = 0.0
        self.peak_workers_gb = 0.0
        self._th = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._th.start()

    def stop(self):
        self._stop = True
        self._th.join(timeout=3)

    def _run(self):
        import resource
        try:
            import psutil
        except Exception:  # e.g. lerobot060 conda env has no psutil
            psutil = None
        while not self._stop:
            try:
                if psutil is not None:
                    p = psutil.Process(os.getpid())
                    s = p.memory_info().rss
                    w = sum(c.memory_info().rss for c in p.children(recursive=True))
                else:
                    # ru_maxrss (KB on Linux) is monotonic peak self RSS.
                    s = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
                    w = 0
            except Exception:
                s = w = 0
            self.peak_self_gb = max(self.peak_self_gb, s / 1e9)
            self.peak_workers_gb = max(self.peak_workers_gb, w / 1e9)
            time.sleep(0.1)


def main() -> None:
    args = parse_args()

    mp_context = None
    if args.num_workers > 0:
        if args.backend == "lance":
            from lerobot_lancedb import lance_mp_context
            mp_context = lance_mp_context()
        else:
            # openpi uses spawn for the torch data loader
            mp_context = multiprocessing.get_context("spawn")

    ds, meta = build_dataset(args)

    if args.decoder_cache is not None:
        # Replace the unbounded module-level decoder cache with a bounded LRU one
        # (torchcodec path only; pyav has no cache). No package modification.
        from lerobot.datasets import video_utils
        video_utils._default_decoder_cache = LruVideoDecoderCache(args.decoder_cache)
        meta["decoder_cache_maxsize"] = args.decoder_cache
        print(f"  decoder_cache: bounded LRU maxsize={args.decoder_cache}")

    if hasattr(ds, "_ensure_hf_dataset_loaded"):
        t0 = time.perf_counter()
        ds._ensure_hf_dataset_loaded()
        meta["hf_preload_s"] = time.perf_counter() - t0

    if args.preload_action:
        if args.num_workers > 0:
            raise SystemExit("--preload-action requires --num-workers 0 "
                             "(in-process reader cache; MethodType override is not spawn-picklable)")
        meta.update(enable_action_preload(ds))

    logging.basicConfig(level=logging.WARNING)
    print(f"=== {meta['kind']} nw={args.num_workers} uint8={args.return_uint8} threadlimit={args.thread_limit} "
          f"len={len(ds)} fps={meta['fps']} | init={meta['init_s']:.2f}s")

    g = torch.Generator()
    g.manual_seed(args.seed)
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,  # openpi train.py passes shuffle=True (RandomSampler)
        drop_last=True,
        collate_fn=openpi_collate,
        generator=g,
        persistent_workers=(args.num_workers > 0),
        multiprocessing_context=mp_context,
        worker_init_fn=worker_init_fn if args.thread_limit else None,
    )

    monitor = RssMonitor()
    monitor.start()

    it = iter(loader)
    for _ in range(args.warmup_batches):  # warmup 10
        next(it)
    batches = []
    t0_all = time.perf_counter()
    for _ in range(args.num_batches):  # timed 100
        t0 = time.perf_counter()
        next(it)
        batches.append(time.perf_counter() - t0)
    total_s = time.perf_counter() - t0_all
    monitor.stop()

    arr = np.array(batches)
    n = len(arr)
    samples_per_s = n * args.batch_size / total_s
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
        "samples_per_s": samples_per_s,
        "camera_frames_per_s": samples_per_s * 3,
        "peak_self_rss_gb": monitor.peak_self_gb,
        "peak_workers_rss_gb": monitor.peak_workers_gb,
        "peak_total_rss_gb": monitor.peak_self_gb + monitor.peak_workers_gb,
    }
    print(f"  mean={stats['mean_s']*1e3:.0f}ms p50={stats['p50_s']*1e3:.0f} p95={stats['p95_s']*1e3:.0f} "
          f"| {samples_per_s:.0f} samples/s | peak total RSS={stats['peak_total_rss_gb']:.1f}GB")

    meta.update({"batch_stats": stats, "args": vars(args)})
    if args.out:
        with open(args.out, "w") as f:
            json.dump(meta, f, indent=2, default=str)
        print(f"wrote -> {args.out}")


if __name__ == "__main__":
    main()
