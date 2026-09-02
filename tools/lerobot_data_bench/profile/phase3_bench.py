#!/usr/bin/env python
"""Phase 3 decode-throughput benchmark: random-access 3-camera decode of a variant.

Mirrors the training decode cost model: each sample = 1 random frame from each of
3 cameras, decoded via torchcodec (same path lerobot uses: fsspec.open +
VideoDecoder seek_mode='approximate'). Reports samples/s, camera frames/s, and
per-sample p50/p90/p95.

Samples uniformly over each file's own frame range (files differ in content but the
decode cost is camera-independent, so throughput is comparable across variants).
"""
from __future__ import annotations

import argparse
import pathlib
import time

import numpy as np
import fsspec
from torchcodec.decoders import VideoDecoder


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--files", nargs="+", required=True)
    p.add_argument("--samples", type=int, default=500)
    p.add_argument("--warmup", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    decoders, nframes = [], []
    for vp in args.files:
        fh = fsspec.open(vp).__enter__()
        dec = VideoDecoder(fh, seek_mode="approximate")
        nf = int(dec.metadata.num_frames)
        decoders.append(dec)
        nframes.append(nf)
        print(f"  {pathlib.Path(vp).name}: nframes={nf}", flush=True)

    rng = np.random.RandomState(args.seed)

    def one_sample() -> float:
        t0 = time.perf_counter()
        for i, dec in enumerate(decoders):
            dec.get_frame_at(int(rng.randint(0, nframes[i])))
        return time.perf_counter() - t0

    for _ in range(args.warmup):
        one_sample()

    per = np.array([one_sample() for _ in range(args.samples)])
    total = float(per.sum())
    sps = args.samples / total
    cam_fps = sps * len(decoders)
    print(f"samples={args.samples} total={total:.1f}s "
          f"samples_per_s={sps:.1f} camera_frames_per_s={cam_fps:.1f} "
          f"per_sample p50={np.median(per) * 1e3:.1f}ms p90={np.percentile(per, 90) * 1e3:.1f}ms "
          f"p95={np.percentile(per, 95) * 1e3:.1f}ms", flush=True)
    with open("/tmp/phase3_bench.txt", "a") as f:
        f.write(f"{' '.join(str(pathlib.Path(x).name) for x in args.files)} | "
                f"{sps:.1f} sps {cam_fps:.1f} camfps | p50={np.median(per) * 1e3:.1f}ms "
                f"p95={np.percentile(per, 95) * 1e3:.1f}ms\n")


if __name__ == "__main__":
    main()
