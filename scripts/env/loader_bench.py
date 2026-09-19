"""Isolated dataloader throughput bench for goai_pi05 on RoboDojo sim.

Replicates the train.py data-loading path (create_torch_dataset + transform_dataset +
TorchDataLoader) but pulls raw items only (framework='pytorch', no GPU device_put),
so timing reflects decode + CPU transforms + collate, not GPU compute.

usage: loader_bench.py --workers W --video-backend {torchcodec,pyav} --action-preload {0,1} --num-items N
"""
import argparse, dataclasses, resource, time

import jax

import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.shared.array_typing as at


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--video-backend", default="torchcodec")
    ap.add_argument("--action-preload", type=int, default=1)
    ap.add_argument("--num-items", type=int, default=96)
    args = ap.parse_args()

    cfg = _config.get_config("goai_pi05_p1")
    cfg = dataclasses.replace(
        cfg,
        data=dataclasses.replace(
            cfg.data,
            video_backend=args.video_backend,
            action_preload=bool(args.action_preload),
        ),
    )
    data_config = cfg.data.create(cfg.assets_dirs, cfg.model)

    t0 = time.time()
    dataset = _data_loader.create_torch_dataset(data_config, cfg.model.action_horizon, cfg.model)
    dataset = _data_loader.transform_dataset(dataset, data_config, skip_norm_stats=False)
    t_build = time.time() - t0
    print(f"dataset frames={len(dataset)} build={t_build:.1f}s workers={args.workers} "
          f"backend={args.video_backend} preload={bool(args.action_preload)}")

    # sanity: one item dtype/shape (verify uint8 + 224 no-resize + structure)
    item = dataset[0]
    for cam, arr in item["observation"]["images"].items() if "observation" in item else []:
        pass
    print("item top keys:", list(item.keys()))
    imgs = item.get("images", item.get("observation", {}).get("images", {}))
    for cam, arr in imgs.items():
        print(f"  {cam}: shape={tuple(arr.shape)} dtype={getattr(arr, 'dtype', None)}")

    loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=1,
        num_workers=args.workers,
        num_batches=args.num_items,
        framework="pytorch",  # skip GPU device_put: pure CPU decode/transform timing
    )
    # warm up 4 items
    it = iter(loader)
    for _ in range(4):
        next(it)
    n = 0
    per_item = []
    t_start = time.time()
    for batch in it:
        n += 1
        per_item.append(time.perf_counter())
    elapsed = time.time() - t_start
    if per_item:
        deltas = [b - a for a, b in zip([t_start] + per_item[:-1], per_item)]
        deltas = deltas[2:]  # drop early ramp
    else:
        deltas = []
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # MiB (linux)
    mean = (sum(deltas) / len(deltas)) if deltas else float("nan")
    print(f"RESULT workers={args.workers} backend={args.video_backend} preload={bool(args.action_preload)} "
          f"items={n} wall={elapsed:.2f}s throughput={n/elapsed:.1f} items/s mean_item={mean*1000:.0f}ms "
          f"mainRSS_MiB={rss:.0f}")


if __name__ == "__main__":
    main()
