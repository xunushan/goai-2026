"""Time the FULL data pull path (decode + CPU->GPU sharded device_put) on an idle GPU.

Mirrors train.py exactly (create_data_loader with NamedSharding on the real device,
num_workers configurable). No training compute runs, so any per-item cost here is
decode + collate + device_put alone.

usage: pull_bench.py --workers W --num-items N
"""
import argparse, time
import dataclasses

import jax

import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.sharding as sharding


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--num-items", type=int, default=96)
    args = ap.parse_args()

    cfg = _config.get_config("goai_pi05_p1")
    cfg = dataclasses.replace(cfg, num_workers=args.workers)
    mesh = sharding.make_mesh(cfg.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))

    t0 = time.time()
    loader = _data_loader.create_data_loader(cfg, sharding=data_sharding, shuffle=False)
    it = iter(loader)
    t_build = time.time() - t0
    print(f"build={t_build:.1f}s workers={args.workers} device={jax.devices()[0]}")

    pulls = []
    for i in range(args.num_items):
        s = time.perf_counter()
        batch = next(it)
        jax.block_until_ready(jax.tree.leaves(batch)[0])  # ensure H2D copy done
        pulls.append(time.perf_counter() - s)
    pulls = pulls[2:]  # drop ramp
    mean = sum(pulls) / len(pulls)
    print(f"PULL workers={args.workers} items={len(pulls)} mean_item={mean*1000:.1f}ms "
          f"throughput={1/mean:.1f} items/s")


if __name__ == "__main__":
    main()
