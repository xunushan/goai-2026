#!/usr/bin/env python3
"""Step 4: 每个任务按类占比抽 10 个验证 episode (分层随机抽样)。

读 <cluster_out>/task*/clusters.csv (episode->cluster 映射), 用最大余数法把
每任务 10 个验证额按类占比分配, 再在类内固定种子随机抽足配额。保证每类至少
覆盖 1 个 (当某类占比四舍五入后仍不足时, 由最大余数法补足)。

产物 (写入 <out>/):
    task<i>_val_manifest.csv   该任务每 episode 一行: cluster + split(val/train)
    val_manifest.csv           三任务合并清单 (val 段集中)

用法:
    python tools/gripper_val_split/gripper_select_val.py
    python tools/gripper_val_split/gripper_select_val.py --n-val 10 --seed 42
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve()
while not (ROOT / "CLAUDE.md").is_file() and ROOT.parent != ROOT:
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.gripper_val_split.gripper_common import alloc_proportional  # noqa: E402


def select_val(df_ep_cluster: pd.DataFrame, n_val: int, seed: int) -> pd.DataFrame:
    """对单任务: 类内分层随机抽样, 返回带 split 列的全量 episode 表。"""
    counts = df_ep_cluster.groupby("cluster").size()
    order_cls = sorted(counts.index.tolist())  # 稳定的类顺序
    quotas = dict(zip(order_cls, alloc_proportional([counts[c] for c in order_cls], n_val)))
    rng = np.random.default_rng(seed)
    val_ids = []
    for c in order_cls:
        members = df_ep_cluster.loc[df_ep_cluster["cluster"] == c, "episode_index"] \
            .sort_values().to_numpy()
        k = quotas[c]
        if len(members) < k:          # 防御: 类太小
            k = len(members)
        val_ids.extend(rng.choice(members, size=k, replace=False).tolist())
    val_ids = set(val_ids)
    out = df_ep_cluster.copy()
    out["split"] = np.where(out["episode_index"].isin(val_ids), "val", "train")
    out["quota"] = out["cluster"].map(quotas)
    return out.sort_values(["split", "cluster", "episode_index"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cluster-dir", default="outputs/gripper_cluster")
    ap.add_argument("--out", default="outputs/val_sets")
    ap.add_argument("--n-val", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cluster_root = Path(args.cluster_dir)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    manifests = []
    for tdir in sorted(cluster_root.glob("task*")):
        t = int(tdir.name.replace("task", ""))
        dfc = pd.read_csv(tdir / "clusters.csv")
        res = select_val(dfc, args.n_val, args.seed)
        res.to_csv(out / f"task{t}_val_manifest.csv", index=False)
        val = res[res["split"] == "val"]
        print(f"\n[task{t}] 验证集 {len(val)} 个:")
        for c, g in val.groupby("cluster", sort=True):
            eps = ",".join(map(str, g["episode_index"]))
            print(f"  cluster{c} (配额{len(g)}): ep {eps}")
        manifests.append(res)

    allm = pd.concat(manifests, ignore_index=True)
    allm.to_csv(out / "val_manifest.csv", index=False)
    print("\n已保存:", out)


if __name__ == "__main__":
    main()
