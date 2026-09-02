#!/usr/bin/env python3
"""生成 train/val 划分 JSON (数据位于 data/sim_lerobot_v30_ee/)。

单命令、确定性复现: 从原始 CSV 出发, 依次完成 1) 左右爪夹插值特征
2) KMeans+轮廓系数自动选 K 聚类 3) 按类占比分层随机抽样, 组装成划分 JSON。

JSON 结构 (字段可依评审反馈调整):
    dataset / n_train / n_val / n_val_per_task / seed{cluster,val_sampling}
    cluster{feature, algorithm, kmax}
    tasks["<task_index>"]: {
        task_index, instruction, n_episodes, n_train, n_val, n_clusters,
        cluster_stats["<cluster>"] = {n_episodes, n_train, n_val},
        train_episode_idx: [...], val_episode_idx: [...]
    }

用法:
    python tools/gripper_val_split/gripper_build_split.py
    python tools/gripper_val_split/gripper_build_split.py \
        --out-json data/sim_lerobot_v30_ee/train_val_split.json --n-val 10 --val-seed 42
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve()
while not (ROOT / "CLAUDE.md").is_file() and ROOT.parent != ROOT:
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.gripper_val_split.gripper_common import (  # noqa: E402
    episode_feature_L100_R100,
    load_grippers,
    load_tasks,
)
from tools.gripper_val_split.gripper_cluster import auto_kmeans  # noqa: E402
from tools.gripper_val_split.gripper_select_val import select_val  # noqa: E402


def _per_task_entry(t: int, g, instruction: str, n_val: int,
                    cluster_seed: int, val_seed: int, kmax: int) -> dict:
    g = g.sort_values("episode_index").reset_index(drop=True)
    X = np.stack([episode_feature_L100_R100(r["grip_L"], r["grip_R"])
                  for _, r in g.iterrows()])
    best_k, labels, _ = auto_kmeans(X, kmax, cluster_seed)

    dfc = g[["task_index", "episode_index"]].copy()
    dfc["cluster"] = labels
    split = select_val(dfc, n_val, val_seed)          # 列: task/cluster/episode/split/quota

    n_tr = int((split["split"] == "train").sum())
    n_vl = int((split["split"] == "val").sum())

    cluster_stats: dict[str, dict] = {}
    for c in sorted(split["cluster"].unique()):
        sub = split[split["cluster"] == c]
        cluster_stats[str(c)] = {
            "n_episodes": int(len(sub)),
            "n_train": int((sub["split"] == "train").sum()),
            "n_val": int((sub["split"] == "val").sum()),
        }

    val_rows = split[split["split"] == "val"].sort_values("episode_index")
    train_rows = split[split["split"] == "train"].sort_values("episode_index")
    return {
        "task_index": int(t),
        "instruction": instruction,
        "n_episodes": int(len(g)),
        "n_train": n_tr,
        "n_val": n_vl,
        "n_clusters": int(best_k),
        "cluster_stats": cluster_stats,
        "train_episode_idx": [int(e) for e in train_rows["episode_index"]],
        "val_episode_idx": [int(e) for e in val_rows["episode_index"]],
    }


def build_split(csv: Path, meta: Path, n_val: int = 10,
                cluster_seed: int = 0, val_seed: int = 42, kmax: int = 10) -> dict:
    task_names = load_tasks(meta)
    df_ep = load_grippers(csv)

    tasks = {}
    n_tr = n_vl = 0
    for t in sorted(df_ep["task_index"].unique()):
        g = df_ep[df_ep["task_index"] == t]
        entry = _per_task_entry(int(t), g, task_names.get(int(t), str(t)),
                                n_val, cluster_seed, val_seed, kmax)
        tasks[str(t)] = entry
        n_tr += entry["n_train"]
        n_vl += entry["n_val"]

    return {
        "dataset": Path(csv).name,
        "description": "sim 遥操数据 train/val 划分: 按左右爪夹波形聚类后, "
                       "每任务各类占比分层随机抽 n_val 个做验证集。",
        "n_train": int(n_tr),
        "n_val": int(n_vl),
        "n_val_per_task": int(n_val),
        "seed": {"cluster": int(cluster_seed), "val_sampling": int(val_seed)},
        "cluster": {
            "feature": "[L100,R100] 200维 (左右爪夹各插值到100点)",
            "algorithm": f"KMeans, K=argmax(轮廓系数) in [2,{kmax}]",
            "kmax": int(kmax),
        },
        "tasks": tasks,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/sim_lerobot_v30_ee.csv")
    ap.add_argument("--meta", default="data/sim_lerobot_v30_ee/meta/tasks.parquet")
    ap.add_argument("--out-json", default="data/sim_lerobot_v30_ee/train_val_split.json")
    ap.add_argument("--n-val", type=int, default=10)
    ap.add_argument("--cluster-seed", type=int, default=0)
    ap.add_argument("--val-seed", type=int, default=42)
    ap.add_argument("--kmax", type=int, default=10)
    args = ap.parse_args()

    data = build_split(Path(args.csv), Path(args.meta), args.n_val,
                       args.cluster_seed, args.val_seed, args.kmax)
    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"已写入 {out}")
    print(f"  train={data['n_train']}  val={data['n_val']}")
    for t, entry in data["tasks"].items():
        print(f"  task{t}: n_clusters={entry['n_clusters']} "
              f"cluster_stats={entry['cluster_stats']} val={entry['val_episode_idx']}")


if __name__ == "__main__":
    main()
