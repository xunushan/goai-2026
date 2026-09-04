#!/usr/bin/env python3
"""real 遥操数据(real_lerobot_v30_ee)验证集划分驱动 —— sim 方法(gripper_val_split)的 real 应用。

与 sim 相同的方法链 (README.md) 应用到 real 数据:
    ① interp_viz  左右爪夹插值前后可视化
    ② cluster     左右爪夹200维特征 KMeans + 轮廓系数自动选 K, 出图 + clusters.csv
    ③ val         每任务按类占比分层随机抽 n_val 个作验证集
    ④ build       从 CSV 单命令组装 train/val 划分 JSON

与 sim 脚本的差异:
    - 默认 CSV/meta 指向 data/real_lerobot_v30_ee; 产物输出 *_real 目录避免覆盖 sim。
    - 轻量读列: 只读 task_index/episode_index/length/observation.state, 避免全列读入 629MB CSV。
    - --tasks 过滤: 单/多任务跑 (sim 脚本 build/viz 默认全任务)。

用法 (仓库根目录, lerobot env; 默认 task5 演示):
    python tools/gripper_val_split/gripper_real_split.py viz     --tasks 5
    python tools/gripper_val_split/gripper_real_split.py cluster --tasks 5
    python tools/gripper_val_split/gripper_real_split.py val     --tasks 5
    python tools/gripper_val_split/gripper_real_split.py build   --tasks 5
    全任务 (产出完整 JSON):
    python tools/gripper_val_split/gripper_real_split.py build --tasks 0 1 2 3 4 5 \
        --out-json data/real_lerobot_v30_ee/train_val_split.json

    # 每步可加 --csv/--meta/--n-val/--cluster-seed/--val-seed/--kmax 覆盖默认。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve()
while not (ROOT / "CLAUDE.md").is_file() and ROOT.parent != ROOT:
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.gripper_val_split.gripper_common import (  # noqa: E402
    GRIP_L,
    GRIP_R,
    episode_feature_L100_R100,
    load_tasks,
    setup_cjk_font,
)

setup_cjk_font()

D_CSV = "data/real_lerobot_v30_ee/real_lerobot_v30_ee.csv"
D_META = "data/real_lerobot_v30_ee/meta/tasks.parquet"
D_INTERP = "outputs/gripper_interp_viz_real"
D_CLUSTER = "outputs/gripper_cluster_real"
D_VAL = "outputs/val_sets_real"
D_JSON = "data/real_lerobot_v30_ee/train_val_split.json"

# 状态读取列 (轻量): state 字符串较大, 只取需要的列
_CSV_COLS = ["task_index", "episode_index", "length", "observation.state"]


def load_grippers_subset(csv: Path, tasks: list[int] | None = None) -> pd.DataFrame:
    """轻量读 CSV -> 每 episode 一行 (列: task_index/episode_index/length/grip_L/grip_R)。

    tasks=None 全任务; 否则只保留这些任务的行 (CSV 仍需整体扫描, 但不全列读入)。
    与 gripper_common.load_grippers 产出结构一致, 供聚类/viz/抽样复用。
    """
    df = pd.read_csv(csv, usecols=_CSV_COLS)
    if tasks is not None:
        df = df[df["task_index"].isin(tasks)]
    if df.empty:
        raise SystemExit(f"CSV 中无任务 {tasks}")

    def _col(idx: int) -> np.ndarray:
        return np.asarray([float(s.split(",")[idx].rstrip("]")) for s in df["observation.state"]])

    df["grip_L"] = _col(GRIP_L)
    df["grip_R"] = _col(GRIP_R)
    rows = []
    for (t, e), g in df.groupby(["task_index", "episode_index"]):
        rows.append({
            "task_index": int(t),
            "episode_index": int(e),
            "length": int(g["length"].iloc[0]),
            "grip_L": g["grip_L"].to_numpy(float),
            "grip_R": g["grip_R"].to_numpy(float),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# ① 插值可视化
# ---------------------------------------------------------------------------
def cmd_viz(args) -> None:
    from tools.gripper_val_split.gripper_interp_viz import make_figs

    task_names = load_tasks(Path(args.meta))
    df_ep = load_grippers_subset(Path(args.csv), args.tasks)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    make_figs(df_ep, task_names, args.n_examples, out)


# ---------------------------------------------------------------------------
# ② 聚类
# ---------------------------------------------------------------------------
def cmd_cluster(args) -> None:
    from tools.gripper_val_split.gripper_cluster import cluster_and_plot

    task_names = load_tasks(Path(args.meta))
    df_ep = load_grippers_subset(Path(args.csv), args.tasks)
    out_root = Path(args.out)
    for t in args.tasks:
        g = df_ep[df_ep["task_index"] == t]
        if g.empty:
            raise SystemExit(f"task {t} 无数据")
        t_out = out_root / f"task{t}"
        t_out.mkdir(parents=True, exist_ok=True)
        r = cluster_and_plot(int(t), g, task_names.get(int(t), str(t)), t_out,
                             args.kmax, args.cluster_seed)
        print(json.dumps(r, ensure_ascii=False))


# ---------------------------------------------------------------------------
# ③ 分层抽样 val
# ---------------------------------------------------------------------------
def cmd_val(args) -> None:
    from tools.gripper_val_split.gripper_select_val import select_val

    cluster_root = Path(args.cluster_dir)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    frames = []
    for t in args.tasks:
        dfc = pd.read_csv(cluster_root / f"task{t}" / "clusters.csv")
        res = select_val(dfc, args.n_val, args.val_seed)
        res.to_csv(out / f"task{t}_val_manifest.csv", index=False)
        val = res[res["split"] == "val"].sort_values(["cluster", "episode_index"])
        print(f"[task{t}] val n={len(val)}")
        for c, g in val.groupby("cluster", sort=True):
            print(f"  cluster{c} (配额{len(g)}): " + ",".join(map(str, g["episode_index"])))
        frames.append(res)
    pd.concat(frames, ignore_index=True).to_csv(out / "val_manifest.csv", index=False)


# ---------------------------------------------------------------------------
# ④ train/val split JSON
# ---------------------------------------------------------------------------
def _per_task_entry(t: int, g: pd.DataFrame, instruction: str, n_val: int,
                    cluster_seed: int, val_seed: int, kmax: int) -> dict:
    from tools.gripper_val_split.gripper_cluster import auto_kmeans
    from tools.gripper_val_split.gripper_select_val import select_val

    g = g.sort_values("episode_index").reset_index(drop=True)
    X = np.stack([episode_feature_L100_R100(r["grip_L"], r["grip_R"])
                  for _, r in g.iterrows()])
    best_k, labels, _ = auto_kmeans(X, kmax, cluster_seed)
    dfc = g[["task_index", "episode_index"]].copy()
    dfc["cluster"] = labels
    split = select_val(dfc, n_val, val_seed)
    n_tr = int((split["split"] == "train").sum())
    n_vl = int((split["split"] == "val").sum())
    cluster_stats = {}
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


def cmd_build(args) -> None:
    task_names = load_tasks(Path(args.meta))
    df_ep = load_grippers_subset(Path(args.csv), args.tasks)
    tasks = {}
    n_tr = n_vl = 0
    for t in sorted(df_ep["task_index"].unique()):
        g = df_ep[df_ep["task_index"] == t]
        entry = _per_task_entry(int(t), g, task_names.get(int(t), str(t)),
                                args.n_val, args.cluster_seed, args.val_seed, args.kmax)
        tasks[str(t)] = entry
        n_tr += entry["n_train"]
        n_vl += entry["n_val"]

    data = {
        "dataset": Path(args.csv).name,
        "description": "real 遥操数据 train/val 划分: 按左右爪夹波形聚类后, "
                       "每任务各类占比分层随机抽 n_val 个做验证集。",
        "n_train": int(n_tr),
        "n_val": int(n_vl),
        "n_val_per_task": int(args.n_val),
        "seed": {"cluster": int(args.cluster_seed), "val_sampling": int(args.val_seed)},
        "cluster": {
            "feature": "[L100,R100] 200维 (左右爪夹各插值到100点)",
            "algorithm": f"KMeans, K=argmax(轮廓系数) in [2,{args.kmax}]",
            "kmax": int(args.kmax),
        },
        "tasks": tasks,
    }
    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"已写入 {out}")
    print(f"  train={data['n_train']}  val={data['n_val']}")
    for t, entry in tasks.items():
        print(f"  task{t}: n_clusters={entry['n_clusters']} "
              f"cluster_stats={entry['cluster_stats']} val={entry['val_episode_idx']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["viz", "cluster", "val", "build"])
    ap.add_argument("--csv", default=D_CSV)
    ap.add_argument("--meta", default=D_META)
    ap.add_argument("--tasks", nargs="+", type=int, default=[5], help="任务索引 (默认 task5)")
    ap.add_argument("--n-val", type=int, default=10)
    ap.add_argument("--cluster-seed", type=int, default=0)
    ap.add_argument("--val-seed", type=int, default=42)
    ap.add_argument("--kmax", type=int, default=10)
    ap.add_argument("--n-examples", type=int, default=6, help="[viz] 插值示例数")
    ap.add_argument("--out", default=D_INTERP, help="[viz/cluster/val] 输出目录")
    ap.add_argument("--cluster-dir", default=D_CLUSTER, help="[val] 聚类产物目录")
    ap.add_argument("--out-json", default=D_JSON, help="[build] JSON 输出路径")
    args = ap.parse_args()

    # 每子命令的默认输出目录不同: 运行时按 cmd 覆盖 --out 默认
    if args.out == D_INTERP and args.cmd in ("cluster", "val"):
        args.out = {"cluster": D_CLUSTER, "val": D_VAL}[args.cmd]

    {"viz": cmd_viz, "cluster": cmd_cluster, "val": cmd_val, "build": cmd_build}[args.cmd](args)


if __name__ == "__main__":
    main()
