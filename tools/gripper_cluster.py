#!/usr/bin/env python3
"""Step 2: 对单个任务的 100 个 episode 按左右爪夹波形聚类 (类别数自动确定)。

特征: 每个 episode -> [L100(100), R100(100)] 共 200 维 (见 gripper_common)。
聚类: KMeans, 类别数 K 在 [2, kmax] 里取轮廓系数(silhouette)最大者。
展示: 每任务输出到 <out>/task<idx>/:
    silhouette.png         K vs 轮廓系数 (说明为何选该 K)
    cluster_2d.png         PCA 与 t-SNE 两个 2D 投影, 按类着色 + 样本量标注
    cluster_<c>_curves.png 每类一张: 成员左右爪夹 100 维曲线叠画 + 类均值
    clusters.csv           episode_index -> cluster 映射 (供后续验证集抽取复用)

用法:
    python tools/gripper_cluster.py --task 0
    python tools/gripper_cluster.py --csv data/sim_lerobot_v30_ee.csv --tasks 0 1 2 --out outputs/gripper_cluster
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.gripper_common import (  # noqa: E402
    C_L,
    C_R,
    N_DIM,
    episode_feature_L100_R100,
    interp_100,
    load_grippers,
    load_tasks,
    setup_cjk_font,
)

setup_cjk_font()

TAB = plt.get_cmap("tab10").colors


def auto_kmeans(X: np.ndarray, kmax: int, seed: int):
    """KMeans + 轮廓系数自动选 K。返回 (best_k, labels, sil_scores{k: s})。"""
    n = len(X)
    kmax = max(2, min(kmax, n - 1))
    sils: dict[int, float] = {}
    for k in range(2, kmax + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)
        sils[k] = float(silhouette_score(X, km.labels_))
    best_k = max(sils, key=sils.get)
    best = KMeans(n_clusters=best_k, n_init=10, random_state=seed).fit(X)
    return best_k, best.labels_, sils


def run_task(t: int, g: pd.DataFrame, title: str, out: Path, kmax: int, seed: int):
    g = g.sort_values("episode_index").reset_index(drop=True)
    X = np.stack([episode_feature_L100_R100(r["grip_L"], r["grip_R"])
                  for _, r in g.iterrows()])
    n = len(X)
    best_k, labels, sils = auto_kmeans(X, kmax, seed)

    # ---- silhouette 曲线 ----
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    ks = sorted(sils)
    ax.plot(ks, [sils[k] for k in ks], marker="o", color="#2c7fb8")
    ax.axvline(best_k, color="gray", ls="--", lw=1)
    ax.annotate(f"选 K={best_k} (轮廓系数 {sils[best_k]:.3f})", xy=(best_k, sils[best_k]),
                xytext=(best_k + 0.4, sils[best_k] - 0.02), fontsize=9)
    ax.set_xlabel("类别数 K")
    ax.set_ylabel("轮廓系数 (越大越好)")
    ax.set_xticks(ks)
    ax.set_title(f"[task{t}] 自动选 K · {title}", fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = out / "silhouette.png"
    fig.savefig(p, dpi=140, bbox_inches="tight"); plt.close(fig); print("saved", p)

    # ---- 2D 投影 (PCA + t-SNE) ----
    pca = PCA(n_components=2, random_state=seed)
    Zp = pca.fit_transform(X)
    tsne = TSNE(n_components=2, init="pca", perplexity=min(30, max(5, n - 1)),
                random_state=seed, learning_rate="auto")
    Zt = tsne.fit_transform(X)
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.2))
    for ax, Z, tag, ev in ((axes[0], Zp, "PCA", pca.explained_variance_ratio_),
                           (axes[1], Zt, "t-SNE", None)):
        for c in range(best_k):
            m = labels == c
            ax.scatter(Z[m, 0], Z[m, 1], s=26, color=TAB[c % 10], alpha=0.85,
                       label=f"类{c} n={int(m.sum())}", edgecolor="white", lw=0.4)
        ax.set_title(tag + (f" ({sum(ev)*100:.1f}% 方差)" if ev is not None else ""), fontsize=10)
        ax.set_xlabel(tag + " 1"); ax.set_ylabel(tag + " 2")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="best")
    fig.suptitle(f"[task{t}] {title} · KMeans K={best_k} 聚类的二维分布", fontsize=11)
    fig.tight_layout()
    p = out / "cluster_2d.png"
    fig.savefig(p, dpi=140, bbox_inches="tight"); plt.close(fig); print("saved", p)

    # ---- 每类曲线图 ----
    xx = np.arange(N_DIM)
    for c in range(best_k):
        members = g[labels == c]
        Lm = np.stack([interp_100(x) for x in members["grip_L"]])
        Rm = np.stack([interp_100(x) for x in members["grip_R"]])
        col = TAB[c % 10]
        fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.0))
        for ax, M, arm, mcol in ((axes[0], Lm, "左臂", col), (axes[1], Rm, "右臂", col)):
            for row in M:
                ax.plot(xx, row, color=mcol, alpha=0.18, lw=0.8)
            ax.plot(xx, M.mean(0), color="black", lw=2.0, label="类均值")
            ax.set_ylim(-0.05, 1.05)
            ax.set_title(f"类{c} · {arm} ({len(M)} ep)", fontsize=10)
            ax.set_xlabel("采样点 0~99", fontsize=9)
            ax.set_yticks([0, 1])
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
        fig.suptitle(f"[task{t}] {title} · 类{c} 成员左右爪夹曲线 (n={len(members)})", fontsize=10.5)
        fig.tight_layout()
        p = out / f"cluster_{c}_curves.png"
        fig.savefig(p, dpi=140, bbox_inches="tight"); plt.close(fig); print("saved", p)

    # ---- CSV 映射 ----
    pd.DataFrame({
        "task_index": int(t), "episode_index": g["episode_index"].to_numpy(),
        "cluster": labels,
    }).sort_values("cluster").to_csv(out / "clusters.csv", index=False)
    print(f"[task{t}] best K={best_k}, silhouette={sils[best_k]:.3f}")
    return best_k, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/sim_lerobot_v30_ee.csv")
    ap.add_argument("--meta", default="data/sim_lerobot_v30_ee/meta/tasks.parquet")
    ap.add_argument("--out", default="outputs/gripper_cluster")
    ap.add_argument("--tasks", nargs="+", type=int, default=[0], help="要聚类的任务索引")
    ap.add_argument("--kmax", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    task_names = load_tasks(Path(args.meta))
    df_ep = load_grippers(Path(args.csv))
    out = Path(args.out)
    for t in args.tasks:
        g = df_ep[df_ep["task_index"] == t]
        if g.empty:
            raise SystemExit(f"task {t} 无数据")
        t_out = out / f"task{t}"
        t_out.mkdir(parents=True, exist_ok=True)
        title = task_names.get(t, str(t))
        run_task(t, g, title, t_out, args.kmax, args.seed)
    print("done")


if __name__ == "__main__":
    main()
