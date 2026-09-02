#!/usr/bin/env python3
"""Step 2/3: 对任务的 100 个 episode 按左右爪夹波形聚类 (类别数自动确定)。

可复用 API (模块级函数):
    silhouette_scores(X, kmax, seed) -> {k: 轮廓系数}
    auto_kmeans(X, kmax, seed)       -> (best_k, labels, sils)
    plot_silhouette(...)             画 K vs 轮廓系数, 标注自动选的 K
    plot_2d(...)                     PCA 与 t-SNE 两个 2D 投影, 按类着色
    plot_cluster_curves(...)         每类一张左右爪夹曲线叠画 + 类均值
    cluster_and_plot(...)            单任务编排: 聚类 + 出图 + 存 clusters.csv

特征: 每 episode -> [L100(100), R100(100)] 共 200 维 (gripper_common.interp_100)。
方法: KMeans, K 在 [2, kmax] 取轮廓系数最大 (K>kmax 时系数持续下降即选到边界)。

产物 (out/task<idx>/): silhouette.png / cluster_2d.png / cluster_<c>_curves.png /
    clusters.csv (episode -> cluster 映射, 供 gripper_select_val.py 分层抽样)。

CLI 用法:
    python tools/gripper_val_split/gripper_cluster.py --task 0      # 单任务
    python tools/gripper_val_split/gripper_cluster.py --tasks 0 1 2 # 多任务
"""

from __future__ import annotations

import argparse
import sys
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

ROOT = Path(__file__).resolve()
while not (ROOT / "CLAUDE.md").is_file() and ROOT.parent != ROOT:
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.gripper_val_split.gripper_common import (  # noqa: E402
    N_DIM,
    episode_feature_L100_R100,
    interp_100,
    load_grippers,
    load_tasks,
    setup_cjk_font,
)

setup_cjk_font()

TAB = plt.get_cmap("tab10").colors  # 聚类配色 (tab10, 按类取模)
XX = np.arange(N_DIM)               # 曲线 x 轴 (采样点 0~99)


# ---------------------------------------------------------------------------
# 聚类 (纯算法, 不画图)
# ---------------------------------------------------------------------------
def silhouette_scores(X: np.ndarray, kmax: int, seed: int = 0) -> dict[int, float]:
    """对 KMeans 的 K=2..kmax 逐一算轮廓系数。"""
    n = len(X)
    kmax = max(2, min(kmax, n - 1))
    return {k: float(silhouette_score(X, KMeans(n_clusters=k, n_init=10,
                                                random_state=seed).fit(X).labels_))
            for k in range(2, kmax + 1)}


def auto_kmeans(X: np.ndarray, kmax: int, seed: int = 0):
    """KMeans + 轮廓系数自动选 K。返回 (best_k, labels, sils)。"""
    sils = silhouette_scores(X, kmax, seed)
    best_k = max(sils, key=sils.get)
    labels = KMeans(n_clusters=best_k, n_init=10, random_state=seed).fit(X).labels_
    return best_k, labels, sils


# ---------------------------------------------------------------------------
# 画图 (聚类结果可视化)
# ---------------------------------------------------------------------------
def plot_silhouette(t: int, title: str, sils: dict[int, float], best_k: int,
                    out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    ks = sorted(sils)
    ax.plot(ks, [sils[k] for k in ks], marker="o", color="#2c7fb8")
    ax.axvline(best_k, color="gray", ls="--", lw=1)
    ax.annotate(f"选 K={best_k} (轮廓系数 {sils[best_k]:.3f})", xy=(best_k, sils[best_k]),
                xytext=(best_k + 0.4, min(sils.values())),
                fontsize=9, arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.set_xlabel("类别数 K"); ax.set_ylabel("轮廓系数 (越大越好)")
    ax.set_xticks(ks)
    ax.set_title(f"[task{t}] 自动选 K · {title}", fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = out / "silhouette.png"
    fig.savefig(p, dpi=140, bbox_inches="tight"); plt.close(fig)
    return p


def plot_2d(X: np.ndarray, labels: np.ndarray, best_k: int, t: int, title: str,
            out: Path, seed: int = 0) -> Path:
    n = len(X)
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
        ax.set_title(tag + (f" ({sum(ev) * 100:.1f}% 方差)" if ev is not None else ""),
                     fontsize=10)
        ax.set_xlabel(tag + " 1"); ax.set_ylabel(tag + " 2")
        ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="best")
    fig.suptitle(f"[task{t}] {title} · KMeans K={best_k} 聚类的二维分布", fontsize=11)
    fig.tight_layout()
    p = out / "cluster_2d.png"
    fig.savefig(p, dpi=140, bbox_inches="tight"); plt.close(fig)
    return p


def plot_cluster_curves(g: pd.DataFrame, labels: np.ndarray, best_k: int, t: int,
                        title: str, out: Path) -> list[Path]:
    """每类一张: 成员左右爪夹 100 维曲线叠画 + 类均值 (L 实线 / R 虚线)。"""
    saved = []
    for c in range(best_k):
        members = g[labels == c]
        Lm = np.stack([interp_100(x) for x in members["grip_L"]])
        Rm = np.stack([interp_100(x) for x in members["grip_R"]])
        col = TAB[c % 10]
        fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.0))
        for ax, M, arm, ls in ((axes[0], Lm, "左臂", "-"), (axes[1], Rm, "右臂", "--")):
            for row in M:
                ax.plot(XX, row, color=col, ls=ls, alpha=0.18, lw=0.8)
            ax.plot(XX, M.mean(0), color="black", lw=2.0, label="类均值")
            ax.set_ylim(-0.05, 1.05)
            ax.set_title(f"类{c} · {arm} ({len(M)} ep)", fontsize=10)
            ax.set_xlabel("采样点 0~99", fontsize=9)
            ax.set_yticks([0, 1]); ax.grid(alpha=0.3); ax.legend(fontsize=8)
        fig.suptitle(f"[task{t}] {title} · 类{c} 成员左右爪夹曲线 (n={len(members)})",
                     fontsize=10.5)
        fig.tight_layout()
        p = out / f"cluster_{c}_curves.png"
        fig.savefig(p, dpi=140, bbox_inches="tight"); plt.close(fig)
        saved.append(p)
    return saved


# ---------------------------------------------------------------------------
# 单任务编排
# ---------------------------------------------------------------------------
def cluster_and_plot(t: int, g: pd.DataFrame, title: str, out: Path,
                     kmax: int = 10, seed: int = 0) -> dict:
    """对单任务聚类并出全部图 + 存 clusters.csv。返回结果 dict。"""
    g = g.sort_values("episode_index").reset_index(drop=True)
    X = np.stack([episode_feature_L100_R100(r["grip_L"], r["grip_R"])
                  for _, r in g.iterrows()])
    best_k, labels, sils = auto_kmeans(X, kmax, seed)

    plot_silhouette(t, title, sils, best_k, out)
    plot_2d(X, labels, best_k, t, title, out, seed)
    plot_cluster_curves(g, labels, best_k, t, title, out)

    pd.DataFrame({"task_index": int(t), "episode_index": g["episode_index"].to_numpy(),
                  "cluster": labels}).sort_values("cluster") \
        .to_csv(out / "clusters.csv", index=False)

    counts = pd.Series(labels).value_counts().sort_index()
    print(f"[task{t}] best K={best_k}, silhouette={sils[best_k]:.3f}, "
          f"counts={counts.to_dict()}")
    return {"task": int(t), "best_k": best_k, "silhouette": sils[best_k],
            "counts": counts.to_dict()}


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
        cluster_and_plot(t, g, task_names.get(t, str(t)), t_out, args.kmax, args.seed)
    print("done")


if __name__ == "__main__":
    main()
