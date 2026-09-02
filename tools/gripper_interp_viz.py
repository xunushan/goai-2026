#!/usr/bin/env python3
"""Step 1: 左右爪夹(gripper)数据插值可视化 —— 插值前后对比确认。

背景
----
data/sim_lerobot_v30_ee.csv 为遥操数据（每行一帧）。state 16 维结构:
    left_ee_pose(7) + left gripper(1, idx=7) + right_ee_pose(7) + right gripper(1, idx=15)
每个任务 100 个 episode, 时长不一。为聚类, 需把每个 episode 左右爪夹时间序列
时间归一化为每臂 100 维。

本脚本为每个任务生成两类图:
    1) *_interp_examples.png  从该任务中按长度分位挑选若干 episode, 每行一个:
       左列=插值前(原始按帧, L 实线 / R 虚线), 右列=插值后(每臂 100 维等间距采样)。
       用于目测插值是否忠实保留开合波形。
    2) *_interp_overlay.png   该任务全部 100 个 episode 插值到 100 维后的叠画
       (左臂 / 右臂两子图), 用于整体确认波形覆盖与干净度。

用法:
    python tools/gripper_interp_viz.py
    python tools/gripper_interp_viz.py --csv data/sim_lerobot_v30_ee.csv \
        --out outputs/gripper_interp_viz --n-examples 6 --seed 0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# state 16 维常量 (与 tools/episode_state_insight.py 一致)
# ---------------------------------------------------------------------------
GRIP_L = 7    # 左臂 gripper (0 闭 ~ 1 开)
GRIP_R = 15   # 右臂 gripper

C_L = "#1f77b4"   # 左臂
C_R = "#ff7f0e"   # 右臂
N_DIM = 100       # 每臂插值目标维数


def load_tasks(meta_path: Path) -> dict[int, str]:
    """读取 tasks.parquet -> {task_index: 指令文本} (指令在 index, task_index 在列)。"""
    df = pd.read_parquet(meta_path)
    return {int(row["task_index"]): str(idx) for idx, row in df.iterrows()}


def load_grippers(csv: Path) -> pd.DataFrame:
    """读 CSV, 每 episode 汇总为一行: L/R 原始 gripper 数组 + 长度。

    返回列: task_index, episode_index, length, grip_L, grip_R
    """
    df = pd.read_csv(csv)
    # 直接按逗号切分字符串化 list, 取第 GRIP_L / GRIP_R 个 token (末 token 带 "]")
    def _col(idx: int):
        out = []
        for s in df["observation.state"]:
            out.append(float(s.split(",")[idx].rstrip("]")))
        return np.asarray(out)
    gL = _col(GRIP_L)
    gR = _col(GRIP_R)
    df["grip_L"] = gL
    df["grip_R"] = gR
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


def interp_100(x: np.ndarray) -> np.ndarray:
    """把任意长度一维序列等间距时间归一化到 N_DIM=100 维(np.interp, 保留端点)。"""
    n = len(x)
    if n == 1:
        return np.full(N_DIM, x[0])
    src = np.linspace(0, n - 1, n)
    dst = np.linspace(0, n - 1, N_DIM)
    return np.interp(dst, src, x)


def plot_episode_pair(ax_before, ax_after, ep: dict, t: int):
    """在左右两个 axes 上画单 episode 的插值前 / 插值后曲线。"""
    L, R = ep["grip_L"], ep["grip_R"]
    L100, R100 = interp_100(L), interp_100(R)

    fr = np.arange(len(L))
    ax_before.plot(fr, L, color=C_L, lw=1.0, label="Left gripper", zorder=3)
    ax_before.plot(fr, R, color=C_R, lw=1.0, ls="--", label="Right gripper", zorder=3)
    ax_before.set_ylim(-0.05, 1.05)

    b100 = np.linspace(0, N_DIM - 1, N_DIM)
    ax_after.plot(b100, L100, color=C_L, lw=1.0, zorder=3)
    ax_after.plot(b100, R100, color=C_R, lw=1.0, ls="--", zorder=3)
    ax_after.set_ylim(-0.05, 1.05)


def make_figs(df_ep: pd.DataFrame, task_names: dict[int, str], n_examples: int, out: Path, seed: int):
    rng = np.random.default_rng(seed)
    for t, g in df_ep.groupby("task_index"):
        g = g.sort_values("episode_index")
        leng = g["length"].to_numpy()
        # 按长度分位挑选样本覆盖短/中/长
        qs = np.linspace(0, 100, n_examples + 2)[1:-1]
        idxs = []
        for q in qs:
            sel = int(np.argsort(np.abs(np.percentile(leng, q) - leng))[0])
            if sel not in idxs:
                idxs.append(sel)
        sel_rows = g.iloc[idxs].to_dict("records")

        slug = f"task{t}_{task_names.get(t, str(t)).lower().replace(' ', '_')[:24]}"

        # ---- 图1: 单 episode 插值前后对比 (示例) ----
        nrow = len(sel_rows)
        fig, axes = plt.subplots(nrow, 2, figsize=(12, 1.9 * nrow), squeeze=False,
                                 gridspec_kw={"width_ratios": [1.35, 1]})
        for r, ep in enumerate(sel_rows):
            axb, axa = axes[r]
            plot_episode_pair(axb, axa, ep, t)
            axb.set_title("原始(按帧)", fontsize=9)
            axa.set_title(f"插值后({N_DIM} 维)", fontsize=9)
            for ax in (axb, axa):
                ax.tick_params(labelsize=8)
                ax.grid(alpha=0.25)
            axa.set_xlabel("采样点 0~99", fontsize=8)
            axa.set_yticks([0, 1])
            axb.set_ylabel(f"ep{ep['episode_index']} len={ep['length']}", fontsize=8)
            axb.set_xlabel("帧", fontsize=8)
            axb.set_yticks([0, 1])
        # 左上角加图例一次
        axes[0, 0].legend(fontsize=8, loc="upper left", framealpha=0.6)
        fig.suptitle(f"[{t}] {task_names.get(t, '')}", fontsize=11, y=0.995)
        fig.tight_layout()
        p = out / f"{slug}_interp_examples.png"
        fig.savefig(p, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print("saved", p)

        # ---- 图2: 全部 100 episode 插值后叠画 ----
        Lm = np.stack([interp_100(x) for x in g["grip_L"]])  # (100,100)
        Rm = np.stack([interp_100(x) for x in g["grip_R"]])
        xx = np.arange(N_DIM)
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
        for ax, M, name, c in ((axes[0], Lm, "Left", C_L), (axes[1], Rm, "Right", C_R)):
            for row in M:
                ax.plot(xx, row, color=c, alpha=0.20, lw=0.7)
            ax.plot(xx, M.mean(0), color="black", lw=1.8, label="mean")
            ax.set_ylim(-0.05, 1.05)
            ax.set_title(f"{name} gripper 全部 {len(M)} ep 插值叠画", fontsize=10)
            ax.set_xlabel("采样点 0~99", fontsize=9)
            ax.set_yticks([0, 1])
            ax.grid(alpha=0.25)
            ax.legend(fontsize=8)
        fig.suptitle(f"[{t}] {task_names.get(t, '')} (全部 {len(g)} ep)", fontsize=11)
        fig.tight_layout()
        p = out / f"{slug}_interp_overlay.png"
        fig.savefig(p, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print("saved", p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/sim_lerobot_v30_ee.csv")
    ap.add_argument("--meta", default="data/sim_lerobot_v30_ee/meta/tasks.parquet")
    ap.add_argument("--out", default="outputs/gripper_interp_viz")
    ap.add_argument("--n-examples", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    task_names = load_tasks(Path(args.meta))
    df_ep = load_grippers(Path(args.csv))
    print("episodes:", len(df_ep), "| tasks:", df_ep["task_index"].nunique())
    print("episodes/task:\n", df_ep.groupby("task_index").size().to_string())
    make_figs(df_ep, task_names, args.n_examples, out, args.seed)


if __name__ == "__main__":
    main()
