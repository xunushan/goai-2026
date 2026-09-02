#!/usr/bin/env python3
"""Step 1: 左右爪夹(gripper)数据插值可视化 —— 插值前后对比确认。

data/sim_lerobot_v30_ee.csv 为遥操数据（每行一帧）, state 16 维:
    left_ee_pose(7) + left gripper(1, idx=7) + right_ee_pose(7) + right gripper(1, idx=15)
每个任务 100 个 episode, 时长不一。为聚类, 把每个 episode 左右爪夹时间序列
时间归一化为每臂 100 维 (见 gripper_common.interp_100)。

每任务生成两类图:
    1) *_interp_examples.png  按长度分位挑选若干 episode, 每行一个:
       左列=插值前(原始按帧, L 实线 / R 虚线), 右列=插值后(每臂 100 维)。
    2) *_interp_overlay.png   全部 100 个 episode 插值到 100 维后的叠画。

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

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # 保证仓库根在 sys.path

from tools.gripper_common import (  # noqa: E402
    C_L,
    C_R,
    N_DIM,
    interp_100,
    load_grippers,
    load_tasks,
    setup_cjk_font,
)

setup_cjk_font()  # 需在首个 figure 前调用


def plot_episode_pair(ax_before, ax_after, ep: dict):
    """在左右两个 axes 上画单 episode 的插值前 / 插值后曲线。"""
    L, R = ep["grip_L"], ep["grip_R"]
    L100, R100 = interp_100(L), interp_100(R)

    fr = np.arange(len(L))
    ax_before.plot(fr, L, color=C_L, lw=1.0, label="左臂", zorder=3)
    ax_before.plot(fr, R, color=C_R, lw=1.0, ls="--", label="右臂", zorder=3)
    ax_before.set_ylim(-0.05, 1.05)

    b100 = np.linspace(0, N_DIM - 1, N_DIM)
    ax_after.plot(b100, L100, color=C_L, lw=1.0, zorder=3)
    ax_after.plot(b100, R100, color=C_R, lw=1.0, ls="--", zorder=3)
    ax_after.set_ylim(-0.05, 1.05)


def make_figs(df_ep: pd.DataFrame, task_names: dict[int, str], n_examples: int, out: Path):
    for t, g in df_ep.groupby("task_index"):
        g = g.sort_values("episode_index")
        leng = g["length"].to_numpy()
        slug = f"task{t}_{task_names.get(int(t), str(t)).lower().replace(' ', '_')[:24]}"
        title = f"[task{t}] {task_names.get(int(t), str(t))}"

        # ---- 图1: 单 episode 插值前后对比 (按长度分位抽 n_examples 个) ----
        qs = np.linspace(0, 100, n_examples + 2)[1:-1]
        idxs = []
        for q in qs:
            sel = int(np.argsort(np.abs(np.percentile(leng, q) - leng))[0])
            if sel not in idxs:
                idxs.append(sel)
        sel_rows = g.iloc[idxs].to_dict("records")

        nrow = len(sel_rows)
        fig, axes = plt.subplots(nrow, 2, figsize=(12, 1.9 * nrow), squeeze=False,
                                 gridspec_kw={"width_ratios": [1.35, 1]})
        for r, ep in enumerate(sel_rows):
            axb, axa = axes[r]
            plot_episode_pair(axb, axa, ep)
            axb.set_title("插值前: 原始按帧", fontsize=9)
            axa.set_title(f"插值后: 每臂 {N_DIM} 维", fontsize=9)
            for ax in (axb, axa):
                ax.tick_params(labelsize=8)
                ax.grid(alpha=0.25)
                ax.set_yticks([0, 1])
            axb.set_ylabel(f"ep{ep['episode_index']} len={ep['length']}", fontsize=8)
            axb.set_xlabel("帧", fontsize=8)
            axa.set_xlabel("采样点 0~99", fontsize=8)
        axes[0, 0].legend(fontsize=8, loc="upper left", framealpha=0.6)
        fig.suptitle(title, fontsize=11, y=0.995)
        fig.tight_layout()
        p = out / f"{slug}_interp_examples.png"
        fig.savefig(p, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print("saved", p)

        # ---- 图2: 全部 100 episode 插值后叠画 ----
        Lm = np.stack([interp_100(x) for x in g["grip_L"]])  # (n_ep, 100)
        Rm = np.stack([interp_100(x) for x in g["grip_R"]])
        xx = np.arange(N_DIM)
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
        for ax, M, name, c in ((axes[0], Lm, "左臂", C_L), (axes[1], Rm, "右臂", C_R)):
            for row in M:
                ax.plot(xx, row, color=c, alpha=0.20, lw=0.7)
            ax.plot(xx, M.mean(0), color="black", lw=1.8, label="均值")
            ax.set_ylim(-0.05, 1.05)
            ax.set_title(f"{name} gripper · 全部 {len(M)} ep 插值叠画", fontsize=10)
            ax.set_xlabel("采样点 0~99", fontsize=9)
            ax.set_yticks([0, 1])
            ax.grid(alpha=0.25)
            ax.legend(fontsize=8)
        fig.suptitle(title, fontsize=11)
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
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    task_names = load_tasks(Path(args.meta))
    df_ep = load_grippers(Path(args.csv))
    print("episodes:", len(df_ep), "| tasks:", df_ep["task_index"].nunique())
    print("episodes/task:\n", df_ep.groupby("task_index").size().to_string())
    make_figs(df_ep, task_names, args.n_examples, out)


if __name__ == "__main__":
    main()
