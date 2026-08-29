#!/usr/bin/env python3
"""训练集 frame_weight=1.5 区间的 episode 可视化工具。

从 data/lerobot_v30_ee.csv（每行一帧）读取 state 16 维向量与 frame_weight,
对每个任务选取若干 episode（默认每任务 1 个, 取该任务最小的 episode_index,
确定性可复现）, 绘制左右爪夹开度时序图, 并标记出所有 frame_weight=1.5
区间的起始与结束位置:

    - 阴影区      : frame_weight=1.5 的连续帧区间 (琥珀色浅填充)
    - 起始竖线    : 绿色实线 + 帧号标注 (start)
    - 结束竖线    : 红色虚线 + 帧号标注 (end)

每个 episode 可能有多个互不重叠的 1.5 区间（对应多次抓取周期的 hold 窗口,
见 tools/frame_weight.py 的权重窗口定义）。绘图风格/调色板与
utils/visualize_distribution_shift.py、tools/keyframe_detect.py 保持一致。

state 16 维结构:
    left_ee_pose(7)=xyz(3)+quat(4) + left_gripper(1)
    + right_ee_pose(7) + right_gripper(1)

用法:
    python tools/frame_weight_visualize.py
    python tools/frame_weight_visualize.py --tasks 0,2,8 --per-task 1
    python tools/frame_weight_visualize.py --out outputs/test
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CSV = ROOT / "data" / "lerobot_v30_ee.csv"
DEFAULT_TASKS = ROOT / "data" / "lerobot_v30_ee" / "meta" / "tasks.parquet"
DEFAULT_OUT = ROOT / "outputs" / "test"

GRIP_L = 7
GRIP_R = 15
WEIGHT_HIT = 1.5

# task_index -> 简短 slug（文件名用）
TASK_SLUGS: dict[int, str] = {
    0: "arrange_largest_number",
    1: "fold_clothes",
    2: "hang_mugs",
    3: "make_toast",
    4: "pack_objects_into_box",
    5: "pour_liquid_into_cup",
    6: "push_T",
    7: "sort_nesting_dolls_by_size",
    8: "stack_blocks",
    9: "stack_bowls",
    10: "headphone_laptop_stand",
    11: "sweep_blocks",
}

# 绘图风格与 visualize_distribution_shift.py / keyframe_detect.py 一致
INK = "#0b0b0b"; MUT = "#898781"; GRID = "#e1e0d9"; SURF = "#fcfcfb"
C_LEFT = "#2a78d6"    # 左臂
C_RIGHT = "#eb6834"   # 右臂
C_START = "#008300"   # 1.5 区间起始（绿, 实线）
C_END = "#e34948"     # 1.5 区间结束（红, 虚线）
FILL_1_5 = "#f2a900"  # 1.5 区间阴影填充（琥珀色, 低透明）
FILL_ALPHA = 0.14


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------

def load_tasks(tasks_parquet: Path) -> dict[int, str]:
    """读取 tasks.parquet -> {task_index: 完整指令}。"""
    import pyarrow.parquet as pq

    t = pq.read_table(str(tasks_parquet)).to_pandas()
    return {int(v): str(k) for k, v in t["task_index"].items()}


def load_frame_df(csv_path: Path) -> pd.DataFrame:
    """读取 CSV 的 episode/task/frame/state/frame_weight 列。"""
    return pd.read_csv(
        csv_path,
        usecols=["episode_index", "task_index", "frame_index",
                 "observation.state", "frame_weight"],
    )


def episode_data(
    df: pd.DataFrame, episode_index: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """取单个 episode 的 (state (T,16), frame_index (T,), weight (T,))。

    frame_index / weight 与 state 逐行对齐, 按 frame_index 升序。
    """
    ep = df[df["episode_index"] == episode_index].sort_values("frame_index")
    state = np.asarray(
        [np.fromstring(s.strip("[]"), sep=",", dtype=float)
         for s in ep["observation.state"]],
        dtype=float,
    )
    return (
        state,
        ep["frame_index"].to_numpy(dtype=int),
        ep["frame_weight"].to_numpy(dtype=float),
    )


# ---------------------------------------------------------------------------
# 1.5 区间检测
# ---------------------------------------------------------------------------

def weight_segments(weight: np.ndarray, frame: np.ndarray) -> list[tuple[int, int]]:
    """frame_weight==1.5 的连续帧区间 -> [(start_frame, end_frame), ...]。

    start/end 均为 frame_index 值（闭区间 [start, end]）。每个区间可能长度
    不同、数量多个, 按帧序排列。
    """
    mask = weight == WEIGHT_HIT
    segments = []
    i = 0
    n = len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            segments.append((int(frame[i]), int(frame[j])))
            i = j + 1
        else:
            i += 1
    return segments


# ---------------------------------------------------------------------------
# 绘图
# ---------------------------------------------------------------------------

def plot_episode_frame_weight(
    task_name: str,
    episode_index: int,
    frame: np.ndarray,
    state: np.ndarray,
    segments: list[tuple[int, int]],
    out_path: Path,
) -> None:
    """绘制单个 episode 的左右爪夹时序图, 标记全部 1.5 区间起止。"""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.patches import Patch
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "axes.edgecolor": MUT, "axes.labelcolor": INK, "axes.titlecolor": INK,
        "text.color": INK, "xtick.color": INK, "ytick.color": INK,
        "figure.facecolor": SURF, "axes.facecolor": SURF, "grid.color": GRID,
        "font.family": "sans-serif", "figure.dpi": 110,
    })

    fig, axes = plt.subplots(2, 1, figsize=(13, 6.8), sharex=True)
    fig.suptitle(
        f"{task_name}\n"
        f"episode {episode_index}  ({len(frame)} frames, "
        f"{len(segments)} × weight=1.5 segment)",
        fontsize=11, y=0.99,
    )

    for ax, name, idx, color in [
        (axes[0], "Left gripper", GRIP_L, C_LEFT),
        (axes[1], "Right gripper", GRIP_R, C_RIGHT),
    ]:
        ax.plot(frame, state[:, idx], color=color, ls="-", lw=1.3)
        ax.set_title(name, loc="left", fontsize=10)
        ax.set_ylabel("gripper (0 closed ~ 1 open)")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.25)
        for start, end in segments:
            ax.axvspan(start, end, color=FILL_1_5, alpha=FILL_ALPHA,
                       zorder=0)
            ax.axvline(start, color=C_START, ls="-", lw=1.2, alpha=0.95)
            ax.axvline(end, color=C_END, ls="--", lw=1.2, alpha=0.95)

    # 起止帧号标注（数据坐标, 双行交替置于曲线顶部之上防重叠）
    ax = axes[1]
    ax.set_ylim(-0.05, 1.15)
    for n_seg, (start, end) in enumerate(segments):
        y = 1.06 if n_seg % 2 == 0 else 1.12
        ax.text(start, y, f"{start}", color=C_START, fontsize=8,
                ha="center", va="bottom")
        ax.text(end, y, f"{end}", color=C_END, fontsize=8,
                ha="center", va="bottom")

    # 图例（置于图上方, 避开曲线与帧号标注）
    handles = [
        plt.Line2D([0], [0], color=C_LEFT, lw=1.5, label="left gripper"),
        plt.Line2D([0], [0], color=C_RIGHT, lw=1.5, label="right gripper"),
        plt.Line2D([0], [0], color=C_START, ls="-", lw=1.2, label="1.5 start"),
        plt.Line2D([0], [0], color=C_END, ls="--", lw=1.2, label="1.5 end"),
        Patch(facecolor=FILL_1_5, alpha=FILL_ALPHA,
              label="weight=1.5 segment"),
    ]
    fig.legend(handles=handles, fontsize=8, ncol=5, frameon=False,
               loc="upper center", bbox_to_anchor=(0.5, 0.955))
    axes[1].set_xlabel("frame index")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=str(DEFAULT_CSV), help="训练集 CSV 路径")
    parser.add_argument("--tasks", default="all",
                        help="要绘制的 task_index, 逗号分隔 (如 '2,8,9') 或 'all'")
    parser.add_argument("--per-task", type=int, default=1,
                        help="每任务绘制的最靠前 episode 数(确定性选择)")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="输出目录")
    args = parser.parse_args()

    tasks_parquet = DEFAULT_TASKS
    task_names = load_tasks(tasks_parquet) if tasks_parquet.is_file() else {}

    print(f"loading {args.csv} ...")
    df = load_frame_df(Path(args.csv))
    episodes_by_task = (
        df.groupby("task_index")["episode_index"]
        .apply(lambda s: np.sort(s.unique()))
        .to_dict()
    )

    selected_tasks = (
        list(episodes_by_task)
        if args.tasks == "all"
        else [int(x) for x in args.tasks.split(",")]
    )
    out_dir = Path(args.out)

    n_saved = 0
    summary = {}
    for ti in selected_tasks:
        if ti not in episodes_by_task:
            print(f"  ! task_index={ti} 不存在, 跳过")
            continue
        eps = episodes_by_task[ti]
        pick = eps[: args.per_task]
        slug = TASK_SLUGS.get(ti, f"task_{ti:03d}")
        name = task_names.get(ti, slug)

        for ep in pick:
            ep = int(ep)
            state, frame, weight = episode_data(df, ep)
            segments = weight_segments(weight, frame)
            out_path = out_dir / f"{slug}_ep{ep:03d}_frame_weight.png"
            plot_episode_frame_weight(name, ep, frame, state, segments, out_path)
            print(f"  task {ti:2d} ({slug:24s}) ep {ep:3d}: "
                  f"{len(frame):4d} frames, {len(segments)} 段 1.5 "
                  f"-> {out_path.name}")
            summary[f"task{ti}_{ep}"] = {
                "task_index": ti,
                "episode_index": ep,
                "n_frames": int(len(frame)),
                "segments": [{"start": s, "end": e} for s, e in segments],
            }
            n_saved += 1

    summary_path = out_dir / "frame_weight_visualize_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\ndone, {n_saved} figures -> {out_dir}")
    print(f"segment summary -> {summary_path}")


if __name__ == "__main__":
    main()
