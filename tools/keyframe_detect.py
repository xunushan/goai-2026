#!/usr/bin/env python3
"""训练集 episode 的爪夹关键帧检测与可视化。

从 data/lerobot_v30_ee.csv（每行一帧）读取 state 16 维向量,
对每个 episode 的左右爪夹 (L=index7, R=index15) 检测关键帧, 并在时序图上
用竖线标出。绘图风格复用 utils/visualize_distribution_shift.py（grid/图例去重/
tight_layout 等）。检测代码只在本文件中, 不写入 visualize_distribution_shift.py。

关键帧定义（对每个爪夹, 一次"关-持-开"为一个 cycle, 周期从 1 降到保持再回到 1）:
    close_start  从打开(≈1)逐渐关闭的起点
    hold_start   水平保持段的起点（下降终点, 之后进入水平段）
    hold_end     水平保持段的终点（上升起点, 之后逐渐打开）
    open_full    从关闭完全打开回到(≈1)的点

state 16 维结构:
    left_ee_pose(7)=xyz(3)+quat(4) + left_gripper(1)
    + right_ee_pose(7) + right_gripper(1)

用法:
    python tools/keyframe_detect.py                      # 每任务随机 3 episode
    python tools/keyframe_detect.py --per-task 3 --seed 0 --tasks 0,2,8
    python tools/keyframe_detect.py --out outputs/keyframe_detect
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

GRIP_L = 7
GRIP_R = 15

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

# 绘图风格复用 visualize_distribution_shift.py
INK = "#0b0b0b"; MUT = "#898781"; GRID = "#e1e0d9"; SURF = "#fcfcfb"


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------

def load_state_frame_df(csv_path: Path) -> pd.DataFrame:
    """读取 CSV 的 state 相关列。"""
    return pd.read_csv(
        csv_path,
        usecols=["episode_index", "task_index", "frame_index", "observation.state"],
    )


def episode_state(
    df: pd.DataFrame, episode_index: int
) -> tuple[np.ndarray, np.ndarray]:
    """取单个 episode 的 (state (T,16), frame_index (T,))。"""
    ep = df[df["episode_index"] == episode_index].sort_values("frame_index")
    state = np.asarray(
        [np.fromstring(s.strip("[]"), sep=",", dtype=float) for s in ep["observation.state"]]
    )
    return state, ep["frame_index"].to_numpy()


def load_tasks(tasks_parquet: Path) -> dict[int, str]:
    """读取 tasks.parquet -> {task_index: 完整指令}。"""
    import pyarrow.parquet as pq

    t = pq.read_table(str(tasks_parquet)).to_pandas()
    return {int(v): str(k) for k, v in t["task_index"].items()}


# ---------------------------------------------------------------------------
# 关键帧检测
# ---------------------------------------------------------------------------

INCOMPLETE = -1  # open_full 哨兵: 周期不完整(结尾仍在保持, 未回开)


def detect_gripper_keyframes(
    gripper: np.ndarray,
    frames: np.ndarray,
    *,
    eps: float = 0.02,
    min_len: int = 2,
    min_prominence: float = 0.2,
    hold_min_len: int = 3,
    open_level: float = 0.9,
    allow_incomplete: bool = True,
) -> dict[str, np.ndarray]:
    """检测单个爪夹的抓取周期关键帧。

    周期结构: 从开(≈1)下降 → 水平保持 → 上升到开(≈1)。
    关键帧: close_start, hold_start, hold_end, open_full。
    allow_incomplete=True 时, 结尾仍在保持(未回开)的抓取也标出,
    其 open_full = INCOMPLETE(-1)。

    参数:
        gripper: (T,) 爪夹开度序列 (0 闭 ~ 1 开)
        frames:  (T,) 对应 frame_index
        eps: 判定"变化"的 diff 阈值（静止噪声 |d|<=0.005, 动作 |d|~0.1）
        min_len: 变化段最少帧数, 过滤单帧毛刺
        min_prominence: 运动段幅度下限; 低于此的上升/下降(如重抓抖动、
            浅捏 1→0.75)并入平段, 不构成独立抓取周期
        hold_min_len: 水平保持段最少帧数
        open_level: 视作"打开"(≈1) 的开度阈值

    返回:
        close_start/hold_start/hold_end/open_full: 各关键帧的 frame_index 数组
    """
    from scipy.ndimage import median_filter

    g = median_filter(np.asarray(gripper, dtype=float), size=3, mode="nearest")
    d = np.diff(g)
    n = len(d)

    def segments(mask: np.ndarray) -> list[tuple[int, int]]:
        """mask 中连续 True 段 -> (帧起点, 帧终点) 闭区间。"""
        segs = []
        i = 0
        m = len(mask)
        while i < m:
            if mask[i]:
                j = i
                while j + 1 < m and mask[j + 1]:
                    j += 1
                segs.append((i, j + 1))  # 帧区间 [i, j+1]
                i = j + 1
            else:
                i += 1
        return segs

    desc_segs = [
        (a, b) for (a, b) in segments(d < -eps)
        if (b - a + 1) >= min_len and (g[a] - g[b]) >= min_prominence
    ]
    asc_segs = [
        (a, b) for (a, b) in segments(d > eps)
        if (b - a + 1) >= min_len and (g[b] - g[a]) >= min_prominence
    ]

    # 每帧标记: -1 下压, +1 上升, 0 平段
    label = np.zeros(n, dtype=int)
    for a, b in desc_segs:
        label[a:b + 1] = -1
    for a, b in asc_segs:
        label[a:b + 1] = 1

    # 帧级事件序列 (连续同标号合为一段)
    events: list[tuple[str, int, int]] = []
    i = 0
    while i < n:
        if label[i] == 0:
            j = i
            while j + 1 < n and label[j + 1] == 0:
                j += 1
            events.append(("flat", i, j))
        else:
            typ = "desc" if label[i] == -1 else "asc"
            j = i
            while j + 1 < n and label[j] == label[j + 1]:
                j += 1
            events.append((typ, i, j))
        i = j + 1

    cycles: list[tuple[int, int, int, int]] = []  # (close_start, hold_start, hold_end, open_full|-1)
    e = len(events)

    def emit(desc_a, closed_flats, open_full) -> None:
        hold = max(closed_flats, key=lambda f: f[1] - f[0])
        if hold[1] - hold[0] + 1 < hold_min_len:
            return
        # close_start 回溯到 open 平台边界(起点值≈1)
        cs = desc_a
        while cs > 0 and g[cs - 1] < open_level:
            cs -= 1
        if cs > 0:
            cs -= 1
        cycles.append((cs, hold[0], hold[1], open_full))

    idx = 0
    while idx < e:
        typ, a, b = events[idx]
        if typ != "desc":
            idx += 1
            continue

        completed = False
        closed_flats: list[tuple[int, int]] = []
        j = idx + 1
        while j < e:
            t2, a2, b2 = events[j]
            if t2 == "desc":
                # 二次下压: 之前的平段只是下压中的停顿, 丢弃
                closed_flats = []
                j += 1
                continue
            if t2 == "flat":
                if g[a2] < open_level:
                    closed_flats.append((a2, b2))
                    j += 1
                    continue
                # 回到 open 平段(逐渐重新打开, 无显著上升段)
                if closed_flats:
                    emit(a, closed_flats, a2)
                    completed = True
                break
            # asc
            if g[b2] >= open_level:
                # 回开到打开, 周期完成
                if closed_flats:
                    emit(a, closed_flats, b2)
                    completed = True
                break
            # 重抓抖动(未回开): 并入保持, 继续找后续回开
            j += 1
            continue
        if not completed and allow_incomplete and closed_flats:
            # 事件跑完仍未回开: 结尾在保持中 -> 不完整周期
            emit(a, closed_flats, INCOMPLETE)
            completed = True
        idx = j + 1 if completed else idx + 1

    return {
        "close_start": np.asarray([c[0] for c in cycles], dtype=int),
        "hold_start": np.asarray([c[1] for c in cycles], dtype=int),
        "hold_end": np.asarray([c[2] for c in cycles], dtype=int),
        "open_full": np.asarray([c[3] for c in cycles], dtype=int),
    }


# ---------------------------------------------------------------------------
# 绘图
# ---------------------------------------------------------------------------

C_LEFT = "#2a78d6"
C_RIGHT = "#eb6834"
# 关键帧: 颜色区分类型, 线型辅助区分
KEYFRAME_STYLES = [
    ("close_start", "#1baf7a", "--", "close start (open~1)"),
    ("hold_start", "#d62728", "-.", "hold start"),
    ("hold_end", "#9467bd", ":", "hold end"),
    ("open_full", "#f2a900", "-", "fully open (~1)"),
]


def plot_episode_keyframes(
    task_name: str,
    episode_index: int,
    frame: np.ndarray,
    state: np.ndarray,
    keyframes: dict[str, dict[str, np.ndarray]],
    out_path: Path,
) -> None:
    """绘制单个 episode 的左右爪夹时序图, 关键帧竖线标出。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "axes.edgecolor": MUT, "axes.labelcolor": INK, "axes.titlecolor": INK,
        "text.color": INK, "xtick.color": INK, "ytick.color": INK,
        "figure.facecolor": SURF, "axes.facecolor": SURF, "grid.color": GRID,
        "font.family": "sans-serif", "figure.dpi": 110,
    })

    fig, axes = plt.subplots(2, 1, figsize=(13, 6.5), sharex=True)
    fig.suptitle(f"{task_name}\nepisode {episode_index}  ({len(frame)} frames)",
                 fontsize=11, y=0.99)

    for ax, name, idx, color in [
        (axes[0], "Left gripper", GRIP_L, C_LEFT),
        (axes[1], "Right gripper", GRIP_R, C_RIGHT),
    ]:
        ax.plot(frame, state[:, idx], color=color, ls="-", lw=1.2)
        side = "left" if idx == GRIP_L else "right"
        for key, linecolor, linestyle, _ in KEYFRAME_STYLES:
            for x in keyframes[side][key]:
                if x == INCOMPLETE:
                    continue  # 不完整周期的 open_full 不画线
                ax.axvline(x, color=linecolor, linestyle=linestyle, alpha=0.8, lw=1.1)
        ax.set_title(name)
        ax.set_ylabel("gripper (0 closed ~ 1 open)")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.25)

    # 图例
    handles = [
        plt.Line2D([0], [0], color=C_LEFT, lw=1.5, label="left"),
        plt.Line2D([0], [0], color=C_RIGHT, lw=1.5, label="right"),
    ] + [
        plt.Line2D([0], [0], color=c, ls=s, lw=1.2, label=lbl)
        for _, c, s, lbl in KEYFRAME_STYLES
    ]
    axes[1].legend(handles=handles, fontsize=8, loc="best", ncol=2)

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
    parser.add_argument("--per-task", type=int, default=3,
                        help="每任务随机挑选的 episode 数（批次数）")
    parser.add_argument("--min-prominence", type=float, default=0.2,
                        help="运动段幅度下限(低于此的抖动/浅捏并入平段)")
    parser.add_argument("--hold-min-len", type=int, default=3,
                        help="水平保持段最少帧数")
    parser.add_argument("--open-level", type=float, default=0.9,
                        help="视作打开的阈值(周期起点/终点回到该值)")
    parser.add_argument("--no-incomplete", action="store_true",
                        help="不标记结尾未回开(不完整)的抓取周期")
    parser.add_argument("--seed", type=int, default=0, help="随机种子(可复现)")
    parser.add_argument("--out", default=str(ROOT / "outputs" / "keyframe_detect"),
                        help="输出目录")
    args = parser.parse_args()

    tasks_parquet = DEFAULT_TASKS
    task_names = load_tasks(tasks_parquet) if tasks_parquet.is_file() else {}

    print(f"loading {args.csv} ...")
    df = load_state_frame_df(Path(args.csv))
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
    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.out)

    detect_kwargs = {
        "min_prominence": args.min_prominence,
        "hold_min_len": args.hold_min_len,
        "open_level": args.open_level,
        "allow_incomplete": not args.no_incomplete,
    }
    n_saved = 0
    summary = {}
    for ti in selected_tasks:
        if ti not in episodes_by_task:
            print(f"  ! task_index={ti} 不存在, 跳过")
            continue
        eps = episodes_by_task[ti]
        pick = rng.choice(eps, size=min(args.per_task, len(eps)), replace=False)
        slug = TASK_SLUGS.get(ti, f"task_{ti:03d}")
        name = task_names.get(ti, slug)

        for ep in pick:
            ep = int(ep)
            state, frame = episode_state(df, ep)
            kf = {
                "left": detect_gripper_keyframes(state[:, GRIP_L], frame, **detect_kwargs),
                "right": detect_gripper_keyframes(state[:, GRIP_R], frame, **detect_kwargs),
            }
            out_path = out_dir / f"{slug}_{ep:03d}.png"
            plot_episode_keyframes(name, ep, frame, state, kf, out_path)
            l_inc = int((kf['left']['open_full'] == INCOMPLETE).sum())
            r_inc = int((kf['right']['open_full'] == INCOMPLETE).sum())
            print(f"  task {ti:2d} ({slug:24s}) ep {ep:3d}: "
                  f"L={len(kf['left']['close_start'])}cyc{'(+'+str(l_inc)+'不完整)' if l_inc else ''} "
                  f"R={len(kf['right']['close_start'])}cyc{'(+'+str(r_inc)+'不完整)' if r_inc else ''} -> {out_path}")
            summary[f"task{ti}_{ep}"] = {
                "task_index": ti,
                "episode_index": ep,
                "left": {k: v.tolist() for k, v in kf["left"].items()},
                "right": {k: v.tolist() for k, v in kf["right"].items()},
            }
            n_saved += 1

    summary_path = out_dir / "keyframes_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\ndone, {n_saved} figures -> {out_dir}")
    print(f"keyframe summary -> {summary_path}")


if __name__ == "__main__":
    main()
