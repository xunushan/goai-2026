#!/usr/bin/env python3
"""训练集 episode 的 state 时序洞察可视化工具。

从 data/lerobot_v30_ee.csv（训练集全量, 每行一帧）读取 state 16 维向量,
对每个任务随机挑选若干 episode, 绘制 3 子图随帧变化:

    子图 1  末端位置 xyz      x/y/z 三色, 左臂实线 / 右臂虚线
    子图 2  姿态旋转角(deg)   相对 episode 首帧四元数的累计旋转角 (compute_rotation_deg)
    子图 3  gripper 开度      左臂实线 / 右臂虚线 (0 闭 ~ 1 开)

state 16 维结构 (与动作同构):
    left_ee_pose(7)=xyz(3)+quat(x,y,z,w)(4) + left_ee_joint_state(1)=gripper
    + right_ee_pose(7) + right_ee_joint_state(1)

用法:
    python tools/episode_state_insight.py
    python tools/episode_state_insight.py --per-task 3 --seed 0 --out outputs/episode_insight
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# 保证可从仓库根 import utils 模块（无论 cwd 在哪）
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.episode_analysis import compute_rotation_deg  # noqa: E402

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

# state 16 维切片索引
POS_L = slice(0, 3)    # 左臂位置 xyz
QUAT_L = slice(3, 7)   # 左臂四元数 (x,y,z,w)
GRIP_L = 7             # 左臂 gripper
POS_R = slice(8, 11)   # 右臂位置 xyz
QUAT_R = slice(11, 15) # 右臂四元数 (x,y,z,w)
GRIP_R = 15            # 右臂 gripper

# task_index -> 简短 slug（文件名用）。图内标题用 tasks.parquet 的完整指令。
# 来源: data/lerobot_v30_ee/meta/tasks.parquet 的 12 条指令
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

# dataviz 参考调色板 (light mode)
INK = "#0b0b0b"; MUT = "#898781"; GRID = "#e1e0d9"; SURF = "#fcfcfb"
C_XYZ = ("#2a78d6", "#eb6834", "#1baf7a")   # x / y / z (blue / orange / aqua)
C_LEFT = "#2a78d6"                           # 左臂
C_RIGHT = "#eb6834"                          # 右臂

DEFAULT_CSV = ROOT / "data" / "lerobot_v30_ee.csv"
DEFAULT_TASKS = ROOT / "data" / "lerobot_v30_ee" / "meta" / "tasks.parquet"


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------

def load_tasks(tasks_parquet: Path) -> dict[int, str]:
    """读取 tasks.parquet -> {task_index: 完整指令}。"""
    import pyarrow.parquet as pq

    t = pq.read_table(str(tasks_parquet)).to_pandas()
    # 结构: index=完整指令, 列 task_index=编号
    return {int(v): str(k) for k, v in t["task_index"].items()}


def load_state_frame_df(csv_path: Path) -> pd.DataFrame:
    """读取 CSV 的 state 相关列 (episode_index, task_index, frame_index, observation.state)。"""
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


# ---------------------------------------------------------------------------
# 绘图
# ---------------------------------------------------------------------------

def plot_episode_state(
    frame: np.ndarray,
    state: np.ndarray,
    task_name: str,
    task_slug: str,
    episode_index: int,
    out_path: Path,
) -> None:
    """绘制单个 episode 的 3 子图 (xyz / 旋转角 / gripper) 并保存 PNG。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_frames = state.shape[0]
    q0_l = state[0, QUAT_L].astype(float)   # 首帧姿态作参考
    q0_r = state[0, QUAT_R].astype(float)

    plt.rcParams.update({
        "axes.edgecolor": MUT, "axes.labelcolor": INK, "axes.titlecolor": INK,
        "text.color": INK, "xtick.color": INK, "ytick.color": INK,
        "figure.facecolor": SURF, "axes.facecolor": SURF, "grid.color": GRID,
        "font.family": "sans-serif", "figure.dpi": 110,
    })

    fig, axes = plt.subplots(2, 2, figsize=(14, 8.5))
    fig.suptitle(f"{task_name}\nepisode {episode_index}  ({n_frames} frames)",
                 fontsize=11, y=0.99)

    # ---- 位置 xyz: 左右臂各一个子图, 散点无实线, 便于看轨迹变化 ----
    for ax, name, sl, prefix in [
        (axes[0, 0], "Left arm", POS_L, "L"),
        (axes[0, 1], "Right arm", POS_R, "R"),
    ]:
        for d, c in enumerate(C_XYZ):
            ax.plot(frame, state[:, sl][:, d], color=c, ls="none",
                    marker="o", ms=3, alpha=0.7,
                    label=f"{prefix} {('xyz'[d])}")
        ax.set_title(f"{name}: end-effector position xyz")
        ax.set_xlabel("frame index"); ax.set_ylabel("position (m)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)

    # ---- 旋转角 (相对首帧姿态, compute_rotation_deg) ----
    ax = axes[1, 0]
    rot_l = compute_rotation_deg(np.tile(q0_l, (n_frames, 1)), state[:, QUAT_L].astype(float))
    rot_r = compute_rotation_deg(np.tile(q0_r, (n_frames, 1)), state[:, QUAT_R].astype(float))
    ax.plot(frame, rot_l, color=C_LEFT, ls="-", lw=1.1, label="Left arm")
    ax.plot(frame, rot_r, color=C_RIGHT, ls="--", lw=1.1, label="Right arm")
    ax.set_title("rotation from initial pose")
    ax.set_xlabel("frame index"); ax.set_ylabel("angle (deg)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

    # ---- gripper 开度 ----
    ax = axes[1, 1]
    ax.plot(frame, state[:, GRIP_L], color=C_LEFT, ls="-", lw=1.1, label="Left arm")
    ax.plot(frame, state[:, GRIP_R], color=C_RIGHT, ls="--", lw=1.1, label="Right arm")
    ax.set_title("gripper opening")
    ax.set_xlabel("frame index"); ax.set_ylabel("gripper (0 closed ~ 1 open)")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

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
    parser.add_argument("--per-task", type=int, default=3, help="每任务随机挑选的 episode 数")
    parser.add_argument("--seed", type=int, default=0, help="随机种子(可复现)")
    parser.add_argument("--out", default=str(ROOT / "outputs" / "episode_insight"),
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

    n_saved = 0
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
            out_path = out_dir / f"{slug}_{ep:03d}.png"
            plot_episode_state(frame, state, name, slug, ep, out_path)
            print(f"  task {ti:2d} ({slug:24s}) ep {ep:3d}: "
                  f"{state.shape[0]:4d} frames -> {out_path}")
            n_saved += 1
    print(f"\ndone, {n_saved} figures -> {out_dir}")


if __name__ == "__main__":
    main()
