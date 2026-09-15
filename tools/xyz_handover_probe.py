#!/usr/bin/env python3
"""plug 交接(handover)候选配对的审查图 —— 只做诊断，不产出标签。

背景：docs/xyz_gripper_event_segmentation.md §12.2 规定 handover 的范围是
递交侧 `place` 与接取侧 `grasp` 的**并集**（重叠只用于配对判断）。本脚本把每个
episode 的候选配对画出来，用于人工核对：

  - 并集窗口的起止帧是否合理（起点是否恰好落在接取手到达最终定位邻域那一帧）；
  - 窗口内双臂末端距离的走向（是否"一手静止在等、另一手飞过来"）；
  - `place` / `grasp` 两个底层区间的相对位置。

配对判据（§12.2 前两条，第三条 D_handover 未落数值故不筛）：
  1. `place` 与 `grasp` 有重叠，或间隔 <= K_gap；
  2. 接取侧 `close_end` <= 递交侧 `open_end`。
第 1 层状态与锚点来自 data/sim_lerobot_v30_ee/xyz_segment/（由
tools/xyz_gripper_segment.py 生成），本脚本不重跑分段。

用法:
    python tools/xyz_handover_probe.py --task 1 --n-per-group 6
    python tools/xyz_handover_probe.py --task 1 --episodes 145,146,148
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import keyframe_events as ke  # noqa: E402
from tools import keyframe_detect as kd  # noqa: E402
from tools import xyz_gripper_segment as xs  # noqa: E402
from tools import xyz_task_event as te  # noqa: E402

ROOT = ke.ROOT
DEFAULT_LABEL_DIR = ROOT / "data" / "sim_lerobot_v30_ee" / "xyz_segment"
DEFAULT_OUT = ROOT / "outputs" / "xyz_handover_probe"


def pairings(anchors: pd.DataFrame, k_gap: int = 25) -> list[dict]:
    """枚举一臂 place × 另一臂 grasp 的候选配对（§12.2 条件 1、2）。

    配对逻辑的唯一实现在生产模块 tools/xyz_task_event.py（第 2 层映射也用同一套），
    此处仅转发，避免两处分叉。
    """
    return te.handover_pairings(anchors, k_gap)


def probe_episode(ep: int, ti: int, anchors: pd.DataFrame, states: pd.DataFrame,
                  out_dir: Path, fps: float = 25.0) -> dict:
    """画一个 episode 的交接审查图，返回统计。"""
    plt = xs._setup_mpl()
    pairs = pairings(anchors)
    T = len(states)

    fig, axes = plt.subplots(4, 1, figsize=(14, 9.2), sharex=True,
                             gridspec_kw={"height_ratios": [1.0, 1.0, 0.3, 0.3]})
    x = np.arange(T)

    # 交接并集窗口（横跨所有行）
    for k, pr in enumerate(pairs):
        for ax in axes:
            ax.axvspan(pr["u"][0], pr["u"][1], color="#2a78d6", alpha=0.10, zorder=0)
            for t in pr["u"]:
                ax.axvline(t, color="#2a78d6", lw=1.0, ls="--", alpha=0.8)

    # 行 0/1: 双臂夹爪开度（数据来自 parquet 的 state，由 main 放进 attrs）
    for row, s in ((0, "left"), (1, "right")):
        ax = axes[row]
        col = kd.C_LEFT if s == "left" else kd.C_RIGHT
        ax.plot(x, states.attrs["grip"][s], color=col, lw=1.8)
        for pr in pairs:
            for key, cn, ls in (("place", "递交手 place", ":"), ("grasp", "接取手 grasp", "-")):
                a_, b_ = pr[key]
                if (key == "place" and s == pr["releaser"]) or \
                   (key == "grasp" and s == pr["receiver"]):
                    ax.axvspan(a_, b_, color="#2a78d6" if key == "grasp" else "#e45756",
                               alpha=0.13, zorder=0)
                    ax.annotate(cn, xy=((a_ + b_) / 2, 1.06), fontsize=7.5, ha="center",
                                va="bottom", color="#2a78d6" if key == "grasp" else "#e45756")
        ax.set_ylim(-0.06, 1.12)
        ax.set_ylabel(f"{s} 夹爪开度", color=col, fontsize=9)
        ax.tick_params(axis="y", labelcolor=col)
        ax.grid(True, lw=0.5)
        ax.legend(handles=[plt.Line2D([], [], color=col, lw=1.8, label=f"{s} 夹爪开度")],
                  loc="upper right", fontsize=8, framealpha=0.9)

    # 行 2: 双臂末端距离
    axd = axes[2]
    dist = states.attrs["ee_dist"]
    axd.plot(x, dist, color="#4a4a4a", lw=1.3)
    for k, pr in enumerate(pairs):
        u0, u1 = pr["u"]
        seg = dist[u0:u1 + 1]
        km = int(np.argmin(seg)) + u0
        axd.plot([km], [dist[km]], "o", color="#e45756", ms=6, zorder=5)
        axd.annotate(f"最近逼近 {dist[km]:.3f} m", xy=(km, dist[km]), fontsize=8,
                     color="#e45756", xytext=(4, 10), textcoords="offset points")
    axd.set_ylabel("|L-R| 末端距离 (m)", fontsize=9)
    axd.set_ylim(bottom=0)
    axd.grid(True, lw=0.5)

    # 行 3: 双臂状态色带
    axb = axes[3]
    for row, s in ((0.0, "left"), (0.5, "right")):
        vals = states[f"{s}_state"].values
        for k in xs.STATES:
            for a_, b_ in xs._runs_of(vals, k):
                axb.broken_barh([(a_, b_ - a_)], (row, 0.46), facecolors=xs.STATE_COLOR[k])
    for s, row in (("left", 0.0), ("right", 0.5)):
        axb.text(-8, row + 0.23, s, ha="right", va="center", fontsize=8)
    axb.set_ylim(0, 1)
    axb.set_yticks([])
    axb.set_xlabel("frame_index")
    axb.grid(False)

    title = [f"plug episode {ep}  T={T}  交接候选配对数 = {len(pairs)}"]
    for k, pr in enumerate(pairs):
        u0, u1 = pr["u"]
        title.append(
            f"  [{k}] 递交={pr['releaser']} 接取={pr['receiver']}  "
            f"place=[{pr['place'][0]},{pr['place'][1]}) grasp=[{pr['grasp'][0]},{pr['grasp'][1]})  "
            f"重叠={pr['overlap']}f  并集=[{u0},{u1}) {u1 - u0}f  "
            f"t={u0 / fps:.2f}~{u1 / fps:.2f}s")
    fig.suptitle("\n".join(title), fontsize=10, y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(out_dir / f"handover_ep{ep}.png")
    plt.close(fig)

    stats = []
    for k, pr in enumerate(pairs):
        u0, u1 = pr["u"]
        seg = dist[u0:u1 + 1]
        stats.append({"episode_index": ep, "pair": k, "releaser": pr["releaser"],
                      "receiver": pr["receiver"],
                      "place_start": pr["place"][0], "place_end": pr["place"][1],
                      "grasp_start": pr["grasp"][0], "grasp_end": pr["grasp"][1],
                      "overlap_frames": pr["overlap"],
                      "union_start": u0, "union_end": u1, "union_len": u1 - u0,
                      "dist_at_union_start": round(float(dist[u0]), 4),
                      "dist_min_in_union": round(float(seg.min()), 4),
                      "argmin_in_union": int(np.argmin(seg)) + u0})
    return {"n_pairs": len(pairs), "stats": stats}


def main() -> None:
    ap = argparse.ArgumentParser(description="plug 交接候选配对审查图")
    ap.add_argument("--task", type=int, default=1)
    ap.add_argument("--episodes", default=None, help="逗号分隔；不给则按配对数自动选")
    ap.add_argument("--n-per-group", type=int, default=6, help="自动选样时每组取几个")
    ap.add_argument("--only-n", type=int, default=1,
                    help="只画「候选配对数 == 该值」的 episode；-1 = 全部组都画")
    ap.add_argument("--k-gap", type=int, default=25, help="§12.2 条件1 的允许配对间隔")
    ap.add_argument("--csv", default=str(ROOT / "data" / "sim_lerobot_v30_ee"
                                         / "sim_lerobot_v30_ee.csv"))
    ap.add_argument("--label-dir", default=str(DEFAULT_LABEL_DIR))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    anchors = pd.read_csv(Path(args.label_dir) / f"gripper_anchors_task{args.task}.csv")
    states_all = pd.read_csv(Path(args.label_dir) / f"arm_states_task{args.task}.csv")
    small = ke.load_state_cols(Path(args.csv))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 先按配对数分组
    by_n: dict[int, list[int]] = {}
    for ep in sorted(anchors.episode_index.unique()):
        n = len(pairings(anchors[anchors.episode_index == ep], args.k_gap))
        by_n.setdefault(n, []).append(int(ep))
    print("配对数分布:", {k: len(v) for k, v in sorted(by_n.items())})

    if args.episodes:
        eps = [int(v) for v in args.episodes.split(",") if v.strip()]
    else:
        eps = []
        groups = (sorted(by_n.items()) if args.only_n < 0
                  else [(args.only_n, by_n.get(args.only_n, []))])
        for n, lst in groups:
            if not lst:
                print(f"  [warn] 没有配对数 == {n} 的 episode")
                continue
            # 等间隔取样，覆盖整个 episode 范围
            idx = np.linspace(0, len(lst) - 1, min(args.n_per_group, len(lst))).round().astype(int)
            eps += [lst[i] for i in idx]
    print("选中 episode:", eps)

    rows = []
    for ep in eps:
        st, _ = kd.episode_state(small, int(ep))
        lay = states_all[states_all.episode_index == ep].reset_index(drop=True)
        assert len(lay) == len(st), f"ep{ep}: 标签 {len(lay)} 帧 != state {len(st)} 帧"
        lay.attrs["grip"] = {"left": st[:, ke.GRIP_L], "right": st[:, ke.GRIP_R]}
        lay.attrs["ee_dist"] = np.linalg.norm(st[:, 0:3] - st[:, 8:11], axis=-1)
        res = probe_episode(ep, args.task, anchors[anchors.episode_index == ep],
                            lay, out_dir, fps=25.0)
        rows += res["stats"]
        print(f"  ep{ep}: 配对数={res['n_pairs']}")

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(out_dir / "handover_pairs.csv", index=False)
        print(f"\n配对数分布(选中): {df.groupby('episode_index').size().to_dict()}")
        print(f"并集长度 p10/p50/p90 = "
              f"{np.percentile(df.union_len, 10):.0f}/"
              f"{df.union_len.median():.0f}/{np.percentile(df.union_len, 90):.0f}")
    print(f"产物目录: {out_dir}")


if __name__ == "__main__":
    main()
