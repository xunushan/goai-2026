#!/usr/bin/env python3
"""sim_lerobot_v30_ee 关键阶段标注: 每帧一个关键阶段标签 + 可视化 (先实现 fill_pen_holder)。

在关键帧检测 (detect_gripper_keyframes, 见 tools/keyframe_detect.py) 的基础上, 把每个
抓取周期解析成关键事件, 再把事件前后扩成"阶段窗口", 给每帧赋一个阶段标签:

    fill_pen_holder 四个阶段 (各阶段 a=t0-前窗口, b=t0+后窗口, [a,b] 闭区间内标该阶段):

        | 阶段      | 作用臂     | t0           | 窗口        |
        |-----------|-----------|--------------|-------------|
        | 抓取笔筒  | 持筒臂    | hold_start   | [t0-15, t0+5]  |
        | 放下笔筒  | 持筒臂    | hold_end     | [t0-15, t0+5]  |
        | 抓笔      | 笔臂      | hold_start   | [t0-15, t0+5]  |
        | 放笔      | 笔臂      | hold_end     | [t0-20, t0+10] |

    其余帧标签为 none(普通运动/静止)。

fill 中两个臂角色不同: 一个臂"始终抓着笔筒"(抓一次、整段保持、最后放下 -> 单条长周期),
另一臂多次短周期抓笔-放笔。故先判定持筒臂: 左右臂各周期中, 保持段长度最大且跨段占比 >=
THREAD_FRAC 的臂为持筒臂, 另一臂为笔臂 (不同 episode 左右臂角色可互换, 须逐 episode 判定)。

窗口/t0 均在"位置空间"(state 行号)计算; sim 数据集 frame_index 从 0 连续, 位置 == 帧号,
统计输出仍给真实 frame_index。持筒臂只取跨段长的周期作为持筒事件; 笔臂所有周期都是抓/放笔事件。

用法:
    python tools/stage_label.py                            # fill(task 0), ep 0,50,99
    python tools/stage_label.py --tasks 0 --episodes 0,50,99 --out outputs/xxx
    python tools/stage_label.py --min-prominence 0.2 --hold-min-len 3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.frame_weight_trapezoid import (  # noqa: E402
    TASK_SLUGS,
    load_state_cols,
    load_tasks,
    parse_state_series,
    pick_episodes,
)
from tools.keyframe_detect import (  # noqa: E402
    GRIP_L,
    GRIP_R,
    INCOMPLETE,
    C_LEFT,
    C_RIGHT,
    detect_gripper_keyframes,
)

DEFAULT_CSV = ROOT / "data" / "sim_lerobot_v30_ee" / "sim_lerobot_v30_ee.csv"
DEFAULT_OUT = ROOT / "outputs" / "stage_label_sim_v30ee"

# 阶段标签调色板 (绘图用; none 帧不画色)
STAGE_COLORS: dict[str, str] = {
    "grasp_holder": "#2a78d6",   # 抓取笔筒
    "place_holder": "#b07bd8",   # 放下笔筒
    "grasp_pen":    "#eb6834",   # 抓笔
    "place_pen":    "#4c9f38",   # 放笔
}

# fill_pen_holder 阶段表: role=持筒臂(holder)/笔臂(pen); anchor=该周期 hold_start/hold_end;
# pre/post = 窗口在 t0 前/后延伸 (用户确认: 放下笔筒也到 holder_end+5, 与抓取窗口对称)
FILL_STAGES: list[dict] = [
    {"key": "grasp_holder", "cn": "抓取笔筒", "role": "holder",
     "anchor": "hold_start", "pre": 15, "post": 5},
    {"key": "place_holder", "cn": "放下笔筒", "role": "holder",
     "anchor": "hold_end", "pre": 15, "post": 5},
    {"key": "grasp_pen", "cn": "抓笔", "role": "pen",
     "anchor": "hold_start", "pre": 15, "post": 5},
    {"key": "place_pen", "cn": "放笔", "role": "pen",
     "anchor": "hold_end", "pre": 20, "post": 10},
]

# 持筒臂周期: 保持跨段须占整条轨迹至少该比例, 才认定为"持筒"而非偶然长保持
THREAD_FRAC = 0.30
# 预设判定: 持筒臂跨段 >= 该占比 视为可信 (低于仅告警)
HOLDER_SURE_FRAC = 0.5


# ---------------------------------------------------------------------------
# 角色判定 (持筒臂 vs 笔臂)
# ---------------------------------------------------------------------------

def cycle_hold_span(cyc: tuple) -> int:
    """周期 (cs, hs, he, of) 的保持长度; he=-1(不完整) 按长保持处理仍可用。"""
    _, hs, he, _ = cyc
    if he == INCOMPLETE:
        return int(1e9)  # 结尾仍保持 -> 视为极长, 优先选为持筒候选
    return he - hs


def resolve_fill_roles(
    state: np.ndarray,
    frame: np.ndarray,
    per_side: dict[str, dict[str, np.ndarray]],
    hold_frac_min: float = HOLDER_SURE_FRAC,
) -> tuple[dict, dict]:
    """fill 每 episode 角色: 返回 (roles, holder_cycles)。

    roles = {"holder": "left"|"right", "pen": ...}
    holder_cycles = {"left": [(cs,hs,he,of)...], "right": [...]} 的持筒臂长周期列表。
    """
    spans: dict[str, int] = {}
    cycle_lists: dict[str, list] = {}
    for side, idx in (("left", GRIP_L), ("right", GRIP_R)):
        kf = per_side[side]
        cycles = [(int(kf["close_start"][i]), int(kf["hold_start"][i]),
                   int(kf["hold_end"][i]), int(kf["open_full"][i]))
                  for i in range(len(kf["close_start"]))]
        cycle_lists[side] = cycles
        # 单周期最大保持时长 (不完整周期记 +inf, 已由 cycle_hold_span 处理)
        spans[side] = max((cycle_hold_span(c) for c in cycles), default=-1)

    if all(s <= 0 for s in spans.values()):
        raise ValueError("两侧均无抓取周期, 无法判定持筒臂")
    holder = max(("left", "right"), key=lambda s: spans[s])
    pen = "right" if holder == "left" else "left"

    # 持筒臂中真正"整段持筒"的周期: 跨段占比达标
    ep_len = len(state)
    holder_cycles = [c for c in cycle_lists[holder]
                     if (c[2] if c[2] != INCOMPLETE else ep_len - 1) - c[1]
                     >= hold_frac_min * ep_len]
    if not holder_cycles:
        print(f"    ! 持筒臂 {holder} 无跨段>= {hold_frac_min:.0%} 的长周期, "
              f"用最长周期兜底")
        holder_cycles = [max(cycle_lists[holder],
                             key=lambda c: (c[2] if c[2] != INCOMPLETE
                                            else ep_len - 1) - c[1])]
    return {"holder": holder, "pen": pen}, {holder: holder_cycles,
                                            pen: cycle_lists[pen]}


# ---------------------------------------------------------------------------
# 阶段标签 (逐帧)
# ---------------------------------------------------------------------------

def label_from_cycle(
    cycle: tuple,
    stages: list[dict],
    role: str,
    T: int,
) -> list[dict]:
    """单个抓取周期 -> 命中该周期的阶段窗口列表 (含 arm/type/frame 元信息)。

    返回 [{key, cn, a, b, t0}], a/b 为位置闭区间 (已 clip 到 [0,T-1])。
    """
    _, hs, he, _ = cycle
    out: list[dict] = []
    for st in stages:
        if st["role"] != role:
            continue
        if (st["role"], st["anchor"]) == ("holder", "hold_end") and he == INCOMPLETE:
            continue  # 持筒臂结尾未放开 -> 无"放下"
        t0 = he if st["anchor"] == "hold_end" else hs
        a = max(0, t0 - st["pre"])
        b = min(T - 1, t0 + st["post"])
        if a > b or b < 0 or a >= T:
            continue
        out.append({**st, "t0": t0, "a": a, "b": b})
    return out


def annotate_fill_episode(
    state: np.ndarray,
    frame: np.ndarray,
    detect_kwargs: dict,
    stages: list[dict] | None = None,
) -> dict:
    """单 fill episode -> 逐帧阶段标签 (labels: 位置空间 int, names: idx->key)。

    返回 dict(labels, names, spans, roles, per_side, holder_sure)。
    """
    stages = FILL_STAGES if stages is None else stages
    T = len(state)
    per_side = {}
    for side, idx in (("left", GRIP_L), ("right", GRIP_R)):
        kf = detect_gripper_keyframes(state[:, idx], frame, **detect_kwargs)
        per_side[side] = {k: np.asarray(v, dtype=int) for k, v in kf.items()}

    roles, holder_cycles = resolve_fill_roles(state, frame, per_side)

    # 收集该 episode 所有阶段窗口
    spans: list[dict] = []
    for side in ("left", "right"):
        role = "holder" if side == roles["holder"] else "pen"
        for cyc in holder_cycles[side]:
            spans.extend(label_from_cycle(cyc, stages, role, T))
    spans.sort(key=lambda s: s["a"])

    # 逐帧标签: 位置空间
    names = [s["key"] for s in stages]
    labels = np.zeros(T, dtype=int)          # 0 = none
    idx_of = {s["key"]: i + 1 for i, s in enumerate(stages)}
    for s in spans:
        labels[s["a"]:s["b"] + 1] = idx_of[s["key"]]

    # 判定可信度: 持筒臂最长周期跨段占比
    hs, he = holder_cycles[roles["holder"]][0][1], holder_cycles[
        roles["holder"]][0][2]
    holder_end = he if he != INCOMPLETE else T - 1
    holder_sure = (holder_end - hs) / T

    return {"labels": labels, "names": ["none"] + [s["key"] for s in stages],
            "spans": spans, "roles": roles, "per_side": per_side,
            "holder_sure": float(holder_sure)}


# ---------------------------------------------------------------------------
# 可视化
# ---------------------------------------------------------------------------

def plot_episode_stages(
    task_name: str,
    episode_index: int,
    frame: np.ndarray,
    state: np.ndarray,
    res: dict,
    out_path: Path,
) -> None:
    """3 行图: [逐帧阶段标签色带] [左爪夹+阶段底纹] [右爪夹+阶段底纹]。

    阶段色带直接展示"每帧一个标签"; 下两行把窗口底纹叠在对应动作臂的爪夹曲线上,
    便于核对窗口是否包住真实抓/放动作。行内虚线 = 各阶段 t0 锚点。
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import colors as mcolors
    from tools.keyframe_detect import GRID, INK, MUT, SURF

    plt.rcParams.update({
        "axes.edgecolor": MUT, "axes.labelcolor": INK, "axes.titlecolor": INK,
        "text.color": INK, "xtick.color": INK, "ytick.color": INK,
        "figure.facecolor": SURF, "axes.facecolor": SURF, "grid.color": GRID,
        "font.family": "sans-serif", "figure.dpi": 110,
        "font.sans-serif": ["PingFang SC", "Hiragino Sans GB", "Heiti SC",
                            "Arial Unicode MS", "DejaVu Sans"],
        "axes.unicode_minus": False,
    })

    names = res["names"]                       # ["none", grasp_holder, ...]
    labels = res["labels"]
    roles = res["roles"]
    stage_keys = names[1:]
    cmap_colors = ["#eceff1"] + [STAGE_COLORS[k] for k in stage_keys]
    cmap = mcolors.ListedColormap(cmap_colors)

    fig, axes = plt.subplots(3, 1, figsize=(13.5, 8.5), sharex=True)
    fig.suptitle(f"{task_name}\nepisode {episode_index}  ({len(frame)} frames)  "
                 f"持筒臂={roles['holder']}  笔臂={roles['pen']}   "
                 f"持筒跨段 {res['holder_sure']:.0%}",
                 fontsize=11, y=0.995)

    T = len(labels)
    # 行0: 逐帧标签色带
    ax = axes[0]
    x = np.arange(T + 1) - 0.5              # 格边 (连续 frame_index)
    ax.pcolormesh(x, [0, 1], labels[None, :], cmap=cmap, vmin=0, vmax=len(names) - 1,
                  shading="flat")
    ax.set_yticks([])
    ax.set_ylim(0, 1)
    ax.set_ylabel("stage")
    ax.set_title("stage label per frame "
                 "(0=none; 蓝=抓取笔筒 紫=放下笔筒 橙=抓笔 绿=放笔)",
                 loc="left", fontsize=9)
    # 色带内标阶段中文名 (取窗口中心, 重叠时可能错位但仅供目检)
    for s in res["spans"]:
        cx = frame[s["a"]] + (frame[min(s["b"], T - 1)] - frame[s["a"]]) / 2
        ax.text(cx, 0.5, s["cn"], ha="center", va="center", fontsize=7,
                color="white", fontweight="bold", clip_on=True)

    # 行1/2: 爪夹曲线 + 各阶段窗口底纹 (只画在动作臂所在行)
    for row_side, idx, color in (("left", GRIP_L, C_LEFT), ("right", GRIP_R, C_RIGHT)):
        axg = axes[1] if row_side == "left" else axes[2]
        # 该臂负责的阶段窗口底纹
        for s in res["spans"]:
            arm = res["roles"]["holder"] if s["role"] == "holder" else res["roles"]["pen"]
            if arm != row_side:
                continue
            axg.axvspan(frame[s["a"]], frame[s["b"]],
                        color=STAGE_COLORS[s["key"]], alpha=0.18, lw=0)
            axg.axvline(frame[s["t0"]], color=STAGE_COLORS[s["key"]],
                        ls="--", lw=0.9, alpha=0.6)
        axg.plot(frame, state[:, idx], color=color, lw=1.1)
        axg.set_ylabel("gripper")
        axg.set_ylim(-0.05, 1.05)
        side_role = ("持筒臂" if res["roles"]["holder"] == row_side else "笔臂")
        axg.set_title(f"{'Left' if row_side == 'left' else 'Right'} gripper "
                      f"({side_role})", loc="left", fontsize=9)
        axg.grid(alpha=0.25)

    # 图例 (阶段色)
    handles = [
        plt.Line2D([0], [0], color=C_LEFT, lw=1.5, label="left gripper"),
        plt.Line2D([0], [0], color=C_RIGHT, lw=1.5, label="right gripper"),
    ]
    for k in stage_keys:
        cn = next(s["cn"] for s in res["spans"] if s["key"] == k) if any(
            s["key"] == k for s in res["spans"]) else k
        handles.append(plt.Rectangle((0, 0), 1, 1, color=STAGE_COLORS[k],
                                     label=cn))
    fig.legend(handles=handles, fontsize=8, ncol=len(handles), frameon=False,
               loc="upper center", bbox_to_anchor=(0.5, 0.985))
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    axes[2].set_xlabel("frame index")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=str(DEFAULT_CSV), help="sim CSV 路径")
    parser.add_argument("--tasks", default="0",
                        help="task_index, 逗号分隔 (目前仅 0=fill_pen_holder 有阶段表)")
    parser.add_argument("--per-task", type=int, default=3)
    parser.add_argument("--episodes", default=None,
                        help="指定 ep(逗号分隔, 覆盖 --per-task)")
    parser.add_argument("--min-prominence", type=float, default=0.2)
    parser.add_argument("--hold-min-len", type=int, default=3)
    parser.add_argument("--open-level", type=float, default=0.9)
    parser.add_argument("--no-incomplete", action="store_true")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    detect_kwargs = {
        "min_prominence": args.min_prominence,
        "hold_min_len": args.hold_min_len,
        "open_level": args.open_level,
        "allow_incomplete": not args.no_incomplete,
    }
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    task_names = load_tasks(ROOT / "data" / "sim_lerobot_v30_ee" / "meta"
                            / "tasks.parquet") if (ROOT / "data"
                            / "sim_lerobot_v30_ee" / "meta"
                            / "tasks.parquet").is_file() else {}

    print(f"loading {args.csv} ...")
    small = load_state_cols(Path(args.csv))
    eps_by = (small.groupby("task_index")["episode_index"]
              .apply(lambda s: np.sort(s.unique())).to_dict())
    selected = [int(x) for x in args.tasks.split(",")] if args.tasks != "all" \
        else sorted(eps_by)
    want_eps = [int(x) for x in args.episodes.split(",")] if args.episodes \
        else None

    summary: dict = {}
    n_saved = 0
    for ti in selected:
        if ti not in eps_by:
            print(f"  ! task_index={ti} 不存在, 跳过")
            continue
        if ti != 0:
            print(f"  ! task_index={ti} 暂无阶段表, 跳过 (当前仅 fill 0 实现)")
            continue
        slug = TASK_SLUGS.get(ti, f"task_{ti:03d}")
        name = task_names.get(ti, slug)
        eps = eps_by[ti]
        pick = (np.asarray(want_eps, dtype=int) if want_eps is not None
                else pick_episodes(eps, args.per_task))
        pick = pick[np.isin(pick, eps)]
        print(f"  task {ti} ({slug}): episodes {pick.tolist()}")
        for ep in [int(x) for x in pick]:
            sub = small[small["episode_index"] == ep].sort_values("frame_index")
            state = parse_state_series(sub["observation.state"])
            frame = sub["frame_index"].to_numpy()
            res = annotate_fill_episode(state, frame, detect_kwargs)
            out_png = out_dir / f"{slug}_ep{ep:03d}_stages.png"
            plot_episode_stages(name, ep, frame, state, res, out_png)
            n_saved += 1

            n_frame = len(frame)
            per_lbl = {k: 0 for k in res["names"]}
            for i, lbl in enumerate(res["labels"]):
                per_lbl[res["names"][int(lbl)]] += 1
            # 文本: 各阶段实际命中帧区间
            lines = [f"    ep{ep:3d} roles holder={res['roles']['holder']} "
                     f"pen={res['roles']['pen']} "
                     f"holder_span={res['holder_sure']:.0%}"]
            for s in res["spans"]:
                cn = s["cn"]
                arm = res["roles"][s["role"]]
                lines.append(f"        {cn:<6}[{s['role']:<7}{arm}] "
                             f"t0={frame[s['t0']]:4d}  frames {frame[s['a']]:4d}"
                             f"-{frame[s['b']]:4d}")
            print("\n".join(lines))
            pcts = {k: round(100.0 * v / n_frame, 2) for k, v in per_lbl.items()}
            print(f"        per-frame share: {pcts}")
            summary[f"task{ti}_ep{ep}"] = {
                "task_index": ti, "episode_index": ep, "n_frames": n_frame,
                "roles": res["roles"], "holder_span_frac": res["holder_sure"],
                "spans": [{"stage": s["key"], "cn": s["cn"], "role": s["role"],
                           "arm": res["roles"][s["role"]], "t0": int(frame[s["t0"]]),
                           "a": int(frame[s["a"]]), "b": int(frame[s["b"]])}
                          for s in res["spans"]],
                "per_label_pct": pcts,
            }

    (out_dir / "stage_bands_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n{n_saved} stage figures -> {out_dir}")
    print(f"summary -> {out_dir / 'stage_bands_summary.json'}")


if __name__ == "__main__":
    main()
