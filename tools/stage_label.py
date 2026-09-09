#!/usr/bin/env python3
"""sim_lerobot_v30_ee 关键阶段标注: 每帧一个关键阶段标签 + 可视化 (fill_pen_holder 先落地)。

在关键帧检测 (detect_gripper_keyframes, 见 tools/keyframe_detect.py) 的基础上, 把每个
抓取周期解析成关键事件, 再把事件前后扩成"阶段窗口", 给每帧赋一个阶段标签:

    fill_pen_holder 四个阶段 (各阶段 a=t0-前窗口, b=t0+后窗口, [a,b] 闭区间内标该阶段):

        | 阶段      | 作用臂     | t0           | 窗口          |
        |-----------|-----------|--------------|---------------|
        | 抓取笔筒  | 持筒臂    | hold_start   | [t0-15, t0+5] |
        | 放下笔筒  | 持筒臂    | hold_end     | [t0-15, t0+5] |  (用户确认窗口右界为 holder_end+5)
        | 抓笔      | 笔臂      | hold_start   | [t0-15, t0+5] |
        | 放笔      | 笔臂      | hold_end     | [t0-20, t0+10] |

    其余帧标签为 none(普通运动/静止)。

fill 中两个臂角色不同: 一个臂"始终抓着笔筒"(抓一次、整段保持、最后放下 -> 单条长周期),
另一臂多次短周期抓笔-放笔。故先判定持筒臂: 左右臂各周期中, 保持段跨段最长且占比达标的臂
为持筒臂, 另一臂为笔臂 (不同 episode 左右臂角色可互换, 须逐 episode 判定)。实测 fill 内
左/右持筒并存 (约 ep99 等为右持筒)。

窗口/t0 均在"位置空间"(state 行号)计算; sim 数据集 frame_index 从 0 连续, 位置 == 帧号,
统计/输出仍给真实 frame_index。持筒臂只取跨段长的周期作为持筒事件; 笔臂所有周期都是抓/放笔事件。

子命令:
  viz   挑若干 episode 画图人工核对 (默认每任务 3 个均匀抽样)
  apply 对全部(或指定任务)episode 逐帧打标: 写精简 CSV(episode/frame/stage_label) +
        每 ep 角色与阶段明细 JSON, 并自动挑若干个"右臂持筒"episode 画验证图

用法:
    python tools/stage_label.py viz --tasks 0 --episodes 0,50,99
    python tools/stage_label.py viz --per-task 3
    python tools/stage_label.py apply --right-n 3
    python tools/stage_label.py apply --tasks 0 --out outputs/xxx --right-n 3
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
DEFAULT_OUT = ROOT / "outputs" / "stage_label_fill_sim_v30ee"

# 阶段标签调色板 (绘图用; none 帧不画色)
STAGE_COLORS: dict[str, str] = {
    "grasp_holder": "#2a78d6",   # 抓取笔筒
    "place_holder": "#b07bd8",   # 放下笔筒
    "grasp_pen":    "#eb6834",   # 抓笔
    "place_pen":    "#4c9f38",   # 放笔
}
NONE_COLOR = "#eceff1"

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
# 持筒臂周期须跨段至少占该比例才认定为"整段持筒"的长周期
HOLDER_SURE_FRAC = 0.5


# ---------------------------------------------------------------------------
# 角色判定 (持筒臂 vs 笔臂)
# ---------------------------------------------------------------------------

def cycle_hold_span(cyc: tuple) -> int:
    """周期 (cs, hs, he, of) 的保持长度; he=-1(不完整, 结尾仍在保持)按极长处理。"""
    _, hs, he, _ = cyc
    return int(1e9) if he == INCOMPLETE else he - hs


def resolve_fill_roles(
    state: np.ndarray,
    per_side: dict[str, dict[str, np.ndarray]],
    holder_min_frac: float = HOLDER_SURE_FRAC,
) -> tuple[dict, dict]:
    """fill 每 episode 角色判定 -> (roles, holder_cycles)。

    roles = {"holder": "left"|"right", "pen": ...}
    holder_cycles = {side: [(cs,hs,he,of)...]} 仅持筒臂长周期进 holder_cycles[holder],
    笔臂全部周期进 holder_cycles[pen]。
    """
    cycle_lists: dict[str, list[tuple]] = {}
    span_max: dict[str, int] = {}
    for side in ("left", "right"):
        kf = per_side[side]
        cycles = [(int(kf["close_start"][i]), int(kf["hold_start"][i]),
                   int(kf["hold_end"][i]), int(kf["open_full"][i]))
                  for i in range(len(kf["close_start"]))]
        cycle_lists[side] = cycles
        span_max[side] = max((cycle_hold_span(c) for c in cycles), default=-1)
    if all(s <= 0 for s in span_max.values()):
        raise ValueError("两侧均无抓取周期, 无法判定持筒臂")
    holder = max(("left", "right"), key=lambda s: span_max[s])
    pen = "right" if holder == "left" else "left"

    ep_len = len(state)
    good = [c for c in cycle_lists[holder]
            if ((c[2] if c[2] != INCOMPLETE else ep_len - 1) - c[1])
            >= holder_min_frac * ep_len]
    if not good:
        good = [max(cycle_lists[holder], key=lambda c: (
            (c[2] if c[2] != INCOMPLETE else ep_len - 1) - c[1]))]
    return {"holder": holder, "pen": pen}, {holder: good, pen: cycle_lists[pen]}


# ---------------------------------------------------------------------------
# 阶段标签 (逐帧)
# ---------------------------------------------------------------------------

def stage_windows_from_cycle(cycle: tuple, stages: list[dict], role: str,
                             T: int) -> list[dict]:
    """单个周期里命中给定 role 的阶段窗口。返回 [{key,cn,a,b,t0}], a/b 闭区间已 clip。"""
    _, hs, he, _ = cycle
    out: list[dict] = []
    for st in stages:
        if st["role"] != role:
            continue
        if st["anchor"] == "hold_end" and he == INCOMPLETE:
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
    """单 fill episode -> 逐帧阶段标签。

    返回 dict(labels: 位置空间 int, names: idx->key, spans, roles, per_side,
    holder_sure)。labels[i]=0 -> "none"; >0 -> names[l]。
    """
    stages = FILL_STAGES if stages is None else stages
    T = len(state)
    per_side: dict[str, dict[str, np.ndarray]] = {}
    for side, idx in (("left", GRIP_L), ("right", GRIP_R)):
        kf = detect_gripper_keyframes(state[:, idx], frame, **detect_kwargs)
        per_side[side] = {k: np.asarray(v, dtype=int) for k, v in kf.items()}

    roles, holder_cycles = resolve_fill_roles(state, per_side)

    spans: list[dict] = []
    for side in ("left", "right"):
        role = "holder" if side == roles["holder"] else "pen"
        for cyc in holder_cycles[side]:
            spans.extend(stage_windows_from_cycle(cyc, stages, role, T))
    spans.sort(key=lambda s: s["a"])

    names = ["none"] + [s["key"] for s in stages]
    labels = np.zeros(T, dtype=int)
    for i, s in enumerate(stages):
        for w in (w for w in spans if w["key"] == s["key"]):
            labels[w["a"]:w["b"] + 1] = i + 1

    hs, he = holder_cycles[roles["holder"]][0][1], holder_cycles[
        roles["holder"]][0][2]
    holder_end = he if he != INCOMPLETE else T - 1
    return {"labels": labels, "names": names, "spans": spans, "roles": roles,
            "per_side": per_side,
            "holder_sure": float((holder_end - hs) / T)}


# ---------------------------------------------------------------------------
# 可视化
# ---------------------------------------------------------------------------

def plot_episode_stages(
    task_name: str,          # 短标题 slug, 如 "fill_pen_holder"
    episode_index: int,
    frame: np.ndarray,
    state: np.ndarray,
    res: dict,
    out_path: Path,
) -> None:
    """3 行图: [逐帧阶段色带] [左爪夹+底纹] [右爪夹+底纹], 标题精简为 slug+ep。

    色带直接展示"每帧一个标签"; 下两行把窗口底纹叠在对应动作臂的爪夹曲线上,
    便于核对窗口是否包住真实抓/放动作。行内虚线 = 各阶段 t0 锚点。
    图例放图下方; 色带内不写字, 避免挤在一起。
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

    names = res["names"]
    labels = res["labels"]
    stage_keys = names[1:]
    cmap = mcolors.ListedColormap(
        [NONE_COLOR] + [STAGE_COLORS[k] for k in stage_keys])

    fig, axes = plt.subplots(3, 1, figsize=(13.5, 7.8), sharex=True)
    fig.suptitle(f"{task_name}  episode {episode_index:03d}",
                 fontsize=12, y=0.99)

    T = len(labels)
    # 行0: 逐帧标签色带
    ax = axes[0]
    x = np.arange(T + 1) - 0.5
    ax.pcolormesh(x, [0, 1], labels[None, :], cmap=cmap,
                  vmin=0, vmax=len(names) - 1, shading="flat")
    ax.set_yticks([]); ax.set_ylim(0, 1); ax.set_ylabel("stage")
    ax.grid(False)
    ax.set_title("浅灰=none(普通帧), 色带=该帧命中的关键阶段", loc="left",
                 fontsize=8)

    # 行1/2: 爪夹曲线 + 该臂负责的阶段窗口底纹
    for row_side, idx, color in (("left", GRIP_L, C_LEFT),
                                 ("right", GRIP_R, C_RIGHT)):
        axg = axes[1] if row_side == "left" else axes[2]
        for s in res["spans"]:
            arm = res["roles"]["holder"] if s["role"] == "holder" \
                else res["roles"]["pen"]
            if arm != row_side:
                continue
            axg.axvspan(frame[s["a"]], frame[s["b"]],
                        color=STAGE_COLORS[s["key"]], alpha=0.20, lw=0)
            axg.axvline(frame[s["t0"]], color=STAGE_COLORS[s["key"]],
                        ls="--", lw=0.9, alpha=0.55)
        axg.plot(frame, state[:, idx], color=color, lw=1.1)
        axg.set_ylabel("gripper")
        axg.set_ylim(-0.05, 1.05)
        role_cn = "持筒臂" if res["roles"]["holder"] == row_side else "笔臂"
        axg.set_title(f"{'Left' if row_side == 'left' else 'Right'} gripper "
                      f"({role_cn})", loc="left", fontsize=9)
        axg.grid(alpha=0.25)
    axes[2].set_xlabel("frame index")

    # 图例 (放图下方, 条目从简)
    stage_cn = {s["key"]: s["cn"] for s in FILL_STAGES}
    handles = [
        plt.Line2D([0], [0], color=C_LEFT, lw=1.5, label="left gripper"),
        plt.Line2D([0], [0], color=C_RIGHT, lw=1.5, label="right gripper"),
        plt.Rectangle((0, 0), 1, 1, color=NONE_COLOR, label="none(普通帧)"),
    ] + [plt.Rectangle((0, 0), 1, 1, color=STAGE_COLORS[k],
                       label=stage_cn[k]) for k in stage_keys]
    fig.legend(handles=handles, fontsize=8, ncol=len(handles), frameon=False,
               loc="lower center", bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 共用: 加载 & 统计
# ---------------------------------------------------------------------------

def load_common(args) -> tuple:
    detect_kwargs = {
        "min_prominence": args.min_prominence,
        "hold_min_len": args.hold_min_len,
        "open_level": args.open_level,
        "allow_incomplete": not args.no_incomplete,
    }
    small = load_state_cols(Path(args.csv))
    eps_by = (small.groupby("task_index")["episode_index"]
              .apply(lambda s: np.sort(s.unique())).to_dict())
    task_names = load_tasks(ROOT / "data" / "sim_lerobot_v30_ee" / "meta"
                            / "tasks.parquet") if (ROOT / "data"
                            / "sim_lerobot_v30_ee" / "meta"
                            / "tasks.parquet").is_file() else {}
    selected = ([int(x) for x in args.tasks.split(",")] if args.tasks != "all"
                else sorted(eps_by))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    return detect_kwargs, small, eps_by, task_names, selected, out_dir


def add_detect_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--csv", default=str(DEFAULT_CSV), help="sim CSV 路径")
    p.add_argument("--tasks", default="0",
                   help="task_index, 逗号分隔 (目前仅 0=fill_pen_holder 有阶段表)")
    p.add_argument("--min-prominence", type=float, default=0.2)
    p.add_argument("--hold-min-len", type=int, default=3)
    p.add_argument("--open-level", type=float, default=0.9)
    p.add_argument("--no-incomplete", action="store_true")
    p.add_argument("--out", default=str(DEFAULT_OUT))


def episode_row_table(res: dict, frame: np.ndarray) -> list[str]:
    """该 ep 的阶段命中明细 -> 文本行。"""
    lines = [f"    ep roles holder={res['roles']['holder']} "
             f"pen={res['roles']['pen']} holder_span={res['holder_sure']:.0%}"]
    cn = {s["key"]: s["cn"] for s in FILL_STAGES}
    for s in res["spans"]:
        arm = res["roles"]["holder"] if s["role"] == "holder" \
            else res["roles"]["pen"]
        lines.append(f"        {cn[s['key']]:<5}[{s['role']:<7}{arm}] "
                     f"t0={frame[s['t0']]:4d}  frames {frame[s['a']]:4d}"
                     f"-{frame[s['b']]:4d}")
    return lines


# ---------------------------------------------------------------------------
# viz: 抽样画图核对
# ---------------------------------------------------------------------------

def cmd_viz(args) -> None:
    p = argparse.ArgumentParser(add_help=False)
    add_detect_args(p)
    p.add_argument("--per-task", type=int, default=3)
    p.add_argument("--episodes", default=None,
                   help="指定 ep(逗号分隔, 覆盖 --per-task)")
    a = p.parse_args(args)
    (dk, small, eps_by, task_names, selected, out_dir) = load_common(a)
    want_eps = [int(x) for x in a.episodes.split(",")] if a.episodes else None

    summary: dict = {}
    n_saved = 0
    for ti in selected:
        if ti != 0:
            print(f"  ! task_index={ti} 暂无阶段表, 跳过 (当前仅 fill 0 实现)")
            continue
        slug = TASK_SLUGS.get(ti, f"task_{ti:03d}")
        full_name = task_names.get(ti, slug)
        eps = eps_by[ti]
        pick = (np.asarray(want_eps, dtype=int) if want_eps is not None
                else pick_episodes(eps, a.per_task))
        pick = pick[np.isin(pick, eps)]
        print(f"  task {ti} ({slug}): episodes {pick.tolist()}")
        for ep in [int(x) for x in pick]:
            sub = small[small["episode_index"] == ep].sort_values("frame_index")
            state = parse_state_series(sub["observation.state"])
            frame = sub["frame_index"].to_numpy()
            res = annotate_fill_episode(state, frame, dk)
            out_png = out_dir / f"{slug}_ep{ep:03d}_stages.png"
            plot_episode_stages(slug, ep, frame, state, res, out_png)
            n_saved += 1
            per_lbl = label_share(res)
            print("\n".join(episode_row_table(res, frame)))
            print(f"        share: {per_lbl}")
            summary[f"task{ti}_ep{ep}"] = episode_summary(res, frame)
    (out_dir / "stage_bands_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n{n_saved} stage figures -> {out_dir}")


# ---------------------------------------------------------------------------
# apply: 全量打标 + 右持筒验证图
# ---------------------------------------------------------------------------

def label_share(res: dict) -> dict:
    n = len(res["labels"])
    return {k: round(100.0 * int((res["labels"] == i).sum()) / n, 2)
            for i, k in enumerate(res["names"])}


def episode_summary(res: dict, frame: np.ndarray) -> dict:
    return {
        "n_frames": int(len(frame)),
        "roles": res["roles"],
        "holder_span_frac": round(res["holder_sure"], 4),
        "spans": [{"stage": s["key"], "cn": s["cn"], "role": s["role"],
                   "arm": res["roles"]["holder"] if s["role"] == "holder"
                   else res["roles"]["pen"],
                   "t0": int(frame[s["t0"]]), "a": int(frame[s["a"]]),
                   "b": int(frame[s["b"]])} for s in res["spans"]],
        "label_pct": label_share(res),
    }


def cmd_apply(args) -> None:
    p = argparse.ArgumentParser(add_help=False)
    add_detect_args(p)
    p.add_argument("--right-n", type=int, default=3,
                   help="自动挑选画验证图的右持筒 episode 数")
    a = p.parse_args(args)
    (dk, small, eps_by, task_names, selected, out_dir) = load_common(a)

    rows: list[pd.DataFrame] = []
    per_episode: dict[str, dict] = {}
    n_frame = n_right = n_left = 0
    agg = {k: 0 for k in ["none", "grasp_holder", "place_holder",
                          "grasp_pen", "place_pen"]}
    for ti in selected:
        if ti != 0:
            print(f"  ! task_index={ti} 暂无阶段表, 跳过 (当前仅 fill 0)")
            continue
        slug = TASK_SLUGS.get(ti, f"task_{ti:03d}")
        full_name = task_names.get(ti, slug)
        eps = [int(e) for e in eps_by[ti]]
        print(f"  task {ti} ({slug}): {len(eps)} episodes ...")
        for ep in eps:
            sub = small[small["episode_index"] == ep].sort_values("frame_index")
            state = parse_state_series(sub["observation.state"])
            frame = sub["frame_index"].to_numpy()
            res = annotate_fill_episode(state, frame, dk)
            rows.append(pd.DataFrame({
                "episode_index": ep, "frame_index": frame,
                "stage_label": [res["names"][int(i)] for i in res["labels"]],
            }))
            n_frame += len(frame)
            if res["roles"]["holder"] == "right":
                n_right += 1
            else:
                n_left += 1
            for i, k in enumerate(res["names"]):
                agg[k] += int((res["labels"] == i).sum())
            per_episode[f"task{ti}_ep{ep}"] = episode_summary(res, frame)

    # 写精简 CSV
    all_rows = pd.concat(rows, ignore_index=True)
    csv_name = f"{slug}_stage_label.csv"
    all_rows.to_csv(out_dir / csv_name, index=False)
    print(f"\nper-frame stage CSV -> {out_dir / csv_name} "
          f"({len(all_rows):,} rows)")

    # 统计 + JSON
    stats = {
        "csv": csv_name, "stage_def": FILL_STAGES,
        "total": {"episodes": len(eps), "frames": n_frame,
                  "holder_left": n_left, "holder_right": n_right,
                  "label_share_pct": {k: round(100.0 * v / n_frame, 2)
                                      for k, v in agg.items()}},
        "per_episode": per_episode,
    }
    stats_path = out_dir / "stage_apply_summary.json"
    stats_path.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"summary -> {stats_path}")

    # 右持筒验证图
    right_eps = sorted(int(e) for e in eps_by[ti]
                       if per_episode[f"task{ti}_ep{e}"]["roles"]["holder"]
                       == "right")
    print(f"\n持筒臂分布: left={n_left}  right={n_right}")
    if right_eps:
        pick = pick_episodes(np.asarray(right_eps), min(a.right_n, len(right_eps)))
        print(f"右持筒 episode 全列: {right_eps}")
        print(f"验证抽样: {pick.tolist()}")
        for ep in [int(x) for x in pick]:
            sub = small[small["episode_index"] == ep].sort_values("frame_index")
            state = parse_state_series(sub["observation.state"])
            frame = sub["frame_index"].to_numpy()
            res = annotate_fill_episode(state, frame, dk)
            out_png = out_dir / f"{slug}_ep{ep:03d}_stages.png"
            plot_episode_stages(slug, ep, frame, state, res, out_png)
            print(f"    ep{ep:03d} (holder=right) -> {out_png.name}")
            print("\n".join(episode_row_table(res, frame)))
    else:
        print("  ! 本批没有右持筒 episode")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", default="viz",
                        choices=["viz", "apply"])
    args, rest = parser.parse_known_args()
    if args.command == "apply":
        cmd_apply(rest)
    else:
        cmd_viz(rest)


if __name__ == "__main__":
    main()
