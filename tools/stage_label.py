#!/usr/bin/env python3
"""sim_lerobot_v30_ee 关键阶段标注: 每帧一个关键阶段标签 + 可视化 (fill/plug 已实现)。

在关键帧检测 (detect_gripper_keyframes, 见 tools/keyframe_detect.py) 基础上, 把每个抓取
周期解析成关键事件, 事件前后扩成"阶段窗口", 给每帧赋一个阶段标签 (默认 none=普通帧)。
窗口/t0 均在"位置空间"(state 行号)计算; sim 数据集 frame_index 从 0 连续, 位置 == 帧号,
统计/输出给真实 frame_index。

各任务阶段表 (a=t0-前窗口, b=t0+后窗口; [a,b] 内标该阶段; 重叠帧按阶段表顺序后者优先):

fill_pen_holder (task 0) —— 持筒臂(抓一次长持到结尾, 可能 L/R) + 笔臂(多次抓/放笔):
    抓取笔筒  持筒臂 hold_start [t0-15, t0+5]
    放下笔筒  持筒臂 hold_end   [t0-15, t0+5]   (用户确认右界 = holder_end+5)
    抓笔      笔臂   hold_start [t0-15, t0+5]
    放笔      笔臂   hold_end   [t0-20, t0+10]
    先逐 episode 判定持筒臂(长保持周期所在臂), 另一臂为笔臂。

plug_in_charger (task 1) —— 左/右角色随 episode 互换, 周期数可为 1 或 2(部分 episode
    双交接); 以"最后一个 hold_end"(插入臂) 为插入锚点:
    抓充电器  每个周期的 hold_start(含对侧接取) [t0-15, t0+5]
    释放充电器 每个非插入周期的 hold_end(交接释放) [t0-10, t0+5]
    插入      插入周期的 hold_end(末次, L/R 均可) [t0-20, t0+10]
    插入臂 = 两侧中 hold_end 最大(最晚放开)的那个周期所在臂。

子命令:
  viz   挑若干 episode 画图人工核对 (默认每任务 3 个均匀抽样)
  apply 对全部(或指定任务)episode 逐帧打标: 写精简 CSV(episode/frame/stage_label) +
        每 ep 阶段明细 JSON, 自动抽若干"目标形态"episode 画验证图
        (fill: 右持筒; plug: 右臂插入)

用法:
    python tools/stage_label.py viz --tasks 0 --episodes 0,50,99
    python tools/stage_label.py viz --tasks 1 --episodes 100,101,102,103
    python tools/stage_label.py apply --tasks 0 --out outputs/xxx --right-n 3
    python tools/stage_label.py apply --tasks 1 --out outputs/xxx
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

NONE_COLOR = "#eceff1"

# 阶段调色板 (none 帧不画色)
STAGE_COLORS: dict[str, str] = {
    # fill
    "grasp_holder": "#2a78d6",
    "place_holder": "#b07bd8",
    "grasp_pen": "#eb6834",
    "place_pen": "#4c9f38",
    # plug
    "grasp_charger": "#2ca25f",
    "release_charger": "#9467bd",
    "insert": "#d62728",
}

# fill_pen_holder 阶段表
FILL_STAGES: list[dict] = [
    {"key": "grasp_holder", "cn": "抓取笔筒", "anchor": "hold_start",
     "pre": 15, "post": 5},
    {"key": "place_holder", "cn": "放下笔筒", "anchor": "hold_end",
     "pre": 15, "post": 5},
    {"key": "grasp_pen", "cn": "抓笔", "anchor": "hold_start",
     "pre": 15, "post": 5},
    {"key": "place_pen", "cn": "放笔", "anchor": "hold_end",
     "pre": 20, "post": 10},
]
# plug_in_charger 阶段表
PLUG_STAGES: list[dict] = [
    {"key": "grasp_charger", "cn": "抓充电器", "anchor": "hold_start",
     "pre": 15, "post": 5},
    {"key": "release_charger", "cn": "释放充电器", "anchor": "hold_end",
     "pre": 10, "post": 5},
    {"key": "insert", "cn": "插入", "anchor": "hold_end",
     "pre": 20, "post": 10},
]
TASK_STAGES: dict[int, list[dict]] = {0: FILL_STAGES, 1: PLUG_STAGES}

# fill: 持筒臂周期须跨段至少该比例才认定为"整段持筒"
HOLDER_SURE_FRAC = 0.5


# ---------------------------------------------------------------------------
# 周期工具
# ---------------------------------------------------------------------------

def side_cycles(per_side: dict, side: str) -> list[tuple[int, int, int, int]]:
    kf = per_side[side]
    return [(int(kf["close_start"][i]), int(kf["hold_start"][i]),
             int(kf["hold_end"][i]), int(kf["open_full"][i]))
            for i in range(len(kf["close_start"]))]


def detect_sides(state: np.ndarray, frame: np.ndarray,
                 detect_kwargs: dict) -> dict:
    per_side = {}
    for side, idx in (("left", GRIP_L), ("right", GRIP_R)):
        kf = detect_gripper_keyframes(state[:, idx], frame, **detect_kwargs)
        per_side[side] = {k: np.asarray(v, dtype=int) for k, v in kf.items()}
    return per_side


def window(t0: int, pre: int, post: int, T: int):
    a = max(0, t0 - pre)
    b = min(T - 1, t0 + post)
    return (a, b) if a <= b else None


# ---------------------------------------------------------------------------
# fill: 角色 + 逐帧标签
# ---------------------------------------------------------------------------

def resolve_fill_roles(state: np.ndarray,
                       per_side: dict) -> dict:
    """返回 {"holder": side, "pen": side, "good_cycles": {side: cycles}}。"""
    cycle_lists = {s: side_cycles(per_side, s) for s in ("left", "right")}
    span_max = {s: max(((c[2] if c[2] != INCOMPLETE else int(1e9)) - c[1]
                        for c in cycle_lists[s]), default=-1)
                for s in ("left", "right")}
    if all(v <= 0 for v in span_max.values()):
        raise ValueError("两侧均无抓取周期, 无法判定持筒臂")
    holder = max(("left", "right"), key=lambda s: span_max[s])
    pen = "right" if holder == "left" else "left"
    ep_len = len(state)
    good = [c for c in cycle_lists[holder]
            if ((c[2] if c[2] != INCOMPLETE else ep_len - 1) - c[1])
            >= HOLDER_SURE_FRAC * ep_len]
    if not good:
        good = [max(cycle_lists[holder], key=lambda c: (
            (c[2] if c[2] != INCOMPLETE else ep_len - 1) - c[1]))]
    return {"holder": holder, "pen": pen,
            "good_cycles": {holder: good, pen: cycle_lists[pen]}}


def annotate_task(ti: int, state: np.ndarray, frame: np.ndarray,
                  detect_kwargs: dict) -> dict:
    """按任务打标单 episode -> 统一 res dict。

    res: labels(位置空间 int), names, stages, spans[{key,cn,side,t0,a,b}],
    per_side, meta(str), good_cycles。
    """
    stages = TASK_STAGES.get(ti)
    if stages is None:
        raise SystemExit(f"task_index={ti} 暂无阶段表")
    T = len(state)
    per_side = detect_sides(state, frame, detect_kwargs)

    if ti == 0:                       # fill: 持筒臂 + 笔臂
        roles = resolve_fill_roles(state, per_side)
        spans = []
        for side in ("left", "right"):
            role = "holder" if side == roles["holder"] else "pen"
            for cyc in roles["good_cycles"][side]:
                for st in stages:
                    anchor_stage = (st["key"] in ("grasp_holder",
                                                  "place_holder")) \
                        if role == "holder" else (st["key"] in ("grasp_pen",
                                                                "place_pen"))
                    if not anchor_stage:
                        continue
                    t0 = cyc[2] if st["anchor"] == "hold_end" else cyc[1]
                    if st["anchor"] == "hold_end" and cyc[2] == INCOMPLETE:
                        continue
                    ab = window(t0, st["pre"], st["post"], T)
                    if ab:
                        spans.append({**st, "side": side, "t0": t0,
                                      "a": ab[0], "b": ab[1]})
        hs, he = roles["good_cycles"][roles["holder"]][0][1], roles[
            "good_cycles"][roles["holder"]][0][2]
        holder_span = ((he if he != INCOMPLETE else T - 1) - hs) / T
        meta = (f"holder={roles['holder']}(持筒臂 {holder_span:.0%}) "
                f"pen={roles['pen']}")
    elif ti == 1:                     # plug: 抓/交接释放/末次插入
        cycle_lists = {s: side_cycles(per_side, s) for s in ("left", "right")}
        # 插入臂 = 所有周期中 hold_end 最大者
        cands = [(s, c) for s in ("left", "right") for c in cycle_lists[s]
                 if c[2] != INCOMPLETE]
        insert_side, insert_cyc = max(cands, key=lambda x: x[1][2])
        spans = []
        for side in ("left", "right"):
            for cyc in cycle_lists[side]:
                for st in stages:
                    is_insert = (st["key"] == "insert"
                                 and cyc is insert_cyc
                                 and side == insert_side)
                    is_grasp = st["key"] == "grasp_charger" \
                        and st["anchor"] == "hold_start"
                    is_release = (st["key"] == "release_charger"
                                  and st["anchor"] == "hold_end"
                                  and not is_insert
                                  and not (side == insert_side
                                           and cyc is insert_cyc))
                    if not (is_insert or is_grasp or is_release):
                        continue
                    t0 = cyc[1] if st["anchor"] == "hold_start" else cyc[2]
                    if t0 == INCOMPLETE:
                        continue
                    ab = window(t0, st["pre"], st["post"], T)
                    if ab:
                        spans.append({**st, "side": side, "t0": t0,
                                      "a": ab[0], "b": ab[1]})
        n_cyc = {s: len(cycle_lists[s]) for s in ("left", "right")}
        meta = (f"insert={insert_side} (插入臂)  cycles L={n_cyc['left']} "
                f"R={n_cyc['right']}")
    else:
        raise SystemExit(f"task_index={ti} 暂无阶段表")

    spans.sort(key=lambda s: s["a"])
    names = ["none"] + [s["key"] for s in stages]
    labels = np.zeros(T, dtype=int)
    for i, st in enumerate(stages):
        for w in (w for w in spans if w["key"] == st["key"]):
            labels[w["a"]:w["b"] + 1] = i + 1

    return {"labels": labels, "names": names, "stages": stages, "spans": spans,
            "per_side": per_side, "meta": meta}


# ---------------------------------------------------------------------------
# 可视化
# ---------------------------------------------------------------------------

def plot_episode_stages(
    task_name: str,          # 短标题 slug
    episode_index: int,
    frame: np.ndarray,
    state: np.ndarray,
    res: dict,
    out_path: Path,
) -> None:
    """3 行图: [逐帧阶段色带] [左爪夹+底纹] [右爪夹+底纹], 标题精简 slug+ep。"""
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
    ax = axes[0]
    x = np.arange(T + 1) - 0.5
    ax.pcolormesh(x, [0, 1], labels[None, :], cmap=cmap,
                  vmin=0, vmax=len(names) - 1, shading="flat")
    ax.set_yticks([]); ax.set_ylim(0, 1); ax.set_ylabel("stage")
    ax.grid(False)
    ax.set_title("浅灰=none(普通帧), 色带=该帧命中的关键阶段", loc="left",
                 fontsize=8)

    # 行1/2: 爪夹曲线 + 动作臂负责的阶段窗口底纹
    for row_side, idx, color in (("left", GRIP_L, C_LEFT),
                                 ("right", GRIP_R, C_RIGHT)):
        axg = axes[1] if row_side == "left" else axes[2]
        for s in res["spans"]:
            if s["side"] != row_side:
                continue
            axg.axvspan(frame[s["a"]], frame[s["b"]],
                        color=STAGE_COLORS[s["key"]], alpha=0.20, lw=0)
            axg.axvline(frame[s["t0"]], color=STAGE_COLORS[s["key"]],
                        ls="--", lw=0.9, alpha=0.55)
        axg.plot(frame, state[:, idx], color=color, lw=1.1)
        axg.set_ylabel("gripper")
        axg.set_ylim(-0.05, 1.05)
        axg.set_title(f"{'Left' if row_side == 'left' else 'Right'} gripper",
                      loc="left", fontsize=9)
        axg.grid(alpha=0.25)
    axes[2].set_xlabel("frame index")

    # 图例 (图下方)
    stage_cn = {s["key"]: s["cn"] for s in res["stages"]}
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
# 共用
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
    tasks_pq = ROOT / "data" / "sim_lerobot_v30_ee" / "meta" / "tasks.parquet"
    task_names = load_tasks(tasks_pq) if tasks_pq.is_file() else {}
    selected = ([int(x) for x in args.tasks.split(",")] if args.tasks != "all"
                else sorted(eps_by))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    return detect_kwargs, small, eps_by, task_names, selected, out_dir


def add_detect_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--csv", default=str(DEFAULT_CSV), help="sim CSV 路径")
    p.add_argument("--tasks", default="0,1",
                   help="task_index, 逗号分隔 (0=fill 1=plug, 需有阶段表)")
    p.add_argument("--min-prominence", type=float, default=0.2)
    p.add_argument("--hold-min-len", type=int, default=3)
    p.add_argument("--open-level", type=float, default=0.9)
    p.add_argument("--no-incomplete", action="store_true")
    p.add_argument("--out", default=str(DEFAULT_OUT))


def label_share(res: dict) -> dict:
    n = len(res["labels"])
    return {k: round(100.0 * int((res["labels"] == i).sum()) / n, 2)
            for i, k in enumerate(res["names"])}


def span_lines(res: dict, frame: np.ndarray) -> list[str]:
    cn = {s["key"]: s["cn"] for s in res["stages"]}
    out = []
    for s in res["spans"]:
        out.append(f"        {cn[s['key']]:<6}[{s['side'][:1].upper()}] "
                   f"t0={frame[s['t0']]:4d}  frames {frame[s['a']]:4d}"
                   f"-{frame[s['b']]:4d}")
    return out


def episode_summary(res: dict, frame: np.ndarray) -> dict:
    return {
        "meta": res["meta"], "n_frames": int(len(frame)),
        "spans": [{"stage": s["key"], "cn": s["cn"], "side": s["side"],
                   "t0": int(frame[s["t0"]]), "a": int(frame[s["a"]]),
                   "b": int(frame[s["b"]])} for s in res["spans"]],
        "label_pct": label_share(res),
    }


# ---------------------------------------------------------------------------
# viz
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
        if ti not in TASK_STAGES:
            print(f"  ! task_index={ti} 暂无阶段表, 跳过")
            continue
        slug = TASK_SLUGS.get(ti, f"task_{ti:03d}")
        eps = eps_by[ti]
        pick = (np.asarray(want_eps, dtype=int) if want_eps is not None
                else pick_episodes(eps, a.per_task))
        pick = pick[np.isin(pick, eps)]
        print(f"  task {ti} ({slug}): episodes {pick.tolist()}")
        for ep in [int(x) for x in pick]:
            sub = small[small["episode_index"] == ep].sort_values("frame_index")
            state = parse_state_series(sub["observation.state"])
            frame = sub["frame_index"].to_numpy()
            res = annotate_task(ti, state, frame, dk)
            out_png = out_dir / f"{slug}_ep{ep:03d}_stages.png"
            plot_episode_stages(slug, ep, frame, state, res, out_png)
            n_saved += 1
            print(f"    ep{ep:03d}  {res['meta']}")
            print("\n".join(span_lines(res, frame)))
            print(f"        share: {label_share(res)}")
            summary[f"task{ti}_ep{ep}"] = episode_summary(res, frame)
    (out_dir / "stage_bands_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n{n_saved} stage figures -> {out_dir}")


# ---------------------------------------------------------------------------
# apply
# ---------------------------------------------------------------------------

def apply_verify_pick(ti: int, summaries: dict) -> list[int]:
    """每种形态抽样画验证图的 episode 集。

    fill: 返回右持筒 (meta 以 'holder=right' 开头) 均匀抽样;
    plug: 返回右臂插入 (meta 'insert=right') 均匀抽样。
    """
    cond = {"0": "holder=right", "1": "insert=right"}.get(str(ti))
    eps = [int(k.split("_ep")[1]) for k, v in summaries.items()
           if cond and v["meta"].startswith(cond)]
    return sorted(eps)


def cmd_apply(args) -> None:
    p = argparse.ArgumentParser(add_help=False)
    add_detect_args(p)
    p.add_argument("--right-n", type=int, default=3,
                   help="每种验证形态(目标臂)抽样画图数")
    a = p.parse_args(args)
    (dk, small, eps_by, task_names, selected, out_dir) = load_common(a)

    rows: list[pd.DataFrame] = []
    per_episode: dict[str, dict] = {}
    agg: dict = {}
    per_task: dict[int, dict] = {}
    for ti in selected:
        if ti not in TASK_STAGES:
            print(f"  ! task_index={ti} 暂无阶段表, 跳过")
            continue
        slug = TASK_SLUGS.get(ti, f"task_{ti:03d}")
        eps = [int(e) for e in eps_by[ti]]
        print(f"  task {ti} ({slug}): {len(eps)} episodes ...")
        t_agg = {k: 0 for k in ["none"] + [s["key"] for s in TASK_STAGES[ti]]}
        for ep in eps:
            sub = small[small["episode_index"] == ep].sort_values("frame_index")
            state = parse_state_series(sub["observation.state"])
            frame = sub["frame_index"].to_numpy()
            res = annotate_task(ti, state, frame, dk)
            rows.append(pd.DataFrame({
                "episode_index": ep, "frame_index": frame,
                "stage_label": [res["names"][int(i)] for i in res["labels"]],
            }))
            for i, k in enumerate(res["names"]):
                t_agg[k] += int((res["labels"] == i).sum())
            per_episode[f"task{ti}_ep{ep}"] = episode_summary(res, frame)
        per_task[ti] = {"slug": slug, "label_share_pct": {
            k: round(100.0 * v / sum(t_agg.values()), 2)
            for k, v in t_agg.items()}}
        agg.update({k: v for k, v in per_task[ti]["label_share_pct"].items()})

    # CSV
    all_rows = pd.concat(rows, ignore_index=True)
    out_csv = out_dir / "stage_label.csv"
    all_rows.to_csv(out_csv, index=False)
    print(f"\nper-frame stage CSV -> {out_csv} ({len(all_rows):,} rows)")

    stats = {
        "csv": str(out_csv),
        "per_task": per_task,
        "per_episode": per_episode,
    }
    stats_path = out_dir / "stage_apply_summary.json"
    stats_path.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"summary -> {stats_path}")
    for ti, s in per_task.items():
        print(f"  task {ti} ({s['slug']}) label share: {s['label_share_pct']}")

    # 验证图
    for ti in selected:
        if ti not in TASK_STAGES:
            continue
        slug = TASK_SLUGS.get(ti, f"task_{ti:03d}")
        verify_eps = apply_verify_pick(ti, per_episode)
        print(f"\n  task {ti} 验证形态 episodes: {verify_eps}")
        if verify_eps:
            pick = pick_episodes(np.asarray(verify_eps),
                                 min(a.right_n, len(verify_eps)))
            for ep in [int(x) for x in pick]:
                sub = small[small["episode_index"] == ep] \
                    .sort_values("frame_index")
                state = parse_state_series(sub["observation.state"])
                frame = sub["frame_index"].to_numpy()
                res = annotate_task(ti, state, frame, dk)
                out_png = out_dir / f"{slug}_ep{ep:03d}_stages.png"
                plot_episode_stages(slug, ep, frame, state, res, out_png)
                print(f"    ep{ep:03d}  {res['meta']} -> {out_png.name}")


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
