#!/usr/bin/env python3
"""sim_lerobot_v30_ee 关键阶段标注 (单标签) + 可视化 —— fill / plug / stack。

在关键帧事件范围界定 (tools/keyframe_events.py) 基础上, 给每帧赋一个"阶段标签"
(默认 none=普通帧; 重叠帧按配置事件顺序后者优先), 用于人工核对事件范围是否合理。
事件定义 (key/cn/anchor/role/窗口) 全部来自 configs/keyframe_weight_config.json。

本工具的标签是"每帧单一标签", 供阶段色带图核对; 训练用的"多标签 + 梯形权重"
见 tools/frame_weight_trapezoid.py apply (输出 data/sim_lerobot_v30_ee/frame_weight.csv)。

各任务事件语义 (角色判定见 tools/keyframe_events.py):
    fill_pen_holder: 持筒臂(保持跨度最大臂)+笔臂 → 抓/放笔筒、抓/放笔
    plug_in_charger: 双臂各周期抓取/接取, 非插入周期交接释放, 末次周期插入
    stack_bowls: 双臂各周期抓碗/放置碗

子命令:
  viz   挑若干 episode 画图人工核对 (默认每任务 3 个均匀抽样)
  apply 对全部(或指定任务)episode 逐帧打标: 写精简 CSV(episode/frame/stage_label) +
        每 ep 阶段明细 JSON, 自动抽若干"目标形态"episode 画验证图
        (fill: 右持筒; plug: 右臂插入)

用法:
    python tools/stage_label.py viz --tasks 0 --episodes 0,50,99
    python tools/stage_label.py viz --tasks 1 --episodes 100,101,102,103
    python tools/stage_label.py apply --tasks 0 --out outputs/xxx --right-n 3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import keyframe_events as ke  # noqa: E402
from tools.keyframe_detect import (  # noqa: E402, F401  (绘图常量复用)
    C_LEFT,
    C_RIGHT,
    GRIP_L,
    GRIP_R,
)

DEFAULT_CSV = ke.DEFAULT_CSV
DEFAULT_OUT = ke.ROOT / "outputs" / "stage_label_sim_v30ee"

NONE_COLOR = ke.NONE_COLOR


# ---------------------------------------------------------------------------
# 事件实例 -> 单标签标注 (复用 keyframe_events)
# ---------------------------------------------------------------------------

def annotate_task(ti: int, state: np.ndarray, frame: np.ndarray,
                  detect_kwargs: dict, config: dict) -> dict:
    """按任务打标单 episode -> res dict。

    res: labels(位置空间 int), names, stages, spans[{key,cn,side,t0,a,b}],
    per_side, meta。
    """
    info = ke.episode_event_instances(ti, state, frame, detect_kwargs, config)
    instances = info["instances"]
    stages = ke.task_events(config, ti)
    T = info["T"]

    spans = [{"key": x["key"], "cn": x["cn"], "side": x["side"], "t0": x["t0"],
              "a": x["a"], "b": x["b"]} for x in instances]
    names = [ke.NONE_LABEL] + [s["key"] for s in stages]
    labels = np.zeros(T, dtype=int)
    for i, st in enumerate(stages):           # 配置顺序后者优先
        for w in (w for w in spans if w["key"] == st["key"]):
            labels[w["a"]:w["b"] + 1] = i + 1

    return {"labels": labels, "names": names, "stages": stages, "spans": spans,
            "per_side": info["per_side"], "meta": info["meta"]}


# ---------------------------------------------------------------------------
# 可视化
# ---------------------------------------------------------------------------

def plot_episode_stages(task_name, episode_index, frame, state, res, out_path):
    """3 行图: [逐帧阶段色带] [左爪夹+底纹] [右爪夹+底纹]。"""
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
        [NONE_COLOR] + [ke.event_color(k) for k in stage_keys])

    fig, axes = plt.subplots(3, 1, figsize=(13.5, 7.8), sharex=True)
    fig.suptitle(f"{task_name}  episode {episode_index:03d}", fontsize=12, y=0.99)

    T = len(labels)
    ax = axes[0]
    x = np.arange(T + 1) - 0.5
    ax.pcolormesh(x, [0, 1], labels[None, :], cmap=cmap,
                  vmin=0, vmax=len(names) - 1, shading="flat")
    ax.set_yticks([]); ax.set_ylim(0, 1); ax.set_ylabel("stage")
    ax.grid(False)
    ax.set_title("浅灰=none(普通帧), 色带=该帧命中的关键阶段", loc="left", fontsize=8)

    for row_side, idx, color in (("left", GRIP_L, C_LEFT),
                                 ("right", GRIP_R, C_RIGHT)):
        axg = axes[1] if row_side == "left" else axes[2]
        for s in res["spans"]:
            if s["side"] != row_side:
                continue
            axg.axvspan(frame[s["a"]], frame[s["b"]],
                        color=ke.event_color(s["key"]), alpha=0.20, lw=0)
            axg.axvline(frame[s["t0"]], color=ke.event_color(s["key"]),
                        ls="--", lw=0.9, alpha=0.55)
        axg.plot(frame, state[:, idx], color=color, lw=1.1)
        axg.set_ylabel("gripper")
        axg.set_ylim(-0.05, 1.05)
        axg.set_title(f"{'Left' if row_side == 'left' else 'Right'} gripper",
                      loc="left", fontsize=9)
        axg.grid(alpha=0.25)
    axes[2].set_xlabel("frame index")

    stage_cn = {s["key"]: s["cn"] for s in res["stages"]}
    handles = [
        plt.Line2D([0], [0], color=C_LEFT, lw=1.5, label="left gripper"),
        plt.Line2D([0], [0], color=C_RIGHT, lw=1.5, label="right gripper"),
        plt.Rectangle((0, 0), 1, 1, color=NONE_COLOR, label="none(普通帧)"),
    ] + [plt.Rectangle((0, 0), 1, 1, color=ke.event_color(k),
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
    config = ke.load_config(Path(args.config) if args.config else None)
    detect_kwargs = {
        "min_prominence": args.min_prominence,
        "hold_min_len": args.hold_min_len,
        "open_level": args.open_level,
        "allow_incomplete": not args.no_incomplete,
    }
    small = ke.load_state_cols(Path(args.csv))
    eps_by = (small.groupby("task_index")["episode_index"]
              .apply(lambda s: np.sort(s.unique())).to_dict())
    task_names = ke.load_tasks(ke.DEFAULT_TASKS)
    selected = ([int(x) for x in args.tasks.split(",")] if args.tasks != "all"
                else sorted(eps_by))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    return config, detect_kwargs, small, eps_by, task_names, selected, out_dir


def add_detect_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--csv", default=str(DEFAULT_CSV), help="sim CSV 路径")
    p.add_argument("--tasks", default="0,1,2",
                   help="task_index, 逗号分隔 (0=fill 1=plug 2=stack)")
    p.add_argument("--config", default=None, help="关键帧配置 JSON")
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


def episode_state(small: pd.DataFrame, ep: int) -> tuple:
    sub = small[small["episode_index"] == ep].sort_values("frame_index")
    return ke.parse_state_series(sub["observation.state"]), sub["frame_index"].to_numpy()


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
    config, dk, small, eps_by, _names, selected, out_dir = load_common(a)
    want_eps = [int(x) for x in a.episodes.split(",")] if a.episodes else None

    summary: dict = {}
    n_saved = 0
    for ti in selected:
        if str(ti) not in config.get("tasks", {}):
            print(f"  ! task_index={ti} 未在配置中定义, 跳过"); continue
        slug = ke.task_slug(config, ti)
        eps = eps_by[ti]
        pick = (np.asarray(want_eps, dtype=int) if want_eps is not None
                else ke.pick_episodes(eps, a.per_task))
        pick = pick[np.isin(pick, eps)]
        print(f"  task {ti} ({slug}): episodes {pick.tolist()}")
        for ep in [int(x) for x in pick]:
            state, frame = episode_state(small, ep)
            res = annotate_task(ti, state, frame, dk, config)
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
    """验证形态 episode 集: 有角色/形态区分的任务过滤该形态; 无则全 episode。"""
    cond = {"0": "holder=right", "1": "insert=right"}.get(str(ti))
    return sorted(int(k.split("_ep")[1]) for k, v in summaries.items()
                  if (not cond or v["meta"].startswith(cond)))


def cmd_apply(args) -> None:
    p = argparse.ArgumentParser(add_help=False)
    add_detect_args(p)
    p.add_argument("--right-n", type=int, default=3,
                   help="每种验证形态(目标臂)抽样画图数")
    a = p.parse_args(args)
    config, dk, small, eps_by, _names, selected, out_dir = load_common(a)

    rows: list[pd.DataFrame] = []
    per_episode: dict[str, dict] = {}
    per_task: dict[int, dict] = {}
    for ti in selected:
        if str(ti) not in config.get("tasks", {}):
            print(f"  ! task_index={ti} 未在配置中定义, 跳过"); continue
        slug = ke.task_slug(config, ti)
        keys = [ev["key"] for ev in ke.task_events(config, ti)]
        eps = [int(e) for e in eps_by[ti]]
        print(f"  task {ti} ({slug}): {len(eps)} episodes ...")
        t_agg = {k: 0 for k in [ke.NONE_LABEL] + keys}
        for ep in eps:
            state, frame = episode_state(small, ep)
            res = annotate_task(ti, state, frame, dk, config)
            rows.append(pd.DataFrame({
                "episode_index": ep, "frame_index": frame,
                "stage_label": [res["names"][int(i)] for i in res["labels"]],
            }))
            for i, k in enumerate(res["names"]):
                t_agg[k] += int((res["labels"] == i).sum())
            per_episode[f"task{ti}_ep{ep}"] = episode_summary(res, frame)
        per_task[ti] = {"slug": slug, "label_share_pct": {
            k: round(100.0 * v / sum(t_agg.values()), 2) for k, v in t_agg.items()}}

    all_rows = pd.concat(rows, ignore_index=True)
    out_csv = out_dir / "stage_label.csv"
    all_rows.to_csv(out_csv, index=False)
    print(f"\nper-frame stage CSV -> {out_csv} ({len(all_rows):,} rows)")

    stats = {"csv": str(out_csv), "per_task": per_task, "per_episode": per_episode}
    stats_path = out_dir / "stage_apply_summary.json"
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n",
                          encoding="utf-8")
    print(f"summary -> {stats_path}")
    for ti, s in per_task.items():
        print(f"  task {ti} ({s['slug']}) label share: {s['label_share_pct']}")

    for ti in selected:
        if str(ti) not in config.get("tasks", {}):
            continue
        slug = ke.task_slug(config, ti)
        verify_eps = apply_verify_pick(ti, per_episode)
        print(f"\n  task {ti} 验证形态 episodes: {verify_eps}")
        if verify_eps:
            pick = ke.pick_episodes(np.asarray(verify_eps),
                                    min(a.right_n, len(verify_eps)))
            for ep in [int(x) for x in pick]:
                state, frame = episode_state(small, ep)
                res = annotate_task(ti, state, frame, dk, config)
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
