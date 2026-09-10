#!/usr/bin/env python3
"""sim_lerobot_v30_ee 关键帧权重赋值: 配置驱动的事件范围 + 非对称梯形权重 + 多标签。

方案/公式以 docs/dual_arm_tasks_failure_and_keyframe_plan.md 为准; 所有事件范围与
权重参数来自 configs/keyframe_weight_config.json (不写死在代码里)。流程:

  1. 关键节点检测 + 事件范围界定  —— tools/keyframe_events.py (复用 keyframe_detect):
     每个抓取周期按任务角色归为具体事件 (抓/放/插入...), 每个事件有参考帧 t0 与
     窗口参数 (L, R, Pl, Pr, W), 关键帧范围 = [t0-L, t0+R]。
  2. 逐帧权重 = 各事件梯形窗口取 max (不累加); 普通帧权重 1。
  3. 逐帧关键帧标签 = 命中窗口的事件 key, 多个用 '|' 连接, 无则 'none'。

三个子命令:
  preview  每任务抽样 n 个 episode, 画 权重曲线+事件色带 / 左右爪夹 3 行图 (人工确认)
  apply    对全部(或指定)episode 逐帧赋权重+标签, 原地更新目标 frame_weight.csv
           (episode_index,frame_index,frame_weight,keyframe_label) + outputs 副本 + stats
  dist     读 apply 产物(含 task_index 的副本), 画权重分布图

配置文件 schema 与范围定义详见 tools/keyframe_events.py 文件头 / 配置文件 _comment。

用法:
    python tools/frame_weight_trapezoid.py preview --per-task 3 --tasks 0,1,2
    python tools/frame_weight_trapezoid.py apply            # 更新 data/.../frame_weight.csv
    python tools/frame_weight_trapezoid.py dist --csv outputs/<...>/frame_weight_with_task.csv
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

ROOT = ke.ROOT
DEFAULT_CSV = ke.DEFAULT_CSV
DEFAULT_CONFIG = ke.DEFAULT_CONFIG
DEFAULT_TARGET = ROOT / "data" / "sim_lerobot_v30_ee" / "frame_weight.csv"
DEFAULT_OUT = ROOT / "outputs" / "frame_weight_sim_v30ee"

WEIGHT_BASE = ke.WEIGHT_BASE


# ---------------------------------------------------------------------------
# 可视化
# ---------------------------------------------------------------------------

def plot_episode_weight(
    task_name: str,
    episode_index: int,
    frame: np.ndarray,
    state: np.ndarray,
    res: dict,
    max_w: float,
    out_path: Path,
) -> None:
    """3 行子图: [权重曲线+事件色带] [左爪夹] [右爪夹]。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from tools.keyframe_detect import (C_LEFT, C_RIGHT, GRID, INK, KEYFRAME_STYLES,
                                       MUT, SURF)

    plt.rcParams.update({
        "axes.edgecolor": MUT, "axes.labelcolor": INK, "axes.titlecolor": INK,
        "text.color": INK, "xtick.color": INK, "ytick.color": INK,
        "figure.facecolor": SURF, "axes.facecolor": SURF, "grid.color": GRID,
        "font.family": "sans-serif", "figure.dpi": 110,
        "font.sans-serif": ["PingFang SC", "Hiragino Sans GB", "Heiti SC",
                            "Arial Unicode MS", "DejaVu Sans"],
        "axes.unicode_minus": False,
    })

    weight = res["weight"]
    instances = res["instances"]
    per_side = res["per_side"]

    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    fig.suptitle(f"{task_name}\nepisode {episode_index}  ({len(frame)} frames, "
                 f"{len(instances)} events)  {res['meta']}",
                 fontsize=11, y=0.995)

    # 行0: 事件色带 + 权重曲线 + t0 标记
    ax = axes[0]
    for inst in instances:
        col = ke.event_color(inst["key"])
        ax.axvspan(frame[inst["a"]], frame[inst["b"]], color=col, alpha=0.16, lw=0)
        ax.axvline(frame[inst["t0"]], color=col, lw=0.9, alpha=0.6, zorder=0)
    ax.fill_between(frame, WEIGHT_BASE, weight, color="#f2a900", alpha=0.35, lw=0)
    ax.plot(frame, weight, color="#c47b00", lw=1.0)
    ax.axhline(WEIGHT_BASE, color="0.5", lw=0.8, ls=":")
    ax.set_ylabel("weight")
    ax.set_ylim(0.9, max_w + 0.25)
    ax.set_yticks(sorted({1.0, 2.0, round(max_w, 1)}))
    ax.set_title("frame weight (max over events) + event ranges (colored bands)",
                 loc="left", fontsize=9)
    ax.grid(alpha=0.25)

    # 行1/2: 爪夹曲线 + 关键节点竖线
    for axg, name, idx, color in [
        (axes[1], "Left gripper", ke.GRIP_L, C_LEFT),
        (axes[2], "Right gripper", ke.GRIP_R, C_RIGHT),
    ]:
        axg.plot(frame, state[:, idx], color=color, lw=1.1)
        axg.set_ylabel("gripper")
        axg.set_ylim(-0.05, 1.05)
        side = "left" if idx == ke.GRIP_L else "right"
        for key, lc, ls, _ in KEYFRAME_STYLES:
            for x in per_side[side][key]:
                if x == ke.INCOMPLETE:
                    continue
                axg.axvline(x, color=lc, ls=ls, alpha=0.8, lw=1.0)
        axg.set_title(name, loc="left", fontsize=9)
        axg.grid(alpha=0.25)
    axes[2].set_xlabel("frame index")

    # 图例: 爪夹 + 本 episode 出现的事件
    keys = sorted({inst["key"] for inst in instances})
    cn = {inst["key"]: inst["cn"] for inst in instances}
    handles = [
        plt.Line2D([0], [0], color=C_LEFT, lw=1.5, label="left gripper"),
        plt.Line2D([0], [0], color=C_RIGHT, lw=1.5, label="right gripper"),
    ] + [plt.Rectangle((0, 0), 1, 1, color=ke.event_color(k),
                       label=f"{k}({cn.get(k, k)})") for k in keys]
    fig.legend(handles=handles, fontsize=8, ncol=min(6, len(handles)),
               frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.975))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_weight_distribution(stats_by_task: dict[int, dict],
                             episode_records: list[dict],
                             max_w: float, out_path: Path) -> None:
    """权重分布图: 上=逐帧权重直方(log y), 下=逐 episode 的 weight=1 占比直方。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {0: "#2a78d6", 1: "#eb6834", 2: "#4c9f38"}
    fig, axes = plt.subplots(2, 1, figsize=(11, 8))
    fig.suptitle("frame weight distribution (sim_lerobot_v30_ee, "
                 "trapezoid max-over-events)", fontsize=12, y=0.99)

    ax = axes[0]
    binw = 0.05
    bins = np.arange(1.0 - binw / 2, max_w + binw, binw)
    for ti, s in sorted(stats_by_task.items()):
        ax.hist(s["_all_weights"], bins=bins, alpha=0.55, log=True,
                color=colors.get(ti, "0.5"), label=f"{s['slug']}: n={s['frames']:,}")
    ax.set_xlim(1.0, max_w + 0.05)
    ax.set_xlabel("frame weight")
    ax.set_ylabel("frames (log)")
    ax.set_title("per-frame weight histogram (log y; spike at 1 = unboosted)",
                 loc="left", fontsize=9)
    ax.legend(fontsize=8); ax.grid(alpha=0.25)

    ax = axes[1]
    for ti in sorted({r["task_index"] for r in episode_records}):
        fr = [r["eq1_pct"] for r in episode_records if r["task_index"] == ti]
        ax.hist(fr, bins=np.arange(0, 101, 5), alpha=0.55,
                color=colors.get(ti, "0.5"), label=stats_by_task[ti]["slug"])
    ax.set_xlabel("episode weight=1 fraction (%)")
    ax.set_ylabel("episodes")
    ax.set_title("per-episode share of weight==1 frames (unboosted coverage)",
                 loc="left", fontsize=9)
    ax.legend(fontsize=8); ax.grid(alpha=0.25)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 公共参数
# ---------------------------------------------------------------------------

def common_parser_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--csv", default=str(DEFAULT_CSV), help="sim CSV 路径")
    p.add_argument("--tasks", default="all",
                   help="task_index, 逗号分隔 (如 '0,2') 或 'all'")
    p.add_argument("--config", default=None,
                   help=f"关键帧配置 JSON (默认 {DEFAULT_CONFIG})")
    p.add_argument("--min-prominence", type=float, default=0.2)
    p.add_argument("--hold-min-len", type=int, default=3)
    p.add_argument("--open-level", type=float, default=0.9)
    p.add_argument("--no-incomplete", action="store_true")
    p.add_argument("--out", default=str(DEFAULT_OUT), help="输出目录")


def load_common(args) -> tuple:
    config = ke.load_config(Path(args.config) if args.config else None)
    detect_kwargs = {
        "min_prominence": args.min_prominence,
        "hold_min_len": args.hold_min_len,
        "open_level": args.open_level,
        "allow_incomplete": not args.no_incomplete,
    }
    task_names = ke.load_tasks(ke.DEFAULT_TASKS)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"loading {args.csv} ...")
    small = ke.load_state_cols(Path(args.csv))
    eps_by = (small.groupby("task_index")["episode_index"]
              .apply(lambda s: np.sort(s.unique())).to_dict())
    selected = (list(eps_by) if args.tasks == "all"
                else [int(x) for x in args.tasks.split(",")])
    return (small, eps_by, task_names, config, detect_kwargs, out_dir, selected)


def episode_df(small: pd.DataFrame, ep: int) -> tuple:
    sub = small[small["episode_index"] == ep].sort_values("frame_index")
    return ke.parse_state_series(sub["observation.state"]), sub["frame_index"].to_numpy()


def max_event_weight(config: dict, ti: int) -> float:
    return max((float(ev["W"]) for ev in ke.task_events(config, ti)), default=2.0)


# ---------------------------------------------------------------------------
# preview
# ---------------------------------------------------------------------------

def cmd_preview(args) -> None:
    parser = argparse.ArgumentParser(add_help=False)
    common_parser_args(parser)
    parser.add_argument("--per-task", type=int, default=3)
    parser.add_argument("--episodes", default=None,
                        help="指定每任务 ep(逗号分隔, 覆盖 --per-task)")
    a = parser.parse_args(args)
    (small, eps_by, task_names, config, dk, out_dir, selected) = load_common(a)
    want = [int(x) for x in a.episodes.split(",")] if a.episodes else None
    summary: dict = {}
    for ti in selected:
        if ti not in eps_by:
            print(f"  ! task_index={ti} 不存在, 跳过"); continue
        slug = ke.task_slug(config, ti)
        name = task_names.get(ti, slug)
        eps = eps_by[ti]
        pick = (np.asarray(want, dtype=int) if want is not None
                else ke.pick_episodes(eps, a.per_task))
        pick = pick[np.isin(pick, eps)]
        mw = max_event_weight(config, ti)
        print(f"  task {ti} ({slug}) maxW={mw}: episodes {pick.tolist()}")
        for ep in [int(x) for x in pick]:
            state, frame = episode_df(small, ep)
            r = ke.episode_weight_and_labels(ti, state, frame, dk, config)
            out_png = out_dir / f"{slug}_ep{ep:03d}_weight.png"
            plot_episode_weight(name, ep, frame, state, r, mw, out_png)
            eq1 = int(np.isclose(r["weight"], WEIGHT_BASE).sum())
            nmulti = int(sum(ke.MULTI_SEP in x for x in r["labels"]))
            print(f"    ep {ep:3d}: T={len(frame):4d} events={len(r['instances']):2d} "
                  f"w=1={eq1:4d} ({100.0*eq1/len(frame):4.1f}%) multi-label={nmulti} "
                  f"-> {out_png.name}")
            summary[f"task{ti}_ep{ep}"] = {
                "task_index": ti, "episode_index": ep, "T": int(len(frame)),
                "meta": r["meta"],
                "instances": [{k: x[k] for k in ("key", "side", "t0", "a", "b")}
                              for x in r["instances"]],
                "weight_eq1": eq1, "weight_eq1_pct": round(100.0*eq1/len(frame), 3),
                "multi_label_frames": nmulti,
            }
    (out_dir / "preview_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\npreview done -> {out_dir}")


# ---------------------------------------------------------------------------
# apply
# ---------------------------------------------------------------------------

def cmd_apply(args) -> None:
    parser = argparse.ArgumentParser(add_help=False)
    common_parser_args(parser)
    parser.add_argument("--target", default=str(DEFAULT_TARGET),
                        help="原地更新的 frame_weight.csv (ep,frame,weight[,label])")
    a = parser.parse_args(args)
    (small, eps_by, task_names, config, dk, out_dir, selected) = load_common(a)

    rows: list[pd.DataFrame] = []
    per_episode: dict[str, dict] = {}
    per_task: dict[int, dict] = {}
    n_ep = n_frame = n_eq1 = 0
    for ti in selected:
        if ti not in eps_by:
            print(f"  ! task_index={ti} 不存在, 跳过"); continue
        slug = ke.task_slug(config, ti)
        keys = [ev["key"] for ev in ke.task_events(config, ti)]
        eps = [int(e) for e in eps_by[ti]]
        print(f"  task {ti} ({slug}): {len(eps)} episodes ...")
        cnt = {k: 0 for k in [ke.NONE_LABEL] + keys}   # 命中该事件(含多标签中)的帧数
        t_frame = t_eq1 = t_multi = 0
        for ep in eps:
            state, frame = episode_df(small, ep)
            r = ke.episode_weight_and_labels(ti, state, frame, dk, config)
            eq1 = int(np.isclose(r["weight"], WEIGHT_BASE).sum())
            multi = int(sum(ke.MULTI_SEP in x for x in r["labels"]))
            rows.append(pd.DataFrame({
                "episode_index": ep, "task_index": ti, "frame_index": frame,
                "frame_weight": r["weight"], "keyframe_label": r["labels"],
            }))
            for x in r["labels"]:
                if x == ke.NONE_LABEL:
                    cnt[ke.NONE_LABEL] += 1
                else:
                    for k in x.split(ke.MULTI_SEP):
                        cnt[k] += 1
            n_ep += 1; n_frame += len(frame); n_eq1 += eq1
            t_frame += len(frame); t_eq1 += eq1; t_multi += multi
            per_episode[f"task{ti}_ep{ep}"] = {
                "task_index": ti, "episode_index": ep, "T": int(len(frame)),
                "meta": r["meta"], "weight_eq1": eq1,
                "weight_gt1": int(len(frame) - eq1),
                "eq1_pct": round(100.0*eq1/len(frame), 3),
                "multi_label_frames": multi,
                "instances": [{k: x[k] for k in ("key", "side", "t0", "a", "b")}
                              for x in r["instances"]],
            }
        per_task[ti] = {
            "slug": slug, "instruction": task_names.get(ti, slug),
            "episodes": len(eps), "frames": t_frame,
            "weight_eq1": t_eq1, "weight_gt1": t_frame - t_eq1,
            "weight_eq1_pct": round(100.0*t_eq1/t_frame, 3),
            "multi_label_frames": t_multi,
            "event_frame_share_pct": {k: round(100.0*v/t_frame, 2)
                                      for k, v in cnt.items()},
        }

    key = pd.concat(rows, ignore_index=True)

    # 1) 更新数据目标 frame_weight.csv (ep, frame, weight, label): 以目标行序为基准对齐
    target = Path(a.target)
    if target.is_file():
        base = pd.read_csv(target)[["episode_index", "frame_index"]]
        merged = base.merge(key.drop(columns=["task_index"]),
                            on=["episode_index", "frame_index"],
                            how="left", validate="one_to_one")
        miss = int(merged["frame_weight"].isna().sum())
        if miss:
            raise RuntimeError(f"{miss} 行未对齐到目标 frame_weight.csv 的 (ep,frame)")
        merged["frame_weight"] = merged["frame_weight"].astype(float)
        out_target = merged
    else:
        out_target = key.drop(columns=["task_index"])
    out_target = out_target[["episode_index", "frame_index",
                             "frame_weight", "keyframe_label"]]
    out_target.to_csv(target, index=False)
    print(f"\n更新目标 -> {target} ({len(out_target):,} rows)")

    # 2) outputs 副本 (带 task_index, 便于 dist/分析)
    copy = key[["episode_index", "task_index", "frame_index",
                "frame_weight", "keyframe_label"]]
    copy_csv = out_dir / "frame_weight_with_task.csv"
    copy.to_csv(copy_csv, index=False)
    print(f"副本 -> {copy_csv}")

    # 3) stats
    total_eq1_pct = 100.0 * n_eq1 / n_frame
    stats = {
        "target": str(target), "copy": str(copy_csv),
        "config": str(Path(a.config) if a.config else DEFAULT_CONFIG),
        "method": "per-event trapezoid weight (max over events) + multi-label column",
        "total": {"episodes": n_ep, "frames": n_frame, "weight_eq1": n_eq1,
                  "weight_gt1": n_frame - n_eq1,
                  "weight_eq1_pct": round(total_eq1_pct, 3),
                  "multi_label_frames": sum(t["multi_label_frames"]
                                            for t in per_task.values())},
        "per_task": {str(ti): v for ti, v in per_task.items()},
        "per_episode": per_episode,
    }
    stats_path = out_dir / "weight_apply_stats.json"
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n",
                          encoding="utf-8")

    print(f"{'task':>4} {'slug':<18} {'episodes':>8} {'frames':>8} "
          f"{'w=1':>8} {'w=1 pct':>8} {'multi':>7}")
    for ti in selected:
        s = per_task.get(ti)
        if s:
            print(f"{ti:>4} {s['slug']:<18} {s['episodes']:>8} {s['frames']:>8} "
                  f"{s['weight_eq1']:>8} {s['weight_eq1_pct']:>7.2f}% "
                  f"{s['multi_label_frames']:>7}")
            print(f"       event share: {s['event_frame_share_pct']}")
    print(f"总计: {n_ep} episodes / {n_frame:,} frames, "
          f"weight=1={n_eq1:,} ({total_eq1_pct:.2f}%)")
    print(f"stats -> {stats_path}")


# ---------------------------------------------------------------------------
# dist
# ---------------------------------------------------------------------------

def cmd_dist(args) -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--csv", required=True,
                        help="apply 副本 frame_weight_with_task.csv (含 task_index)")
    parser.add_argument("--tasks", default="all")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    a = parser.parse_args(args)
    out_dir = Path(a.out); out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(a.csv, usecols=["episode_index", "task_index",
                                     "frame_index", "frame_weight"])
    df["frame_weight"] = df["frame_weight"].astype(float)
    selected = (sorted(df["task_index"].unique()) if a.tasks == "all"
                else [int(x) for x in a.tasks.split(",")])
    stats_by_task: dict[int, dict] = {}
    records: list[dict] = []
    n_frame = n_eq1 = 0
    for ti in selected:
        sub = df[df["task_index"] == ti]
        eq1 = int(np.isclose(sub["frame_weight"], WEIGHT_BASE).sum())
        n_frame += len(sub); n_eq1 += eq1
        for ep, g in sub.groupby("episode_index"):
            e1 = int(np.isclose(g["frame_weight"], WEIGHT_BASE).sum())
            records.append({"task_index": ti, "episode_index": int(ep),
                            "eq1_pct": round(100.0*e1/len(g), 2)})
        stats_by_task[ti] = {
            "slug": ke.TASK_SLUGS.get(ti, f"task_{ti:03d}"),
            "frames": int(len(sub)), "eq1": eq1,
            "eq1_pct": round(100.0*eq1/len(sub), 2),
            "_all_weights": sub["frame_weight"].to_numpy(),
        }
        print(f"  task {ti} ({stats_by_task[ti]['slug']}): frames={len(sub):,} "
              f"w=1={eq1:,} ({stats_by_task[ti]['eq1_pct']:.2f}%)")
    print(f"total: {n_frame:,} frames, w=1={n_eq1:,} "
          f"({100.0*n_eq1/n_frame:.2f}%)")
    out_png = out_dir / "weight_distribution.png"
    plot_weight_distribution(stats_by_task, records, df["frame_weight"].max(), out_png)
    print(f"distribution figure -> {out_png}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("preview", help="每任务抽样 n ep 画 权重+事件+爪夹 图")
    sub.add_parser("apply", help="全量赋权重+标签, 更新 frame_weight.csv + 副本 + stats")
    sub.add_parser("dist", help="读 apply 副本, 画权重分布图")
    args, rest = parser.parse_known_args()
    if args.command == "preview":
        cmd_preview(rest)
    elif args.command == "apply":
        cmd_apply(rest)
    elif args.command == "dist":
        cmd_dist(rest)


if __name__ == "__main__":
    main()
