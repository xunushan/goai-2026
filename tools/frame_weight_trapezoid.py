#!/usr/bin/env python3
"""sim_lerobot_v30_ee 关键帧训练权重生成 (配置驱动): 梯形 loss 权重 + 常数采样权重 + 多标签。

方案/公式以 docs/dual_arm_tasks_failure_and_keyframe_plan.md (v2) 为准; 所有事件范围与
权重参数来自 configs/keyframe_weight_config.json (不写死在代码里)。流程:

  1. 关键节点检测 + 事件范围界定  —— tools/keyframe_events.py (复用 keyframe_detect):
     每个抓取周期按任务角色归为具体事件 (抓/放/接取/交接/插入), 每个事件有参考帧 t0 与
     窗口参数 (L, R, Pl, Pr, W), 关键帧范围 = [t0-L, t0+R]。
  2. frame_weight_loss     = 各事件梯形窗口逐帧取 max (不累加); 窗口外为 1。
  3. is_key_frame          = 落在任一事件窗口闭区间的二值标记 (不由 loss>1 反推)。
  4. frame_weight_sampling = 关键帧常数(默认 2) / 普通帧 1。
  5. keyframe_label        = 命中窗口的事件 key, 多个用 '|' 连接, 无则 'none'。

子命令:
  preview       每任务抽样 n 个 episode, 画 loss/sampling 曲线+事件色带 / 左右爪夹 (人工确认)
  apply         对全部(或指定)episode 逐帧赋权重+标签, 原地更新目标 frame_weight.csv
                (episode_index,frame_index,keyframe_label,is_key_frame,
                 frame_weight_loss,frame_weight_sampling) + outputs 副本 + stats
  dist          读 apply 产物(含 task_index 的副本), 画权重分布图
  align         每任务抽样 n ep, 把 loss/sampling 与左右爪夹曲线对齐标注 (人工确认)
  merge         把权重表三列按 (episode_index,frame_index) 并入训练集主表 CSV
  merge-dataset 把三个训练字段(frame_weight_loss / frame_weight_sampling /
                is_key_frame)写进 LeRobot v3 数据集: 数据 parquet 追加列 +
                meta/info.json features + meta/stats.json (按 (episode_index,frame_index)
                对齐, 行序不变; 幂等, 重复执行只覆盖这三列)

配置文件 schema 与范围定义详见 tools/keyframe_events.py 文件头 / 配置文件 _comment。

用法:
    python tools/frame_weight_trapezoid.py preview --per-task 3 --tasks 0,1,2
    python tools/frame_weight_trapezoid.py apply            # 更新 data/.../frame_weight.csv
    python tools/frame_weight_trapezoid.py dist --csv outputs/<...>/frame_weight_with_task.csv
    python tools/frame_weight_trapezoid.py align --per-task 3 --out outputs/frame_weight_align
    python tools/frame_weight_trapezoid.py merge --dataset data/.../sim_lerobot_v30_ee.csv
    python tools/frame_weight_trapezoid.py merge-dataset --dataset-dir data/sim_lerobot_v30_ee
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

    weight = res["weight_loss"]
    sampling = res["weight_sampling"]
    instances = res["instances"]
    per_side = res["per_side"]
    key_v = float(sampling.max()) if len(sampling) else ke.SAMPLING_KEY_VALUE

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
    ax.plot(frame, weight, color="#c47b00", lw=1.0, label="frame_weight_loss")
    ax.step(frame, sampling, where="mid", color="#1f7a8c", lw=1.0, ls="--",
            label="frame_weight_sampling")
    ax.axhline(WEIGHT_BASE, color="0.5", lw=0.8, ls=":")
    ax.set_ylabel("weight (loss / sampling)")
    ax.set_ylim(0.9, max(max_w, key_v) + 0.25)
    ax.set_yticks(sorted({1.0, round(max_w, 2), round(key_v, 2)}))
    ax.set_title("frame_weight_loss (trapezoid max over events) + event ranges "
                 "(colored bands);  dashed step = frame_weight_sampling",
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
        plt.Line2D([0], [0], color="#c47b00", lw=1.5, label="frame_weight_loss"),
        plt.Line2D([0], [0], color="#1f7a8c", lw=1.5, ls="--",
                   label="frame_weight_sampling"),
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
    """权重分布图: 上=逐帧 frame_weight_loss 直方(log y), 下=逐 episode 的 loss=1 占比直方。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {0: "#2a78d6", 1: "#eb6834", 2: "#4c9f38"}
    fig, axes = plt.subplots(3, 1, figsize=(11, 11))
    fig.suptitle("frame_weight_loss / frame_weight_sampling distribution "
                 "(sim_lerobot_v30_ee)", fontsize=12, y=0.995)

    ax = axes[0]
    binw = 0.025
    bins = np.arange(1.0 - binw / 2, max_w + binw, binw)
    for ti, s in sorted(stats_by_task.items()):
        ax.hist(s["_all_loss"], bins=bins, alpha=0.55, log=True,
                color=colors.get(ti, "0.5"), label=f"{s['slug']}: n={s['frames']:,}")
    ax.set_xlim(1.0, max_w + 0.05)
    ax.set_xlabel("frame_weight_loss")
    ax.set_ylabel("frames (log)")
    ax.set_title("per-frame frame_weight_loss histogram (log y; spike at 1 = 窗口外)",
                 loc="left", fontsize=9)
    ax.legend(fontsize=8); ax.grid(alpha=0.25)

    ax = axes[1]
    for ti in sorted({r["task_index"] for r in episode_records}):
        fr = [r["eq1_pct"] for r in episode_records if r["task_index"] == ti]
        ax.hist(fr, bins=np.arange(0, 101, 5), alpha=0.55,
                color=colors.get(ti, "0.5"), label=stats_by_task[ti]["slug"])
    ax.set_xlabel("episode frame_weight_loss==1 fraction (%)")
    ax.set_ylabel("episodes")
    ax.set_title("per-episode share of frame_weight_loss==1 frames", loc="left",
                 fontsize=9)
    ax.legend(fontsize=8); ax.grid(alpha=0.25)

    ax = axes[2]
    for ti in sorted({r["task_index"] for r in episode_records}):
        fr = [r["key_pct"] for r in episode_records if r["task_index"] == ti]
        ax.hist(fr, bins=np.arange(0, 101, 5), alpha=0.55,
                color=colors.get(ti, "0.5"), label=stats_by_task[ti]["slug"])
    ax.set_xlabel("episode is_key_frame fraction (%)")
    ax.set_ylabel("episodes")
    ax.set_title("per-episode share of is_key_frame frames "
                 "(= frame_weight_sampling>1)", loc="left", fontsize=9)
    ax.legend(fontsize=8); ax.grid(alpha=0.25)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
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
        key_v, _ = ke.sampling_values(config)
        print(f"  task {ti} ({slug}) loss_maxW={mw} sampling_key={key_v}: "
              f"episodes {pick.tolist()}")
        for ep in [int(x) for x in pick]:
            state, frame = episode_df(small, ep)
            r = ke.episode_weight_and_labels(ti, state, frame, dk, config)
            out_png = out_dir / f"{slug}_ep{ep:03d}_weight.png"
            plot_episode_weight(name, ep, frame, state, r, mw, out_png)
            eq1 = int(np.isclose(r["weight_loss"], WEIGHT_BASE).sum())
            nkey = int(r["is_key_frame"].sum())
            nmulti = int(sum(ke.MULTI_SEP in x for x in r["labels"]))
            print(f"    ep {ep:3d}: T={len(frame):4d} events={len(r['instances']):2d} "
                  f"key={nkey:4d} ({100.0*nkey/len(frame):4.1f}%) "
                  f"loss=1 {100.0*eq1/len(frame):.1f}% multi-label={nmulti} "
                  f"-> {out_png.name}")
            summary[f"task{ti}_ep{ep}"] = {
                "task_index": ti, "episode_index": ep, "T": int(len(frame)),
                "meta": r["meta"],
                "instances": [{k: x[k] for k in ("key", "side", "t0", "a", "b")}
                              for x in r["instances"]],
                "is_key_frames": nkey, "is_key_frame_pct": round(100.0*nkey/len(frame), 3),
                "loss_eq1": eq1, "loss_eq1_pct": round(100.0*eq1/len(frame), 3),
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
    n_ep = n_frame = n_eq1 = n_key = 0
    for ti in selected:
        if ti not in eps_by:
            print(f"  ! task_index={ti} 不存在, 跳过"); continue
        slug = ke.task_slug(config, ti)
        keys = [ev["key"] for ev in ke.task_events(config, ti)]
        eps = [int(e) for e in eps_by[ti]]
        print(f"  task {ti} ({slug}): {len(eps)} episodes ...")
        cnt = {k: 0 for k in [ke.NONE_LABEL] + keys}   # 命中该事件(含多标签中)的帧数
        t_frame = t_eq1 = t_multi = t_key = 0
        for ep in eps:
            state, frame = episode_df(small, ep)
            r = ke.episode_weight_and_labels(ti, state, frame, dk, config)
            eq1 = int(np.isclose(r["weight_loss"], WEIGHT_BASE).sum())
            multi = int(sum(ke.MULTI_SEP in x for x in r["labels"]))
            rows.append(pd.DataFrame({
                "episode_index": ep, "task_index": ti, "frame_index": frame,
                "keyframe_label": r["labels"],
                "is_key_frame": r["is_key_frame"].astype(int),
                "frame_weight_loss": r["weight_loss"],
                "frame_weight_sampling": r["weight_sampling"],
            }))
            for x in r["labels"]:
                if x == ke.NONE_LABEL:
                    cnt[ke.NONE_LABEL] += 1
                else:
                    for k in x.split(ke.MULTI_SEP):
                        cnt[k] += 1
            n_ep += 1; n_frame += len(frame); n_eq1 += eq1
            n_key += int(r["is_key_frame"].sum())
            t_frame += len(frame); t_eq1 += eq1; t_multi += multi
            t_key += int(r["is_key_frame"].sum())
            per_episode[f"task{ti}_ep{ep}"] = {
                "task_index": ti, "episode_index": ep, "T": int(len(frame)),
                "meta": r["meta"], "loss_eq1": eq1,
                "loss_gt1": int(len(frame) - eq1),
                "loss_eq1_pct": round(100.0*eq1/len(frame), 3),
                "is_key_frames": int(r["is_key_frame"].sum()),
                "is_key_frame_pct": round(100.0*r["is_key_frame"].mean(), 3),
                "multi_label_frames": multi,
                "instances": [{k: x[k] for k in ("key", "side", "t0", "a", "b")}
                              for x in r["instances"]],
            }
        per_task[ti] = {
            "slug": slug, "instruction": task_names.get(ti, slug),
            "episodes": len(eps), "frames": t_frame,
            "loss_eq1": t_eq1, "loss_gt1": t_frame - t_eq1,
            "loss_eq1_pct": round(100.0*t_eq1/t_frame, 3),
            "is_key_frames": t_key,
            "is_key_frame_pct": round(100.0*t_key/t_frame, 3),
            "multi_label_frames": t_multi,
            "event_frame_share_pct": {k: round(100.0*v/t_frame, 2)
                                      for k, v in cnt.items()},
        }

    key = pd.concat(rows, ignore_index=True)

    # 1) 更新数据目标 frame_weight.csv: 以目标行序为基准对齐 (缺文件则按计算顺序写)
    target = Path(a.target)
    cols = ["episode_index", "frame_index", "keyframe_label", "is_key_frame",
            "frame_weight_loss", "frame_weight_sampling"]
    if target.is_file():
        base = pd.read_csv(target)[["episode_index", "frame_index"]]
        merged = base.merge(key.drop(columns=["task_index"]),
                            on=["episode_index", "frame_index"],
                            how="left", validate="one_to_one")
        miss = int(merged["frame_weight_loss"].isna().sum())
        if miss:
            raise RuntimeError(f"{miss} 行未对齐到目标 frame_weight.csv 的 (ep,frame)")
        merged["frame_weight_loss"] = merged["frame_weight_loss"].astype(float)
        merged["frame_weight_sampling"] = merged["frame_weight_sampling"].astype(float)
        out_target = merged
    else:
        out_target = key.drop(columns=["task_index"])
    out_target = out_target[cols]
    out_target.to_csv(target, index=False)
    print(f"\n更新目标 -> {target} ({len(out_target):,} rows, cols={cols})")

    # 2) outputs 副本 (带 task_index, 便于 dist/分析)
    copy = key[["episode_index", "task_index"] + cols[1:]]
    copy_csv = out_dir / "frame_weight_with_task.csv"
    copy.to_csv(copy_csv, index=False)
    print(f"副本 -> {copy_csv}")

    # 3) stats
    total_eq1_pct = 100.0 * n_eq1 / n_frame
    key_v, normal_v = ke.sampling_values(config)
    ess = (n_key * key_v + (n_frame - n_key) * normal_v) ** 2 / (
        n_key * key_v ** 2 + (n_frame - n_key) * normal_v ** 2)
    stats = {
        "target": str(target), "copy": str(copy_csv),
        "config": str(Path(a.config) if a.config else DEFAULT_CONFIG),
        "method": ("frame_weight_loss = per-event trapezoid (max over events), no "
                   "alpha mapping; frame_weight_sampling = key/normal constant; "
                   "is_key_frame = closed event windows"),
        "sampling": {"key_value": key_v, "normal_value": normal_v},
        "total": {"episodes": n_ep, "frames": n_frame,
                  "is_key_frames": n_key,
                  "is_key_frame_pct": round(100.0*n_key/n_frame, 3),
                  "loss_eq1": n_eq1, "loss_gt1": n_frame - n_eq1,
                  "loss_eq1_pct": round(total_eq1_pct, 3),
                  "ess_with_sampling": round(ess, 1),
                  "expected_key_share_after_sampling": round(
                      100.0*n_key*key_v/(n_key*key_v + (n_frame-n_key)*normal_v), 2),
                  "multi_label_frames": sum(t["multi_label_frames"]
                                            for t in per_task.values())},
        "per_task": {str(ti): v for ti, v in per_task.items()},
        "per_episode": per_episode,
    }
    stats_path = out_dir / "weight_apply_stats.json"
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n",
                          encoding="utf-8")

    print(f"{'task':>4} {'slug':<18} {'episodes':>8} {'frames':>8} "
          f"{'key':>8} {'key pct':>8} {'loss=1':>8} {'multi':>7}")
    for ti in selected:
        s = per_task.get(ti)
        if s:
            print(f"{ti:>4} {s['slug']:<18} {s['episodes']:>8} {s['frames']:>8} "
                  f"{s['is_key_frames']:>8} {s['is_key_frame_pct']:>7.2f}% "
                  f"{s['loss_eq1']:>8} {s['multi_label_frames']:>7}")
            print(f"       event share: {s['event_frame_share_pct']}")
    print(f"总计: {n_ep} episodes / {n_frame:,} frames, "
          f"is_key_frame={n_key:,} ({100.0*n_key/n_frame:.2f}%), "
          f"loss=1 {n_eq1:,} ({total_eq1_pct:.2f}%)")
    print(f"采样: key={key_v} normal={normal_v} -> 采样后关键帧期望占比 "
          f"{stats['total']['expected_key_share_after_sampling']}%, "
          f"ESS={stats['total']['ess_with_sampling']}")
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
    df = pd.read_csv(a.csv, usecols=["episode_index", "task_index", "frame_index",
                                     "is_key_frame", "frame_weight_loss"])
    df["frame_weight_loss"] = df["frame_weight_loss"].astype(float)
    selected = (sorted(df["task_index"].unique()) if a.tasks == "all"
                else [int(x) for x in a.tasks.split(",")])
    stats_by_task: dict[int, dict] = {}
    records: list[dict] = []
    n_frame = n_eq1 = n_key = 0
    for ti in selected:
        sub = df[df["task_index"] == ti]
        eq1 = int(np.isclose(sub["frame_weight_loss"], WEIGHT_BASE).sum())
        nkey = int(sub["is_key_frame"].sum())
        n_frame += len(sub); n_eq1 += eq1; n_key += nkey
        for ep, g in sub.groupby("episode_index"):
            e1 = int(np.isclose(g["frame_weight_loss"], WEIGHT_BASE).sum())
            records.append({"task_index": ti, "episode_index": int(ep),
                            "eq1_pct": round(100.0*e1/len(g), 2),
                            "key_pct": round(100.0*g["is_key_frame"].mean(), 2)})
        stats_by_task[ti] = {
            "slug": ke.TASK_SLUGS.get(ti, f"task_{ti:03d}"),
            "frames": int(len(sub)), "eq1": eq1,
            "eq1_pct": round(100.0*eq1/len(sub), 2),
            "is_key_frames": nkey,
            "is_key_pct": round(100.0*nkey/len(sub), 2),
            "_all_loss": sub["frame_weight_loss"].to_numpy(),
        }
        print(f"  task {ti} ({stats_by_task[ti]['slug']}): frames={len(sub):,} "
              f"key={nkey:,} ({stats_by_task[ti]['is_key_pct']:.2f}%) "
              f"loss=1 {eq1:,} ({stats_by_task[ti]['eq1_pct']:.2f}%)")
    print(f"total: {n_frame:,} frames, key={n_key:,} ({100.0*n_key/n_frame:.2f}%), "
          f"loss=1 {n_eq1:,} ({100.0*n_eq1/n_frame:.2f}%)")
    out_png = out_dir / "weight_distribution.png"
    plot_weight_distribution(stats_by_task, records,
                             df["frame_weight_loss"].max(), out_png)
    print(f"distribution figure -> {out_png}")


# ---------------------------------------------------------------------------
# align: 权重与爪夹曲线对齐可视化 (人工确认每帧权重)
# ---------------------------------------------------------------------------

def plot_task_align(task_name: str, picks: list[dict], max_w: float,
                    out_path: Path) -> None:
    """每任务一张图, 每个 episode 一行: 左轴=左右爪夹开度, 右轴=逐帧权重曲线,
    事件范围用彩色底纹标出, 最大权重区间标出权重数值。三行 sharex=False。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from tools.keyframe_detect import C_LEFT, C_RIGHT, GRID, INK, MUT, SURF

    plt.rcParams.update({
        "axes.edgecolor": MUT, "axes.labelcolor": INK, "axes.titlecolor": INK,
        "text.color": INK, "xtick.color": INK, "ytick.color": INK,
        "figure.facecolor": SURF, "axes.facecolor": SURF, "grid.color": GRID,
        "font.family": "sans-serif", "figure.dpi": 110,
        "font.sans-serif": ["PingFang SC", "Hiragino Sans GB", "Heiti SC",
                            "Arial Unicode MS", "DejaVu Sans"],
        "axes.unicode_minus": False,
    })

    n = len(picks)
    fig, axes = plt.subplots(n, 1, figsize=(14, 3.4 * n), squeeze=False)
    fig.suptitle(f"{task_name}  —  frame weight aligned with gripper curves",
                 fontsize=12, y=0.995)
    keys_seen: set[str] = set()
    for r, pk in enumerate(picks):
        ep, frame, state, res = pk["ep"], pk["frame"], pk["state"], pk["res"]
        weight = res["weight_loss"]
        sampling = res["weight_sampling"]
        ax = axes[r][0]
        ax2 = ax.twinx()                                   # 右轴: 权重
        # 事件范围底纹 (左轴坐标, 只作背景)
        for inst in res["instances"]:
            keys_seen.add(inst["key"])
            ax.axvspan(frame[inst["a"]], frame[inst["b"]],
                       color=ke.event_color(inst["key"]), alpha=0.14, lw=0)
            ax.axvline(frame[inst["t0"]], color=ke.event_color(inst["key"]),
                       ls="--", lw=0.9, alpha=0.55, zorder=1)
        # 权重曲线 (右轴)
        ax2.fill_between(frame, WEIGHT_BASE, weight, color="#f2a900",
                         alpha=0.22, lw=0)
        ax2.plot(frame, weight, color="#c47b00", lw=1.1,
                 label="frame_weight_loss")
        ax2.step(frame, sampling, where="mid", color="#1f7a8c", lw=1.0, ls="--",
                 label="frame_weight_sampling")
        ax2.axhline(WEIGHT_BASE, color="0.6", lw=0.7, ls=":")
        key_v = float(sampling.max()) if len(sampling) else ke.SAMPLING_KEY_VALUE
        ax2.set_ylim(0.9, max(max_w, key_v) + 0.35)
        ax2.set_yticks(sorted({1.0, round(max_w, 2), round(key_v, 2)}))
        ax2.set_ylabel("loss / sampling", color="#c47b00")
        ax2.tick_params(axis="y", colors="#c47b00")
        # 每个连续 weight>1 区段标注一次峰值 (避免相邻事件重复标注/与图例打架)
        hi = weight > WEIGHT_BASE + 1e-9
        i = 0
        while i < len(hi):
            if hi[i]:
                j = i
                while j + 1 < len(hi) and hi[j + 1]:
                    j += 1
                k = int(np.argmax(weight[i:j + 1])) + i
                ax2.text(frame[k], float(weight[k]) + 0.05, f"{float(weight[k]):g}",
                         ha="center", va="bottom", fontsize=7.5, color="#8a5700",
                         zorder=6)
                i = j + 1
            else:
                i += 1
        # 爪夹曲线 (左轴)
        ax.plot(frame, state[:, ke.GRIP_L], color=C_LEFT, lw=1.2, label="left gripper")
        ax.plot(frame, state[:, ke.GRIP_R], color=C_RIGHT, lw=1.2, label="right gripper")
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel("gripper (0闭~1开)")
        ax.set_xlim(frame[0], frame[-1])
        ax.grid(alpha=0.22, zorder=0)
        eq1 = int(np.isclose(weight, WEIGHT_BASE).sum())
        nmulti = int(sum(ke.MULTI_SEP in x for x in res["labels"]))
        ax.set_title(f"ep{ep:03d}  T={len(frame)}  {res['meta']}  "
                     f"key={100.0*res['is_key_frame'].mean():.1f}%  "
                     f"loss=1 {100.0*eq1/len(frame):.1f}%  multi={nmulti}",
                     loc="left", fontsize=9)
    axes[-1][0].set_xlabel("frame index")
    cn = {i["key"]: i["cn"] for pk in picks for i in pk["res"]["instances"]}
    # 图例统一置于图外底部 (避免遮挡右上角峰值标注)
    line_handles = [
        plt.Line2D([0], [0], color=C_LEFT, lw=1.5, label="left gripper"),
        plt.Line2D([0], [0], color=C_RIGHT, lw=1.5, label="right gripper"),
        plt.Line2D([0], [0], color="#c47b00", lw=1.5, label="frame_weight_loss"),
        plt.Line2D([0], [0], color="#1f7a8c", lw=1.5, ls="--",
                   label="frame_weight_sampling"),
    ]
    span_handles = [plt.Rectangle((0, 0), 1, 1, color=ke.event_color(k), alpha=0.5,
                                  label=f"{k}({cn.get(k, k)})")
                    for k in sorted(keys_seen)]
    handles = line_handles + span_handles
    fig.legend(handles=handles, fontsize=8, frameon=False, ncol=min(6, len(handles)),
               loc="lower center", bbox_to_anchor=(0.5, -0.004))
    fig.tight_layout(rect=(0, 0.035, 1, 0.965))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def cmd_align(args) -> None:
    parser = argparse.ArgumentParser(add_help=False)
    common_parser_args(parser)
    parser.add_argument("--per-task", type=int, default=3,
                        help="每任务抽样 episode 数 (默认 3)")
    parser.add_argument("--episodes", default=None,
                        help="指定每任务 ep(逗号分隔, 覆盖 --per-task)")
    a = parser.parse_args(args)
    (small, eps_by, task_names, config, dk, out_dir, selected) = load_common(a)
    want = [int(x) for x in a.episodes.split(",")] if a.episodes else None
    for ti in selected:
        if ti not in eps_by:
            print(f"  ! task_index={ti} 不存在, 跳过"); continue
        slug = ke.task_slug(config, ti)
        name = task_names.get(ti, slug)
        eps = eps_by[ti]
        pick = (np.asarray(want, dtype=int) if want is not None
                else ke.pick_episodes(eps, a.per_task))
        pick = [int(x) for x in pick[np.isin(pick, eps)]]
        mw = max_event_weight(config, ti)
        picks = []
        for ep in pick:
            state, frame = episode_df(small, ep)
            res = ke.episode_weight_and_labels(ti, state, frame, dk, config)
            picks.append({"ep": ep, "frame": frame, "state": state, "res": res})
        out_png = out_dir / f"{slug}_align.png"
        plot_task_align(name, picks, mw, out_png)
        print(f"  task {ti} ({slug}): episodes {pick} -> {out_png}")
    print(f"\nalign figures -> {out_dir}")


# ---------------------------------------------------------------------------
# merge: 把 frame_weight/keyframe_label 并入数据集 CSV
# ---------------------------------------------------------------------------

def cmd_merge(args) -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--dataset", required=True,
                        help="要并入的训练集 CSV (如 data/.../sim_lerobot_v30_ee.csv)")
    parser.add_argument("--weights", default=str(DEFAULT_TARGET),
                        help="权重 CSV (episode_index,frame_index,keyframe_label,"
                             "is_key_frame,frame_weight_loss,frame_weight_sampling)")
    parser.add_argument("--out", default=None,
                        help="输出路径 (默认原地更新 --dataset)")
    a = parser.parse_args(args)
    ds_path = Path(a.dataset)
    fw = pd.read_csv(a.weights)
    NEW_COLS = ["keyframe_label", "is_key_frame", "frame_weight_loss",
                "frame_weight_sampling"]
    need = {"episode_index", "frame_index", *NEW_COLS}
    if not need.issubset(fw.columns):
        raise SystemExit(f"{a.weights} 缺列: {sorted(need - set(fw.columns))}")
    ds = pd.read_csv(ds_path)
    # 幂等: 若已存在旧列则先剔除 (含已废止的 frame_weight)
    ds = ds.drop(columns=[c for c in (*NEW_COLS, "frame_weight")
                          if c in ds.columns])
    merged = ds.merge(fw[["episode_index", "frame_index", *NEW_COLS]],
                      on=["episode_index", "frame_index"], how="left",
                      validate="one_to_one")
    miss = int(merged["frame_weight_loss"].isna().sum())
    if miss:
        raise RuntimeError(f"{miss} 行未对齐到权重 (ep,frame)")
    out = Path(a.out) if a.out else ds_path
    merged.to_csv(out, index=False)
    print(f"并入 {len(merged):,} 行 -> {out}  (新增列: {', '.join(NEW_COLS)})")
    print(f"  label 分布: "
          f"{merged['keyframe_label'].value_counts().head(8).to_dict()}")
    print(f"  is_key_frame={int(merged['is_key_frame'].sum()):,} "
          f"({100.0*merged['is_key_frame'].mean():.2f}%), "
          f"loss 范围 [{merged['frame_weight_loss'].min()}, "
          f"{merged['frame_weight_loss'].max()}], "
          f"sampling 取值 {sorted(merged['frame_weight_sampling'].unique())}")


# ---------------------------------------------------------------------------
# 并入 LeRobot v3 数据集 (parquet + meta)
# ---------------------------------------------------------------------------

# 写进 LeRobot 数据集的三个训练字段 (不含 keyframe_label —— 字符串列不入 features)
DS_COLS = ["is_key_frame", "frame_weight_loss", "frame_weight_sampling"]
DS_DTYPES = {"is_key_frame": "int64",           # 二值 0/1
             "frame_weight_loss": "float32",
             "frame_weight_sampling": "float32"}
DS_QUANTILES = [("q01", 0.01), ("q10", 0.10), ("q50", 0.50),
                ("q90", 0.90), ("q99", 0.99)]


def _stats_entry(v: np.ndarray) -> dict:
    """LeRobot stats.json 单字段格式 (见 meta/stats.json 现有条目)。"""
    e = {"min": [float(v.min())], "max": [float(v.max())],
         "mean": [float(v.mean())], "std": [float(v.std())],
         "count": [int(v.size)]}
    for name, q in DS_QUANTILES:
        e[name] = [float(np.quantile(v, q))]
    return e


def cmd_merge_dataset(args) -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--dataset-dir", default=str(DEFAULT_TARGET.parents[0]),
                        help="LeRobot v3 数据集根目录 (含 data/ 与 meta/)")
    parser.add_argument("--weights", default=str(DEFAULT_TARGET),
                        help="权重 CSV (episode_index,frame_index,...)")
    parser.add_argument("--dry-run", action="store_true",
                        help="只校验对齐与统计, 不写任何文件")
    a = parser.parse_args(args)
    ddir = Path(a.dataset_dir).resolve()
    info_p, stats_p = ddir / "meta" / "info.json", ddir / "meta" / "stats.json"
    if not info_p.exists():
        raise SystemExit(f"{info_p} 不存在, 不是 LeRobot v3 数据集目录")
    info = json.loads(info_p.read_text())

    # 定位数据 parquet (按 info.json 的 data_path 模板, 目前仅支持单文件数据集)
    pqs = sorted(ddir.glob("data/chunk-*/file-*.parquet"))
    if len(pqs) != 1:
        raise SystemExit(f"预期 1 个数据 parquet, 实际 {len(pqs)} 个: {pqs}")

    w = pd.read_csv(a.weights)
    need = {"episode_index", "frame_index", *DS_COLS}
    if not need.issubset(w.columns):
        raise SystemExit(f"{a.weights} 缺列: {sorted(need - set(w.columns))}")
    ww = w[["episode_index", "frame_index", *DS_COLS]]
    if ww.duplicated(["episode_index", "frame_index"]).any():
        raise SystemExit(f"{a.weights} 的 (episode_index,frame_index) 不唯一")

    df = pd.read_parquet(pqs[0])
    keys = pd.MultiIndex.from_arrays([df["episode_index"], df["frame_index"]])
    ww = ww.set_index(["episode_index", "frame_index"])
    vals = {}
    for c in DS_COLS:
        s = ww[c].reindex(keys)
        miss = int(s.isna().sum())
        if miss:
            raise SystemExit(f"{miss} 行 parquet (ep,frame) 在权重表中找不到, 已中止")
        vals[c] = s.to_numpy()

    # 幂等: 已存在则直接覆盖; 新增列一律追加在末尾
    for c in DS_COLS:
        df[c] = (vals[c].astype("int64") if DS_DTYPES[c] == "int64"
                 else vals[c].astype("float32"))
        info["features"][c] = {"dtype": DS_DTYPES[c], "shape": [1], "names": None}

    print(f"数据集: {pqs[0].relative_to(ddir)}  ({len(df):,} 行, "
          f"col_order={list(df.columns)})")
    for c in DS_COLS:
        v = df[c].to_numpy()
        print(f"  {c:<22} {DS_DTYPES[c]:<8} "
              f"min={v.min():g} max={v.max():g} mean={v.mean():.4f} "
              f"nonzero={int((v != 0).sum()):,}")
    if list(df["is_key_frame"]) != [int(x != "none") for x in
                                    w.set_index(["episode_index",
                                                 "frame_index"])["keyframe_label"]
                                    .reindex(keys).to_numpy()]:
        raise SystemExit("is_key_frame 与 keyframe_label!=none 不一致, 已中止")

    if a.dry_run:
        print("[dry-run] 未写文件")
        return

    df.to_parquet(pqs[0], index=False)          # 行序/原列 dtype 已验无损往返
    info["data_files_size_in_mb"] = round(pqs[0].stat().st_size / 1e6, 3)
    info_p.write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n")
    if stats_p.exists():
        stats = json.loads(stats_p.read_text())
        for c in DS_COLS:
            stats[c] = _stats_entry(df[c].to_numpy())
        stats_p.write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n")
    print(f"已写入 {pqs[0]} + meta/info.json"
          f"{' + meta/stats.json' if stats_p.exists() else ''}")
    print(f"  features 新增: {', '.join(DS_COLS)}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("preview", help="每任务抽样 n ep 画 权重+事件+爪夹 图")
    sub.add_parser("apply", help="全量赋权重+标签, 更新 frame_weight.csv + 副本 + stats")
    sub.add_parser("dist", help="读 apply 副本, 画权重分布图")
    sub.add_parser("align", help="每任务抽样 n ep: 权重曲线与爪夹曲线对齐标注(人工确认)")
    sub.add_parser("merge", help="把 frame_weight/keyframe_label 并入训练集 CSV")
    sub.add_parser("merge-dataset",
                   help="把 loss/sampling/is_key_frame 写进 LeRobot v3 parquet + meta")
    args, rest = parser.parse_known_args()
    if args.command == "preview":
        cmd_preview(rest)
    elif args.command == "apply":
        cmd_apply(rest)
    elif args.command == "dist":
        cmd_dist(rest)
    elif args.command == "align":
        cmd_align(rest)
    elif args.command == "merge":
        cmd_merge(rest)
    elif args.command == "merge-dataset":
        cmd_merge_dataset(rest)


if __name__ == "__main__":
    main()
