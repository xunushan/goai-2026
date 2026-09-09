#!/usr/bin/env python3
"""sim_lerobot_v30_ee 关键帧非对称梯形权重: 检测 / 预览 / 全量赋值 / 分布图。

参考 docs/dual_arm_tasks_failure_and_keyframe_plan.md:
  - 关键事件 t0 = 每个抓取周期(左右臂)的 hold_start 与 hold_end;
  - 每个 t0 生成非对称梯形权重窗, 逐帧权重 = 所有事件窗口取 max (不累加);
  - 普通帧权重 1, 最大权重 wmax (默认 2); 窗口参数按任务可配 (见下方 schema)。

窗口参数 (计划 §1.1, a=t0-L, b=t0+Pl, c=t0+Pr, d=t0+R):
    w(t): t<a 为 1; [a,b) 线性 1->wmax; [b,c] 恒 wmax; (c,d] 线性 wmax->1; t>d 为 1

三个子命令:
  preview   每任务抽样 n 个 episode, 画 权重曲线/左右爪夹 3 行图 (人工确认窗口落点)
  apply     对全部(或指定)episode 逐帧赋权重, 追加 frame_weight 列写副本 CSV,
            打印 weight=1 / weight>1 占比, 写 stats JSON
  dist      读 apply 产物 *_weight.csv, 画权重分布图 (逐帧权重直方 + 逐 episode 的
            weight=1 占比分布)

按任务配置权重参数 (参数可按任务配置):
    JSON: {"wmax": 2.0, "tasks": {"<task_index>": {"L":..,"R":..,"Pl":..,"Pr":..}}}
    默认配置文件 tools/frame_weight_config.json (sim 三任务, 见 doc §5), 用
    --config <path> 覆盖。task_index 语义随数据集变化 (sim 0/1/2 与 real 不同)。

state 16 维: left_ee_pose(7)+left_gripper(idx7) + right_ee_pose(7)+right_gripper(idx15);
sim 每 episode frame_index 从 0 连续, 检测返回位置 == 帧号。

用法:
    python tools/frame_weight_trapezoid.py preview                # 每任务 3 个 episode
    python tools/frame_weight_trapezoid.py preview --per-task 3 --episodes 0,50,99
    python tools/frame_weight_trapezoid.py apply --out outputs/xxx
    python tools/frame_weight_trapezoid.py dist --csv <apply产物> --out outputs/xxx
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

from tools.keyframe_detect import (  # noqa: E402
    GRIP_L,
    GRIP_R,
    INCOMPLETE,
    detect_gripper_keyframes,
)

DEFAULT_CSV = ROOT / "data" / "sim_lerobot_v30_ee" / "sim_lerobot_v30_ee.csv"
DEFAULT_TASKS = ROOT / "data" / "sim_lerobot_v30_ee" / "meta" / "tasks.parquet"
DEFAULT_SPLIT = ROOT / "data" / "sim_lerobot_v30_ee" / "train_val_split.json"
DEFAULT_CONFIG = ROOT / "tools" / "frame_weight_config.json"
DEFAULT_OUT = ROOT / "outputs" / "frame_weight_trapezoid_sim_v30ee"

WEIGHT_BASE = 1.0

# sim 数据集固定 3 任务 task_index -> slug (文件名用)
TASK_SLUGS: dict[int, str] = {
    0: "fill_pen_holder",
    1: "plug_in_charger",
    2: "stack_bowls",
}


# ---------------------------------------------------------------------------
# 按任务配置加载
# ---------------------------------------------------------------------------

def _default_task_params() -> dict:
    """回退默认: 计划 §5 sim 三任务参数。"""
    return {
        "0": {"L": 20, "R": 10, "Pl": -8, "Pr": -2},   # fill_pen_holder
        "1": {"L": 15, "R": 10, "Pl": -5, "Pr": 5},    # plug_in_charger(短ep, 收紧窗口)
        "2": {"L": 20, "R": 10, "Pl": -8, "Pr": -2},   # stack_bowls
    }


def load_config(config_path: Path | None) -> dict:
    """读取权重配置 -> {"wmax": float, "tasks": {str(task): params}}。"""
    if config_path is not None:
        return json.loads(Path(config_path).read_text(encoding="utf-8"))
    if DEFAULT_CONFIG.is_file():
        return json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    return {"wmax": 2.0, "tasks": _default_task_params()}


# ---------------------------------------------------------------------------
# 权重函数
# ---------------------------------------------------------------------------

def trapezoid_weight(
    t0: int,
    frame: np.ndarray,
    params: dict[str, int],
    wmax: float,
) -> np.ndarray:
    """单个事件 t0 的非对称梯形权重窗 (逐帧, 与 frame 对齐)。"""
    L, R, Pl, Pr = params["L"], params["R"], params["Pl"], params["Pr"]
    a, b, c, d = t0 - L, t0 + Pl, t0 + Pr, t0 + R
    t = np.asarray(frame, dtype=float)
    w = np.full_like(t, WEIGHT_BASE, dtype=float)
    rise = (t > a) & (t < b)
    w[rise] = WEIGHT_BASE + (wmax - WEIGHT_BASE) * (t[rise] - a) / (b - a)
    w[(t >= b) & (t <= c)] = wmax
    fall = (t > c) & (t < d)
    w[fall] = wmax - (wmax - WEIGHT_BASE) * (t[fall] - c) / (d - c)
    return w


# ---------------------------------------------------------------------------
# 关键事件 (t0) 检测 + episode 权重
# ---------------------------------------------------------------------------

def detect_episode_events(
    state: np.ndarray,
    frame: np.ndarray,
    detect_kwargs: dict,
) -> dict:
    """单 episode 关键事件: events(hold_start/hold_end, 含 side/type/frame) +
    per_side 四类关键点位置 (供完整周期可视化)。"""
    per_side: dict[str, dict[str, np.ndarray]] = {}
    events: list[dict] = []
    for side, idx in (("left", GRIP_L), ("right", GRIP_R)):
        kf = detect_gripper_keyframes(state[:, idx], frame, **detect_kwargs)
        per_side[side] = {k: np.asarray(v, dtype=int) for k, v in kf.items()}
        for key in ("hold_start", "hold_end"):
            for p in per_side[side][key]:
                if p < 0:
                    continue
                events.append({"side": side, "type": key, "frame": int(p)})
    events.sort(key=lambda e: e["frame"])
    return {"events": events, "per_side": per_side}


def episode_weight(
    state: np.ndarray,
    frame: np.ndarray,
    params: dict[str, int],
    wmax: float,
    detect_kwargs: dict,
) -> tuple[np.ndarray, list[dict]]:
    """单 episode 逐帧权重 (取各事件窗口 max); 返回 (w, events)。"""
    res = detect_episode_events(state, frame, detect_kwargs)
    w = np.full(len(frame), WEIGHT_BASE, dtype=float)
    for ev in res["events"]:
        w = np.maximum(w, trapezoid_weight(ev["frame"], frame, params, wmax))
    return w, res["events"]


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------

def load_tasks(tasks_parquet: Path) -> dict[int, str]:
    import pyarrow.parquet as pq

    t = pq.read_table(str(tasks_parquet)).to_pandas()
    return {int(v): str(k) for k, v in t["task_index"].items()}


def load_state_cols(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(
        csv_path,
        usecols=["episode_index", "task_index", "frame_index", "observation.state"],
    )


def parse_state_series(s: pd.Series) -> np.ndarray:
    return np.asarray(
        [np.fromstring(x.strip("[]"), sep=",", dtype=float) for x in s],
        dtype=float,
    )


def load_train_episodes(split_json: Path) -> set[int] | None:
    if not split_json.is_file():
        return None
    d = json.loads(split_json.read_text(encoding="utf-8"))
    train: set[int] = set()
    for cfg in d["tasks"].values():
        train.update(int(e) for e in cfg["train_episode_idx"])
    return train


# ---------------------------------------------------------------------------
# 可视化
# ---------------------------------------------------------------------------

def plot_episode_weight(
    task_name: str,
    episode_index: int,
    frame: np.ndarray,
    state: np.ndarray,
    weight: np.ndarray,
    events: list[dict],
    per_side: dict[str, dict[str, np.ndarray]],
    out_path: Path,
) -> None:
    """3 行子图: [权重曲线] [左爪夹] [右爪夹], 关键帧竖线标出。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from tools.keyframe_detect import C_LEFT, C_RIGHT, INK, KEYFRAME_STYLES, MUT

    plt.rcParams.update({
        "axes.edgecolor": MUT, "axes.labelcolor": INK, "axes.titlecolor": INK,
        "text.color": INK, "xtick.color": INK, "ytick.color": INK,
        "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
        "grid.color": "#e1e0d9", "font.family": "sans-serif", "figure.dpi": 110,
    })

    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    fig.suptitle(f"{task_name}\nepisode {episode_index}  ({len(frame)} frames, "
                 f"{len(events)} events)",
                 fontsize=11, y=0.99)

    ax = axes[0]
    ax.fill_between(frame, WEIGHT_BASE, weight, color="#f2a900", alpha=0.35, lw=0)
    ax.plot(frame, weight, color="#c47b00", lw=1.0)
    ax.axhline(WEIGHT_BASE, color="0.5", lw=0.8, ls=":")
    ax.set_ylabel("weight")
    ax.set_ylim(0.9, 2.12)
    ax.set_yticks([1.0, 1.5, 2.0])
    ax.set_title("frame weight (max over events, 1 normal / 2 key)", loc="left",
                 fontsize=9)
    for ev in events:
        col = "#d62728" if ev["type"] == "hold_start" else "#9467bd"
        ax.axvline(ev["frame"], color=col, lw=0.8, alpha=0.5, zorder=0)
        marker = "^" if ev["side"] == "left" else "v"
        ax.scatter([ev["frame"]], [2.05], marker=marker, s=14, color=col, zorder=3)
    ax.grid(alpha=0.25)

    for axg, name, idx, color in [
        (axes[1], "Left gripper", GRIP_L, C_LEFT),
        (axes[2], "Right gripper", GRIP_R, C_RIGHT),
    ]:
        axg.plot(frame, state[:, idx], color=color, lw=1.1)
        axg.set_ylabel("gripper")
        axg.set_ylim(-0.05, 1.05)
        side = "left" if idx == GRIP_L else "right"
        for key, lc, ls, _ in KEYFRAME_STYLES:
            for x in per_side[side][key]:
                if x == INCOMPLETE:
                    continue
                axg.axvline(x, color=lc, ls=ls, alpha=0.8, lw=1.0)
        axg.set_title(name, loc="left", fontsize=9)
        axg.grid(alpha=0.25)
    axes[2].set_xlabel("frame index")

    handles = [
        plt.Line2D([0], [0], color=C_LEFT, lw=1.5, label="left gripper"),
        plt.Line2D([0], [0], color=C_RIGHT, lw=1.5, label="right gripper"),
    ] + [
        plt.Line2D([0], [0], color=c, ls=s, lw=1.2, label=lbl)
        for _, c, s, lbl in KEYFRAME_STYLES
    ] + [
        plt.Line2D([0], [0], color="#d62728", lw=1.4, label="event t0 (hold_start)"),
        plt.Line2D([0], [0], color="#9467bd", lw=1.4, label="event t0 (hold_end)"),
    ]
    fig.legend(handles=handles, fontsize=8, ncol=5, frameon=False,
               loc="upper center", bbox_to_anchor=(0.5, 0.975))
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_weight_distribution(
    stats_by_task: dict[int, dict],
    episode_records: list[dict],
    wmax: float,
    out_path: Path,
) -> None:
    """权重分布图: 上=逐帧权重直方(log y), 下=逐 episode 的 weight=1 占比直方。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {0: "#2a78d6", 1: "#eb6834", 2: "#4c9f38"}
    fig, axes = plt.subplots(2, 1, figsize=(11, 8))
    fig.suptitle("frame weight distribution (sim_lerobot_v30_ee, "
                 "trapezoid max-over-events)",
                 fontsize=12, y=0.99)

    # 行0: 逐帧权重直方 (含 weight=1 尖峰, log y)
    ax = axes[0]
    binw = 0.05
    bins = np.arange(1.0 - binw / 2, wmax + binw, binw)
    for ti, s in sorted(stats_by_task.items()):
        ax.hist(s["_all_weights"], bins=bins, alpha=0.55, log=True,
                color=colors.get(ti, "0.5"),
                label=f"{s['slug']}: n={s['frames']:,}")
    ax.set_xlim(1.0, wmax + 0.05)
    ax.set_xlabel("frame weight")
    ax.set_ylabel("frames (log)")
    ax.set_title("per-frame weight histogram (log y; spike at 1 = unboosted)",
                 loc="left", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    # 行1: 逐 episode 的 weight==1 占比分布
    ax = axes[1]
    ax2x = None
    for ti in sorted({r["task_index"] for r in episode_records}):
        fr = [r["eq1_pct"] for r in episode_records if r["task_index"] == ti]
        ax.hist(fr, bins=np.arange(0, 101, 5), alpha=0.55,
                color=colors.get(ti, "0.5"),
                label=stats_by_task[ti]["slug"])
    ax.set_xlabel("episode weight=1 fraction (%)")
    ax.set_ylabel("episodes")
    ax.set_title("per-episode share of weight==1 frames (unboosted coverage)",
                 loc="left", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 各子命令实现
# ---------------------------------------------------------------------------

def pick_episodes(eps: np.ndarray, n: int) -> np.ndarray:
    """在 ep 排序列表里均匀取 n 个 (首/中/尾), 确定性覆盖不同长度轨迹。"""
    if n >= len(eps):
        return eps
    return np.asarray(
        [eps[int(round(i * (len(eps) - 1) / (n - 1)))] for i in range(n)],
        dtype=int,
    )


def common_parser_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--csv", default=str(DEFAULT_CSV), help="sim CSV 路径")
    p.add_argument("--tasks", default="all",
                   help="task_index, 逗号分隔 (如 '0,2') 或 'all'")
    p.add_argument("--config", default=None, help="权重配置 JSON (默认 tools/"
                   "frame_weight_config.json)")
    p.add_argument("--min-prominence", type=float, default=0.2)
    p.add_argument("--hold-min-len", type=int, default=3)
    p.add_argument("--open-level", type=float, default=0.9)
    p.add_argument("--no-incomplete", action="store_true")
    p.add_argument("--out", default=str(DEFAULT_OUT), help="输出目录")


def load_common(args) -> tuple:
    """解析公共参数 -> (small, episodes_by_task, task_names, config,
    detect_kwargs, train_eps, out_dir)。"""
    config = load_config(Path(args.config) if args.config else None)
    detect_kwargs = {
        "min_prominence": args.min_prominence,
        "hold_min_len": args.hold_min_len,
        "open_level": args.open_level,
        "allow_incomplete": not args.no_incomplete,
    }
    task_names = load_tasks(DEFAULT_TASKS) if DEFAULT_TASKS.is_file() else {}
    train_eps = load_train_episodes(DEFAULT_SPLIT)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"loading {args.csv} ...")
    small = load_state_cols(Path(args.csv))
    episodes_by_task = (
        small.groupby("task_index")["episode_index"]
        .apply(lambda s: np.sort(s.unique()))
        .to_dict()
    )
    selected_tasks = (
        list(episodes_by_task)
        if args.tasks == "all"
        else [int(x) for x in args.tasks.split(",")]
    )
    return (small, episodes_by_task, task_names, config, detect_kwargs,
            train_eps, out_dir, selected_tasks)


def require_task_params(ti: int, config: dict) -> dict[str, int]:
    key = str(ti)
    if key not in config["tasks"]:
        raise SystemExit(f"task_index={ti} 未在权重配置中定义, 请加进 "
                         f"{DEFAULT_CONFIG} 或 --config 文件")
    return config["tasks"][key]


def cmd_preview(args) -> None:
    parser = argparse.ArgumentParser(add_help=False)
    common_parser_args(parser)
    parser.add_argument("--per-task", type=int, default=3)
    parser.add_argument("--episodes", default=None,
                        help="指定每任务 ep(逗号分隔, 覆盖 --per-task)")
    a = parser.parse_args(args)
    (small, eps_by, task_names, config, dk, _train, out_dir,
     selected) = load_common(a)
    wmax = config["wmax"]
    want_eps = ([int(x) for x in a.episodes.split(",")]
                if a.episodes else None)
    summary = {}
    for ti in selected:
        if ti not in eps_by:
            print(f"  ! task_index={ti} 不存在, 跳过")
            continue
        params = require_task_params(ti, config)
        eps = eps_by[ti]
        pick = (np.asarray(want_eps, dtype=int) if want_eps is not None
                else pick_episodes(eps, a.per_task))
        pick = pick[np.isin(pick, eps)]
        slug = TASK_SLUGS.get(ti, f"task_{ti:03d}")
        name = task_names.get(ti, slug)
        p = params
        print(f"  task {ti} ({slug}) L{p['L']} R{p['R']} Pl{p['Pl']} "
              f"Pr{p['Pr']}: episodes {pick.tolist()}")
        for ep in [int(x) for x in pick]:
            sub = small[small["episode_index"] == ep].sort_values("frame_index")
            state = parse_state_series(sub["observation.state"])
            frame = sub["frame_index"].to_numpy()
            res = detect_episode_events(state, frame, dk)
            w, events = episode_weight(state, frame, params, wmax, dk)
            out_png = out_dir / f"{slug}_ep{ep:03d}_weight.png"
            plot_episode_weight(name, ep, frame, state, w, res["events"],
                                res["per_side"], out_png)
            eq1 = int(np.isclose(w, WEIGHT_BASE).sum())
            print(f"    ep {ep:3d}: T={len(frame):4d} events={len(events):2d} "
                  f"weight=1={eq1:4d} ({100.0*eq1/len(frame):4.1f}%) -> "
                  f"{out_png.name}")
            summary[f"task{ti}_ep{ep}"] = {
                "task_index": ti, "episode_index": ep, "n_frames": int(len(frame)),
                "events": events,
                "weight_eq1": eq1,
                "weight_eq1_pct": round(100.0 * eq1 / len(frame), 3),
            }
    (out_dir / "preview_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\npreview done -> {out_dir}")


def cmd_apply(args) -> None:
    parser = argparse.ArgumentParser(add_help=False)
    common_parser_args(parser)
    a = parser.parse_args(args)
    (small, eps_by, task_names, config, dk, train_eps, out_dir,
     selected) = load_common(a)
    wmax = config["wmax"]

    key_rows: list[pd.DataFrame] = []
    per_episode: dict[str, dict] = {}
    per_task: dict[int, dict] = {}
    n_ep = n_frame = 0
    n_eq1_total = n_gt1_total = 0
    train_frame = train_eq1 = n_train_ep = 0
    for ti in selected:
        if ti not in eps_by:
            print(f"  ! task_index={ti} 不存在, 跳过")
            continue
        params = require_task_params(ti, config)
        eps = eps_by[ti]
        slug = TASK_SLUGS.get(ti, f"task_{ti:03d}")
        name = task_names.get(ti, slug)
        p = params
        print(f"  task {ti} ({slug}) L{p['L']} R{p['R']} Pl{p['Pl']} "
              f"Pr{p['Pr']}: {len(eps)} episodes")
        t_frame = t_eq1 = t_gt1 = 0
        ep_list = [int(e) for e in eps]
        for ep in ep_list:
            sub = small[small["episode_index"] == ep].sort_values("frame_index")
            state = parse_state_series(sub["observation.state"])
            frame = sub["frame_index"].to_numpy()
            w, events = episode_weight(state, frame, params, wmax, dk)
            eq1 = int(np.isclose(w, WEIGHT_BASE).sum())
            gt1 = len(frame) - eq1
            key_rows.append(pd.DataFrame({
                "episode_index": ep, "frame_index": frame,
                "frame_weight": w,
            }))
            n_frame += len(frame); n_eq1_total += eq1; n_gt1_total += gt1
            t_frame += len(frame); t_eq1 += eq1; t_gt1 += gt1
            if train_eps is not None and ep in train_eps:
                train_frame += len(frame); train_eq1 += eq1; n_train_ep += 1
            per_episode[f"task{ti}_ep{ep}"] = {
                "task_index": ti, "episode_index": ep, "n_frames": int(len(frame)),
                "events": events, "weight_eq1": eq1,
                "weight_gt1": int(gt1),
                "eq1_pct": round(100.0 * eq1 / len(frame), 3),
            }
        n_ep += len(eps)
        per_task[ti] = {
            "slug": slug, "instruction": name, "episodes": len(eps),
            "frames": t_frame, "weight_eq1": t_eq1, "weight_gt1": t_gt1,
            "weight_eq1_pct": round(100.0 * t_eq1 / t_frame, 3),
        }

    # 写副本 CSV (追加 frame_weight 列, 保留源列序)
    full = pd.read_csv(str(a.csv))
    key = pd.concat(key_rows, ignore_index=True)
    full = full.merge(key, on=["episode_index", "frame_index"], how="left",
                      validate="one_to_one")
    missing = int(full["frame_weight"].isna().sum())
    if missing:
        raise RuntimeError(f"{missing} 行未对齐到 episode/frame key")
    out_csv = out_dir / f"{Path(a.csv).stem}_weight.csv"
    full.to_csv(out_csv, index=False)
    print(f"\nannotated copy -> {out_csv}")

    total_eq1_pct = 100.0 * n_eq1_total / n_frame
    stats = {
        "csv": str(a.csv),
        "method": "asymmetric trapezoid weight, max over events "
                  "(t0=hold_start/hold_end of each grasp cycle, L/R arms)",
        "wmax": wmax, "weight_base": WEIGHT_BASE,
        "total": {"episodes": n_ep, "frames": n_frame,
                  "weight_eq1": n_eq1_total, "weight_gt1": n_gt1_total,
                  "weight_eq1_pct": round(total_eq1_pct, 3)},
        "per_task": {str(ti): v for ti, v in per_task.items()},
        "per_episode": per_episode,
    }
    if train_eps is not None:
        stats["train_subset"] = {
            "episodes": len([e for e in train_eps
                             if e in set().union(*(set(map(int, v))
                             for v in eps_by.values())) or e in train_eps]),
            "frames": train_frame, "weight_eq1": train_eq1,
            "weight_eq1_pct": round(100.0 * train_eq1 / train_frame, 3),
        }
    stats_path = out_dir / "weight_apply_stats.json"
    stats_path.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"{'task':>4} {'slug':<18} {'episodes':>8} {'frames':>8} "
          f"{'weight=1':>9} {'w=1 pct':>9}")
    for ti in selected:
        s = per_task.get(ti)
        if s:
            print(f"{ti:>4} {s['slug']:<18} {s['episodes']:>8} {s['frames']:>8} "
                  f"{s['weight_eq1']:>9} {s['weight_eq1_pct']:>8.2f}%")
    print(f"总计    : {n_ep} episodes / {n_frame} frames, "
          f"weight=1={n_eq1_total} ({total_eq1_pct:.2f}%), "
          f"weight>1={n_gt1_total} ({100.0-total_eq1_pct:.2f}%)")
    if train_eps is not None:
        tr = stats["train_subset"]
        print(f"train   : {tr['episodes']} episodes / {tr['frames']} frames, "
              f"weight=1={tr['weight_eq1']} ({tr['weight_eq1_pct']:.2f}%)")
    print(f"stats -> {stats_path}")


def cmd_dist(args) -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--csv", required=True,
                        help="apply 产物 *_weight.csv (含 frame_weight 列)")
    parser.add_argument("--tasks", default="all")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    a = parser.parse_args(args)
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(a.csv, usecols=["episode_index", "task_index",
                                     "frame_index", "frame_weight"])
    df["frame_weight"] = df["frame_weight"].astype(float)
    selected = (
        sorted(df["task_index"].unique())
        if a.tasks == "all"
        else [int(x) for x in a.tasks.split(",")]
    )
    stats_by_task: dict[int, dict] = {}
    records: list[dict] = []
    n_frame = n_eq1 = 0
    for ti in selected:
        sub = df[df["task_index"] == ti]
        eq1 = int(np.isclose(sub["frame_weight"], WEIGHT_BASE).sum())
        n_frame += len(sub); n_eq1 += eq1
        recs = []
        for ep, g in sub.groupby("episode_index"):
            e1 = int(np.isclose(g["frame_weight"], WEIGHT_BASE).sum())
            recs.append({"task_index": ti, "episode_index": int(ep),
                         "eq1_pct": round(100.0 * e1 / len(g), 2)})
        records.extend(recs)
        stats_by_task[ti] = {
            "slug": TASK_SLUGS.get(ti, f"task_{ti:03d}"),
            "frames": int(len(sub)), "eq1": eq1,
            "eq1_pct": round(100.0 * eq1 / len(sub), 2),
            "_all_weights": sub["frame_weight"].to_numpy(),
        }
        print(f"  task {ti} ({stats_by_task[ti]['slug']}): frames="
              f"{len(sub):,} weight=1={eq1:,} ({stats_by_task[ti]['eq1_pct']:.2f}%)")
    print(f"total: {n_frame:,} frames, weight=1={n_eq1:,} "
          f"({100.0*n_eq1/n_frame:.2f}%)")
    out_png = out_dir / "weight_distribution.png"
    plot_weight_distribution(stats_by_task, records, df["frame_weight"].max(),
                             out_png)
    print(f"distribution figure -> {out_png}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("preview", help="每任务抽样 n ep 画 权重+爪夹 图")
    sub.add_parser("apply", help="全部 episode 逐帧赋权重, 写副本 CSV + stats")
    sub.add_parser("dist", help="读 apply 产物, 画权重分布图")
    args, rest = parser.parse_known_args()
    if args.command == "preview":
        cmd_preview(rest)
    elif args.command == "apply":
        cmd_apply(rest)
    elif args.command == "dist":
        cmd_dist(rest)


if __name__ == "__main__":
    main()
