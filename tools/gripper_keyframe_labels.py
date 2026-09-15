#!/usr/bin/env python3
"""真实数据集 episode 的爪夹关键节点标注与可视化（配置驱动）。

复用 tools/keyframe_detect.py 的检测引擎 detect_gripper_keyframes，**不修改**该文件
（其绝对阈值服务于仿真数据，改动会波及仿真流水线）。本工具在其上补三层：

1. **逐臂归一化**：真实数据两臂的"闭合值"不等于 0 且因任务而异
   （实测 task 0 右臂闭合到 ≈0.67、task 2 两臂 ≈0.35、task 4 两臂 ≈0.55），
   绝对阈值会漏检。按 docs/xyz_gripper_event_segmentation.md §3.2 要求
   "阈值应使用归一化开度，而不是直接绑定某一种夹爪的原始数值"，
   对每个 (episode, 臂) 用自身 [p_low, p_high] 分位区间线性归一化后再检测。
2. **配置驱动的 episode 选择**：--config 复用 configs/real_lerobot_v30_ee.json
   的 csv / tasks_parquet / task_slugs / episodes，保证与 HTML 可视化同一批 episode。
3. **标签落盘 + 逐任务总览图**：输出四类节点的逐帧标签 CSV 与每任务一张的
   多 episode 总览图（Feishu 直发友好）。

四类节点与用户命名的对应（见 docs/xyz_gripper_event_segmentation.md §3.2
与 configs/xyz_segment_config.json::_anchor_map）：

    close_start                  夹爪开始有效闭合
    hold_start   ≡ close_end     闭合结束并进入稳定保持   （用户称 close_end / holder_start）
    hold_end     ≡ open_start    夹爪开始有效张开         （用户称 holder_end）
    open_full    ≡ open_end      打开结束并回到稳定状态

**注意 hold_start 与 close_end 是同一帧**（同一节点的两套叫法），故图上两者共用一条竖线。
标签列中写作 `close_end|holder_start`，以同时保留两套命名。

用法:
    python tools/gripper_keyframe_labels.py --config configs/real_lerobot_v30_ee.json \\
        --out outputs/gripper_labels_real
    python tools/gripper_keyframe_labels.py --config ... --episodes 0,300 --no-normalize
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
    episode_state,
    load_state_frame_df,
    load_tasks,
)

# 绘图风格与 keyframe_detect.py 保持一致
INK = "#0b0b0b"; MUT = "#898781"; GRID = "#e1e0d9"; SURF = "#fcfcfb"
C_LEFT = "#2a78d6"
C_RIGHT = "#eb6834"

# 引擎节点 -> (绘图用颜色, 线型, 图例文案, 标签列文案)
NODE_STYLE = [
    ("close_start", "#1baf7a", "--", "close_start", "close_start"),
    ("hold_start", "#d62728", "-.", "close_end / holder_start", "close_end|holder_start"),
    ("hold_end", "#9467bd", ":", "holder_end", "holder_end"),
    ("open_full", "#f2a900", "-", "open_full", "open_full"),
]
NODE_ORDER = [n for n, *_ in NODE_STYLE]


# ---------------------------------------------------------------------------
# 配置 / 路径
# ---------------------------------------------------------------------------

def resolve_csv(raw: str) -> Path:
    """解析配置里的 csv 路径。

    configs/real_lerobot_v30_ee.json 的 "csv" 指向 data/real_lerobot_v30_ee.csv，
    实际文件已移到同名目录下 (data/real_lerobot_v30_ee/real_lerobot_v30_ee.csv)。
    两种布局都兼容，并回报实际使用的那一个。
    """
    p = ROOT / raw if not Path(raw).is_absolute() else Path(raw)
    if p.is_file():
        return p
    alt = p.parent / p.stem / p.name
    if alt.is_file():
        return alt
    raise FileNotFoundError(f"csv 不存在: {p} (回退候选 {alt} 也不存在)")


def load_config(path: Path) -> dict:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    cfg["_csv_path"] = resolve_csv(cfg["csv"])
    tp = cfg.get("tasks_parquet")
    cfg["_tasks_path"] = (ROOT / tp) if tp and (ROOT / tp).is_file() else None
    cfg["_slugs"] = {int(k): v for k, v in cfg.get("task_slugs", {}).items()}
    cfg["_episodes"] = [int(e) for e in cfg["episodes"]]
    return cfg


# ---------------------------------------------------------------------------
# 检测（含逐臂归一化）
# ---------------------------------------------------------------------------

def normalize_per_arm(g: np.ndarray, lo_pct: float, hi_pct: float) -> tuple[np.ndarray, float, float]:
    """把单个 (episode, 臂) 的开度曲线按其自身 [lo_pct, hi_pct] 分位线性归一化到 [0,1]。

    真实数据的两臂闭合值不同且不为 0，绝对阈值无法通用（doc §3.2）。
    返回 (归一化曲线, lo, hi)，lo/hi 供报告与复现。
    """
    lo, hi = np.percentile(g, [lo_pct, hi_pct])
    if hi - lo < 1e-6:
        return np.zeros_like(g), float(lo), float(hi)
    return np.clip((g - lo) / (hi - lo), 0.0, 1.0), float(lo), float(hi)


def detect_side(
    g: np.ndarray,
    frames: np.ndarray,
    *,
    normalize: bool,
    lo_pct: float,
    hi_pct: float,
    detect_kwargs: dict,
) -> tuple[dict[str, np.ndarray], dict]:
    """单臂检测。normalize=True 时先按自身分位区间归一化再检测。"""
    if normalize:
        gd, lo, hi = normalize_per_arm(g, lo_pct, hi_pct)
    else:
        gd, lo, hi = g, float(np.nanmin(g)), float(np.nanmax(g))
    kf = detect_gripper_keyframes(gd, frames, **detect_kwargs)
    return kf, {"norm_lo": lo, "norm_hi": hi}


def frame_labels(
    n_frames: int, frames: np.ndarray, kf: dict[str, np.ndarray]
) -> np.ndarray:
    """把四类节点摊成逐帧标签数组（用户命名，'|' 连接多标签）。"""
    frame_to_pos = {int(f): i for i, f in enumerate(frames)}
    out = np.full(n_frames, "none", dtype=object)
    for key, _c, _ls, _lbl, text in NODE_STYLE:
        for x in kf[key]:
            x = int(x)
            if x == INCOMPLETE:
                continue
            pos = frame_to_pos.get(x)
            if pos is None:
                continue
            out[pos] = text if out[pos] == "none" else f"{out[pos]}|{text}"
    return out


# ---------------------------------------------------------------------------
# 绘图
# ---------------------------------------------------------------------------

def _style():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "axes.edgecolor": MUT, "axes.labelcolor": INK, "axes.titlecolor": INK,
        "text.color": INK, "xtick.color": INK, "ytick.color": INK,
        "figure.facecolor": SURF, "axes.facecolor": SURF, "grid.color": GRID,
        "font.family": "sans-serif", "figure.dpi": 110,
    })
    return plt


def plot_task_overview(
    task_name: str,
    slug: str,
    entries: list[dict],
    out_path: Path,
    *,
    n_cols: int = 2,
) -> None:
    """每任务一张总览图：行 = episode，列 = 左/右臂。

    entries: [{"episode_index", "frames", "left"(原始曲线), "right", "kf", "lo/hi"}]
    """
    plt = _style()
    n = len(entries)
    fig, axes = plt.subplots(n, n_cols, figsize=(15, 1.6 * n + 1.2), sharex=False)
    axes = np.atleast_2d(axes)
    if axes.shape[1] != n_cols:  # n==1 时的形状修正
        axes = axes.reshape(n, n_cols)

    fig.suptitle(
        f"{task_name}\n{slug} — keyframe nodes per gripper cycle "
        f"(close_start / close_end·holder_start / holder_end / open_full)",
        fontsize=10.5, y=0.997,
    )

    for r, ent in enumerate(entries):
        for c, (side_key, idx, color, side_name) in enumerate([
            ("left", GRIP_L, C_LEFT, "left"), ("right", GRIP_R, C_RIGHT, "right"),
        ]):
            ax = axes[r, c]
            fr = ent["frames"]
            raw = ent[side_key]                      # 原始开度（画图用真实值）
            ax.plot(fr, raw, color=color, ls="-", lw=1.0)
            for key, lc, ls, _lbl, _txt in NODE_STYLE:
                for x in ent["kf"][side_key][key]:
                    if x == INCOMPLETE:
                        continue
                    ax.axvline(x, color=lc, linestyle=ls, alpha=0.85, lw=1.0)
            lo, hi = ent["norm"][side_key]["norm_lo"], ent["norm"][side_key]["norm_hi"]
            ax.axhline(lo, color=MUT, lw=0.6, ls=":", alpha=0.6)
            ax.axhline(hi, color=MUT, lw=0.6, ls=":", alpha=0.6)
            ax.set_ylim(min(-0.05, lo - 0.05), max(1.05, hi + 0.05))
            ax.grid(alpha=0.22)
            if c == 0:
                ax.set_ylabel(f"ep {ent['episode_index']}\n{side_name}", fontsize=8)
            else:
                ax.set_ylabel(side_name, fontsize=8)
            ax.tick_params(labelsize=7)
            if r == 0:
                ax.set_title(f"{side_name} gripper", fontsize=9)
            if r == n - 1:
                ax.set_xlabel("frame index", fontsize=8)

    handles = [
        plt.Line2D([0], [0], color=lc, ls=ls, lw=1.3, label=lbl)
        for _k, lc, ls, lbl, _t in NODE_STYLE
    ]
    axes[0, 0].legend(handles=handles, fontsize=7.5, loc="best", ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs" / "real_lerobot_v30_ee.json"),
                        help="数据集配置 JSON (csv / tasks_parquet / task_slugs / episodes)")
    parser.add_argument("--episodes", default=None,
                        help="覆盖配置的 episode 白名单, 逗号分隔")
    parser.add_argument("--no-normalize", action="store_true",
                        help="关闭逐臂归一化, 直接用原始开度 + 绝对阈值 (对照用)")
    parser.add_argument("--lo-pct", type=float, default=2.0, help="归一化下分位")
    parser.add_argument("--hi-pct", type=float, default=98.0, help="归一化上分位")
    parser.add_argument("--min-prominence", type=float, default=0.15,
                        help="运动段幅度下限 (归一化空间, doc §3.2: 完整开合范围的 10~15%%)")
    parser.add_argument("--eps", type=float, default=0.02, help="diff 变化阈值")
    parser.add_argument("--hold-min-len", type=int, default=3, help="水平保持段最少帧数")
    parser.add_argument("--open-level", type=float, default=0.9, help="视作打开的阈值")
    parser.add_argument("--no-incomplete", action="store_true",
                        help="不标记结尾未回开的周期")
    parser.add_argument("--out", default=str(ROOT / "outputs" / "gripper_labels_real"),
                        help="输出目录")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    out_dir = Path(args.out)
    episodes = ([int(x) for x in args.episodes.split(",")] if args.episodes
                else cfg["_episodes"])

    print(f"csv       : {cfg['_csv_path'].relative_to(ROOT)}")
    print(f"tasks     : {cfg['_tasks_path']}")
    print(f"episodes  : {len(episodes)}")
    print(f"normalize : {not args.no_normalize} "
          f"(p{args.lo_pct:g}~p{args.hi_pct:g}, min_prominence={args.min_prominence})")

    df = load_state_frame_df(cfg["_csv_path"])
    task_names = load_tasks(cfg["_tasks_path"]) if cfg["_tasks_path"] else {}

    detect_kwargs = {
        "eps": args.eps,
        "min_prominence": args.min_prominence,
        "hold_min_len": args.hold_min_len,
        "open_level": args.open_level,
        "allow_incomplete": not args.no_incomplete,
    }

    # --- 逐 episode 检测 ---
    per_ep: dict[int, dict] = {}
    label_frames: list[pd.DataFrame] = []
    report_rows: list[dict] = []

    for ep in episodes:
        state, frames = episode_state(df, ep)
        ti = int(df[df["episode_index"] == ep]["task_index"].iloc[0])
        ent = {"episode_index": ep, "task_index": ti, "frames": frames,
               "kf": {}, "norm": {}, "left": state[:, GRIP_L], "right": state[:, GRIP_R]}
        for side, gi in [("left", GRIP_L), ("right", GRIP_R)]:
            kf, nrm = detect_side(state[:, gi], frames, normalize=not args.no_normalize,
                                  lo_pct=args.lo_pct, hi_pct=args.hi_pct,
                                  detect_kwargs=detect_kwargs)
            ent["kf"][side] = kf
            ent["norm"][side] = nrm
            n_cyc = len(kf["close_start"])
            n_inc = int((kf["open_full"] == INCOMPLETE).sum())
            report_rows.append(dict(
                episode_index=ep, task_index=ti, side=side, n_cycles=n_cyc,
                n_incomplete=n_inc,
                norm_lo=round(nrm["norm_lo"], 3), norm_hi=round(nrm["norm_hi"], 3),
                **{k: len(kf[k]) for k in NODE_ORDER},
            ))
        per_ep[ep] = ent

        label_frames.append(pd.DataFrame({
            "episode_index": ep,
            "task_index": ti,
            "frame_index": frames,
            "left_keyframe_label": frame_labels(len(frames), frames, ent["kf"]["left"]),
            "right_keyframe_label": frame_labels(len(frames), frames, ent["kf"]["right"]),
        }))

    # --- 标签 CSV ---
    lab = pd.concat(label_frames, ignore_index=True).sort_values(
        ["episode_index", "frame_index"]).reset_index(drop=True)
    lab["left_is_keyframe"] = (lab["left_keyframe_label"] != "none").astype(int)
    lab["right_is_keyframe"] = (lab["right_keyframe_label"] != "none").astype(int)
    out_dir.mkdir(parents=True, exist_ok=True)
    lab_path = out_dir / "gripper_keyframe_labels.csv"
    lab.to_csv(lab_path, index=False)

    rep = pd.DataFrame(report_rows)
    rep_path = out_dir / "detection_report.csv"
    rep.to_csv(rep_path, index=False)

    # --- 每任务总览图 ---
    by_task: dict[int, list[int]] = {}
    for ep in episodes:
        by_task.setdefault(per_ep[ep]["task_index"], []).append(ep)

    fig_paths = []
    for ti, eps in sorted(by_task.items()):
        slug = cfg["_slugs"].get(ti, f"task_{ti:03d}")
        name = task_names.get(ti, slug)
        p = out_dir / f"{slug}_overview.png"
        plot_task_overview(name, slug, [per_ep[e] for e in sorted(eps)], p)
        fig_paths.append(p)
        sub = rep[rep["task_index"] == ti]
        print(f"  task {ti} ({slug:26s}) {len(eps)} eps | cycles "
              f"L={sub[sub.side=='left'].n_cycles.sum()} "
              f"R={sub[sub.side=='right'].n_cycles.sum()} -> {p.name}")

    print(f"\nlabels  -> {lab_path}  ({len(lab)} rows)")
    print(f"report  -> {rep_path}")
    print(f"figures -> {out_dir}  ({len(fig_paths)} png)")


if __name__ == "__main__":
    main()
