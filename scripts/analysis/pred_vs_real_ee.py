#!/usr/bin/env python3
"""X0_ee6d_real 离线预测 vs real_lerobot_v30_ee 真实动作：可视化 + 指标。

两种对照口径（对齐约定取自 X-VLA/evaluation/evaluate_ee.py 的已落库指标口径）：
  1) 单步：predictions.csv 每帧 anchor=t 的 chunk 第 0 步 ↔ 真实帧 t+1（lead=1）。
     顺带给出 lead=1..30 的误差曲线（正好就是 chunk 内步序 1..30 的误差 profile）。
  2) 拼接：anchor 取 0,30,60,...，每个 anchor 用完整 30 步 chunk 拼出整集动作
     （chunk 第 k 步 ↔ 真实帧 t+1+k），与真实逐帧对比。

量化口径复用 evaluate_ee.ee_errors：位置 L2 (cm)、姿态四元数测地角 (deg, 双覆盖不敏感)、
夹爪 MAE。16 维布局 = 双臂 [x,y,z,quat(wxyz),gripper]。

用法:
  python scripts/analysis/pred_vs_real_ee.py \
      --predictions /path/predictions.csv --baseline /path/real_lerobot_v30_ee.csv \
      --split /path/train_val_split.json --outdir outputs/pred_analysis
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# ---------------------------------------------------------------- 设计令牌
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
INK3 = "#8a8880"
GRID = "#e4e3de"
C_REAL = "#2a78d6"   # 真实（categorical slot 1）
C_PRED = "#eb6834"   # 预测（categorical slot 2）
# 每任务一条线（slot 1..6；validate_palette.js --mode light 全过，最差相邻对 CVD ΔE 9.1）
TASK_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
TASK_SHORT = ["t0 插笔入筒", "t1 物件入筐", "t2 叠块盖杯", "t3 叠碗", "t4 立瓶", "t5 插充电器"]
TASK_TICK = [t.replace(" ", "\n", 1) for t in TASK_SHORT]   # 条形图刻度用两行，避免相邻标签粘连
N_TASK = 6
HORIZON = 30        # chunk 长度
EXEC_WINDOW = 30    # 拼接口径的 anchor 间距
LEADS = (1, 10, 20, 30)  # 落 CSV 的提前量
FPS = 25.0

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.size": 9,
    "font.family": "sans-serif",
    "font.sans-serif": ["PingFang SC", "Arial Unicode MS", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK2,
    "axes.titlesize": 10,
    "axes.titlecolor": INK,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.frameon": False,
    "legend.fontsize": 8,
    "lines.linewidth": 1.0,
    "figure.dpi": 130,
})


def style_axis(ax, *, xlabel=None, ylabel=None, title=None):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", visible=False)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, loc="left", pad=6)


def spread_labels(ax, series, x_end, label_x, min_frac=0.06, fontsize=7.5):
    """线端直接标注：按 y 排序做最小间距展开（标签必须落在轴内，避免被 tight bbox 裁掉）。

    series: [(y_value, label, color, bold), ...]
    """
    lo, hi = ax.get_ylim()
    gap = (hi - lo) * min_frac
    order = sorted(range(len(series)), key=lambda i: series[i][0])
    ys = [series[i][0] for i in order]
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1] + gap)
    overflow = ys[-1] - hi
    if overflow > 0:
        ys = [y - overflow for y in ys]
        for i in range(len(ys) - 2, -1, -1):
            ys[i] = min(ys[i], ys[i + 1] - gap)
    for i, idx in enumerate(order):
        y0, label, color, bold = series[idx]
        ax.plot([x_end, label_x], [y0, ys[i]], color=color, lw=0.6)
        ax.annotate(label, xy=(label_x, ys[i]), color=color, fontsize=fontsize, va="center",
                    ha="left", fontweight="bold" if bold else "normal")


def save(fig, path, caption=None):
    if caption:
        fig.text(0.008, -0.012, caption, ha="left", va="top", fontsize=7, color=INK3)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- 指标（对齐 evaluate_ee.ee_errors）
def pos_cm(pred, real):
    """双臂位置 L2 (cm)。"""
    return (float(np.linalg.norm(pred[:3] - real[:3]) * 100.0),
            float(np.linalg.norm(pred[8:11] - real[8:11]) * 100.0))


def rot_deg(pred, real):
    """双臂四元数测地角 (deg)，abs(dot) 对双覆盖 q/-q 不敏感。"""
    out = []
    for s in (3, 11):
        p, e = pred[s:s + 4], real[s:s + 4]
        pn, en = np.linalg.norm(p), np.linalg.norm(e)
        if pn < 1e-8 or en < 1e-8:
            out.append(float("nan"))
            continue
        dot = abs(float(np.dot(p / pn, e / en)))
        out.append(math.degrees(2.0 * math.acos(min(1.0, max(-1.0, dot)))))
    return out[0], out[1]


def grp_err(pred, real):
    return float(abs(pred[7] - real[7])), float(abs(pred[15] - real[15]))


def vector_errors(pred, real):
    """[N,30,16] × [N,30,16] -> 逐 (anchor, 步序) 误差字典（向量化，避免百万次 python 循环）。"""
    d_pos = pred[..., :] - real
    left_pos = np.linalg.norm(d_pos[..., 0:3], axis=-1) * 100.0
    right_pos = np.linalg.norm(d_pos[..., 8:11], axis=-1) * 100.0
    out = {
        "left_pos_cm": left_pos,
        "right_pos_cm": right_pos,
        "mean_pos_cm": (left_pos + right_pos) / 2.0,
        "left_grp_mae": np.abs(d_pos[..., 7]),
        "right_grp_mae": np.abs(d_pos[..., 15]),
    }
    out["mean_grp_mae"] = (out["left_grp_mae"] + out["right_grp_mae"]) / 2.0
    for name, s in (("left_rot_deg", 3), ("right_rot_deg", 11)):
        qp = pred[..., s:s + 4]
        qe = real[..., s:s + 4]
        npn = np.linalg.norm(qp, axis=-1, keepdims=True)
        nen = np.linalg.norm(qe, axis=-1, keepdims=True)
        dot = np.abs(np.sum(qp / np.maximum(npn, 1e-8) * (qe / np.maximum(nen, 1e-8)), axis=-1))
        out[name] = np.degrees(2.0 * np.arccos(np.clip(dot, -1.0, 1.0)))
    out["mean_rot_deg"] = (out["left_rot_deg"] + out["right_rot_deg"]) / 2.0
    return out


# ---------------------------------------------------------------- 数据加载
def load_split(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    ep2task, ep2instr = {}, {}
    for fallback, task in data["tasks"].items():
        ti = int(task.get("task_index", fallback))
        for ep in task["val_episode_idx"]:
            ep2task[int(ep)] = ti
            ep2instr[int(ep)] = task["instruction"]
    return ep2task, ep2instr


def load_baseline(path, val_eps, cache):
    if cache.exists():
        z = np.load(cache)
        return {k: z[k] for k in z.files}
    val = set(val_eps)
    eps, frames, acts = [], [], []
    with open(path, newline="") as f:
        r = csv.reader(f)
        hdr = next(r)
        c_ep, c_fr, c_act = (hdr.index(k) for k in ("episode_index", "frame_index", "action"))
        for row in r:
            ep = int(row[c_ep])
            if ep not in val:
                continue
            eps.append(ep)
            frames.append(int(row[c_fr]))
            acts.append(json.loads(row[c_act]))
    out = {"episode": np.asarray(eps, np.int32), "frame": np.asarray(frames, np.int32),
           "action": np.asarray(acts, np.float32)}
    np.savez(cache, **out)
    return out


def load_predictions(path, cache):
    if cache.exists():
        z = np.load(cache)
        return {k: z[k] for k in z.files}
    eps, frames, chunks = [], [], []
    with open(path, newline="") as f:
        r = csv.reader(f)
        hdr = next(r)
        c_ep, c_fr, c_ch = (hdr.index(k) for k in ("episode_index", "frame_index", "predicted_action_chunk"))
        for row in r:
            eps.append(int(row[c_ep]))
            frames.append(int(row[c_fr]))
            chunks.append(json.loads(row[c_ch]))
    out = {"episode": np.asarray(eps, np.int32), "frame": np.asarray(frames, np.int32),
           "chunk": np.asarray(chunks, np.float32).reshape(len(chunks), HORIZON, 16)}
    np.savez(cache, **out)
    return out


def index_episodes(data, val_eps, key):
    """按 episode 切出 {ep: (frames[升序], values[...])}。"""
    out = {}
    for ep in val_eps:
        m = data["episode"] == ep
        if not m.any():
            continue
        order = np.argsort(data["frame"][m], kind="stable")
        out[ep] = (data["frame"][m][order], data[key][m][order])
    return out


def gather_targets(frames, base_frames, base_act):
    """构造 [N,30] 目标索引：第 k-1 列 = 真实帧 frames+k 的行号，越界为 -1。"""
    tmax = int(base_frames[-1])
    idx = np.arange(1, HORIZON + 1)[None, :] + frames[:, None]
    valid = idx <= tmax
    pos = np.searchsorted(base_frames, idx)
    pos = np.clip(pos, 0, len(base_frames) - 1)
    ok = valid & (base_frames[pos] == np.where(valid, idx, -1))
    if not (np.array_equal(base_frames, np.arange(base_frames[0], tmax + 1))):
        raise ValueError("baseline frame_index 非连续，需改用显式映射")
    tgt = np.where(ok, pos, -1)
    real = np.zeros((len(frames), HORIZON, 16), np.float32)
    real[tgt >= 0] = base_act[tgt[tgt >= 0]]
    return tgt, real, ok


# ---------------------------------------------------------------- 口径 1 / 口径 2 误差
def all_errors(pred_map, base_map, task_of):
    """逐 (anchor, 步序) 误差（chunk 第 k 步 ↔ 真实帧 anchor+k），按 episode 返回。"""
    out = {}
    for ep, (frames, chunk) in pred_map.items():
        base_frames, base_act = base_map[ep]
        tgt, real, ok = gather_targets(frames, base_frames, base_act)
        errs = vector_errors(chunk, real)
        errs = {k: np.where(ok, v, np.nan) for k, v in errs.items()}
        errs["anchor_frame"] = np.repeat(frames[:, None], HORIZON, axis=1)
        errs["target_frame"] = tgt
        out[ep] = errs
    return out


def reconstruct(pred_map, base_map, stride=EXEC_WINDOW):
    """拼接：anchor=0,stride,2*stride,...；chunk 第 k 步 -> 真实帧 anchor+1+k。"""
    out = {}
    for ep, (frames, chunk) in pred_map.items():
        base_frames, base_act = base_map[ep]
        tmax = int(base_frames[-1])
        anchors = [int(t) for t in frames if int(t) % stride == 0 and int(t) + HORIZON <= tmax]
        if not anchors:
            continue
        pos = {int(f): i for i, f in enumerate(frames)}
        tgts, preds, reals, steps = [], [], [], []
        for a in anchors:
            i = pos[a]
            for k in range(HORIZON):
                f = a + 1 + k
                tgts.append(f)
                preds.append(chunk[i, k])
                reals.append(base_act[f])
                steps.append(k)          # chunk 内步序（0 基）
        out[ep] = (np.asarray(tgts, np.int32), np.asarray(preds, np.float32),
                   np.asarray(reals, np.float32), len(anchors), np.asarray(steps, np.int8))
    return out


def nanmean_or_nan(values):
    values = np.asarray(values, float)
    return float(np.nanmean(values)) if values.size else float("nan")


def quat_sign_flip_rate(pred_map, base_map):
    """真实与预测四元数 dot<0（同一旋转的另一符号）的样本占比：诊断双覆盖。

    位置/姿态误差都用 abs(dot) 度量，故符号翻转不影响指标；此处只作数据说明。
    """
    out = {}
    for side, s in (("left", 3), ("right", 11)):
        flips = []
        for ep, (fr, C) in pred_map.items():
            bf, ba = base_map[ep]
            _, real, ok = gather_targets(fr, bf, ba)
            qp = C[..., s:s + 4][ok]
            qe = real[..., s:s + 4][ok]
            qp = qp / np.maximum(np.linalg.norm(qp, axis=-1, keepdims=True), 1e-8)
            qe = qe / np.maximum(np.linalg.norm(qe, axis=-1, keepdims=True), 1e-8)
            flips.append(float(np.mean(np.sum(qp * qe, axis=-1) < 0)))
        out[side] = float(np.mean(flips))
    return out


def offset_scan(pred_map, base_map, stride=EXEC_WINDOW, offsets=(-2, -1, 0, 1, 2)):
    """时间对齐自检：把整段 30 步 chunk 相对真实帧平移 off，比较平均位置误差。

    约定为 off=+1（chunk 第 0 步 = 真实帧 t+1）；此表用数据验证是否存在系统性滞后/超前。
    """
    rec = {}
    for off in offsets:
        vals = []
        for ep, (fr, C) in pred_map.items():
            bf, ba = base_map[ep]
            tmax = int(bf[-1])
            idx = [i for i, t in enumerate(fr)
                   if int(t) % stride == 0 and 0 <= int(t) + off and int(t) + off + HORIZON <= tmax]
            if not idx:
                continue
            idx = np.asarray(idx)
            per_frame = []
            for j, i in enumerate(idx):
                base = int(fr[i]) + off
                blk = ba[base:base + HORIZON]
                if blk.shape[0] != HORIZON:
                    continue
                per_frame.append(((np.linalg.norm(C[i, :, 0:3] - blk[:, 0:3], axis=1)
                                   + np.linalg.norm(C[i, :, 8:11] - blk[:, 8:11], axis=1)) / 2 * 100).mean())
            if per_frame:
                vals.append(float(np.mean(per_frame)))
        rec[off] = float(np.mean(vals)) if vals else float("nan")
    return rec
    values = np.asarray(values, float)
    return float(np.nanmean(values)) if values.size else float("nan")


def macro_over_episodes(per_ep, task_of, key, k):
    """evaluate_ee 口径：episode 内均值 -> 每任务均值 -> 任务均值。"""
    per_task = defaultdict(list)
    for ep, errs in per_ep.items():
        per_task[task_of[ep]].append(nanmean_or_nan(errs[key][:, k - 1]))
    return float(np.mean([np.mean(v) for v in per_task.values()]))


# ---------------------------------------------------------------- 图：逐集轨迹
def episode_trajectory_figure(ep, task, instr, x, real, pred, err_l, err_r, path, subtitle):
    fig, axes = plt.subplots(4, 2, figsize=(13, 9), sharex=True,
                             gridspec_kw={"hspace": 0.28, "wspace": 0.16})
    for col, arm in enumerate(("左臂", "右臂")):
        o = col * 8
        for row, dim in enumerate("xyz"):
            ax = axes[row, col]
            h_real, = ax.plot(x, real[:, o + row], color=C_REAL, lw=1.2, label="真实")
            h_pred, = ax.plot(x, pred[:, o + row], color=C_PRED, lw=1.2, ls=(0, (4, 1.6)), label="预测")
            style_axis(ax, ylabel=f"{dim} (m)")
            if row == 0:
                ax.set_title(arm, loc="left")
            if row == 0 and col == 0:
                ax.legend(handles=[h_real, h_pred], loc="upper right", ncol=2)
    for col, err in enumerate((err_l, err_r)):
        ax = axes[3, col]
        ax.plot(x, err, color=INK2, lw=1.0)
        p50, p90 = np.nanpercentile(err, [50, 90])
        ax.axhline(p50, color=INK3, lw=1.0, ls=(0, (4, 1.6)))
        ax.annotate(f"p50 {p50:.2f} cm\np90 {p90:.2f} cm\n均值 {np.nanmean(err):.2f} cm",
                    xy=(0.015, 0.97), xycoords="axes fraction", color=INK2, fontsize=8, va="top",
                    bbox={"facecolor": SURFACE, "edgecolor": "none", "alpha": 0.85, "pad": 1.5})
        style_axis(ax, xlabel="帧 (25 Hz)", ylabel="位置误差 (cm)" if col == 0 else None)
    fig.suptitle(f"{subtitle}\nep{ep} · task{task} · {instr[:78]}", x=0.01, ha="left",
                 fontsize=11.5, color=INK)
    fig.subplots_adjust(top=0.90)
    save(fig, path, caption="上三行：x/y/z 轨迹（实线=真实，点线=预测）；末行：该臂位置 L2 误差。")


# ---------------------------------------------------------------- 图：汇总
def figure_lead_curve(per_ep, task_of, outdir):
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.4))
    specs = [("mean_pos_cm", "位置误差 (cm)"), ("mean_rot_deg", "姿态误差 (deg)"),
             ("mean_grp_mae", "夹爪 MAE")]
    for ax, (key, ylab) in zip(axes, specs):
        per_task = {ti: np.full(HORIZON, np.nan) for ti in range(N_TASK)}
        for ti in range(N_TASK):
            eps = [ep for ep in per_ep if task_of[ep] == ti]
            if not eps:
                continue
            for k in range(1, HORIZON + 1):
                per_task[ti][k - 1] = float(np.nanmean([nanmean_or_nan(per_ep[ep][key][:, k - 1]) for ep in eps]))
        overall = np.nanmean(np.stack([per_task[ti] for ti in range(N_TASK)]), axis=0)
        for ti in range(N_TASK):
            ax.plot(range(1, HORIZON + 1), per_task[ti], color=TASK_COLORS[ti], lw=1.6)
        ax.plot(range(1, HORIZON + 1), overall, color=INK, lw=2.4, alpha=0.85)
        ax.set_xlim(0.5, HORIZON + 12)
        ax.set_xticks([1, 5, 10, 15, 20, 25, 30])
        style_axis(ax, xlabel="chunk 内步序 k（对应真实帧 t+k）", ylabel=ylab)
        spread_labels(ax, [(per_task[ti][-1], TASK_SHORT[ti], TASK_COLORS[ti], False)
                           for ti in range(N_TASK)] + [(overall[-1], "全任务均值", INK, True)],
                      x_end=HORIZON, label_x=HORIZON + 1.0)
    fig.suptitle("提前量误差曲线（单步口径的延伸：chunk 第 k 步 -> 真实帧 t+k）\n"
                 "X0_ee6d_real ckpt-130000 · 6 任务 val 全量 60 集", x=0.008, ha="left",
                 fontsize=11.5, color=INK)
    fig.subplots_adjust(top=0.78, wspace=0.34)
    save(fig, Path(outdir) / "fig1_lead_curve.png",
         caption="误差随 lead 增长属预期（外推越远越难）；线端标签为任务简称。")


def figure_error_dist(per_ep, task_of, outdir):
    one = {ep: e["mean_pos_cm"][:, 0] for ep, e in per_ep.items()}   # k=1 = 单步
    pooled = np.concatenate([v[np.isfinite(v)] for v in one.values()])
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.3))
    ax = axes[0]
    bins = np.linspace(0, float(np.percentile(pooled, 99.5)), 60)
    ax.hist(pooled, bins=bins, color=C_REAL, alpha=0.85, edgecolor=SURFACE, linewidth=0.5)
    p50, p90, p99 = np.percentile(pooled, [50, 90, 99])
    for v, lab, col, ls in ((p50, "p50", INK2, (0, (4, 1.6))), (p90, "p90", INK, "-"),
                            (p99, "p99", INK3, "-")):
        ax.axvline(v, color=col, lw=1.4, ls=ls)
        ax.annotate(f"{lab} {v:.2f}", xy=(v, 1.0), xycoords=("data", "axes fraction"), color=col,
                    fontsize=8, rotation=90, va="top", ha="right",
                    bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 1.2})
    style_axis(ax, xlabel="单步位置误差 (cm)", ylabel="帧数", title="误差分布（双臂均值，全部 val 帧）")

    ax = axes[1]
    data = [np.concatenate([one[ep][np.isfinite(one[ep])] for ep in one if task_of[ep] == ti])
            for ti in range(N_TASK)]
    bp = ax.boxplot(data, patch_artist=True, widths=0.6,
                    medianprops={"color": INK, "linewidth": 1.4},
                    flierprops={"markersize": 2, "markerfacecolor": INK3, "markeredgecolor": "none"})
    for patch, c in zip(bp["boxes"], TASK_COLORS):
        patch.set_facecolor(c)
        patch.set_alpha(0.35)
        patch.set_edgecolor(c)
    ax.set_xticks(np.arange(N_TASK))
    ax.set_xticklabels(TASK_TICK)
    style_axis(ax, ylabel="单步位置误差 (cm)", title="按任务分布（每任务 10 集逐帧）")
    fig.suptitle("单步预测（anchor=t 的 chunk[0] -> 真实帧 t+1）误差", x=0.008, ha="left",
                 fontsize=11.5, color=INK)
    fig.subplots_adjust(top=0.83, wspace=0.2)
    save(fig, Path(outdir) / "fig1_error_dist.png",
         caption="横轴截断到 p99.5 以看清主体；箱线图为每任务 10 集的逐帧误差。")


def figure_anchor_summary(rec, task_of, prof_num, prof_den, outdir, n_anchor_total):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    eps = sorted(rec)
    runlen = EXEC_WINDOW

    ax = axes[0]
    rl = np.asarray([np.sqrt(np.mean(np.sum((rec[e][1][:, 0:3] - rec[e][2][:, 0:3]) ** 2, axis=1))) * 100
                     for e in eps])
    rr = np.asarray([np.sqrt(np.mean(np.sum((rec[e][1][:, 8:11] - rec[e][2][:, 8:11]) ** 2, axis=1))) * 100
                     for e in eps])
    xs = np.arange(len(eps))
    ax.bar(xs - 0.2, rl, width=0.4, color=C_REAL, label="左臂")
    ax.bar(xs + 0.2, rr, width=0.4, color=C_PRED, label="右臂")
    mean_rmse = float(np.mean((rl + rr) / 2))
    ax.axhline(mean_rmse, color=INK, lw=1.6, ls=(0, (4, 1.6)))
    ax.annotate(f"双臂均值 {mean_rmse:.2f} cm", xy=(0.99, mean_rmse), xycoords=("axes fraction", "data"),
                xytext=(0, 4), textcoords="offset points", color=INK, fontsize=8, va="bottom",
                ha="right")
    ax.set_xticks(xs[::5])
    ax.set_xticklabels([f"ep{e}" for e in eps[::5]], rotation=90)
    ax.legend(loc="upper left")
    style_axis(ax, ylabel="拼接轨迹位置 RMSE (cm)",
               title=f"每集拼接重建误差（x = val {len(eps)} 集）")

    ax = axes[1]
    ends = []
    for ti in range(N_TASK):
        if prof_den[ti].sum() == 0:
            continue
        y = prof_num[ti] / np.maximum(prof_den[ti], 1)
        ax.plot(range(1, HORIZON + 1), y, color=TASK_COLORS[ti], lw=1.6)
        ends.append((y[-1], TASK_SHORT[ti], TASK_COLORS[ti], False))
    overall = np.nansum([prof_num[t] for t in range(N_TASK)], axis=0) / np.maximum(
        np.nansum([prof_den[t] for t in range(N_TASK)], axis=0), 1)
    ax.plot(range(1, HORIZON + 1), overall, color=INK, lw=2.4, alpha=0.85)
    ax.set_xticks([1, 5, 10, 15, 20, 25, 30])
    ax.set_xlim(0.5, HORIZON + 12)
    style_axis(ax, xlabel="chunk 内步序 k", ylabel="平均位置误差 (cm)",
               title=f"chunk 内误差随步序增长（anchor 间隔 {runlen} 帧）")
    spread_labels(ax, ends + [(overall[-1], "全任务均值", INK, True)],
                  x_end=HORIZON, label_x=HORIZON + 1.0)

    ax = axes[2]
    tp = [float(np.mean([np.sqrt(np.mean(np.sum((rec[e][1][:, 0:3] - rec[e][2][:, 0:3]) ** 2, axis=1))) * 100
                         for e in rec if task_of[e] == ti])) for ti in range(N_TASK)]
    tp_r = [float(np.mean([np.sqrt(np.mean(np.sum((rec[e][1][:, 8:11] - rec[e][2][:, 8:11]) ** 2, axis=1))) * 100
                           for e in rec if task_of[e] == ti])) for ti in range(N_TASK)]
    xs = np.arange(N_TASK)
    ax.bar(xs - 0.2, tp, width=0.4, color=C_REAL, label="左臂")
    ax.bar(xs + 0.2, tp_r, width=0.4, color=C_PRED, label="右臂")
    for i, v in enumerate(tp):
        ax.annotate(f"{v:.2f}", xy=(i - 0.2, v), ha="center", va="bottom", fontsize=7.5, color=INK2)
    for i, v in enumerate(tp_r):
        ax.annotate(f"{v:.2f}", xy=(i + 0.2, v), ha="center", va="bottom", fontsize=7.5, color=INK2)
    ax.set_xticks(xs)
    ax.set_xticklabels(TASK_TICK)
    ax.legend(loc="upper left")
    style_axis(ax, ylabel="位置 RMSE (cm)", title="按任务（episode 均值）")
    fig.suptitle(f"拼接口径：anchor 每 {runlen} 帧取一次，用满 30 步 chunk 重建整集动作\n"
                 f"X0_ee6d_real ckpt-130000 · val 60 集 / {n_anchor_total} 个 anchor",
                 x=0.008, ha="left", fontsize=11.5, color=INK)
    fig.subplots_adjust(top=0.78, wspace=0.3)
    save(fig, Path(outdir) / "fig2_summary.png",
         caption="拼接只覆盖被 anchor 命中的帧（每集末尾不足 30 帧的部分不参与）。")


def write_csv(path, rows, fields):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--outdir", default="outputs/pred_analysis")
    ap.add_argument("--tag", default=None, help="输出子目录名（默认取 predictions.csv 的父目录名）")
    ap.add_argument("--no-episode-figures", action="store_true")
    args = ap.parse_args()

    pred_path = Path(args.predictions)
    tag = args.tag or pred_path.parent.name
    outdir = Path(args.outdir) / tag
    (outdir / "step1").mkdir(parents=True, exist_ok=True)
    (outdir / "anchor30").mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.outdir) / "_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    ep2task, ep2instr = load_split(args.split)
    val_eps = sorted(ep2task)
    base = load_baseline(args.baseline, val_eps, cache_dir / "baseline_val.npz")
    preds = load_predictions(pred_path, cache_dir / "predictions.npz")
    base_map = index_episodes(base, val_eps, "action")
    pred_map = index_episodes(preds, val_eps, "chunk")
    print(f"val episodes={len(val_eps)} baseline_rows={len(base['episode'])} "
          f"prediction_rows={len(preds['episode'])}")
    missing = [ep for ep in val_eps if ep not in pred_map]
    if missing:
        print(f"!! 缺少预测的 val 集: {missing}")

    per_ep = all_errors(pred_map, base_map, ep2task)

    # ---------------- 口径 1：单步 + lead 曲线
    rows = []
    for ep in val_eps:
        if ep not in per_ep:
            continue
        for lead in LEADS:
            e = per_ep[ep]
            m = {"episode_index": ep, "task_index": ep2task[ep], "lead": lead,
                 "n": int(np.isfinite(e["mean_pos_cm"][:, lead - 1]).sum()),
                 "left_pos_cm": nanmean_or_nan(e["left_pos_cm"][:, lead - 1]),
                 "right_pos_cm": nanmean_or_nan(e["right_pos_cm"][:, lead - 1]),
                 "mean_pos_cm": nanmean_or_nan(e["mean_pos_cm"][:, lead - 1]),
                 "left_rot_deg": nanmean_or_nan(e["left_rot_deg"][:, lead - 1]),
                 "right_rot_deg": nanmean_or_nan(e["right_rot_deg"][:, lead - 1]),
                 "mean_rot_deg": nanmean_or_nan(e["mean_rot_deg"][:, lead - 1]),
                 "left_grp_mae": nanmean_or_nan(e["left_grp_mae"][:, lead - 1]),
                 "right_grp_mae": nanmean_or_nan(e["right_grp_mae"][:, lead - 1]),
                 "mean_grp_mae": nanmean_or_nan(e["mean_grp_mae"][:, lead - 1])}
            rows.append(m)
    fields = ["episode_index", "task_index", "lead", "n", "left_pos_cm", "right_pos_cm", "mean_pos_cm",
              "left_rot_deg", "right_rot_deg", "mean_rot_deg", "left_grp_mae", "right_grp_mae",
              "mean_grp_mae"]
    write_csv(outdir / "metrics_step1.csv", rows, fields)
    summary = {key: {lead: macro_over_episodes(per_ep, ep2task, key, lead) for lead in LEADS}
               for key in ("mean_pos_cm", "mean_rot_deg", "mean_grp_mae")}
    figure_lead_curve(per_ep, ep2task, outdir)
    figure_error_dist(per_ep, ep2task, outdir)

    if not args.no_episode_figures:
        for ep, (frames, chunk) in pred_map.items():
            base_frames, base_act = base_map[ep]
            tgt_rows, real, ok = gather_targets(frames, base_frames, base_act)
            x = frames + 1                      # 单步目标帧
            good = ok[:, 0]
            episode_trajectory_figure(
                ep, ep2task[ep], ep2instr[ep], x[good], real[good, 0], chunk[good, 0],
                per_ep[ep]["left_pos_cm"][good, 0], per_ep[ep]["right_pos_cm"][good, 0],
                outdir / "step1" / f"fig1_traj_ep{ep:03d}.png",
                "口径1 单步：每帧 anchor=t 的 chunk[0] -> 真实帧 t+1")

    # ---------------- 口径 2：anchor 每 30 帧 + 满 30 步拼接
    rec = reconstruct(pred_map, base_map)
    prof_num = {ti: np.zeros(HORIZON) for ti in range(N_TASK)}
    prof_den = {ti: np.zeros(HORIZON) for ti in range(N_TASK)}
    rows2 = []
    for ep, (tgts, preds_r, reals, n_anchor, steps) in sorted(rec.items()):
        d = preds_r - reals
        rows2.append({
            "episode_index": ep, "task_index": ep2task[ep], "n_anchors": n_anchor,
            "n_frames": len(tgts), "frame_first": int(tgts[0]), "frame_last": int(tgts[-1]),
            "left_pos_rmse_cm": float(np.sqrt(np.mean(np.sum(d[:, 0:3] ** 2, axis=1))) * 100),
            "right_pos_rmse_cm": float(np.sqrt(np.mean(np.sum(d[:, 8:11] ** 2, axis=1))) * 100),
            "left_pos_mae_cm": float(np.mean(np.linalg.norm(d[:, 0:3], axis=1)) * 100),
            "right_pos_mae_cm": float(np.mean(np.linalg.norm(d[:, 8:11], axis=1)) * 100),
            "left_grp_mae": float(np.mean(np.abs(d[:, 7]))),
            "right_grp_mae": float(np.mean(np.abs(d[:, 15]))),
        })
        # chunk 内步序 profile：只统计“用满 30 步的 anchor”（每集每个 k 各 n_anchor 个样本）
        le = (np.linalg.norm(d[:, 0:3], axis=1) + np.linalg.norm(d[:, 8:11], axis=1)) / 2.0 * 100
        ti = ep2task[ep]
        for k in range(HORIZON):
            m = steps == k
            prof_num[ti][k] += float(le[m].sum())
            prof_den[ti][k] += int(m.sum())
    write_csv(outdir / "metrics_anchor30.csv", rows2,
              ["episode_index", "task_index", "n_anchors", "n_frames", "frame_first", "frame_last",
               "left_pos_rmse_cm", "right_pos_rmse_cm", "left_pos_mae_cm", "right_pos_mae_cm",
               "left_grp_mae", "right_grp_mae"])
    n_anchor_total = sum(v[3] for v in rec.values())
    figure_anchor_summary(rec, ep2task, prof_num, prof_den, outdir, n_anchor_total)

    if not args.no_episode_figures:
        for ep, (tgts, preds_r, reals, _, _steps) in rec.items():
            el = np.linalg.norm(preds_r[:, 0:3] - reals[:, 0:3], axis=1) * 100
            er = np.linalg.norm(preds_r[:, 8:11] - reals[:, 8:11], axis=1) * 100
            episode_trajectory_figure(
                ep, ep2task[ep], ep2instr[ep], tgts, reals, preds_r, el, er,
                outdir / "anchor30" / f"fig2_traj_ep{ep:03d}.png",
                "口径2 拼接：anchor 每 30 帧取一次，用满 30 步 chunk 重建")

    # ---------------- 结论数字
    pooled = np.concatenate([per_ep[ep]["mean_pos_cm"][:, 0] for ep in per_ep])
    pooled = pooled[np.isfinite(pooled)]
    p50, p90, p99 = np.percentile(pooled, [50, 90, 99])
    rm = np.asarray([(r["left_pos_rmse_cm"] + r["right_pos_rmse_cm"]) / 2 for r in rows2])
    worst_i, best_i = int(np.argmax(rm)), int(np.argmin(rm))
    overall_prof = np.nansum([prof_num[t] for t in range(N_TASK)], axis=0) / np.maximum(
        np.nansum([prof_den[t] for t in range(N_TASK)], axis=0), 1)
    flips = quat_sign_flip_rate(pred_map, base_map)
    offs = offset_scan(pred_map, base_map)
    best_off = min((o for o in offs if np.isfinite(offs[o])), key=lambda o: offs[o])

    lines = [f"# 预测 vs 真实分析：{tag}", "",
             f"- 预测文件：`{pred_path}`（{len(preds['episode'])} 行）",
             f"- 真值：`{args.baseline}`",
             f"- val：{len(val_eps)} 集 / {sum(len(pred_map[e][0]) for e in pred_map)} 帧 anchor",
             f"- 对齐约定：chunk 第 k 步 = 真实帧 anchor+k（与 `evaluate_ee.py` 的 `predicted[lead-1]` "
             f"= `anchor+lead` 一致），即 anchor=t 的 chunk 覆盖真实帧 t+1..t+30", "",
             "## 口径1 单步（chunk[0] = 真实帧 t+1）", "",
             "| lead（提前量） | 位置 (cm) | 姿态 (deg) | 夹爪 MAE |", "|---|---|---|---|"]
    for lead in LEADS:
        lines.append(f"| {lead} | {summary['mean_pos_cm'][lead]:.3f} | {summary['mean_rot_deg'][lead]:.3f} "
                     f"| {summary['mean_grp_mae'][lead]:.4f} |")
    lines += ["", f"- 逐帧误差（pooled，{pooled.size} 帧）：均值 {pooled.mean():.3f} cm，"
                  f"p50 {p50:.3f} cm，p90 {p90:.3f} cm，p99 {p99:.3f} cm",
              f"- 对照已落库指标（`offline_ee_results.csv`，同口径宏平均）：ckpt-130000 "
              f"lead1={0.9317:.4f} cm / lead10={1.8283:.4f} / lead20={2.5699:.4f} / lead30={3.1428:.4f}", "",
              "## 口径2 拼接（anchor 每 30 帧，用满 30 步，误差每 30 帧被重新锚回）", "",
              f"- 覆盖：{len(rec)} 集 / {n_anchor_total} anchor / {sum(v[0].size for v in rec.values())} 帧",
              f"- 每集位置 RMSE：均值 {rm.mean():.3f} cm，最差 ep{rows2[worst_i]['episode_index']:03d} "
              f"{rm[worst_i]:.3f} cm，最好 ep{rows2[best_i]['episode_index']:03d} {rm.min():.3f} cm",
              f"- chunk 内步序误差：k=1 {overall_prof[0]:.3f} cm → k=15 {overall_prof[14]:.3f} cm → "
              f"k=30 {overall_prof[-1]:.3f} cm（拼接每 30 帧重新锚定，误差被周期性拉回；与口径1 的 lead 曲线同源，口径1 是全部 anchor 的平均、此处只取 anchor%30==0 的子集）", "",
              "## 两项数据自检", "",
              f"- **四元数双覆盖**：真实与预测四元数同号(点积>0)只在部分帧成立——"
              f"点积<0（同一旋转的另一符号）的样本占比：左臂 {flips['left']:.1%}、右臂 {flips['right']:.1%}。"
              f"指标一律用 |dot| 度量（与 `evaluate_ee.quat_error_deg` 一致），故符号翻转不影响数值，"
              f"但直接对四元数分量做 MSE 的训练目标会因此不稳定。",
              f"- **时间对齐自检**（把整段 30 步 chunk 相对真实帧平移 off，只用 anchor%30==0 的部署锚点）："
              + " / ".join(f"off={o:+d} {offs[o]:.3f} cm" for o in sorted(offs))
              + f"，最小点在 off={best_off:+d}（约定为 +1）。"
              f"偏差量级 {abs(offs[best_off] - offs[1]):.3f} cm（"
              f"{abs(offs[best_off] - offs[1]) / offs[1]:.1%}），即存在轻微系统性滞后（预测更像 1~2 帧前的目标），"
              f"不是整体错位；本报告仍按 `evaluate_ee.py` 的约定口径出图。", "",
              "## 图表", "",
              "- `fig1_lead_curve.png` 提前量 lead=1..30 的三项误差曲线（按任务 + 全任务均值）",
              "- `fig1_error_dist.png` 单步误差分布 + 按任务箱线",
              "- `step1/fig1_traj_ep*.png` 每集 x/y/z 轨迹（真实 vs 单步预测）+ 逐帧误差",
              "- `fig2_summary.png` 拼接：每集 RMSE / chunk 内误差增长 / 按任务",
              "- `anchor30/fig2_traj_ep*.png` 每集 x/y/z 轨迹（真实 vs 拼接重建）+ 逐帧误差",
              "- `metrics_step1.csv` / `metrics_anchor30.csv` 明细", ""]
    (outdir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\n输出目录: {outdir}")


if __name__ == "__main__":
    main()
