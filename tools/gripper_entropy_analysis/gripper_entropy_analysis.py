#!/usr/bin/env python3
"""gripper(夹持器) entropy 分析 v2: 逐条轨迹逐帧 H = -g * ln(g)。

定义(用户 2026-09-04 明确): 对每条 episode, 每帧用该帧夹持器值 g∈[0,1](0闭~1开),
  H(g) = -g * ln(g);   g->0+ 时极限为 0, g=1(全开) 时为 0, g≈0.368 时最大 ≈ 0.531 nat。
不跨 episode、不做时间归一化/对齐; x 轴为该条 episode 原始 frame_index。
任务内聚合: 100 条逐条浅线 + 均值(±σ)带(在 frame 轴上按该帧存在的 episode 计算)。

产物(outputs/gripper_entropy_pointwise_<date>/):
  1. entropy_frame_task_{0..2}.png      每任务 L/R 两面板 100 条逐帧曲线+均值带
  2. histogram_entropy_{all,task}.png   熵直方图(L/R 对比)
  3. entropy_mean_series.csv            每任务每帧 均值/sd/n (L、R)
  4. entropy_stats_summary.csv          每任务每臂统计量 + 每episode均值统计
  5. entropy_LR_test.csv                L/R 配对(Wilcoxon, 以每episode均值配对)
  6. report_entropy.md                  报告
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from datetime import date

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "gripper_val_split"))

from gripper_common import load_grippers, setup_cjk_font  # noqa: E402

CSV = ROOT / "data" / "sim_lerobot_v30_ee" / "sim_lerobot_v30_ee.csv"
SPLIT_JSON = ROOT / "data" / "sim_lerobot_v30_ee" / "train_val_split.json"
OUT = ROOT / "outputs" / f"gripper_entropy_pointwise_{date.today().isoformat()}"

EPSILON = 1e-9
ARMS = {"L": "grip_L", "R": "grip_R"}
C_L, C_R = "#1f77b4", "#ff7f0e"
MIN_N_BAND = 10   # 均值带在样本数少于该值的帧位淡出


def log(msg): print(f"[{date.today().isoformat()}] {msg}", flush=True)


def H_of_g(g: np.ndarray) -> np.ndarray:
    """H = -g*ln(g), g∈[0,1] 截断; g<=0 按极限取 0。"""
    g = np.clip(np.asarray(g, float), 0.0, 1.0)
    out = np.zeros_like(g)
    m = g > EPSILON
    out[m] = -g[m] * np.log(g[m])
    return out


def load_episodes() -> pd.DataFrame:
    cache = OUT / "_episodes_cache.pkl"
    if cache.exists():
        log(f"load from cache {cache}")
        return pd.read_pickle(cache)
    df = load_grippers(CSV)
    df.to_pickle(cache)
    log(f"loaded {len(df)} episodes, tasks={sorted(int(x) for x in df.task_index.unique())}")
    return df


def task_instructions() -> dict[int, str]:
    d = json.load(open(SPLIT_JSON))
    return {int(t): info["instruction"] for t, info in d["tasks"].items()}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    setup_cjk_font()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import stats as sps

    eps = load_episodes()
    instr = task_instructions()
    tasks = sorted(int(x) for x in eps["task_index"].unique())

    # 每 (task, arm): 每 episode 一序列 H, 长度=帧数
    Hs = {}     # (task, arm) -> list[np.ndarray]
    Gs = {}     # (task, arm) -> list[np.ndarray] 原始 g (供解读/极值统计)
    for t in tasks:
        te = eps[eps["task_index"] == t]
        for arm, col in ARMS.items():
            seqs, gseqs = [], []
            for g in te[col]:
                g = np.asarray(g, float)
                gseqs.append(g)
                seqs.append(H_of_g(g))
            Hs[(t, arm)] = seqs
            Gs[(t, arm)] = gseqs

    # ---------- 1) 逐条曲线 + 均值带 ----------
    for t in tasks:
        fig, axes = plt.subplots(1, 2, figsize=(15, 4.6), sharey=False)
        for ax, arm, c in zip(axes, ARMS, [C_L, C_R]):
            seqs = Hs[(t, arm)]
            # 逐条浅线(各自原始 frame_index 0..len-1)
            for s in seqs:
                ax.plot(np.arange(len(s)), s, color=c, alpha=0.07, lw=0.7)
            # 均值/σ 带: 每帧取该帧存在的 episode (mask 矩阵)
            maxL = max(len(s) for s in seqs)
            cols = np.full((len(seqs), maxL), np.nan)
            for i, s in enumerate(seqs):
                cols[i, :len(s)] = s
            with np.errstate(all="ignore"):
                mean = np.nanmean(cols, axis=0)
                sd = np.nanstd(cols, axis=0)
                valid = (~np.isnan(cols)).sum(axis=0)
            x = np.arange(maxL)
            ax.fill_between(x, mean - sd, mean + sd, color=c, alpha=0.18)
            # 样本少的尾段降透明度提示
            ax.plot(x[valid >= MIN_N_BAND], mean[valid >= MIN_N_BAND], color=c, lw=2.0)
            ax.plot(x[valid < MIN_N_BAND], mean[valid < MIN_N_BAND], color=c, lw=1.4,
                    ls="--", alpha=0.7)
            ax.set_xlim(0, maxL)
            ax.set_ylim(-0.02, 0.56)
            ax.set_xlabel("frame_index (原始)")
            ax.set_ylabel("H = -g·ln(g)  (nat)")
            ax.set_title(f"task {t} · {arm}臂 (100条episode, 均值±σ深色带)")
            ax.grid(alpha=0.3)
        fig.suptitle(f"task {t}: {instr[t]}    H(g)=-g·ln(g), g=夹持器值[0闭~1开], g=1时H=0, g≈0.37最大",
                     fontsize=9.5)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        fig.savefig(OUT / f"entropy_frame_task_{t}.png", dpi=140)
        plt.close(fig)

    # 均值序列 CSV
    rows = []
    for t in tasks:
        maxL = max(len(s) for arm in ARMS for s in Hs[(t, arm)])
        for arm in ARMS:
            cols = np.full((len(Hs[(t, arm)]), maxL), np.nan)
            for i, s in enumerate(Hs[(t, arm)]):
                cols[i, :len(s)] = s
            with np.errstate(all="ignore"):
                m = np.nanmean(cols, axis=0); sd = np.nanstd(cols, axis=0)
                n = (~np.isnan(cols)).sum(axis=0)
            for k in range(maxL):
                rows.append({"task_index": t, "arm": arm, "frame": k,
                             "mean": round(float(m[k]), 5) if m[k] == m[k] else "",
                             "sd": round(float(sd[k]), 5) if sd[k] == sd[k] else "",
                             "n_ep": int(n[k])})
    pd.DataFrame(rows).to_csv(OUT / "entropy_mean_series.csv", index=False, encoding="utf-8-sig")

    # ---------- 2) 直方图 ----------
    # 每任务每臂: 把所有(episode,frames)的 H 拼起来
    allH = {arm: np.concatenate([h for t in tasks for h in Hs[(t, arm)]]) for arm in ARMS}
    # 全量 + 分任务 2x2
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ax = axes[0, 0]
    ax.hist(allH["L"], bins=60, alpha=0.6, color=C_L, label="L臂", density=True)
    ax.hist(allH["R"], bins=60, alpha=0.6, color=C_R, label="R臂", density=True)
    ax.set_title("全部任务 (每episode每帧一个H)"); ax.legend(); ax.grid(alpha=0.3)
    ax.set_xlabel("H = -g·ln(g)"); ax.set_ylabel("密度")
    sub_axes = [axes[0, 1], axes[1, 0], axes[1, 1]]
    for j, t in enumerate(tasks):
        ax = sub_axes[j]
        ax.hist(np.concatenate(Hs[(t, "L")]), bins=50, alpha=0.6,
                color=C_L, label="L臂", density=True)
        ax.hist(np.concatenate(Hs[(t, "R")]), bins=50, alpha=0.6,
                color=C_R, label="R臂", density=True)
        ax.set_title(f"task {t}: {instr[t][:30]}...")
        ax.legend(); ax.grid(alpha=0.3)
        ax.set_xlabel("H"); ax.set_ylabel("密度")
    fig.suptitle("gripper 逐帧熵 H(g)=-g·ln(g) 直方图", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / "histogram_entropy_all.png", dpi=140)
    plt.close(fig)
    for t in tasks:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(np.concatenate(Hs[(t, "L")]), bins=60, alpha=0.6, color=C_L, label="L臂", density=True)
        ax.hist(np.concatenate(Hs[(t, "R")]), bins=60, alpha=0.6, color=C_R, label="R臂", density=True)
        ax.set_title(f"task {t} 逐帧熵直方图 (L vs R)")
        ax.set_xlabel("H = -g·ln(g)"); ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT / f"histogram_entropy_task_{t}.png", dpi=140)
        plt.close(fig)

    # ---------- 3) 统计 ----------
    # 帧级统计 (以所有帧为样本) + episode级均值(每episode一个标量, 等权)
    rows = []
    for t in tasks:
        for arm in ARMS:
            h_all = np.concatenate(Hs[(t, arm)])          # 帧级样本
            ep_mean = np.array([s.mean() for s in Hs[(t, arm)]])   # 每episode均值
            ep_med = np.array([np.median(s) for s in Hs[(t, arm)]])
            g_all = np.concatenate(Gs[(t, arm)])
            rows.append({
                "task_index": t, "arm": arm,
                "n_frames": int(len(h_all)), "n_ep": len(Hs[(t, arm)]),
                "frame_mean": round(float(h_all.mean()), 5),
                "frame_std": round(float(h_all.std()), 5),
                "frame_median": round(float(np.median(h_all)), 5),
                "frame_q25": round(float(np.percentile(h_all, 25)), 5),
                "frame_q75": round(float(np.percentile(h_all, 75)), 5),
                "frame_p90": round(float(np.percentile(h_all, 90)), 5),
                "frame_max": round(float(h_all.max()), 5),
                "ep_mean_avg": round(float(ep_mean.mean()), 5),
                "ep_mean_std": round(float(ep_mean.std()), 5),
                "ep_median_med": round(float(np.median(ep_med)), 5),
                "pct_g_lt0.1": round(float((g_all < 0.1).mean()), 4),
                "pct_g_gt0.9": round(float((g_all > 0.9).mean()), 4),
            })
    stats_df = pd.DataFrame(rows)
    stats_df.to_csv(OUT / "entropy_stats_summary.csv", index=False, encoding="utf-8-sig")

    # L vs R 配对(每episode均值, 同index匹配)
    test_rows = []
    for t in tasks:
        eL = np.array([s.mean() for s in Hs[(t, "L")]])
        eR = np.array([s.mean() for s in Hs[(t, "R")]])
        try:
            w, p = sps.wilcoxon(eL - eR)
        except ValueError:
            w, p = np.nan, np.nan
        test_rows.append({
            "task_index": t,
            "ep_mean_L": round(float(eL.mean()), 5),
            "ep_mean_R": round(float(eR.mean()), 5),
            "diff(L-R)": round(float((eL - eR).mean()), 5),
            "wilcoxon_p": (None if p != p else round(float(p), 4)),
            "interpret": ("L臂显著更高" if (eL.mean() > eR.mean()) and (p == p) and p < 0.05 else
                          "R臂显著更高" if (eR.mean() > eL.mean()) and (p == p) and p < 0.05 else
                          "两臂差异不显著"),
        })
    pd.DataFrame(test_rows).to_csv(OUT / "entropy_LR_test.csv", index=False, encoding="utf-8-sig")

    # ---------- 4) 报告 ----------
    lnmax = np.exp(-1)  # 0.368
    L = [
        "# Gripper 逐帧熵报告 (H = -g·ln g)",
        "",
        f"- 日期: {date.today().isoformat()}",
        f"- 数据: `{CSV.name}` ({len(eps)} episodes, tasks {tasks})",
        "- 定义: 对每条 episode 每一帧, 取该帧夹持器值 g∈[0,1] (0闭~1开), 计算",
        "  `H(g) = -g·ln(g)` (自然对数, 单位 nat)。g→0⁺ 按极限取 0; g=1(全开) 时 H=0;",
        f"  g={lnmax:.3f} 时最大 ≈ 0.531 nat。",
        "- 不做跨 episode 分布/对齐/时间归一化; 逐条轨迹画, 任务内取均值±σ 带。",
        "",
        "> 语义提示: H 在夹持器**全开或全闭**(极端态)为 0, 在中间开度(≈0.37)最大。",
        "> 因此该曲线刻画夹持器离开两极端、处于『半开合/中间位』的程度随时间的变化,",
        "> 峰值高 = 夹持动作较多、开度常在中间态; 恒 0 = 全程贴死一端(如一直张开)。",
        "",
        "## 任务与动作语义",
        "",
        "| task | 指令 |",
        "|---|---|",
    ]
    for t in tasks:
        L.append(f"| {t} | {instr[t]} |")
    L += ["", "## 1. 统计量 (帧级样本: 该任务该臂所有 episode×帧 的 H)", "",
          "| task | arm | n_frames | frame_mean | frame_std | frame_median | [q25,q75] | p90 | max | ep均值avg | ep均值std |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in stats_df.iterrows():
        L.append(f"| {r['task_index']} | {r['arm']} | {r['n_frames']} | {r['frame_mean']:.4f} | "
                 f"{r['frame_std']:.4f} | {r['frame_median']:.4f} | "
                 f"[{r['frame_q25']:.4f},{r['frame_q75']:.4f}] | {r['frame_p90']:.4f} | "
                 f"{r['frame_max']:.4f} | {r['ep_mean_avg']:.4f} | {r['ep_mean_std']:.4f} |")
    L += ["", "## 2. L vs R (按每episode均值成对 Wilcoxon)", "",
          "| task | ep_mean L | ep_mean R | diff | p | 结论 |", "|---|---|---|---|---|---|"]
    for _, r in pd.DataFrame(test_rows).iterrows():
        pstr = "—" if r["wilcoxon_p"] is None else f"{r['wilcoxon_p']:.4f}"
        L.append(f"| {r['task_index']} | {r['ep_mean_L']:.4f} | {r['ep_mean_R']:.4f} | "
                 f"{r['diff(L-R)']:+.4f} | {pstr} | {r['interpret']} |")
    L += ["", "## 3. 说明", "",
          f"- 均值带在样本数 ≥ {MIN_N_BAND} 条 episode 处画实线, 之后虚线(尾部只有少数长 episode, 参考意义弱)。",
          "- g 极值占比: 全程贴死开(>0.9)占比高 → 该臂夹持器大多时间张开(如仅握持的辅助手)。",
          ""]
    (OUT / "report_entropy.md").write_text("\n".join(L), encoding="utf-8")
    log(f"done -> {OUT}")


if __name__ == "__main__":
    main()
