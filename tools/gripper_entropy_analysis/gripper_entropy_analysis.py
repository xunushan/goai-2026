#!/usr/bin/env python3
"""gripper(夹持器) target 二元熵分析 (最终口径, 用户确认)。

定义(用户提供的代码语义, 对应 pi05/X-VLA 20D action 的 gripper 通道):
    target = action[..., [l_g, r_g]].clamp(1e-6, 1 - 1e-6)
    H = -( target * log(target) + (1 - target) * log(1 - target) )

  - 本 CSV action 为 16D, gripper 在 l_g=idx7 / r_g=idx15;
    与 20D 布局(l_g=idx9 / r_g=idx19)数值一致(见 scripts/convert_lerobot_ee16_to_xvla20.py),
    且二元熵对 p<->1-p 对称, 不受 gripper 方向翻转影响。
  - log = torch.log = 自然对数, 单位 nat; H ∈ [0, ln2≈0.693], p=0.5 时最大。

对每条 episode 逐帧算 H(L)/H(R); 不做时间归一化, x 轴 = 原始 frame_index。
同任务 100 条 episode: 逐条浅线 + 均值(±σ)带。

产物(outputs/gripper_entropy_action_binary_<date>/):
  entropy_frame_task_{0..2}.png   每任务 L/R 两面板 逐帧曲线+均值带
  histogram_entropy_{all,task}.png 熵直方图
  entropy_mean_series.csv         每任务每帧 均值/sd/n
  entropy_stats_summary.csv       每任务每臂统计
  entropy_LR_test.csv             L/R 配对(Wilcoxon, 每episode均值)
  report_entropy.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from datetime import date

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

CSV = ROOT / "data" / "sim_lerobot_v30_ee" / "sim_lerobot_v30_ee.csv"
SPLIT_JSON = ROOT / "data" / "sim_lerobot_v30_ee" / "train_val_split.json"
OUT = ROOT / "outputs" / f"gripper_entropy_action_binary_{date.today().isoformat()}"

A_G_L, A_G_R = 7, 15     # 16D action 中 gripper 下标
CLAMP = (1e-6, 1.0 - 1e-6)
ARMS = {"L": A_G_L, "R": A_G_R}
C_L, C_R = "#1f77b4", "#ff7f0e"
MIN_N_BAND = 10
H_MAX = float(np.log(2))          # ln2 ≈ 0.693, 二元熵上界


def log(msg): print(f"[{date.today().isoformat()}] {msg}", flush=True)


def H_of_p(p: np.ndarray) -> np.ndarray:
    """二元熵 H = -p ln p - (1-p) ln(1-p), p 先 clamp 到 (1e-6,1-1e-6)。"""
    p = np.clip(np.asarray(p, float), *CLAMP)
    return -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))


def load_action_grippers() -> pd.DataFrame:
    """读 CSV 的 action 列, 每 episode 一行 {task_index, episode_index, length, grip_L, grip_R}
    (grip = action 夹持器目标序列, L=idx7 / R=idx15)。"""
    cache = OUT / "_episodes_cache.pkl"
    if cache.exists():
        log(f"load from cache {cache}")
        return pd.read_pickle(cache)
    df = pd.read_csv(CSV)
    cols_L, cols_R = [], []
    for s in df["action"]:
        tok = s.split(",")
        cols_L.append(float(tok[A_G_L]))
        cols_R.append(float(tok[A_G_R].rstrip("]")))
    gl = np.asarray(cols_L); gr = np.asarray(cols_R)
    rows = []
    for (t, e), g in df.groupby(["task_index", "episode_index"]):
        rows.append({
            "task_index": int(t), "episode_index": int(e),
            "length": int(g["length"].iloc[0]),
            "grip_L": gl[g.index].astype(float),
            "grip_R": gr[g.index].astype(float),
        })
    out = pd.DataFrame(rows)
    out.to_pickle(cache)
    log(f"loaded {len(out)} episodes, tasks={sorted(out.task_index.unique())}")
    return out


def task_instructions() -> dict[int, str]:
    d = json.load(open(SPLIT_JSON))
    return {int(t): info["instruction"] for t, info in d["tasks"].items()}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    # CJK 字体 (含 axis 用 ASCII 连字符避免缺字)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import stats as sps
    import sys as _sys
    _sys.path.insert(0, str(ROOT / "tools" / "gripper_val_split"))
    from gripper_common import setup_cjk_font
    setup_cjk_font()

    eps = load_action_grippers()
    instr = task_instructions()
    tasks = sorted(int(x) for x in eps["task_index"].unique())

    # 每 (task, arm): list[np.ndarray] 熵序列
    Hs = {}
    for t in tasks:
        te = eps[eps["task_index"] == t]
        for arm, col in ARMS.items():
            Hs[(t, arm)] = [H_of_p(g) for g in te["grip_" + arm]]

    # ---------- 1) 逐条曲线 + 均值带 ----------
    for t in tasks:
        fig, axes = plt.subplots(1, 2, figsize=(15, 4.6), sharey=False)
        for ax, arm, c in zip(axes, ["L", "R"], [C_L, C_R]):
            seqs = Hs[(t, arm)]
            for s in seqs:
                ax.plot(np.arange(len(s)), s, color=c, alpha=0.07, lw=0.7)
            maxL = max(len(s) for s in seqs)
            cols = np.full((len(seqs), maxL), np.nan)
            for i, s in enumerate(seqs):
                cols[i, :len(s)] = s
            with np.errstate(all="ignore"):
                mean = np.nanmean(cols, axis=0); sd = np.nanstd(cols, axis=0)
                valid = (~np.isnan(cols)).sum(axis=0)
            x = np.arange(maxL)
            ax.fill_between(x, mean - sd, mean + sd, color=c, alpha=0.18)
            ax.plot(x[valid >= MIN_N_BAND], mean[valid >= MIN_N_BAND], color=c, lw=2.0)
            ax.plot(x[valid < MIN_N_BAND], mean[valid < MIN_N_BAND], color=c, lw=1.4,
                    ls="--", alpha=0.7)
            ax.set_xlim(0, maxL); ax.set_ylim(-0.02, H_MAX * 1.05)
            ax.set_xlabel("frame_index (原始)")
            ax.set_ylabel("H = -p ln p - (1-p) ln(1-p)  (nat)")
            ax.set_title(f"task {t} · {arm}臂 (action gripper, 100条episode, 均值±σ深色带)")
            ax.grid(alpha=0.3)
        fig.suptitle(f"task {t}: {instr[t]}\n"
                     f"action gripper 目标 p 的二元熵;  p∈(0,1) clamp 后 H(p)∈[0,{H_MAX:.3f}], p=0.5 最大",
                     fontsize=9.5)
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        fig.savefig(OUT / f"entropy_frame_task_{t}.png", dpi=140)
        plt.close(fig)

    # 均值序列 CSV
    rows = []
    for t in tasks:
        for arm in ["L", "R"]:
            seqs = Hs[(t, arm)]
            maxL = max(len(s) for s in seqs)
            cols = np.full((len(seqs), maxL), np.nan)
            for i, s in enumerate(seqs):
                cols[i, :len(s)] = s
            with np.errstate(all="ignore"):
                m = np.nanmean(cols, axis=0); sd = np.nanstd(cols, axis=0)
                n = (~np.isnan(cols)).sum(axis=0)
            for k in range(maxL):
                rows.append({"task_index": t, "arm": arm, "frame": k,
                             "mean": round(float(m[k]), 5), "sd": round(float(sd[k]), 5),
                             "n_ep": int(n[k])})
    pd.DataFrame(rows).to_csv(OUT / "entropy_mean_series.csv", index=False, encoding="utf-8-sig")

    # ---------- 2) 直方图 ----------
    allH = {arm: np.concatenate([h for t in tasks for h in Hs[(t, arm)]]) for arm in ["L", "R"]}
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ax = axes[0, 0]
    ax.hist(allH["L"], bins=60, alpha=0.6, color=C_L, label="L臂", density=True)
    ax.hist(allH["R"], bins=60, alpha=0.6, color=C_R, label="R臂", density=True)
    ax.set_title("全部任务 (每episode每帧一个 H)"); ax.legend(); ax.grid(alpha=0.3)
    ax.set_xlabel("H (nat)"); ax.set_ylabel("密度")
    for j, t in enumerate(tasks):
        ax = [axes[0, 1], axes[1, 0], axes[1, 1]][j]
        ax.hist(np.concatenate(Hs[(t, "L")]), bins=50, alpha=0.6, color=C_L, label="L臂", density=True)
        ax.hist(np.concatenate(Hs[(t, "R")]), bins=50, alpha=0.6, color=C_R, label="R臂", density=True)
        ax.set_title(f"task {t}: {instr[t][:30]}...")
        ax.legend(); ax.grid(alpha=0.3); ax.set_xlabel("H (nat)"); ax.set_ylabel("密度")
    fig.suptitle("action gripper 目标二元熵直方图 (H = -p ln p - (1-p) ln(1-p))", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / "histogram_entropy_all.png", dpi=140)
    plt.close(fig)
    for t in tasks:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(np.concatenate(Hs[(t, "L")]), bins=60, alpha=0.6, color=C_L, label="L臂", density=True)
        ax.hist(np.concatenate(Hs[(t, "R")]), bins=60, alpha=0.6, color=C_R, label="R臂", density=True)
        ax.set_title(f"task {t} 二元熵直方图 (L vs R)")
        ax.set_xlabel("H (nat)"); ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT / f"histogram_entropy_task_{t}.png", dpi=140)
        plt.close(fig)

    # ---------- 3) 统计 ----------
    rows = []
    for t in tasks:
        for arm in ["L", "R"]:
            h_all = np.concatenate(Hs[(t, arm)])
            ep_mean = np.array([s.mean() for s in Hs[(t, arm)]])
            rows.append({
                "task_index": t, "arm": arm,
                "n_frames": int(len(h_all)), "n_ep": len(Hs[(t, arm)]),
                "frame_mean": round(float(h_all.mean()), 5),
                "frame_std": round(float(h_all.std()), 5),
                "frame_median": round(float(np.median(h_all)), 5),
                "frame_q25": round(float(np.percentile(h_all, 25)), 5),
                "frame_q75": round(float(np.percentile(h_all, 75)), 5),
                "frame_max": round(float(h_all.max()), 5),
                "ep_mean_avg": round(float(ep_mean.mean()), 5),
                "ep_mean_std": round(float(ep_mean.std()), 5),
            })
    stats_df = pd.DataFrame(rows)
    stats_df.to_csv(OUT / "entropy_stats_summary.csv", index=False, encoding="utf-8-sig")

    test_rows = []
    for t in tasks:
        eL = np.array([s.mean() for s in Hs[(t, "L")]])
        eR = np.array([s.mean() for s in Hs[(t, "R")]])
        try:
            _, p = sps.wilcoxon(eL - eR)
        except ValueError:
            p = np.nan
        test_rows.append({
            "task_index": t,
            "ep_mean_L": round(float(eL.mean()), 5),
            "ep_mean_R": round(float(eR.mean()), 5),
            "diff(L-R)": round(float((eL - eR).mean()), 5),
            "wilcoxon_p": (None if p != p else round(float(p), 4)),
            "interpret": ("L臂显著更高" if eL.mean() > eR.mean() and p == p and p < 0.05 else
                          "R臂显著更高" if eR.mean() > eL.mean() and p == p and p < 0.05 else
                          "两臂差异不显著"),
        })
    pd.DataFrame(test_rows).to_csv(OUT / "entropy_LR_test.csv", index=False, encoding="utf-8-sig")

    # ---------- 4) 报告 ----------
    L = [
        "# action gripper 目标二元熵报告 (H = -p·ln p - (1-p)·ln(1-p))",
        "",
        f"- 日期: {date.today().isoformat()}",
        f"- 数据: `{CSV.name}` ({len(eps)} episodes, tasks {tasks})",
        "- 指标(按你提供代码语义): 每帧取 action 夹持器目标 p, clamp(1e-6, 1-1e-6) 后",
        "  H(p) = -p·ln(p) - (1-p)·ln(1-p), 自然对数(nat), H∈[0, ln2≈0.693], p=0.5 最大。",
        "- 本 CSV action 16D, gripper 下标 L=7 / R=15; 与 20D 布局(L=9/R=19)数值一致",
        "  (见 `scripts/convert_lerobot_ee16_to_xvla20.py`), 二元熵对 p↔(1-p) 对称。",
        "- 对每条 episode 逐帧计算; 不做跨episode分布/对齐/归一化, x 轴为原始 frame_index;",
        "  同任务 100 条逐条浅线 + 均值±σ 带。",
        "",
        "> 语义提示: H 越大 = 该帧夹持器命令越接近『半开半闭/不确定』(p≈0.5);",
        "> H≈0 = 夹持器命令贴向一端(全开 p→1 或全闭 p→0, 动作确定)。",
        "",
        "## 任务与动作语义",
        "",
        "| task | 指令 |", "|---|---|",
    ]
    for t in tasks:
        L.append(f"| {t} | {instr[t]} |")
    L += ["", "## 1. 统计量 (帧级样本)", "",
          "| task | arm | n_frames | frame_mean | frame_std | frame_median | [q25,q75] | max | ep均值avg | ep均值std |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in stats_df.iterrows():
        L.append(f"| {r['task_index']} | {r['arm']} | {r['n_frames']} | {r['frame_mean']:.4f} | "
                 f"{r['frame_std']:.4f} | {r['frame_median']:.4f} | "
                 f"[{r['frame_q25']:.4f},{r['frame_q75']:.4f}] | {r['frame_max']:.4f} | "
                 f"{r['ep_mean_avg']:.4f} | {r['ep_mean_std']:.4f} |")
    L += ["", "## 2. L vs R (每episode均值成对 Wilcoxon)", "",
          "| task | ep_mean L | ep_mean R | diff | p | 结论 |", "|---|---|---|---|---|---|"]
    for _, r in pd.DataFrame(test_rows).iterrows():
        pstr = "-" if r["wilcoxon_p"] is None else f"{r['wilcoxon_p']:.4f}"
        L.append(f"| {r['task_index']} | {r['ep_mean_L']:.4f} | {r['ep_mean_R']:.4f} | "
                 f"{r['diff(L-R)']:+.4f} | {pstr} | {r['interpret']} |")
    L += ["", "## 3. 说明", "",
          f"- 均值带实线要求该帧样本 ≥ {MIN_N_BAND} 条 episode, 之后为虚线(尾部仅少数长轨迹)。",
          ""]
    (OUT / "report_entropy.md").write_text("\n".join(L), encoding="utf-8")
    log(f"done -> {OUT}")


if __name__ == "__main__":
    main()
