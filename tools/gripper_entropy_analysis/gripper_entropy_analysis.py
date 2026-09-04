#!/usr/bin/env python3
"""gripper(夹持器) entropy 分析: 分任务按归一化帧位计算跨 episode 分布熵。

方法(与用户确认):
  - 熵的对象: 同一任务在某个归一化帧位上, 100 条 episode 夹持器取值的分布
    → 该帧上各演示夹持器策略的分歧/一致性。
  - 帧对齐: 每 episode 线性插值到 [0,100] 共 101 点 (沿用 interp_100 约定)。
  - 分箱: 夹持器值(归一化, 0闭~1开)固定 K=10 个等宽 bin, 计算 Shannon 熵(bit)。

产物(写入 outputs/gripper_entropy_<date>/):
  1. entropy_vs_frame_task_{0..2}.png   每任务 L/R 两面板 熵随归一化帧位曲线
  2. histogram_entropy_{all,per_task}.png 夹持器熵直方图(全量+分任务, L/R 对比)
  3. entropy_series.csv                 每任务每帧位 H_L / H_R
  4. entropy_stats_summary.csv          每任务每臂统计量 + L/R 成对检验
  5. report_entropy.md                  文字小结
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

from gripper_common import load_grippers, interp_100, setup_cjk_font  # noqa: E402

CSV = ROOT / "data" / "sim_lerobot_v30_ee" / "sim_lerobot_v30_ee.csv"
SPLIT_JSON = ROOT / "data" / "sim_lerobot_v30_ee" / "train_val_split.json"
OUT = ROOT / "outputs" / f"gripper_entropy_{date.today().isoformat()}"

GRID_N = 101          # 归一化帧位点数 (0..100)
K_BINS = 10           # 等宽分箱数
GRIP_RANGE = (0.0, 1.0)  # 夹持器物理范围 0闭 ~ 1开, 越界值截断
EPS = 1e-12

ARMS = {"L": 7, "R": 15}   # observation.state 中左/右 gripper 下标

# 绘图配色沿用 gripper_common: 左蓝 / 右橙
C_L, C_R = "#1f77b4", "#ff7f0e"


def log(msg: str) -> None:
    print(f"[{date.today().isoformat()}] {msg}", flush=True)


def load_episodes() -> pd.DataFrame:
    """读 CSV -> 每 episode 一行 {task_index, episode_index, length, grip_L, grip_R}。"""
    cache = OUT / "_episodes_cache.pkl"
    if cache.exists():
        log(f"load episodes from cache {cache}")
        return pd.read_pickle(cache)
    df = load_grippers(CSV)
    df.to_pickle(cache)
    log(f"episodes loaded & cached: {len(df)} episodes, tasks={sorted(df.task_index.unique())}")
    return df


def interp_grip_series(g: np.ndarray) -> np.ndarray:
    """单条夹持器序列 -> [0,100] 共 GRID_N 个点 (保留端点)。"""
    return interp_100(np.asarray(g, float), n=GRID_N)


def step_entropy(vals: np.ndarray, k: int = K_BINS, rng: tuple = GRIP_RANGE) -> float:
    """把 vals(该帧位所有 episode 的夹持器值) 等宽分箱 -> Shannon 熵(bit)。"""
    v = np.clip(np.asarray(vals, float), rng[0], rng[1])
    edges = np.linspace(rng[0], rng[1], k + 1)
    idx = np.searchsorted(edges, v, side="right") - 1
    idx = np.clip(idx, 0, k - 1)
    counts = np.bincount(idx, minlength=k).astype(float)
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def task_instructions() -> dict[int, str]:
    d = json.load(open(SPLIT_JSON))
    return {int(t): info["instruction"] for t, info in d["tasks"].items()}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    setup_cjk_font()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    eps = load_episodes()
    instr = task_instructions()
    tasks = sorted(eps["task_index"].unique())

    series_rows = []
    entropy = {}   # (task, arm) -> np.ndarray[GRID_N]
    for t in tasks:
        te = eps[eps["task_index"] == t]
        log(f"task {t}: {len(te)} episodes, instruction: {instr[t][:60]}...")
        for arm, _idx in ARMS.items():
            mat = np.stack(te[f"grip_{arm}"].map(interp_grip_series).to_numpy())  # (n_ep, GRID_N)
            H = np.array([step_entropy(mat[:, s]) for s in range(GRID_N)])
            entropy[(t, arm)] = H
            for s in range(GRID_N):
                series_rows.append({"task_index": int(t), "step": s,
                                    f"H_{arm}": round(float(H[s]), 5)})

    series_df = pd.DataFrame(series_rows)
    # 每 (task,step) 合并 L/R 成一行
    wide = series_df.set_index(["task_index", "step"])["H_L"].to_frame().join(
        series_df.set_index(["task_index", "step"])["H_R"])
    wide.to_csv(OUT / "entropy_series.csv", encoding="utf-8-sig")
    log(f"series saved: {OUT/'entropy_series.csv'}")

    # ---------------- 1) 熵随归一化帧位曲线 (每任务一图, 左右两面板) ----------------
    for t in tasks:
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), sharey=True)
        x = np.arange(GRID_N)
        for ax, arm, c in zip(axes, ["L", "R"], [C_L, C_R]):
            H = entropy[(t, arm)]
            ax.plot(x, H, color=c, lw=2)
            ax.fill_between(x, H, color=c, alpha=0.12)
            ax.set_xlim(0, GRID_N - 1)
            ax.set_ylim(0, np.log2(K_BINS))
            ax.set_xticks([0, 20, 40, 60, 80, 100])
            ax.set_xlabel("归一化帧位 (0–100, 每 episode 线性插值)")
            ax.set_ylabel("夹持器熵 (bit)")
            ax.set_title(f"task {t} · {arm}臂")
            ax.grid(alpha=0.3)
            pk = int(np.argmax(H))
            ax.annotate(f"峰值 {H[pk]:.2f} @帧位{pk}", xy=(pk, H[pk]),
                        xytext=(pk + 4, H[pk] + 0.12 * np.log2(K_BINS)),
                        fontsize=9, arrowprops=dict(arrowstyle="->", lw=0.8, color="0.4"))
        fig.suptitle(f"task {t}: {instr[t]}   (跨 100 episode 同帧位夹持器分布熵, K={K_BINS} bins)",
                     fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        fig.savefig(OUT / f"entropy_vs_frame_task_{t}.png", dpi=150)
        plt.close(fig)
    log("figs: entropy_vs_frame_task_{0..2}.png done")

    # ---------------- 2) 熵直方图 ----------------
    # (a) 全量 L vs R
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ha = np.concatenate([entropy[(t, "L")] for t in tasks])
    hb = np.concatenate([entropy[(t, "R")] for t in tasks])
    ax = axes[0, 0]
    ax.hist(ha, bins=40, alpha=0.55, color=C_L, label="L臂", density=True)
    ax.hist(hb, bins=40, alpha=0.55, color=C_R, label="R臂", density=True)
    ax.set_xlabel("熵 (bit)"); ax.set_ylabel("密度"); ax.set_title("全任务 每帧位熵 分布 (L vs R)")
    ax.legend(); ax.grid(alpha=0.3)
    for j, t in enumerate(tasks):
        ax = axes[1, j] if j < 2 else axes[0, 1]
        # 重排: 用 (0,1) 与 (1,0),(1,1) 放三个任务; 左上已被全量占用
        sub_axes = [axes[0, 1], axes[1, 0], axes[1, 1]]
        ax = sub_axes[j]
        ax.hist(entropy[(t, "L")], bins=30, alpha=0.55, color=C_L, label="L臂", density=True)
        ax.hist(entropy[(t, "R")], bins=30, alpha=0.55, color=C_R, label="R臂", density=True)
        ax.set_xlabel("熵 (bit)"); ax.set_ylabel("密度")
        ax.set_title(f"task {t}: {instr[t][:32]}...")
        ax.legend(); ax.grid(alpha=0.3)
    fig.suptitle(f"gripper entropy 直方图 (每归一化帧位一个熵值, K={K_BINS} bins)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / "histogram_entropy.png", dpi=150)
    plt.close(fig)

    # (b) 每任务单独一张 L/R 对比直方图
    for t in tasks:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(entropy[(t, "L")], bins=30, alpha=0.55, color=C_L, label="L臂", density=True)
        ax.hist(entropy[(t, "R")], bins=30, alpha=0.55, color=C_R, label="R臂", density=True)
        ax.set_xlabel("熵 (bit)"); ax.set_ylabel("密度")
        ax.set_title(f"task {t} gripper entropy 直方图 (L vs R)")
        ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT / f"histogram_entropy_task_{t}.png", dpi=150)
        plt.close(fig)
    log("figs: histogram done")

    # ---------------- 3) 统计量 ----------------
    stats_rows = []
    for t in tasks:
        for arm in ARMS:
            H = entropy[(t, arm)]
            Hq = np.percentile(H, [25, 75])
            stats_rows.append({
                "task_index": int(t),
                "arm": arm,
                "instruction": instr[t],
                "mean": round(H.mean(), 4), "std": round(H.std(), 4),
                "median": round(np.median(H), 4),
                "q25": round(Hq[0], 4), "q75": round(Hq[1], 4),
                "min": round(H.min(), 4), "max": round(H.max(), 4),
                "sum_bits": round(H.sum(), 4),
                "frac_gt_1bit": round(float((H > 1.0).mean()), 4),
                "frac_lt_0.1bit": round(float((H < 0.1).mean()), 4),
                "peak_step": int(np.argmax(H)), "peak_value": round(H.max(), 4),
            })
    stats_df = pd.DataFrame(stats_rows)
    # L vs R 成对检验(同帧位配对)
    from scipy import stats as sps
    test_rows = []
    for t in tasks:
        HL, HR = entropy[(t, "L")], entropy[(t, "R")]
        diff = HL - HR
        try:
            w, p = sps.wilcoxon(diff)
        except ValueError:
            w, p = np.nan, np.nan
        test_rows.append({
            "task_index": int(t),
            "mean(L-R)": round(float(diff.mean()), 4),
            "wilcoxon_p": round(float(p), 4) if p == p else None,
            "sum(L)-sum(R)": round(float(HL.sum() - HR.sum()), 4),
            "interpret": ("R臂显著更活跃/分歧" if diff.mean() < 0 and p < 0.05 else
                          "L臂显著更活跃/分歧" if diff.mean() > 0 and p < 0.05 else
                          "两臂差异不显著"),
        })
    test_df = pd.DataFrame(test_rows)
    stats_df.to_csv(OUT / "entropy_stats_summary.csv", index=False, encoding="utf-8-sig")
    test_df.to_csv(OUT / "entropy_LR_test.csv", index=False, encoding="utf-8-sig")

    # ---------------- 4) markdown 报告 ----------------
    lines = [
        "# Gripper Entropy 分析报告 (sim_lerobot_v30_ee)",
        "",
        f"- 日期: {date.today().isoformat()}",
        f"- 数据: `{CSV.name}` ({len(eps)} episodes, tasks {','.join(map(str, tasks))})",
        "- 熵定义: 同一任务在某**归一化帧位**(0–100, 每 episode 线性插值)上, "
        "跨 100 条 episode 的夹持器取值分布之 Shannon 熵(bit)。",
        f"- 分箱: 夹持器值(0闭~1开)固定 **K={K_BINS}** 个等宽 bin; 最大可能熵 log2({K_BINS})={np.log2(K_BINS):.2f} bit。",
        "",
        "> 熵高 = 该任务阶段各演示的夹持器策略分歧大(有人开有人关/状态分散); 熵≈0 = 全部演示夹持器同处一个状态(如全程张开或全程闭合)。",
        "",
        "## 任务与动作语义",
        "",
        "| task | 指令 | 预期夹持器行为 |",
        "|---|---|---|",
    ]
    role = {
        0: "一手持笔筒, 另一手逐支插笔并放回 → 插笔臂反复开合夹笔, 持筒臂基本只握持",
        1: "把充电器插头插入插线板 → 以插头抓手(近似单臂)闭合为主, 另一手可能只是辅助",
        2: "把三只碗叠起 → 双手交替夹取-叠放, 两臂都应出现阶段性开合",
    }
    for t in tasks:
        lines.append(f"| {t} | {instr[t]} | {role[t]} |")
    lines += [
        "",
        "## 1. 每帧位熵统计 (bit)",
        "",
        "| task | arm | mean | std | median | [q25,q75] | max@step | frac>1bit | frac<0.1bit | sum_bits |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r in stats_df.iterrows():
        lines.append(
            f"| {r['task_index']} | {r['arm']} | {r['mean']:.3f} | {r['std']:.3f} | {r['median']:.3f} "
            f"| [{r['q25']:.2f},{r['q75']:.2f}] | {r['max']:.2f}@{r['peak_step']} "
            f"| {r['frac_gt_1bit']:.2f} | {r['frac_lt_0.1bit']:.2f} | {r['sum_bits']:.1f} |")
    lines += ["", "## 2. 左右臂成对差异 (同帧位 Wilcoxon signed-rank)", ""]
    lines += ["| task | mean(L-R) | sum(L)-sum(R) | Wilcoxon p | 解释 |",
              "|---|---|---|---|---|"]
    for _, r in test_df.iterrows():
        lines.append(f"| {r['task_index']} | {r['mean(L-R)']:.4f} | {r['sum(L)-sum(R)']:.1f} | "
                     f"{'' if pd.isna(r['wilcoxon_p']) else r['wilcoxon_p']:.3f} | {r['interpret']} |")
    lines += ["", "## 3. 关键结论", ""]

    def _phase(s: int) -> str:
        return ("前期" if s < 33 else "中段" if s < 67 else "后段")

    # 主峰(全局 max)所在臂与帧位
    for t in tasks:
        HL, HR = entropy[(t, "L")], entropy[(t, "R")]
        active = "L" if HL.mean() > HR.mean() else "R"
        idle = "R" if active == "L" else "L"
        # 用 max 对应臂的峰值来定位主要动作峰
        main_arm = "L" if HL.max() >= HR.max() else "R"
        main_pk = int(np.argmax(HL)) if main_arm == "L" else int(np.argmax(HR))
        pk_other = int(np.argmax(HR)) if main_arm == "L" else int(np.argmax(HL))
        lines.append(
            f"- **task {t}** ({instr[t]}): 夹持器平均熵 L={HL.mean():.3f} bit vs R={HR.mean():.3f} bit,"
            f" 整体 **{active} 臂更活跃/分歧更大**(夹持策略在演示间更不一致)。"
            f" 峰值: {main_arm}臂 {max(HL.max(), HR.max()):.2f} bit @帧位{main_pk}({_phase(main_pk)}),"
            f" 另一臂峰 @{pk_other}({_phase(pk_other)})。")
    lines.append(
        "> 解读注意: 归一化帧位 0/100 两端熵必为 0(所有演示起止夹持器均为张开 1.0); "
        "中段熵高说明该阶段各演示夹持器所处状态分散(同一相对时刻有人已闭合有人仍张开等)。")
    lines += ["", "## 产物文件", "",
              "- `entropy_vs_frame_task_{0,1,2}.png`: 每任务 L/R 两面板, 熵随归一化帧位曲线",
              "- `histogram_entropy.png` / `histogram_entropy_task_{0,1,2}.png`: 熵直方图",
              "- `entropy_series.csv`: 每任务每帧位 H_L / H_R",
              "- `entropy_stats_summary.csv` / `entropy_LR_test.csv`: 统计量",
              ""]
    (OUT / "report_entropy.md").write_text("\n".join(lines), encoding="utf-8")
    log(f"report saved: {OUT / 'report_entropy.md'}")
    log(f"ALL DONE -> {OUT}")


if __name__ == "__main__":
    main()
