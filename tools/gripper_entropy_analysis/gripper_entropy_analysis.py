#!/usr/bin/env python3
"""gripper(夹持器) action 目标二元熵分析 (H = -p ln p - (1-p) ln(1-p)).

指标(用户确认):
    target = action[..., l_g / r_g].clamp(1e-6, 1-1e-6)
    H = -(target*ln(target) + (1-target)*ln(1-target))     # nat, H∈[0, ln2]
  16D action 中 gripper 下标 l_g=7 / r_g=15 (与 20D 布局 9/19 数值一致, 见
  scripts/convert_lerobot_ee16_to_xvla20.py; 二元熵对 p<->1-p 对称)。

用法:
  python gripper_entropy_analysis.py                     # sim 全任务(默认)
  python gripper_entropy_analysis.py --csv data/lerobot_v30_ee/lerobot_v30_ee.csv \
      --instr-json data/lerobot_v30_ee/meta/tasks.json --tasks 8 \
      --tag real_stack_blocks                            # real 数据指定任务

产物(outputs/gripper_entropy_action_binary[_<tag>]_<date>/):
  entropy_frame_task_{t}.png   逐帧曲线(L/R 面板, 逐条 episode 浅线+均值±σ带)
  histogram_entropy_all/task_{t}.png  直方图
  entropy_mean_series.csv / entropy_stats_summary.csv / entropy_LR_test.csv
  report_entropy.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from datetime import date

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

SIM_CSV = ROOT / "data" / "sim_lerobot_v30_ee" / "sim_lerobot_v30_ee.csv"
SIM_INSTR = ROOT / "data" / "sim_lerobot_v30_ee" / "train_val_split.json"

A_G_L, A_G_R = 7, 15
CLAMP = (1e-6, 1.0 - 1e-6)
ARMS = {"L": "grip_L", "R": "grip_R"}
C_L, C_R = "#1f77b4", "#ff7f0e"
MIN_N_BAND = 10
H_MAX = float(np.log(2))          # ln2 ≈ 0.693


def log(msg): print(f"[{date.today().isoformat()}] {msg}", flush=True)


def H_of_p(p: np.ndarray) -> np.ndarray:
    """H = -p ln p - (1-p) ln(1-p); p 先 clamp 到 (1e-6, 1-1e-6)。"""
    p = np.clip(np.asarray(p, float), *CLAMP)
    return -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))


def load_action_grippers(csv: Path, only_tasks: set[int] | None) -> pd.DataFrame:
    """流式读 CSV 的 action 列, 每 episode 汇总一行。

    列: task_index, episode_index, length, grip_L, grip_R (夹持器目标序列, L=idx7/R=idx15)。
    CSV 行须按 (episode, frame) 升序排列(与 lerobot 导出一致), 组内即帧序。
    """
    cache = OUT / f"_eps_{csv.stem}_{('all' if not only_tasks else 't'+''.join(map(str, sorted(only_tasks))))}.pkl"
    if cache.exists():
        log(f"load from cache {cache}")
        return pd.read_pickle(cache)

    groups: dict = {}   # (task, ep) -> ([gL],[gR]), dict 保序 = 首见顺序
    for chunk in pd.read_csv(csv, usecols=["task_index", "episode_index", "action"],
                             chunksize=150000):
        for ti, ei, a in zip(chunk["task_index"], chunk["episode_index"], chunk["action"]):
            if only_tasks and int(ti) not in only_tasks:
                continue
            tok = a.split(",")
            gL = float(tok[A_G_L])
            gR = float(tok[A_G_R].rstrip("]"))
            b = groups.setdefault((int(ti), int(ei)), ([], []))
            b[0].append(gL); b[1].append(gR)
    rows = [{"task_index": k[0], "episode_index": k[1], "length": len(v[0]),
             "grip_L": np.asarray(v[0], float), "grip_R": np.asarray(v[1], float)}
            for k, v in groups.items()]
    out = pd.DataFrame(rows)
    out.to_pickle(cache)
    log(f"loaded {len(out)} episodes (tasks {sorted(out.task_index.unique())}) from {csv.name}")
    return out


def task_instructions(path: Path) -> dict[int, str]:
    """兼容两种 JSON: {task_index: instruction} 或 {tasks:{idx:{instruction}},...}。"""
    d = json.load(open(path))
    if isinstance(d, dict) and isinstance(d.get("tasks"), dict):
        first = next(iter(d["tasks"].values()))
        if isinstance(first, dict) and "instruction" in first:
            return {int(k): v["instruction"] for k, v in d["tasks"].items()}
    return {int(k): str(v) for k, v in d.items()}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path, default=SIM_CSV)
    ap.add_argument("--instr-json", type=Path, default=SIM_INSTR)
    ap.add_argument("--tasks", type=int, nargs="+", default=None,
                    help="限定 task_index 列表; 缺省=处理文件内全部任务")
    ap.add_argument("--tag", default="",
                    help="输出目录后缀, 例如 real_stack_blocks")
    return ap.parse_args()


def main() -> None:
    global OUT
    args = parse_args()
    tag = f"_{args.tag}" if args.tag else ""
    OUT = ROOT / "outputs" / f"gripper_entropy_action_binary{tag}_{date.today().isoformat()}"
    OUT.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import stats as sps
    sys.path.insert(0, str(ROOT / "tools" / "gripper_val_split"))
    from gripper_common import setup_cjk_font
    setup_cjk_font()

    instr = task_instructions(args.instr_json)
    eps = load_action_grippers(args.csv, set(args.tasks) if args.tasks else None)
    tasks = sorted(int(x) for x in eps["task_index"].unique())

    Hs = {}
    for t in tasks:
        te = eps[eps["task_index"] == t]
        for arm, col in ARMS.items():
            Hs[(t, arm)] = [H_of_p(g) for g in te[col]]

    # ---------- 1) 逐帧曲线(L/R 面板, 逐条+均值带) ----------
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
            ax.set_title(f"task {t} · {arm}臂 (action gripper, 均值±σ深色带, 虚=少样本尾部)")
            ax.grid(alpha=0.3)
        nm = instr.get(t, f"task {t}")
        fig.suptitle(f"{args.csv.parent.name} | task {t}: {nm}\n"
                     f"action gripper 目标 p 的二元熵; p clamp 后 H∈[0,{H_MAX:.3f}], p=0.5 最大",
                     fontsize=9.5)
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        fig.savefig(OUT / f"entropy_frame_task_{t}.png", dpi=140)
        plt.close(fig)
    log("curve figs done")

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
    if len(tasks) == 1:
        t = tasks[0]
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        ax.hist(np.concatenate(Hs[(t, "L")]), bins=60, alpha=0.6, color=C_L,
                label="L臂", density=True)
        ax.hist(np.concatenate(Hs[(t, "R")]), bins=60, alpha=0.6, color=C_R,
                label="R臂", density=True)
        ax.set_title(f"task {t} {instr.get(t,'')[:40]}... 二元熵直方图 (L vs R)")
        ax.set_xlabel("H (nat)"); ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT / "histogram_entropy_task.png", dpi=140)
        plt.close(fig)
        hist_file = ["histogram_entropy_task.png"]
    else:
        allH = {arm: np.concatenate([h for t in tasks for h in Hs[(t, arm)]])
                for arm in ["L", "R"]}
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        ax = axes[0, 0]
        ax.hist(allH["L"], bins=60, alpha=0.6, color=C_L, label="L臂", density=True)
        ax.hist(allH["R"], bins=60, alpha=0.6, color=C_R, label="R臂", density=True)
        ax.set_title("所选任务合计 (每episode每帧一个 H)"); ax.legend(); ax.grid(alpha=0.3)
        ax.set_xlabel("H (nat)"); ax.set_ylabel("密度")
        for j, t in enumerate(tasks[:3]):
            ax = [axes[0, 1], axes[1, 0], axes[1, 1]][j]
            ax.hist(np.concatenate(Hs[(t, "L")]), bins=50, alpha=0.6, color=C_L,
                    label="L臂", density=True)
            ax.hist(np.concatenate(Hs[(t, "R")]), bins=50, alpha=0.6, color=C_R,
                    label="R臂", density=True)
            ax.set_title(f"task {t}: {instr.get(t,'')[:28]}...")
            ax.legend(); ax.grid(alpha=0.3); ax.set_xlabel("H (nat)"); ax.set_ylabel("密度")
        fig.suptitle("action gripper 目标二元熵直方图", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(OUT / "histogram_entropy_all.png", dpi=140)
        plt.close(fig)
        for t in tasks:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.hist(np.concatenate(Hs[(t, "L")]), bins=60, alpha=0.6, color=C_L,
                    label="L臂", density=True)
            ax.hist(np.concatenate(Hs[(t, "R")]), bins=60, alpha=0.6, color=C_R,
                    label="R臂", density=True)
            ax.set_title(f"task {t} 二元熵直方图 (L vs R)")
            ax.set_xlabel("H (nat)"); ax.legend(); ax.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(OUT / f"histogram_entropy_task_{t}.png", dpi=140)
            plt.close(fig)
        hist_file = ["histogram_entropy_all.png"] + [f"histogram_entropy_task_{t}.png"
                                                     for t in tasks]
    log("hist figs done")

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
        "# action gripper 目标二元熵报告 (H = -p ln p - (1-p) ln(1-p))",
        "",
        f"- 日期: {date.today().isoformat()}",
        f"- 数据: `{args.csv}`",
        f"- 任务: {', '.join(f'{t}={instr.get(t, t)}' for t in tasks)}",
        "- 指标: 每帧取 action 夹持器目标 p (16D 下标 L=7/R=15), clamp(1e-6,1-1e-6) 后",
        "  H = -p·ln(p) - (1-p)·ln(1-p), 自然对数(nat), H∈[0, ln2≈0.693], p=0.5 最大。",
        "- 逐条 episode 逐帧计算, 原始 frame_index; 同任务多条 episode: 逐条浅线+均值±σ 带",
        "  (带样本 ≥ {} 的帧画实线, 否则虚线)。".format(MIN_N_BAND),
        "",
        "> 语义: H 大 = 该帧夹持器命令接近 p≈0.5(半开半闭/不确定); H≈0 = 命令贴向全开/全闭。",
        "",
        "## 1. 统计量 (帧级样本)",
        "",
        "| task | arm | n_frames | frame_mean | frame_std | frame_median | [q25,q75] | max | ep均值avg | ep均值std |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
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
    L += ["", "## 产物文件", "", "- " + "\n- ".join(
        [f"entropy_frame_task_{t}.png" for t in tasks] + hist_file +
        ["entropy_mean_series.csv", "entropy_stats_summary.csv", "entropy_LR_test.csv"]), ""]
    (OUT / "report_entropy.md").write_text("\n".join(L), encoding="utf-8")
    log(f"done -> {OUT}")


if __name__ == "__main__":
    main()
