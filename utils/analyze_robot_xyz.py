#!/usr/bin/env python3
"""真机评测 xyz 轨迹分析：验证「开动后原地抖动、不前进」的问题。

输入：utils/extract_robot_log_csv.py 生成的总表（robot_test/real_episodes.csv）
输出：
  - 每个 episode × 每条手臂一行的指标 CSV
  - 汇总打印 + 每个模型一张位移-时间图（png）

指标定义（帧 = 一个控制步，chunk = execute_steps=30 帧）：
  first_move_frame  首次离开起点 >2cm 的帧号（0 表示首帧即开始动）
  stall_frames      首次移动之后，最长的「停滞段」帧数。停滞判据：
                    连续帧满足 |p(t) - p(t-chunk)| < 5mm（一个 chunk 的净位移
                    < 5mm，即没有推进），段长按 30 帧窗口滑窗统计
  stall_start/end   该停滞段的起止帧号
  stall_pct         停滞帧数 / 首次移动后的总帧数
  jitter_mm         停滞段内相邻帧指令位移均值（mm）—— 抖动幅度
  gyro_mm           停滞段内指令相对本 chunk 均值的平均偏离（mm）—— 颤动半径
  boundary_jump_mm  停滞段内 chunk 首帧相对上一帧的跳变均值（mm）—— 拼接抖动
  net_move_m        首末位置直线距离（m）
  path_len_m        累计路程（m）
  progress_ratio    net/path，越低说明「走得多、没走远」
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CHUNK = 30  # execute_steps
MOVE_EPS = 0.02  # 首次移动判据：离开起点 2cm
STALL_EPS = 0.005  # 停滞判据：单个 chunk 净位移 < 5mm


def _traj(g: pd.DataFrame, arm: str, kind: str) -> np.ndarray:
    return g[[f"{kind}_{arm}_x", f"{kind}_{arm}_y", f"{kind}_{arm}_z"]].to_numpy(float)


def analyze_episode(g: pd.DataFrame, arm: str, stall_eps: float = STALL_EPS) -> dict:
    """g: 单 episode 全帧、按 frame_index 升序。用密集指令流（action）做判据，
    用实测 state（chunk 边界）做交叉验证。"""
    pa = _traj(g, arm, "action")
    n = len(pa)

    # 首次移动：相对首帧位移 > MOVE_EPS
    disp = np.linalg.norm(pa - pa[0], axis=1)
    moved = np.flatnonzero(disp > MOVE_EPS)
    first_move = int(moved[0]) if len(moved) else -1

    # 停滞判据：与一个 chunk 之前的位置相比几乎没动
    drift = np.full(n, np.nan)
    drift[CHUNK:] = np.linalg.norm(pa[CHUNK:] - pa[:-CHUNK], axis=1)

    start = max(first_move, CHUNK) if first_move >= 0 else CHUNK
    stall = np.zeros(n, dtype=bool)
    if n > start:
        stall[start:] = drift[start:] < stall_eps

    # 最长连续 True 段
    best_len = best_start = 0
    i = 0
    while i < n:
        if stall[i]:
            j = i
            while j + 1 < n and stall[j + 1]:
                j += 1
            if j - i + 1 > best_len:
                best_len, best_start = j - i + 1, i
            i = j + 1
        else:
            i += 1
    stall_end = best_start + best_len - 1 if best_len else best_start

    seg = pa[best_start : stall_end + 1]
    d = np.linalg.norm(np.diff(pa, axis=0), axis=1) if n > 1 else np.array([0.0])
    boundary = np.arange(CHUNK - 1, len(d), CHUNK)
    intra = np.setdiff1d(np.arange(len(d)), boundary)

    def _gyro(a: np.ndarray) -> float:
        if len(a) < 2:
            return float("nan")
        return float(np.mean([np.linalg.norm(a[k : k + CHUNK] - a[k : k + CHUNK].mean(0), axis=1).mean()
                              for k in range(0, len(a) - 1, CHUNK)]))

    seg_boundary = boundary[(boundary >= best_start) & (boundary <= stall_end)]
    path = float(d.sum())
    net = float(np.linalg.norm(pa[-1] - pa[0]))
    return {
        "frames": n,
        "chunks": int(np.ceil(n / CHUNK)),
        "first_move_frame": first_move,
        "stall_frames": best_len,
        "stall_start": best_start,
        "stall_end": stall_end,
        "stall_pct": round(best_len / max(n - start, 1), 3),
        "jitter_mm": round(float(d[intra].mean()) * 1000, 3) if len(intra) else np.nan,
        "gyro_mm": round(_gyro(seg) * 1000, 3),
        "boundary_jump_mm": round(float(d[seg_boundary].mean()) * 1000, 3)
        if len(seg_boundary)
        else np.nan,
        "net_move_m": round(net, 4),
        "path_len_m": round(path, 4),
        "progress_ratio": round(net / path, 3) if path > 0 else np.nan,
    }


def analyze(df: pd.DataFrame, stall_eps: float = STALL_EPS) -> pd.DataFrame:
    rows = []
    for (model, episode), g in df.groupby(["model_name", "episode"], sort=False):
        g = g.sort_values("frame_index")
        if len(g) < CHUNK + 1:
            print(f"skip: {model}/{episode} 仅 {len(g)} 帧，不足以判定", file=sys.stderr)
            continue
        for arm in ("l", "r"):
            rows.append({"model_name": model, "episode": episode, "arm": arm,
                         **analyze_episode(g, arm, stall_eps)})
    return pd.DataFrame(rows)


def plot(df: pd.DataFrame, stats: pd.DataFrame, outdir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for model, gmodel in df.groupby("model_name"):
        eps = sorted(gmodel["episode"].unique())
        fig, axes = plt.subplots(len(eps), 1, figsize=(11, 2.2 * len(eps)), squeeze=False)
        for ax, ep in zip(axes[:, 0], eps):
            g = gmodel[gmodel["episode"] == ep].sort_values("frame_index")
            for arm, color in (("l", "tab:blue"), ("r", "tab:red")):
                for kind, ls in (("action", "-"), ("state", "--")):
                    a = _traj(g.dropna(subset=[f"state_{arm}_x"]) if kind == "state" else g,
                              arm, kind)
                    if not len(a):
                        continue
                    ref = a[0]
                    ax.plot(
                        np.arange(len(a)) * (CHUNK if kind == "state" else 1),
                        np.linalg.norm(a - ref, axis=1) * 100,
                        ls, color=color, lw=1.0,
                        label=f"{arm} {kind}" if ax is axes[0, 0] else None,
                    )
            s = stats[(stats.model_name == model) & (stats.episode == ep)]
            hit = s[s.stall_pct.fillna(0) > 0.3]
            title = f"{model} / {ep}"
            if len(hit):
                title += "   stall: " + ", ".join(
                    f"{r.arm} {r.stall_start}-{r.stall_end} ({r.stall_pct:.0%}, "
                    f"jitter {r.jitter_mm:.1f}mm/frame, radius {r.gyro_mm:.1f}mm)"
                    for r in hit.itertuples()
                )
            ax.set_title(title, fontsize=8)
            ax.set_ylabel("dist. from start (cm)")
            ax.grid(alpha=0.3)
        axes[0, 0].legend(fontsize=6, ncol=4)
        axes[-1, 0].set_xlabel("frame")
        fig.tight_layout()
        fig.savefig(outdir / f"xyz_{model}.png", dpi=110)
        plt.close(fig)


def _weighted_stall_pct(g: pd.DataFrame) -> float:
    """帧加权停滞占比 = Σ停滞帧 / Σ(首次移动后的帧数)。"""
    post = (g["frames"] - np.maximum(g["first_move_frame"], CHUNK)).clip(lower=1)
    return float(g["stall_frames"].sum() / post.sum())


def summarize(stats: pd.DataFrame, outdir: Path) -> None:
    """按模型聚合停滞占比，打印并写 model_summary.csv。"""
    rows = []
    by_model = stats.groupby("model_name")
    for model, g in by_model:
        row = {
            "model_name": model,
            "n_episodes": g.groupby("episode").ngroups,
            "n_arm_samples": len(g),
            "stall_pct_weighted": round(_weighted_stall_pct(g), 3),
            "stall_pct_mean": round(g["stall_pct"].mean(), 3),
            "stall_pct_median": round(g["stall_pct"].median(), 3),
            "stall_pct_min": round(g["stall_pct"].min(), 3),
            "stall_pct_max": round(g["stall_pct"].max(), 3),
            "stall_chunk_median": round((g["stall_frames"] / CHUNK).median(), 1),
            "stall_chunk_max": round((g["stall_frames"] / CHUNK).max(), 1),
            "stall_start_median": int(g["stall_start"].median()),
            "ep_with_stall_gt50pct": int(
                (g.groupby("episode")["stall_pct"].max() > 0.5).sum()
            ),
        }
        for arm in ("l", "r"):
            sub = g[g["arm"] == arm]
            row[f"stall_pct_{arm}_weighted"] = round(_weighted_stall_pct(sub), 3) if len(sub) else np.nan
        rows.append(row)
    summary = pd.DataFrame(rows).sort_values("stall_pct_weighted", ascending=False)
    summary.to_csv(outdir / "model_summary.csv", index=False)

    print("\n=== 按模型汇总（停滞占比 = 停滞帧 / 首次移动后的帧数）===")
    print(summary.to_string(index=False))

    print("\n=== 按 模型 × 手臂（帧加权停滞占比）===")
    piv = stats.assign(post=(stats["frames"] - np.maximum(stats["first_move_frame"], CHUNK)).clip(lower=1))
    arm_tab = piv.groupby(["model_name", "arm"]).apply(
        lambda g: g["stall_frames"].sum() / g["post"].sum(), include_groups=False
    ).unstack().round(3)
    arm_tab.columns = [f"stall_pct_{c}" for c in arm_tab.columns]
    print(arm_tab.sort_values("stall_pct_l", ascending=False).to_string())
    print(f"\nsummary -> {outdir}/model_summary.csv")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="robot_test/real_episodes.csv")
    parser.add_argument("--outdir", default="outputs/real_xyz")
    parser.add_argument("--plot", action="store_true", help="额外输出每个模型的轨迹图")
    parser.add_argument("--detail", action="store_true", help="打印 episode×手臂 明细")
    parser.add_argument("--stall-mm", type=float, default=STALL_EPS * 1000,
                        help="停滞判据：单个 chunk 净位移阈值 (mm)，默认 5")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.csv)
    stats = analyze(df, args.stall_mm / 1000)
    stats.to_csv(outdir / "episode_arm_stats.csv", index=False)

    if args.plot:
        plot(df, stats, outdir)
        print(f"plots -> {outdir}/xyz_*.png")
    if args.detail:
        print(stats.sort_values(["model_name", "episode", "arm"]).to_string(index=False))

    summarize(stats, outdir)
    print("\n停滞段抖动幅度 (mm/帧): "
          f"中位 {stats.jitter_mm.median():.2f}, 最大 {stats.jitter_mm.max():.2f}")
    print("停滞段 chunk 边界跳变 (mm): "
          f"中位 {stats.boundary_jump_mm.median():.2f}, 最大 {stats.boundary_jump_mm.max():.2f}")
    print(f"stats -> {outdir}/episode_arm_stats.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
