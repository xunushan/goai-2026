#!/usr/bin/env python3
"""把 PACE（tools/pace.py）跑在策略 rollout 的每个 action chunk 上，看它输出什么。

PACE 的输入必须是**一次 query 预测出的那个 chunk**（论文的 L 步预测视野），输出是
**执行视野 h —— 一个「步数」（整数），即执行完 h 步后重新观测**。所以对每个 30 帧
chunk 跑一次，得到的 h ∈ [h_min, h_max] 就是「这个 chunk 只执行前 h 步」的那个 h。

布局差异（本脚本必须处理的唯一坑）
----------------------------------
PACE 按 X-VLA 的 **20 维 ee 布局**取手臂 xyz（`ARM_XYZ_IDX`）：
    0:3 左臂 xyz | 3:9 左 rot6d | 9 左夹爪 | 10:13 右臂 xyz | 13:19 右 rot6d | 19 右夹爪
而 `utils/extract_policy_log_csv.py` 输出的 CSV 是 **16 维**（四元数，无 rot6d）：
    0:3 l_xyz | 3:7 l_quat | 7 l_g | 8:11 r_xyz | 11:15 r_quat | 15 r_g
PACE 的速度剖面只用 xyz（`arm_speed_profile` 只读 `ARM_XYZ_IDX`），所以把 xyz 平移到
PACE 的槽位即可，rot6d 槽位留 0 不影响结果。

阈值 δ_T 从 policy 的 deploy.yml 读（`pace.threshold_delta_by_task.stack_bowls`），
不写死在代码里。注意：这里读的是**仓库内**的 deploy.yml，与服务器上实际生效的那份
可能不同，结论要标注用的是哪份。

用法:
    python tools/pace_chunk_probe.py \
        --csv outputs/simu_analysis/stack_bowls_policy_log.csv --envs all
    python tools/pace_chunk_probe.py --csv ... --envs 1 --verbose
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

from utils.extract_policy_log_csv import STATE_NAMES  # noqa: E402
from tools.pace import (  # noqa: E402
    ARM_XYZ_IDX, PaceConfig, arm_speed_profile, select_execution_horizon,
    smooth_profile, valley_prominences,
)

# dataviz 调色板（与 tools/policy_chunk_keyframe_locate.py 保持一致）
INK = "#0b0b0b"; MUT = "#898781"; GRID = "#e1e0d9"; SURF = "#fcfcfb"
C_XYZ = ("#2a78d6", "#eb6834", "#1baf7a")
C_LEFT = "#2a78d6"
C_RIGHT = "#eb6834"
C_ACC = "#1baf7a"        # 被接受的谷值
C_REJ = "#b8b6ae"        # 未达 δ_T 的候选
C_HPICK = "#9467bd"      # 选中的 h 边界

DEFAULT_DEPLOY_YML = ROOT / "RoboDojo" / "XPolicyLab" / "policy" / "X_VLA" / "deploy.yml"
DEFAULT_RESULT = ROOT / "outputs" / "simu_analysis" / "stack_bowls" / "_result.json"
TASK_NAME = "stack_bowls"

# CSV(16 维) -> PACE(20 维) 的 xyz/夹爪槽位映射
CSV_L_XYZ, CSV_L_GRIP = 0, 7
CSV_R_XYZ, CSV_R_GRIP = 8, 15
PACE_L_XYZ, PACE_L_GRIP = 0, 9
PACE_R_XYZ, PACE_R_GRIP = 10, 19


def to_pace_layout(arr16: np.ndarray) -> np.ndarray:
    """CSV 的 16 维动作 -> PACE 的 20 维 ee 布局（rot6d 槽位留 0，剖面用不到）。"""
    out = np.zeros((arr16.shape[0], 20), dtype=np.float64)
    out[:, PACE_L_XYZ:PACE_L_XYZ + 3] = arr16[:, CSV_L_XYZ:CSV_L_XYZ + 3]
    out[:, PACE_L_GRIP] = arr16[:, CSV_L_GRIP]
    out[:, PACE_R_XYZ:PACE_R_XYZ + 3] = arr16[:, CSV_R_XYZ:CSV_R_XYZ + 3]
    out[:, PACE_R_GRIP] = arr16[:, CSV_R_GRIP]
    return out


def load_pace_cfg(deploy_yml: Path, task: str) -> tuple[PaceConfig, float | None, dict]:
    """从 deploy.yml 的 `pace` 段构造 PaceConfig，并取该任务的 δ_T。"""
    import yaml
    raw = yaml.safe_load(deploy_yml.read_text())
    apc = int(raw.get("actions_per_chunk", 30))
    cfg = PaceConfig.from_model_cfg(raw, apc)
    return cfg, cfg.threshold_for(task), raw


def load_success(path: Path) -> dict[int, bool]:
    if not path.exists():
        return {}
    d = json.loads(path.read_text())
    return {int(k): bool(v.get("success")) for k, v in d.get("details", {}).items()}


def plot_pace_chunk(arr20: np.ndarray, a: int, b: int, dec, cfg: PaceConfig,
                    delta_t: float | None, task: str, env: int, tag: str, T: int,
                    out_path: Path) -> None:
    """把 PACE 的决策标在速度剖面 + 夹爪 + 左右臂 xyz 上。

    x 轴一律用**绝对帧号**，四联共用。速度剖面的第 j 个元素 v[j] 描述
    a[j]→a[j+1] 这一步，故画在其跨度中点 (a+j+0.5) 上，避免与点号错位一格。
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({
        "axes.edgecolor": MUT, "axes.labelcolor": INK, "axes.titlecolor": INK,
        "text.color": INK, "xtick.color": INK, "ytick.color": INK,
        "figure.facecolor": SURF, "axes.facecolor": SURF, "grid.color": GRID,
        "font.family": "sans-serif", "figure.dpi": 110,
    })
    chunk = arr20[a:b + 1]
    f = np.arange(a, b + 1)
    # h 步 = 点 a[0..h-1] = 绝对帧 a..a+h-1；边界落在 a+h-1 与 a+h 之间
    h_frame = a + dec.h - 1 + 0.5
    h_txt = "fallback -> h_max" if dec.fallback else f"h={dec.h}"

    fig, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=True,
                             gridspec_kw={"height_ratios": [1.35, 1, 1, 1]})
    fig.suptitle(
        f"{task}   |   env {env} [{tag}]"
        f"   |   chunk ({a}-{b} of {T})   |   "
        f"PACE {h_txt}   δ_T={delta_t:.2e}\n"
        f"purple dashed = re-observe boundary (execute points 0..{dec.h - 1}, "
        f"i.e. frames {a}..{a + dec.h - 1})",
        fontsize=10, y=0.985)

    # ---- 面板 1: 速度剖面（PACE 的输入） ----
    ax = axes[0]
    for arm, color, ls in (("left", C_LEFT, "-"), ("right", C_RIGHT, "--")):
        v = arm_speed_profile(chunk, ARM_XYZ_IDX[arm])
        vs = smooth_profile(v, cfg.smooth_window, cfg.smooth)
        x = a + np.arange(len(v)) + 0.5
        ax.plot(x, v, color=color, ls=ls, lw=1.0, alpha=0.45,
                label=f"v {arm} (raw)")
        ax.plot(x, vs, color=color, ls=ls, lw=1.8, label=f"v {arm} (smoothed)")
        idx, pr = valley_prominences(vs, 0, len(chunk) - 1)
        for j, p in zip(idx, pr):
            acc = delta_t is not None and p >= delta_t
            ax.plot([a + j + 0.5], [vs[j]], marker="v" if acc else "o",
                    ms=8 if acc else 5, color=C_ACC if acc else C_REJ,
                    mec=INK if acc else "none", mew=0.6, zorder=5)
            if acc:
                ax.annotate(f"Φ={p:.1e}\nh={j + 1}", (a + j + 0.5, vs[j]),
                            textcoords="offset points", xytext=(6, -26),
                            fontsize=7.5, color=INK)
    if delta_t is not None:
        ax.axhline(delta_t, color=C_ACC, lw=1.0, ls="-.", alpha=0.8)
        ax.annotate(f"δ_T={delta_t:.2e}", (b, delta_t), fontsize=7.5,
                    color=C_ACC, ha="right", va="bottom", annotation_clip=False)
    ax.set_ylabel("arm speed\nstep length (m/step)")
    ax.legend(fontsize=7.5, ncol=2, loc="upper right")

    # ---- 面板 2: 夹爪 ----
    ax = axes[1]
    ax.plot(f, arr20[a:b + 1, PACE_L_GRIP], color=C_LEFT, ls="-", lw=1.4,
            marker="o", ms=2.6, label="Left")
    ax.plot(f, arr20[a:b + 1, PACE_R_GRIP], color=C_RIGHT, ls="--", lw=1.4,
            marker="o", ms=2.6, label="Right")
    ax.set_ylabel("gripper\n(0 closed ~ 1 open)")
    ax.set_ylim(-0.05, 1.05); ax.legend(fontsize=8, ncol=2, loc="lower right")

    # ---- 面板 3/4: 左右臂 xyz ----
    for ax, arm, name in ((axes[2], "left", "Left arm"), (axes[3], "right", "Right arm")):
        o = PACE_L_XYZ if arm == "left" else PACE_R_XYZ
        for d, c in enumerate(C_XYZ):
            ax.plot(f, arr20[a:b + 1, o + d], color=c, ls="-", lw=1.3,
                    marker="o", ms=2.6, label=f"{arm[0].upper()} {'xyz'[d]}")
        ax.set_ylabel(f"{name}\nxyz (m)"); ax.legend(fontsize=8, ncol=3, loc="upper right")

    for ax in axes:
        ax.grid(alpha=0.3)
        ax.axvline(h_frame, color=C_HPICK, lw=1.6, ls="--", alpha=0.9)
    axes[0].legend(handles=axes[0].get_legend_handles_labels()[0] + [
        Line2D([], [], color=C_HPICK, ls="--", lw=1.6, label="PACE h boundary")],
        fontsize=7.5, ncol=2, loc="upper right")
    axes[-1].set_xlabel("frame index")
    axes[-1].set_xlim(a - 1, b + 1)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--envs", default="all")
    ap.add_argument("--chunk", type=int, default=30)
    ap.add_argument("--deploy-yml", default=str(DEFAULT_DEPLOY_YML))
    ap.add_argument("--result-json", default=str(DEFAULT_RESULT))
    ap.add_argument("--task", default=TASK_NAME)
    ap.add_argument("-v", "--verbose", action="store_true", help="打印每个候选谷值")
    ap.add_argument("--out", default=None, help="可选：把逐 chunk 结果写成 CSV")
    ap.add_argument("--plot", default=None, metavar="DIR",
                    help="把 PACE 决策标到速度剖面+夹爪+xyz 四联图上，输出到该目录")
    ap.add_argument("--plot-all", action="store_true",
                    help="默认只画非回落块（PACE 真正生效的），加此项则每块都画")
    args = ap.parse_args()

    cfg, delta_t, raw = load_pace_cfg(Path(args.deploy_yml), args.task)
    print(f"PACE 配置来源: {args.deploy_yml}")
    print(f"  enabled={cfg.enabled}  h_max={cfg.h_max} (actions_per_chunk="
          f"{raw.get('actions_per_chunk')})  h_min={cfg.h_min}  d_min={cfg.d_min}  "
          f"smooth={cfg.smooth}/{cfg.smooth_window}  arms={cfg.arms}")
    print(f"  δ_T[{args.task}] = {delta_t}   (全局兜底 threshold_delta="
          f"{cfg.threshold_delta})")
    if delta_t is None:
        print("  !! 该任务无 δ_T -> PACE 会直接回落 h_max（固定视野行为），结论无意义")

    df = pd.read_csv(args.csv)
    success = load_success(Path(args.result_json))
    envs = (sorted(int(x) for x in df["env_idx"].unique())
            if args.envs == "all" else [int(x) for x in args.envs.split(",")])

    rows = []
    for e in envs:
        grp = df[df["env_idx"] == e].sort_values("frame_index")
        arr16 = grp[[f"action_{n}" for n in STATE_NAMES]].to_numpy(dtype=float)
        arr20 = to_pace_layout(arr16)
        T = len(arr20)
        tag = {True: "成功", False: "失败", None: "未知"}[success.get(e)]
        # 图上只用 ASCII 标签：matplotlib 默认字体 DejaVu Sans 无 CJK 字形，中文会变豆腐块
        tag_ascii = {True: "PASS", False: "FAIL", None: "N/A"}[success.get(e)]
        n_chunks = (T + args.chunk - 1) // args.chunk
        print(f"\n=== env {e} [{tag}]  T={T}  chunk={n_chunks} (每块 {args.chunk} 帧) ===")
        print("  chunk   帧范围      h    回落?  候选数  被接受(Φ>=δ_T)   最强的 3 个候选(arm,h,Φ)")
        hs = []
        for ci in range(n_chunks):
            a, b = ci * args.chunk, min((ci + 1) * args.chunk, T) - 1
            chunk = arr20[a:b + 1]
            d = select_execution_horizon(chunk, cfg, delta_t, task_name=args.task)
            hs.append(d.h)
            acc = [c for c in d.candidates if c["accepted"]]
            top = sorted(d.candidates, key=lambda c: -c["prominence"])[:3]
            top_txt = "  ".join(f"({c['arm'][0]},{c['h']},{c['prominence']:.2e})"
                                + ("*" if c["accepted"] else "") for c in top)
            print(f"  c{ci:02d}   [{a:4d},{b:4d}]  h={d.h:3d}  "
                  f"{'YES' if d.fallback else ' - ':>4s}  {len(d.candidates):5d}  "
                  f"{len(acc):5d}            {top_txt}")
            if args.verbose:
                for c in d.candidates:
                    print(f"        {c['arm']:5s} h={c['h']:3d} Φ={c['prominence']:.6e} "
                          f"{'ACCEPT' if c['accepted'] else ''}")
            rows.append({"env": e, "success": success.get(e), "chunk": ci,
                         "start": a, "end": b, "h": d.h, "fallback": d.fallback,
                         "n_candidates": len(d.candidates), "n_accepted": len(acc),
                         "delta_t": delta_t})
            if args.plot and (args.plot_all or not d.fallback):
                plot_pace_chunk(arr20, a, b, d, cfg, delta_t, args.task, e, tag_ascii, T,
                                Path(args.plot) / f"pace_ep{e}_c{ci:03d}_f{a}-{b}_h{d.h}"
                                                  f"{'_fb' if d.fallback else ''}.png")
        hs = np.array(hs)
        print(f"  --> h 分布: mean={hs.mean():.1f} median={np.median(hs):.0f} "
              f"min={hs.min()} max={hs.max()}  |  回落(h==h_max) {int((hs == cfg.h_max).sum())}/"
              f"{len(hs)} 块  |  贴 h_min 的 {int((hs == cfg.h_min).sum())} 块")
        # 本次 rollout 里 PACE 是关的（deploy.yml enabled: false），策略确实每 30 帧
        # 预测一次并整段执行，所以固定 30 帧窗口 == 真实预测 chunk，是**精确**对应，
        # 不是近似。但这不是「PACE 开启后的行为预测」：那时 chunk 边界会随选出的 h
        # 移动，不再落在固定网格上，需在仿真里做真正的 A/B。
        print("  注: 本次 rollout PACE 关闭且无 temporal ensemble，策略确实每 30 帧"
              "预测一次并整段执行，故 30 帧窗口与真实预测 chunk 精确对应；"
              "但 PACE 一旦开启，chunk 边界会随 h 移动，本表不能当作上线行为预测。")

    if args.out:
        pd.DataFrame(rows).to_csv(args.out, index=False)
        print(f"\n逐 chunk 结果 -> {args.out}  ({len(rows)} 行)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
