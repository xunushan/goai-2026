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
from tools.pace import PaceConfig, select_execution_horizon  # noqa: E402

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
