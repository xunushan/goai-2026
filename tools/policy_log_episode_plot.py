#!/usr/bin/env python3
"""把策略服务日志 CSV 的某个 episode 画成 state 时序图。

桥接 [utils/extract_policy_log_csv.py](../utils/extract_policy_log_csv.py)
(策略日志 → 逐帧 CSV) 与
[tools/episode_state_insight.py](episode_state_insight.py) (state 3 子图绘图)：
前者输出的是 `state_<dim>` 宽表 + `env_idx`/`episode_uuid`，后者要求
`observation.state` 列表列并按 `episode_index` 过滤，两者不直接对接。

**关键事实（决定图长什么样）**：服务端只在 action chunk 边界上报 state，
即每 `actions_per_chunk` 帧才有一个 state 采样点，chunk 内其余帧 `state_*`
为空。因此 state 曲线是稀疏采样，不是逐帧轨迹；`action_*` 才是逐帧的。
本脚本只画真实采样点，不做插值/补齐。

并行评测的 episode 划分：`eval-num=N` 时 N 个 episode 跑在同一 reset 组内的
N 个并行 env，`episode_index` 恒为 0，须按 `env_idx` 或 `episode_uuid` 区分。

用法:
    python tools/policy_log_episode_plot.py \
        --csv outputs/simu_analysis/stack_bowls_policy_log.csv \
        --out outputs/simu_analysis
    # 只画指定 env / uuid:
    python tools/policy_log_episode_plot.py --csv ... --env 3
    python tools/policy_log_episode_plot.py --csv ... --uuid f43c4451
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.extract_policy_log_csv import STATE_NAMES  # noqa: E402
from tools.episode_state_insight import plot_episode_state  # noqa: E402


def load_episode(df: pd.DataFrame, env_idx: int) -> tuple[np.ndarray, np.ndarray]:
    """取单个 env 的 (state (T,16), frame (T,))，只保留有 state 采样的帧。

    `state_*` 在 chunk 内为 NaN（日志里是空字符串），这些帧没有 state 可画，
    丢弃而不是前向填充——填充会伪造出并不存在的采样密度。
    """
    ep = df[df["env_idx"] == env_idx].sort_values("frame_index")
    has_state = ep[f"state_{STATE_NAMES[0]}"].notna()
    ep = ep[has_state]
    state = ep[[f"state_{n}" for n in STATE_NAMES]].to_numpy(dtype=float)
    return state, ep["frame_index"].to_numpy()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True, help="extract_policy_log_csv.py 输出的 CSV")
    ap.add_argument("--out", required=True, help="输出目录")
    ap.add_argument("--env", type=int, default=None, help="只画该 env_idx")
    ap.add_argument("--uuid", default=None, help="只画该 episode_uuid")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if "episode_uuid" not in df.columns:
        df["episode_uuid"] = ""
    envs = sorted(df["env_idx"].unique())
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    n = 0
    for e in envs:
        if args.env is not None and int(e) != args.env:
            continue
        sub = df[df["env_idx"] == e]
        uuid = str(sub["episode_uuid"].iloc[0]) if len(sub) else ""
        if args.uuid and uuid != args.uuid:
            continue
        task = str(sub["task"].iloc[0]) if len(sub) else ""
        state, frame = load_episode(df, int(e))
        if state.shape[0] < 2:
            print(f"  ! env {e} 采样点不足 ({state.shape[0]}), 跳过")
            continue
        out_path = out_dir / f"stack_bowls_ep{e}_{uuid or 'nouuid'}_state.png"
        total = int(sub["frame_index"].max()) + 1
        plot_episode_state(
            frame, state,
            task, "stack_bowls", int(e), out_path,
            # 副标题只用 ASCII：matplotlib 默认字体 DejaVu Sans 无 CJK 字形，
            # 中文会渲染成豆腐块。
            subtitle=(f"env {e} / {uuid or '-'}   |   {total} frames   |   "
                      f"state sampled at {state.shape[0]} chunk boundaries only"),
        )
        print(f"  env {e} uuid={uuid or '-':8s} frames={int(sub['frame_index'].max()) + 1:4d} "
              f"state_samples={state.shape[0]:3d} -> {out_path}")
        n += 1
    print(f"done, {n} figures -> {out_dir}")
    return 0 if n else 1


if __name__ == "__main__":
    raise SystemExit(main())
