#!/usr/bin/env python3
"""为 Lerobot 训练集每帧计算 frame_weight（关键帧时刻加权）。

对每个 episode 的左右爪夹（observation.state idx7=left, idx15=right）复用
tools/keyframe_detect.py 的 detect_gripper_keyframes（参数与之前一致）检测
hold_start / hold_end，按任务定义权重窗口，帧命中任一窗口 frame_weight=1.5，
其余 =1.0。左右爪窗口取并集，任一命中即 1.5。

权重窗口（帧单位，越界自动裁剪）:
    默认任务:   [hold_start-15, hold_start+10] ∪ [hold_end-10, hold_end+5]
    pour_liquid_into_cup (task 5): [hold_start-15, hold_start+100]  (仅 hold_start)
    push_T              (task 6): [hold_start-15, hold_start+15]    (仅 hold_start)

权重以 parquet 为唯一计算源（episode_index/frame_index/task_index/state 全量），
再按 (episode_index, frame_index) 对齐写入 CSV 或 parquet，保证两处一致。

用法:
    # 先写 CSV（供 review）: 给 data/lerobot_v30_ee.csv 追加 frame_weight 列
    python tools/frame_weight.py --target csv

    # review 后写 parquet: 给 data/chunk-000/file-000.parquet 追加 frame_weight 列
    python tools/frame_weight.py --target parquet

    # 输出到其它路径（不覆盖原文件）
    python tools/frame_weight.py --target csv --out /tmp/weighted.csv
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.keyframe_detect import GRIP_L, GRIP_R, detect_gripper_keyframes  # noqa: E402

DEFAULT_CSV = ROOT / "data" / "lerobot_v30_ee.csv"
DEFAULT_PARQUET = (
    ROOT / "data" / "lerobot_v30_ee" / "data" / "chunk-000" / "file-000.parquet"
)

WEIGHT_HIT = 1.5
WEIGHT_BASE = 1.0

# 任务特化窗口: 'hs'=(hold_start 前/后帧数), 'he'=(hold_end 前/后帧数) 或 None
WINDOWS_OVERRIDE: dict[int, dict[str, tuple[int, int] | None]] = {
    5: {"hs": (15, 100), "he": None},  # pour_liquid_into_cup: 持瓶+倒水全程
    6: {"hs": (15, 15), "he": None},   # push_T: 下压-保持
}
DEFAULT_WINDOWS = {"hs": (15, 10), "he": (10, 5)}


# ---------------------------------------------------------------------------
# 权重计算
# ---------------------------------------------------------------------------

def _apply_window(weight: np.ndarray, frame: np.ndarray, hold: int,
                  spec: tuple[int, int] | None) -> None:
    """把 hold 关键帧的 [hold-pre, hold+post] 窗口内帧权重置为 WEIGHT_HIT。"""
    if spec is None:
        return
    pre, post = spec
    lo, hi = hold - pre, hold + post
    mask = (frame >= lo) & (frame <= hi)
    weight[mask] = WEIGHT_HIT


def episode_weight(
    state: np.ndarray,
    frame: np.ndarray,
    task_index: int,
    detect_kwargs: dict,
) -> np.ndarray:
    """单个 episode 的权重数组 (T,)，与 state/frame 逐行对齐。"""
    weight = np.full(len(state), WEIGHT_BASE)
    spec = WINDOWS_OVERRIDE.get(int(task_index), DEFAULT_WINDOWS)
    for idx in (GRIP_L, GRIP_R):
        kf = detect_gripper_keyframes(state[:, idx], frame, **detect_kwargs)
        for h in kf["hold_start"]:
            _apply_window(weight, frame, int(h), spec["hs"])
        he_spec = spec["he"]
        if he_spec is not None:
            for h in kf["hold_end"]:
                _apply_window(weight, frame, int(h), he_spec)
    return weight


# ---------------------------------------------------------------------------
# 读取 / 写出
# ---------------------------------------------------------------------------

def load_parquet_frames(parquet_path: Path):
    """读取 parquet 帧表 -> (df[episode/frame/task], state (N,16), 物理行序)。"""
    table = pq.read_table(
        str(parquet_path),
        columns=["index", "episode_index", "frame_index", "task_index",
                 "observation.state"],
    )
    df = table.select(
        ["episode_index", "frame_index", "task_index", "index"]
    ).to_pandas()
    state = np.vstack(table.column("observation.state").to_numpy())  # (N,16)
    return table, df, state


def compute_weights_from_parquet(
    parquet_path: Path, detect_kwargs: dict
) -> np.ndarray:
    """对全部帧计算权重，数组与 parquet 物理行序对齐。"""
    table, df, state = load_parquet_frames(parquet_path)
    weights = np.full(len(df), WEIGHT_BASE)
    for ep, positions in df.groupby("episode_index", sort=True).groups.items():
        positions = np.asarray(positions)
        ep_state = state[positions]
        ep_frame = df["frame_index"].to_numpy()[positions]
        ep_task = int(df["task_index"].to_numpy()[positions[0]])
        weights[positions] = episode_weight(ep_state, ep_frame, ep_task, detect_kwargs)
    return weights


def write_csv(csv_path: Path, parquet_path: Path, detect_kwargs: dict,
              out_path: Path | None = None) -> tuple[int, int]:
    """读 CSV 全量列，按 (episode_index, frame_index) 对齐补 frame_weight 列。"""
    weights = compute_weights_from_parquet(parquet_path, detect_kwargs)
    wdf = pd.read_csv(str(csv_path))
    _, df, _ = load_parquet_frames(parquet_path)
    key = pd.DataFrame({
        "episode_index": df["episode_index"].to_numpy(),
        "frame_index": df["frame_index"].to_numpy(),
        "frame_weight": weights,
    })
    wdf = wdf.merge(key, on=["episode_index", "frame_index"], how="left",
                    validate="one_to_one")
    missing = int(wdf["frame_weight"].isna().sum())
    if missing:
        raise RuntimeError(f"CSV {missing} 行未对齐到 parquet key")
    target = out_path or csv_path
    wdf.to_csv(target, index=False)
    return int((weights == WEIGHT_HIT).sum()), int((weights == WEIGHT_BASE).sum())


def write_parquet(parquet_path: Path, detect_kwargs: dict,
                  out_path: Path | None = None) -> tuple[int, int]:
    """读 parquet 全表，追加 frame_weight 列写回（保留 schema metadata）。"""
    weights = compute_weights_from_parquet(parquet_path, detect_kwargs)
    table = pq.read_table(str(parquet_path))
    table = table.append_column(
        "frame_weight", pa.array(weights, type=pa.float32())
    )
    if table.schema.metadata is None:
        raise RuntimeError("parquet schema metadata 丢失，中止以防破坏 LeRobot 读取")
    target = out_path or parquet_path
    tmp = target.with_name(target.name + ".tmp")
    pq.write_table(table, str(tmp))
    os.replace(tmp, target)
    return int((weights == WEIGHT_HIT).sum()), int((weights == WEIGHT_BASE).sum())


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def per_task_stats(parquet_path: Path, detect_kwargs: dict) -> str:
    _, df, _ = load_parquet_frames(parquet_path)
    weights = compute_weights_from_parquet(parquet_path, detect_kwargs)
    stats = df.assign(frame_weight=weights)
    lines = [f"{'task':>4} {'episodes':>8} {'frames':>8} {'hit(1.5)':>9} "
             f"{'pct':>6}"]
    total_hit = total = 0
    for ti in sorted(df["task_index"].unique()):
        sub = stats[stats["task_index"] == ti]
        hit = int((sub["frame_weight"] == WEIGHT_HIT).sum())
        n = len(sub)
        total_hit += hit
        total += n
        lines.append(f"{ti:>4} {sub['episode_index'].nunique():>8} {n:>8} "
                     f"{hit:>9} {100.0*hit/n:>5.1f}%")
    lines.append(f"总计: {total} 帧, hit(1.5)={total_hit} "
                 f"({100.0*total_hit/total:.1f}%), 其余为 1.0")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=["csv", "parquet"], default=None,
                        help="写入 CSV 还是 parquet（--stats-only 时可不填）")
    parser.add_argument("--csv", default=str(DEFAULT_CSV),
                        help="CSV 路径（仅 target=csv 时用）")
    parser.add_argument("--parquet", default=str(DEFAULT_PARQUET),
                        help="parquet 路径（权重计算源 + target=parquet 写入目标）")
    parser.add_argument("--out", default=None,
                        help="输出路径; 缺省则覆盖原文件")
    parser.add_argument("--min-prominence", type=float, default=0.2,
                        help="运动段幅度下限(与 keyframe_detect 一致)")
    parser.add_argument("--hold-min-len", type=int, default=3,
                        help="水平保持段最少帧数")
    parser.add_argument("--open-level", type=float, default=0.9,
                        help="视作打开的阈值")
    parser.add_argument("--no-incomplete", action="store_true",
                        help="不标记结尾未回开(不完整)的抓取周期")
    parser.add_argument("--stats-only", action="store_true",
                        help="只打印每任务权重统计，不写任何文件")
    args = parser.parse_args()

    detect_kwargs = {
        "min_prominence": args.min_prominence,
        "hold_min_len": args.hold_min_len,
        "open_level": args.open_level,
        "allow_incomplete": not args.no_incomplete,
    }
    parquet_path = Path(args.parquet)
    print(per_task_stats(parquet_path, detect_kwargs))
    print()
    if args.stats_only:
        return

    if args.target is None:
        raise SystemExit("--stats-only 已执行; 需要 --target csv|parquet 才写文件")

    if args.target == "csv":
        hit, base = write_csv(Path(args.csv), parquet_path, detect_kwargs,
                              Path(args.out) if args.out else None)
        out = args.out or args.csv
        print(f"csv -> {out}: {hit} 帧 weight=1.5, {base} 帧 weight=1.0")
    else:
        hit, base = write_parquet(parquet_path, detect_kwargs,
                                  Path(args.out) if args.out else None)
        out = args.out or args.parquet
        print(f"parquet -> {out}: {hit} 帧 weight=1.5, {base} 帧 weight=1.0")


if __name__ == "__main__":
    main()
