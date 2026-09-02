"""验证本地 lerobot 数据集 CSV 与权威基准 CSV (extract 自 hdf5) 的一致性。

任务 A (逐帧对比): 每任务随机选 per-task 个 episode, 逐帧对比 state/action 每个维度。
任务 B (视频帧数):  统计已下载视频覆盖 episode 的三路相机帧数, 与基准 CSV 帧数对比。

约定:
- 基准 CSV (sim_all.csv 等) 为权威提取, episode_index 全局递增。
- 本地 CSV observation.state 16 维 = [left x,y,z,qw,qx,qy,qz,g, right ...]。
- action[i] == state[i+1]（末帧 action=自身 state）。
- 基准 ee 列为 7 维四元数 (x,y,z,qw,qx,qy,qz), 四元数顺序 wxyz (w 在首位)。
  sim 及后续更新的数据集均为该表示, 不支持 6 维欧拉角。

用法:
    # sim 数据集 (3 任务)
    conda run -n lerobot python scripts/verify_dataset_vs_csv.py \
        --ref-csv data/sim_all.csv --local-csv data/sim_lerobot_v30_ee.csv \
        --data-root data/sim_lerobot_v30_ee \
        --config configs/sim_lerobot_v30_ee.json --seed 42 --per-task 3

    # real 数据集 (6 任务, 同样为 7 维四元数 wxyz)
    conda run -n lerobot python scripts/verify_dataset_vs_csv.py \
        --ref-csv data/real_all.csv --local-csv data/real_lerobot_v30_ee.csv \
        --data-root data/real_lerobot_v30_ee \
        --config configs/real_lerobot_v30_ee.json --seed 42 --per-task 3
"""
from __future__ import annotations

import argparse
import ast
import json
import random
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

# 每个 dataset 的 ee 四元数列 (x,y,z,qw,qx,qy,qz,g per arm, 与本地 state 顺序一致)
# 四元数顺序 wxyz (qw 在首位)
EE_COLS = (["left_eef_x", "left_eef_y", "left_eef_z",
            "left_eef_qw", "left_eef_qx", "left_eef_qy", "left_eef_qz", "left_gripper"] +
           ["right_eef_x", "right_eef_y", "right_eef_z",
            "right_eef_qw", "right_eef_qx", "right_eef_qy", "right_eef_qz", "right_gripper"])


def parse_state(v: object) -> list[float]:
    return list(v) if isinstance(v, (list, np.ndarray)) else ast.literal_eval(v)


def load_local_csv(local_csv: Path, episodes: set[int]) -> pd.DataFrame:
    """加载本地 CSV 中指定 episode 的帧 (state/action/task/视频时间戳)。"""
    print(f"[load] 读取本地 CSV {local_csv.name} 指定列 ...")
    df = pd.read_csv(
        local_csv,
        usecols=["episode_index", "frame_index", "task_index",
                 "observation.state", "action",
                 "high_video_from_timestamp", "high_video_to_timestamp",
                 "left_video_from_timestamp", "left_video_to_timestamp",
                 "right_video_from_timestamp", "right_video_to_timestamp",
                 "high_video_path", "left_video_path", "right_video_path"],
    )
    df = df[df["episode_index"].isin(episodes)]
    df["observation.state"] = df["observation.state"].apply(ast.literal_eval)
    df["action"] = df["action"].apply(ast.literal_eval)
    return df


def derive_task_ranges(ref: pd.DataFrame) -> dict[int, tuple[int, str]]:
    """从基准 CSV 按任务分组, 返回 {task_index: (min_ep, task_name)}。

    task_index 按任务的 min episode 升序编号 (与 lerobot task_index 一致)。
    """
    grp = ref.groupby("task")["episode_index"].agg(["min", "max"])
    grp = grp.sort_values("min")
    return {i: (int(r["min"]), int(r["max"]), name) for i, (name, r) in enumerate(grp.iterrows())}


def verify_task_a(ref_csv: Path, local_csv: Path, seed: int, per_task: int) -> None:
    """每任务随机选 per_task 个 episode, 逐帧对比 state/action 与基准 CSV。"""
    print("=" * 76)
    print(f"[任务A] 每任务随机选 {per_task} 个 episode, 逐帧对比 state/action (seed={seed})")
    print("=" * 76)

    ref = pd.read_csv(ref_csv)  # 全量列
    task_map = derive_task_ranges(ref)
    print("任务范围:", {t: (name, lo, hi) for t, (lo, hi, name) in task_map.items()})

    rng = random.Random(seed)
    selected: dict[int, list[int]] = {}
    all_eps: set[int] = set()
    for tidx, (lo, hi, _name) in task_map.items():
        picked = sorted(rng.sample(list(range(lo, hi + 1)), per_task))
        selected[tidx] = picked
        all_eps.update(picked)
    print("选中 episodes:", {t: e for t, e in selected.items()})

    local = load_local_csv(local_csv, all_eps)
    local = local.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)

    # 准备每 episode 的 ee 数值矩阵 (基准侧, 均为 7 维四元数 wxyz)
    ref_ee = ref[["episode_index", "frame_index"] + EE_COLS].copy()

    grand_s_mis = grand_a_mis = 0
    max_s = max_a = 0.0
    n_frames = 0

    for tidx in sorted(selected):
        for ep in selected[tidx]:
            lrow = local[local["episode_index"] == ep].sort_values("frame_index").reset_index(drop=True)
            rrow = ref_ee[ref_ee["episode_index"] == ep].sort_values("frame_index").reset_index(drop=True)

            n_local, n_real = len(lrow), len(rrow)
            frames_ok = n_local == n_real
            # 任务核对: 本地 task_index vs 基准任务编号
            task_ok = bool(len(lrow)) and int(lrow["task_index"].iloc[0]) == tidx
            task_name = task_map[tidx][2]

            s_mis = a_mis = 0
            s_max = a_max = 0.0
            for i in range(min(n_local, n_real)):
                rv = rrow[EE_COLS].iloc[i].to_numpy(dtype=float)
                lv = np.array(parse_state(lrow["observation.state"].iloc[i]), dtype=float)
                d = np.abs(lv - rv)
                s_max = max(s_max, d.max())
                s_mis += int((d > 1e-4).any())
                # action: 本帧 == 后一帧 state; 末帧 == 自身 state
                ns = rrow[EE_COLS].iloc[i + 1].to_numpy(dtype=float) if i + 1 < n_real else rv
                av = np.array(parse_state(lrow["action"].iloc[i]), dtype=float)
                ad = np.abs(av - ns)
                a_max = max(a_max, ad.max())
                a_mis += int((ad > 1e-4).any())

            n_frames += n_local
            ok = frames_ok and task_ok and s_mis == 0 and a_mis == 0
            print(f"  task{tidx}({task_name}) ep{ep:03d}: frames local={n_local} ref={n_real} "
                  f"task={task_ok} | state mis={s_mis} max={s_max:.2e} | "
                  f"action mis={a_mis} max={a_max:.2e}  [{'OK' if ok else 'FAIL'}]")
            grand_s_mis += s_mis
            grand_a_mis += a_mis
            max_s = max(max_s, s_max)
            max_a = max(max_a, a_max)

    print("-" * 76)
    print(f"[任务A 汇总] {sum(len(v) for v in selected.values())} episodes / {n_frames} 帧")
    print(f"  state  mismatch={grand_s_mis}  max_abs_diff={max_s:.2e}")
    print(f"  action mismatch={grand_a_mis}  max_abs_diff={max_a:.2e}")
    print(f"  结论: {'PASS' if grand_s_mis == 0 and grand_a_mis == 0 else 'FAIL'}")


def count_video_frames(video_path: Path, from_ts: float, to_ts: float) -> int:
    """用 ffmpeg 从 from 到 to 解码计数帧数 (逐帧读取, 不重编码)。"""
    cmd = ["ffmpeg", "-nostdin", "-ss", f"{from_ts:.6f}", "-to", f"{to_ts:.6f}",
           "-i", str(video_path), "-map", "0:v", "-f", "null", "-"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        return -1
    import re
    frames = re.findall(r"frame=\s*(\d+)", r.stderr)
    return int(frames[-1]) if frames else -1


def verify_task_b(ref_csv: Path, local_csv: Path, data_root: Path, config_path: Path | None) -> None:
    """统计已下载视频覆盖 episode 的三路帧数, 与基准 CSV 对比。"""
    print()
    print("=" * 76)
    print("[任务B] 已下载视频覆盖 episode 的三路帧数 vs 基准 CSV")
    print("=" * 76)

    eps: list[int] = []
    if config_path and config_path.exists():
        eps = json.loads(config_path.read_text()).get("episodes", [])
    if not eps:
        df = pd.read_csv(local_csv, usecols=["episode_index"])
        eps = sorted(df["episode_index"].unique())

    local = load_local_csv(local_csv, set(eps))
    real = pd.read_csv(ref_csv, usecols=["task", "episode_index", "frame_index"])
    real_len = real.groupby("episode_index").size()

    print(f"共 {len(eps)} 个 episode, 三路视频逐路解码计数 ...")
    bad = 0
    for ep in eps:
        row = local[local["episode_index"] == ep].iloc[0]
        rlen = int(real_len.get(ep, -1))
        cam_results = []
        for cam in ["high", "left", "right"]:
            path = data_root / row[f"{cam}_video_path"]
            frm = float(row[f"{cam}_video_from_timestamp"])
            to = float(row[f"{cam}_video_to_timestamp"])
            n = count_video_frames(path, frm, to) if path.exists() else -2
            cam_results.append(n)
        ok = all(n == rlen for n in cam_results if n >= 0)
        bad += 0 if ok else 1
        print(f"  ep{ep:03d}: ref={rlen} | high={cam_results[0]} "
              f"left={cam_results[1]} right={cam_results[2]}  [{'OK' if ok else 'FAIL'}]")

    print("-" * 76)
    print(f"[任务B 汇总] {len(eps)} episodes, {bad} 个不一致")
    print(f"  结论: {'PASS' if bad == 0 else 'FAIL'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref-csv", required=True, help="权威基准 CSV (real_all.csv / sim_all.csv)")
    parser.add_argument("--local-csv", required=True, help="本地生成的 lerobot CSV")
    parser.add_argument("--data-root", required=True, help="数据集根目录 (含 videos/, 任务B用)")
    parser.add_argument("--config", default=None, help="数据集配置文件 (任务B episodes 来源)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--per-task", type=int, default=3)
    parser.add_argument("--skip-a", action="store_true", help="跳过任务A")
    parser.add_argument("--skip-b", action="store_true", help="跳过任务B")
    args = parser.parse_args()

    if not args.skip_a:
        verify_task_a(Path(args.ref_csv), Path(args.local_csv), args.seed, args.per_task)
    if not args.skip_b:
        verify_task_b(Path(args.ref_csv), Path(args.local_csv), Path(args.data_root),
                      Path(args.config) if args.config else None)


if __name__ == "__main__":
    main()
