"""验证本地 lerobot 数据集 CSV 与权威基准 CSV (extract 自 hdf5) 的一致性。

检查1 CSV 逐帧一致性: 确认 lerobot 导出的本地 CSV 与权威基准 CSV 完全一致。
    不依赖视频, 默认对 CSV 中的全部 episode 全量逐帧对比 state/action 每个维度
    (含帧数/任务归属); 开发期可用 --per-task N 每任务随机抽 N 个 episode 快速抽查。
检查2 视频帧数核对:   确认视频实际帧数与 CSV 中该 episode 帧数一致 (需要视频)。
    对已下载视频覆盖的 episode, 统计三路相机解码帧数, 与基准 CSV 帧数对比。

约定:
- 基准 CSV (sim_all.csv 等) 为权威提取, episode_index 全局递增。
- 本地 CSV observation.state 16 维 = [left x,y,z,qw,qx,qy,qz,g, right ...]。
- action[i] == state[i+1]（末帧 action=自身 state）。
- 基准 ee 列为 7 维四元数 (x,y,z,qw,qx,qy,qz), 四元数顺序 wxyz (w 在首位)。
  sim 及后续更新的数据集均为该表示, 不支持 6 维欧拉角。

用法:
    # sim 数据集: 全量 CSV 对比 (检查1) + 视频帧数核对 (检查2, 需已下载视频)
    conda run -n lerobot python scripts/verify_dataset_vs_csv.py \
        --ref-csv data/sim_all.csv --local-csv data/sim_lerobot_v30_ee.csv \
        --data-root data/sim_lerobot_v30_ee \
        --config configs/sim_lerobot_v30_ee.json

    # real 数据集 (6 任务, 同样为 7 维四元数 wxyz)
    conda run -n lerobot python scripts/verify_dataset_vs_csv.py \
        --ref-csv data/real_all.csv --local-csv data/real_lerobot_v30_ee.csv \
        --data-root data/real_lerobot_v30_ee \
        --config configs/real_lerobot_v30_ee.json

    # 只做 CSV 全量对比 (不碰视频): 加 --skip-video-check
    conda run -n lerobot python scripts/verify_dataset_vs_csv.py \
        --ref-csv data/sim_all.csv --local-csv data/sim_lerobot_v30_ee.csv \
        --skip-video-check
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


def verify_csv_consistency(ref_csv: Path, local_csv: Path,
                           per_task: int | None = None, seed: int = 42) -> None:
    """检查1: 逐帧对比本地 CSV 的 state/action 与权威基准 CSV (纯数据, 不依赖视频)。

    默认对 CSV 中的全部 episode 做全量逐帧对比; 传 --per-task N 时每任务随机
    抽 N 个 episode 做快速抽查 (供开发期 smoke)。
    """
    print("=" * 76)
    print("[检查1 · CSV逐帧一致性] 本地 CSV vs 权威基准 CSV (不依赖视频)")
    print("=" * 76)

    ref = pd.read_csv(ref_csv)  # 全量列
    task_map = derive_task_ranges(ref)
    print("任务范围:", {t: (name, lo, hi) for t, (lo, hi, name) in task_map.items()})

    if per_task:
        rng = random.Random(seed)
        selected: dict[int, list[int]] = {}
        for tidx, (lo, hi, _name) in task_map.items():
            selected[tidx] = sorted(rng.sample(list(range(lo, hi + 1)), per_task))
        eps_all = sorted(e for v in selected.values() for e in v)
        print(f"抽查模式: 每任务随机选 {per_task} 个 episode -> {sorted(selected.items())}")
    else:
        # 全量: 对比基准 CSV 中的全部 episode (本地 CSV 应有同样全量数据)
        selected = {t: list(range(lo, hi + 1)) for t, (lo, hi, _n) in task_map.items()}
        eps_all = sorted(e for v in selected.values() for e in v)
        print(f"全量模式: 对比全部 {len(eps_all)} 个 episode (每任务 {len(eps_all)//max(len(task_map),1)} 个)")

    local = load_local_csv(local_csv, set(eps_all))
    local = local.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)

    # 基准 ee 数值矩阵 (7 维四元数 wxyz + gripper)
    ref_ee = ref[["episode_index", "frame_index"] + EE_COLS].copy()

    # 一次解析本地 state/action 为 (N,16) 矩阵
    l_ep = local["episode_index"].to_numpy()
    l_state = np.vstack([np.asarray(parse_state(v), dtype=float) for v in local["observation.state"]])
    l_action = np.vstack([np.asarray(parse_state(v), dtype=float) for v in local["action"]])

    grand_s_mis = grand_a_mis = grand_frame_mis = 0
    max_s = max_a = 0.0
    n_frames = 0

    for tidx in sorted(selected):
        task_name = task_map[tidx][2]
        for ep in selected[tidx]:
            lmask = l_ep == ep
            n_local = int(lmask.sum())
            lstate = l_state[lmask]
            laction = l_action[lmask]

            rrow = ref_ee[ref_ee["episode_index"] == ep].sort_values("frame_index")
            rstate = rrow[EE_COLS].to_numpy(dtype=float) if len(rrow) else np.empty((0, 16))
            n_real = len(rstate)

            n_cmp = min(n_local, n_real)
            frames_ok = n_local == n_real
            grand_frame_mis += int(not frames_ok)

            if n_cmp:
                # state: 本地 state vs 基准 ee
                d = np.abs(lstate[:n_cmp] - rstate[:n_cmp])
                s_mis = int((d > 1e-4).any(axis=1).sum())
                s_max = float(d.max())
                # action: 本帧 == 后一帧 state; 末帧 == 自身 state
                nxt = np.empty_like(rstate)
                nxt[:-1] = rstate[1:]
                nxt[-1] = rstate[-1]
                ad = np.abs(laction[:n_cmp] - nxt[:n_cmp])
                a_mis = int((ad > 1e-4).any(axis=1).sum())
                a_max = float(ad.max())
            else:
                s_mis = a_mis = 0
                s_max = a_max = 0.0

            # 任务核对: 本地 task_index vs 基准任务编号
            task_ok = n_local > 0 and int(local.loc[lmask, "task_index"].iloc[0]) == tidx

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
    print(f"[检查1 汇总] {len(eps_all)} episodes / {n_frames} 帧")
    print(f"  state  mismatch={grand_s_mis}  max_abs_diff={max_s:.2e}")
    print(f"  action mismatch={grand_a_mis}  max_abs_diff={max_a:.2e}")
    print(f"  结论: {'PASS' if grand_s_mis == 0 and grand_a_mis == 0 and grand_frame_mis == 0 else 'FAIL'}")


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


def verify_video_frame_count(ref_csv: Path, local_csv: Path, data_root: Path,
                             config_path: Path | None = None) -> None:
    """检查2: 统计已下载视频覆盖 episode 的三路帧数, 与基准 CSV 该 episode 帧数对比。"""
    print()
    print("=" * 76)
    print("[检查2 · 视频帧数核对] 视频实际帧数 vs 基准 CSV 该 episode 帧数 (需要视频)")
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
    print(f"[检查2 汇总] {len(eps)} episodes, {bad} 个不一致")
    print(f"  结论: {'PASS' if bad == 0 else 'FAIL'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref-csv", required=True, help="权威基准 CSV (real_all.csv / sim_all.csv)")
    parser.add_argument("--local-csv", required=True, help="本地生成的 lerobot CSV")
    parser.add_argument("--data-root", default=None,
                        help="数据集根目录 (含 videos/); 仅检查2视频帧数核对需要")
    parser.add_argument("--config", default=None,
                        help="数据集配置文件 (检查2的 episodes 来源; 缺省取本地 CSV 全部 episode)")
    parser.add_argument("--per-task", type=int, default=None,
                        help="检查1抽查: 每任务随机抽 N 个 episode (缺省为全量逐帧对比)")
    parser.add_argument("--seed", type=int, default=42, help="抽查模式的随机种子")
    parser.add_argument("--skip-csv-check", action="store_true", help="跳过检查1 (CSV逐帧一致性)")
    parser.add_argument("--skip-video-check", action="store_true", help="跳过检查2 (视频帧数核对)")
    args = parser.parse_args()

    if not args.skip_csv_check:
        verify_csv_consistency(Path(args.ref_csv), Path(args.local_csv),
                               args.per_task, args.seed)
    if not args.skip_video_check:
        if not args.data_root:
            parser.error("检查2 (视频帧数核对) 需要 --data-root 指向含 videos/ 的数据集目录; "
                         "只做 CSV 对比请加 --skip-video-check")
        verify_video_frame_count(Path(args.ref_csv), Path(args.local_csv),
                                 Path(args.data_root),
                                 Path(args.config) if args.config else None)


if __name__ == "__main__":
    main()
