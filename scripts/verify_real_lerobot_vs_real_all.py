"""验证本地下载的 real_lerobot_v30_ee 数据集与权威基准 real_all.csv 的一致性。

任务 2：每个任务随机选 3 个 episode，逐帧对比 state/action 的每个维度。
任务 3：统计已下载视频覆盖的 episode 三路相机帧数，与 real_all.csv 对比。

约定：
- real_all.csv 为 hdf5 权威提取，episode_index 全局递增 0..599。
- 本地 CSV observation.state 16 维 = [left x,y,z,qw,qx,qy,qz,g, right ...]。
- action[i] == state[i+1]（末帧 action=自身 state）。
- real_all ee 列已是 7 维四元数（x,y,z,qw,qx,qy,qz）；若遇到 6 维欧拉角用 eef_to_pose 转换。

用法：
    conda run -n lerobot python scripts/verify_real_lerobot_vs_real_all.py \
        --seed 42 --per-task 3 --config configs/real_lerobot_v30_ee.json
"""
from __future__ import annotations

import argparse
import ast
import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REAL_ALL_CSV = ROOT / "data" / "real_all.csv"
LOCAL_CSV = ROOT / "data" / "real_lerobot_v30_ee.csv"

TASK_RANGES = {
    0: (0, 99),      # fill_pen_holder
    1: (100, 199),   # put_objects_into_basket
    2: (200, 299),   # stack_and_cover_blocks
    3: (300, 399),   # stack_bowls
    4: (400, 499),   # stand_up_bottles
    5: (500, 599),   # insert_charger
}

EE_COLS = (["left_eef_x", "left_eef_y", "left_eef_z",
            "left_eef_qw", "left_eef_qx", "left_eef_qy", "left_eef_qz", "left_gripper"] +
           ["right_eef_x", "right_eef_y", "right_eef_z",
            "right_eef_qw", "right_eef_qx", "right_eef_qy", "right_eef_qz", "right_gripper"])


def eef_to_pose(eef: np.ndarray) -> np.ndarray:
    """real eef 统一为 [x,y,z,qw,qx,qy,qz]。

    - 7 维：已是四元数，直接返回。
    - 6 维：[x,y,z, 欧拉角×3]。欧拉角顺序为 xyz 内旋。
    """
    if eef.shape[1] == 7:
        return eef
    if eef.shape[1] == 6:
        from scipy.spatial.transform import Rotation
        pos = eef[:, :3]
        q_xyzw = Rotation.from_euler("xyz", eef[:, 3:6], degrees=False).as_quat()
        return np.concatenate([pos, q_xyzw[:, 3:4], q_xyzw[:, :3]], axis=1)
    raise ValueError(f"eef dims {eef.shape[1]} not supported")


def load_local_csv(episodes: set[int]) -> pd.DataFrame:
    """加载本地 CSV 中指定 episode 的帧 (state/action/task)。"""
    print(f"[load] 读取本地 CSV 指定列 ...")
    df = pd.read_csv(
        LOCAL_CSV,
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


def parse_state(v: object) -> list[float]:
    return list(v) if isinstance(v, (list, np.ndarray)) else ast.literal_eval(v)


def verify_task2(seed: int, per_task: int) -> None:
    """每个任务随机选 per_task 个 episode，逐帧对比 state/action 与 real_all。"""
    print("=" * 72)
    print(f"[任务2] 每个任务随机选 {per_task} 个 episode, 逐帧对比 state/action (seed={seed})")
    print("=" * 72)

    real = pd.read_csv(REAL_ALL_CSV)  # 31 列全量
    rng = random.Random(seed)
    selected: dict[int, list[int]] = {}
    all_eps: set[int] = set()
    for tidx, (lo, hi) in TASK_RANGES.items():
        picked = sorted(rng.sample(list(range(lo, hi + 1)), per_task))
        selected[tidx] = picked
        all_eps.update(picked)
    print("选中 episodes:", {t: e for t, e in selected.items()})

    local = load_local_csv(all_eps)
    local = local.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)

    grand_state_mismatch = 0
    grand_action_mismatch = 0
    max_state_diff = 0.0
    max_action_diff = 0.0

    for tidx in sorted(selected):
        for ep in selected[tidx]:
            lrow = local[local["episode_index"] == ep]
            rrow = real[real["episode_index"] == ep]
            lrow = lrow.sort_values("frame_index").reset_index(drop=True)
            rrow = rrow.sort_values("frame_index").reset_index(drop=True)

            n_local = len(lrow)
            n_real = len(rrow)
            frames_ok = n_local == n_real
            # 任务名核对
            task_name = rrow["task"].iloc[0] if n_real else "?"
            if n_local:
                lt = int(lrow["task_index"].iloc[0])
                task_ok = (lt == tidx)
            else:
                task_ok = False

            # state 对比 (16 维)
            s_mismatch = 0
            s_max = 0.0
            a_mismatch = 0
            a_max = 0.0
            for i in range(min(n_local, n_real)):
                rv = rrow[EE_COLS].iloc[i].to_numpy(dtype=float)
                lv = np.array(parse_state(lrow["observation.state"].iloc[i]), dtype=float)
                d = np.abs(lv - rv)
                s_max = max(s_max, d.max())
                s_mismatch += int((d > 1e-4).any())
                # action: 本帧 == 后一帧 state; 末帧 == 自身 state
                next_state = rrow[EE_COLS].iloc[i + 1].to_numpy(dtype=float) if i + 1 < n_real else rv
                av = np.array(parse_state(lrow["action"].iloc[i]), dtype=float)
                ad = np.abs(av - next_state)
                a_max = max(a_max, ad.max())
                a_mismatch += int((ad > 1e-4).any())

            status = "OK" if (frames_ok and task_ok and s_mismatch == 0 and a_mismatch == 0) else "FAIL"
            print(f"  task{tidx}({task_name}) ep{ep:03d}: frames local={n_local} real={n_real} "
                  f"task={task_ok} | state mismatch={s_mismatch} max={s_max:.2e} | "
                  f"action mismatch={a_mismatch} max={a_max:.2e}  [{status}]")
            grand_state_mismatch += s_mismatch
            grand_action_mismatch += a_mismatch
            max_state_diff = max(max_state_diff, s_max)
            max_action_diff = max(max_action_diff, a_max)

    print("-" * 72)
    print(f"[任务2 汇总] 共 {sum(len(v) for v in selected.values())} 个 episode / "
          f"{len(local)} 帧")
    print(f"  state  mismatch={grand_state_mismatch}  max_abs_diff={max_state_diff:.2e}")
    print(f"  action mismatch={grand_action_mismatch}  max_abs_diff={max_action_diff:.2e}")
    verdict = "PASS" if (grand_state_mismatch == 0 and grand_action_mismatch == 0) else "FAIL"
    print(f"  结论: {verdict}")


def count_video_frames(video_path: Path, from_ts: float, to_ts: float) -> int:
    """用 ffmpeg 从 from 到 to 解码计数帧数 (逐帧读取, 不重编码)。"""
    cmd = [
        "ffmpeg", "-nostdin", "-ss", f"{from_ts:.6f}", "-to", f"{to_ts:.6f}",
        "-i", str(video_path), "-map", "0:v", "-f", "null", "-",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        return -1
    # 从 stderr 解析 frame=  计数
    import re
    frames = re.findall(r"frame=\s*(\d+)", r.stderr)
    return int(frames[-1]) if frames else -1


def verify_task3(config_path: Path | None) -> None:
    """统计已下载视频覆盖 episode 的三路帧数，与 real_all.csv 对比。"""
    print()
    print("=" * 72)
    print("[任务3] 已下载视频覆盖 episode 的三路帧数 vs real_all.csv")
    print("=" * 72)

    eps: list[int] = []
    if config_path and config_path.exists():
        cfg = json.loads(config_path.read_text())
        eps = cfg.get("episodes", [])
    if not eps:
        # 兜底: 从本地 CSV 中有视频的 episode 里取前若干
        df = pd.read_csv(LOCAL_CSV, usecols=["episode_index"])
        eps = sorted(df["episode_index"].unique())

    local = load_local_csv(set(eps))
    real = pd.read_csv(REAL_ALL_CSV, usecols=["task", "episode_index", "frame_index"])
    real_len = real.groupby("episode_index").size()

    print(f"共 {len(eps)} 个 episode, 三路视频逐路解码计数 ...")
    bad = 0
    for ep in eps:
        row = local[local["episode_index"] == ep].iloc[0]
        rlen = int(real_len.get(ep, -1))
        cam_results = []
        for cam in ["high", "left", "right"]:
            path = ROOT / "data" / "real_lerobot_v30_ee" / row[f"{cam}_video_path"]
            frm = float(row[f"{cam}_video_from_timestamp"])
            to = float(row[f"{cam}_video_to_timestamp"])
            n = count_video_frames(path, frm, to) if path.exists() else -2
            cam_results.append(n)
        ok = all(n == rlen for n in cam_results if n >= 0)
        status = "OK" if ok else "FAIL"
        if not ok:
            bad += 1
        print(f"  ep{ep:03d}: real_all={rlen} | high={cam_results[0]} "
              f"left={cam_results[1]} right={cam_results[2]}  [{status}]")

    print("-" * 72)
    print(f"[任务3 汇总] {len(eps)} episodes, {bad} 个不一致")
    print(f"  结论: {'PASS' if bad == 0 else 'FAIL'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--per-task", type=int, default=3)
    parser.add_argument("--config", default=str(ROOT / "configs" / "real_lerobot_v30_ee.json"))
    parser.add_argument("--skip-task2", action="store_true")
    parser.add_argument("--skip-task3", action="store_true")
    args = parser.parse_args()

    if not args.skip_task2:
        verify_task2(args.seed, args.per_task)
    if not args.skip_task3:
        verify_task3(Path(args.config))


if __name__ == "__main__":
    main()
