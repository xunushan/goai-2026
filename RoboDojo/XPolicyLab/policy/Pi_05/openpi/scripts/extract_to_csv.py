"""从 hdf5 提取全部数值列到 CSV（后续 joint/ee 数据校对的权威基准）。

CSV 每行 = 一帧，列：
  task, episode_index, frame_index,
  left_joint_0..5, left_gripper, right_joint_0..5, right_gripper,   # joint state(14)
  left_eef_x/y/z/qw/qx/qy/qz, right_eef_x/y/z/qw/qx/qy/qz           # ee state(14, gripper 复用)

注意：6 维 eef 任务在提取时按 xyz 内旋转四元数（eef_to_pose），CSV 存转换后四元数。
"""
import argparse
import csv
import os
import sys
from pathlib import Path

import h5py
import numpy as np

from process_data import SCHEMAS, eef_to_pose, _extract_arm_parts

ROOT_PATH = Path(__file__).parent.parent.parent.parent.parent
DATA_ROOT = Path(os.environ.get("XDATA_ROOT", str(ROOT_PATH / "data")))

CSV_COLUMNS = (
    ["task", "episode_index", "frame_index"]
    + [f"left_joint_{i}" for i in range(6)]
    + ["left_gripper"]
    + [f"right_joint_{i}" for i in range(6)]
    + ["right_gripper"]
    + [f"left_eef_{n}" for n in ["x", "y", "z", "qw", "qx", "qy", "qz"]]
    + [f"right_eef_{n}" for n in ["x", "y", "z", "qw", "qx", "qy", "qz"]]
)


def extract_episode(ep_path: Path, schema: dict, task: str, ep_index: int, writer) -> int:
    """读取单个 episode，逐帧写 CSV。返回帧数。"""
    with h5py.File(ep_path, "r") as ep:
        left_joint = _extract_arm_parts(ep, schema["state_parts_joint"][0])  # (T,7) joint+gripper
        right_joint = _extract_arm_parts(ep, schema["state_parts_joint"][1])
        left_ee = _extract_arm_parts(ep, schema["state_parts_ee"][0])  # (T,8) eef+gripper
        right_ee = _extract_arm_parts(ep, schema["state_parts_ee"][1])

    # joint: [joint(6), gripper] 拼接为 [left_joint0..5, left_gripper, right_joint0..5, right_gripper]
    left_j, left_g = left_joint[:, :6], left_joint[:, 6:7]
    right_j, right_g = right_joint[:, :6], right_joint[:, 6:7]
    # ee: eef_to_pose 已转四元数，eef(7) + gripper(1) → 每臂 ee(8)，取 eef 前 7 列
    left_e, right_e = left_ee[:, :7], right_ee[:, :7]

    n = left_joint.shape[0]
    for i in range(n):
        row = (
            [task, ep_index, i]
            + [f"{v:.7g}" for v in left_j[i]]
            + [f"{left_g[i, 0]:.7g}"]
            + [f"{v:.7g}" for v in right_j[i]]
            + [f"{right_g[i, 0]:.7g}"]
            + [f"{v:.7g}" for v in left_e[i]]
            + [f"{v:.7g}" for v in right_e[i]]
        )
        writer.writerow(row)
    return n


def main():
    parser = argparse.ArgumentParser(description="Extract hdf5 numeric columns to CSV")
    parser.add_argument("bench_name", type=str, default="real")
    parser.add_argument("env_cfg_type", type=str, default="piper_x")
    parser.add_argument("output", type=str, help="Output CSV path")
    parser.add_argument(
        "raw_task_dirs", type=str, nargs="?", default=None,
        help="Comma-separated task dirs under data/<bench>/<task>/<env_cfg_type>; default: all dirs.",
    )
    parser.add_argument(
        "--max_episodes", type=int, default=None,
        help="Only extract first N episodes per task (small-scale verification).",
    )
    args = parser.parse_args()

    schema = SCHEMAS[(args.bench_name, args.env_cfg_type)]
    task_root = DATA_ROOT / args.bench_name
    if args.raw_task_dirs:
        task_dirs = [d for d in (args.raw_task_dirs or "").split(",") if d]
    else:
        task_dirs = sorted(p.name for p in task_root.iterdir() if (task_root / p.name / args.env_cfg_type).is_dir())

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total_frames = 0
    total_episodes = 0
    epi_counter = 0  # 全局 episode_index（与 process_data 的 save_episode 全局递增语义一致）
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_COLUMNS)
        for task in task_dirs:
            data_dir = task_root / task / args.env_cfg_type / "data"
            ep_files = sorted(data_dir.glob("episode_*.hdf5"))
            if not ep_files:
                print(f"[warn] no episodes under {data_dir}")
                continue
            if args.max_episodes is not None:
                ep_files = ep_files[: args.max_episodes]
            for ep_file in ep_files:
                n = extract_episode(ep_file, schema, task, epi_counter, writer)
                epi_counter += 1
                total_frames += n
                total_episodes += 1
                if total_episodes % 50 == 0:
                    print(f"  {total_episodes} episodes, {total_frames} frames", flush=True)
            print(f"[done] task={task}: {len(ep_files)} episodes", flush=True)

    print(f"CSV written: {out_path} ({total_episodes} episodes, {total_frames} frames, {len(CSV_COLUMNS)} cols)")


if __name__ == "__main__":
    main()
