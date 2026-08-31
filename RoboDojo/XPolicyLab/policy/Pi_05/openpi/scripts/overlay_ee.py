"""在 joint 数据集基础上覆盖生成 ee 数据集（不重复编码视频）。

原理：
- joint 数据集已含完整视频（videos/），ee 的视频与之逐帧相同 → videos 软链接复用。
- 只重写 data parquet 的 state/action 列（14 维 joint → 16 维 ee，数值来自 CSV）。
- 重写 meta/info.json（features 16 维命名）与 meta/stats.json（数值部分重算）。
- meta/episodes、meta/tasks.parquet 与 joint 完全一致，直接复制。

校验：生成后必须用 verify_from_csv.py 以 CSV 为基准校验 ee parquet。
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EE_DIM_NAMES = ["x", "y", "z", "w", "wx", "wy", "wz", "g"]

# CSV 中 ee state 的 16 列（与 extract_to_csv.py 输出一致）
EE_STATE_COLS = (
    [f"left_eef_{n}" for n in ["x", "y", "z", "qw", "qx", "qy", "qz"]]
    + ["left_gripper"]
    + [f"right_eef_{n}" for n in ["x", "y", "z", "qw", "qx", "qy", "qz"]]
    + ["right_gripper"]
)
EE_MOTOR_NAMES = [f"l_{n}" for n in EE_DIM_NAMES] + [f"r_{n}" for n in EE_DIM_NAMES]


def _load_csv(csv_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """读 CSV → (episode_index, frame_index, ee_state(N,16), ee_action(N,16))。"""
    df = pd.read_csv(csv_path)
    for col in EE_STATE_COLS:
        assert col in df.columns, f"CSV missing column: {col}"
    epi = df["episode_index"].to_numpy()
    frame = df["frame_index"].to_numpy()
    state = df[EE_STATE_COLS].to_numpy(dtype=np.float32)

    # action = state 前移（per episode）：action[t]=state[t+1]，episode 末帧 action=自身 state
    action = np.empty_like(state)
    action[:-1] = state[1:]
    action[-1] = state[-1]
    # 位置 t 是 episode 末帧，当 t+1 是新的 episode 首帧（或 t 为全局最后一帧）
    epi_last = np.concatenate([epi[1:] != epi[:-1], [True]]).astype(bool)
    action[epi_last] = state[epi_last]
    return epi, frame, state, action


def _build_lookup(epi, frame, state, action):
    """(episode_index, frame_index) → (state16, action16) 字典。"""
    lut = {}
    for i in range(len(epi)):
        lut[(int(epi[i]), int(frame[i]))] = (state[i], action[i])
    return lut


def _compute_numeric_stats(state: np.ndarray, action: np.ndarray) -> dict:
    """按 lerobot stats.json 格式计算数值特征统计量（(C,1,1) shape）。"""
    def stats_for(arr: np.ndarray) -> dict:
        # arr: (N, C) float32
        arr = arr.astype(np.float64)
        qs = [0.01, 0.1, 0.5, 0.9, 0.99]
        out = {
            "count": float(len(arr)),
            "min": np.min(arr, axis=0).reshape(-1, 1, 1).tolist(),
            "max": np.max(arr, axis=0).reshape(-1, 1, 1).tolist(),
            "mean": np.mean(arr, axis=0).reshape(-1, 1, 1).tolist(),
            "std": np.std(arr, axis=0).reshape(-1, 1, 1).tolist(),
        }
        for q, name in zip(qs, ["q01", "q10", "q50", "q90", "q99"]):
            out[name] = np.percentile(arr, q * 100, axis=0).reshape(-1, 1, 1).tolist()
        return out

    return {"observation.state": stats_for(state), "action": stats_for(action)}


def main():
    parser = argparse.ArgumentParser(description="Overlay ee dataset on top of joint dataset")
    parser.add_argument("joint_dir", type=str, help="Joint dataset dir (with videos)")
    parser.add_argument("ee_dir", type=str, help="Output ee dataset dir")
    parser.add_argument("csv", type=str, help="CSV with ee numeric columns (extract_to_csv.py output)")
    parser.add_argument("--robot_type", type=str, default="piper_x")
    parser.add_argument("--fps", type=int, default=25)
    args = parser.parse_args()

    joint_dir = Path(args.joint_dir)
    ee_dir = Path(args.ee_dir)
    csv_path = Path(args.csv)
    assert joint_dir.is_dir(), f"joint dir missing: {joint_dir}"
    assert csv_path.is_file(), f"CSV missing: {csv_path}"

    # ---- 读 CSV 全量数值 ----
    print("[1/5] loading CSV...", flush=True)
    epi, frame, state, action = _load_csv(csv_path)
    lut = _build_lookup(epi, frame, state, action)
    print(f"  {len(state)} frames loaded", flush=True)

    # ---- 建立 ee 目录结构（排除 videos，软链接） ----
    print("[2/5] creating ee dir structure...", flush=True)
    if ee_dir.exists():
        shutil.rmtree(ee_dir)
    (ee_dir / "data").mkdir(parents=True)
    (ee_dir / "meta").mkdir(parents=True)
    # videos 软链接到 joint
    os.symlink(joint_dir / "videos", ee_dir / "videos")
    # episodes + tasks 复制（与 joint 一致）
    shutil.copytree(joint_dir / "meta" / "episodes", ee_dir / "meta" / "episodes")
    shutil.copy2(joint_dir / "meta" / "tasks.parquet", ee_dir / "meta" / "tasks.parquet")

    # ---- 遍历 joint 的 data chunk，替换 state/action ----
    print("[3/5] rewriting data parquets (joint→ee)...", flush=True)
    joint_chunks = sorted((joint_dir / "data").glob("chunk-*/file-*.parquet"))
    if not joint_chunks:
        raise SystemExit(f"no data parquet under {joint_dir / 'data'}")
    ee_chunks = []
    n_missing = 0
    for jc in joint_chunks:
        rel = jc.relative_to(joint_dir / "data")
        ec = ee_dir / "data" / rel
        ec.parent.mkdir(parents=True, exist_ok=True)
        df = pd.read_parquet(jc)
        keys = list(zip(df["episode_index"].to_numpy(), df["frame_index"].to_numpy()))
        states, actions = [], []
        for k in keys:
            hit = lut.get((int(k[0]), int(k[1])))
            if hit is None:
                n_missing += 1
                states.append(np.zeros(16, dtype=np.float32))
                actions.append(np.zeros(16, dtype=np.float32))
            else:
                states.append(hit[0])
                actions.append(hit[1])
        df["observation.state"] = [s.tolist() for s in states]
        df["action"] = [a.tolist() for a in actions]
        df.to_parquet(ec, index=False)
        ee_chunks.append(ec)
    if n_missing:
        print(f"  [warn] {n_missing} frames missing in CSV (filled zeros)", flush=True)

    # ---- info.json：16 维 features ----
    print("[4/5] writing info.json...", flush=True)
    with open(joint_dir / "meta" / "info.json") as f:
        info = json.load(f)
    feats = info["features"]
    feats["observation.state"] = {
        "dtype": "float32", "shape": [16], "names": EE_MOTOR_NAMES,
    }
    feats["action"] = {
        "dtype": "float32", "shape": [16], "names": [EE_MOTOR_NAMES],
    }
    info["robot_type"] = args.robot_type
    info["fps"] = args.fps
    # 更新 data 文件大小
    info["data_files_size_in_mb"] = round(
        sum(p.stat().st_size for p in ee_chunks) / 1e6, 3
    )
    with open(ee_dir / "meta" / "info.json", "w") as f:
        json.dump(info, f, indent=2)

    # ---- stats.json：数值重算，视频保留 joint ----
    print("[5/5] writing stats.json...", flush=True)
    with open(joint_dir / "meta" / "stats.json") as f:
        stats = json.load(f)
    num_stats = _compute_numeric_stats(state, action)
    stats["observation.state"] = num_stats["observation.state"]
    stats["action"] = num_stats["action"]
    with open(ee_dir / "meta" / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"DONE: ee dataset at {ee_dir} ({len(joint_chunks)} chunks, videos -> {joint_dir}/videos)", flush=True)
    print("NEXT: run verify_from_csv.py --dataset <ee_dir> --csv <csv> --action_type ee", flush=True)


if __name__ == "__main__":
    main()
