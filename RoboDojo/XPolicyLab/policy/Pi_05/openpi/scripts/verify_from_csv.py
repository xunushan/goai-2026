"""以 CSV 为权威基准校验 lerobot v30 数据集的 state/action 数值。

用法：
  verify_from_csv.py --dataset <joint_or_ee_dir> --csv <csv> --action_type joint|ee [--atol 1e-4]

对每个 chunk parquet，用 (episode_index, frame_index) 从 CSV 匹配期望 state，
action 按「state 前移，episode 末帧为自身」规则构造，与 parquet 逐元素对比。
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

JOINT_STATE_COLS = (
    [f"left_joint_{i}" for i in range(6)]
    + ["left_gripper"]
    + [f"right_joint_{i}" for i in range(6)]
    + ["right_gripper"]
)
# 四元数 xyzw：left_eef_qx/qy/qz/qw（w 在最后，与 extract_to_csv.py 输出一致）
EE_STATE_COLS = (
    [f"left_eef_{n}" for n in ["x", "y", "z", "qx", "qy", "qz", "qw"]]
    + ["left_gripper"]
    + [f"right_eef_{n}" for n in ["x", "y", "z", "qx", "qy", "qz", "qw"]]
    + ["right_gripper"]
)
STATE_COLS = {"joint": JOINT_STATE_COLS, "ee": EE_STATE_COLS}


def expected_state_action(csv_df: pd.DataFrame, action_type: str):
    """返回 (dict[(epi,frame)] -> state_np, 期望 action 构造需逐 episode)。"""
    cols = STATE_COLS[action_type]
    epi = csv_df["episode_index"].to_numpy()
    frame = csv_df["frame_index"].to_numpy()
    state = csv_df[cols].to_numpy(dtype=np.float32)

    # action = state 前移，episode 末帧 action=自身
    action = np.empty_like(state)
    action[:-1] = state[1:]
    action[-1] = state[-1]
    epi_last = np.concatenate([epi[1:] != epi[:-1], [True]]).astype(bool)
    action[epi_last] = state[epi_last]

    lut = {}
    for i in range(len(epi)):
        lut[(int(epi[i]), int(frame[i]))] = (state[i], action[i])
    return lut


def main():
    parser = argparse.ArgumentParser(description="Verify dataset parquet against CSV baseline")
    parser.add_argument("--dataset", required=True, help="Dataset dir (joint or ee)")
    parser.add_argument("--csv", required=True, help="CSV baseline (extract_to_csv.py output)")
    parser.add_argument("--action_type", required=True, choices=["joint", "ee"])
    parser.add_argument("--atol", type=float, default=1e-4, help="Per-element abs tolerance")
    args = parser.parse_args()

    ds_dir = Path(args.dataset)
    csv_df = pd.read_csv(args.csv)
    print(f"CSV: {len(csv_df)} rows, {len(STATE_COLS[args.action_type])} state cols")
    lut = expected_state_action(csv_df, args.action_type)

    chunks = sorted((ds_dir / "data").glob("chunk-*/file-*.parquet"))
    if not chunks:
        raise SystemExit(f"no parquet under {ds_dir}/data")

    total_frames = 0
    n_mismatch_frames = 0
    n_missing = 0
    max_abs_diff = 0.0
    worst = None  # (epi, frame, col, parq_val, exp_val)
    mismatch_samples = []
    per_col_max = {}

    for c in chunks:
        df = pd.read_parquet(c)
        keys = list(zip(df["episode_index"].to_numpy(), df["frame_index"].to_numpy()))
        parq_state = [np.asarray(v, dtype=np.float32) for v in df["observation.state"]]
        parq_action = [np.asarray(v, dtype=np.float32) for v in df["action"]]
        for idx, k in enumerate(keys):
            total_frames += 1
            hit = lut.get((int(k[0]), int(k[1])))
            if hit is None:
                n_missing += 1
                continue
            exp_s, exp_a = hit
            for name, parq, exp in (("state", parq_state[idx], exp_s), ("action", parq_action[idx], exp_a)):
                diff = np.abs(parq.astype(np.float64) - exp.astype(np.float64))
                m = float(diff.max()) if diff.size else 0.0
                if m > max_abs_diff:
                    max_abs_diff = m
                    worst = (int(k[0]), int(k[1]), name, parq, exp)
                if m > args.atol:
                    n_mismatch_frames += 1
                    if len(mismatch_samples) < 10:
                        # 找最大偏差位置
                        pos = int(np.argmax(diff))
                        mismatch_samples.append(
                            f"  ep{int(k[0])} frame{int(k[1])} {name}[{pos}] parq={parq[pos]:.6f} csv={exp[pos]:.6f} diff={m:.2e}"
                        )
                    break  # 该帧已算不匹配

    print(f"\n=== verify {args.action_type} ===")
    print(f"frames checked: {total_frames}")
    print(f"missing in CSV: {n_missing}")
    print(f"mismatch frames (>{args.atol}): {n_mismatch_frames}")
    print(f"max abs diff: {max_abs_diff:.3e}")
    if worst is not None:
        epi, frame, name, parq, exp = worst
        print(f"worst: ep{epi} frame{frame} {name} dims={parq.shape}")
        for j in range(len(parq)):
            if abs(float(parq[j]) - float(exp[j])) == max_abs_diff:
                print(f"  col{j}: parq={parq[j]:.6f} csv={exp[j]:.6f}")
    if mismatch_samples:
        print("first mismatches:")
        for s in mismatch_samples:
            print(s)
    ok = n_mismatch_frames == 0 and n_missing == 0
    print(f"RESULT: {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
