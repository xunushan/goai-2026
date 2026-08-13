#!/usr/bin/env python3
"""从 X-VLA 策略服务日志提取仿真 episode 数据，输出 pandas DataFrame。

日志来源：outputs/20260813/ckpt-6000-bench/server_ckpt6000_bench.log

双模式 episode 边界（自动检测）：
- 新日志（client 注入、server 记录的 episode_idx uuid 字段）：每个 uuid =
  一个 episode，task 优先取 task_name 字段（可区分 base/_random），
  instruction 映射仅作回退；
- 旧日志（无该字段）：以服务端 model.reset() 为边界（reset 清空
  policy_noise_draws，故每次 episode 的首次推理 policy_noise_draw == 1）。

对齐逻辑：
- 每个 env 内 observation / actions 严格交替、request 成对（已校验）；
- 一次推理产出 execute_steps 个动作（本日志 = 30），覆盖 frame_idx 区间
  [obs_frame, obs_frame + 30)；
- state（16 维）只在 observation 帧出现，其余帧为 None；
- action（16 维）= left_ee_pose(7) + left_ee_joint_state(1) +
  right_ee_pose(7) + right_ee_joint_state(1)，与 state 同构。

输出列：
    episode_idx  全局唯一 episode id（旧日志为整数序号；新日志为 8 位 uuid）
    task         任务名（新日志取 task_name；旧日志由 instruction 映射）
    env_idx      环境编号
    frame_idx    episode 内帧号（0 起）
    state        16 维 list 或 None（仅 observation 帧有值）
    action       16 维 list
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

# instruction -> task 名映射（来源：RoboDojo task templates / task_instructions.csv）
INSTRUCTION_TO_TASK = {
    "Arrange the numbers from left to right to form the largest possible number, and place them on the pad.": "arrange_largest_number",
    "Hang all the mugs on the mug rack.": "hang_mugs",
    "Place all the objects into the box with their front sides facing left.": "pack_objects_into_box",
    "Pour the liquid from the bottle into the cup.": "pour_liquid_into_cup",
    "Push the T-shaped block to align it precisely with the gray T-shaped pad.": "push_T",
    "Arrange the five nesting dolls in a row from left to right, from smallest to largest.": "sort_nesting_dolls_by_size",
    "Stack the three blocks with different textures.": "stack_blocks",
    "Stack the three bowls together.": "stack_bowls",
    "Pick up the broom, hand it over to the right hand, then use the dustpan to sweep the blocks.": "sweep_blocks",
}

IO_LINE_RE = re.compile(r"\[x_vla\]\[io\] (.*)")


def parse_log(path: Path) -> list[dict]:
    """逐行解析 [x_vla][io] JSON。"""
    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            m = IO_LINE_RE.search(line)
            if not m:
                continue
            try:
                rows.append(json.loads(m.group(1)))
            except json.JSONDecodeError:
                continue
    return rows


def group_by_env(rows: list[dict]) -> dict[int, list[dict]]:
    """按 env_idx 分组，保持日志出现顺序。"""
    by_env: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        by_env[r["env_idx"]].append(r)
    return dict(sorted(by_env.items()))


def build_episodes(rows: list[dict]) -> list[dict]:
    """把每个 env 的推理序列切分成 episode，返回扁平化的 frame 记录。

    双模式（自动检测）：
    - uuid 模式：日志含 episode_idx（8 位 hex）时，以 uuid 为 episode 主键，
      task 优先取 obs 的 task_name（可区分 base/_random），instruction 映射回退；
    - 旧模式：无该字段时，以 server_actions 的 policy_noise_draw == 1 为
      episode 边界（model.reset() 清空 noise_draws 后重新从 1 计数）。
    """
    by_env = group_by_env(rows)

    has_episode_idx = any(
        r.get("episode_idx") is not None
        for r in rows
        if r["event"] == "server_actions"
    )
    print(f"episode_idx field present in log: {has_episode_idx} "
          f"(-> {'uuid 模式' if has_episode_idx else '旧模式（noise_draw 边界）'})")

    records: list[dict] = []
    # 旧模式下按日志首次出现顺序全局编号，保证确定且唯一
    episode_counter = 0

    for env_idx, seq in by_env.items():
        # 每个 env 内 seq 严格按 [obs, act, obs, act, ...] 交替（已校验），
        # 按位置配对，避免 request 数值在 episode 间重置导致 dict 碰撞。
        obs_list = [r for r in seq if r["event"] == "client_observation"]
        act_list = [r for r in seq if r["event"] == "server_actions"]
        assert len(obs_list) == len(act_list), (
            f"env_idx={env_idx} obs/act 数量不一致"
        )

        frame_in_episode = 0  # 本次 episode 内已推进的帧数
        episode_idx = None
        cur_episode_records: list[dict] = []

        def flush():
            """把 cur_episode_records 里的 action 展开成逐帧记录并落盘。"""
            nonlocal episode_idx, cur_episode_records
            episode = episode_idx
            if episode is None:
                cur_episode_records = []
                return
            for rec in cur_episode_records:
                obs = rec["obs"]
                chunk = rec["chunk"]
                n = len(chunk)
                task = obs.get("task_name") or INSTRUCTION_TO_TASK.get(
                    obs.get("instruction", ""),
                    obs.get("instruction", ""),
                )
                for j in range(n):
                    records.append(
                        {
                            "episode_idx": episode,
                            "task": task,
                            "env_idx": env_idx,
                            "frame_idx": rec["base_frame"] + j,
                            # 仅 observation 帧（chunk 内第 0 个）才有 state
                            "state": (
                                state_16d(obs["state"]) if j == 0 else None
                            ),
                            "action": chunk[j],
                        }
                    )
            cur_episode_records = []

        for obs, act in zip(obs_list, act_list):
            if has_episode_idx:
                # uuid 模式：episode_idx 变化即新 episode；取 obs/act 任一值
                ep = act.get("episode_idx") or obs.get("episode_idx")
                if ep is not None and ep != episode_idx:
                    flush()  # 结束上一段 episode
                    episode_idx = ep
                    frame_in_episode = 0
            else:
                # 旧模式：reset() 后首次推理 noise_draw 重新从 1 计数
                if act["policy_noise_draw"] == 1:
                    flush()
                    episode_idx = episode_counter
                    episode_counter += 1
                    frame_in_episode = 0
            cur_episode_records.append(
                {
                    "obs": obs,
                    "chunk": act["actions_16d"],
                    "base_frame": frame_in_episode,
                }
            )
            frame_in_episode += len(act["actions_16d"])
        flush()

    return records


def state_16d(state: dict) -> list[float]:
    """state -> 16 维 [left_ee_pose(7), left_ee_joint_state(1),
    right_ee_pose(7), right_ee_joint_state(1)]。"""
    return (
        state["left_ee_pose"]
        + state["left_ee_joint_state"]
        + state["right_ee_pose"]
        + state["right_ee_joint_state"]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "log",
        nargs="?",
        default="outputs/20260813/ckpt-6000-bench/server_ckpt6000_bench.log",
        help="X-VLA 策略服务日志路径",
    )
    parser.add_argument("--out", default=None, help="parquet 输出路径（可选）")
    args = parser.parse_args()

    log_path = Path(args.log)
    if not log_path.exists():
        raise SystemExit(f"log not found: {log_path}")

    rows = parse_log(log_path)
    events = defaultdict(int)
    for r in rows:
        events[r["event"]] += 1
    print(f"parsed {len(rows)} io rows: {dict(events)}")

    records = build_episodes(rows)
    df = pd.DataFrame(records, columns=[
        "episode_idx", "task", "env_idx", "frame_idx", "state", "action",
    ])

    print(f"\ndataframe shape: {df.shape[0]} rows (frames) x {df.shape[1]} cols")
    n_episodes = df["episode_idx"].nunique()
    n_envs = df["env_idx"].nunique()
    print(f"episodes: {n_episodes}, envs: {n_envs}")
    print("\ntask distribution (frames / episodes):")
    task_grp = df.groupby("task")
    for task, g in task_grp:
        print(f"  {task:28s} frames={len(g):6d} episodes={g['episode_idx'].nunique():3d}")

    print("\nper-env episode counts:")
    print(df.groupby("env_idx")["episode_idx"].nunique().to_string())

    # 每个 chunk 固定 execute_steps 帧，仅 chunk 首帧带 state
    chunk_len = int(round(1.0 / (df["state"].notna().mean())))
    print(f"\nstate coverage: {df['state'].notna().mean():.4f} "
          f"(=1/{chunk_len}，仅 observation 帧)")
    print("frame_idx max per episode:",
          df.groupby(["env_idx", "episode_idx"])["frame_idx"].max().max())

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out, index=False)
        print(f"\nsaved -> {out}")

    # 便捷查看
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    print("\nhead:\n", df.head(8).to_string())
    print("\ntail:\n", df.tail(8).to_string())


if __name__ == "__main__":
    main()
