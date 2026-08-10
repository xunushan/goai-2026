#!/usr/bin/env python3
"""从 X_VLA 策略服务日志（[x_vla][io] 行）提取仿真轨迹并导出 CSV。

日志来源：RoboDojo/XPolicyLab/policy/X_VLA/model.py 的 get_action_batch()
打印的 `[x_vla][io] {json}` 事件，成对出现：
  - client_observation：含 instruction（task）与 16 维 state（左右 7 维 ee
    pose + 1 维夹爪），仅在 action chunk 边界（每 execute_steps 帧）上报；
  - server_actions：含 model_chunk_size / execute_steps，及动作数据。

CSV 字段（一行 = 一个仿真帧）：
  task            任务指令（instruction）
  episode_index   按 request 计数器归零切分（服务端收到 reset 帧会重置
                  _request_index），一个 reset 组 = 一个 episode
  env_idx         并行仿真环境编号（episode 内多 env 并行，各自独立轨迹）
  frame_index     该 env 在 episode 内的帧号；首帧 0 即仿真初始 state
  state_*         16 维 state（l_x..l_g, r_x..r_g），仅 chunk 边界帧有值，
                  其余帧为空字符串
  action_*        16 维 action，顺序与 info.json 的 action 特征完全一致
                  （6D 旋转已由服务端转成四元数）

新旧日志兼容：
  - 新日志（model.py 含 actions_16d 字段）：action 16 维全量写入；
  - 旧日志（仅 xyz min/max + 夹爪概率）：action 仅夹爪两维可写，其余为空，
    并打印警告。

用法：
  python utils/extract_policy_log_csv.py \
      --log outputs/abtest/_logs/policy_X_VLA_port80.log --out out.csv
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

# info.json 中 observation.state / action 的 16 维命名与顺序
STATE_NAMES = [
    "l_x", "l_y", "l_z", "l_w", "l_wx", "l_wy", "l_wz", "l_g",
    "r_x", "r_y", "r_z", "r_w", "r_wx", "r_wy", "r_wz", "r_g",
]
STATE_PREFIX = "state_"
ACTION_PREFIX = "action_"

_IO_RE = re.compile(r"^\[([^]]+)\] \[x_vla\]\[io\] (.*)$")


def parse_events(log_path: str) -> list[tuple[str, dict]]:
    """解析日志中的 [x_vla][io] 事件，返回 [(ts, payload_dict), ...]。"""
    events: list[tuple[str, dict]] = []
    with open(log_path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = _IO_RE.match(line.strip())
            if not m:
                continue
            try:
                payload = json.loads(m.group(2))
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict) or "event" not in payload:
                continue
            events.append((m.group(1), payload))
    return events


def split_episodes(events: list[tuple[str, dict]]) -> list[list[tuple[str, dict]]]:
    """按 request 计数器归零切分 episode。

    服务端收到 reset 帧 → model.reset() → _request_index=0，因此观测事件里
    request 从高值回到 0 即为 episode 边界。
    """
    episodes: list[list[tuple[str, dict]]] = []
    cur: list[tuple[str, dict]] = []
    prev_request: int | None = None
    for ts, payload in events:
        if payload["event"] == "client_observation":
            request = int(payload["request"])
            if prev_request is not None and request < prev_request:
                episodes.append(cur)
                cur = []
            prev_request = request
        cur.append((ts, payload))
    if cur:
        episodes.append(cur)
    return episodes


def flatten_state16(state: dict) -> list[float | None]:
    """把 obs.state 展平为 info.json 顺序的 16 维列表。

    顺序：left_ee_pose(7) + left_ee_joint_state(1) +
          right_ee_pose(7) + right_ee_joint_state(1)。
    """
    parts = [
        state.get("left_ee_pose", []),
        state.get("left_ee_joint_state", []),
        state.get("right_ee_pose", []),
        state.get("right_ee_joint_state", []),
    ]
    return [float(v) for part in parts for v in part]


def episode_to_rows(episode: list[tuple[str, dict]], ep_index: int) -> tuple[list[list], int]:
    """把一个 episode 展开为帧级行。

    返回 (rows, missing_action_requests)，missing_action_requests 为缺少
    actions_16d 字段的请求数（旧日志为全部请求）。
    """
    obs_by_req: dict[int, dict] = {}
    act_by_req: dict[int, dict] = {}
    for _, payload in episode:
        key = int(payload["request"])
        if payload["event"] == "client_observation":
            obs_by_req[key] = payload
        elif payload["event"] == "server_actions":
            act_by_req[key] = payload

    # 按 env 分组；同 env 内按 request 升序即时间顺序
    env_requests: dict[int, list[int]] = defaultdict(list)
    for request, obs in sorted(obs_by_req.items()):
        env_requests[int(obs["env_idx"])].append(request)

    rows: list[list] = []
    missing_action = 0
    for env_idx, requests in sorted(env_requests.items()):
        task = str(obs_by_req[requests[0]].get("instruction", ""))
        for k, request in enumerate(requests):
            obs = obs_by_req[request]
            act = act_by_req.get(request)
            if act is None:
                # 观测与动作未配对（正常应为 1:1），跳过该请求
                continue
            frame0 = k * int(act.get("execute_steps", 30))
            state16 = flatten_state16(obs.get("state", {}))
            has_full = "actions_16d" in act
            if not has_full:
                missing_action += 1
            action16s = act.get("actions_16d")
            steps = len(action16s) if action16s else int(act.get("execute_steps", 30))
            for step in range(steps):
                frame = frame0 + step
                # state 仅在该请求对应的 chunk 边界帧（首帧）有值
                state_row = state16 if step == 0 else [""] * len(state16)
                if has_full:
                    action_row = action16s[step]
                else:
                    # 旧日志：仅夹爪可恢复（command 即下发夹爪值），其余留空
                    action_row = [""] * 16
                    left_cmd = act["left_gripper_probability"]["command"]
                    right_cmd = act["right_gripper_probability"]["command"]
                    if step < len(left_cmd):
                        action_row[7] = left_cmd[step]
                    if step < len(right_cmd):
                        action_row[15] = right_cmd[step]
                rows.append([task, ep_index, env_idx, frame] + state_row + action_row)
    return rows, missing_action


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", required=True, help="策略服务日志路径")
    parser.add_argument("--out", default="policy_log.csv", help="输出 CSV 路径")
    args = parser.parse_args()

    events = parse_events(args.log)
    if not events:
        print(f"error: 日志中未找到 [x_vla][io] 事件: {args.log}", file=sys.stderr)
        return 1
    episodes = split_episodes(events)

    rows: list[list] = []
    total_missing = 0
    for ep_index, episode in enumerate(episodes):
        ep_rows, missing = episode_to_rows(episode, ep_index)
        total_missing += missing
        rows.extend(ep_rows)

    columns = (
        ["task", "episode_index", "env_idx", "frame_index"]
        + [f"{STATE_PREFIX}{name}" for name in STATE_NAMES]
        + [f"{ACTION_PREFIX}{name}" for name in STATE_NAMES]
    )
    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(args.out, index=False)

    print(f"episodes={len(episodes)} rows={len(df)} output={args.out}")
    print(f"action rows missing actions_16d (旧日志格式，仅夹爪): {total_missing}")
    if total_missing:
        print(
            "warning: 部分/全部 action 缺 16 维（旧日志仅含夹爪与 xyz min/max，"
            "旋转缺失）。建议使用含 actions_16d 的新日志获得完整动作。",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
