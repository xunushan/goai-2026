#!/usr/bin/env python3
"""从 X_VLA 策略服务日志（[x_vla][io] 行）提取**真机**评测轨迹并导出 CSV。

与仿真版 utils/extract_policy_log_csv.py 的差异（真机日志特有）：
  1. 真机日志的 client_observation / server_actions 带 `episode` 字段
     （形如 `ep001_afbc`，与评测视频文件名中的段一致），episode 边界以该字段
     变化为准，比「request 计数器归零」更可靠（真机 request 亦每集归零），
     CSV 的 `episode` 列即该字段原值；
  2. 新增 `model_name` 列，从所在文件夹名截取（去掉末尾 `_YYYYMMDD_HHMMSS`），
     如 robot_test/x0_90k_20260919_190445 → x0_90k；
  3. 新增 `task_name` / `instruction` / `model_prompt` 三列（原仿真版只有
     `task` = instruction）。

CSV 字段（一行 = 一个真机执行帧）：
  model_name      模型名，取自运行目录名（去掉末尾时间戳）
  task_name       任务目录名，如 stand_up_bottles
  instruction     客户端下发的指令，如 "Stand the bottle upright."
  model_prompt    服务端实际喂给模型的 prompt
  episode_index   本次转换内 episode 序号，从 0 起
  env_idx         并行环境编号（真机恒为 0）
  episode         episode 字段原值，如 ep001_afbc；与评测视频文件名一致
  frame_index     该 episode 内的帧号，首帧 0
  state_*         16 维 state（l_x..l_g, r_x..r_g），仅 chunk 边界帧有值，
                  其余帧为空字符串
  action_*        16 维 action，顺序与 info.json 的 action 特征一致

用法：
  # 单个运行目录 → CSV 写到 robot_test/ 下
  python utils/extract_robot_log_csv.py --run robot_test/x0_90k_20260919_190445
  # 批量：robot_test 下所有运行目录 → 汇总为一张总表
  python utils/extract_robot_log_csv.py --root robot_test
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

# 同目录的仿真版脚本提供日志解析与 16 维展平等公共实现
sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_policy_log_csv import STATE_NAMES, flatten_state16, parse_events  # noqa: E402

STATE_PREFIX = "state_"
ACTION_PREFIX = "action_"

# 运行目录名末尾的时间戳：_20260919_190445
_RUN_TS_RE = re.compile(r"_\d{8}_\d{6}$")


def model_name_from_dir(run_dir: Path) -> str:
    """从运行目录名截取模型名：去掉末尾 `_YYYYMMDD_HHMMSS`。"""
    return _RUN_TS_RE.sub("", run_dir.name) or run_dir.name


def split_by_episode(events: list[tuple[str, dict]]) -> list[tuple[str, list[tuple[str, dict]]]]:
    """按 `episode` 字段变化切分，返回 [(episode 名, 事件列表), ...]。

    `episode` 缺失时退化为仿真版逻辑：按 request 计数器归零切分，名为空串。
    """
    episodes: list[tuple[str, list[tuple[str, dict]]]] = []
    cur_name: str | None = None
    cur: list[tuple[str, dict]] = []
    prev_request: int | None = None

    for ts, payload in events:
        name: str | None = None
        if payload["event"] == "client_observation":
            name = str(payload.get("episode") or "")
            request = int(payload["request"])
            if prev_request is not None and request < prev_request and not name:
                name = f"ep{len(episodes) + 1:03d}"
            prev_request = request

        if name and cur and name != cur_name:
            episodes.append((cur_name or "", cur))
            cur = []
        if name:
            cur_name = name
        cur.append((ts, payload))

    if cur:
        episodes.append((cur_name or "", cur))
    return episodes


def episode_to_rows(
    episode: list[tuple[str, dict]], ep_index: int, episode_name: str, model_name: str
) -> list[list]:
    """把一个 episode 展开为帧级行（结构与仿真版一致，前面多几列元信息）。"""
    obs_by_req: dict[int, dict] = {}
    act_by_req: dict[int, dict] = {}
    for _, payload in episode:
        key = int(payload["request"])
        if payload["event"] == "client_observation":
            obs_by_req[key] = payload
        elif payload["event"] == "server_actions":
            act_by_req[key] = payload

    env_requests: dict[int, list[int]] = defaultdict(list)
    for request, obs in sorted(obs_by_req.items()):
        env_requests[int(obs.get("env_idx", 0))].append(request)

    rows: list[list] = []
    for env_idx, requests in sorted(env_requests.items()):
        head = obs_by_req[requests[0]]
        task_name = str(head.get("task_name", ""))
        instruction = str(head.get("instruction", ""))
        model_prompt = str(head.get("model_prompt", ""))
        for k, request in enumerate(requests):
            obs = obs_by_req[request]
            act = act_by_req.get(request)
            if act is None:
                # 观测与动作未配对（正常应为 1:1），跳过该请求
                continue
            execute_steps = int(act.get("execute_steps", 30))
            frame0 = k * execute_steps
            state16 = flatten_state16(obs.get("state", {}))
            action16s = act.get("actions_16d")
            steps = len(action16s) if action16s else execute_steps
            for step in range(steps):
                frame = frame0 + step
                # state 仅在该请求对应的 chunk 边界帧（首帧）有值
                state_row = state16 if step == 0 else [""] * len(state16)
                action_row = action16s[step] if action16s else [""] * 16
                rows.append(
                    [
                        model_name,
                        task_name,
                        instruction,
                        model_prompt,
                        ep_index,
                        env_idx,
                        episode_name,
                        frame,
                    ]
                    + state_row
                    + action_row
                )
    return rows


CSV_COLUMNS = (
    ["model_name", "task_name", "instruction", "model_prompt", "episode_index", "env_idx",
     "episode", "frame_index"]
    + [f"{STATE_PREFIX}{name}" for name in STATE_NAMES]
    + [f"{ACTION_PREFIX}{name}" for name in STATE_NAMES]
)


def convert_run(log_path: Path, model_name: str) -> pd.DataFrame:
    events = parse_events(str(log_path))
    if not events:
        raise ValueError(f"日志中未找到 [x_vla][io] 事件: {log_path}")

    rows: list[list] = []
    for ep_index, (episode_name, episode) in enumerate(split_by_episode(events)):
        rows.extend(episode_to_rows(episode, ep_index, episode_name, model_name))
    return pd.DataFrame(rows, columns=CSV_COLUMNS)


def collect_runs(run_dirs: list[Path]) -> tuple[pd.DataFrame, list[str]]:
    """把多个运行目录的 episode 帧合成一张总表，返回 (df, 每个目录的摘要行)。"""
    frames: list[pd.DataFrame] = []
    summary: list[str] = []
    for run_dir in run_dirs:
        log_path = find_log(run_dir)
        if log_path is None:
            print(f"skip: {run_dir} 无 .log", file=sys.stderr)
            continue
        model_name = model_name_from_dir(run_dir)
        df = convert_run(log_path, model_name)
        frames.append(df)
        eps = df["episode"].dropna().unique().tolist()
        summary.append(
            f"  {run_dir.name}: model={model_name} episodes={len(eps)} {eps} rows={len(df)}"
        )
    if not frames:
        raise ValueError("没有可转换的日志")
    return pd.concat(frames, ignore_index=True), summary


def find_log(run_dir: Path) -> Path | None:
    logs = sorted(run_dir.glob("*.log"))
    if not logs:
        return None
    if len(logs) > 1:
        print(f"warning: {run_dir} 有多个日志，取 {logs[0].name}", file=sys.stderr)
    return logs[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--run", help="单个运行目录（如 robot_test/x0_90k_20260919_190445）")
    src.add_argument("--root", help="批量：根目录（如 robot_test），汇总为一张总表")
    parser.add_argument(
        "--out",
        default=None,
        help="CSV 输出路径。--run 默认 <运行目录>.csv（写到上级目录），"
        "--root 默认 <根目录>/real_episodes.csv",
    )
    args = parser.parse_args()

    if args.run:
        run_dirs = [Path(args.run)]
    else:
        run_dirs = sorted(p for p in Path(args.root).iterdir() if p.is_dir())

    df, summary = collect_runs(run_dirs)

    if args.out:
        out_path = Path(args.out)
    elif args.run:
        out_path = run_dirs[0].parent / f"{run_dirs[0].name}.csv"
    else:
        out_path = Path(args.root) / "real_episodes.csv"
    df.to_csv(out_path, index=False)

    print("\n".join(summary))
    print(f"total: runs={len(summary)} rows={len(df)} -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
