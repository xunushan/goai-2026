#!/usr/bin/env python3
"""仿真评测结果 (exp1_1cam) 的视频 + action 轨迹联动 HTML 生成工具。

与 tools/episode_video_sync.py (训练集) 的差异:
- 数据源: 评测日志 (X_VLA_*.log) + 按 task 目录组织的 episode 视频;
- episode 标识: 日志里的 8 位 hex uuid (episode_idx), 视频文件名含同一 uuid;
- 轨迹: 用 episode 的 action (16 维, 逐帧), 不是 state;
- 对齐: 视频文件 start_time=0、三路同长, 直接整文件取; 视频帧数为
  实际执行长度, action 可能更长 (策略整 chunk 预测, 尾部未执行), 统一截到
  视频帧数。

目录结构 (exp 目录):
    <exp>/
        X_VLA_*.log                    # 策略服务日志
        <task>/
            episode_{uuid}_cam_head_{success|fail}.mp4
            episode_{uuid}_cam_left_wrist_{success|fail}.mp4
            episode_{uuid}_cam_right_wrist_{success|fail}.mp4
            _result.json

用法:
    python tools/episode_video_sync_sim.py --exp outputs/20260813/exp1_1cam
    python tools/episode_video_sync_sim.py --exp ... --regen-video
    python tools/episode_video_sync_sim.py --exp ... --out outputs/.../interactive_sim
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.episode_video_sync import (  # noqa: E402
    HEADER,
    SCRIPT,
    compose_mosaic_video,
    get_plotly_js,
)
from utils.episode_analysis import compute_rotation_deg  # noqa: E402

# action 16 维 (extract_policy_log_csv.STATE_NAMES 顺序)
POS_L = slice(0, 3)     # l_x, l_y, l_z
QUAT_L = slice(3, 7)    # 四元数 (重排对 compute_rotation_deg 标量结果不变)
GRIP_L = 7
POS_R = slice(8, 11)
QUAT_R = slice(11, 15)
GRIP_R = 15

FPS = 25.0

IO_LINE_RE = re.compile(r"\[x_vla\]\[io\] (.*)")
VIDEO_RE = re.compile(
    r"episode_([0-9a-f]+)_cam_(head|left_wrist|right_wrist)_(success|fail)\.mp4"
)


def extract_action_episodes(log_path: Path) -> dict[str, dict]:
    """从日志提取 {uuid: {'task': str, 'actions': list[[16],...]}}。

    action 16 维由服务端 actions_16d 按出现顺序拼接 (uuid 唯一对应一个
    env, 出现顺序即时间顺序)。
    """
    eps: dict[str, dict] = {}
    with open(log_path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = IO_LINE_RE.search(line)
            if not m:
                continue
            try:
                p = json.loads(m.group(1))
            except json.JSONDecodeError:
                continue
            if not isinstance(p, dict):
                continue
            if p.get("event") == "server_actions" and "actions_16d" in p:
                u = p.get("episode_idx")
                if not u:
                    continue
                eps.setdefault(u, {"task": "", "actions": []})
                eps[u]["actions"].extend(p["actions_16d"])
            elif p.get("event") == "client_observation" and p.get("task_name"):
                u = p.get("episode_idx")
                if u and u in eps:
                    eps[u]["task"] = p["task_name"]
    return eps


def scan_videos(exp_dir: Path) -> dict[str, dict]:
    """扫描 exp 目录下全部 episode 视频 -> {uuid: {cam: path, 'task': str}}。"""
    out: dict[str, dict] = {}
    for v in exp_dir.rglob("*.mp4"):
        m = VIDEO_RE.match(v.name)
        if not m:
            continue
        u, cam = m.group(1), m.group(2)
        rec = out.setdefault(u, {"task": v.parent.name})
        rec[cam] = v
        rec.setdefault("task", v.parent.name)
    return out


def video_frames(path: Path) -> int:
    """视频总帧数 (用时长 x fps 取整)。"""
    o = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True)
    return int(round(float(o.stdout.strip()) * FPS))


def episode_payload(uuid: str, ep: dict, videos: dict, n_frames: int) -> dict:
    """从 action 轨迹构建 HTML 数据 (xyz / 旋转角 / gripper / 帧时刻)。"""
    act = np.asarray(ep["actions"], dtype=float)[:n_frames]  # 截到视频帧数
    n = act.shape[0]
    t = np.arange(n) / FPS
    rot_l = compute_rotation_deg(
        np.tile(act[0, QUAT_L], (n, 1)), act[:, QUAT_L])
    rot_r = compute_rotation_deg(
        np.tile(act[0, QUAT_R], (n, 1)), act[:, QUAT_R])
    return {
        "uuid": uuid,
        "task": ep["task"],
        "frames": int(n),
        "fps": FPS,
        "from": 0.0,
        "t": t.tolist(),
        "xyz_l": np.asarray(act[:, POS_L]).tolist(),
        "xyz_r": np.asarray(act[:, POS_R]).tolist(),
        "rot_l": rot_l.tolist(),
        "rot_r": rot_r.tolist(),
        "grip_l": [float(v) for v in act[:, GRIP_L]],
        "grip_r": [float(v) for v in act[:, GRIP_R]],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp", required=True, help="评测结果目录 (exp1_1cam)")
    parser.add_argument("--out",
                        default=str(ROOT / "outputs" / "episode_insight" / "interactive_sim"))
    parser.add_argument("--regen-video", action="store_true", help="强制重合成马赛克视频")
    args = parser.parse_args()

    exp_dir = Path(args.exp)
    log_paths = sorted(exp_dir.glob("X_VLA_*.log")) + sorted(exp_dir.glob("*.log"))
    if not log_paths:
        raise SystemExit(f"no log found in {exp_dir}")
    log_path = log_paths[0]

    eps = extract_action_episodes(log_path)
    vids = scan_videos(exp_dir)
    # 取交集: 有 action 且有完整三路视频的 uuid
    uuids = sorted(u for u in eps if set(vids.get(u, {})) >= {"head", "left_wrist", "right_wrist"})
    print(f"log episodes={len(eps)}, 三路视频齐全 uuid={len(uuids)}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    video_dir = out_dir / "videos"

    # 1) 合成马赛克 (幂等) + 组装 payload
    tasks: dict[str, dict] = {}
    eps_payload: dict[str, dict] = {}
    for u in uuids:
        rec = vids[u]
        slug = re.sub(r"[^a-z0-9_]", "_", rec["task"]) or "task"
        n_act = len(eps[u]["actions"])
        n_vid = video_frames(rec["head"])
        n = min(n_act, n_vid)
        mp4 = video_dir / f"{slug}_{u}.mp4"
        video_rel = os.path.relpath(mp4, out_dir)
        if not mp4.is_file() or args.regen_video:
            # 三路同长, 从 0 整文件取, 时长截到对齐帧数; 相机名映射到
            # compose_mosaic_video 期望的 high/left/right 键
            cam_map = {"head": "high", "left_wrist": "left",
                       "right_wrist": "right"}
            segs = {cam_map[c]: (os.path.relpath(rec[c], exp_dir), 0.0, n / FPS)
                    for c in ("head", "left_wrist", "right_wrist")}
            print(f"  compose {u} ({slug}) n={n} "
                  f"act={n_act} vid={n_vid} ...")
            compose_mosaic_video(exp_dir, segs, mp4)
        payload = episode_payload(u, eps[u], rec, n)
        payload["video"] = video_rel
        eps_payload[u] = payload
        key = rec["task"]
        if key not in tasks:
            tasks[key] = {"name": key, "slug": slug, "episodes": []}
        tasks[key]["episodes"].append(u)

    # 2) 单 HTML (内嵌 Plotly + 数据)
    data = {"tasks": tasks, "eps": eps_payload}
    plotly_js = get_plotly_js(out_dir / "plotly.min.js")
    html = (
        HEADER
        + SCRIPT
            .replace("__PLOTLY__", plotly_js)
            .replace("__DATA__", json.dumps(data))
    )
    out_html = out_dir / "episode_sync_sim.html"
    out_html.write_text(html, encoding="utf-8")

    print(f"\nHTML -> {out_html}  ({out_html.stat().st_size/1e6:.1f} MB, "
          f"{len(uuids)} episodes, {len(tasks)} tasks)")
    print("tasks:", ", ".join(tasks))


if __name__ == "__main__":
    main()
