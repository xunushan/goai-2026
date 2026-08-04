#!/usr/bin/env python3
"""Find possible failure/recovery behavior from LeRobot numeric trajectories.

This script does not claim that an object was grasped, dropped, or recovered.
It creates a ranked review queue whose semantics must be confirmed from video.
Main parquet files are read one row group at a time.

================================================================================
检测类别
================================================================================

rapid_regrasp
    短时间内出现闭合—打开—再次闭合的抓取模式。
    判据：pattern=(1,0,1,0)，且 t2-t0 <= rapid_regrasp_seconds，末端返回距离 <= regrasp_return_distance_m。

sustained_tracking_error
    Action 与下一帧 state 持续不一致，可能指示碰撞、阻挡或控制失败。
    判据：translation error > tracking_translation_m 或 rotation error > tracking_rotation_deg 连续 >=3 帧。

local_oscillation
    小范围内反复运动且净位移很小，可能指示精细对准或无效振荡。
    判据：路径长 >0.025m，净位移 <0.006m，方向反转 >=4 次。

trajectory_return
    末端执行器返回较早访问过的位置，可能指示重试或恢复。
    判据：大范围移动（>0.12m）后回到之前位置附近（<0.012m）。

================================================================================
关键参数（命令行参数）
================================================================================

--open-threshold       0.8       夹爪张开阈值（0~1）
--closed-threshold     0.5       夹爪闭合阈值（0~1）
--debounce-frames      4         夹爪状态去抖帧数
--rapid-regrasp-seconds 3.0      快速重抓的时间窗口（秒）
--regrasp-return-distance-m 0.08 重抓后末端允许的最大返回距离（米）
--merge-gap-frames     25        同一臂事件合并的最大间隔帧数
--max-merged-window-frames 125   合并后事件的最大窗口帧数
--tracking-translation-m 0.005   位移误差阈值（米/帧）
--tracking-rotation-deg 5.0      旋转误差阈值（度/帧）
--stagnation-window    25        振荡检测窗口大小（帧）
--negative-controls    8         随机正常对照片段数量
--max-candidates       300       最大候选数量

================================================================================
输出文件
================================================================================

recovery_candidates.jsonl
    所有检测到的候选片段（含 rapid_regrasp / sustained_tracking_error /
    local_oscillation / trajectory_return 四类，以及 negative_control）。

review_queue.jsonl
    合并 integrity_findings.jsonl（可选）和 recovery_candidates.jsonl 后的
    完整待确认队列，按 severity 排序：critical > high > medium > low > control。

recovery_summary.json
    统计摘要，包含 task、task_index、episodes_scanned、各类别计数、
    negative_controls 数量、review_queue_size。

recovery_report.md
    人类可读的 Markdown 报告。

================================================================================
用法
================================================================================

    python find_recovery_candidates.py \
        --dataset data/lerobot_v30_ee \
        --task "Stack the three blocks with different textures." \
        --output-dir outputs/recovery

    # 自定义阈值示例
    python find_recovery_candidates.py \
        --dataset data/lerobot_v30_ee \
        --task "Stack the three blocks with different textures." \
        --output-dir outputs/recovery \
        --open-threshold 0.85 \
        --closed-threshold 0.45 \
        --rapid-regrasp-seconds 2.0 \
        --tracking-translation-m 0.003 \
        --negative-controls 12

================================================================================
注意
================================================================================

所有数值规则检测的结果仅供视频确认用，不能直接判定抓取、碰撞、掉落或恢复
是否真实发生。视频确认需根据 episode_index / start_frame / end_frame
从对应相机视频中提取片段回放。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq


ARMS = {"left": slice(0, 8), "right": slice(8, 16)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--task", required=True, help="Exact task text")
    parser.add_argument("--integrity-findings", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fps", type=float, help="Override info.json fps")
    parser.add_argument("--open-threshold", type=float, default=0.8)
    parser.add_argument("--closed-threshold", type=float, default=0.5)
    parser.add_argument("--debounce-frames", type=int, default=4)
    parser.add_argument("--rapid-regrasp-seconds", type=float, default=3.0)
    parser.add_argument("--regrasp-return-distance-m", type=float, default=0.08)
    parser.add_argument("--merge-gap-frames", type=int, default=25)
    parser.add_argument("--max-merged-window-frames", type=int, default=125)
    parser.add_argument("--tracking-translation-m", type=float, default=0.005)
    parser.add_argument("--tracking-rotation-deg", type=float, default=5.0)
    parser.add_argument("--stagnation-window", type=int, default=25)
    parser.add_argument("--negative-controls", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--max-candidates", type=int, default=300)
    return parser.parse_args()


def load_task_index(dataset: Path, task: str) -> int:
    rows = pq.read_table(dataset / "meta" / "tasks.parquet").to_pylist()
    mapping = {
        str(row.get("task") or row.get("__index_level_0__")): int(row["task_index"])
        for row in rows
    }
    if task not in mapping:
        raise ValueError(f"Unknown task {task!r}. Available:\n  " + "\n  ".join(sorted(mapping)))
    return mapping[task]


def quaternion_angle_deg(q0: np.ndarray, q1: np.ndarray) -> np.ndarray:
    q0 = q0 / np.maximum(np.linalg.norm(q0, axis=-1, keepdims=True), 1e-12)
    q1 = q1 / np.maximum(np.linalg.norm(q1, axis=-1, keepdims=True), 1e-12)
    dot = np.abs(np.sum(q0 * q1, axis=-1))
    return np.rad2deg(2 * np.arccos(np.clip(dot, 0, 1)))


def stable_id(category: str, episode: int, arm: str, start: int, end: int) -> str:
    raw = f"{category}|{episode}|{arm}|{start}|{end}"
    return f"{category}_ep{episode:04d}_{arm}_{start:06d}_{end:06d}_" + hashlib.sha1(
        raw.encode()
    ).hexdigest()[:8]


def debounce_gripper(values: np.ndarray, closed: float, opened: float, frames: int) -> np.ndarray:
    raw = np.full(len(values), -1, dtype=np.int8)
    raw[values <= closed] = 0
    raw[values >= opened] = 1
    result = np.full(len(values), -1, dtype=np.int8)
    stable = int(raw[0]) if raw[0] >= 0 else 1
    pending = stable
    count = 0
    for i, value in enumerate(raw):
        if value < 0 or value == stable:
            pending = stable
            count = 0
        elif value == pending:
            count += 1
            if count >= frames:
                stable = int(value)
                count = 0
        else:
            pending = int(value)
            count = 1
        result[i] = stable
    return result


def event(
    category: str,
    episode: int,
    task_index: int,
    task: str,
    arm: str,
    start: int,
    end: int,
    peak: int,
    score: int,
    reason_cn: str,
    metrics: dict[str, Any],
    questions: list[str],
) -> dict[str, Any]:
    return {
        "source": "recovery_candidate",
        "category": category,
        "category_cn": reason_cn,
        "episode_index": episode,
        "task_index": task_index,
        "task": task,
        "arm": arm,
        "start_frame": max(0, int(start)),
        "peak_frame": max(0, int(peak)),
        "end_frame": max(0, int(end)),
        "score": int(score),
        "reason_scores": {category: int(score)},
        "reasons": [category],
        "metrics": metrics,
        "questions": questions,
        "review_required": True,
        "review_status": "unreviewed",
    }


def detect_episode(
    episode: int,
    task_index: int,
    task: str,
    state: np.ndarray,
    action: np.ndarray,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    n = len(state)
    for arm, arm_slice in ARMS.items():
        s = state[:, arm_slice]
        a = action[:, arm_slice]
        grip = debounce_gripper(
            a[:, 7], args.closed_threshold, args.open_threshold, args.debounce_frames
        )
        transitions = np.flatnonzero(np.diff(grip) != 0) + 1
        # A rapid close-open-close is suspicious; multiple widely spaced grasps are
        # normal for a three-block task and are deliberately not flagged.
        for i in range(len(transitions) - 2):
            t0, t1, t2 = map(int, transitions[i : i + 3])
            pattern = tuple(grip[[t0 - 1, t0, t1, t2]])
            return_distance = float(np.linalg.norm(s[t2, :3] - s[t0, :3]))
            if (
                pattern == (1, 0, 1, 0)
                and t2 - t0 <= int(args.rapid_regrasp_seconds * (args.fps or 25))
                and return_distance <= args.regrasp_return_distance_m
            ):
                result.append(
                    event(
                        "rapid_regrasp", episode, task_index, task, arm,
                        t0, t2, t1, 3, "短时间内出现闭合—打开—再次闭合",
                        {
                            "transition_frames": [t0, t1, t2],
                            "duration_frames": t2 - t0,
                            "return_distance_m": return_distance,
                        },
                        [
                            "第一次闭合是否真正抓住物体？",
                            "中间打开是正常放置还是抓取失败？",
                            "再次闭合是否属于恢复动作，最终是否成功？",
                        ],
                    )
                )

        tracking_t = np.linalg.norm(a[:-1, :3] - s[1:, :3], axis=1)
        tracking_r = quaternion_angle_deg(a[:-1, 3:7], s[1:, 3:7])
        tracking_bad = (tracking_t > args.tracking_translation_m) | (
            tracking_r > args.tracking_rotation_deg
        )
        for start, end in contiguous_ranges(tracking_bad):
            if end - start + 1 < 3:
                continue
            local_score = tracking_t[start : end + 1] / args.tracking_translation_m
            local_score += tracking_r[start : end + 1] / args.tracking_rotation_deg
            peak = start + int(np.argmax(local_score))
            result.append(
                event(
                    "sustained_tracking_error", episode, task_index, task, arm,
                    start, end + 1, peak, 3, "Action 与下一帧 state 持续不一致",
                    {
                        "max_translation_m": float(np.max(tracking_t[start : end + 1])),
                        "max_rotation_deg": float(np.max(tracking_r[start : end + 1])),
                        "duration_frames": end - start + 1,
                    },
                    [
                        "是否发生碰撞、阻挡或控制执行失败？",
                        "是否由该异常触发后续纠正或恢复动作？",
                    ],
                )
            )

        displacement = np.diff(s[:, :3], axis=0)
        speed = np.linalg.norm(displacement, axis=1)
        cosine = np.ones(len(displacement))
        if len(displacement) > 1:
            denom = np.linalg.norm(displacement[1:], axis=1) * np.linalg.norm(
                displacement[:-1], axis=1
            )
            cosine[1:] = np.sum(displacement[1:] * displacement[:-1], axis=1) / np.maximum(
                denom, 1e-12
            )
        reverse = np.r_[False, cosine < -0.5]
        window = max(8, args.stagnation_window)
        for start in range(0, max(0, n - window + 1), max(4, window // 2)):
            end = start + window - 1
            xyz = s[start : end + 1, :3]
            path = float(np.sum(np.linalg.norm(np.diff(xyz, axis=0), axis=1)))
            net = float(np.linalg.norm(xyz[-1] - xyz[0]))
            reversals = int(np.sum(reverse[start : end + 1]))
            radius = float(np.linalg.norm(np.ptp(xyz, axis=0)))
            if path > 0.025 and net < 0.006 and reversals >= 4:
                result.append(
                    event(
                        "local_oscillation", episode, task_index, task, arm,
                        start, end, start + window // 2, 2, "小范围内反复运动且净位移很小",
                        {
                            "path_length_m": path,
                            "net_displacement_m": net,
                            "range_m": radius,
                            "direction_reversals": reversals,
                        },
                        [
                            "这是正常精细对准、失败后的纠正，还是无效振荡？",
                            "物体是否发生移动、碰撞或掉落？",
                        ],
                    )
                )

        # Return close to a much earlier pose after meaningful travel.
        stride = max(5, int((args.fps or 25) // 2))
        for current in range(3 * stride, n, stride):
            prior = s[: current - 2 * stride : stride, :3]
            if len(prior) == 0:
                continue
            distance = np.linalg.norm(prior - s[current, :3], axis=1)
            nearest = int(np.argmin(distance))
            earlier = nearest * stride
            if distance[nearest] < 0.012:
                traveled = float(
                    np.sum(np.linalg.norm(np.diff(s[earlier : current + 1, :3], axis=0), axis=1))
                )
                if traveled > 0.12:
                    result.append(
                        event(
                            "trajectory_return", episode, task_index, task, arm,
                            max(0, current - 2 * stride),
                            min(n - 1, current + stride),
                            current,
                            1,
                            "末端执行器返回较早访问过的位置",
                            {
                                "earlier_frame": earlier,
                                "return_distance_m": float(distance[nearest]),
                                "path_length_m": traveled,
                                "time_gap_frames": current - earlier,
                            },
                            [
                                "返回是正常抓取下一个方块，还是失败后的重试？",
                                "返回之前是否发生碰撞、掉落或放置失败？",
                            ],
                        )
                    )
    return result


def contiguous_ranges(mask: np.ndarray) -> list[tuple[int, int]]:
    ids = np.flatnonzero(mask)
    if len(ids) == 0:
        return []
    cuts = np.flatnonzero(np.diff(ids) > 1)
    starts = np.r_[ids[0], ids[cuts + 1]]
    ends = np.r_[ids[cuts], ids[-1]]
    return [(int(a), int(b)) for a, b in zip(starts, ends)]


def merge_events(
    events: list[dict[str, Any]], gap: int, max_window: int, episode_length: int
) -> list[dict[str, Any]]:
    events = sorted(events, key=lambda x: (x["arm"], x["start_frame"], x["end_frame"]))
    merged: list[dict[str, Any]] = []
    for item in events:
        if (
            merged
            and merged[-1]["arm"] == item["arm"]
            and item["start_frame"] <= merged[-1]["end_frame"] + gap
            and max(merged[-1]["end_frame"], item["end_frame"])
            - min(merged[-1]["start_frame"], item["start_frame"])
            <= max_window
        ):
            target = merged[-1]
            target["start_frame"] = min(target["start_frame"], item["start_frame"])
            target["end_frame"] = max(target["end_frame"], item["end_frame"])
            if item["score"] > target["score"]:
                target["peak_frame"] = item["peak_frame"]
            target["reasons"] = sorted(set(target["reasons"] + item["reasons"]))
            target.setdefault("reason_scores", {}).update(item.get("reason_scores", {}))
            target["score"] = sum(target["reason_scores"].values())
            target["questions"] = list(dict.fromkeys(target["questions"] + item["questions"]))
            target["metrics"]["merged_events"] = target["metrics"].get("merged_events", 1) + 1
        else:
            merged.append(json.loads(json.dumps(item)))
    for item in merged:
        item["severity"] = "high" if item["score"] >= 5 else "medium" if item["score"] >= 3 else "low"
        item["start_frame"] = max(0, item["start_frame"])
        item["end_frame"] = min(episode_length - 1, item["end_frame"])
        before = max(0, item["start_frame"] - 25)
        after = min(episode_length - 1, item["end_frame"] + 25)
        item["key_frames"] = sorted(
            set([before, item["start_frame"], item["peak_frame"], item["end_frame"], after])
        )
        item["clip_padding_frames"] = 50
        item["issue_id"] = stable_id(
            item["category"], item["episode_index"], item["arm"],
            item["start_frame"], item["end_frame"],
        )
    return merged


def load_integrity(path: Path | None, task: str) -> list[dict[str, Any]]:
    if path is None or not path.is_file():
        return []
    result = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            item = json.loads(line)
            if item.get("task") == task and item.get("review_required"):
                result.append(item)
    return result


def main() -> None:
    args = parse_args()
    args.dataset = args.dataset.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    info = json.loads((args.dataset / "meta" / "info.json").read_text())
    args.fps = args.fps or float(info["fps"])
    task_index = load_task_index(args.dataset, args.task)
    candidates: list[dict[str, Any]] = []
    episode_lengths: dict[int, int] = {}
    normal_pool: list[tuple[int, int]] = []
    columns = ["observation.state", "action", "episode_index", "frame_index", "task_index"]
    for path in sorted((args.dataset / "data").rglob("*.parquet")):
        parquet = pq.ParquetFile(path)
        for row_group in range(parquet.num_row_groups):
            table = parquet.read_row_group(row_group, columns=columns)
            raw = table.to_pydict()
            tasks = np.asarray(raw["task_index"], dtype=np.int64)
            if not np.any(tasks == task_index):
                continue
            episodes = np.asarray(raw["episode_index"], dtype=np.int64)
            frames = np.asarray(raw["frame_index"], dtype=np.int64)
            for episode in np.unique(episodes[tasks == task_index]):
                mask = (episodes == episode) & (tasks == task_index)
                order = np.argsort(frames[mask])
                state = np.asarray(raw["observation.state"], dtype=np.float64)[mask][order]
                action = np.asarray(raw["action"], dtype=np.float64)[mask][order]
                episode = int(episode)
                episode_lengths[episode] = len(state)
                candidates.extend(
                    merge_events(
                        detect_episode(episode, task_index, args.task, state, action, args),
                        args.merge_gap_frames,
                        args.max_merged_window_frames,
                        len(state),
                    )
                )
                if len(state) > 150:
                    normal_pool.extend((episode, frame) for frame in range(75, len(state) - 75, 100))

    candidates.sort(key=lambda item: (-item["score"], item["episode_index"], item["start_frame"]))
    candidates = candidates[: args.max_candidates]
    occupied: dict[int, list[tuple[int, int]]] = {}
    for item in candidates:
        occupied.setdefault(item["episode_index"], []).append(
            (item["start_frame"] - 50, item["end_frame"] + 50)
        )
    controls = [
        pair for pair in normal_pool
        if all(not (start <= pair[1] <= end) for start, end in occupied.get(pair[0], []))
    ]
    random.Random(args.seed).shuffle(controls)
    for episode, frame in controls[: args.negative_controls]:
        start, end = max(0, frame - 25), min(episode_lengths[episode] - 1, frame + 25)
        candidates.append({
            "issue_id": stable_id("normal_control", episode, "both", start, end),
            "source": "negative_control",
            "category": "normal_control",
            "category_cn": "随机正常对照片段",
            "severity": "control",
            "review_required": True,
            "task_index": task_index,
            "task": args.task,
            "episode_index": episode,
            "arm": "both",
            "start_frame": start,
            "peak_frame": frame,
            "end_frame": end,
            "key_frames": [start, frame, end],
            "clip_padding_frames": 25,
            "score": 0,
            "reasons": ["normal_control"],
            "metrics": {},
            "questions": ["该片段是否确实为正常任务进程？", "数值规则是否遗漏了可见异常？"],
            "review_status": "unreviewed",
        })

    integrity = load_integrity(args.integrity_findings, args.task)
    queue: dict[str, dict[str, Any]] = {item["issue_id"]: item for item in integrity}
    queue.update({item["issue_id"]: item for item in candidates})
    queue_items = sorted(
        queue.values(),
        key=lambda x: (
            {"critical": 0, "high": 1, "medium": 2, "low": 3, "control": 4}.get(
                x.get("severity", "low"), 5
            ),
            x["episode_index"],
            x["start_frame"],
        ),
    )
    for name, items in (
        ("recovery_candidates.jsonl", candidates),
        ("review_queue.jsonl", queue_items),
    ):
        with (args.output_dir / name).open("w", encoding="utf-8") as file:
            for item in items:
                file.write(json.dumps(item, ensure_ascii=False) + "\n")
    summary = {
        "task": args.task,
        "task_index": task_index,
        "episodes_scanned": len(episode_lengths),
        "recovery_candidates": sum(item["source"] == "recovery_candidate" for item in candidates),
        "negative_controls": sum(item["source"] == "negative_control" for item in candidates),
        "integrity_findings_in_queue": len(integrity),
        "review_queue_size": len(queue_items),
        "categories": dict(Counter(item["category"] for item in candidates)),
    }
    (args.output_dir / "recovery_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    lines = [
        "# 异常恢复候选检索",
        "",
        f"- 任务：{args.task}",
        f"- 扫描 episodes：{len(episode_lengths)}",
        f"- 恢复候选：{summary['recovery_candidates']}",
        f"- 随机正常对照：{summary['negative_controls']}",
        f"- 最终待确认项：{summary['review_queue_size']}",
        "",
        "这些结果仅表示数值轨迹可疑，不能仅凭 state/action 判断是否真实抓取、碰撞、",
        "掉落或恢复。请用 `review_queue.jsonl` 在有视频的服务器上提取画面后确认。",
        "",
        "## 候选分类",
        "",
    ]
    lines += [f"- `{key}`：{value}" for key, value in sorted(summary["categories"].items())]
    (args.output_dir / "recovery_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
