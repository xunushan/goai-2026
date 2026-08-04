#!/usr/bin/env python3
"""LeRobot v3 数据集质量审计（仅数值数据，不解码视频）。

按 row group 逐批读取 parquet 主数据文件，仅加载数值列，检测数据完整性问题。
输出人类可读报告、episode 粒度指标和 findings JSONL（可驱动后续视频提取确认）。

================================================================================
检测类别
================================================================================

frame_sequence              帧编号不连续              critical  不需要视频
timestamp_sequence          时间戳不连续              critical  需要视频
global_index_sequence       全局索引不连续            critical  不需要视频
non_finite_values           存在 NaN 或 Inf           critical  不需要视频
metadata_length_mismatch    元数据与主数据长度不一致   critical  不需要视频
metadata_task_mismatch      任务文本与 task_index 不一致 critical  不需要视频
video_metadata_missing      缺少三路视频定位元数据     critical  不需要视频
action_state_tracking       action 与下一帧 state 跟踪误差异常 high    需要视频
action_discontinuity        相邻 action 存在异常跳变   high      需要视频
quaternion_norm             四元数范数异常（≠1）        critical  不需要视频
quaternion_sign_flip        四元数发生 q/-q 符号翻转   medium    不需要视频

================================================================================
关键参数（命令行参数）
================================================================================

--tracking-translation-m     0.005   action 与 next-state 位移误差阈值（米）
--tracking-rotation-deg      5.0     action 与 next-state 旋转误差阈值（度）
--jump-translation-m         0.05    相邻 action 位移跳变阈值（米）
--jump-rotation-deg          15.0    相邻 action 旋转跳变阈值（度）
--quaternion-norm-tolerance  0.02    四元数范数容差（偏离 1 的最大允许量）
--fps-tolerance              1e-4    时间戳帧间隔容差（秒）
--chunk-size                 50      chunk 大小，用于 padding 比例统计
--max-findings-per-type      200     每类问题最多保留数量
--task                       None    精确任务文本；不指定则审计所有任务

================================================================================
输出文件
================================================================================

episode_metrics.csv
    每个 episode 的统计指标：tracking/jump/rotation/max 等数值，以及
    padding_fraction（用于判断训练时 future action 是否充足）。

integrity_findings.jsonl
    所有检测到的问题条目（review_required=true 的需要视频确认），
    包含 issue_id、severity、category、start_frame、end_frame、key_frames、
    metrics、questions 等字段。

audit_summary.json
    统计摘要，包含已审计 episode 数、元数据 episode 数、各类别 findings 数量、
    review_required 数量、当前阈值配置。

audit_report.md
    人类可读的 Markdown 报告，含结论说明、自动判定结果、findings 分类表、
    padding 风险统计。

================================================================================
用法
================================================================================

    python audit_dataset_quality.py \
        --dataset data/lerobot_v30_ee \
        --task "Stack the three blocks with different textures." \
        --output-dir outputs/audit

    # 不指定 task，审计所有任务
    python audit_dataset_quality.py \
        --dataset data/lerobot_v30_ee \
        --output-dir outputs/audit_all

    # 自定义阈值
    python audit_dataset_quality.py \
        --dataset data/lerobot_v30_ee \
        --task "Stack the three blocks with different textures." \
        --output-dir outputs/audit \
        --tracking-translation-m 0.003 \
        --jump-translation-m 0.02

================================================================================
注意
================================================================================

所有 findings 仅基于数值数据判定。review_required=true 的条目需要根据
episode_index / start_frame / end_frame / key_frames 从三路视频中提取
对应片段进行人工确认。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyarrow.parquet as pq


ARM_SLICES = {"left": slice(0, 8), "right": slice(8, 16)}
CAMERAS = (
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)


@dataclass
class Episode:
    episode_index: int
    task_index: int
    frame_index: np.ndarray
    timestamp: np.ndarray
    global_index: np.ndarray
    state: np.ndarray
    action: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task", help="Exact task text; omit to audit all tasks")
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--fps-tolerance", type=float, default=1e-4)
    parser.add_argument("--tracking-translation-m", type=float, default=0.005)
    parser.add_argument("--tracking-rotation-deg", type=float, default=5.0)
    parser.add_argument("--jump-translation-m", type=float, default=0.05)
    parser.add_argument("--jump-rotation-deg", type=float, default=15.0)
    parser.add_argument("--quaternion-norm-tolerance", type=float, default=0.02)
    parser.add_argument("--max-findings-per-type", type=int, default=200)
    return parser.parse_args()


def json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def load_tasks(dataset: Path) -> tuple[dict[int, str], dict[str, int]]:
    rows = pq.read_table(dataset / "meta" / "tasks.parquet").to_pylist()
    by_index: dict[int, str] = {}
    for row in rows:
        text = str(row.get("task") or row.get("__index_level_0__"))
        by_index[int(row["task_index"])] = text
    return by_index, {text: index for index, text in by_index.items()}


def load_episode_metadata(dataset: Path) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for path in sorted((dataset / "meta" / "episodes").rglob("*.parquet")):
        for row in pq.read_table(path).to_pylist():
            index = int(row["episode_index"])
            if index in result:
                raise ValueError(f"Duplicate episode_index in metadata: {index}")
            result[index] = row
    if not result:
        raise FileNotFoundError("No episode metadata parquet files found")
    return result


def quaternion_angle_deg(q0: np.ndarray, q1: np.ndarray) -> np.ndarray:
    n0 = np.linalg.norm(q0, axis=-1, keepdims=True)
    n1 = np.linalg.norm(q1, axis=-1, keepdims=True)
    a = q0 / np.maximum(n0, 1e-12)
    b = q1 / np.maximum(n1, 1e-12)
    dots = np.abs(np.sum(a * b, axis=-1))
    return np.rad2deg(2.0 * np.arccos(np.clip(dots, 0.0, 1.0)))


def contiguous_ranges(mask: np.ndarray) -> list[tuple[int, int]]:
    indices = np.flatnonzero(mask)
    if len(indices) == 0:
        return []
    breaks = np.flatnonzero(np.diff(indices) > 1)
    starts = np.r_[indices[0], indices[breaks + 1]]
    ends = np.r_[indices[breaks], indices[-1]]
    return [(int(start), int(end)) for start, end in zip(starts, ends)]


def stable_issue_id(category: str, episode: int, arm: str, start: int, end: int) -> str:
    raw = f"{category}|{episode}|{arm}|{start}|{end}"
    suffix = hashlib.sha1(raw.encode()).hexdigest()[:8]
    return f"{category}_ep{episode:04d}_{arm}_{start:06d}_{end:06d}_{suffix}"


def finding(
    category: str,
    severity: str,
    episode: Episode,
    task: str,
    arm: str,
    start: int,
    end: int,
    metrics: dict[str, Any],
    reason_cn: str,
    review_required: bool,
    questions: list[str],
) -> dict[str, Any]:
    peak = int(metrics.pop("peak_frame", (start + end) // 2))
    before = max(0, start - 25)
    after = min(len(episode.frame_index) - 1, end + 25)
    return {
        "issue_id": stable_issue_id(category, episode.episode_index, arm, start, end),
        "source": "integrity_audit",
        "category": category,
        "category_cn": reason_cn,
        "severity": severity,
        "review_required": review_required,
        "task_index": episode.task_index,
        "task": task,
        "episode_index": episode.episode_index,
        "arm": arm,
        "start_frame": int(start),
        "peak_frame": peak,
        "end_frame": int(end),
        "key_frames": sorted(set([before, start, peak, end, after])),
        "clip_padding_frames": 50,
        "reasons": [category],
        "metrics": metrics,
        "questions": questions,
        "review_status": "unreviewed",
    }


def iter_episodes(dataset: Path, selected_task: int | None) -> Iterable[Episode]:
    columns = [
        "observation.state",
        "action",
        "timestamp",
        "frame_index",
        "episode_index",
        "index",
        "task_index",
    ]
    for path in sorted((dataset / "data").rglob("*.parquet")):
        parquet = pq.ParquetFile(path)
        for row_group in range(parquet.num_row_groups):
            table = parquet.read_row_group(row_group, columns=columns)
            raw = table.to_pydict()
            episode_ids = np.asarray(raw["episode_index"], dtype=np.int64)
            task_ids = np.asarray(raw["task_index"], dtype=np.int64)
            for episode_index in np.unique(episode_ids):
                mask = episode_ids == episode_index
                if selected_task is not None and not np.any(task_ids[mask] == selected_task):
                    continue
                order = np.argsort(np.asarray(raw["frame_index"], dtype=np.int64)[mask])
                yield Episode(
                    episode_index=int(episode_index),
                    task_index=int(task_ids[mask][order][0]),
                    frame_index=np.asarray(raw["frame_index"], dtype=np.int64)[mask][order],
                    timestamp=np.asarray(raw["timestamp"], dtype=np.float64)[mask][order],
                    global_index=np.asarray(raw["index"], dtype=np.int64)[mask][order],
                    state=np.asarray(raw["observation.state"], dtype=np.float64)[mask][order],
                    action=np.asarray(raw["action"], dtype=np.float64)[mask][order],
                )


def summarize(values: np.ndarray) -> dict[str, float]:
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return {"mean": math.nan, "p95": math.nan, "p99": math.nan, "max": math.nan}
    return {
        "mean": float(np.mean(finite)),
        "p95": float(np.percentile(finite, 95)),
        "p99": float(np.percentile(finite, 99)),
        "max": float(np.max(finite)),
    }


def audit_episode(
    episode: Episode,
    task_text: str,
    metadata: dict[str, Any] | None,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    n = len(episode.frame_index)
    findings: list[dict[str, Any]] = []
    expected_frames = np.arange(n)
    frame_bad = episode.frame_index != expected_frames
    timestamp_delta = np.diff(episode.timestamp)
    expected_dt = 1.0 / float(json.loads((args.dataset / "meta" / "info.json").read_text())["fps"])
    timestamp_bad = np.abs(timestamp_delta - expected_dt) > args.fps_tolerance
    global_bad = np.diff(episode.global_index) != 1
    finite_bad = ~(np.all(np.isfinite(episode.state), axis=1) & np.all(np.isfinite(episode.action), axis=1))

    generic_checks = (
        ("frame_sequence", frame_bad, "帧编号不连续", False),
        ("timestamp_sequence", np.r_[False, timestamp_bad], "时间戳不连续", True),
        ("global_index_sequence", np.r_[False, global_bad], "全局索引不连续", False),
        ("non_finite_values", finite_bad, "存在 NaN 或 Inf", False),
    )
    for category, mask, label, needs_video in generic_checks:
        for start, end in contiguous_ranges(mask):
            findings.append(
                finding(
                    category, "critical", episode, task_text, "both", start, end,
                    {"count": end - start + 1}, label, needs_video,
                    ["三路画面是否在该时间点发生跳变或不同步？"] if needs_video else [],
                )
            )

    metrics: dict[str, Any] = {
        "episode_index": episode.episode_index,
        "task_index": episode.task_index,
        "task": task_text,
        "frames": n,
        "metadata_length": int(metadata["length"]) if metadata else -1,
        "frame_sequence_errors": int(np.sum(frame_bad)),
        "timestamp_errors": int(np.sum(timestamp_bad)),
        "global_index_errors": int(np.sum(global_bad)),
        "non_finite_frames": int(np.sum(finite_bad)),
        "padding_fraction_mean": float(np.mean(np.maximum(0, args.chunk_size - np.minimum(args.chunk_size, n - np.arange(n))) / args.chunk_size)),
        "padding_fraction_gt_half": float(np.mean((n - np.arange(n)) < args.chunk_size / 2)),
    }

    if metadata is None or int(metadata["length"]) != n:
        findings.append(
            finding(
                "metadata_length_mismatch", "critical", episode, task_text, "both", 0, max(0, n - 1),
                {"observed_length": n, "metadata_length": int(metadata["length"]) if metadata else None},
                "Episode 元数据长度与主数据不一致", False, [],
            )
        )
    if metadata is not None:
        metadata_tasks = [str(value) for value in (metadata.get("tasks") or [])]
        if task_text not in metadata_tasks:
            findings.append(
                finding(
                    "metadata_task_mismatch", "critical", episode, task_text, "both", 0, max(0, n - 1),
                    {"metadata_tasks": metadata_tasks, "numeric_task": task_text},
                    "Episode 的任务文本与主数据 task_index 不一致", False, [],
                )
            )
        missing_video_metadata = []
        for camera in CAMERAS:
            for suffix in ("chunk_index", "file_index", "from_timestamp", "to_timestamp"):
                key = f"videos/{camera}/{suffix}"
                if metadata.get(key) is None:
                    missing_video_metadata.append(key)
        if missing_video_metadata:
            findings.append(
                finding(
                    "video_metadata_missing", "critical", episode, task_text, "both", 0, max(0, n - 1),
                    {"missing_fields": missing_video_metadata},
                    "Episode 缺少三路视频定位元数据", False, [],
                )
            )

    for arm, arm_slice in ARM_SLICES.items():
        state = episode.state[:, arm_slice]
        action = episode.action[:, arm_slice]
        q_state = state[:, 3:7]
        q_action = action[:, 3:7]
        qn_state = np.linalg.norm(q_state, axis=1)
        qn_action = np.linalg.norm(q_action, axis=1)
        norm_bad = (np.abs(qn_state - 1) > args.quaternion_norm_tolerance) | (
            np.abs(qn_action - 1) > args.quaternion_norm_tolerance
        )
        sign_state = np.r_[False, np.sum(q_state[1:] * q_state[:-1], axis=1) < 0]
        sign_action = np.r_[False, np.sum(q_action[1:] * q_action[:-1], axis=1) < 0]

        tracking_t = np.linalg.norm(action[:-1, :3] - state[1:, :3], axis=1)
        tracking_r = quaternion_angle_deg(action[:-1, 3:7], state[1:, 3:7])
        jump_t = np.r_[0.0, np.linalg.norm(np.diff(action[:, :3], axis=0), axis=1)]
        jump_r = np.r_[0.0, quaternion_angle_deg(action[:-1, 3:7], action[1:, 3:7])]

        metrics.update({
            f"{arm}_tracking_translation_mean_m": summarize(tracking_t)["mean"],
            f"{arm}_tracking_translation_p99_m": summarize(tracking_t)["p99"],
            f"{arm}_tracking_translation_max_m": summarize(tracking_t)["max"],
            f"{arm}_tracking_rotation_p99_deg": summarize(tracking_r)["p99"],
            f"{arm}_tracking_rotation_max_deg": summarize(tracking_r)["max"],
            f"{arm}_action_jump_translation_max_m": summarize(jump_t)["max"],
            f"{arm}_action_jump_rotation_max_deg": summarize(jump_r)["max"],
            f"{arm}_state_quaternion_sign_flips": int(np.sum(sign_state)),
            f"{arm}_action_quaternion_sign_flips": int(np.sum(sign_action)),
            f"{arm}_quaternion_norm_errors": int(np.sum(norm_bad)),
        })

        checks = [
            (
                "action_state_tracking",
                np.r_[tracking_t > args.tracking_translation_m, False]
                | np.r_[tracking_r > args.tracking_rotation_deg, False],
                "high", "Action 与下一帧 state 的跟踪误差异常", True,
                tracking_t, tracking_r,
            ),
            (
                "action_discontinuity",
                (jump_t > args.jump_translation_m) | (jump_r > args.jump_rotation_deg),
                "high", "相邻 action 存在异常跳变", True,
                jump_t, jump_r,
            ),
            (
                "quaternion_norm",
                norm_bad, "critical", "四元数范数异常", False,
                np.abs(qn_state - 1), np.abs(qn_action - 1),
            ),
            (
                "quaternion_sign_flip",
                sign_state | sign_action, "medium", "四元数发生 q/-q 符号翻转", False,
                sign_state.astype(float), sign_action.astype(float),
            ),
        ]
        for category, mask, severity, label, needs_video, first, second in checks:
            for start, end in contiguous_ranges(mask):
                local = slice(start, end + 1)
                combined = np.maximum(
                    np.asarray(first[local], dtype=float),
                    np.asarray(second[local], dtype=float),
                )
                peak = start + int(np.nanargmax(combined)) if np.any(np.isfinite(combined)) else start
                findings.append(
                    finding(
                        category, severity, episode, task_text, arm, start, end,
                        {
                            "peak_frame": peak,
                            "max_metric_1": float(np.nanmax(first[local])),
                            "max_metric_2": float(np.nanmax(second[local])),
                        },
                        label, needs_video,
                        [
                            "机械臂或场景是否发生碰撞、阻挡或突然跳变？",
                            "画面、state 与 action 是否保持时间同步？",
                        ] if needs_video else [],
                    )
                )
    return metrics, findings


def write_report(
    path: Path,
    args: argparse.Namespace,
    metrics: list[dict[str, Any]],
    findings: list[dict[str, Any]],
    metadata_count: int,
) -> None:
    categories = Counter(item["category"] for item in findings)
    review = [item for item in findings if item["review_required"]]
    critical = [item for item in findings if item["severity"] == "critical"]
    lines = [
        "# LeRobot 训练数据完整性审计",
        "",
        f"- 数据集：`{args.dataset.resolve()}`",
        f"- 任务过滤：`{args.task or '全部任务'}`",
        f"- 已审计 episodes：{len(metrics)}",
        f"- 元数据 episodes：{metadata_count}",
        f"- 数值 findings：{len(findings)}",
        f"- 需要视频确认：{len(review)}",
        f"- Critical findings：{len(critical)}",
        "",
        "## 结论说明",
        "",
        "本报告只基于 meta/parquet 数值数据，不解码视频。`review_required=true` 的",
        "条目已写入 `integrity_findings.jsonl`，需要服务器端抽取三视角画面后确认。",
        "四元数 q/-q 翻转、索引错误和非法数值可直接由数值数据确认。",
        "",
        "## 自动判定",
        "",
        (
            "- 在当前阈值下未发现数值完整性异常。仍需通过随机正常对照和候选视频"
            "确认规则是否存在漏报。"
            if not findings
            else f"- 当前阈值下发现 {len(findings)} 条数值问题，其中 {len(review)} 条需要视频确认。"
        ),
        f"- 左臂 action→next-state 平移误差最大值：{max(row['left_tracking_translation_max_m'] for row in metrics):.6f} m",
        f"- 右臂 action→next-state 平移误差最大值：{max(row['right_tracking_translation_max_m'] for row in metrics):.6f} m",
        f"- 左臂 action→next-state旋转误差最大值：{max(row['left_tracking_rotation_max_deg'] for row in metrics):.3f}°",
        f"- 右臂 action→next-state旋转误差最大值：{max(row['right_tracking_rotation_max_deg'] for row in metrics):.3f}°",
        "",
        "## Findings 分类",
        "",
        "| 类别 | 数量 |",
        "|---|---:|",
    ]
    lines += [f"| `{category}` | {count} |" for category, count in sorted(categories.items())]
    lines += [
        "",
        "## Padding 风险",
        "",
        f"- 平均 padding 比例：{np.mean([row['padding_fraction_mean'] for row in metrics]):.3%}",
        f"- future action 少于半个 chunk 的训练起点比例：{np.mean([row['padding_fraction_gt_half'] for row in metrics]):.3%}",
        "",
        "具体 episode 指标见 `episode_metrics.csv`。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.dataset = args.dataset.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    info = json.loads((args.dataset / "meta" / "info.json").read_text())
    task_by_index, index_by_task = load_tasks(args.dataset)
    if args.task and args.task not in index_by_task:
        available = "\n  ".join(sorted(index_by_task))
        raise ValueError(f"Unknown task {args.task!r}. Available:\n  {available}")
    selected_task = index_by_task.get(args.task) if args.task else None
    metadata = load_episode_metadata(args.dataset)

    all_metrics: list[dict[str, Any]] = []
    all_findings: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for episode in iter_episodes(args.dataset, selected_task):
        task_text = task_by_index.get(episode.task_index, f"task_index={episode.task_index}")
        metrics, findings = audit_episode(episode, task_text, metadata.get(episode.episode_index), args)
        all_metrics.append(metrics)
        for item in findings:
            if counts[item["category"]] < args.max_findings_per_type:
                all_findings.append(item)
                counts[item["category"]] += 1

    if not all_metrics:
        raise ValueError("No episodes matched the requested task")
    with (args.output_dir / "episode_metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(all_metrics[0]))
        writer.writeheader()
        writer.writerows(all_metrics)
    with (args.output_dir / "integrity_findings.jsonl").open("w", encoding="utf-8") as file:
        for item in all_findings:
            file.write(json.dumps(item, ensure_ascii=False, default=json_value) + "\n")
    summary = {
        "dataset": str(args.dataset),
        "task": args.task,
        "fps": info["fps"],
        "episodes_audited": len(all_metrics),
        "metadata_episodes": len(metadata),
        "findings": len(all_findings),
        "review_required": sum(bool(item["review_required"]) for item in all_findings),
        "findings_by_category": dict(Counter(item["category"] for item in all_findings)),
        "thresholds": {
            "chunk_size": args.chunk_size,
            "tracking_translation_m": args.tracking_translation_m,
            "tracking_rotation_deg": args.tracking_rotation_deg,
            "jump_translation_m": args.jump_translation_m,
            "jump_rotation_deg": args.jump_rotation_deg,
            "quaternion_norm_tolerance": args.quaternion_norm_tolerance,
        },
    }
    (args.output_dir / "audit_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_report(
        args.output_dir / "audit_report.md", args, all_metrics, all_findings, len(metadata)
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
