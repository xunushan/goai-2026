"""
Episode 数据分析工具集（基于 step1 输出的 DataFrame）。

================================================================================
功能概览
================================================================================

1. add_action_delta_columns(df)
   向 DataFrame 添加 8 列 action delta 指标：
   - state_to_action_left/right_translation_m  ：action 与当前 state 的位置差
   - state_to_action_left/right_rotation_deg   ：action 与当前 state 的旋转角差
   - consecutive_action_left/right_translation_m：相邻两帧 action 的位置差
   - consecutive_action_left/right_rotation_deg ：相邻两帧 action 的旋转角差

2. analyze_episode_gripper(df, episode_action, task_index)
   分析单个 episode 的夹爪行为：
   - 统计量：count/min/p01/p10/median/p90/p99/max/fraction_le_0.5/fraction_ge_0.9
   - 4 面板可视化：动作分布直方图、夹爪时序图、归一化曲线+P10-P90区间、精细张开状态直方图

3. compose_episode_video(df, episode_index, output_path, crf=23)
   将单个 episode 的三个相机视频（俯视、左腕、右腕）合成为一个 2x1 马赛克 MP4。

4. get_task_names(dataset_root)
   从 tasks.parquet 读取任务名称映射，返回 {task_index: task_name, ...}。

5. analyze_task_distribution(df, dataset_root)
   统计每个 task 的 episode 数量和总帧数，返回 DataFrame：
   [task_index, task_name, episode_count, frame_count]。

================================================================================
用法示例
================================================================================

    from load_lerobot_dataset import load_lerobot_as_dataframe
    from episode_analysis import (
        add_action_delta_columns,
        analyze_episode_gripper,
        compose_episode_video,
        analyze_task_distribution,
    )

    # Step1: 加载数据
    df = load_lerobot_as_dataframe("data/lerobot_v30_ee", n_episodes=100)
    df.attrs["dataset_root"] = "data/lerobot_v30_ee"

    # 添加 8 列 delta 指标
    df = add_action_delta_columns(df)

    # 夹爪分析
    task8_df = df[df["task_index"] == 8]
    ep_idx = task8_df["episode_index"].unique()[0]
    ep_action = np.array(df[df["episode_index"] == ep_idx]["action"].tolist())
    stats, fig = analyze_episode_gripper(df, ep_action, task_index=8)

    # 合成视频
    compose_episode_video(df, episode_index=0, output_path="outputs/ep0_mosaic.mp4")

    # Task 分布统计
    result = analyze_task_distribution(df, "data/lerobot_v30_ee")
    print(result.to_string())
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =============================================================================
# 常量
# =============================================================================

LEFT_GRIPPER = 7
RIGHT_GRIPPER = 15

# =============================================================================
# 核心计算函数（输入为两个位置/四元数数组，计算差值）
# =============================================================================


def compute_position_delta_m(pos_ref: np.ndarray, pos_target: np.ndarray) -> np.ndarray:
    """
    计算两个位置之间的欧几里得距离（米）

    Args:
        pos_ref: shape (N, 3) 参考位置 [x, y, z]
        pos_target: shape (N, 3) 目标位置 [x, y, z]

    Returns:
        shape (N,) 的距离数组（米）
    """
    return np.linalg.norm(pos_target - pos_ref, axis=1)


def compute_rotation_deg(
    quaternion_ref: np.ndarray, quaternion_target: np.ndarray
) -> np.ndarray:
    """
    计算两个四元数之间的旋转角度差（度）

    使用公式: angle = 2 * arccos(|dot_product|)

    Args:
        quaternion_ref: shape (N, 4) 参考四元数 [qw, qx, qy, qz]
        quaternion_target: shape (N, 4) 目标四元数 [qw, qx, qy, qz]

    Returns:
        shape (N,) 的角度数组（度）
    """
    # 归一化四元数
    ref_norm = np.linalg.norm(quaternion_ref, axis=1, keepdims=True)
    target_norm = np.linalg.norm(quaternion_target, axis=1, keepdims=True)

    # 检查无效四元数（范数为0）
    valid = (ref_norm[:, 0] > 1e-8) & (target_norm[:, 0] > 1e-8)

    # 点积
    dots = np.sum(
        quaternion_ref[valid]
        / ref_norm[valid]
        * (quaternion_target[valid] / target_norm[valid]),
        axis=1,
    )

    # 计算角度（度），使用 2*arccos(|dot|) 避免 2π 问题
    result = np.zeros(len(quaternion_ref), dtype=np.float64)
    result[valid] = np.rad2deg(2.0 * np.arccos(np.clip(np.abs(dots), 0.0, 1.0)))

    return result


# =============================================================================
# Per-frame 计算函数（无时间依赖，可批量计算）
# =============================================================================


def compute_state_to_action_delta_per_frame(
    state: np.ndarray, action: np.ndarray
) -> dict[str, np.ndarray]:
    """
    计算每个 frame 的 state→action 差值（无时间依赖）

    Args:
        state: shape (T, 16) 状态序列
        action: shape (T, 16) 动作序列

    Returns:
        dict，包含 4 个 key 对应的 (T,) 数组
    """
    # 左臂: 位置 0:3, 四元数 3:7
    # 右臂: 位置 8:11, 四元数 11:15

    # 平移距离 = ||action_pos - state_pos||
    left_translation = compute_position_delta_m(state[:, 0:3], action[:, 0:3])
    right_translation = compute_position_delta_m(state[:, 8:11], action[:, 8:11])

    # 旋转角度
    left_rotation = compute_rotation_deg(state[:, 3:7], action[:, 3:7])
    right_rotation = compute_rotation_deg(state[:, 11:15], action[:, 11:15])

    return {
        "state_to_action_left_translation_m": left_translation,
        "state_to_action_right_translation_m": right_translation,
        "state_to_action_left_rotation_deg": left_rotation,
        "state_to_action_right_rotation_deg": right_rotation,
    }


# =============================================================================
# Per-episode 计算函数（使用 numpy 滑动窗口批量实现）
# =============================================================================


def compute_consecutive_action_delta_per_episode(
    action: np.ndarray,
) -> dict[str, np.ndarray]:
    """
    计算整个 episode 内连续动作之间的差值（使用滑动窗口）

    使用 np.diff 实现滑动窗口：action[1:] - action[:-1]

    Args:
        action: shape (T, 16) 动作序列

    Returns:
        dict，包含 4 个 key 对应的 (T,) 数组
        注意：第一帧的 consecutive 差值设为 0
    """
    if len(action) <= 1:
        # 只有一个 frame，返回全0
        return {
            "consecutive_action_left_translation_m": np.zeros(len(action)),
            "consecutive_action_right_translation_m": np.zeros(len(action)),
            "consecutive_action_left_rotation_deg": np.zeros(len(action)),
            "consecutive_action_right_rotation_deg": np.zeros(len(action)),
        }

    # 使用 np.diff 计算连续差分
    # action_diff[i] = action[i+1] - action[i]
    action_diff = np.diff(action, axis=0)

    # 计算平移距离（四元数分量的差值也参与 norm 计算，但这里只用位置分量）
    left_diff = action_diff[:, 0:3]  # shape (T-1, 3)
    right_diff = action_diff[:, 8:11]  # shape (T-1, 3)

    left_translation = np.linalg.norm(left_diff, axis=1)
    right_translation = np.linalg.norm(right_diff, axis=1)

    # 计算旋转角度（使用 diff 后的四元数差值）
    left_quat_diff = action_diff[:, 3:7]
    right_quat_diff = action_diff[:, 11:15]

    left_rotation = compute_rotation_deg(action[:-1, 3:7], action[1:, 3:7])
    right_rotation = compute_rotation_deg(action[:-1, 11:15], action[1:, 11:15])

    # 拼接第一帧（设为0）
    return {
        "consecutive_action_left_translation_m": np.concatenate(
            [[0], left_translation]
        ),
        "consecutive_action_right_translation_m": np.concatenate(
            [[0], right_translation]
        ),
        "consecutive_action_left_rotation_deg": np.concatenate([[0], left_rotation]),
        "consecutive_action_right_rotation_deg": np.concatenate([[0], right_rotation]),
    }


# =============================================================================
# Per-episode 处理的包装函数（用于 groupby.apply）
# =============================================================================


def _compute_episode_deltas(episode_df: pd.DataFrame) -> pd.DataFrame:
    """
    计算单个 episode 的 8 个 delta 指标

    Args:
        episode_df: 包含单个 episode 数据的 DataFrame

    Returns:
        包含 8 个 delta 列的 DataFrame（与输入等长）
    """
    # 转换数据为 numpy array
    state = np.array(episode_df["observation.state"].tolist())
    action = np.array(episode_df["action"].tolist())

    # Per-frame 计算：state_to_action（无时间依赖）
    sta_delta = compute_state_to_action_delta_per_frame(state, action)

    # Per-episode 计算：consecutive_action（使用滑动窗口）
    ca_delta = compute_consecutive_action_delta_per_episode(action)

    # 合并结果
    result = {**sta_delta, **ca_delta}
    return pd.DataFrame(result, index=episode_df.index)


# =============================================================================
# 主函数：向 DataFrame 添加 8 列
# =============================================================================


def add_action_delta_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    向 DataFrame 添加 8 列 action delta 指标

    使用 groupby.apply 批量处理，避免 for 循环遍历 episodes

    Args:
        df: step1 输出的 DataFrame，包含 episode_index, observation.state, action 等列

    Returns:
        添加了 8 列的新 DataFrame
    """
    # 定义新列名
    new_columns = [
        "state_to_action_left_translation_m",
        "state_to_action_right_translation_m",
        "state_to_action_left_rotation_deg",
        "state_to_action_right_rotation_deg",
        "consecutive_action_left_translation_m",
        "consecutive_action_right_translation_m",
        "consecutive_action_left_rotation_deg",
        "consecutive_action_right_rotation_deg",
    ]

    # 按 episode 分组，使用 apply 批量计算
    delta_df = df.groupby("episode_index", group_keys=False).apply(
        _compute_episode_deltas, include_groups=False
    )

    # 合并到原始 DataFrame
    result_df = pd.concat([df, delta_df[new_columns]], axis=1)

    return result_df


# =============================================================================
# 夹爪分析函数
# =============================================================================


def _compute_gripper_stats(values: np.ndarray) -> dict[str, Any]:
    """计算夹爪值的统计量"""
    return {
        "count": int(values.size),
        "min": float(np.min(values)),
        "p01": float(np.percentile(values, 1)),
        "p10": float(np.percentile(values, 10)),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "p99": float(np.percentile(values, 99)),
        "max": float(np.max(values)),
        "fraction_le_0.5": float(np.mean(values <= 0.5)),
        "fraction_ge_0.9": float(np.mean(values >= 0.9)),
    }


def _normalized_gripper_curve(action: np.ndarray, arm: int, points: int = 101) -> np.ndarray:
    """
    将夹爪动作归一化到 episode 进度 [0, 1]

    Args:
        action: shape (n_frame, 16) 的 action 数组
        arm: 0=左臂, 1=右臂
        points: 归一化后的采样点数

    Returns:
        shape (points,) 的归一化夹爪曲线
    """
    gripper_idx = LEFT_GRIPPER if arm == 0 else RIGHT_GRIPPER
    values = action[:, gripper_idx]
    destination = np.linspace(0.0, 1.0, points)
    source = np.linspace(0.0, 1.0, len(values))
    return np.interp(destination, source, values)


def _compute_all_episode_curves(
    df: pd.DataFrame, task_index: int, arm: int, points: int = 101
) -> np.ndarray:
    """
    计算 df 中同一 task 所有 episode 的归一化夹爪曲线

    Args:
        df: step1 输出的 DataFrame
        task_index: 任务编号
        arm: 0=左臂, 1=右臂
        points: 归一化后的采样点数

    Returns:
        shape (n_episodes, points) 的曲线矩阵
    """
    gripper_idx = LEFT_GRIPPER if arm == 0 else RIGHT_GRIPPER

    # 筛选同一 task 的 episode
    task_df = df[df["task_index"] == task_index]
    episode_indices = task_df["episode_index"].unique()

    curves = []
    for ep_idx in episode_indices:
        ep_action = np.array(task_df[task_df["episode_index"] == ep_idx]["action"].tolist())
        if len(ep_action) > 1:
            curve = _normalized_gripper_curve(ep_action, arm, points)
            curves.append(curve)

    return np.stack(curves) if curves else np.array([])


def analyze_episode_gripper(
    df: pd.DataFrame,
    episode_action: np.ndarray,
    task_index: int,
) -> tuple[dict[str, Any], plt.Figure]:
    """
    分析单个 episode 的夹爪行为

    Args:
        df: step1 输出的 DataFrame
        episode_action: shape (n_frame, 16) 的 action 数组
        task_index: 当前 episode 的任务编号

    Returns:
        (stats_dict, figure):
        - stats_dict: 包含 action/state 统计量和 per-episode 统计
        - figure: 4 面板可视化
    """
    n_frames = len(episode_action)

    # 在 df 中找到匹配的 episode
    episode_mask = (df["length"] == n_frames) & (df["task_index"] == task_index)
    if not episode_mask.any():
        raise ValueError(f"Cannot find episode with {n_frames} frames and task_index={task_index}")

    # 如果有多个匹配，取第一个
    matched_episodes = df[episode_mask]["episode_index"].unique()
    episode_index = int(matched_episodes[0])

    ep_df = df[df["episode_index"] == episode_index]

    # 提取夹爪值
    gripper_indices = [LEFT_GRIPPER, RIGHT_GRIPPER]
    train_action = episode_action[:, gripper_indices]
    episode_state = np.array(ep_df["observation.state"].tolist())[:, gripper_indices]

    # 计算 per-episode 统计
    per_episode = {}
    names = ("left", "right")
    for arm, name in enumerate(names):
        values = episode_action[:, gripper_indices[arm]]
        indices = np.flatnonzero(values <= 0.5)
        first_closed_progress = None
        if indices.size:
            first_closed_progress = float(indices[0] / max(len(values) - 1, 1))
        per_episode[name] = {
            "episodes_with_action_le_0.5": int(bool(indices.size)),
            "total_episodes": 1,
            "median_first_action_le_0.5_progress": first_closed_progress,
        }

    report = {
        "episode_index": episode_index,
        "task_index": task_index,
        "training_frame_count": int(len(episode_action)),
        "training": {
            "action": {
                "left": _compute_gripper_stats(train_action[:, 0]),
                "right": _compute_gripper_stats(train_action[:, 1]),
            },
            "state": {
                "left": _compute_gripper_stats(episode_state[:, 0]),
                "right": _compute_gripper_stats(episode_state[:, 1]),
            },
            "per_episode": per_episode,
        },
    }

    # 生成可视化
    figure, axes = plt.subplots(2, 2, figsize=(15, 10))
    colors = ("tab:blue", "tab:orange")

    for arm, (name, color) in enumerate(zip(names, colors)):
        gripper_idx = gripper_indices[arm]

        # 左上: 夹爪动作分布直方图（所有同 task episode）
        task_df = df[df["task_index"] == task_index]
        all_task_actions = np.array(task_df["action"].tolist())[:, gripper_idx]
        axes[0, 0].hist(
            all_task_actions,
            bins=np.linspace(0, 1, 51),
            alpha=0.45,
            color=color,
            label=f"Task {name}",
        )
        axes[0, 0].axvline(
            np.median(train_action[:, arm]),
            color=color,
            linestyle="--",
            label=f"Episode Median {name}",
        )

        # 右上: 夹爪时序图
        frame_indices = np.arange(len(episode_action))
        axes[0, 1].plot(
            frame_indices,
            episode_action[:, gripper_idx],
            color=color,
            label=f"{name} action",
        )
        axes[0, 1].plot(
            frame_indices,
            episode_state[:, arm],
            color=color,
            linestyle=":",
            alpha=0.7,
            label=f"{name} state",
        )

        # 左下: 归一化曲线 + P10-P90 区间（基于 df 中同 task 所有 episode）
        all_curves = _compute_all_episode_curves(df, task_index, arm)
        episode_curve = _normalized_gripper_curve(episode_action, arm)
        x = np.linspace(0, 100, len(episode_curve))

        if len(all_curves) > 0:
            median = np.median(all_curves, axis=0)
            low, high = np.percentile(all_curves, [10, 90], axis=0)
            axes[1, 0].fill_between(x, low, high, color=color, alpha=0.18, label=f"Task P10-P90")
            axes[1, 0].plot(x, median, color=color, linestyle="--", alpha=0.7, label=f"Task Median")

        axes[1, 0].plot(x, episode_curve, color=color, linewidth=2, label=f"Episode {name}")

        # 右下: 夹爪值在 0.9-1.0 范围的直方图
        axes[1, 1].hist(
            train_action[:, arm],
            bins=np.linspace(0.9, 1.0, 51),
            alpha=0.45,
            color=color,
            label=f"Episode {name}",
        )

    # 设置各面板标题和标签
    axes[0, 0].set_title(f"Task {task_index} action gripper distribution")
    axes[0, 0].set_xlabel("Gripper action")
    axes[0, 0].set_ylabel("Frames")
    axes[0, 0].set_yscale("log")

    axes[0, 1].set_title("Episode gripper timeline")
    axes[0, 1].set_xlabel("Frame index")
    axes[0, 1].set_ylabel("Gripper value")

    axes[1, 0].set_title(f"Normalized gripper action over episode progress (task {task_index})")
    axes[1, 0].set_xlabel("Episode progress (%)")
    axes[1, 0].set_ylabel("Gripper action")

    axes[1, 1].set_title("Gripper action detail near open state (0.9-1.0)")
    axes[1, 1].set_xlabel("Gripper action")
    axes[1, 1].set_ylabel("Frames")

    for axis in axes.flat:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)

    figure.tight_layout()

    return report, figure


# =============================================================================
# 视频合成函数
# =============================================================================

import shutil
import subprocess
from dataclasses import dataclass


CAMERAS = (
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)


@dataclass(frozen=True)
class VideoSegment:
    path: str
    start: float
    duration: float


def compose_episode_video(
    df: pd.DataFrame,
    episode_index: int,
    output_path: str,
    crf: int = 23,
) -> None:
    """
    将单个 episode 的三个相机视频合成为一个马赛克视频

    Args:
        df: step1 输出的 DataFrame
        episode_index: episode 编号
        output_path: 输出 MP4 路径
        crf: H.264 质量 (越低越好/文件越大, 默认23)

    Returns:
        None (直接生成视频文件)
    """
    import json
    from pathlib import Path

    # 检查 ffmpeg
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found in PATH")

    # 获取 episode 数据
    ep_df = df[df["episode_index"] == episode_index]
    if len(ep_df) == 0:
        raise ValueError(f"Episode {episode_index} not found in df")

    # 获取视频信息和 frame count
    row = ep_df.iloc[0]
    frame_count = int(row["length"])
    fps = 25  # 从数据集 metadata 可获取，此处固定

    # 构建 video segments
    # 列名映射: CAMERAS -> df 列名前缀
    camera_prefix = {
        "observation.images.cam_high": "high_video",
        "observation.images.cam_left_wrist": "left_video",
        "observation.images.cam_right_wrist": "right_video",
    }

    segments = []
    for camera in CAMERAS:
        prefix = camera_prefix[camera]
        from_ts = row[f"{prefix}_from_timestamp"]
        to_ts = row[f"{prefix}_to_timestamp"]
        video_path = row[f"{prefix}_path"]

        segments.append(
            VideoSegment(
                path=str(Path(df.attrs.get("dataset_root", ".")) / video_path),
                start=float(from_ts),
                duration=float(to_ts) - float(from_ts),
            )
        )

    # 检查文件存在
    missing = [seg.path for seg in segments if not Path(seg.path).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing video(s):\n" + "\n".join(missing))

    # 确保输出目录存在
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 构建 ffmpeg 命令
    command = ["ffmpeg", "-hide_banner", "-y"]
    for segment in segments:
        command += [
            "-ss",
            f"{segment.start:.6f}",
            "-t",
            f"{segment.duration:.6f}",
            "-i",
            segment.path,
        ]

    # 640x480 overhead view above two 320x240 wrist views
    command += [
        "-filter_complex",
        (
            "[0:v]scale=640:480,setsar=1[top];"
            "[1:v]scale=320:240,setsar=1[left];"
            "[2:v]scale=320:240,setsar=1[right];"
            "[left][right]hstack=inputs=2[bottom];"
            "[top][bottom]vstack=inputs=2[out]"
        ),
        "-map",
        "[out]",
        "-frames:v",
        str(frame_count),
        "-r",
        f"{fps}",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    ]

    print(f"Episode {episode_index}: frames={frame_count}, fps={fps}")
    for camera, segment in zip(CAMERAS, segments):
        print(
            f"  {camera}: {segment.path}, "
            f"{segment.start:.3f}s..{segment.start + segment.duration:.3f}s"
        )

    subprocess.run(command, check=True)
    print(f"Saved: {output_path}")


# =============================================================================
# Task 分布统计函数
# =============================================================================

def get_task_names(dataset_root: str) -> dict[int, str]:
    """
    从 tasks.parquet 读取任务名称映射

    Args:
        dataset_root: 数据集根目录

    Returns:
        {task_index: task_name, ...}
    """
    from pathlib import Path
    import pyarrow.parquet as pq

    tasks_path = Path(dataset_root) / "meta" / "tasks.parquet"
    if not tasks_path.is_file():
        return {}

    pf = pq.ParquetFile(tasks_path)
    table = pf.read()
    tasks_df = table.to_pandas()

    task_names = {}
    for _, row in tasks_df.iterrows():
        task_idx = int(row["task_index"])
        # task_name 在 index 中，index 名称是 __index_level_0__
        task_name = str(row.name) if row.name else f"task_{task_idx}"
        task_names[task_idx] = task_name

    return task_names


def analyze_task_distribution(df: pd.DataFrame, dataset_root: str) -> pd.DataFrame:
    """
    统计每个 task 的 episode 数量和总帧数

    Args:
        df: step1 输出的 DataFrame
        dataset_root: 数据集根目录（用于读取 tasks.parquet）

    Returns:
        DataFrame，包含 task_index, task_name, episode_count, frame_count
    """
    # 获取任务名称映射
    task_names = get_task_names(dataset_root)

    # 按 task_index 分组统计
    # 每个 episode 的 length 是相同的，取第一个即可
    task_stats = (
        df.groupby("task_index")
        .agg(
            episode_count=("episode_index", "nunique"),
            frame_count=("length", "first"),  # length 是定值
        )
        .reset_index()
    )

    # 重新计算 frame_count（每个 episode 的 length * episode_count 不对，因为 df 是按 frame 展开的）
    frame_counts = df.groupby("task_index").size().reset_index(name="total_frames")
    task_stats = task_stats.merge(frame_counts, on="task_index")
    task_stats = task_stats.drop(columns=["frame_count"])
    task_stats = task_stats.rename(columns={"total_frames": "frame_count"})

    # 添加 task_name
    task_stats["task_name"] = task_stats["task_index"].map(
        lambda x: task_names.get(x, f"task_{x}")
    )

    # 排序列
    task_stats = task_stats[["task_index", "task_name", "episode_count", "frame_count"]]

    return task_stats
