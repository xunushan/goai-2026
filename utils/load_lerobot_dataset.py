"""
LeRobot v3.0 数据格式转换为 pandas DataFrame

功能：
- 读取 lerobot_v30_ee 格式数据集
- 支持随机抽取 n 个 episode 或按比例抽取
- 输出宽表格式的 DataFrame

输出列：
- episode_index, length, observation.state, action, task_index, frame_index
- high_video_path, left_video_path, right_video_path
- high_video_from_timestamp, high_video_to_timestamp
- left_video_from_timestamp, left_video_to_timestamp
- right_video_from_timestamp, right_video_to_timestamp

Usage:
    conda run -n lerobot python utils/load_lerobot_dataset.py data/lerobot_v30_ee 100
    conda run -n lerobot python utils/load_lerobot_dataset.py data/lerobot_v30_ee 0.1
"""

import json
import random
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


# =============================================================================
# 常量定义
# =============================================================================

# 视频字段名与相机视角的映射关系
VIDEO_FIELDS = {
    "cam_high": "observation.images.cam_high",
    "cam_left_wrist": "observation.images.cam_left_wrist",
    "cam_right_wrist": "observation.images.cam_right_wrist",
}

# 需要的 episode 元信息列（只读取这些列，避免读取全部 121 列）
EPISODE_META_COLUMNS = [
    "episode_index",
    "length",
    "dataset_from_index",
    "dataset_to_index",
    "videos/observation.images.cam_high/chunk_index",
    "videos/observation.images.cam_high/file_index",
    "videos/observation.images.cam_high/from_timestamp",
    "videos/observation.images.cam_high/to_timestamp",
    "videos/observation.images.cam_left_wrist/chunk_index",
    "videos/observation.images.cam_left_wrist/file_index",
    "videos/observation.images.cam_left_wrist/from_timestamp",
    "videos/observation.images.cam_left_wrist/to_timestamp",
    "videos/observation.images.cam_right_wrist/chunk_index",
    "videos/observation.images.cam_right_wrist/file_index",
    "videos/observation.images.cam_right_wrist/from_timestamp",
    "videos/observation.images.cam_right_wrist/to_timestamp",
]

# 主数据表的列
MAIN_DATA_COLUMNS = [
    "episode_index",
    "frame_index",
    "observation.state",
    "action",
    "task_index",
]


# =============================================================================
# 数据读取函数
# =============================================================================

def _read_info_json(dataset_path: Path) -> dict:
    """读取 info.json 获取数据集配置"""
    info_path = dataset_path / "meta" / "info.json"
    with open(info_path, "r") as f:
        return json.load(f)


def _read_episode_meta(dataset_path: Path, selected_episodes: set[int]) -> pd.DataFrame:
    """
    读取 episode 元信息表并筛选指定的 episodes

    Args:
        dataset_path: 数据集根目录
        selected_episodes: 要筛选的 episode 索引集合

    Returns:
        筛选后的 episode 元信息 DataFrame
    """
    meta_path = dataset_path / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    pf = pq.ParquetFile(meta_path)
    table = pf.read(columns=EPISODE_META_COLUMNS)
    df = table.to_pandas()
    df = df[df["episode_index"].isin(selected_episodes)]
    return df


def _read_main_data(
    dataset_path: Path,
    data_path_template: str,
    selected_episodes: set[int],
) -> pd.DataFrame:
    """
    读取主数据表并筛选指定的 episodes

    Args:
        dataset_path: 数据集根目录
        data_path_template: 数据文件路径模板（来自 info.json）
        selected_episodes: 要筛选的 episode 索引集合

    Returns:
        筛选后的主数据 DataFrame
    """
    data_path = dataset_path / data_path_template.format(chunk_index=0, file_index=0)
    pf = pq.ParquetFile(data_path)
    table = pf.read()
    df = table.to_pandas()
    df = df[df["episode_index"].isin(selected_episodes)]
    return df


# =============================================================================
# 视频信息处理函数
# =============================================================================

def _build_video_info_map(episodes_df: pd.DataFrame) -> dict[int, dict]:
    """
    从 episode 元信息构建视频路径和时间戳映射

    Args:
        episodes_df: episode 元信息 DataFrame

    Returns:
        {episode_index: {
            "high_video_path": ...,
            "left_video_path": ...,
            "right_video_path": ...,
            "high_video_from_timestamp": ...,
            "high_video_to_timestamp": ...,
            "left_video_from_timestamp": ...,
            "left_video_to_timestamp": ...,
            "right_video_from_timestamp": ...,
            "right_video_to_timestamp": ...,
            "length": ...,
        }, ...}
    """
    video_info = {}
    for _, row in episodes_df.iterrows():
        ep_idx = int(row["episode_index"])

        # 俯视相机
        high_chunk = int(row["videos/observation.images.cam_high/chunk_index"])
        high_file = int(row["videos/observation.images.cam_high/file_index"])

        # 左腕相机
        left_chunk = int(row["videos/observation.images.cam_left_wrist/chunk_index"])
        left_file = int(row["videos/observation.images.cam_left_wrist/file_index"])

        # 右腕相机
        right_chunk = int(row["videos/observation.images.cam_right_wrist/chunk_index"])
        right_file = int(row["videos/observation.images.cam_right_wrist/file_index"])

        video_info[ep_idx] = {
            "high_video_path": (
                f"videos/observation.images.cam_high/"
                f"chunk-{high_chunk:03d}/file-{high_file:03d}.mp4"
            ),
            "left_video_path": (
                f"videos/observation.images.cam_left_wrist/"
                f"chunk-{left_chunk:03d}/file-{left_file:03d}.mp4"
            ),
            "right_video_path": (
                f"videos/observation.images.cam_right_wrist/"
                f"chunk-{right_chunk:03d}/file-{right_file:03d}.mp4"
            ),
            "high_video_from_timestamp": row["videos/observation.images.cam_high/from_timestamp"],
            "high_video_to_timestamp": row["videos/observation.images.cam_high/to_timestamp"],
            "left_video_from_timestamp": row["videos/observation.images.cam_left_wrist/from_timestamp"],
            "left_video_to_timestamp": row["videos/observation.images.cam_left_wrist/to_timestamp"],
            "right_video_from_timestamp": row["videos/observation.images.cam_right_wrist/from_timestamp"],
            "right_video_to_timestamp": row["videos/observation.images.cam_right_wrist/to_timestamp"],
            "length": int(row["length"]),
        }

    return video_info


def _merge_video_info(
    main_df: pd.DataFrame,
    video_info: dict[int, dict],
) -> pd.DataFrame:
    """
    将视频信息合并到主数据表

    Args:
        main_df: 主数据 DataFrame
        video_info: 视频信息映射

    Returns:
        合并视频信息后的 DataFrame
    """
    # 为每一行添加视频信息
    video_rows = []
    for ep_idx in main_df["episode_index"]:
        video_rows.append(video_info.get(ep_idx, {}))

    # 注意：必须 reset_index，否则 concat 时会因为索引不同导致数据错位
    video_df = pd.DataFrame(video_rows).reset_index(drop=True)
    main_subset = main_df[MAIN_DATA_COLUMNS].reset_index(drop=True)

    result_df = pd.concat([main_subset, video_df], axis=1)

    # 添加 length 列（从 video_info 中获取）
    result_df["length"] = result_df["episode_index"].map(
        lambda x: video_info.get(x, {}).get("length", 0)
    )

    return result_df


# =============================================================================
# 数据类型处理函数
# =============================================================================

def _convert_numpy_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    将 numpy array 转换为 list，便于 pandas 存储和后续处理

    Args:
        df: 输入 DataFrame

    Returns:
        转换后的 DataFrame
    """
    def to_list(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        return x

    df["observation.state"] = df["observation.state"].apply(to_list)
    df["action"] = df["action"].apply(to_list)

    return df


def _reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    重新排列 DataFrame 的列顺序

    Args:
        df: 输入 DataFrame

    Returns:
        重新排序列后的 DataFrame
    """
    columns_order = [
        "episode_index",
        "length",
        "observation.state",
        "action",
        "task_index",
        "frame_index",
        "high_video_path",
        "left_video_path",
        "right_video_path",
        "high_video_from_timestamp",
        "high_video_to_timestamp",
        "left_video_from_timestamp",
        "left_video_to_timestamp",
        "right_video_from_timestamp",
        "right_video_to_timestamp",
    ]
    return df[columns_order]


# =============================================================================
# 主函数
# =============================================================================

def load_lerobot_as_dataframe(
    dataset_path: str,
    n_episodes: Optional[int] = None,
    ratio: Optional[float] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    将 LeRobot v3.0 数据集转换为 pandas DataFrame

    Args:
        dataset_path: 数据集根目录，如 "data/lerobot_v30_ee"
        n_episodes: 随机抽取的 episode 数量（与 ratio 二选一）
        ratio: 随机抽取的比例（0~1），如 0.1 表示抽取 10% 的 episodes
        seed: 随机种子

    Returns:
        pandas DataFrame，每行代表一帧，包含以下列：
        - episode_index: episode 编号
        - length: 该 episode 的总帧数
        - observation.state: 16维状态向量（list）
        - action: 16维动作向量（list）
        - task_index: 任务编号 (0~11)
        - frame_index: episode 内帧序号
        - high_video_path: 俯视相机视频相对路径
        - left_video_path: 左腕相机视频相对路径
        - right_video_path: 右腕相机视频相对路径
        - high_video_from_timestamp: 俯视视频起始时间戳
        - high_video_to_timestamp: 俯视视频结束时间戳
        - left_video_from_timestamp: 左腕视频起始时间戳
        - left_video_to_timestamp: 左腕视频结束时间戳
        - right_video_from_timestamp: 右腕视频起始时间戳
        - right_video_to_timestamp: 右腕视频结束时间戳
    """
    root = Path(dataset_path)

    # -------------------------------------------------------------------------
    # Step 1: 读取 info.json 获取数据集配置
    # -------------------------------------------------------------------------
    info = _read_info_json(root)
    total_episodes = info["total_episodes"]
    data_path_template = info["data_path"]

    # -------------------------------------------------------------------------
    # Step 2: 确定要抽取的 episode 索引
    # -------------------------------------------------------------------------
    if n_episodes is not None:
        n_to_sample = min(n_episodes, total_episodes)
    elif ratio is not None:
        n_to_sample = max(1, int(total_episodes * ratio))
    else:
        n_to_sample = total_episodes  # 默认加载全部

    random.seed(seed)
    selected_episodes = sorted(random.sample(range(total_episodes), n_to_sample))
    selected_set = set(selected_episodes)

    # -------------------------------------------------------------------------
    # Step 3: 读取 episode 元信息表
    # -------------------------------------------------------------------------
    episodes_df = _read_episode_meta(root, selected_set)

    # -------------------------------------------------------------------------
    # Step 4: 构建视频路径和时间戳映射
    # -------------------------------------------------------------------------
    video_info = _build_video_info_map(episodes_df)

    # -------------------------------------------------------------------------
    # Step 5: 读取主数据表
    # -------------------------------------------------------------------------
    main_df = _read_main_data(root, data_path_template, selected_set)

    # -------------------------------------------------------------------------
    # Step 6: 合并视频信息
    # -------------------------------------------------------------------------
    result_df = _merge_video_info(main_df, video_info)

    # -------------------------------------------------------------------------
    # Step 7: 数据类型处理
    # -------------------------------------------------------------------------
    result_df = _convert_numpy_types(result_df)
    result_df = _reorder_columns(result_df)

    return result_df


# =============================================================================
# 命令行入口
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="LeRobot v3.0 数据集 -> pandas DataFrame 宽表, 可选保存 CSV",
    )
    parser.add_argument("dataset_path", help="数据集根目录, 如 data/lerobot_v30_ee")
    parser.add_argument("n_or_ratio", nargs="?", default=None,
                        help="抽取数量(整数)或比例(浮点, 0~1); 缺省加载全部 episode")
    parser.add_argument("--output", default=None,
                        help="保存 CSV 路径 (如 data/real_lerobot_v30_ee.csv); "
                             "缺省只打印摘要")
    parser.add_argument("--episodes", default=None,
                        help="只加载这些 episode (逗号分隔白名单), 与 n_or_ratio 互斥")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    # 解析抽取参数
    if args.episodes is not None:
        want = [int(e) for e in args.episodes.split(",") if e.strip()]

        def load(dataset_path, **kwargs):
            df = load_lerobot_as_dataframe(dataset_path, **kwargs)
            return df[df["episode_index"].isin(want)].reset_index(drop=True)

    elif args.n_or_ratio is not None:
        val = args.n_or_ratio
        if "." in val:
            load = lambda p, **k: load_lerobot_as_dataframe(p, ratio=float(val), **k)
        else:
            load = lambda p, **k: load_lerobot_as_dataframe(p, n_episodes=int(val), **k)
    else:
        load = load_lerobot_as_dataframe

    df = load(args.dataset_path, seed=args.seed)

    # 保存 CSV
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        print(f"\nCSV 已保存 -> {out}  ({out.stat().st_size / 1e6:.1f} MB)")

    # 打印结果摘要
    print(f"DataFrame shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Episode range: {df['episode_index'].min()} ~ {df['episode_index'].max()}")
    print(f"Unique episodes: {df['episode_index'].nunique()}")
    print(f"\nFirst few rows:")
    print(df.head())
