"""
左右爪夹 gripper 时序图（按任务分组）

功能：
- 读取 lerobot_v30_ee.csv
- 每个任务随机抽取 3 个 episode
- 按 frame_index 绘制时序散点图（不用连线，便于观察是突变还是连续变化）
- 输出图像到 outputs/，代码保留在 scripts/

Usage:
    conda run -n lerobot python scripts/plot_gripper_timeseries.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

# l_g = index 7, r_g = index 15
L_GRIPPER_IDX = 7
R_GRIPPER_IDX = 15


def extract_gripper(df: pd.DataFrame) -> pd.DataFrame:
    """从 observation.state 提取左右爪夹值"""
    df = df.copy()
    df["l_g"] = df["observation.state"].apply(lambda x: x[L_GRIPPER_IDX] if isinstance(x, (list, np.ndarray)) else np.nan)
    df["r_g"] = df["observation.state"].apply(lambda x: x[R_GRIPPER_IDX] if isinstance(x, (list, np.ndarray)) else np.nan)
    return df


def plot_timeseries(df: pd.DataFrame, output_dir: str = "outputs", seed: int = 42):
    """
    按任务绘制时序散点图

    每个任务 3 个 episode，l_g 和 r_g 并排显示
    使用 scatter 而不是 plot，便于观察突变 vs 连续变化
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    random.seed(seed)
    tasks = sorted(df["task_index"].unique())

    # 颜色映射：每个 episode 用不同颜色
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
              "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

    for task in tasks:
        task_df = df[df["task_index"] == task]
        episodes = task_df["episode_index"].unique()
        # 随机选 3 个 episode（如果不足 3 个就全部选）
        n_select = min(3, len(episodes))
        selected_eps = random.sample(list(episodes), n_select)

        fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        fig.suptitle(f"Task {task} - Gripper Time Series ({n_select} episodes)", fontsize=13)

        for ep_idx, ep in enumerate(selected_eps):
            ep_data = task_df[task_df["episode_index"] == ep].sort_values("frame_index")
            color = colors[ep_idx % len(colors)]

            # 左爪夹
            axes[0].scatter(ep_data["frame_index"], ep_data["l_g"],
                           color=color, alpha=0.7, s=8,
                           label=f"Ep {ep}")
            # 右爪夹
            axes[1].scatter(ep_data["frame_index"], ep_data["r_g"],
                           color=color, alpha=0.7, s=8,
                           label=f"Ep {ep}")

        axes[0].set_ylabel("Left Gripper (l_g)")
        axes[0].legend(fontsize=8, loc="upper right")
        axes[1].set_ylabel("Right Gripper (r_g)")
        axes[1].set_xlabel("Frame Index")
        axes[1].legend(fontsize=8, loc="upper right")

        plt.tight_layout()
        output_path = f"{output_dir}/gripper_timeseries_task{task}.png"
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"Saved {output_path}")

    # 全局汇总图：所有任务的所有 episode 叠加在一起
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle("All Tasks - Gripper Time Series Overlay", fontsize=13)

    for task in tasks:
        task_df = df[df["task_index"] == task]
        episodes = task_df["episode_index"].unique()
        n_select = min(3, len(episodes))
        selected_eps = random.sample(list(episodes), n_select)

        for ep in selected_eps:
            ep_data = task_df[task_df["episode_index"] == ep].sort_values("frame_index")
            axes[0].scatter(ep_data["frame_index"], ep_data["l_g"], alpha=0.3, s=3, label=f"Task {task}")
            axes[1].scatter(ep_data["frame_index"], ep_data["r_g"], alpha=0.3, s=3, label=f"Task {task}")

    axes[0].set_ylabel("Left Gripper (l_g)")
    axes[1].set_ylabel("Right Gripper (r_g)")
    axes[1].set_xlabel("Frame Index")
    # 只保留一个图例
    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    axes[0].legend(by_label.values(), by_label.keys(), fontsize=6, ncol=6, loc="upper right")
    axes[1].legend(by_label.values(), by_label.keys(), fontsize=6, ncol=6, loc="upper right")

    plt.tight_layout()
    output_path = f"{output_dir}/gripper_timeseries_all_tasks.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved {output_path}")


if __name__ == "__main__":
    print("Loading data...")
    df = pd.read_csv("data/lerobot_v30_ee.csv", converters={"observation.state": eval})

    print("Extracting gripper values...")
    df = extract_gripper(df)

    print("Plotting time series...")
    plot_timeseries(df, output_dir="outputs", seed=42)
    print("Done!")
