"""
左右爪夹 gripper 分布直方图（按任务分组）

功能：
- 读取 lerobot_v30_ee.csv
- 按 task_index 分任务统计左/右爪夹 (l_g, r_g) 的分布直方图
- 输出图像到 outputs/，代码保留在 scripts/

Usage:
    conda run -n lerobot python scripts/plot_gripper_histograms.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# l_g = index 7, r_g = index 15
L_GRIPPER_IDX = 7
R_GRIPPER_IDX = 15


def extract_gripper(df: pd.DataFrame) -> pd.DataFrame:
    """从 observation.state 提取左右爪夹值"""
    df = df.copy()
    df["l_g"] = df["observation.state"].apply(lambda x: x[L_GRIPPER_IDX] if isinstance(x, (list, np.ndarray)) else np.nan)
    df["r_g"] = df["observation.state"].apply(lambda x: x[R_GRIPPER_IDX] if isinstance(x, (list, np.ndarray)) else np.nan)
    return df


def plot_histograms(df: pd.DataFrame, output_dir: str = "outputs"):
    """按任务绘制左右爪夹分布直方图"""
    import os
    os.makedirs(output_dir, exist_ok=True)

    tasks = sorted(df["task_index"].unique())
    n_tasks = len(tasks)

    # 1. 合在一起的全局 histogram（所有任务堆叠）
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Gripper Distribution (All Tasks Combined)", fontsize=14)

    for ax, (gripper_name, gripper_col) in zip(axes, [("Left Gripper (l_g)", "l_g"), ("Right Gripper (r_g)", "r_g")]):
        for task in tasks:
            subset = df[df["task_index"] == task][gripper_col].dropna()
            ax.hist(subset, bins=50, alpha=0.5, label=f"Task {task}", density=True)
        ax.set_xlabel(gripper_name)
        ax.set_ylabel("Density")
        ax.set_title(gripper_name)
        ax.legend(fontsize=6, ncol=3, loc="upper right")

    plt.tight_layout()
    plt.savefig(f"{output_dir}/gripper_histogram_all_tasks.png", dpi=150)
    plt.close()
    print(f"Saved {output_dir}/gripper_histogram_all_tasks.png")

    # 2. 每个任务单独一行，l_g 和 r_g 并排
    n_cols = 2
    n_rows = n_tasks
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 3 * n_rows))
    if n_tasks == 1:
        axes = axes.reshape(1, -1)

    for row, task in enumerate(tasks):
        subset = df[df["task_index"] == task]
        l_g_vals = subset["l_g"].dropna()
        r_g_vals = subset["r_g"].dropna()

        axes[row, 0].hist(l_g_vals, bins=50, color="steelblue", alpha=0.7)
        axes[row, 0].set_ylabel(f"Task {task}")
        axes[row, 0].set_xlabel("Left Gripper (l_g)")
        axes[row, 0].set_title(f"Task {task} - Left Gripper")

        axes[row, 1].hist(r_g_vals, bins=50, color="darkorange", alpha=0.7)
        axes[row, 1].set_ylabel(f"Task {task}")
        axes[row, 1].set_xlabel("Right Gripper (r_g)")
        axes[row, 1].set_title(f"Task {task} - Right Gripper")

    plt.tight_layout()
    plt.savefig(f"{output_dir}/gripper_histogram_by_task.png", dpi=150)
    plt.close()
    print(f"Saved {output_dir}/gripper_histogram_by_task.png")


if __name__ == "__main__":
    print("Loading data...")
    df = pd.read_csv("data/lerobot_v30_ee.csv", converters={"observation.state": eval})

    print("Extracting gripper values...")
    df = extract_gripper(df)

    print("Plotting histograms...")
    plot_histograms(df, output_dir="outputs")
    print("Done!")
