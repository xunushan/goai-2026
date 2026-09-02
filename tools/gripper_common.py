#!/usr/bin/env python3
"""gripper 分析共享工具 (开发期放 tools/, 收尾再决定去留/归位)。

集中: state 16 维常量、CSV 加载、时间归一化插值、matplotlib 中文字体配置。
供 tools/gripper_interp_viz.py 与 tools/gripper_cluster.py 复用。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# state 16 维结构 (与 tools/episode_state_insight.py 一致)
#   left_ee_pose(7) + left gripper + right_ee_pose(7) + right gripper
# ---------------------------------------------------------------------------
GRIP_L = 7     # 左臂 gripper 在 state 向量中的下标 (0 闭 ~ 1 开)
GRIP_R = 15    # 右臂 gripper
N_DIM = 100    # 每臂时间归一化插值目标维数

# 绘图配色: 左臂蓝 / 右臂橙
C_L = "#1f77b4"
C_R = "#ff7f0e"

# matplotlib 中文渲染候选字体 (按优先级), macOS 可用
CJK_FONTS = ["PingFang SC", "Hiragino Sans GB", "Heiti SC", "Arial Unicode MS", "Songti SC"]


def setup_cjk_font() -> str:
    """把 matplotlib 默认字体切到支持中文的字体, 返回实际选中的字体名。

    需在 pyplot 首次创建 figure 前调用。同时关闭坐标负号的 unicode_minus 以免
    负号显示为方块。
    """
    import matplotlib
    from matplotlib import font_manager
    chosen = None
    for name in CJK_FONTS:
        try:
            font_manager.findfont(name, fallback_to_default=False)
            chosen = name
            break
        except Exception:
            continue
    if chosen is None:  # 找不到则尝试字体列表里自动补齐
        chosen = "sans-serif"
    else:
        rc_sans = list(matplotlib.rcParams.get("font.sans-serif", []))
        if chosen in rc_sans:
            rc_sans.remove(chosen)
        matplotlib.rcParams["font.sans-serif"] = [chosen, *rc_sans]
    matplotlib.rcParams["font.family"] = "sans-serif"
    matplotlib.rcParams["axes.unicode_minus"] = False
    return chosen


def load_tasks(meta_path: Path) -> dict[int, str]:
    """读取 tasks.parquet -> {task_index: 指令文本} (指令在 index, task_index 在列)。"""
    df = pd.read_parquet(meta_path)
    return {int(row["task_index"]): str(idx) for idx, row in df.iterrows()}


def load_grippers(csv: Path) -> pd.DataFrame:
    """读 CSV, 每 episode 汇总为一行。

    返回列: task_index, episode_index, length, grip_L, grip_R
        grip_L / grip_R 为该 episode 左右臂原始 gripper 序列 (按帧)。
    """
    df = pd.read_csv(csv)
    # 直接按逗号切分字符串化 list, 取第 GRIP_L / GRIP_R 个 token (末 token 带 "]")
    def _col(idx: int) -> np.ndarray:
        out = []
        for s in df["observation.state"]:
            out.append(float(s.split(",")[idx].rstrip("]")))
        return np.asarray(out)

    df["grip_L"] = _col(GRIP_L)
    df["grip_R"] = _col(GRIP_R)
    rows = []
    for (t, e), g in df.groupby(["task_index", "episode_index"]):
        rows.append({
            "task_index": int(t),
            "episode_index": int(e),
            "length": int(g["length"].iloc[0]),
            "grip_L": g["grip_L"].to_numpy(float),
            "grip_R": g["grip_R"].to_numpy(float),
        })
    return pd.DataFrame(rows)


def interp_100(x: np.ndarray, n: int = N_DIM) -> np.ndarray:
    """把任意长度一维序列等间距时间归一化到 n 维 (np.interp, 保留端点)。"""
    m = len(x)
    if m == 1:
        return np.full(n, float(x[0]))
    src = np.linspace(0, m - 1, m)
    dst = np.linspace(0, m - 1, n)
    return np.interp(dst, src, x)


def episode_feature_L100_R100(grip_L: np.ndarray, grip_R: np.ndarray) -> np.ndarray:
    """把单个 episode 的左右 gripper 拼成聚类特征: [L100(100), R100(100)] -> 200 维。"""
    return np.concatenate([interp_100(grip_L), interp_100(grip_R)])
