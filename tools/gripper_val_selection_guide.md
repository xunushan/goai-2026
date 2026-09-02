# 遥操数据验证集选取指南（左右爪夹聚类 → 分层抽样）

> 针对 `data/sim_lerobot_v30_ee.csv`（sim 遥操数据, 3 任务 × 100 episode, 每行一帧）。
> 目标: 每个任务从 100 个 episode 中按**爪夹开合行为**分层选出 10 个验证集,
> 使验证集覆盖该任务不同的左右手分工/操作时序风格。

所有代码位于 `tools/`, 共享逻辑集中在 `tools/gripper_common.py`, 均可用 `python tools/<脚本>.py` 运行。

---

## 0. 数据说明

`observation.state` 每帧 16 维（与 tools/episode_state_insight.py 同约定）:

```
left_ee_pose(7) + left_gripper(1, idx=7) + right_ee_pose(7) + right_gripper(1, idx=15)
```

gripper 取值 0(闭)~1(开)。三个任务的指令见 `data/sim_lerobot_v30_ee/meta/tasks.parquet`。

---

## 1. 完整流程（四步 + 产出 JSON）

| 步骤 | 脚本 | 做什么 | 产物 |
|---|---|---|---|
| ① 插值前/后可视化 | `tools/gripper_interp_viz.py` | 每 episode 左右爪夹各插值到 100 点, 出图确认方法忠实 | `outputs/gripper_interp_viz/*.png` |
| ② 单任务聚类演示 | `tools/gripper_cluster.py --task 0` | KMeans + 轮廓系数自动选 K, 出二维分布/每类曲线图 | `outputs/gripper_cluster/task0/*` |
| ③ 全部任务聚类 | `tools/gripper_cluster.py --tasks 0 1 2` | 同上推广到其余任务 | `outputs/gripper_cluster/task{1,2}/*` |
| ④ 分层抽样验证集 | `tools/gripper_select_val.py` | 每任务 10 个名额按类占比分配, 类内固定种子随机抽 | `outputs/val_sets/task*_val_manifest.csv`, `val_manifest.csv` |
| ⑤ 生成划分 JSON | `tools/gripper_build_split.py` | 从 CSV 一步复现聚类+抽样, 组装正式 JSON | `data/sim_lerobot_v30_ee/train_val_split.json` |

> ③④⑤ 均为**确定性可复现**: 聚类 seed=0, 抽样 seed=42。
> ④的结果不落盘也没关系——⑤会从原始 CSV 重算并产出同一套划分。

### 常用命令

```bash
# ① 插值前后对比图（默认读 data/sim_lerobot_v30_ee.csv）
python tools/gripper_interp_viz.py

# ③ 聚类全部任务（task0 演示: --task 0）
python tools/gripper_cluster.py --tasks 0 1 2

# ④ 分层抽样验证集（读 outputs/gripper_cluster/task*/clusters.csv）
python tools/gripper_select_val.py --n-val 10 --seed 42

# ⑤ 生成 train/val 划分 JSON（单命令复现全部）
python tools/gripper_build_split.py --n-val 10 --val-seed 42
#    --out-json 默认 data/sim_lerobot_v30_ee/train_val_split.json
```

---

## 2. JSON 字段说明（`data/sim_lerobot_v30_ee/train_val_split.json`）

```jsonc
{
  "dataset": "sim_lerobot_v30_ee.csv",
  "n_train": 270,              // 3 任务总训练样本数
  "n_val": 30,                 // 3 任务总验证样本数
  "n_val_per_task": 10,        // 每任务验证样本数
  "seed": {"cluster": 0, "val_sampling": 42},
  "cluster": {"feature": "...200维...", "algorithm": "KMeans K=argmax(轮廓系数)", "kmax": 10},
  "tasks": {
    "0": {
      "task_index": 0,
      "instruction": "任务指令文本",
      "n_episodes": 100,
      "n_train": 90, "n_val": 10,
      "n_clusters": 2,          // 自动选出的类别数
      "cluster_stats": {        // 每类样本量统计
        "0": {"n_episodes": 55, "n_train": 49, "n_val": 6},
        "1": {"n_episodes": 45, "n_train": 41, "n_val": 4}
      },
      "train_episode_idx": [0, 1, ...],   // 训练集 episode 编号
      "val_episode_idx":   [14, 39, ...]  // 验证集 episode 编号
    }
    // "1", "2" 同理
  }
}
```

验证集选取逻辑: 每任务各类取 `n=min(10, 类样本数)`, 用**最大余数法**把 10 个名额按
类占比分配到各类, 各类内**随机**抽满（固定 seed）, 保证每类至少覆盖 1 个。

---

## 3. 代码复用方式

共享库 `tools/gripper_common.py`（数据/数学原语, 其它脚本 import 它）:

```python
import sys; sys.path.insert(0, ".")
from tools.gripper_common import (
    interp_100,                 # 任意长序列 -> 100 维 (np.interp)
    episode_feature_L100_R100,  # 左右爪夹 -> 200 维特征 [L100, R100]
    load_grippers,              # CSV -> DataFrame(每episode一行: grip_L/grip_R 序列)
    alloc_proportional,         # 最大余数法分额
    setup_cjk_font,             # matplotlib 中文字体
)
```

`tools/gripper_cluster.py` 提供**聚类与画图**函数: `auto_kmeans(X,kmax,seed)`、
`plot_silhouette(...)`、`plot_2d(...)`、`plot_cluster_curves(...)`、`cluster_and_plot(...)`。
`tools/gripper_select_val.py` 提供 `select_val(df_ep_cluster, n_val, seed)`（分层抽样）。
`tools/gripper_build_split.py` 提供 `build_split(csv, meta, ...)`（返回完整 JSON dict）。

---

## 4. 任务聚类结论速查（本轮 sim_lerobot_v30_ee）

| 任务 | K(自动) | 轮廓系数 | 各类 n | 结构解读 |
|---|---|---|---|---|
| task0 笔筒 | 2 | 0.24 | 55/45 | 左右手分工互换 |
| task1 充电器插排 | 4 | 0.89 | 26/27/28/19 | 闭合动作时序相位 |
| task2 叠碗 | 6 | 0.33 | 22/23/26/10/14/5 | 左右闭合时机组合, 含仅单臂动作小类 |
