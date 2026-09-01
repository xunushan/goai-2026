# GOAI-2026 real 数据 → lerobot v3.0 转换与验证流程

本文档记录 GOAI-2026 双臂赛真实数据（real/piper_x，hdf5）转换为 lerobot v3.0 格式（openpi/Pi_05 训练用）的完整流程、脚本调用方法与验证手段。

## 1. 背景与目标

- **输入**：`/cloud/cloud-ssd1/GOAI-2026_real/data/real/<task>/piper_x/data/episode_*.hdf5`
  - 6 任务 × 100 episode = 600 episode，666,002 帧
- **输出**：lerobot v3.0 数据集两种 action 类型
  - `real_lerobot_v30_joint`：state/action 14 维（左右各 7：6 关节 + 1 gripper）
  - `real_lerobot_v30_ee`：state/action 16 维（左右各 8：x,y,z,qx,qy,qz,qw,g，四元数 xyzw）
  - 两数据集**视频逐帧相同**，ee 通过软链接复用 joint 的视频
- **上传目标**：HuggingFace `tianSeconds/goai_2026_lerobot`（一个仓库两个子文件夹）

## 2. 核心思路

1. **先生成 CSV 基准**：从 hdf5 提取全部数值列（joint 14 + ee 16 + task/episode/frame），作为后续一切校对的权威基准。
2. **joint 数据集**：process_data.py 直接从 hdf5 转换（含视频编码）。
3. **ee 数据集**：基于 joint 目录 + CSV 覆盖生成——不重新编码视频，只重写 state/action 数值。
4. **上传前必须校验**：以 CSV 为基准，对比 parquet 的 state/action 逐元素一致。

```
hdf5 ──extract_to_csv.py──> real_all.csv（权威基准）
  │                              │
  │ process_data.py              │
  │ (joint+视频编码)             │ overlay_ee.py
  ▼                              ▼
real_lerobot_v30_joint ──verify_from_csv.py──> joint 校验 PASS
       │
       └──overlay_ee.py──> real_lerobot_v30_ee ──verify_from_csv.py──> ee 校验 PASS
```

## 3. 脚本与调用方法

所有脚本位于 `RoboDojo/XPolicyLab/policy/Pi_05/openpi/scripts/`，服务器环境：
- Python：`/data/venvs/pi05_openpi/bin/python`
- 环境变量：`HF_LEROBOT_HOME=/cloud/cloud-ssd1/lerobot_data`、`XDATA_ROOT=/cloud/cloud-ssd1/GOAI-2026_real/data`

### 3.1 extract_to_csv.py —— 生成 CSV 基准

从 hdf5 提取全部数值列。CSV 每行 = 一帧，31 列：
`task, episode_index, frame_index` + joint 14 列 + ee 16 列。

```bash
python scripts/extract_to_csv.py real piper_x \
    /cloud/cloud-ssd1/lerobot_data/real_all.csv \
    fill_pen_holder,put_objects_into_basket,stack_and_cover_blocks,stack_bowls,stand_up_bottles,insert_charger
```

**关键实现**：
- **episode_index 全局递增**（跨任务连续 0..599），与 process_data 的 `save_episode` 语义严格一致——否则后续 (episode_index, frame_index) 匹配会键冲突（早期 bug 根因）。
- 6 维 eef 任务在提取时按 **xyz 内旋转**转为四元数（`eef_to_pose`，已验证为强候选）。
- `--max_episodes N`：小规模验证时限制每任务 episode 数。

### 3.2 process_data.py —— 转换 joint 数据集（含视频）

```bash
python scripts/process_data.py real v30_full piper_x joint \
    fill_pen_holder,put_objects_into_basket,stack_and_cover_blocks,stack_bowls,stand_up_bottles,insert_charger \
    --mode video --repo_id real_lerobot_v30_joint
```

**关键实现**：
- `SCHEMAS[(real, piper_x)]`：joint 用 `left/right_arm/joint(6)+gripper(1)`；ee 用 `left/right_arm/eef(7/6)+gripper(1)`。
- **streaming_encoding=True**（lerobot 0.4.4）：线程流式编码，~30s/episode，比非 streaming 快 ~5 倍。
- **分块解码**（block_size=256 帧）：图像按块生成器解码，避免单 episode 峰值 ~5GB 叠加触发服务器 cgroup OOM（9664MB）。
- action = state 前移（per episode 末帧 action=自身）。
- 视频按 lerobot 标准分块：每相机 chunk-000 内多个 ~200MB 的 mp4，meta/episodes 记录 `file_index + from/to_timestamp` 定位。

### 3.3 overlay_ee.py —— 基于 joint + CSV 覆盖生成 ee

```bash
python scripts/overlay_ee.py \
    /cloud/cloud-ssd1/lerobot_data/real_lerobot_v30_joint \
    /cloud/cloud-ssd1/lerobot_data/real_lerobot_v30_ee \
    /cloud/cloud-ssd1/lerobot_data/real_all.csv
```

**原理**（不重复编码视频）：
- videos 软链接复用 joint 的视频（两数据集视频逐帧相同）。
- 只重写 data parquet 的 state/action 列（14 维 → 16 维，数值来自 CSV，按 (epi, frame) 匹配）。
- 重写 meta/info.json（features 16 维命名 `l_x..l_g/r_x..r_g`）与 meta/stats.json（数值部分重算）。
- meta/episodes、meta/tasks.parquet 与 joint 完全一致，直接复制。

### 3.4 verify_from_csv.py —— 以 CSV 为基准校验

```bash
python scripts/verify_from_csv.py --dataset <joint_or_ee_dir> \
    --csv /cloud/cloud-ssd1/lerobot_data/real_all.csv --action_type joint|ee
```

对每个 chunk parquet，用 `(episode_index, frame_index)` 从 CSV 匹配期望 state/action（action 按「state 前移，末帧自身」构造），逐元素对比（atol 默认 1e-4）。输出帧数 / mismatch 数 / max abs diff / RESULT PASS|FAIL。

## 4. 验证结果（全量 600）

| 验证项 | 结果 |
|---|---|
| joint vs CSV | 666,002 帧，missing 0，mismatch 0，max abs diff 4.77e-07，**PASS** |
| ee vs CSV | 666,002 帧，missing 0，mismatch 0，max abs diff 0，**PASS** |
| LeRobotDataset 加载 | 两数据集 666,002 帧 / 600 episodes / 3 路视频解码正常（3×480×640）|
| episode 帧数对齐 | parquet 每 episode 帧数 == hdf5 原始帧数，**600/600 OK** |
| 视频文件 | 66 个 mp4，共 12.9G，单个 57~210MB（均值 ~200MB）|

**CSV 与 joint 数据核对说明**：joint 数据直接来自 hdf5，CSV 是 hdf5 数值的提取副本，两者同源。verify 以 CSV 为基准反向核对 parquet——joint 0 mismatch 证明 parquet 与 hdf5 数值完全一致（同源 + 转换无损）。ee 由 CSV 生成后 round-trip 校验 0 mismatch。

## 5. episode 帧数对齐验证

反向用 joint 数据评估每个 episode 帧数与 hdf5 是否对齐：

```python
import pandas as pd, h5py
from pathlib import Path

df = pd.read_parquet("/cloud/cloud-ssd1/lerobot_data/real_lerobot_v30_joint/data/chunk-000/file-000.parquet")
parq_counts = df["episode_index"].value_counts().sort_index()

# hdf5 原始帧数
for t in ["fill_pen_holder","put_objects_into_basket","stack_and_cover_blocks",
          "stack_bowls","stand_up_bottles","insert_charger"]:
    eps = sorted(Path(f"/cloud/cloud-ssd1/GOAI-2026_real/data/real/{t}/piper_x/data").glob("episode_*.hdf5"))
    for ep in eps:
        with h5py.File(ep, "r") as f:
            print(ep.name, "frames:", f["left_arm/joint"].shape[0])
```
结果：600/600 全部一致。

## 6. 上传 HF

```python
from huggingface_hub import HfApi
api = HfApi()

# 创建仓库（已存在则跳过）
api.create_repo(repo_id="tianSeconds/goai_2026_lerobot", repo_type="dataset", private=False)

# 上传 joint（13.2G）
api.upload_folder(repo_id="tianSeconds/goai_2026_lerobot", repo_type="dataset",
    folder_path="/cloud/cloud-ssd1/lerobot_data/real_lerobot_v30_joint",
    path_in_repo="real_lerobot_v30_joint", commit_message="upload joint")

# 上传 ee（13G）
api.upload_folder(repo_id="tianSeconds/goai_2026_lerobot", repo_type="dataset",
    folder_path="/cloud/cloud-ssd1/lerobot_data/real_lerobot_v30_ee",
    path_in_repo="real_lerobot_v30_ee", commit_message="upload ee")

# 上传 CSV 基准
api.upload_file(repo_id="tianSeconds/goai_2026_lerobot", repo_type="dataset",
    path_or_fileobj="/cloud/cloud-ssd1/lerobot_data/real_all.csv",
    path_in_repo="real_all.csv", commit_message="upload CSV")
```

**注意**：huggingface_hub 上传**不追踪软链接**。本地 ee/videos 是软链（省磁盘），但上传前必须转成真实拷贝（`rm videos && cp -r joint/videos ee/videos`），否则 HF 上会得到无效链接。

## 7. 小规模验证命令（每任务 2 episode）

```bash
TASKS=fill_pen_holder,put_objects_into_basket,stack_and_cover_blocks,stack_bowls,stand_up_bottles,insert_charger

# 1. CSV（限制每任务 2 episode）
python scripts/extract_to_csv.py real piper_x real_small.csv $TASKS --max_episodes 2
# 2. joint
python scripts/process_data.py real v30_smoke piper_x joint 2 $TASKS --mode video --repo_id real_lerobot_v30_joint_verify
# 3. 校验 joint
python scripts/verify_from_csv.py --dataset real_lerobot_v30_joint_verify --csv real_small.csv --action_type joint
# 4. overlay ee
python scripts/overlay_ee.py real_lerobot_v30_joint_verify real_lerobot_v30_ee_verify real_small.csv
# 5. 校验 ee
python scripts/verify_from_csv.py --dataset real_lerobot_v30_ee_verify --csv real_small.csv --action_type ee
```

## 8. 历史 bug 记录

- **episode_index 语义冲突**：早期 CSV 的 episode_index 按每任务从 0 计数，而 process_data 全局递增，导致 (epi,frame) 键冲突——fill_pen_holder（第一个任务）全部 2414 帧校验 FAIL，且 ee 校验假阳性（overlay 与 verify 用同一个冲突 lut，自洽但错位）。修复：extract_to_csv 改用全局计数器。
- **OOM**：单 episode 峰值 5GB 物理页在 episode 间不归还，触发 cgroup 限制。修复：load_data 图像分块解码（block_size=256）。
