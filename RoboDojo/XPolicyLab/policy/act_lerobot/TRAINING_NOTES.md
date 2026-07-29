# ACT 训练踩坑记录与运行指南

> 环境: LeRobot 0.6.0 / Tesla T4 (15GB) / Python 3.12 / CUDA 12.8

## 一、踩坑记录

### 1. 模块路径变更
```
旧: python -m lerobot.scripts.train
新: python -m lerobot.scripts.lerobot_train
```
0.6.0 把所有 CLI 入口重命名为 `lerobot_*` 前缀。

### 2. 缺少 accelerate 包
```
ImportError: 'accelerate' is required but not installed
```
**解决**: 安装 `lerobot[training]` extra（含 accelerate, wandb 等）
```bash
pip install 'lerobot[training]'
```
已在 `setup_lerobot.sh` 中改为安装 `lerobot[training]`。

### 3. HF Hub 下载 ACT 配置 401
```
--policy.path=lerobot/act
RepositoryNotFoundError: 401 Client Error ... lerobot/act/resolve/main/config.json
```
**原因**: `--policy.path` 会从 HF Hub 下载预训练 config，未登录或仓库变更会 401。
**解决**: 用注册名直接实例化，不走网络下载
```
--policy.type=act    # 正确
# --policy.path=lerobot/act   # 错误
```

### 4. 参数名变更
```
旧: --offline_steps=100 --lr=1e-5 --device=cuda
新: --steps=100 --policy.optimizer_lr=1e-5 (移除 device，accelerate 自动检测)
```
0.6.0 参数变化:
- `offline_steps` → `steps`
- `lr` → `--policy.optimizer_lr`（ACT 配置自带该字段）
- `device` 移除，accelerate 自动检测 CUDA

### 5. dataset.root 路径规则
```
错误: --dataset.root=/workspace/data
正确: --dataset.root=/workspace/data/lerobot_v30_ee
```
0.6.0 中 `root` 直接作为数据集根目录（需含 `meta/info.json`），不会拼接 `repo_id`。

### 6. 输出目录不能预先创建
```
FileExistsError: Output directory ... already exists and resume is False
```
**解决**: 脚本中不要 `mkdir -p` 预创建目录，让 LeRobot 自己创建。它要求目录不存在（除非 `resume=true`）。

### 7. push_to_hub 默认开启
```
ValueError: 'repo_id' argument missing. Please specify it to push the model to the hub.
```
**原因**: ACT 默认 `push_to_hub=True`。
**解决**: 显式关闭
```
--policy.push_to_hub=false
```

### 8. torchcodec 加载失败 → 切换 pyav
```
RuntimeError: Could not load libtorchcodec
libtiff.so.6: undefined symbol: jpeg12_write_raw_data
libavutil.so.59: cannot open shared object file
```
**原因**: torchcodec 依赖 FFmpeg 共享库，conda 安装的 FFmpeg 与 torchcodec 不兼容。
**解决**: 切换视频解码后端为 PyAV（纯 Python 绑定，自带 FFmpeg）
```
--dataset.video_backend=pyav
```
PyAV 比 torchcodec 稍慢（不支持精确 seek），但稳定可用。

### 9. FP16 启用失败
```
--policy.dtype=fp16
DecodingError: The fields `dtype` are not valid for ACTConfig
```
**原因**: ACTConfig 没有 `dtype` 字段，draccus 严格校验。`lerobot_train.py` 通过 `policy.dtype` 驱动 accelerate autocast，但 ACT 未继承该字段。
**现状**: 移除该参数，默认 FP32 训练。T4 15GB 显存下 batch=16 显存充足（约 12GB）。

### 10. EPOCHS 变量名误导
脚本原变量名 `EPOCHS` 实际是步数（`--steps`），改为 `STEPS` 避免混淆。

## 二、运行命令

### 1. 首次创建固定数据划分

单任务 stack_blocks 使用 90% train / 10% val：

```bash
export TASKS_JSON='["Stack the three blocks with different textures."]'
bash /workspace/act/create_split.sh
```

不设置 `TASKS_JSON` 时默认使用上述 stack-blocks 任务；
`TASKS_JSON='[]'` 表示保留全部任务。任务列表参与默认 split 文件名哈希，
避免单任务 split 与全任务 split 相互覆盖。正式训练需要使用同一个
`TASKS_JSON`；划分文件不存在时训练会直接退出。

### 2. 正式训练
```bash
export TASKS_JSON='["Stack the three blocks with different textures."]'
nohup bash /workspace/act/train_act.sh 2>&1 | tee /workspace/outputs/train_$(date +%Y%m%d_%H%M%S).log &
```

### 3. 查看实时日志
```bash
tail -f /workspace/outputs/train_*.log
```

### 4. 手动转换 checkpoint（可选，训练脚本已自动执行）
```bash
python /workspace/convert_to_ckpt.py /workspace/outputs/act_xxx/checkpoints/last
```
输出到 `/workspace/outputs/act_xxx/checkpoints/model.ckpt`。

### 5. 续训（从 checkpoint 恢复）
```bash
nohup python -m lerobot.scripts.lerobot_train \
    --resume=true \
    --config_path=/workspace/outputs/act_xxx/checkpoints/last/pretrained_model/train_config.json \
    2>&1 | tee /workspace/outputs/resume_$(date +%Y%m%d_%H%M%S).log &
```

## 三、关键参数说明

| 参数 | 值 | 说明 |
|---|---|---|
| BATCH_SIZE | 16 | T4 显存够用（batch=8 时 5.64GB，batch=16 约 12GB） |
| TARGET_EPOCHS | 3 | 根据 split 中实际训练帧数和 batch size 自动换算 STEPS |
| STEPS | 自动计算 | `ceil(train_frames × TARGET_EPOCHS / BATCH_SIZE)`；可用环境变量显式覆盖 |
| LR | 1e-5 | ACT 默认值 |
| CHUNK_SIZE | 50 | 预测未来 50 步动作（2秒@25fps） |
| N_ACTION_STEPS | 10 | 实际执行 10 步 |
| NUM_WORKERS | 6 | 8核CPU留2核给主进程 |
| SEED | 42 | 复现 |
| SAVE_FREQ | 自动计算 | 约每个 epoch 保存一次；可用环境变量显式覆盖 |
| LOG_FREQ | 100 | 每 100 步打印 loss |
| VAL_MAX_SAMPLES | 0 | 独立验证脚本的 frame 上限；0 表示评估完整 val 集 |
| video_backend | pyav | torchcodec 不可用 |

本次训练过程中不访问 val 集。训练完成后，单独运行 `evaluate_act.sh`，其内部调用 `evaluate_val_loss.py` 并输出：

```bash
bash /workspace/act/evaluate_act.sh \
    /workspace/outputs/act_xxx/checkpoints/last/pretrained_model
```

```text
eval_loss:...
val_l1_loss:...
physical_mae:
  first_step:...
  execution_window:...
  full_chunk:...
  per_dimension:...
  groups:...
```

- `eval_loss`：ACT 在 eval 模式下的验证损失；
- `val_l1_loss`：归一化 action 空间中，全部有效 chunk 步和 16 个动作维度的平均绝对误差；
- `physical_mae`：复用原 checkpoint 分析逻辑，在反归一化后的数据集原始单位中输出首步、前10步、完整chunk、逐维及左右臂/夹爪分组 MAE。

验证只遍历一次 val DataLoader、每个 batch 只执行一次模型前向，同时得到 loss 与上述 MAE，避免重复评测 val 集。

## 四、输出结构

```
/workspace/outputs/act_YYYYMMDD_HHMMSS/
├── checkpoints/
│   ├── 005000/pretrained_model/    # 各 checkpoint
│   ├── 010000/pretrained_model/
│   ├── ...
│   ├── 030000/pretrained_model/    # 最后一步
│   ├── last -> 030000              # 软链接指向最新
│   └── model.ckpt                  # 自动转换的 ckpt 文件
└── train_config.json               # 训练配置
```

## 五、性能参考

| Batch | updt_s | data_s | mem_gb | 步数/3小时 |
|---|---|---|---|---|
| 8 | 0.816 | 0.019 | 5.64 | ~13000 |
| 16 | 1.549 | 0.034 | ~12 | ~6500 |
