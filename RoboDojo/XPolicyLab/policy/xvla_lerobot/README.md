# X-VLA LeRobot：ARX-X5 训练与部署

本目录用于在 RoboDojo 的 LeRobot v3 EE 数据上微调和部署 X-VLA。完整决策背景见
[`../X_VLA/ARX_X5_FINETUNING_DESIGN.md`](../X_VLA/ARX_X5_FINETUNING_DESIGN.md)。

`xvla_lerobot` 的运行时代码是独立 policy 实现：它只使用 XPolicyLab 的
公共 `ModelTemplate`、checkpoint resolver 和本目录内的 X-VLA 模型代码，
不继承或导入已经废弃的 `policy/X_VLA/model.py`、`policy/X_VLA/deploy.py`。
上面的设计文档链接仅用于记录决策，不构成代码依赖。

## 已确定的训练定义

- 三路逐帧同步相机：`cam_high`、`cam_left_wrist`、`cam_right_wrist`；
- 原始双臂动作16维：每臂 `xyz + quaternion(wxyz) + continuous gripper`；
- 模型内部20维：每臂 `xyz + rotation6d + continuous gripper`；
- 数据真实频率25 Hz，未来窗口1秒，统一生成30个 action anchor；
- 使用 `domain_id=6`，复用 RoboTwin2/ARX-X5 平台的 soft prompt 和
  domain-specific action encoder/decoder；
- `action_mode=arx_ee6d`，连续夹爪使用 MSE，不使用 BCE/sigmoid；
- prompt warm-up 阶段训练 soft prompt 与 action encoder/decoder，随后联合训练；
- 部署端把一秒内30个 anchor 重采样为25个控制点。

## 安装

```bash
cd /path/to/RoboDojo/XPolicyLab/policy/xvla_lerobot
bash install.sh
```

默认 Conda 环境名为 `XVLA`，可通过 `XVLA_CONDA_ENV` 修改。

## 下载基座模型

第一阶段以 RoboTwin2 checkpoint 为基座并使用 domain 6：

```bash
cd /path/to/RoboDojo/XPolicyLab/policy/xvla_lerobot
mkdir -p checkpoints/shared

hf download \
  2toINF/X-VLA-RoboTwin2 \
  --local-dir checkpoints/shared/X-VLA-Pt
```



至少确认以下文件存在：

```bash
test -f checkpoints/shared/X-VLA-RoboTwin2/config.json
test -f checkpoints/shared/X-VLA-RoboTwin2/model.safetensors
test -f checkpoints/shared/X-VLA-RoboTwin2/preprocessor_config.json
```

## Episode 划分

推荐把固定划分保存在独立 JSON 中：

```json
{
  "train": [0, 1, 2, 5, 8],
  "val": [3, 4],
  "test": [6, 7]
}
```

训练脚本会检查 ID 范围、重复值以及不同 split 是否相交。筛选单位是
`episode_index`，不会按 Parquet chunk 切分，也不会跨 episode 构造未来动作。

## 训练

使用固定 split：

```bash
cd /path/to/RoboDojo/XPolicyLab/policy/xvla_lerobot

MODEL_PATH=$PWD/checkpoints/shared/X-VLA-RoboTwin2 \
DATASET_ROOT=/absolute/path/to/lerobot_v30_ee \
SPLIT_PATH=/absolute/path/to/episode_split.json \
TASKS_JSON='["Stack the three blocks with different textures."]' \
OUTPUT_DIR=$PWD/checkpoints/arx-ee-stack-blocks \
GPU_IDS=0 \
bash train_lerobot.sh
```

直接指定 episode 列表时，`EPISODES_JSON` 可以是 JSON 字符串或 JSON 文件路径：

```bash
DATASET_ROOT=/absolute/path/to/lerobot_v30_ee \
EPISODES_JSON='[0,1,2,5,8]' \
TASKS_JSON='[]' \
bash train_lerobot.sh
```

只有明确需要全部 episode 时才使用：

```bash
DATASET_ROOT=/absolute/path/to/lerobot_v30_ee \
ALLOW_ALL_EPISODES=1 \
bash train_lerobot.sh
```

常用环境变量：

| 变量                 |                               默认值 | 含义                                             |
| -------------------- | -----------------------------------: | ------------------------------------------------ |
| `MODEL_PATH`         | `checkpoints/shared/X-VLA-RoboTwin2` | 基座 checkpoint                                  |
| `DATASET_ROOT`       |                                 必填 | LeRobot v3 根目录                                |
| `SPLIT_PATH`         |                                   无 | 含 `train` 列表的固定 split JSON                 |
| `EPISODES_JSON`      |                                   无 | 内联列表或列表文件路径                           |
| `ALLOW_ALL_EPISODES` |                                  `0` | 显式允许使用全部 episode                         |
| `TASKS_JSON`         |                                 `[]` | 可选的精确任务文本过滤                           |
| `BATCH_SIZE`         |                                 `32` | 每进程 batch size                                |
| `ITERS`              |                              `30000` | optimizer steps                                  |
| `FREEZE_STEPS`       |                               `1000` | prompt/action-head warm-up 步数                  |
| `WARMUP_STEPS`       |                               `2000` | 联合训练学习率 warm-up                           |
| `GPU_IDS`            |                                  `0` | 可见 GPU                                         |
| `NUM_PROCESSES`      |                                  `1` | 当前固定为1；上游 iterable loader 未做进程级分片 |
| `MIXED_PRECISION`    |                               `auto` | Ampere及更新GPU用bf16，否则fp16                  |

脚本在输出目录保存实际使用的 `train_meta.json`。每个 checkpoint 同时保存模型
config、权重和 processor/tokenizer 文件，`config.json` 中的 action mode 应为
`arx_ee6d`。

## 启动策略服务

使用微调后的 checkpoint：

```bash
cd /path/to/RoboDojo/XPolicyLab/policy/xvla_lerobot

bash deploy.sh \
  0 \
  XVLA \
  "$PWD/checkpoints/arx-ee-stack-blocks/ckpt-30000" \
  6000 \
  cuda
```

参数依次为 GPU ID、Conda 环境、checkpoint 路径、端口和设备。服务启动时打印：

- checkpoint、监听地址和设备；
- domain、action mode、anchor/control 时间配置；
- 每次请求的三路图像摘要、输入状态、输出范围和连续夹爪语义。

图像像素和完整动作默认不写日志；调试时可在 `deploy.yml` 中设置：

```yaml
log_full_actions: true
```

## 服务冒烟测试

服务启动后运行：

```bash
cd /path/to/RoboDojo
PYTHONPATH=$PWD:$PWD/XPolicyLab \
python XPolicyLab/policy/xvla_lerobot/mock_client.py \
  --url ws://127.0.0.1:6000
```

客户端发送与仿真一致的三相机和16维绝对 EE 状态，验证返回动作：

- 形状为 `[actions_per_chunk, 16]`；
- quaternion 有效且归一化；
- 连续夹爪处于 `[0,1]`；
- 所有数值有限。

## RoboDojo 完整评测

`eval.sh` 会启动策略服务、等待端口就绪，然后启动仿真客户端：

```bash
cd /path/to/RoboDojo/XPolicyLab/policy/xvla_lerobot

bash eval.sh \
  RoboDojo \
  stack_blocks \
  /absolute/path/to/ckpt-30000 \
  arx_x5 \
  ee \
  0 \
  0 \
  1 \
  XVLA \
  RoboDojo
```

部署配置位于 `deploy.yml`。默认每次模型预测30个一秒 anchor，重采样为25 Hz后
返回前5个控制点，再获取新观测重新规划。可通过 `actions_per_chunk` 调整重规划频率。
