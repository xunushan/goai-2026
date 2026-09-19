# X-VLA 独立服务

## 1. 安装环境

```bash
cd RoboDojo/XPolicyLab/policy/X_VLA_OPT
bash install.sh
conda activate XVLA
```

默认创建 Conda 环境 `XVLA`。启动服务前需要激活该环境。

## 2. 下载模型

下载 RoboDojo seed 0 微调权重：

```bash
hf download RoboDojo-Benchmark/RoboDojo \
  --repo-type dataset \
  --include "ckpt/RoboDojo/X_VLA/RoboDojo-sim-arx_x5-ee-0/ckpt-100000/*" \
  --local-dir /data/checkpoints/xvla/
```

下载 processor/tokenizer（仅作为训练/微调时复制到 checkpoint 的基座来源；部署时 processor 从模型 checkpoint 目录加载，无需单独指定）：

```bash
hf download 2toINF/X-VLA-Pt \
  --local-dir /data/checkpoints/x_vla/X-VLA-Pt
```

## 3. 启动服务

服务启动方式统一参见 [docs/策略服务启动指南.md](../../../../docs/策略服务启动指南.md)。

仿真机按照 [`docs/仿真测试指南.md`](../../../../docs/仿真测试指南.md) 连接服务机 IP 和端口 `80`，评测动作类型使用 `ee`。

## 4. 动作与日志配置

修改 `deploy.yml`：

```yaml
# 流匹配去噪迭代次数，不是执行动作数
steps: 10

# 官方 checkpoint 每次预测并执行完整的 30 步，再重新获取观测并推理
actions_per_chunk: 30

# RoboDojo 使用 0～1 连续夹爪位置，直接输出 X-VLA sigmoid 后的值
gripper_mode: continuous
gripper_threshold: 0.7

# 仿真传入 task name，服务端映射成完整自然语言指令
task_prompt_map:
  stack_blocks: "Stack the three blocks with different textures."

log_io: true
```

完整预测长度来自 checkpoint 的 `config.num_actions`。官方 RoboDojo
`ckpt-100000/config.json` 中明确为 `30`，实际加载值也会在服务启动时打印为
`model_chunk_size`。因此：

- `steps`：一次预测内部的去噪次数，增大通常会增加推理时间；
- `actions_per_chunk`：一次预测后真正交给仿真执行的动作数，范围为
  `1..model_chunk_size`；
- 官方评测实现会遍历并执行模型返回的整个 chunk，所以对齐官方时使用
  `actions_per_chunk: 30`。`10` 或 `5` 只作为后续提高重规划频率的实验配置。
- `gripper_mode: continuous`：保留 `[0,1]` 连续夹爪值；只有对照实验改成
  `threshold` 时，`gripper_threshold` 才生效；
- `task_prompt_map`：将仿真 task name 映射为模型输入的完整自然语言指令，
  未命中的 instruction 保持原样。

### Temporal ensemble（在线动作集成，参照 act_lerobot）

默认 `actions_per_chunk: 30` 是「一次推理执行整段 30 步」的官方对齐行为。启用
temporal ensemble 后按 **actions_per_chunk 滑动窗口**对齐多条预测做指数加权平均，
可提升动作平滑度（ACT 式，LeRobot 默认系数 0.01，正值更信任更早的预测）。推理
成本为官方整段的 1/K：每隔 K 步才重规划一次，一次返回 K 个 ensembled action：

```yaml
actions_per_chunk: 5     # 每隔 5 步重规划一次，一次返回 5 个 ensembled action
temporal_ensemble_coeff: 0.01   # horizon 默认取 ckpt 的 num_actions=30，无需配置
```

- `temporal_ensemble_coeff: null`（默认）关闭，恢复整段 chunk 执行；
- `temporal_ensemble_horizon` 可选，缺省即 ckpt 的 `num_actions=30`（整段长度），
  通常无需配置，仅在需要缩小平滑窗口时设置（范围 `[2, 30]`）；
- 启用时要求 `actions_per_chunk < temporal_ensemble_horizon`（相邻重规划的预测
  窗口在时间上有重叠才有平滑意义）；`actions_per_chunk == horizon`（含 30 整段
  直出）时 ensemble 自动退化为直通，不产生额外计算；
- 每次推理仍产出完整 30 步 chunk，按时间对齐做指数加权平均后返回前 K 个动作。

夹爪连续值直出（默认，推荐先测试）：

```yaml
gripper_mode: continuous
```

使用阈值将夹爪指令二值化：

```yaml
gripper_mode: threshold
gripper_threshold: 0.7
```

阈值模式下，预测值大于 `0.7` 时发送 `1`（开），否则发送 `0`（关）。
连续模式下 `gripper_threshold` 不生效。

`log_io: true` 时，每次推理会输出两类 `[x_vla][io]` JSON 日志：

- `client_observation`：环境编号、原始任务指令、实际模型 prompt、原始状态、
  20D proprio，以及模型输入图像的 shape/dtype/min/max/mean；
- `server_actions`：完整 chunk 长度、实际执行长度、左右夹爪概率的最小/最大
  值、执行段概率、实际发送的连续夹爪指令以及左右臂 XYZ 范围。

日志不会打印图像像素。
日志会包含任务文本和机器人 EE/夹爪状态；如不需要诊断，请设置
`log_io: false`。

### 仿真图片落盘（`save_images`）

`log_io` 的日志只打印图像的 shape/dtype/min/max/mean，看不到画面本身。需要看
「模型到底看到了什么」时打开 `save_images`（默认关）：

```yaml
save_images:
  enabled: true
  root: /data/outputs/sim_images   # 默认值即此处
  jpeg_quality: 90
```

落盘位置与命名：

```
/data/outputs/sim_images/<服务启动时间戳>_<task_name>/env<env_idx>_<episode_idx>/<相机名>/<帧号>.jpg
```

- 存的是**模型输入**图像（与推理同源，`camera_names` 顺序、已解码的 HWC uint8
  RGB），不是协议里的压缩字节；
- 每 episode 一个文件夹、每路相机一个子文件夹（相机名取 `camera_names`；未配置
  时按位置为 `view_0`/`view_1`…），帧号从 `000000` 起、按推理次数递增；
- 最外层带服务启动时间戳，同一台机器上同 task 反复评测不会互相覆盖。

写在每次推理之前（`model.py` 的 `_prep_env`），因此 `eval_batch` 批评测下存到的是
**每次重规划时的画面**，不含 chunk 中间步（批路径本就把中间步 obs 留在客户端）。
单张 480×640 JPEG q=90 约 30~50KB，长评测注意 `/data` 磁盘占用；不需要画面时保持
`enabled: false`，此时完全不碰磁盘。
