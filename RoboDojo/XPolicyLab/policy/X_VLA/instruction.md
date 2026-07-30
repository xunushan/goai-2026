# X-VLA 独立服务

## 1. 安装环境

```bash
cd RoboDojo/XPolicyLab/policy/X_VLA
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
  --local-dir /data/checkpoints/x_vla/robodojo
```

下载 processor/tokenizer：

```bash
hf download 2toINF/X-VLA-Pt \
  --local-dir /data/checkpoints/x_vla/X-VLA-Pt
```

## 3. 启动服务

```bash
cd RoboDojo/XPolicyLab/policy/X_VLA
conda activate XVLA

bash serve_remote.sh \
  /data/checkpoints/x_vla/robodojo/ckpt/RoboDojo/X_VLA/RoboDojo-sim-arx_x5-ee-0/ckpt-100000 \
  /data/checkpoints/x_vla/X-VLA-Pt \
  stack_blocks 0 80 0.0.0.0 XVLA
```

参数依次为：微调模型目录、processor目录、任务名、GPU、端口、监听地址、Conda环境。

`serve_remote.sh` 会再次确认并激活最后一个参数指定的 Conda 环境。仿真机按照 [`docs/仿真测试指南.md`](../../../../docs/仿真测试指南.md) 连接服务机 IP 和端口 `80`。评测动作类型使用 `ee`。

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

`log_io: true` 时，每次推理会输出两类 `[x_vla][io]` JSON 日志：

- `client_observation`：环境编号、原始任务指令、实际模型 prompt、原始状态、
  20D proprio，以及模型输入图像的 shape/dtype/min/max/mean；
- `server_actions`：完整 chunk 长度、实际执行长度、左右夹爪概率的最小/最大
  值、执行段概率、实际发送的连续夹爪指令以及左右臂 XYZ 范围。

日志不会打印图像像素。
日志会包含任务文本和机器人 EE/夹爪状态；如不需要诊断，请设置
`log_io: false`。
