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

# 每次模型预测一个完整 chunk，只执行前 10 步便重新获取观测并推理
actions_per_chunk: 10

# X-VLA 输出 0～1 的夹爪概率；大于阈值发送 1（开），否则发送 0（关）
gripper_threshold: 0.7

log_io: true
```

完整预测长度来自 checkpoint 的 `config.num_actions`，X-VLA 默认配置为
`30`，实际值会在服务启动时打印为 `model_chunk_size`。因此：

- `steps`：一次预测内部的去噪次数，增大通常会增加推理时间；
- `actions_per_chunk`：一次预测后真正交给仿真执行的动作数，范围为
  `1..model_chunk_size`；
- 当前建议先使用 `actions_per_chunk: 10`，若接触物体后纠偏仍不及时，再测试
  `5`。

`log_io: true` 时，每次推理会输出一行 `[x_vla][io]` JSON 日志，包括完整
chunk 长度、实际执行长度、左右夹爪概率的最小/最大值、执行段概率、阈值化
后的 0/1 指令以及左右臂 XYZ 范围。不会打印图像像素。
