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
