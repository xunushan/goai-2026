# X-VLA v1 预测服务

本目录将 LeRobot X-VLA checkpoint 接入 RoboDojo policy server。服务接收
ARX-X5 的三路同步图像、语言指令和 16D 绝对 EE 状态；模型内部使用 20D
rotation-6D 表示，每次返回并执行完整的 30 步 action chunk。

默认 checkpoint 地址为 `/data/checkpoints/xvla_v1`。该目录应当是包含
`config.json`、`model.safetensors` 和预/后处理配置的 LeRobot
`pretrained_model` 目录。也可通过 `--ckpt` 指定其他地址。

## 启动服务

在 RoboDojo 根目录执行：

```bash
bash scripts/robodojo.sh server \
  --policy-dir XPolicyLab/policy/xvla_v1 \
  --task stack_blocks \
  --ckpt /data/checkpoints/xvla_v1 \
  --policy-env lerobot \
  --env-cfg arx_x5 \
  --action-type ee \
  --seed 0 \
  --policy-gpu 0 \
  --policy-port 80 \
  --bind-host 0.0.0.0
```

服务启动时打印 checkpoint、设备、chunk 配置和相机数量。每次推理会以
`[xvla_v1][io]` 为前缀打印请求编号、指令、16D/20D 状态和输出摘要；不会
打印图像像素。端口 80 可能需要 root 权限或系统授予低端口绑定能力。

动作转换顺序固定为：模型 20D 输出 → LeRobot 后处理/反归一化 →
`utils/xvla_ee.py` 转换为 RoboDojo 16D EE 动作。
