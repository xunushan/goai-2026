# LeRobot 0.6 ACT 部署与 RoboDojo 仿真验证

本目录把 LeRobot 0.6 训练的 ACT 模型接入 RoboDojo 的 WebSocket policy server。

## 1. 环境配置

执行本目录的安装脚本，创建 `lerobot` Conda 环境：

```bash
bash XPolicyLab/policy/act_lerobot/install.sh
```

## 2. 准备 checkpoint

模型下载到 `/data/checkpoints/act/` 下：

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate lerobot

hf download tianSeconds/goai \
  --include "act-30k/*" \
  --local-dir /data/checkpoints/act
```

## 3. 启动 policy server

```bash
cd /data/RoboDojo

bash scripts/robodojo.sh server \
  --policy-dir XPolicyLab/policy/act_lerobot \
  --task stack_blocks \
  --ckpt /data/checkpoints/act/<模型目录> \
  --policy-env lerobot \
  --env-cfg arx_x5 \
  --action-type ee \
  --seed 0 \
  --policy-gpu 0 \
  --policy-port <端口> \
  --bind-host <监听地址>
```

参数说明：

- `--policy-port`：服务监听端口，需在安全组/防火墙中开放
- `--bind-host`：`127.0.0.1` 仅本机访问；`0.0.0.0` 允许远程访问

日志输出到 `/data/outputs/`，查看日志：

```bash
tail -f /data/outputs/act_lerobot_server.log
```

## 4. Mock 验证

保持 policy server 运行，另开终端执行：

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate lerobot
cd /data/RoboDojo

PYTHONPATH=/data/RoboDojo:/data/RoboDojo/XPolicyLab \
python XPolicyLab/policy/act_lerobot/mock_client.py \
  --url ws://127.0.0.1:<端口> \
  --dataset /data/lerobot_v30_ee \
  --episode 0 \
  --stride 25 \
  --max-samples 1 \
  --action-steps 10 \
  --output-dir /data/outputs/act_lerobot_mock_episode0
```

输出文件：

```text
summary.json       汇总指标和输出合法性检查
requests.jsonl     每次请求、预测及对比结果
curves.csv         可用于绘制 state/action/prediction 曲线
images/*.jpg       实际发送给服务的图像
```

进入 Isaac Sim 前至少确认：

- 请求成功数与计划发送数一致；
- 每个请求返回 10 个 action（对应当前 `n_action_steps=10`）；
- action 全部是有限数，没有 `NaN` 或 `Inf`；
- 左右 quaternion norm 接近 1；
- gripper 全部位于 `[0,1]`；
- 图像键、shape、dtype、数值范围均正确；
- MAE 和逐维曲线没有明显的维度错位。
