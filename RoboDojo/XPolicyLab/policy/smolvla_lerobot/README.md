# LeRobot 0.6 SmolVLA 部署与 RoboDojo 仿真评测

本目录将 LeRobot 0.6 的 SmolVLA checkpoint 接入 RoboDojo WebSocket
policy server。当前只支持 ARX-X5 双臂 `14D absolute joint`：

```text
左臂 6 关节 + 左夹爪 + 右臂 6 关节 + 右夹爪
```

夹爪约定为 `0=关闭，1=打开`。本实现直接使用已有的 `lerobot` Conda
环境，不会创建新环境，也不会加载 `policy/SmolVLA/smovla` 中的旧版
LeRobot checkout。

## 1. 检查环境

若环境缺少 SmolVLA extra，执行：

```bash
bash XPolicyLab/policy/smolvla_lerobot/install.sh
```

脚本只激活既有的 `lerobot` 环境，并安装
`lerobot[smolvla]>=0.6,<0.7`。

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate lerobot
python -c "import lerobot; print(lerobot.__version__)"
```

服务器应输出 `0.6.x`。

运行不加载模型权重的单元测试：

```bash
cd /data/RoboDojo
PYTHONPATH=/data/RoboDojo:/data/RoboDojo/XPolicyLab \
python -m unittest XPolicyLab.policy.smolvla_lerobot.test_model -v
```

## 2. 下载 checkpoint

```bash
mkdir -p /data/checkpoints/smolvla

hf download DaMiTian/smolvla-aloha-bimanual \
  --local-dir /data/checkpoints/smolvla/DaMiTian--smolvla-aloha-bimanual
```

checkpoint 目录中必须包含 `model.safetensors`、`config.json` 以及模型的
preprocessor/postprocessor 文件。

## 3. 启动 policy server

```bash
cd /data/RoboDojo

bash scripts/robodojo.sh server \
  --policy-dir XPolicyLab/policy/smolvla_lerobot \
  --task stack_blocks \
  --ckpt /data/checkpoints/smolvla/DaMiTian--smolvla-aloha-bimanual \
  --policy-env lerobot \
  --env-cfg arx_x5 \
  --action-type joint \
  --seed 0 \
  --policy-gpu 0 \
  --policy-port 80 \
  --bind-host 0.0.0.0
```

注意：`action-type` 必须是 `joint`，不能使用 `ee`。


## 4. 配置

默认每次规划后执行前 10 个动作，可在 `deploy.yml` 修改：

```yaml
actions_per_chunk: 10
```

效果调优时建议测试 `1、5、10、20`。较小的值重规划更频繁，对扰动更稳，
但推理开销更高。

服务启动时会强校验：

- observation state 为 14D；
- action 为 14D；
- 三路相机全部存在；
- 所有输出有限，不含 NaN/Inf；
- checkpoint 是完整的 LeRobot pretrained model 目录。

如果服务报告 LeRobot 版本不是 0.6，检查 `PYTHONPATH` 中是否残留旧版
`policy/SmolVLA/smovla/src`。
