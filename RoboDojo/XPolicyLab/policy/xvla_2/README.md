# xvla_2 — X-VLA policy-server（RoboDojo ws 协议）

把 X-VLA 模型（`action_mode=arx_ee6d`，20 维动作，`num_actions=30`）封装为
XPolicyLab 的 ws policy-server，供 RoboDojo（Isaac Sim）评测。模型代码来自 pip
安装的 `xvla` 包（`evaluation.robodojo.RoboDojoPolicyClient`），本文件夹只含
适配与部署脚本。仿真端**零 X-VLA import、零仿真代码改动**。

## 准备工作

### 1. 下载模型到本地

模型在 HF `tianSeconds/goai/xvla-ee6d/<ckpt>`（如 `018000`）。下载到
`/data/checkpoints/xvla-ee6d/<ckpt>`。CLI 不接受 repo_id 内嵌子路径，用 python
子目录下载：

```bash
python - <<'PY'
from huggingface_hub import hf_hub_download, list_repo_files
import os
repo, sub = "tianSeconds/goai", "xvla-ee6d/018000"
target = f"/data/checkpoints/{sub}"
os.makedirs(target, exist_ok=True)
for f in list_repo_files(repo):
    if f.startswith(sub + "/"):
        hf_hub_download(repo, filename=f[len(sub) + 1:], subfolder=sub, local_dir=target)
PY
```

目录须含 `config.json` + `model.safetensors` + `preprocessor_config.json`。

### 2. 安装策略服务器环境

policy-server 端 conda 环境默认名 **`XVLA`**（本机已配好 torch+CUDA 与 HF 凭据）：

```bash
bash install.sh /data/X-VLA     # 第二个参数 = 本地已克隆的 X-VLA 仓库路径
```

`install.sh` 会 `pip install -e <X-VLA 仓库>` + ws 依赖（websockets/msgpack/
pydantic/pyyaml/opencv），末尾 import 冒烟。**环境已有 torch 时自动跳过**，不会
降级；`XVLA_CONDA_ENV` 可换环境名，`XVLA_SKIP_CONDA_CREATE=1` 跳过建环境。

## 启动服务

工作目录 `<repo>/RoboDojo`（goai-2026 仓库）。

**完整评测**（策略服务器 + 仿真端同机，`eval.sh` 编排：起 ws 服务器 → 等端口 →
起仿真 client）：

```bash
bash scripts/robodojo.sh eval \
  --policy-dir XPolicyLab/policy/xvla_2 \
  --task <TASK> \
  --ckpt /data/checkpoints/xvla-ee6d/018000 \
  --policy-env XVLA \
  --policy-gpu 0 --env-gpu 0 \
  --eval-env <仿真conda环境>
```

**只起策略服务器**（供远端仿真 client 连接）：

```bash
bash scripts/robodojo.sh server \
  --policy-dir XPolicyLab/policy/xvla_2 \
  --task <TASK> \
  --ckpt /data/checkpoints/xvla-ee6d/018000 \
  --policy-env XVLA --policy-gpu 0 --policy-port 6000
```

**手动调试启动**（不走 robodojo.sh；`deploy.yml` 默认即 018000 模型）：

```bash
cd <repo>/RoboDojo/XPolicyLab
PYTHONPATH=<repo>/RoboDojo:<repo>/RoboDojo/XPolicyLab:<repo>/X-VLA \
  CUDA_VISIBLE_DEVICES=0 python setup_policy_server.py \
  --config_path policy/xvla_2/deploy.yml --overrides port=6000
```

## 验证

无仿真时用任意 ws client 连 `ws://<host>:<port>`，`reset` 后 `infer` 一个含
3 相机 uint8 RGB + 16d state 的 observation，期望返回 **30×16** 动作（全有限、
quat 范数≈1、gripper∈[0,1]）。服务器日志每预测一行 `[xvla_2][io]`（state16/20 +
完整 30×16 action16），还原 episode：

```bash
python -m evaluation.robodojo.parse_log policy_server.log --out preds.csv
```

## 关键注意点

- **gripper 双向反转**（`invert_gripper=True`）：输入 16→20、输出 20→16 都反转，
  与训练一致；官方 X_VLA 不反转（其训练数据未反转），勿照抄。
- **图像管线与训练一致**：`Resize(224,224,BICUBIC) → ToTensor → ImageNet 归一化`；
  不用 HF processor 处理图像（只用于 `encode_language`）。
- `domain_id=6`（`DATA_DOMAIN_ID["arx_x5_ee"]`），flow-matching `steps=10`。
- **全 chunk 模式**：每次 infer 返回 30 动作，仿真执行完再预测；`deploy.py`
  每步 `get_obs()` → 完整逐帧视频。`sim_step_log`（deploy.yml，默认 false）可在
  仿真端每步打 `[xvla_2][sim]`（可选）。
