# xvla_2 — X-VLA policy-server（RoboDojo ws 协议）

把 X-VLA 模型（`action_mode=arx_ee6d`，20 维动作，`num_actions=30`）封装为
XPolicyLab 的 ws policy-server，供 RoboDojo（Isaac Sim）仿真端评测。模型代码
来自 pip 安装的 `xvla` 包（`evaluation.robodojo.RoboDojoPolicyClient`），本文件夹
只提供适配与部署脚本。

**服务边界**：我们交付的是策略服务器（ws 服务）+ 本策略文件夹；仿真端（RoboDojo
eval）是独立基础设施，**零仿真代码改动**。仿真端只需本文件夹的
`deploy.py` + `deploy.yml` + `__init__.py` 骨架，**不 import X-VLA**。

---

## 一、模型下载（提前到本地）

模型已上传 HF：`tianSeconds/goai/xvla-ee6d/002000`。评测前**提前下载到本地**：

```bash
huggingface-cli download tianSeconds/goai/xvla-ee6d/002000 \
  --local-dir /data/checkpoints/xvla-ee6d/002000
```

`deploy.yml` 的 `model:` 默认指向该本地目录；`from_pretrained` 对本地目录和
HF repo id 都直接支持（目录须含 `config.json` + `model.safetensors` +
`preprocessor_config.json`）。`setup_eval_policy_server.sh` 会用 `robodojo.sh
server --ckpt` 覆盖该值：绝对路径 / 已存在相对路径 / `/data/checkpoints/<raw>` /
都不是时按 HF repo id 透传。

## 二、环境安装（策略服务器端）

```bash
bash install.sh                     # conda env=xvla，默认期望 ${POLICY_DIR}/X-VLA 本地仓库
bash install.sh /path/to/X-VLA     # 指定本地 X-VLA 仓库路径
```

- conda 环境默认名 `xvla`（`XVLA_CONDA_ENV` 可覆盖），python 3.10。
- torch/torchvision 与训练一致（`torch==2.1.2 torchvision==0.16.2`，
  CUDA 12.1 wheel；`XVLA_SKIP_TORCH=1` 跳过）。
- 私有仓库 X-VLA 需 git 凭据/token（本机 `~/Documents/token/github`）；脚本不写
  明文。也可自行 `pip install 'git+https://github.com/xunushan/X-VLA.git'`。
- ws 依赖：`websockets msgpack msgpack-numpy pydantic pyyaml opencv-python`。
- 末尾做 import 冒烟；模型真正加载在服务启动时。

## 三、服务启动

```bash
# 策略服务器端（GPU 0，端口自动/指定）
bash scripts/robodojo.sh server \
  --policy-dir XPolicyLab/policy/xvla_2 \
  --task stack_blocks \
  --ckpt /data/checkpoints/xvla-ee6d/002000 \
  --policy-env xvla \
  --policy-gpu 0 \
  --policy-port 6000

# 仿真端（另机/另一进程）
bash scripts/robodojo.sh eval \
  --policy-dir XPolicyLab/policy/xvla_2 \
  --task stack_blocks \
  --ckpt /data/checkpoints/xvla-ee6d/002000 \
  --policy-env <仿真env>
```

`robodojo.sh eval` 会要求本文件夹存在 `setup_eval_policy_server.sh` /
`setup_eval_env_client.sh`（已提供）；仿真端 eval 循环 `importlib
XPolicyLab.policy.xvla_2.deploy` → `eval_one_episode(TASK_ENV, model_client)`。

## 四、本地离线测试（先于仿真）

1. 单测（X-VLA 仓库内，conda lerobot / CPU / Fake 模型）：
   ```bash
   python -m pytest test/test_robodojo_client.py -v   # 13 项，含真实 ws 往返
   ```
2. 用数据集回放 mock（需服务已起 + XPolicyLab 在 PYTHONPATH）：
   ```bash
   cd XPolicyLab/policy/xvla_2
   PYTHONPATH=$PWD/../../.. python mock_client.py \
     --url ws://127.0.0.1:6000 \
     --dataset /data/data/lerobot_v30_ee_6d \
     --episode 0 --stride 25 --max-samples 5 --action-steps 30
   ```
   mock 读**我们 20d 数据集**的 `observation.state`，转 16d（gripper 反转）构造
   仿真观测，逐请求校验返回 **30×16** 动作（形状/有限/quat 范数≈1/gripper∈[0,1]），
   产物写 `outputs/xvla_2_mock/`（requests.jsonl / curves.csv / summary.json /
   images/）。

## 五、日志与 episode 还原

**策略服务器 `[xvla_2][io]`**（每 30 步一次预测；`request` 每 env 递增、reset 清零）：
- `{"event":"init","model":"<本地路径或HF id>",...}`
- `{"event":"client_observation","request":N,"env_idx":0,"instruction":"...",
   "state16":[...16],"state20":[...20],"images":{cam:统计}}`
- `{"event":"server_actions","request":N,"env_idx":0,"num_actions":30,
   "action16":[[...16]×30]}`（完整 chunk，可还原 episode）

**仿真端 `[xvla_2][sim]`**（可选，`deploy.yml` 的 `sim_step_log: true` 开启；
deploy.py 每步打印 `step_observation`，落入仿真 stdout，去向不可控）。注意
**完整逐帧视频不依赖该开关**：deploy.py 每步 `get_obs()` 即触发仿真自带的
`_stream_vision` 逐帧录制（与官方 X_VLA 同机制）。

**解析**（X-VLA 仓库内）：
```bash
# 服务器日志 → 每预测 (state16/20 + 30×16 动作)
python -m evaluation.robodojo.parse_log policy_server.log --out preds.csv
# 合并仿真每步日志 → 每步 (state16, action16)
python -m evaluation.robodojo.parse_log policy_server.log \
  --sim-log sim.log --merge --out steps.parquet
```

## 六、16d / 20d 布局与 gripper 反转

- 仿真端 16d：`[L_xyz3, L_quat_wxyz4, L_g1, R_xyz3, R_quat_wxyz4, R_g1]`（1=开）。
- 模型 20d（arx_ee6d）：`[L_xyz3, L_rot6d6, L_g1, R_xyz3, R_rot6d6, R_g1]`（1=闭）。
- **gripper 双向反转**（`invert_gripper=True`）：输入 16→20、输出 20→16 都反转。
  · 与我们训练一致（训练用 `ee16_to_xvla20(invert_gripper=True)`）。
  · 官方 X_VLA **不反转**，因为其训练数据没反转 —— 不要照抄官方。
- 输出再做四元数再归一化 + gripper clip [0,1]（连续回归，无阈值二值化）。

## 七、与官方 X_VLA 的差异

| 项 | 官方 X_VLA | xvla_2 |
|---|---|---|
| gripper | 不反转（其训练未反转） | 双向反转（我们训练已反转） |
| 图像管线 | HF processor | 训练一致：Resize(224,224,BICUBIC)+ImageNet 归一化 |
| 语言编码 | HF processor | HF processor（同） |
| 推理 | generate_actions steps=10 | 同 |
| 全 chunk | 30 动作/预测 | 同（actions_per_chunk 可调小=截断） |
| 日志 | `[x_vla][io]` | `[xvla_2][io]`（服务器）+ 可选 `[xvla_2][sim]` |
| ws ping | 关闭 | 同 |

## 八、文件清单

**策略服务器端**（需 xvla 包 + 模型）：`model.py`、`deploy.yml`、`install.sh`、
`setup_eval_policy_server.sh`、`mock_client.py`。
**仿真端**（零 X-VLA import）：`deploy.py`、`deploy.yml`、`__init__.py`、
`setup_eval_env_client.sh`。
**两端都有**：`README.md`。
