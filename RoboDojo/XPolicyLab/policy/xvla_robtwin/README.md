# X-VLA-RoboTwin2 在 RoboDojo 中的仿真评测

本目录用于在 RoboDojo/Isaac Sim 中评测官方发布的
[`2toINF/X-VLA-RoboTwin2`](https://huggingface.co/2toINF/X-VLA-RoboTwin2)
checkpoint。当前目标是先完成零微调仿真评测，不包含训练。

适配器复用 `policy/X_VLA` 中的 X-VLA 模型实现，并固定使用
RoboTwin2 checkpoint 的运行协议：

- `domain_id=6`；
- 单张 head camera；另外两个图像槽由 processor 补零并 mask；
- absolute 双臂 EE action；
- 模型内部 action/proprio 为 20 维 EE6D；
- RoboDojo 环境输入输出为 16 维 `xyz + quaternion(wxyz) + gripper`；
- RoboDojo 夹爪命令为 `1=open`、`0=closed`；
- 返回完整 X-VLA action chunk；
- 不添加 ACT temporal ensemble、action limiter 或只执行第一步等逻辑。

## 1. 目录与模型接口

RoboDojo 输入：

```text
state16 =
[L_xyz(3), L_quat_wxyz(4), L_gripper_opening(1),
 R_xyz(3), R_quat_wxyz(4), R_gripper_opening(1)]
```

X-VLA 输入和输出：

```text
action20 =
[L_xyz(3), L_rotation6d(6), L_gripper_logit/probability(1),
 R_xyz(3), R_rotation6d(6), R_gripper_logit/probability(1)]
```

适配器完成：

```text
RoboDojo state16
    → quaternion(wxyz) 转 rotation6d
    → X-VLA proprio20
    → X-VLA absolute action20
    → rotation6d 转 quaternion(wxyz)
    → RoboDojo absolute action16
```

### 夹爪语义

X-VLA 的 EE6D action space 使用 `BCEWithLogitsLoss` 训练两个夹爪通道，
推理时分别经过 sigmoid，不是 softmax。

X-VLA-RoboTwin2 的训练数据将 gripper target 定义为 `1=closed`。官方
RoboTwin2 客户端使用：

```text
p_close > 0.7  → closed
p_close <= 0.7 → open
```

官方客户端随后把这个二值结果转换成 RoboTwin 仿真执行接口要求的
`+1=open / -1=closed`。这个 `[-1,1]` 转换只属于 RoboTwin 环境的执行
命令格式，并不是 X-VLA 网络的训练 target，也不是 RoboDojo 数据预处理。

本适配器不经过 RoboTwin 的 `±1` 接口，而是直接翻译成 RoboDojo 原生命令：

```text
p_close > 0.7  → 0（closed）
p_close <= 0.7 → 1（open）
```

这里不使用 `1-p`，因为 checkpoint 输出是二分类概率，不是连续夹爪开度。

注意：RoboDojo 保存的原始夹爪数据范围是 `[0,1]`，语义为 `1=open`、
`0=closed`。`policy/X_VLA` 当前的 `RoboDojoHandler` 读取
`left_ee_joint_states` 和 `right_ee_joint_states` 后直接拼接进 action target，
既没有反转，也没有转换到 `[-1,1]`。因此它与 RoboTwin2 checkpoint 使用的
`1=closed` 语义相反。以后使用 LeRobot v3 数据微调时，Dataset 必须把夹爪
target 转成：

```text
close_target = 1 - robodojo_opening
```

不能直接用原始 opening 值训练。

## 2. 安装环境

进入本目录：

```bash
cd /data/RoboDojo/XPolicyLab/policy/xvla_robtwin
```

本目录有独立的安装脚本：

```bash
bash install.sh
```

默认创建或复用名为 `XVLA` 的 conda 环境。可通过环境变量改名：

```bash
XVLA_CONDA_ENV=xvla_robtwin bash install.sh
```

若环境已经创建，只安装或更新依赖：

```bash
XVLA_SKIP_CONDA_CREATE=1 \
XVLA_CONDA_ENV=XVLA \
bash install.sh
```

安装脚本使用共享源码：

```text
XPolicyLab/policy/X_VLA/xvla
```

但不会调用 `policy/X_VLA/install.sh`。

## 3. 下载模型

激活环境并安装 Hugging Face CLI：

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate XVLA
pip install -U huggingface_hub
```

从本目录下载到默认位置：

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1

huggingface-cli download \
  2toINF/X-VLA-RoboTwin2 \
  --local-dir checkpoints/shared/X-VLA-RoboTwin2
```

检查必要文件：

```bash
test -f checkpoints/shared/X-VLA-RoboTwin2/config.json
test -f checkpoints/shared/X-VLA-RoboTwin2/model.safetensors
test -f checkpoints/shared/X-VLA-RoboTwin2/preprocessor_config.json
```

默认配置位于 `deploy.yml`：

```yaml
model_path: checkpoints/shared/X-VLA-RoboTwin2
processor_path: checkpoints/shared/X-VLA-RoboTwin2
domain_id: 6
action_type: ee
```

如果模型位于其他目录，可以修改 YAML，也可以在启动服务时传绝对路径。

## 4. 启动策略服务

### 4.1 使用默认模型路径

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate XVLA

cd /data/RoboDojo/XPolicyLab

CUDA_VISIBLE_DEVICES=0 \
python setup_policy_server.py \
  --config_path policy/xvla_robtwin/deploy.yml \
  --overrides \
    host=0.0.0.0 \
    port=6000 \
    device=cuda
```

### 4.2 使用任意模型路径

```bash
cd /data/RoboDojo/XPolicyLab/policy/xvla_robtwin

bash deploy.sh \
  0 \
  XVLA \
  /absolute/path/to/X-VLA-RoboTwin2 \
  /absolute/path/to/X-VLA-RoboTwin2 \
  6000 \
  cuda
```

参数依次为：

```text
GPU ID
conda 环境
model_path
processor_path
port
device
```

服务启动后应保持运行，不需要随 task 重启。

## 5. 启动 Isaac Sim 前的 mock 测试

`mock_client.py` 会连接真实 WebSocket policy server，发送与 RoboDojo 相同
的嵌套 observation：

- 三张 `uint8 HWC RGB` 图像；
- 双臂 16 维 EE state；
- 四元数顺序 `wxyz`；
- 夹爪输入 `1=open`；
- 英文任务 instruction。

适配器只会把 `cam_head` 送给 X-VLA-RoboTwin2。

新开一个终端：

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate XVLA

cd /data/RoboDojo/XPolicyLab/policy/xvla_robtwin

PYTHONPATH=../.. python mock_client.py \
  --url ws://127.0.0.1:6000 \
  --instruction "Stack the three blocks with different textures." \
  --requests 2 \
  --output outputs/xvla_robtwin_mock.json
```

mock 会验证：

- 服务握手和 reset 成功；
- 返回非空 action chunk；
- action shape 为 `[T,16]`；
- 所有值均为有限值；
- 左右 quaternion 范数接近 1；
- gripper 只能是 0 或 1；
- 返回的是完整 chunk。

成功输出示例：

```text
request=0 shape=(T, 16) grippers=...
request=1 shape=(T, 16) grippers=...
PASSED: saved .../outputs/xvla_robtwin_mock.json
```

mock 未通过时不要启动 Isaac Sim。

## 6. 服务日志

默认记录前 10 次请求：

```yaml
log_io: true
log_max_requests: 10
log_full_actions: false
```

每个 request 输出两行：

```text
[xvla_robtwin][io] {"event":"client_observation", ...}
[xvla_robtwin][io] {"event":"server_actions", ...}
```

输入日志包含：

- instruction；
- RoboDojo `state16`；
- 转换后的 `xvla_proprio20`；
- head image shape、dtype、min、max 和 mean；
- `valid_views=1`；
- `domain_id=6`。

输出日志包含：

- `[T,16]` action shape；
- 16 个 action name；
- 完整 RoboDojo action chunk；
- 每维 min/max；
- quaternion norm；
- 原始左右 gripper close probability；
- 最终 `1=open/0=closed` 语义。

需要记录完整原始 20 维输出时：

```yaml
log_full_actions: true
```

需要记录整个 episode 时：

```yaml
log_max_requests: 0
```

## 7. Isaac Sim 客户端公共配置

仿真客户端与策略服务使用相互独立的环境：

- Isaac/RoboDojo 环境只导入 `xvla_robtwin.deploy`，不加载模型；
- XVLA 环境只在 policy server 进程中加载 torch、transformers、timm 和权重；
- 不要在 Isaac/RoboDojo 环境中执行 `install.sh` 或补装 X-VLA 依赖。

`xvla_robtwin/__init__.py` 使用 lazy import，导入 `deploy` 不会触发
`model.py`。如果仿真端出现缺少 `timm`、`transformers` 或 `peft`，说明
使用的仍是修复前代码，应同步本目录，而不是修改仿真环境。

在 Isaac Sim 服务器执行：

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate RoboDojo

cd /data/RoboDojo

export ACCEPT_EULA=Y
export PRIVACY_CONSENT=Y
export OMNI_KIT_ACCEPT_EULA=yes

export POLICY_SERVER_HOST=<policy-server-ip>
export POLICY_SERVER_PORT=6000
```

如果 policy server 和 Isaac Sim 在同一台机器：

```bash
export POLICY_SERVER_HOST=127.0.0.1
```

先检查端口：

```bash
nc -vz "$POLICY_SERVER_HOST" "$POLICY_SERVER_PORT"
```

端口不可达时不要启动 Isaac Sim。

## 8. 第一阶段：stack_blocks 冒烟测试

先 dry-run 检查最终命令：

```bash
bash scripts/robodojo.sh client \
  --task stack_blocks \
  --policy-dir XPolicyLab/policy/xvla_robtwin \
  --policy-host "$POLICY_SERVER_HOST" \
  --policy-port "$POLICY_SERVER_PORT" \
  --env-cfg arx_x5 \
  --action-type ee \
  --ckpt X-VLA-RoboTwin2 \
  --seed 0 \
  --env-gpu 0 \
  --eval-num 1 \
  --dry-run
```

确认后运行一个 episode：

```bash
bash scripts/robodojo.sh client \
  --task stack_blocks \
  --policy-dir XPolicyLab/policy/xvla_robtwin \
  --policy-host "$POLICY_SERVER_HOST" \
  --policy-port "$POLICY_SERVER_PORT" \
  --env-cfg arx_x5 \
  --action-type ee \
  --ckpt X-VLA-RoboTwin2 \
  --seed 0 \
  --env-gpu 0 \
  --eval-num 1
```

首个 episode 重点检查接口，不以成功率为第一目标：

- client 成功连接 policy server；
- server 收到 head RGB、instruction 和 state16；
- `xvla_proprio20` 数值有限；
- 每次请求返回 `[T,16]` action；
- 左右臂没有交换；
- quaternion 没有跳变；
- `1=open/0=closed` 与实际夹爪一致；
- action 是 absolute EE target，不是 delta；
- episode 正常结束；
- 生成 `_result.json` 和视频。

如出现机械臂大幅跳变、左右臂互换、夹爪反向或 NaN/Inf，应停止批量评测，
优先查看服务端的两行 I/O 日志。

结果通常位于：

```text
eval_result/RoboDojo/stack_blocks/
```

## 9. 第二阶段：标准任务和 random 任务

冒烟测试通过后，再扩展到与 RoboTwin2/RoboDojo 重合度高的任务。示例：

```bash
TASKS=(
  stack_blocks
  stack_bowls
  fold_clothes
  hang_mugs
  make_toast
  pack_objects_into_box
  pour_liquid_into_cup
  sweep_blocks
)
```

每个任务先运行一个 episode：

```bash
mkdir -p outputs/xvla_robtwin_eval
echo "task,client_status" > outputs/xvla_robtwin_eval/status.csv

for TASK_NAME in "${TASKS[@]}"; do
  LOG_FILE="outputs/xvla_robtwin_eval/${TASK_NAME}.log"

  bash scripts/robodojo.sh client \
    --task "$TASK_NAME" \
    --policy-dir XPolicyLab/policy/xvla_robtwin \
    --policy-host "$POLICY_SERVER_HOST" \
    --policy-port "$POLICY_SERVER_PORT" \
    --env-cfg arx_x5 \
    --action-type ee \
    --ckpt X-VLA-RoboTwin2 \
    --seed 0 \
    --env-gpu 0 \
    --eval-num 1 \
    2>&1 | tee "$LOG_FILE"

  CLIENT_STATUS=${PIPESTATUS[0]}
  echo "${TASK_NAME},${CLIENT_STATUS}" \
    >> outputs/xvla_robtwin_eval/status.csv
done
```

标准任务能够完整运行后，再将任务名替换为对应的 `_random` 配置。

`client_status=0` 只表示客户端正常结束，不代表任务成功。成功率必须读取
`_result.json`，并结合视频判断。

## 10. 本地一体化评测入口

如果 policy server 和仿真环境在同一台机器，可使用仓库统一的 `smoke`
入口。先只生成命令，确认环境、checkpoint 和任务配置：

```bash
bash scripts/robodojo.sh smoke \
  --policy-dir XPolicyLab/policy/xvla_robtwin \
  --ckpt X-VLA-RoboTwin2 \
  --policy-env XVLA \
  --eval-env RoboDojo \
  --only stack_blocks \
  --env-cfg arx_x5 \
  --action-type ee \
  --policy-gpu 0 \
  --env-gpu 0 \
  --seed 0 \
  --dry-run
```

去掉 `--dry-run` 后执行单任务、单 episode 冒烟测试：

```bash
bash scripts/robodojo.sh smoke \
  --policy-dir XPolicyLab/policy/xvla_robtwin \
  --ckpt X-VLA-RoboTwin2 \
  --policy-env XVLA \
  --eval-env RoboDojo \
  --only stack_blocks \
  --env-cfg arx_x5 \
  --action-type ee \
  --policy-gpu 0 \
  --env-gpu 0 \
  --seed 0
```

冒烟通过后，使用统一的 `benchmark` 入口扩展任务：

```bash
bash scripts/robodojo.sh benchmark \
  --policy-dir XPolicyLab/policy/xvla_robtwin \
  --ckpt X-VLA-RoboTwin2 \
  --policy-env XVLA \
  --eval-env RoboDojo \
  --only stack_blocks,stack_bowls,fold_clothes,hang_mugs \
  --env-cfg arx_x5 \
  --action-type ee \
  --policy-gpu 0 \
  --env-gpu 0 \
  --seed 0 \
  --eval-num 1
```

`smoke/benchmark` 会按任务调用本目录的 `eval.sh`，适用于服务和 Isaac Sim
同机的一体化评测；上一节的 `robodojo.sh client` 适用于已经单独启动服务、
仿真端只连接远程 WebSocket 的方式。

也可以直接调用底层 `eval.sh`：

```bash
cd /data/RoboDojo/XPolicyLab/policy/xvla_robtwin

bash eval.sh \
  RoboDojo \
  stack_blocks \
  X-VLA-RoboTwin2 \
  arx_x5 \
  ee \
  0 \
  0 \
  0 \
  XVLA \
  RoboDojo
```

参数依次为：

```text
bench_name
task_name
ckpt_name
env_cfg_type
action_type
seed
policy_gpu_id
env_gpu_id
policy_conda_env
eval_env_conda_env
```

## 11. 使用 LeRobot v3 EE 数据做 LoRA 后训练

训练入口以 `X-VLA-RoboTwin2` 为基座，并固定使用与基座一致的
`domain_id=6`。它直接读取 LeRobot v3 parquet/video，不生成 HDF5。

### T4 机器安装与代码冒烟

安装脚本只读取本目录的 `requirements.txt`，不依赖
`policy/X_VLA/xvla/requirements.txt`。该文件包含 X-VLA 和 LeRobot v3
reader 的联合依赖。默认安装 PyTorch CUDA 12.8 wheel，适用于安装了
CUDA 12.8 或更新版 NVIDIA 驱动的 T4/4090 机器。CUDA 13.x 驱动可以
向后运行 cu128 wheel。

本目录同时内置训练所需的 `xvla/` Python 源码。`install.sh` 和
`train_lerobot.sh` 都只根据自身目录计算绝对路径，因此可以把
`xvla_robtwin` 单独放在 `/workspace/xvla_robtwin` 运行，不要求存在
`/workspace/XPolicyLab`。

```bash
cd /data/RoboDojo/XPolicyLab/policy/xvla_robtwin
bash install.sh
conda activate XVLA
```

安装结束时脚本会打印 PyTorch、Transformers、PEFT、Accelerate、PyAV、
Datasets 和 LeRobot 版本，并实际导入 LeRobot Dataset reader，同时打印
CUDA 是否可用和 GPU 名称。T4 应看到
`CUDA available: True` 和对应 Tesla T4 设备。

这里有意使用 `lerobot==0.4.4 --no-deps`：训练只调用 LeRobot v3 Dataset
reader。LeRobot 的完整应用依赖包含 `rerun-sdk`、机器人驱动和 UI 组件，
其中 `rerun-sdk` 要求 NumPy 2，与 X-VLA 的 NumPy 1.26 冲突；这些组件均
不参与本训练链路。

先执行不反向传播的结构检查，再做一个 optimizer step：

```bash
DRY_RUN=1 MIXED_PRECISION=fp16 \
MODEL_PATH=/data/RoboDojo/XPolicyLab/policy/xvla_robtwin/checkpoints/shared/X-VLA-RoboTwin2 \
DATASET_ROOT=/data/lerobot_v30_ee \
SPLIT_PATH=/data/splits/lerobot_v30_ee_train90_seed42.json \
TASKS_JSON='["Stack the three blocks with different textures."]' \
bash XPolicyLab/policy/xvla_robtwin/train_lerobot.sh

MIXED_PRECISION=fp16 BATCH_SIZE=1 GRAD_ACCUM_STEPS=1 STEPS=1 \
MODEL_PATH=/data/RoboDojo/XPolicyLab/policy/xvla_robtwin/checkpoints/shared/X-VLA-RoboTwin2 \
DATASET_ROOT=/data/lerobot_v30_ee \
SPLIT_PATH=/data/splits/lerobot_v30_ee_train90_seed42.json \
TASKS_JSON='["Stack the three blocks with different textures."]' \
bash XPolicyLab/policy/xvla_robtwin/train_lerobot.sh
```

T4 不支持原生 bf16，必须使用 `fp16`；脚本的 `auto` 模式也会自动做这个
选择。T4 主要用于代码链路验证，正式超参数和吞吐建议在 4090 上确定。

数据转换如下：

```text
observation.state/action: 双臂 16D
  xyz + quaternion(wxyz) + gripper_opening
                    ↓
X-VLA proprio/action: 双臂 20D
  xyz + rotation6d + gripper_close_target
```

其中：

```text
gripper_close_target = 1 - gripper_opening
```

如果使用 ACT 已生成的固定 episode split：

```bash
cd /data/RoboDojo

MODEL_PATH=/data/RoboDojo/XPolicyLab/policy/xvla_robtwin/checkpoints/shared/X-VLA-RoboTwin2 \
DATASET_ROOT=/data/lerobot_v30_ee \
SPLIT_PATH=/data/splits/lerobot_v30_ee_train90_seed42.json \
TASKS_JSON='["Stack the three blocks with different textures."]' \
BATCH_SIZE=1 \
GRAD_ACCUM_STEPS=16 \
STEPS=10000 \
GPU_IDS=0 \
bash XPolicyLab/policy/xvla_robtwin/train_lerobot.sh
```

先执行数据和模型 dry-run，不开始反向传播：

```bash
DRY_RUN=1 \
MODEL_PATH=/data/RoboDojo/XPolicyLab/policy/xvla_robtwin/checkpoints/shared/X-VLA-RoboTwin2 \
DATASET_ROOT=/data/lerobot_v30_ee \
SPLIT_PATH=/data/splits/lerobot_v30_ee_train90_seed42.json \
bash XPolicyLab/policy/xvla_robtwin/train_lerobot.sh
```

主要环境变量：

| 变量 | 默认值 | 含义 |
|---|---:|---|
| `MODEL_PATH` | 本目录共享 checkpoint | X-VLA-RoboTwin2 基座 |
| `DATASET_ROOT` | 项目 `data/lerobot_v30_ee` | LeRobot v3 数据根目录 |
| `SPLIT_PATH` | 无 | ACT 格式的固定 train/val episode JSON；默认强制要求 |
| `TASKS_JSON` | 无 | 精确任务文本列表；默认强制要求 |
| `DOMAIN_ID` | `6` | RoboTwin2 domain；当前训练入口禁止静默改成 0 |
| `BATCH_SIZE` | `1` | 4090 保守 micro batch；有余量再增大 |
| `GRAD_ACCUM_STEPS` | `16` | 梯度累积步数 |
| `STEPS` | `10000` | optimizer steps |
| `MIXED_PRECISION` | `auto` | T4 自动使用 fp16；Ampere/4090 自动使用 bf16 |
| `GPU_IDS` | `0` | 可见 GPU |
| `NUM_PROCESSES` | `1` | 训练进程数；双卡时设为 `2` |

为了防止 train/val 泄漏和误训全部任务，脚本不会静默接受空 split 或空任务
列表。确实需要使用全部 episode 或全部任务时，必须分别显式设置
`ALLOW_ALL_EPISODES=1`、`ALLOW_ALL_TASKS=1`。

双卡示例：

```bash
GPU_IDS=0,1 NUM_PROCESSES=2 \
bash XPolicyLab/policy/xvla_robtwin/train_lerobot.sh
```

训练产物保存在 `OUTPUT_DIR/ckpt-<step>/`，包含 PEFT adapter、
processor/tokenizer 文件、训练配置和固定数据划分。加载自训练 LoRA 的
`lora_path` 接入将在后续单独处理。

## 12. 推荐执行顺序

```text
安装 XVLA 环境并下载 checkpoint
        ↓
启动 policy server
        ↓
运行 mock_client.py
        ↓
检查两行一组的 server I/O 日志
        ↓
stack_blocks dry-run
        ↓
stack_blocks 单 episode
        ↓
检查 result JSON、三路视频和夹爪方向
        ↓
少量标准任务各 1 episode
        ↓
对应 random 任务各 1 episode
        ↓
最后再扩大 seed 和 eval_num
```

任何系统性输入、坐标系、夹爪或输出格式问题，都应在批量评测前解决。
