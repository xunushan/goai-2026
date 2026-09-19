# xvla_mumugo

GOAI 2026 双臂协同操作任务的 X-VLA 策略服务包（WebSocket 服务端），包含两个评测 checkpoint 的自动下载与一键启动。

## 目录结构

```
xvla_mumugo/
├── start_services.sh      # 一键「下载模型 + 启动服务」脚本
├── readme.md              # 本文档
├── install.sh             # 创建 conda 环境并安装依赖（首次使用运行一次）
├── deploy.yml             # 服务配置（已适配，勿修改）
├── model.py               # 策略服务核心（模型加载 + 推理）
├── deploy.py / eval.sh / setup_eval_*.sh   # RoboDojo 集成入口
├── gripper_hysteresis.py / temporal_ensemble.py  # 实验功能（未启用）
└── xvla/                  # X-VLA 模型实现
```

## 使用步骤

### 1. 解压

解压到已部署 RoboDojo 的服务器上，最终路径为：

```
RoboDojo/XPolicyLab/policy/xvla_mumugo
```

（服务依赖 RoboDojo 的 `scripts/robodojo.sh`，路径需保持上述层级。）

### 2. 安装 conda 环境

所需环境与 **X_VLA 策略完全一致**（Python 3.10）。

- **首次使用**：运行 `bash install.sh`，创建默认 `XVLA` 环境并安装全部依赖；
- **已装过**（包括已存在 X_VLA 用的环境）：跳过本步，启动时用 `--policy-env <环境名>` 直接指定即可。

### 3. 模型说明

两个评测模型，权重各约 3.3GB：

| model 名 | 对应 checkpoint |
| -------- | --------------- |
| `xvla-fw` | `T-formal-12000/ckpt-12000` |
| `xvla-sf` | `A2/ckpt-2000` |

> **⚠️ 请对两个模型都进行评测**：`xvla-fw` 与 `xvla-sf` 为两个独立的评测提交，分别评测 `xvla-fw` 和 `xvla-sf`。

启动脚本时传入模型名称即可。脚本会从公开仓库 `tianSeconds/finetunning` 自动下载对应权重到本目录 `checkpoints/` 下（无需登录）。

### 4. deploy.yml

评测相关配置已适配完成（`policy_seed: null` 等）。**评测期间请勿修改 `deploy.yml`**，该文件为标准评测配置。

### 5. 启动服务

最简启动（默认端口 `6000`、本地监听、后台运行）：

```bash
bash start_services.sh xvla-fw
bash start_services.sh xvla-sf
```

首次运行自动下载模型，之后直接启动。服务地址为 `ws://127.0.0.1:<端口>`。

脚本执行流程：

1. 打印将要执行的服务启动命令；
2. 自动下载模型到本目录 `checkpoints/`（已存在则复用）；
3. 后台启动服务，并轮询等待就绪（模型加载需数分钟）；
4. 输出 `服务已就绪：ws://<host>:<port>` 后，即可开始仿真测试；
5. 启动失败会打印错误信息并给出日志位置。

可调参数（均有默认值，按需指定）：

| 参数 | 说明 | 默认值 |
| ---- | ---- | ------ |
| `--port <port>` | 服务端口 | `6000` |
| `--host <ip>` | 监听地址 | `127.0.0.1` |
| `--gpu <gpu>` | GPU 编号 | `0` |
| `--policy-env <env>` | conda 环境名 | 当前已激活环境；未激活则 `XVLA` |

示例（对外监听 + 指定端口 / GPU / 环境）：

```bash
bash start_services.sh xvla-fw --host 0.0.0.0 --port 8080 --gpu 1 --policy-env myenv
```

> 使用 `--host 0.0.0.0` 对外暴露时，请放行对应端口。

服务日志自动输出到本目录 `logs/` 下（`xvla_mumugo_<model>_<时间戳>.log`）。

### 6. 仿真测试

服务输出 `服务已就绪` 后，在仿真服务器（Isaac Sim 客户端）上执行仿真命令连接本服务。执行前请先将本目录（xvla_mumugo）同步/复制到仿真服务器的 `RoboDojo/XPolicyLab/policy/` 下。

完整评测命令示例（本机测试，跑全部任务、每任务 25 个 episode）：

```bash
cd <仿真端 RoboDojo 根目录>
bash scripts/robodojo.sh benchmark \
  --policy-dir XPolicyLab/policy/xvla_mumugo \
  --ckpt xvla-fw \
  --policy-host 127.0.0.1 \
  --policy-port 6000 \
  --env-cfg arx_x5 \
  --action-type ee \
  --env-gpu 0 \
  --eval-num 25
```

仿真 client 的以下参数须与本服务对齐：

| 仿真 client 参数 | 对齐要求 |
| ---------------- | -------- |
| `--policy-dir XPolicyLab/policy/xvla_mumugo` | 与本服务目录一致 |
| `--ckpt xvla-fw` | 与启动服务传入的模型名一致（client 端 `--ckpt` 仅用于评测结果命名，不定位模型文件） |
| `--policy-host 127.0.0.1` | 与服务输出的一致（默认 `127.0.0.1`） |
| `--policy-port 6000` | 与服务输出的一致（默认 `6000`） |

### 7. 停止服务

启动时会打印进程 PID，直接停止：

```bash
kill <PID>
```

找不到 PID 时：

```bash
ps aux | grep setup_policy_server
kill <对应PID>
```
