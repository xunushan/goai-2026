# codex_agent（本地 Codex 作为 stateful 高层策略）

`codex_agent` 不训练模型。它把**本机 macOS 上的 Codex CLI（模型 `gpt-6-astra`）**
当成高层具身策略：一个 episode = 一个 Codex thread，每个决策点把当前图像 +
本体状态作为新 turn 喂进去，Codex 输出**目标末端位姿**（不是低层轨迹），由
确定性 Motion Interpolator 展开成 ActionChunk 交仿真执行。

设计原则：**大模型负责看、判断、选目标、闭环纠偏；确定性 controller 负责
速度、插值、边界、安全与低层执行。**

## 架构

```
Mac(本地)                              服务器(issac-server)
┌────────────────────────┐             ┌──────────────────────────────┐
│ codex_bridge :8765     │             │ codex_agent policy server    │
│  └ codex exec / resume │◄── ssh -R ──┤  └ 拨号 localhost:8765       │
│    (gpt-6-astra)       │             │      sim client (ws://)      │
└────────────────────────┘             └──────────────────────────────┘
```

策略侧每轮链路：

```
obs (3 路相机 + 双臂位姿/夹爪)
  └─ prompt.py  组装 turn prompt（图像转 JPEG q88 内联）
       └─ bridge_client.py  POST /v1/decide  ──►  本地 codex_bridge
            └─ protocol.py 解析 JSON → TargetCommand（严格校验）
                 └─ motion.py LERP+SLERP → action chunk（每次从当前观测位姿重锚定）
                      └─ deploy.py 逐步 take_action
```

## 目录结构

```
codex_agent/
├── model.py                # ModelTemplate 适配器：状态机、预算记账、调 bridge / 调插值器
├── motion.py               # LERP+SLERP 插值器 + rpy↔quat + guardrail（纯函数，可单测）
├── protocol.py             # TargetCommand 解析与校验（rpy / quat 双形式）
├── prompt.py               # 系统提示词 + 每轮模板 + 输出 JSON Schema
├── bridge_client.py        # bridge 的 HTTP 客户端（仅 stdlib + ThreadPoolExecutor 做超时）
├── tasks/<task>.json       # 任务知识卡（只有任务语义 + HOME + 工作空间，无阈值无分布）
├── bridge/                 # ★ 跑在本地 macOS 上的 Codex HTTP 桥（见 bridge/README.md）
│   ├── server.py           #   POST /v1/decide，GET /healthz
│   └── codex_runner.py     #   命令构造、JSONL 解析、超时杀进程、thread 管理
├── tests/                  # 零 Codex 额度：fake_codex 桩 + mock_bridge
├── tools/
│   ├── tunnel_mac.sh       # 本地开 ssh -R 隧道
│   └── render_prompt.py    # 离线渲染提示词（不调用 Codex）
├── deploy.py / eval.sh / setup_eval_*.sh   # RoboDojo 集成入口（与 patch_policy 一致）
├── install.sh              # 只校验依赖，不安装任何包
└── serve_remote.sh         # 便捷启动脚本
```

## 部署（issac-server）

1. 同步本目录到服务器 `/data/RoboDojo/XPolicyLab/policy/codex_agent`。
2. 校验依赖：`bash install.sh`（复用 XVLA 环境，不安装）。
3. **本地**起 bridge 与隧道（顺序重要）：
   ```bash
   # Mac 终端 1
   python3 RoboDojo/XPolicyLab/policy/codex_agent/bridge/server.py --quiet
   # Mac 终端 2
   bash RoboDojo/XPolicyLab/policy/codex_agent/tools/tunnel_mac.sh issac-server
   ```
4. **服务器**上确认链路：`curl -s http://localhost:8765/healthz`。
5. 跑评测（`--ckpt` 是 `robodojo.sh` 的必填项，本策略无 checkpoint，填 `none`）：
   ```bash
   cd /data/RoboDojo
   bash scripts/robodojo.sh eval \
     --policy-dir XPolicyLab/policy/codex_agent \
     --task plug_in_charger \
     --ckpt none \
     --policy-env XVLA \
     --eval-num 1
   ```

## 关键约束

- **每次 episode 最多 15 次 Codex 调用**（`episode.max_codex_calls`）。调用很贵
  （~25-30 s、token 可观），预算耗尽后不再联网，直接返回 hold chunk。
- **`eval_batch: false`**：一个 episode = 一个 Codex thread，不支持多环境并行。
- **两条硬不变量**：返回的 chunk 长度恒 `>= 1`（空 chunk 会让评测永久挂住）；
  `get_action` 永不抛异常（抛异常 → `WsError` → episode 被静默丢弃，`eval_time=0`）。
- **超时**：bridge 侧 75 s（首轮 90 s）杀 Codex 子进程；策略侧 110 s 硬闸，
  均小于评测客户端写死的 120 s。任何超时都降级为 hold，绝不抛异常。
- **信息边界**：提示词里**不出现**任何从 benchmark 源码挖出的内部信息
  （reward 阈值、物体随机化分布、分数分档）。判分细节只以任务语义表达
  （"插到底、保持竖直，最后双臂回初始位姿并张开夹爪"），不给数值。
  `tests/test_prompt.py` 对此做回归。

## 关键配置（deploy.yml）

| 段 | 字段 | 说明 |
| --- | --- | --- |
| `bridge` | `base_url` / `request_timeout_s` / `decision_wall_budget_s` | bridge 地址与各级超时；`CODEX_BRIDGE_URL` 优先 |
| `episode` | `max_codex_calls` / `max_sim_steps` / `min_sim_steps_per_call` | 预算记账（当前 15 / 400 / 5） |
| `motion` | `delta_p_max_m` / `delta_theta_max_deg` / `settle_steps` | 每仿真步限速（1.5 cm / 5°）与收尾 hold 步数 |
| `guardrail` | `workspace` / `reject_margin_m` / `max_validation_retries` | 越界 clamp 或拒绝（拒绝会回结构化 feedback 让 Codex 重规划） |
| `home` / `workspace` | — | 取自 `data/sim_lerobot_v30_ee`（首帧 std=0 / q01-q99），非基准源码 |

## 测试（零 Codex 额度）

```bash
bash tests/run_all.sh
```

`tests/fake_codex` 桩住 CLI、`tests/mock_bridge.py` 桩住 bridge，两层加起来
覆盖真实 `bridge/server.py` 与真实 `model.py`，全程不调用 Codex——额度留给正式
评测。离线看提示词用 `python tools/render_prompt.py --turn 3`。
