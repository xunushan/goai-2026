# codex_agent（本地 Codex 作为 stateful 高层策略）

`codex_agent` 不训练模型。它把**本机 macOS 上的 Codex CLI（模型 `gpt-6-astra`）**
当成高层具身策略：一个 episode = 一个 Codex thread，每个决策点把当前图像 +
本体状态作为新 turn 喂进去，Codex 输出**目标末端位姿**（不是低层轨迹），由
确定性 Motion Interpolator 展开成 ActionChunk 交仿真执行。

设计原则：**大模型负责看、判断、选目标、闭环纠偏；确定性 controller 负责
速度、插值、边界、安全与低层执行。**

策略内容与传输是分离的：**Codex 自己读 `workspace/`**（`AGENTS.md` 讲本体、
`.agents/skills/codex_agent/SKILL.md` 讲决策流程），`bridge/` 只负责传输、校验与落盘，
GPU 侧的 `model.py` 只发**结构化观测包**，不写一个字的提示词。

## 架构

```
Mac(本地)                                   服务器(issac-server)
┌──────────────────────────────┐            ┌──────────────────────────────┐
│ bridge/  :8765               │            │ codex_agent policy server    │
│  ├ workspace/  ← Codex 读这个 │◄── ssh -R ─┤  └ 拨号 localhost:8765       │
│  └ codex app-server --stdio  │            │      sim client (ws://)      │
│    (gpt-6-astra，常驻进程)     │            └──────────────────────────────┘
└──────────────────────────────┘
```

策略侧每轮链路：

```
obs (3 路相机 + 双臂位姿/夹爪)
  └─ observation.py 组装观测包（位姿、预算、feedback、图像；图像转 JPEG q88，不裁不缩）
       └─ bridge_client.py  POST /v1/decide  ──►  本地 bridge
            └─ bridge/bridge.py  用观测包组装本轮 turn 文本 ──► App Server
                 └─ bridge/schema.py 校验模型回复的形状
                      └─ protocol.py 解析 JSON → TargetCommand（严格校验）
                           └─ motion.py LERP+SLERP → action chunk（每次从当前观测位姿重锚定）
                                └─ deploy.py 逐步 take_action
```

## 目录结构

```
codex_agent/
├── model.py                # ModelTemplate 适配器：状态机、预算记账、调 bridge / 调插值器
├── observation.py          # 结构化观测包（EpisodeContext + build_request + load_task_card）
├── motion.py               # LERP+SLERP 插值器 + rpy↔quat + guardrail（纯函数，可单测）
├── protocol.py             # TargetCommand 解析与校验（rpy / quat 双形式）
├── bridge_client.py        # bridge 的 HTTP 客户端（仅 stdlib）+ 图像 JPEG 编码
├── tasks/<task>.json       # 任务知识卡（任务语义 + 两个预算，无阈值无分布）
├── bridge/                 # ★ 跑在本地 macOS 上的 Codex 策略服务（见 bridge/README.md）
│   ├── bridge.py           #   HTTP：POST /v1/decide、GET /healthz、单飞锁、错误映射
│   ├── app_server.py       #   常驻 codex app-server 的 JSON-RPC 客户端 + thread 管理与轮换
│   ├── bridge.py           #   HTTP 服务与 turn 文本组装
│   ├── schema.py           #   观测包严格校验 + outputSchema + 回复校验
│   └── record.py           #   原样落盘图片（不裁不重编码）+ rollout.jsonl
├── workspace/              # ★ Codex 的工作空间：策略内容在这里，不在代码里
│   ├── AGENTS.md           #   本体契约：双臂、坐标系、夹爪标度、单次决策上界、相机视角
│   ├── .agents/skills/codex_agent/SKILL.md   # 决策流程（可执行步骤 + 回复格式）
│   └── .codex/config.toml  #   权限模型（workspace 只读、output/<ep>/scratch 可写）
├── tests/                  # 零 Codex 额度：fake_app_server 桩 + mock_bridge
├── tools/
│   ├── tunnel_mac.sh       # 本地开 ssh -R 隧道
│   └── render_prompt.py    # 离线渲染本轮 turn 文本（不调用 Codex）
├── deploy.py / eval.sh / setup_eval_*.sh   # RoboDojo 集成入口（与 patch_policy 一致）
├── install.sh              # 只校验依赖，不安装任何包
└── serve_remote.sh         # 便捷启动脚本
```

## 部署（issac-server）

1. 同步本目录到服务器 `/data/RoboDojo/XPolicyLab/policy/codex_agent`。
   > `workspace/` 会随目录一起上传，但**它只在 Mac 上有意义**：读它的是 Mac 上的
   > Codex。服务器上的那份不参与运行，改它不会影响任何一次决策——要改策略行为，
   > 改 Mac 上那一份。

2. 校验依赖：`bash install.sh`（复用 XVLA 环境，不安装）。

3. **本地**起 bridge 与隧道（顺序重要）：
   ```bash
   # Mac 终端 1：从 codex_agent 目录起，--workspace 默认相对包目录解析
   cd RoboDojo/XPolicyLab/policy/codex_agent
   python3 -m bridge.bridge --quiet \
     --codex-bin /Applications/ChatGPT.app/Contents/Resources/codex
   # Mac 终端 2
   bash tools/tunnel_mac.sh issac-server
   ```
   > `--codex-bin` 必须给出绝对路径或指向 PATH 上的名字；**本机 `codex` 不在 PATH 上**，
   > 实际二进制在 `/Applications/ChatGPT.app/Contents/Resources/codex`。
   > `bridge/` 需要一个 `codex-cli 0.154.0-alpha.6.2` 及以上版本：它说的是
   > `codex app-server --stdio` 的 JSON-RPC，这个子命令在 `--help` 里标注为
   > `[experimental]`，跨版本可能变动。详见 [bridge/README.md](bridge/README.md)。

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

- **每次 episode 最多 100 次 Codex 调用**（任务卡的 `max_decisions`，`deploy.yml`
  只在卡片没写时兜底）。调用很贵（~25-30 s、token 可观），预算耗尽后不再联网，
  直接返回 hold chunk。100 次 × 25-30 s ≈ 45-50 min，需注意评测客户端的单集墙钟。
- **单次决策有目标上界**（`guardrail.max_target_distance_m` / `max_target_rotation_rad`，
  5 cm / 0.35 rad）：超限的目标被**拒绝**而非 clamp，下一轮 feedback 说明原因。
  这条上界决定了每次决策的仿真步数开销，因此和 `episode:` 的两个预算耦合。
- **`eval_batch: false`**：一个 episode = 一个 Codex thread，不支持多环境并行。
- **两条硬不变量**：返回的 chunk 长度恒 `>= 1`（空 chunk 会让评测永久挂住）；
  `get_action` 永不抛异常（抛异常 → `WsError` → episode 被静默丢弃，`eval_time=0`）。
- **超时**：bridge 侧单轮 75 s（新建 thread 的首轮 90 s，thread 轮换后的那一轮同样走 90 s），
  策略侧 105 s、硬闸 110 s，均小于评测客户端写死的 120 s。任何超时都降级为 hold，
  绝不抛异常。
- **信息边界**：提示词与 turn 文本里**不出现**任何从 benchmark 源码挖出的内部信息
  （reward 阈值、物体随机化分布、分数分档）。判分细节只以任务语义表达
  （"插到底、保持竖直，最后双臂回初始位姿并张开夹爪"），不给数值。
  数值只允许两类：**单次决策增量上界**（0.05 m / 0.35 rad）与**夹爪标度端点**（0 / 1）；
  其余出现的数只能是本轮的实测值或操作员设的预算。
  `tests/test_workspace.py` 与 `tests/test_turn_text.py` 对此做回归。
- **图像不裁不缩**：相机帧按仿真器给的分辨率原样发出，bridge 也原样落盘（逐字节相同），
  所以录下来的一集能证明模型当时看到的就是什么。这不是省 token 的选择——见
  `deploy.yml` 的 `images:` 注释。

## 关键配置（deploy.yml）

| 段 | 字段 | 说明 |
| --- | --- | --- |
| `bridge` | `base_url` / `connect_timeout_s` / `request_timeout_s` / `decision_wall_budget_s` | bridge 地址与各级超时；`CODEX_BRIDGE_URL` 优先 |
| `episode` | `max_codex_calls` / `max_sim_steps` / `min_sim_steps_per_call` | 预算记账，仅在任务卡未写明时兜底（550 / 100 / 5） |
| `motion` | `delta_p_max_m` / `delta_theta_max_rad` / `settle_steps` | 插值粒度（5 mm / 0.035 rad，对齐 GPT-Policy 的 `cartesian_step_m` / `cartesian_step_rad`）与收尾 hold 步数 |
| `guardrail` | `max_target_distance_m` / `max_target_rotation_rad` / `workspace` / `reject_margin_m` | 单次决策的目标上界（5 cm / 0.35 rad，超限**拒绝**）与工作空间盒（发货配置里盒子是开的） |
| `images` | `camera_names` / `jpeg_quality` | 相机与编码；**没有任何尺寸键**，写 `max_edge` 或 `max_width` 会直接报错 |
| `codex` | `bin` / `model` / `reasoning_effort` | **只用于日志行**，不会传给 Codex；真正生效的是 bridge 的命令行参数或 Mac 上的 `~/.codex/config.toml` |

> 起始位姿（`home`）**不在配置里**：它取自本集第一次观测，既是 `orientation`
> 的相对参考，也是集末强制归位的目标。写死一个常量就是第二个真相源，还会被
> 提示词引用而使模型信以为真。

## 测试（零 Codex 额度）

```bash
bash tests/run_all.sh
```

`tests/fake_app_server` 桩住 `codex app-server` 的 JSON-RPC、`tests/mock_bridge.py`
桩住 bridge 本身，两层加起来覆盖真实 `bridge/` 与真实 `model.py`，全程不调用 Codex
——额度留给正式评测。`mock_bridge.py` 不是宽容的替身：它把收到的观测包真的过一遍
`Observation.parse`、回复真的过一遍 `validate_response`，所以一个形状不对的用例会
在测试里就失败，而不是在真机上。离线看本轮文本用 `python tools/render_prompt.py --turn 3`。
