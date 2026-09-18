# X-VLA 与 Codex Agent 稀疏融合设计

## 1. 目标与原则

X-VLA 每次根据当前 observation 输出长度为 30 的绝对 EEF action chunk。策略服务每轮把 observation 和 VLA proposal 交给独立 Bridge；Bridge 粗略发现夹爪变化，或上一段 VLA 动作需要结果验证时，才调用 Codex Agent，否则直接返回原 VLA chunk。

原则：

- 不修改现有 `X_VLA` 策略服务。
- 不把 X-VLA 依赖装进现有 `codex_agent` 策略目录。
- 新建独立融合策略目录，在 X-VLA 副本上做最小改动。
- Codex bridge 保持独立，只通过 HTTP 接收 observation 和 VLA proposal。
- 在线协议只提供模型判断和 Bridge 路由必需的信息。
- Bridge 的程序逻辑只发现夹爪曲线变化候选，不判断抓空、放歪或任务成功。
- 完整数据、进程内状态和请求/响应分别设计，不混在一个 JSON 中。

## 2. Codex 输出 schema

### 2.1 Codex-only 保持不变

没有 `vla_review` 的请求继续使用当前 schema：

```json
{
  "left": {"position": "keep", "orientation": "keep", "gripper": "keep"},
  "right": {"position": [0.0, 0.0, 0.0], "orientation": [1.0, 0.0, 0.0, 0.0], "gripper": "keep"},
  "note": "修正右臂位置",
  "phase": "align"
}
```

示例数值仅说明字段形状，不是机器人或场景常量。

### 2.2 VLA-review 使用二选一结构

不要在选择 VLA 时输出无意义的 `left/right: keep`，也不要使用大量 null。VLA-review 是两个互斥对象。

#### 选择 VLA full/prefix

```json
{
  "mode": "vla",
  "vla_steps": 30,
  "verify_next": true,
  "note": "轨迹合理，执行后复核",
  "phase": "grasp"
}
```

- `vla_steps=30`：执行完整 chunk。
- `1 <= vla_steps < 30`：只执行前缀，永久丢弃 suffix。
- `verify_next`：这段 VLA 前缀执行后，是否必须用 fresh observation 再调用 Codex。

#### 使用 Codex EEF 修正

```json
{
  "mode": "eef",
  "left": {"position": "keep", "orientation": "keep", "gripper": "keep"},
  "right": {"position": [0.0, 0.0, 0.0], "orientation": [1.0, 0.0, 0.0, 0.0], "gripper": "keep"},
  "note": "抓取点偏右，先对齐",
  "phase": "align"
}
```

EEF 修正沿用 Codex-only 的绝对目标语义。Bridge 使用统一的 LERP/SLERP 插值器生成 correction chunk；步数由平移和旋转距离决定，不保留或补齐为 VLA 的 30 步。执行后丢弃旧 VLA suffix，并从 fresh observation 重新调用 X-VLA。

VLA-review 的 EEF 修正只用于位置和姿态调整，夹爪应保持 `keep`。夹爪动作仍由经过 Codex 审查的 VLA prefix 执行，因此 EEF 修正后不自动产生 `pending_verify`；下一轮是否调用 Codex重新根据新 VLA chunk 判断。

### 2.3 `phase` 的唯一语义

`phase` 表示“本次返回、即将执行的 action 属于哪个阶段”，不是当前画面的阶段，也不是上一动作的结果标签。

例如：

```text
当前画面显示上一轮抓空，本轮输出重新对齐 EEF
→ phase = align 或 retry

当前画面正在接近，Codex批准包含闭合的 VLA prefix
→ phase = grasp

当前画面验证抓取成功，本轮批准搬运 VLA
→ phase = transport
```

这个定义必须写进通用 skill。当前 skill 只给了 phase 示例，没有明确区分“当前状态”和“返回动作”，需要补充。

### 2.4 `verify_next` 的判断

`verify_next` 只存在于 `mode=vla`：

```text
true
  Codex 判断获准执行的 VLA prefix 包含抓取或释放等必须看下一帧才能确认结果的夹爪事件。

false
  获准执行的 prefix 不包含这类事件，或只保留了事件前的 approach。
```

不能因为 Bridge 报告了粗粒度曲线变化就机械地设为 true；Codex 必须检查实际选择的 `vla_steps` 范围和完整轨迹。它也不能表示“当前动作已经成功”。这些规则必须写入 VLA-review reference。

### 2.5 Bridge 返回给策略服务

上面的对象是 Codex 原始决策。Bridge 验证它之后，直接生成最终 action chunk，再返回策略服务：

```json
{
  "request_id": "ep-1-0004",
  "source": "vla",
  "planned_steps": 30,
  "action_chunk": [],
  "decision": {
    "mode": "vla",
    "vla_steps": 30,
    "verify_next": true,
    "note": "轨迹合理，执行后复核",
    "phase": "grasp"
  }
}
```

- `mode=vla`：Bridge 返回输入 VLA chunk 的选定前缀。
- `mode=eef`：Bridge 使用统一插值器生成 correction chunk 后返回。
- 无 Codex 触发：Bridge 直接返回完整 VLA chunk，`decision` 为 null。

因此所有接入策略都执行 Bridge 返回的 `action_chunk`，不各自实现插值或 VLA prefix 截取。

## 3. 请求 Codex 的最小增量

现有 observation、task、budget、feedback 和三路图片保持不变。VLA-review 只增加：

```json
{
  "vla_review": {
    "reason": "gripper_change",
    "chunk": {
      "horizon": 30,
      "left": {
        "position": [[0.0, 0.0, 0.0]],
        "orientation": [[1.0, 0.0, 0.0, 0.0]],
        "gripper": [1.0]
      },
      "right": {
        "position": [[0.0, 0.0, 0.0]],
        "orientation": [[1.0, 0.0, 0.0, 0.0]],
        "gripper": [1.0]
      }
    },
    "summary": {
      "left": {
        "endpoint_translation_m": 0.0,
        "endpoint_rotation_rad": 0.0,
        "gripper_start": 1.0,
        "gripper_end": 1.0,
        "gripper_min": 1.0,
        "gripper_max": 1.0
      },
      "right": {
        "endpoint_translation_m": 0.0,
        "endpoint_rotation_rad": 0.0,
        "gripper_start": 1.0,
        "gripper_end": 0.5,
        "gripper_min": 0.5,
        "gripper_max": 1.0
      }
    }
  }
}
```

示例只展示数组形状；实际 `position/orientation/gripper` 三个数组长度都必须等于 `horizon=30`。

### 3.1 为什么采用 arm-major 数组

每只手内部维护三个同长度数组：

```text
left.position[step]
left.orientation[step]
left.gripper[step]
```

相比 30 个重复的 `{left:{...}, right:{...}}` 对象，它更紧凑，也更容易让 Codex 查看夹爪曲线和 EEF 变化。第 `i` 个 position、orientation 和 gripper 必须属于同一 action step。

### 3.2 最小统计摘要

摘要只针对当前 chunk，每只手保留六个值：

- 首尾 EEF 的平移距离。
- 首尾 EEF 的旋转距离。
- gripper 首值、末值、最小值、最大值。

它帮助 Codex快速理解当前 proposal，但不替代完整 H30，也不携带历史 state，不判断事件类型或成功失败。

### 3.3 `reason`

只保留：

```text
gripper_change
  当前 VLA chunk 的某只夹爪曲线明显变化。

verify_previous
  上一次 mode=vla 决策返回 verify_next=true；按照仿真接口契约，Bridge 返回的整个 action chunk 已执行，当前 observation 是其执行后状态。
```

### 3.4 不额外传执行轨迹

仿真端不会逐步回传 action 或中间 state，只会在整个 action chunk 执行完后返回新的 observation/state。策略服务给出多少步，仿真端就完整执行多少步。

因此第一版不增加 `recent_execution`、`previous_execution` 或 execution ACK：

- Bridge 已经生成最终 chunk，因此知道其确定长度。
- 下一次请求到达即表示上一响应已执行完毕。
- 当前 observation/state 就是上一 chunk 的执行后结果。
- Codex thread 中保留上一轮参与过的 Codex 决策；没有必要伪造不存在的逐步实测轨迹。

## 4. Bridge 路由与 action 生成

VLA prefix 截取和 EEF 插值属于可复用 Bridge，不放进各个策略服务。Bridge 延续现有 `codex_agent` 的状态模型，只维护 Codex App Server thread/history，不新增一套 EpisodeState 或 `routing.json`。

VLA-review 唯一需要跨请求传递的信息是“下一帧是否必须验证”。Bridge 在响应里生成最小 continuation，策略服务不解释内容，只在下一请求原样带回：

```json
{
  "continuation": {
    "previous_request_id": "ep-1-0004",
    "verify_previous": true
  }
}
```

`previous_request_id` 用于发现错序或重复请求；`verify_previous` 直接来自上一轮 Codex 的 `verify_next`。它不是执行 ACK，也不包含执行步数。

策略服务每轮只做三件事：生成 VLA proposal（如果有）、构造请求并原样回传 continuation、执行 Bridge 返回的完整 action chunk。

Bridge 路由：

```python
if no_vla_review:
    decision = call_codex_only()
elif continuation.verify_previous:
    decision = call_codex(reason="verify_previous")
elif coarse_gripper_change(vla_chunk):
    decision = call_codex(reason="gripper_change")
else:
    return vla_chunk
```

粗检测只使用配置化的低噪声阈值判断 `max(gripper)-min(gripper)` 是否明显非零。它不判断精确事件步、物体大小、抓取、释放或成功失败。

### 4.1 `mode=vla`

Bridge 取 `chunk[0:vla_steps]` 作为最终 action chunk 并丢弃 suffix。Codex 可以借此保留正确 approach、丢弃错误 grasp/release。

Bridge 把本次 `verify_next` 写入响应的 continuation。按照接口契约，下一请求到达时上一 chunk 已完整执行，因此：

```text
decision.verify_next = true
→ 下一请求以 reason=verify_previous 调用 Codex，使用 fresh observation 验证结果

decision.verify_next = false
→ 下一请求重新依据新的 VLA proposal 路由
```

### 4.2 `mode=eef`

Bridge 不执行旧 VLA proposal，使用与 Codex-only 相同的 LERP/SLERP 将绝对目标插值成 correction action chunk，并返回策略服务。旧 suffix 丢弃。

EEF correction 的 `pending_verify` 固定为 false。下一轮先由 X-VLA 基于 fresh observation 产生新 chunk，Bridge 再根据新 chunk 路由。

## 5. 数据落盘与两端边界

融合策略服务运行在服务器，Bridge 运行在本机，两端没有共享文件系统。因此不能把服务器文件路径交给 Codex 读取。Bridge 生成并返回 action chunk，其长度就是仿真端随后执行的步数。

### 5.1 Bridge 的 `workspace/output/<episode>/`

```text
output/<episode>/
├── rollout.jsonl
├── observations/
│   ├── cam_head/
│   ├── cam_left_wrist/
│   └── cam_right_wrist/
└── vla_chunks/
    └── <request_id>.json
```

Bridge 收到请求时已经拥有完整 VLA chunk，应在调用 Codex 前保存为 `vla_chunks/<request_id>.json`。路径由 Bridge 自己确定，不需要 Codex 填写，也不需要策略服务提前知道。

`rollout.jsonl` 的 VLA-review 行只增加：

```json
{
  "request_id": "ep-1-0004",
  "vla_review_reason": "gripper_change",
  "vla_chunk_path": "vla_chunks/ep-1-0004.json",
  "decision": {
    "mode": "vla",
    "vla_steps": 30,
    "verify_next": true,
    "note": "轨迹合理，执行后复核",
    "phase": "grasp"
  }
}
```

决策行可以记录 Bridge 返回的 `planned_steps`，它就是返回 action chunk 的长度。

不添加：

- `codex_request_id`：现有 `request_id` 已经是本次 Codex 请求 id。
- `planned_vla_steps`：已经由 `decision.vla_steps` 表达。
- `executed_steps`：接口约定整个返回 chunk 都会执行，不需要再复制一个同值字段。
- H30 数组：已经单独保存到 `vla_chunk_path`。

### 5.2 策略服务日志不属于 Bridge 协议

Bridge 不规定策略服务的日志文件名、字段或保存方式。不同策略服务可以按自身框架记录日志；这不是接入 Bridge 的前置条件，也不进入 Bridge 请求协议。

Bridge 只记录自己收到的请求、Codex 决策以及最终返回的 chunk 元数据。没有触发 Codex 的 VLA-only chunk 也经过 Bridge 快速路由，可以按同一方式记录。

## 6. Codex 数据流

```text
仿真 observation
→ 新融合策略服务调用本进程 X-VLA，得到 H30
→ 融合策略服务发送当前 observation、H30 和 summary
→ Bridge 读取请求中原样回传的 continuation，并做粗粒度夹爪变化检测
→ 未触发 Codex：Bridge 直接返回完整 VLA action chunk
→ 触发 Codex：Bridge 构造 App Server turn
→ Codex 读取当前图片、state、VLA proposal 和已有 thread 历史
→ Bridge 校验 mode=vla/eef，完成 prefix 截取或 EEF 插值
→ Bridge 返回最终 action chunk 和 planned_steps
→ 融合策略服务把最终 action chunk 交给仿真端完整执行
```

Codex 不需要全部历史 VLA proposal。它需要的是当前 proposal、当前视觉事实、当前实测 state，以及以前真正参与过的 Codex 决策历史。

## 7. Skill 路由设计

目录建议：

```text
.agents/skills/codex_agent/
├── SKILL.md
└── references/
    ├── codex-only.md
    └── vla-review.md
```

### 7.1 `SKILL.md`

只保留两类通用规则：

- 图像、实测 state、预算、工具和安全使用规则。
- `phase` 的统一语义：本次返回 action 的阶段。

然后按请求路由：

```text
请求没有 vla_review
→ 读取 references/codex-only.md

请求包含 vla_review
→ 读取 references/vla-review.md
```

一轮只读取一个 reference，避免两套输出 schema 混淆。

### 7.2 `codex-only.md`

保存当前单目标 EEF 决策、夹爪时机和 `{left,right,note,phase}` 输出规则。

### 7.3 `vla-review.md`

明确：

- Bridge 的曲线变化检测只是候选，不代表抓取或释放。
- `mode=vla` 可保留完整 chunk 或只保留错误动作前的 prefix。
- `mode=eef` 只做夹爪保持不变的绝对 EEF 局部修正。
- `verify_next=true` 仅当获准 VLA prefix 实际包含必须用下一帧验证的夹爪事件。
- `reason=verify_previous` 时比较上一轮 Codex 图像与当前 fresh 图像，判断抓空、滑落、放歪或释放失败。
- 夹爪命令、gripper state 和 EEF 上抬不能单独证明抓取成功。
- 失败后不能原样重复同一个姿态。
- `phase` 描述本次将执行的 VLA prefix 或 EEF correction。

## 8. 部署方案

### 8.1 Bridge 从策略目录独立出来

当前 `policy/codex_agent` 同时包含策略服务适配器和 Bridge/App Server 逻辑。目标结构应拆分为：

```text
XPolicyLab/
├── codex_policy_bridge/        独立、可复用的本机服务
│   ├── bridge/
│   ├── workspace/
│   ├── experience_library/
│   └── docs/
└── policy/
    ├── codex_agent/            Codex-only 薄策略客户端
    ├── X_VLA/                  原比赛模型，零修改
    └── X_VLA_Codex/            新融合策略服务
```

独立 Bridge 固化：

- Codex App Server session/thread。
- workspace、skill 和经验库。
- 请求校验、图片缓存和审计日志。
- Codex-only 目标插值。
- VLA 路由、prefix 截取和 EEF correction 插值。
- 响应生成、请求校验和无语义透传的 continuation。

各策略目录只保留薄 HTTP client 和各自模型推理。其它 VLA 只需遵守同一个 observation/VLA-review 协议即可接入。迁移可以后做，但新融合代码不应继续向 `policy/codex_agent` 内增加策略专属逻辑。

### 8.2 保持现有模型服务不变

- 现有 `policy/X_VLA`：不修改，继续作为比赛主模型服务。
- 独立 `codex_policy_bridge`：不加载 X-VLA。
- Codex App Server 和 workspace：仍运行在本机，通过反向 SSH 被服务器访问。

### 8.3 新建融合策略目录

从已验证的 `X_VLA` 固定版本复制一个新目录，例如：

```text
policy/X_VLA_Codex/
```

它运行在与 X-VLA 相同的 Python/GPU 环境，包含原有 X-VLA 依赖和权重加载逻辑。只在副本中增加：

```text
bridge_client.py       发送 observation/VLA proposal，接收最终 action chunk
```

副本的 `model.py` 保持原有 observation 预处理、X-VLA 推理和 action chunk 生成逻辑，只在得到 chunk 后调用 `bridge_client`，并执行 Bridge 返回的 action chunk。不要在副本中实现路由、Codex EEF 插值，也不要重写原 X-VLA 主推理链路。

### 8.4 安装与数据流

```text
服务器：X_VLA_Codex policy server
  加载 X-VLA
  接收仿真 observation
  生成 H30
  每轮 HTTP 请求 localhost:<reverse-tunnel-port>
  执行 Bridge 返回的最终 action chunk

本机：codex_policy_bridge
  接收 observation + VLA review 数据
  快速路由；只有需要时才调用 Codex
  生成最终 action chunk
  维护 Codex thread 和 workspace/output
  返回 action chunk + decision metadata
```

因为两端独立：

- 请求必须携带 Codex 判断所需的图像、state、H30、summary 和最近 N 步轨迹。
- 不能传服务器本地路径让本机 Codex 打开。
- bridge 记录模型看到和返回的内容。
- 策略服务是否以及如何记录日志，由各自实现决定。

这种方式保护现有 `X_VLA`，也避免把重模型依赖引入 `codex_agent`。

## 9. 第一版实施范围

1. 复制 `X_VLA` 为新的融合策略目录，原目录零修改。
2. 将 Bridge/App Server/workspace 从策略目录抽成独立 `codex_policy_bridge`。
3. 新策略 `model.py` 只在生成 H30 后调用 Bridge，并执行其返回的 action chunk。
4. 请求只增加 arm-major `chunk` 和六项/臂 summary。
5. VLA-review 输出实现 `mode=vla | eef` 二选一结构。
6. Bridge 通过无状态 continuation 传递 `verify_previous` 并生成最终 action chunk。
7. Bridge 保存请求、决策和返回 chunk 元数据，不规定策略服务日志。
8. skill 改成通用入口加两个 references 的路由结构。

第一版不实现图像 embedding 触发、程序侧视觉判断、复杂任务状态机或修改现有 X-VLA 服务。
