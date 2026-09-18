# xVLA 与 Codex Agent 稀疏融合设计

## 1. 设计目标

xVLA 每次根据当前 observation 输出一个长度为 30 的绝对 EEF action chunk。正常情况下由 xVLA 连续控制；只有 Host 粗略发现夹爪变化，或者上一段动作需要视觉复核时，才调用 Codex Agent。

设计原则：

- 在当前 observation 请求和 Codex 输出上做最小增量。
- 在线请求只包含 Codex 做本轮判断真正需要的内容。
- Host 内存只保存路由状态，不缓存完整 episode 数据。
- 完整 proposal、图片和执行结果落盘，避免扩大 `rollout.jsonl`。
- Host 只发现“可能有夹爪变化”，不判断抓取、释放或任务成功。
- Codex 根据图像和轨迹决定保留 VLA、截取前缀或输出修正 EEF。

## 2. 运行方式

### 2.1 Codex-only

没有 VLA 时保持当前行为和输出 schema，不做任何兼容性改动：

```json
{
  "left": {"position": "keep", "orientation": "keep", "gripper": "keep"},
  "right": {"position": [0.0, 0.0, 0.0], "orientation": [1.0, 0.0, 0.0, 0.0], "gripper": "keep"},
  "note": "修正右臂位置",
  "phase": "align"
}
```

示例数值仅说明字段形状，不是机器人或场景常量。

### 2.2 VLA-review

请求中存在 `vla_review` 时，bridge 使用 VLA-review 输出 schema。相比现有输出只增加三个字段：

```json
{
  "use_vla": true,
  "vla_steps": 30,
  "verify_next": true,
  "left": {"position": "keep", "orientation": "keep", "gripper": "keep"},
  "right": {"position": "keep", "orientation": "keep", "gripper": "keep"},
  "note": "保留轨迹，执行后复核",
  "phase": "grasp"
}
```

字段含义：

| 字段 | 含义 |
|---|---|
| `use_vla` | true 表示执行 VLA proposal；false 表示使用 Codex 的 EEF 目标 |
| `vla_steps` | 执行 VLA 前多少步；完整 chunk 为 30；不用 VLA 时为 0 |
| `verify_next` | 动作执行后，下一次 fresh observation 是否必须再调用 Codex |
| `left/right` | 沿用当前绝对 EEF 目标结构 |
| `note/phase` | 沿用当前简短审计字段 |

一致性规则：

```text
use_vla=true
  vla_steps 必须在 1..当前 horizon
  left/right 的 position、orientation、gripper 必须全部为 keep

use_vla=false
  vla_steps 必须为 0
  left/right 是本次 Codex 修正目标，语义与 Codex-only 完全相同
```

`use_vla=true, vla_steps=30` 表示保留整个 chunk；小于 30 表示只保留前缀，其余 suffix 丢弃。

## 3. 请求 Codex 的最小增量

当前 observation 请求保持不变，包括 episode/request/step/turn id、task、预算、当前双臂实测 EEF、三路 RGB 和上一次 Codex 执行反馈。

只有触发 VLA 审查时，增加一个 `vla_review`：

```json
{
  "vla_review": {
    "reason": "gripper_change",
    "chunk": [
      {
        "left": {"position": [0, 0, 0], "orientation": [1, 0, 0, 0], "gripper": 1.0},
        "right": {"position": [0, 0, 0], "orientation": [1, 0, 0, 0], "gripper": 1.0}
      }
    ]
  }
}
```

实际请求中的 `chunk` 必须是完整 H30 绝对 EEF 轨迹。示例只展示单行和字段形状。

`reason` 只保留两个值：

```text
gripper_change
  Host 在当前 proposal 中发现明显夹爪变化候选。

verify_previous
  上一次 Codex 决策的 verify_next=true，并且相关动作已经实际执行。
```

不增加 task progress、event type、proposal diagnostics、历史文件列表等在线字段。当前图像、当前 state、完整 VLA chunk、Codex thread 历史和 `reason` 已足够完成本轮判断。

## 4. 数据分别放在哪里

### 4.1 请求输入

每次真正调用 Codex 时发送：

```text
现有 observation 请求
+ vla_review.reason
+ vla_review.chunk（完整 H30）
```

VLA-only chunk 不请求 Codex，也不产生 Codex turn。

### 4.2 Codex 输出

Codex-only 使用当前 schema。VLA-review 使用现有 `left/right/note/phase` 加 `use_vla/vla_steps/verify_next`。Host 不解析 `note` 做控制路由。

### 4.3 Host 内存状态

每个 episode 只需保存：

```text
pending_verify: bool
current_vla_chunk: 当前尚未执行的 H30 proposal
```

可选保存 `vla_chunks/codex_calls/sim_steps` 计数器。不在内存维护完整历史图片、全部 VLA chunk 或另一套任务状态机。Codex 的历史决策由 App Server thread 保存；需要审计的数据落盘。

### 4.4 `workspace/output/<episode>/`

```text
output/<episode>/
├── rollout.jsonl
├── observations/
│   ├── cam_head/
│   ├── cam_left_wrist/
│   └── cam_right_wrist/
└── vla_chunks/
    ├── 000000.json
    ├── 000001.json
    └── ...
```

`observations/` 沿用当前实现，保存仿真端发送的图片。

每次 xVLA inference 保存一份 `vla_chunks/NNNNNN.json`，无论是否触发 Codex：

```json
{
  "step_id": 90,
  "chunk": [],
  "codex_triggered": true,
  "trigger_reason": "gripper_change",
  "codex_request_id": "ep-1-0004",
  "planned_vla_steps": 30,
  "executed_source": "vla",
  "executed_steps": 30,
  "verify_next": true
}
```

文件先写 proposal 和路由结果，执行完成后补充实际执行来源和步数。`chunk` 只在这里完整保存一次。

`rollout.jsonl` 继续一行记录一次 Codex 调用，只增加小字段：

```json
{
  "vla_chunk_path": "vla_chunks/000003.json",
  "vla_review_reason": "gripper_change",
  "decision": {},
  "executed_steps": 30
}
```

不要把 H30 再复制进 `rollout.jsonl`。没有调用 Codex 的 VLA-only chunk 只记录在 `vla_chunks/`。

## 5. Host 路由与执行流程

### 5.1 每轮先调用 xVLA

```text
最新 observation
→ xVLA
→ H30 绝对 EEF chunk
→ 保存 vla_chunks/NNNNNN.json
```

### 5.2 决定是否调用 Codex

```python
if pending_verify:
    call_codex(reason="verify_previous")
elif coarse_gripper_change(current_vla_chunk):
    call_codex(reason="gripper_change")
else:
    execute_vla(steps=30)
```

`pending_verify` 优先于当前曲线检测，因为上一段抓取或释放结果尚未验证。

### 5.3 粗粒度夹爪检测

Host 只判断整个 chunk 中某只夹爪是否明显变化。不要根据导数、局部极值、精确 action index、固定的完全闭合值或曲线推断物体大小和抓取结果。

第一版用配置化的低噪声阈值比较 `max(gripper)-min(gripper)` 即可。它只决定要不要让 Codex 看，不决定动作语义。阈值宁可产生少量额外调用，也不要过细地分类曲线。

### 5.4 处理 Codex 决策

#### 保留完整 VLA

```json
{"use_vla": true, "vla_steps": 30, "verify_next": false}
```

执行完整 H30。

#### 保留 VLA 前缀

```json
{"use_vla": true, "vla_steps": 18, "verify_next": false}
```

执行 `chunk[0:18]`，丢弃 suffix。典型情况：approach 合理，但 Codex 判断后续抓取位置不好。

#### Codex EEF 修正

```json
{
  "use_vla": false,
  "vla_steps": 0,
  "right": {"position": [0, 0, 0], "orientation": [1, 0, 0, 0], "gripper": "keep"}
}
```

不执行旧 VLA chunk。Host 使用当前 Codex-only 相同的 LERP/SLERP，根据当前位置到目标的平移和旋转距离决定 correction chunk 长度，不保留或补齐到原来的 30 步。

修正完成后获取 fresh observation，丢弃旧 VLA suffix，并重新调用 xVLA。

### 5.5 更新 `pending_verify`

只有动作实际执行成功后才更新：

```python
pending_verify = bool(decision.verify_next and executed_steps > 0)
```

下一轮如果 `pending_verify=true`，以 `reason=verify_previous` 调用 Codex。Codex 根据上一轮 thread 中的图像和决策，与当前 fresh observation 对比，判断是否抓空、放歪或释放失败。

当前验证完成但又执行了新的夹爪事件时，Codex可以继续返回 `verify_next=true`；否则返回 false，Host 清除 pending 状态。

## 6. Codex 数据流

```text
策略服务构造 observation + vla_review
→ bridge 校验请求
→ bridge 将当前 observation、H30 chunk 和 reason 加入 App Server turn
→ Codex 读取当前图片、state、VLA 轨迹和已有 thread 历史
→ Codex 返回结构化决策
→ bridge 按 VLA-review schema 校验并写 rollout.jsonl
→ 策略服务执行 VLA full/prefix 或 Codex EEF correction
→ 策略服务将实际执行结果写回 vla_chunks 文件
→ 根据实际执行结果更新 pending_verify
```

稀疏期间没有进入 Codex thread 的每个 VLA-only observation 不需要补进对话。Codex 被触发时以当前事实为准；如果确实需要更早证据，可以使用 workspace 中的 observation 缓存，但主路径不依赖模型主动查文件。

## 7. 写入 skill 的规则

VLA-review 模式增加以下规则，不改变 Codex-only 规则。

### 7.1 `reason=gripper_change`

- Host 只报告曲线变化候选，不能假设它一定是抓取或释放。
- 先看当前主相机和相关腕部相机，再看完整 H30 EEF/gripper 轨迹。
- 整段合理：`use_vla=true, vla_steps=30`。
- approach 合理但后续抓取或释放位置不好：保留错误动作前的 VLA 前缀。
- 当前已经适合做明确的局部调整：`use_vla=false`，输出绝对 EEF 目标。
- 不决定插值步数或速度；Host 负责插值。
- 如果获准执行的动作需要在下一帧确认抓取或释放结果，设置 `verify_next=true`。

### 7.2 `reason=verify_previous`

- 比较上一轮 Codex observation 与当前 fresh observation。
- 夹爪命令、gripper state 或 EEF 上抬本身不能证明抓取成功。
- 抓取后必须看物体是否随夹爪移动、是否滑落或仍留在原位。
- 释放后必须看物体是否脱离手指、得到支撑并保持合理姿态。
- 如果上一动作成功，正常审查当前 VLA chunk。
- 如果抓空或放歪，不得原样重复上一次失败动作；使用 VLA 前缀或 EEF 修正进入恢复。
- 恢复动作仍需下一轮视觉确认时继续设置 `verify_next=true`。

### 7.3 输出规则

- `use_vla=true` 时不输出伪造目标位姿，左右臂全部使用 `keep`。
- `use_vla=false` 时 `vla_steps=0`，目标位姿语义与当前 Codex-only 完全一致。
- 只保留 approach 时，`vla_steps` 必须停在模型认为不合适的抓取/释放动作之前。
- `verify_next` 表示下一轮必须做视觉结果检查，不代表当前已经成功。
- `note` 只写关键视觉依据和本次选择，Host 不解析它做路由。

## 8. xVLA 接入方式

### 8.1 第一版：Host 进程直接加载 xVLA

推荐第一版在策略服务 Host 中加载一次 xVLA 模型：

```text
仿真 observation
→ Host 内 xVLA.infer()
→ H30 chunk
→ Host 路由
→ 必要时调用远端 Codex bridge
```

优点：组件和协议最少，不需要再次编码图片和 state，没有额外网络延迟，proposal 与 observation 天然一一对应。条件是 xVLA 与策略服务可以运行在同一 Python/GPU 环境中，显存也足够。

### 8.2 需要时再拆成独立 VLA 服务

只有出现以下情况再独立部署：

- xVLA 依赖与策略服务冲突。
- 推理需要独立 GPU 或独立容器。
- 多个 Host 需要共享同一个模型实例。
- 模型加载和崩溃需要进程隔离。

无论本地还是远程，Host 只依赖一个很小的接口：

```python
chunk = vla.infer(observation)
```

返回值始终是 H30 绝对 EEF chunk。可以先实现 `LocalVLAAdapter`，未来换成 `RemoteVLAAdapter` 时不改变路由、Codex 请求和日志设计。

不建议让 Codex bridge 调用 VLA。VLA 属于服务器侧高频控制链路，Codex bridge 只负责低频大模型决策。

## 9. 第一版实施范围

只实现：

1. Host 直接加载 xVLA，并提供统一 adapter。
2. 可配置的 gripper range 粗检测。
3. 请求增加 `vla_review.reason/chunk`。
4. VLA-review 输出增加 `use_vla/vla_steps/verify_next`。
5. VLA full、VLA prefix 和现有 EEF correction。
6. 内存中的一个 `pending_verify`。
7. 独立 `vla_chunks/` 文件与精简 `rollout.jsonl` 引用。

不实现额外 task-progress 状态机、图像 embedding 触发器、复杂 proposal diagnostics 或单独 VLA 网络服务。先验证抓空率、放置稳定率、重复抓放次数、Codex 调用次数和 episode 墙钟。
