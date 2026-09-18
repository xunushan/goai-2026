# xVLA 与 Codex Agent 稀疏融合设计

## 1. 目标与边界

xVLA 是默认控制器，每次根据当前 observation 产生长度为 30 的绝对 EEF action chunk。Codex Agent 不参与每个 chunk，而是在夹爪候选事件、事件结果验证和恢复阶段进行视觉审查与修正。

职责边界：

```text
xVLA
  负责正常分布内的连续 H30 EEF 控制

策略服务（Host）
  保存 episode 状态
  对夹爪曲线做宽松候选检测
  决定是否触发 Codex
  执行 VLA full/prefix
  将 Codex EEF 目标插值为 correction chunk
  记录实际执行结果

Codex Agent
  根据当前图像、状态、VLA proposal 和历史判断动作意图
  决定保留整个 VLA chunk、只保留前缀，或给出修正 EEF
  根据前后图像判断抓空、放歪、滑落和释放失败
  在失败后规划不同的恢复动作
```

Host 不根据轨迹判断是否抓住物体。夹爪曲线检测只是调用 Codex 的触发信号，不是成功、失败、抓取或释放的语义结论。

## 2. 两种运行模式

同一个 bridge endpoint 支持两种模式，请求通过 `control.mode` 区分。

### 2.1 `codex_only`

保持当前链路：Codex 直接从 observation 选择下一目标 EEF，Host 进行 LERP/SLERP 插值。

输出 schema 保持不变：

```json
{
  "left": {
    "position": "keep",
    "orientation": "keep",
    "gripper": "keep"
  },
  "right": {
    "position": [0.0, 0.0, 0.0],
    "orientation": [1.0, 0.0, 0.0, 0.0],
    "gripper": "keep"
  },
  "note": "修正右臂位置",
  "phase": "align"
}
```

`left/right` 的 position 是绝对 EEF xyz，orientation 是绝对 wxyz；示例数值只表示字段形状，不是机器人或场景常量。

### 2.2 `vla_review`

xVLA 已经产生 H30 绝对 EEF proposal，Codex 对 proposal 进行仲裁。使用单独的固定输出 schema：

```json
{
  "mode": "vla_full",
  "execute_steps": 30,
  "left": {
    "position": "keep",
    "orientation": "keep",
    "gripper": "keep"
  },
  "right": {
    "position": "keep",
    "orientation": "keep",
    "gripper": "keep"
  },
  "review": {
    "event_type": "grasp",
    "event_arm": "right",
    "event_executed": true,
    "postcheck_required": true,
    "verification": "pending"
  },
  "note": "轨迹可执行，抓后复核",
  "phase": "grasp"
}
```

字段约束：

| 字段 | 含义 |
|---|---|
| `mode` | `vla_full`、`vla_prefix` 或 `eef` |
| `execute_steps` | `vla_full` 必须等于 horizon；`vla_prefix` 为 1..horizon；`eef` 固定为 0 |
| `left/right` | 仅 `eef` 时作为绝对目标；VLA 模式下必须全部为 `keep` |
| `review.event_type` | `none`、`grasp`、`release` 或 `uncertain` |
| `review.event_arm` | `none`、`left`、`right` 或 `both` |
| `review.event_executed` | 本次获准执行的范围是否包含该事件 |
| `review.postcheck_required` | 下一次 fresh observation 是否必须再次调用 Codex |
| `review.verification` | `not_applicable`、`pending`、`success`、`failed` 或 `uncertain` |
| `note/phase` | 保持当前简短审计字段约束 |

三种动作模式：

```text
vla_full
  执行完整 H30 proposal。

vla_prefix
  只执行 proposal[0:execute_steps]；其余 suffix 丢弃。
  典型用途是保留正确的 approach，丢弃不合适的 grasp/release。

eef
  不执行旧 VLA proposal。Codex 给出一个或双臂绝对 EEF 目标；Host 使用现有
  LERP/SLERP 插值器生成短 correction chunk。完成后丢弃旧 suffix，并重新调用 xVLA。
```

不要把 VLA 模式字段添加为当前 Codex-only schema 的可选字段。Bridge 根据 `control.mode` 选择对应 output schema 和验证器，避免两种语义混杂。

## 3. 策略服务请求 Codex 的数据结构

现有 observation 字段继续保留：

```json
{
  "episode_id": "ep-1",
  "request_id": "ep-1-0004",
  "step_id": 90,
  "turn_index": 2,
  "task": {
    "name": "stack_bowls",
    "instruction": "..."
  },
  "budget": {
    "max_decisions": 100,
    "max_sim_steps": 550,
    "remaining_decisions": 97,
    "remaining_steps": 460
  },
  "observation": {
    "left": {"position": [0, 0, 0], "orientation": [1, 0, 0, 0], "gripper": 1.0},
    "right": {"position": [0, 0, 0], "orientation": [1, 0, 0, 0], "gripper": 1.0}
  },
  "feedback": [],
  "images": []
}
```

VLA 模式增加 `control`：

```json
{
  "control": {
    "mode": "vla_review",
    "trigger": {
      "type": "proposal_gripper_candidate",
      "reasons": ["right_gripper_varies"],
      "recovery_active": false
    },
    "vla_proposal": {
      "proposal_id": "vla-ep1-0007",
      "action_space": "absolute_eef",
      "quaternion_order": "wxyz",
      "horizon": 30,
      "left": [
        {"index": 0, "position": [0, 0, 0], "orientation": [1, 0, 0, 0], "gripper": 1.0}
      ],
      "right": [
        {"index": 0, "position": [0, 0, 0], "orientation": [1, 0, 0, 0], "gripper": 1.0}
      ]
    },
    "proposal_diagnostics": {
      "finite": true,
      "left_gripper_range": [1.0, 1.0],
      "right_gripper_range": [0.55, 1.0],
      "candidate_arms": ["right"]
    },
    "previous_codex_decision": null,
    "pending_review": null,
    "since_last_codex": {
      "vla_chunks": 3,
      "sim_steps": 90,
      "start_step": 0,
      "record_paths": [
        "vla_chunks/0001.json",
        "vla_chunks/0002.json",
        "vla_chunks/0003.json"
      ]
    }
  }
}
```

以上位姿数组只说明 schema。实际请求必须包含完整 H30 双臂轨迹，不允许用示例零值填充。

### 3.1 `trigger`

`trigger.type` 使用固定枚举：

```text
proposal_gripper_candidate
post_gripper_check
recovery
```

`reasons` 是 Host 的事实描述，不是视觉结论，例如：

```text
right_gripper_varies
previous_decision_requires_postcheck
recovery_not_resolved
```

不要写 `grasp_failed`、`object_misaligned` 等 Host 无法从轨迹证明的判断。

### 3.2 `proposal_diagnostics`

只描述当前 H30 proposal：

- 数组是否有限。
- 每只手当前 chunk 的 gripper min/max。
- 哪些手出现超过噪声范围的变化候选。

它不携带历史 state，不判断物体大小，不使用固定“完全闭合”阈值，也不判断抓取或释放语义。

### 3.3 `previous_codex_decision` 与 `pending_review`

不要解析上一轮 `note`。策略服务保存上一条 VLA-review 结构化输出，并在下一次请求中原样引用其关键状态：

```json
{
  "previous_codex_decision": {
    "request_id": "ep-1-0003",
    "mode": "vla_full",
    "proposal_id": "vla-ep1-0006",
    "executed_steps": 30,
    "event_type": "grasp",
    "event_arm": "right",
    "postcheck_required": true
  },
  "pending_review": {
    "type": "post_gripper_check",
    "event_type": "grasp",
    "event_arm": "right",
    "before_observation_step": 60,
    "after_observation_step": 90
  }
}
```

`executed_steps` 必须由 Host 根据仿真实际回执填写，不能直接相信模型计划执行的步数。只有动作确实执行后，Host 才建立 `pending_review`。

### 3.4 前后图像

当前三路 RGB 总是作为 `images` 直接附加。`post_gripper_check` 还应直接附加对应事件执行前的三路图像，明确标记为 reference/before，而不是要求模型用 shell 查找。

更早的 VLA-only observation 和 chunk 保存在 episode 目录，只提供精确只读路径。Codex 仅在当前图像和 reference 图像仍有一个具体歧义时，才按现有 skill 规则使用 `view_image`。

## 4. 策略服务状态

每个 episode 维护：

```text
last_codex_decision     上一次结构化 Codex 输出及实际执行回执
pending_review          是否必须做夹爪事件后视觉验证
recovery_active         是否由 Codex 接管恢复
vla_since_last_codex    未进入 Codex 对话的 VLA-only chunk 记录
```

这些状态写入 rollout 审计日志，同时保存在进程内。Host 不能通过扫描 `note` 恢复控制状态。

## 5. Host 触发与执行流程

### 5.1 生成 proposal

每轮都先根据最新 observation 调用 xVLA，得到 H30 绝对 EEF chunk，并保存原始 proposal。

### 5.2 查询上一轮 Codex 状态

优先级：

```text
pending_review == true
  → 必须触发 post_gripper_check

recovery_active == true
  → 必须继续触发 recovery

否则
  → 对当前 proposal 做粗粒度夹爪变化检测
```

### 5.3 粗粒度曲线检测

检测目标仅是：当前 chunk 的某只夹爪是否“明显不恒定”。

不要依赖：

- 曲线导数或局部极值。
- 精确事件步。
- 固定 open/closed 物理阈值。
- 根据最终 opening 推断物体尺寸或是否夹住。

第一版可以使用可配置的低噪声阈值判断 `max-min`，宁可少量误触发 Codex，也不要把它解释为抓取成功/失败。原始 30 点 gripper 曲线必须一并交给 Codex。

### 5.4 是否调用 Codex

```python
trigger_codex = (
    pending_review is not None
    or recovery_active
    or proposal_has_coarse_gripper_candidate
)
```

没有触发：

```text
执行完整 H30
→ 保存 proposal、执行回执和新 observation
→ 追加到 vla_since_last_codex
```

触发：构造第 3 节的数据包并调用 bridge。

### 5.5 处理 Codex 输出

#### `vla_full`

校验 `execute_steps == horizon`，执行完整 proposal。若 Codex 声明事件已执行且要求 post-check，动作完成后建立 `pending_review`。

#### `vla_prefix`

校验 `1 <= execute_steps <= horizon`，只执行 proposal 前缀，suffix 永久丢弃。若前缀没有包含获准的夹爪事件，`event_executed` 必须为 false，也不能建立 post-check。

该模式用于：

```text
approach 合理、抓取位置不好
→ 保留 approach
→ 丢弃开始闭合及后续动作
→ fresh observation 重新调用 xVLA
```

#### `eef`

不执行 VLA proposal。Host 使用当前 Codex-only 相同的 LERP/SLERP 插值逻辑，将绝对目标转换成短 correction chunk。

插值步数根据当前位置到目标的平移和旋转距离确定，不保留原 H30 步数，也不填充到 30。执行完毕后丢弃旧 proposal，获取 fresh observation 并重新调用 xVLA。

#### 仿真回执

无论哪种模式，Host 根据实际执行回执更新：

```text
executed_steps
end_step
terminated/truncated
实际执行的 action 范围
pending_review
recovery_active
```

模型输出描述的是计划，仿真回执才是事实。

## 6. Codex 数据流

```text
策略服务 POST /v1/decide
  → bridge 严格校验 control.mode 和对应 schema
  → 保存当前 observation 图片与请求
  → 将当前图片、状态、预算、trigger、H30 proposal、diagnostics 渲染为 turn
  → post-check 时同时附加事件前 reference 图片
  → App Server 在 episode thread 中调用 Codex
  → Codex 根据 skill 做视觉审查和单次决策
  → bridge 按 control.mode 对输出做结构校验
  → 写 rollout.jsonl
  → 将结构化 decision 返回策略服务
  → 策略服务执行并记录仿真回执
```

VLA-only chunk 不产生 Codex turn，但必须由 Host 记录。下次触发时通过 `since_last_codex` 告知间隔，并提供记录路径。Thread 轮换时，bridge 回放已有 Codex turn；未进入对话的大量 VLA-only 数据仍留在文件中，不把整段 action history塞进上下文。

## 7. 写入 skill 的 Codex 决策规则

VLA 模式应在现有 skill 中增加独立章节，规则如下。

### 7.1 先判断本轮审查类型

- `proposal_gripper_candidate`：审查当前 proposal 的意图和局部可执行性。
- `post_gripper_check`：比较事件前后图像，验证真实结果。
- `recovery`：根据已经确认的失败选择不同的恢复目标。

Host 的 candidate 只是提示。必须结合图像、当前 state、完整 EEF trajectory 和任务阶段独立判断它是否是抓取、释放或无意义变化。

### 7.2 审查当前 proposal

- 图像和实测 state 是当前事实；VLA trajectory 只是未来机器人目标，不是物体未来状态。
- 不因轻微不美观或单纯不确定而接管。
- proposal 全部合理时用 `vla_full`。
- approach 合理但后面的抓取、释放或局部位置不合理时用 `vla_prefix`，保留有用前缀并在错误动作之前停止。
- 当前位置已经适合做明确局部修正时用 `eef`。
- 不要让 `eef` 延续旧 proposal；修正后将由 xVLA 基于 fresh observation 重新规划。

### 7.3 EEF 修正规则

- 输出绝对 xyz 和绝对 wxyz；不要输出 delta。
- 不决定速度、持续时间或插值步数，Host 拥有插值。
- 未修改的分量使用 `keep`。
- 对齐修正时保持夹爪，不要用闭合来探测物体大小。
- correction 完成不代表抓取、释放或到达成功，必须等待新 observation。

### 7.4 夹爪事件前审查

判断：

- 目标物体是否位于预期夹爪之间。
- 主相机和对应腕部相机是否支持当前抓取/释放位置。
- approach 是否值得保留。
- 当前 proposal 的闭合/打开是否过早。
- 放置前物体是否已经得到目标表面、容器或另一只手的支撑。

抓取位置不好但 approach 有用时，优先 `vla_prefix`，不要为了一个末端错误丢弃全部接近动作。

### 7.5 事件后验证

- 夹爪命令、gripper state 或 EEF 上抬本身不能证明抓取成功。
- 抓取后比较 before/current 图像，确认物体是否随夹爪移动、是否滑落或仍留在原位。
- 释放后确认物体是否与手指脱离、是否得到支撑、是否放正。
- `verification=success` 后才清除 post-check。
- `verification=failed` 时进入 recovery；不得原样重复上一次失败姿态。
- `verification=uncertain` 时保持安全状态并请求能消除歧义的短动作，不能谎报成功。

### 7.6 恢复规则

- 抓空后至少改变位置、高度、方向或抓取对象选择中的一项。
- 放歪后先判断物体当前是否仍被夹持、是否可安全重抓，再选择修正。
- recovery 未经 fresh observation 验证成功前，不交回普通 VLA-only 执行。
- 恢复成功后清除 `recovery_active`，再让 xVLA 从当前真实状态重新预测。

### 7.7 输出一致性

- `vla_full/vla_prefix` 时左右臂目标字段必须全部为 `keep`。
- `eef` 时 `execute_steps=0`，且只输出本次修正所需目标。
- `event_executed=false` 时不能要求该事件的 post-check。
- 只有本次获准执行范围包含事件时，才设置 `event_executed=true`。
- `note` 不超过现有简短约束，不能依赖 note 驱动 Host 状态机。

## 8. 审计记录

每次 VLA inference、Codex decision 和实际执行都使用稳定 id 关联：

```text
proposal_id
codex request_id（仅触发时存在）
execution_id
```

建议 rollout 记录：

```json
{
  "proposal_id": "vla-ep1-0007",
  "codex_triggered": true,
  "trigger": "proposal_gripper_candidate",
  "codex_decision": {},
  "planned_steps": 30,
  "executed_steps": 30,
  "pending_review_after_execution": true,
  "recovery_active": false
}
```

这使后续可以统计：

- Codex 调用次数和墙钟开销。
- VLA full/prefix/EEF 的比例。
- 候选检测误触发率。
- 抓取与释放的 pre/post-check 成功率。
- 抓空、放歪和重复失败是否下降。

## 9. 第一版实施范围

第一版只实现：

1. `control.mode` 和两套输出 schema。
2. H30 绝对 EEF proposal 传输。
3. 可配置的 gripper 曲线粗检测。
4. `vla_full / vla_prefix / eef`。
5. 结构化 `pending_review` 和 `recovery_active`。
6. 事件前/后图像审查。
7. VLA-only chunk、Codex decision 和执行回执的统一日志。

暂不实现：

- 根据图像 embedding 自动判断失败。
- 物体检测、深度估计或像素到世界坐标映射。
- Host 根据轨迹判断 grasp/release 成功。
- 对每个普通 VLA chunk 调用 Codex。

先用抓空率、放置稳定率、重复抓放次数、Codex 调用次数和 episode 墙钟验证这套稀疏机制，再决定是否增加更复杂的视觉触发器。
