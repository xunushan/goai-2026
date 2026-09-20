# 经验库与示例注入设计

## 目标

经验库为每个任务提供一个成功示例，给策略模型提供阶段顺序、手臂分工、末端姿态和夹爪时机等先验。它由 bridge 读取，不属于 Codex workspace，智能体不能自行遍历，也不会因为 skill 指令而重复读取。

设计约束：

- 任务名精确匹配，避免加载相似但错误的任务经验。
- 一个 episode 只加载、校验和 render 一次 demo。
- 同一个 Codex thread 不重复注入。
- 更换 thread 时回放 bridge 保存的完整 episode 上下文，包括最初的 demo。
- `demo.json` 保留所有原始图片路径；进入上下文的视图由配置选择。
- bridge 只提供历史参考，模型仍须根据当前 observation 决策。

## 目录结构

```text
codex_agent/
├── experience_library/
│   ├── sim_experience_library/
│   │   ├── index.json
│   │   ├── stack_bowls/
│   │   └── plug_in_charger/
│   └── real_experience_library/
│       ├── index.json
│       ├── fill_pen_holder/
│       └── ...
├── bridge/
│   └── experience.py
└── tools/
    └── render_experience.py
```

`--experience-library` 指向一个含 `index.json` 的具体经验库。默认使用
`experience_library/sim_experience_library`；真机启动时改为
`experience_library/real_experience_library`。该参数在 Bridge 进程启动时确定，
切换环境需要重启 Bridge，不会在一个 episode 中途切换。

## 任务索引与视图配置

所选经验库的 `index.json` 是任务选择和图片选择的唯一配置入口：

```json
{
  "stack_bowls": {
    "demo": "stack_bowls/demo.json",
    "views": {
      "default": ["cam_high"],
      "grasp": ["cam_high", "cam_left_wrist", "cam_right_wrist"],
      "place": ["cam_high", "cam_left_wrist", "cam_right_wrist"]
    }
  }
}
```

策略请求中的 `task.name` 必须与索引键完全一致。没有匹配项时不注入 demo，也不阻断策略调用。

`views` 按关键帧的 `stage` 选择图片：优先使用同名 stage 配置，否则使用 `default`。Python 代码不维护 `grasp/place` 等业务规则，只校验配置中的相机属于固定观测协议且没有重复。

## `demo.json`

每个 demo 包含任务说明和按时间排序的关键帧。关键帧至少包含：

```json
{
  "frame_index": 52,
  "stage": "grasp",
  "observation": "right jaws just closed on the bowl rim",
  "roles": {"left": "idle", "right": "active"},
  "state": {
    "left": {"position": [0, 0, 0], "orientation": [1, 0, 0, 0], "gripper": "open"},
    "right": {"position": [0, 0, 0], "orientation": [1, 0, 0, 0], "gripper": "closed"}
  },
  "decision": {
    "left": {"position": [0, 0, 0], "orientation": [1, 0, 0, 0], "gripper": "keep"},
    "right": {"position": [0, 0, 0], "orientation": [1, 0, 0, 0], "gripper": "keep"}
  },
  "result": "grasped",
  "images": {
    "observation.images.cam_high": "images/001-grasp-cam_high.jpg",
    "observation.images.cam_left_wrist": "images/001-grasp-cam_left_wrist.jpg",
    "observation.images.cam_right_wrist": "images/001-grasp-cam_right_wrist.jpg"
  }
}
```

`state` 是关键帧的实测状态；`decision` 是该关键帧关联的历史指令，仅作为经验，不会被当成当前待执行命令。所有三路图片路径都留在 JSON 中，选择发生在 render 阶段。

## 注入文本

每个 demo 先生成一段总说明：

```text
HISTORICAL SUCCESSFUL DEMONSTRATION
Task name: stack_bowls
Goal: ...
Reference only: reuse stage order, arm roles, grasp orientation and gripper timing;
adapt positions to the current images and measured state.
```

每个关键帧生成一个简短块：

```text
[EXAMPLE 2/5 | grasp | frame 52]
Observed: ...
Roles: left=idle, right=active
State: ...
Decision: ...
Outcome: grasped
```

文本块后紧跟配置选中的图片。图片由 bridge 读取并转换成 `data:image/...;base64,...`，本机文件路径不会发送给模型。完整人工审查样例见 `experience_library/sim_experience_library/stack_bowls/rendered_example.md`。

## Episode 上下文与 thread 轮换

bridge 维护两部分 episode 状态：

```text
initial_context = render 后的 demo 文本和图片 items
history         = 每轮 observation、图片缓存位置、结构化 decision
```

第一轮处理：

```text
按 task.name 查 index.json
→ 读取并校验 demo.json
→ 根据 views 选择图片并 render App Server items
→ 保存为 initial_context
→ initial_context + 当前 observation 发给第一个 thread
```

同一 thread 的后续轮次：

```text
只发送当前 observation
```

达到图片窗口、超时恢复或其它原因创建新 thread 时：

```text
回放已保存的 initial_context
→ 回放完整历史 observation/decision
→ 历史中只保留最近一轮的实时图片编码，旧图片保留受限缓存路径
→ 追加当前 observation
```

这里必须向新 thread 重新发送内容，因为 App Server 的新 thread 无法访问旧 thread；但 bridge 不会重新读取或 render demo，而是回放同一个 episode 已保存的 `initial_context`。因此不存在同一 thread 的重复注入，也不存在轮换时的重复 render。

## 校验与失败行为

bridge 启动时校验 `index.json`；首次使用任务经验时校验 demo：

- 任务名、demo 路径和视图配置合法。
- demo 的 `task_slug` 等于请求任务名。
- 每个关键帧包含双臂 state、decision 和固定三路图片路径。
- 位姿向量长度和数值合法。
- 图片路径不能逃出任务目录，文件必须存在且为 JPEG/PNG。

经验配置或文件错误返回 `experience_invalid` 并写入本轮审计日志，不会让模型在缺失或错配经验的情况下静默执行。

## 增加任务

1. 在目标库中新建 `<task>/demo.json` 和 `images/`。
2. 在该库的 `index.json` 增加与策略请求 `task.name` 完全一致的键。
3. 配置 `views.default` 和需要覆盖的 stage。
4. 运行 `tools/render_experience.py <demo.json>` 检查文本。
5. 更新或生成 `rendered_example.md` 做图片与文本联合 review。
6. 运行 `python -m pytest codex_agent/tests/test_experience.py -q`。
