# codex_agent

用本地 Codex CLI 当**高层策略**：一个 episode 一个 Codex thread，每次决策输出一个
**目标末端位姿**，由 `motion.py` 的 LERP+SLERP 插值器展开成 action chunk。

> 本文件是**手写**的进展与路线图，与同级目录由模板生成的 README 不同——若将来统一重刷
> 模板，请保留本文件。逐层的调用链路（含 `mode` 全表与计数器语义）见
> [CALL_FLOW.md](CALL_FLOW.md)；运维与接口细节见 [instruction.md](instruction.md)，
> Mac 侧策略服务的部署见 [bridge/README.md](bridge/README.md)。

---

## 改造进展（第二步 · 已完成）

**策略服务与工作空间隔离**。出发点：把「谁读策略、谁只搬字节」这件事彻底分开，
对齐参照项目 `agent_policy` 的分工——一个常驻 Codex App Server、一个**由 Codex 自己读的
workspace**、一个只做传输与校验的 bridge。

| # | 问题 | 状态 | 落地位置 |
|---|---|---|---|
| E | 每轮新起 `codex exec` 子进程：thread 靠 `--resume` + cwd 过滤，read-only 沙箱，图片只能附在命令行 | ✅ | 换成**常驻 `codex app-server --stdio`**（JSON-RPC）：`bridge/app_server.py`，thread 由 bridge 持有并按 episode 记忆 |
| F | 提示词在 GPU 侧整段渲染后过隧道：文本有多个作者 | ✅ | 当轮文本由 `bridge/bridge.py` 直接组装，常驻规则在 `workspace/AGENTS.md` / `SKILL.md`；GPU 侧只发结构化观测包（`observation.py`） |
| G | 提示词里写着一堆**我们说不清出处的数字**：起始位姿、示范数据统计出来的工作空间盒、桌面高度 z=0.76 | ✅ | 全部删除。只留两类：单次决策增量上界（0.05 m / 0.35 rad）与夹爪标度端点（0 / 1）；其余数值只能是本轮实测值或操作员预算 |
| H | 图片在 GPU 侧被缩到长边 480，而原图后续另有用途 | ✅ | **不裁不缩**（照 GPT-Policy）：`encode_image` 只做 JPEG 质量转换，bridge 原样落盘收到的字节；`max_edge` / `max_width` 现在都是硬报错 |
| I | 没有审计留痕：收发的图片、每轮 turn 只存在于日志行里 | ✅ | `workspace/output/<episode_id>/`：`rollout.jsonl` 一行一轮、`observations/<camera>/NNNNNN_<request_id>.jpg|png` 逐字节原样、`scratch/` 是唯一可写目录 |
| J | 推理过程不可见，模型有没有真去读 workspace、有没有回看历史帧都无从判断 | ✅ | 权限模型（`workspace/.codex/config.toml` + bridge 用 `-c` 显式下发）：`view_image` / `shell_tool` 开，`web_search` / `computer_use` / `multi_agent` 关 |

**第一步**（已完成，保留在此）：`guardrail:` 段补上单次决策上界、
`_load_task_card` 翻转优先级（任务卡优先、配置兜底，`deploy.py` 未动）、
提示词外置并补上三条关键措辞、角度全仓统一为 rad、
插值粒度换成 GPT-Policy 的 `0.005 m / 0.035 rad`。

### 数值规则（决策 4，两个 suite 盯着）

提示词、turn 文本与 `AGENTS.md` 里**只允许**出现：

1. **单次决策增量上界**：`0.05 m` / `0.35 rad`；
2. **夹爪标度端点**：`0` 全闭 / `1` 全开。

此外出现的每一个数，只能是**本轮实测值**（手臂当前位姿、夹爪读数）或**操作员设的预算**
（决策/步数上限）。**禁止**再出现任何坐标、工作空间盒、桌面高度，也**禁止**用示范数据的
统计量（p90/q99）冒充世界事实——模型会照着一个我们其实站不住的数去避让。

`tests/test_workspace.py` 与 `tests/test_turn_text.py` 逐字盯着这条规则：
前者管 `AGENTS.md` / `SKILL.md`，后者管组装出来的 turn 文本，并且每张任务卡都会被渲染一遍，
所以新增任务不会绕过检查。

### 验证（离线，零 Codex 额度、不联网）

```bash
bash tests/run_all.sh                  # 7 个 suite 共 18062 checks passed / all suites passed
                                       # motion 3020 · protocol 74 · workspace 190 · app server 60
                                       # turn text 1364 · bridge 122 · adapter 13232
python tools/render_prompt.py --turn 3 # 离线看本轮 turn 文本
```

端到端干跑（决策 13 的回归点，零额度）：`--codex-bin tests/fake_app_server` 起 bridge，
用 `observation.build_request` 造一个真报文 POST `/v1/decide`，确认四件事——返回的
`decision` 过得了 `protocol.parse_decision`（`problems` 为空、`gripper: "close"` → `0.0`）、
`rollout.jsonl` 每轮追加一行、`observations/NNNNNN_<request_id>/` 下出现三张图、
**落盘字节与请求里 `b64` 解码后逐字节相同且分辨率仍是 640×480**。同一 episode 的第二个
turn 复用同一目录、日志递增到两行。

- **未触碰**（按约束）：`deploy.py`（与 17 个同级策略逐字节相同，
  md5 `fc9d031ed6efaabf34e7fc69c862eaa0`，改它就是分叉共享适配器）、
  `eval.sh`（`robodojo.sh` 靠 grep 它决定参数个数，必须保持 10 个参数且不出现 `expert_num`）。
- `tests/fake_codex` **已删除**（决策 14）：它模拟的是 `codex exec --json` 的 argv 与事件流，
  而新的 bridge 说的是 `codex app-server --stdio` 的 JSON-RPC，留着它无法离线验证新 bridge。
  替代品 `tests/fake_app_server` 保留其中一条实质约束：**未知 CLI 参数 exit 2**。
- `motion.py` **曾**在「不要碰」之列，现已解锁并改动：改动面限于角度单位改 rad 与两个粒度取值
  （`0.015 m / 5°` → `0.005 m / 0.035 rad`，对齐 GPT-Policy 的 `cartesian_step_m` /
  `cartesian_step_rad`），插值算法、guardrail 判据、chunk 语义一律未动。
  当时给的理由是"改它就是分叉共享适配器"，但该理由**在 `motion.py` 上不成立**：
  43 个同级 policy 里只有 `codex_agent` 有 `motion.py`，它是本项目独有的文件；
  真正共享的是 `deploy.py`（43 个都有，其中 17 个同一份 md5），其逐字节约束不受影响。
- **第一步里那条 `max_width: 640` → `max_edge: 480` 的记录已被取代**（本 README 的表格
  随之改写）：图像现在**不做任何缩放**，按仿真器给的分辨率发送。代价是实打实的
  （面积 token 约 ×1.8，3 视角/轮），而当年引入 `max_edge` 正是为了省这部分——所以这里
  记一句为什么改回来：**帧要在别处也用，裁掉就找不回来了**，而模型选目标位姿靠的正是
  这些像素。还剩的唯一旋钮是 bridge 的 thread 轮换窗口（带图 turn 数，默认 8）。
  `deploy.yml` 的 `images:` 注释里留着同一段账。

### 实测：`episode:` 的两个预算如何耦合

单次决策的仿真步开销 = `max(ceil(距离/0.005), ceil(角度/0.035)) + settle`，满额 **13 步**。
**两个轴是并行的，不是相加**：需要更多步的那条轴决定整体速度，所以取 `max` 而非求和。
（本表与 `deploy.yml` 的 `guardrail:` 注释同源；两处此前都把公式写成加法，按加法算是 12，
与它们自己给出的 8 矛盾，同批修正。）
以「每次都要求满 5 cm」实测（表也存在 `deploy.yml` 的 `guardrail:` 注释里）：

| `max_codex_calls` | 拿到完整 5 cm 的决策数 | 消耗仿真步 |
|---|---|---|
| **42** | 42 / 42 | 546 / 550（**浪费 4 步**） |
| 60 | 31 / 60 | 550 / 550 |
| 69 | 25 / 69 | 550 / 550 |
| **100（当前）** | **6 / 100** | 550 / 550 |
| 110 | 0 / 110 | 550 / 550 |

结论：100 与 550 不冲突（无溢出），但
**第 7 次决策起上限被压到 7 个 chunk 步**——"每次最多 5 cm"这句话只对前 6 次成立。
42 是「宣称的上界全程可用」的预算点。这是取舍，不是 bug；见下方「待定」。

> 粒度从 `0.015 m / 5°` 换成 `0.005 m / 0.035 rad` 后，满额开销由 8 步涨到 13 步，
> 于是「上界全程可用」的预算点从 69 落到 42、当前 100 次预算下的满额决策数从 16 落到 6。
> 预算本身**不动**：550 / 400 是 RoboDojo 仿真器自己的单集步数上限，加不了。
> 可回调的旋钮是 `min_sim_steps_per_call`（5→3 可把满额决策数从 6 提到 25），不是粒度。

### 计划外顺手修的

- **`bridge/app_server.py` 的换进程竞态**（真 bug，非测试瑕疵）：`start()` 会换一个新
  `queue.Queue()`，理由是「旧 reader 线程退出时在旧队列里留了 end-of-stream 标记，重启后
  先读到它就会以为自己已经死了」——注释写对了，实现漏了半步：`_read_stdout` 的
  `finally` 里写的是 `self._messages.put(EOFError(...))`，**属性是那一刻才解析的**。
  旧进程退出 → reader 线程醒来，若晚于 `start()` 换队列，它就把上一进程的 EOF（以及
  管道里最后一行）投进了**新进程的队列**。后果是下一轮 turn 立刻读到 EOF，报
  `app_server_closed`，而不是 stub 真正发来的 `policy_turn_failed` /
  `policy_usage_exhausted` / `policy_invalid_response`。`test_app_server.py` 里
  「each way a turn can fail」一组每个用例都要重启一次进程，所以**实测 4/20 复现**。
  修法是把队列作为参数绑给读它的线程（`_read_stdout(stream, messages)`），并加一条
  按**顺序**而非靠抢跑复现的回归用例；对同一处做变异（改回 `self._messages`）该用例
  会红。生产路径同样吃这个 bug：每换一个 episode 就重启一次 app server。
- **`scratch/` 目录没有测试盯着**：`record.prepare()` 建它是有理由的（写权限指着这个
  路径，缺失会在 turn 内表现为一次莫名其妙的拒绝），但没人验证它真的建了。已补断言。
- **`tools/render_prompt.py` 本来就是坏的**（与改造无关）：调 `_ensure_prompt_context()`
  时没有 episode；读 `model.home_left`（该属性不存在，起始位姿已改为取自首帧观测）；
  示例 feedback 编造了适配器从不输出的一行、又漏了它每次都带的一行。
  **`tools/` 下没有测试覆盖**，所以一直漂移。
- `instruction.md` 里一行 `max_validation_retries`——**全仓库不存在这个字段**。
- `instruction.md` 与 `bridge/README.md` 对 bridge 的启动目录互相矛盾（一个说 repo 根、
  一个说 `bridge/` 内），两处都提到的 `--config bridge/config.json` 指向一个**不存在的文件**。
  已统一为：`cd codex_agent && python3 -m bridge.bridge`。
- `bridge/README.md` 里写死的「15 in the shipped `deploy.yml`」也一并清掉了
  （预算现在只在任务卡与 `deploy.yml` 的 `episode:` 里，两处都不写 15）。

---

## TODO（未做）

- [ ] **起一集真实评测**并在 `workspace/output/<episode>/rollout.jsonl` 里确认：
      `mode=target_rejected` 出现时 `reason` 是距离/角度超限、
      `remaining_decisions` 从 100 递减、**整集墙钟**（决定 550 步与 100 次是否真的匹配）。
      顺带核对 `observations/` 里的图片**逐字节等于**当时发出去的字节、分辨率与报文一致。

      ⚠️ 100 次 × 25–30 s ≈ **45–50 分钟**；评测客户端有没有单集墙钟上限**尚未核实**。
      ⚠️ 会消耗 Codex 额度，跑之前先确认配额。

- [ ] **确认 workspace 真的生效**（验证第 4 步）：跑一次真 `codex app-server`，确认
      `AGENTS.md` / `SKILL.md` 被读到、`view_image` 能打开历史帧。这一条**必须消耗额度**，
      所以放在真实评测之前一起做。

- [ ] **轮换阈值 8 未经实测**：100 次决策 ÷ 8 ≈ 12 次重建 thread，回放开销多大、
      重建后模型还能不能接上，都要等实测。这也是图像不裁之后**唯一剩下的 token 旋钮**。

---

## 待定（需要拍板）

1. **`max_codex_calls` 用 100 还是 42。** 100 的代价见上表（前 6 次才有完整 5 cm）。
   100 = 后半程更细的决策；42 = 宣称的上界全程可用。当前值 100 是明确要求后保留的
   ——粒度变细使这个取舍明显变陡（满额决策 16 → 6），若实测发现任务推不动，
   先调 `min_sim_steps_per_call`（5→3 可把满额决策数提到 25），再考虑这一个。
2. **`tools/render_prompt.py` 的 `RENDER_POSE` 与 `tests/mock_bridge.py` /
   `tests/test_turn_text.py` 的常量重复**，会各自漂移。是否合并到一处。
   （其中 `tests/test_turn_text.py` 的渲染位姿**故意**从起始位姿偏移，
   否则信息边界检查会把自己的测量值当成泄漏——合并时别把这个区别抹平。）
3. **`tools/` 无测试覆盖**——上面那个工具的三个陈旧 bug 就是这么积累的。
   是否给 `render_prompt.py` 加一条冒烟用例。
4. ~~**`prompts/system.md` 与 `prompts/turn.md` 怎么办**~~ ——**已删除**（决策 12 的"保留供对照"
   被撤销）。删除前逐项核对过无独有内容：`system.md` 的 DISCIPLINE / WHO EXECUTES /
   WHAT A DECISION DOES NOT TELL YOU 已分别落在 `workspace/AGENTS.md`（本体）与
   `SKILL.md`（流程，含 `note` / `phase` / `keep` / 一轮一个动作）里，
   `turn.md` 的五个字段由 `bridge/bridge.py` 渲染。策略内容以 `AGENTS.md` / `SKILL.md`
   为准。
