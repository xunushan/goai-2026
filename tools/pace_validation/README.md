# PACE 离线标定与验证（sim_lerobot_v30_ee）

对 [PACE: Phase-Aware Chunk Execution for Robot Policies with Action Chunking](https://arxiv.org/abs/2606.00537)
（arXiv:2606.00537v2）在 `data/sim_lerobot_v30_ee/sim_lerobot_v30_ee.csv` 上做的标定与验证记录。

- 算法实现：`RoboDojo/XPolicyLab/policy/X_VLA_OPT/pace.py`（策略服务运行时用的同一份）
- 单测：`RoboDojo/XPolicyLab/policy/X_VLA_OPT/test_pace.py`（27 项，纯 numpy）
- 标定/验证 CLI：`tools/pace_calibrate.py`
- 地面真值：`tools/keyframe_events.py` 的 per-arm 关键事件帧 `t0`（grasp / place / insert …）
- 产物（未进仓库，`outputs/` 已 gitignore）：`outputs/pace_validation/`

```bash
PY=/opt/anaconda3/envs/lerobot/bin/python
$PY tools/pace_calibrate.py calibrate --percentile 85 --stride 5
$PY tools/pace_calibrate.py validate  --tolerance 5 --per-task 2
```

## 1. 论文前提成立：相位切换点确实是速度谷值

对每个标注事件 `t0`，在同一臂的全轨迹平滑速度剖面上找最近的严格局部极小：

| 指标 | 值 |
| --- | --- |
| 事件在 ±5 帧内有谷值的比例 | **0.851** |
| 最近谷值距离中位数 / p75 / p90 | 3 / 4 / 7 帧 |
| 有符号偏移 `valley_idx - t0` 中位数 | **−1**（减速略早于夹爪关键帧，符合物理） |
| 事件处最近谷值的 prominence 中位数 | 8.3e−03（全池中位数 4.2e−04，**差 ~20 倍**） |

论文 §3.2 的「相位过渡表现为减速谷」在这一数据上成立。

## 2. 论文默认 ρ=5 的标定规则在本数据上不可用

论文 §3.3：「每个任务用训练示范标定一次 δ_T」，Table 3 默认 ρ=5。
实测（`fill_pen_holder`，stride 25 冒烟）得到 `delta_t(p5) = 1.19e-08`。

**根因是候选集被静止段污染，不是阈值取小了**：

- 示范动作以 float32 存储，静止步的坐标增量是 **2⁻²⁴ 的整数倍**（实测速度取值
  `5.96e-08, 8.94e-08, …`），即数值噪声；
- 左臂 **23.6% 的步位移为 0**、39.5% 低于 1e-5（双臂协作里一条臂常静止等待）；
- 静止段上的随机游走必然产生大量严格局部极小，其 prominence 就是噪声量级。

ρ→δ_T→mean_h 的完整映射（`fill_pen_holder`，train 池 n=45380）：

| ρ | δ_T | mean_h |
| --- | --- | --- |
| 5 | 1.19e−08 | 12.6 |
| 50 | 7.21e−05 | 15.6 |
| 75 | 1.54e−03 | 20.3 |
| **85** | **2.99e−03** | **23.5** |
| 95 | 5.66e−03 | 27.5 |

即 ρ=5 让几乎所有谷值被接受、h 塌向 `h_min`，与论文报告的「平均执行视野 24.3/50」
相差甚远。

### 已排除的两个解释

- **「论文的平滑算子 S 更强」**——**证伪**。窗口 1→27 扫描：p05 始终在
  1e−09…5e−08，强化平滑同时压低噪声底与信号，ρ=5 永远在噪声层。
- **「prominence 应该归一化（论文措辞里的 relative）」**——**不成立**。改用
  `Φ_rel = (base−v)/base` 后 ρ=5 给出 δ_rel=0.008/0.016/0.004，mean_h 仅
  8.5/13.8/15.5，仍远低于 24.3。噪声谷在两种量纲下都占多数，**按构造 ρ=5 必然
  落在噪声层**。（第三方实现 `Bookhou/smolvla-rgbd-so101` 确实用了相对比值
  `PROMINENCE_RATIOS=(0.08,0.12,0.16)`，但它同样配了 `LOW_SPEED_QUANTILE=0.45`
  之类的守门项，且自述为 diagnostic 而非复现。）

### 采用的取值

取 **ρ=85**（仍只用训练 split 示范，不含验证集、不含事件标签），因为它恰好复现论文
的操作点。交付到 `deploy.yml` 的值：

| 任务 | δ_T |
| --- | --- |
| `fill_pen_holder` | 2.990e−03 |
| `plug_in_charger` | 7.088e−03 |
| `stack_bowls` | 2.344e−03 |
| 全局兜底（`stack_blocks` 无示范） | 3.219e−03 |

## 3. 验证结果

验证 split 30 个 episode / 202 个标注事件 / 533 次重规划查询。

| 任务 | δ_T | mean_h | 事件命中率 | 等预算固定 H | 固定 H 命中率 |
| --- | --- | --- | --- | --- | --- |
| `fill_pen_holder` | 2.99e−03 | 23.1 | 0.618 | 23 | 0.559 |
| `plug_in_charger` | 7.09e−03 | 24.7 | 0.417 | 25 | 0.104 |
| `stack_bowls` | 2.34e−03 | 27.0 | 0.519 | 27 | 0.308 |
| **合计** | | **24.47** | | | |

- **操作点复现**：`mean_h = 24.47`，论文在 `H_max=50` 上报告 24.3。
- **等预算对照**（唯一有判别力的口径）：把固定视野调到与 PACE 相同的平均执行长度，
  命中率 **0.386 → 0.545（相对 +41%）**。这与论文 Table 4 的结论同向：收益不只是
  「平均多查询了几次」，而是**查在该查的位置**。

### 度量口径的两个坑（写进 `pace_validation.json` 的 `metrics_note`）

1. **命中率随边界数单调上升**：`H=1` 时边界数等于帧数，必然 100% 命中，故
   `best_fixed_h` 恒为 1，**不可用于比较**。只能用等预算口径。
2. **本验证测不出 PACE 的真实收益**。用示范动作充当预测 chunk（论文标定同法），
   而示范 chunk 在相位边界两侧都是「正确」的，因此测不到 PACE 真正要解决的问题
   —— 执行一段跨越相位的 chunk 会让后半段动作失效。要测这个必须在仿真里做 A/B。

## 4. 未决事项与建议

1. **真实收益必须仿真 A/B**。示范 chunk 与策略预测 chunk 的**噪声层不同源**
   （前者是 float32 量化，后者是 flow-matching 输出抖动），所以示范标定出的 δ_T
   不能默认迁移。建议在仿真上扫 δ_T ∈ {1e−03, 3e−03, 1e−02}，同时跑
   `enabled: false` 的固定视野基线。
2. **第三方独立复现给出负面结果**，与本节的谨慎结论一致：
   `LiuQiongwen/owg-mujoco-soarm` 显式实现了 PACE 式的 `pace_execute_length()`
   （无平滑、无 prominence 阈值），作者自己的实验是 22/32 → 20/32（Fisher p=0.79），
   已回退到固定 execute_steps。
3. PACE **没有官方开源实现**（论文全文 / arXiv HTML / 作者主页 / HF / S2 / alphaxiv
   均无 code 链接；`paperswithcode` 已下线，HF Papers 未收录该论文）。若能找到官方
   代码，应优先核对 S 与 Φ 的具体形式。
