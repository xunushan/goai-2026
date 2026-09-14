"""PACE: Phase-Aware Chunk Execution（arXiv:2606.00537v2）执行层实现。

论文把「预测视野 L」与「执行视野 h」分开：策略每次 query 预测 L 步动作，机器人
执行前 h 步后重新观测。论文实验显示 h 对成功率的影响是任务相关且非单调的
（Fig.1：同一任务上 H≈6 峰值、H≈25 掉近 40 点、H≈35 又回升），固定 h 不可靠；
PACE 改为只从**预测 chunk 自身**的运动学结构在线选 h：

  1. 对每个「执行臂」把 chunk 的手臂运动分量映射为一维速度剖面 v（长度 L-1）；
  2. 平滑 v 得到 ṽ，抑制短程抖动造成的伪谷值（Table 2：平滑在两个运动空间下都
     提升成功率，且抬高了平均执行视野）；
  3. 在 ṽ 上取低速谷值作为候选相位边界，每个候选带 prominence 分数 Φ，衡量它
     相对周边运动是不是一次「明显减速」——接触前准备、对准、抓取、释放等相变点
     通常表现为减速谷（§3.2）；
  4. 多臂候选取并集，Φ ≥ δ_T 的候选被接受，取**最早**的一个作为 h；没有可接受
     候选时回落到 H_max（§3.3）。δ_T 每个任务用训练示范标定一次。

PACE 只用预测 chunk，不需要策略内部信号、辅助头、重训练或推理引擎改动，因此可以
直接挂在既有的 20 维中间表示上（与 gripper_hysteresis 同层，见 model.py
`_finalize_chunk`）。

本 harness 的语义映射
----------------------
`XPolicyLab/client_server/ws/model_client.py` 中 `update_obs` 只在客户端本地缓存，
只有 `get_action` 会走网络，因此**观测刷新周期 = 服务端返回的动作数**。所以 PACE
选出的 h 直接就是每次返回给客户端的动作数，等价于论文的 execution horizon；
PACE 关闭时该值恒为 deploy.yml 的 `actions_per_chunk`（现状行为）。

与论文的已知偏离（论文正文/附录 A 未给出全部细节，全部做成可配置项）
------------------------------------------------------------------
- 平滑算子 S：论文只做「平滑 / 不平滑」消融，未给具体形式。默认对称滑动平均
  （`smooth_window=5`，边缘复制填充），可用 `smooth: none` 关闭。
- prominence Φ：论文只说它衡量「相对周边运动是否显著减速」，未给解析式。这里采用
  标准地形 prominence（等价于 scipy.signal.peak_prominences 作用于取负剖面），
  搜索范围限制在 [0, H_max-1]，见 `valley_prominences`。
  论文措辞里的「relative」也可读成归一化 Φ_rel = (base - v)/base，第三方实现
  （Bookhou/smolvla-rgbd-so101 的 `PROMINENCE_RATIOS`）就是这么做的。但在本数据上
  实测**两种定义都救不了 ρ=5**：Φ_rel 的 δ_T 仍是 0.008/0.016/0.004 量级，mean_h
  只有 8.5/13.8/15.5（论文操作点是 24.3/50）。原因是噪声谷在两种尺度下都占多数，
  按构造 ρ=5 必然落在噪声层 —— 所以病根是候选集被静止段污染，不是量纲。故保留
  绝对定义（与 prominence 的通用术语一致，且已通过离线验证），改用更大的 ρ。
- h_min：论文规定 h ∈ {1,...,H_max}，没有下限。本 harness 里一次 query 就是一次
  完整 VLA 前向，h 过小会成倍拉长评测耗时，故加 `h_min`（默认 5）作成本保护。
- 速度剖面分量：X-VLA 动作是 EE 空间的 xyz(3)+rot6d(6)+夹爪(1)，与论文的关节空间
  不同。论文 Table 2 显示平滑后 Cartesian 与 joint 相当（64.5 vs 64.2），这里取
  xyz 位移模长（= 论文 Cartesian 变体），单位为米/步，与示范数据同量纲。

δ_T 标定与验证结论（tools/pace_calibrate.py，2026-09-14 于 data/sim_lerobot_v30_ee）
----------------------------------------------------------------------------------
论文默认 ρ=5 的百分位规则在本数据上不可用（δ_T≈1.2e-08 落在 float32 噪声层，h 塌
到 h_min）；改用 ρ=85 后 mean_h=24.47，复现论文报告的 24.3/50 操作点。等预算对照
（PACE vs 固定 H = PACE 平均视野）在 202 个标注事件上为 0.545 vs 0.386。
以上只测相位边界的**定位质量**；PACE 真正的收益（不执行跨相位的 chunk，避免后半段
动作失效）无法用示范 chunk 离线测出，必须在仿真里做 A/B。详见
tools/pace_validation/README.md。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

# X-VLA 20 维 ee 动作布局（xvla/models/action_hub.py::EE6DActionSpace）：
#   0:3 左臂 xyz | 3:9 左臂 rotate6d | 9 左夹爪
#   10:13 右臂 xyz | 13:19 右臂 rotate6d | 19 右夹爪
ARM_XYZ_IDX: dict[str, tuple[int, int, int]] = {
    "left": (0, 1, 2),
    "right": (10, 11, 12),
}
DEFAULT_ARMS: tuple[str, ...] = ("left", "right")

_SUPPORTED_SMOOTH = ("moving_average", "none")
_SUPPORTED_PROFILE = ("xyz",)

_TRUE_TOKENS = ("1", "true", "yes", "on", "y", "t")
_FALSE_TOKENS = ("", "0", "false", "no", "off", "null", "none", "n", "f")


def _as_bool(value: Any, *, name: str) -> bool:
    if isinstance(value, str):
        token = value.strip().lower()
        if token in _FALSE_TOKENS:
            return False
        if token in _TRUE_TOKENS:
            return True
        raise ValueError(f"{name} must be a boolean, got {value!r}")
    return bool(value)


def _as_float(value: Any, *, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in ("", "null", "none"):
        return None
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"{name} must be finite or null, got {value!r}")
    return number


@dataclass(frozen=True)
class PaceConfig:
    """deploy.yml 的 `pace` 配置段解析结果（仅配置，无状态）。"""

    enabled: bool = False
    h_max: int = 30
    h_min: int = 5
    d_min: int = 10
    smooth: str = "moving_average"
    smooth_window: int = 5
    threshold_delta: float | None = None
    threshold_delta_by_task: dict[str, float] = field(default_factory=dict)
    arms: tuple[str, ...] = DEFAULT_ARMS
    profile: str = "xyz"

    def __post_init__(self) -> None:
        if self.profile not in _SUPPORTED_PROFILE:
            raise ValueError(
                f"pace.profile must be one of {_SUPPORTED_PROFILE}, got {self.profile!r}"
            )
        if self.smooth not in _SUPPORTED_SMOOTH:
            raise ValueError(
                f"pace.smooth must be one of {_SUPPORTED_SMOOTH}, got {self.smooth!r}"
            )
        if self.smooth == "moving_average" and self.smooth_window < 1:
            raise ValueError(
                f"pace.smooth_window must be >= 1, got {self.smooth_window}"
            )
        if not self.arms:
            raise ValueError("pace.arms must not be empty")
        for arm in self.arms:
            if arm not in ARM_XYZ_IDX:
                raise ValueError(
                    f"pace.arms contains unknown arm {arm!r}; "
                    f"known arms are {sorted(ARM_XYZ_IDX)}"
                )
        if self.h_max < 1:
            raise ValueError(f"pace.h_max must be >= 1, got {self.h_max}")
        if self.h_min < 1:
            raise ValueError(f"pace.h_min must be >= 1, got {self.h_min}")
        if self.h_min > self.h_max:
            raise ValueError(
                f"pace.h_min must be <= h_max, got h_min={self.h_min}, h_max={self.h_max}"
            )
        if self.d_min < 1:
            raise ValueError(f"pace.d_min must be >= 1, got {self.d_min}")
        for task, value in self.threshold_delta_by_task.items():
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(
                    f"pace.threshold_delta_by_task[{task!r}] must be a finite "
                    f"non-negative number, got {value!r}"
                )

    def threshold_for(self, task_name: str | None) -> float | None:
        """按任务取 δ_T；未命中则回落到全局 `threshold_delta`（可能为 None）。"""
        if task_name is not None:
            hit = self.threshold_delta_by_task.get(str(task_name))
            if hit is not None:
                return float(hit)
        return self.threshold_delta

    @classmethod
    def from_model_cfg(
        cls, model_cfg: dict[str, Any], actions_per_chunk: int
    ) -> "PaceConfig":
        """从 deploy.yml 解析 `pace` 段。`h_max: null`（默认）取 actions_per_chunk，
        即「论文 H_max = 固定 horizon 基线」，保证 PACE 关闭/回落时行为与现状一致。"""
        raw = model_cfg.get("pace") or {}
        if not isinstance(raw, dict):
            raise TypeError(
                f"pace config must be a dict, got {type(raw).__name__}"
            )
        configured_h_max = raw.get("h_max")
        h_max = (
            int(actions_per_chunk)
            if configured_h_max is None
            or (isinstance(configured_h_max, str)
                and configured_h_max.strip().lower() in ("", "null", "none"))
            else int(configured_h_max)
        )
        # h_min 未显式配置时用默认 5，但不得超过 h_max：h_max 缺省跟随
        # actions_per_chunk，既有的 actions_per_chunk < 5 配置（PACE 关闭）不应因为
        # 本模块的默认值而在 __init__ 报错。
        raw_h_min = raw.get("h_min")
        if raw_h_min is None or (
            isinstance(raw_h_min, str)
            and raw_h_min.strip().lower() in ("", "null", "none")
        ):
            h_min = min(5, h_max)
        else:
            h_min = int(raw_h_min)
        raw_table = raw.get("threshold_delta_by_task") or {}
        if not isinstance(raw_table, dict):
            raise TypeError(
                "pace.threshold_delta_by_task must be a dict of task_name -> delta, "
                f"got {type(raw_table).__name__}"
            )
        table = {
            str(task): float(value) for task, value in raw_table.items()
        }
        raw_arms = raw.get("arms")
        if raw_arms is None:
            arms = DEFAULT_ARMS
        elif isinstance(raw_arms, str):
            arms = tuple(part.strip() for part in raw_arms.split(",") if part.strip())
        else:
            arms = tuple(str(part).strip() for part in raw_arms if str(part).strip())
        return cls(
            enabled=_as_bool(raw.get("enabled", False), name="pace.enabled"),
            h_max=h_max,
            h_min=h_min,
            d_min=int(raw.get("d_min", 10)),
            smooth=str(raw.get("smooth") or "moving_average").strip().lower(),
            smooth_window=int(raw.get("smooth_window", 5)),
            threshold_delta=_as_float(
                raw.get("threshold_delta"), name="pace.threshold_delta"
            ),
            threshold_delta_by_task=table,
            arms=arms,
            profile=str(raw.get("profile") or "xyz").strip().lower(),
        )


# ---------------------------------------------------------------------------
# 算法（纯函数，便于离线复现与单测）
# ---------------------------------------------------------------------------

def arm_speed_profile(
    chunk: np.ndarray, xyz_idx: tuple[int, int, int]
) -> np.ndarray:
    """论文 §3.2 的 ψ^b：单臂位置分量的逐步位移模长。

    返回长度 L-1 的数组，`v[j] = ||a[j+1] - a[j]||`（0-based），即 v[j] 是
    a[j] 与 a[j+1] 之间的那一步速度。因此若 v[j] 是谷值、在其处重规划，
    对应执行前 h = j+1 步。
    """
    points = np.asarray(chunk, dtype=np.float64)[:, list(xyz_idx)]
    if points.shape[0] < 2:
        return np.zeros(0, dtype=np.float64)
    return np.linalg.norm(np.diff(points, axis=0), axis=1)


def smooth_profile(
    profile: np.ndarray, window: int, mode: str = "moving_average"
) -> np.ndarray:
    """论文 §3.2 的 S：抑制短程波动，避免伪谷值触发过频重规划。

    对称滑动平均 + 边缘复制填充（不缩短序列、不引入零值边界）。
    """
    values = np.asarray(profile, dtype=np.float64)
    if mode == "none" or values.size < 3 or window <= 1:
        return values.copy()
    effective = min(int(window), values.size)
    if effective % 2 == 0:
        effective -= 1
    if effective <= 1:
        return values.copy()
    pad = effective // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    kernel = np.full(effective, 1.0 / effective)
    return np.convolve(padded, kernel, mode="valid")


def valley_prominences(
    profile: np.ndarray, lo: int = 0, hi: int | None = None
) -> tuple[list[int], list[float]]:
    """低速谷值候选 + 各自的 prominence Φ（论文 §3.3）。

    候选：严格局部极小 `v[j] < v[j-1] and v[j] < v[j+1]`，j ∈ [lo, hi]。
    Φ 用标准地形 prominence（等价于 scipy.signal.peak_prominences 作用于 -v）：

        Φ(j) = min( 左鞍点-谷底之间最高速度, 右鞍点-谷底之间最高速度 ) - v[j]

    即「这个谷比它两侧的包围鞍点低多少」——谷在已经很慢的段上时 Φ 小，不会触发
    不必要的重规划（§3.3 对 Φ 的描述）。左右均以 [lo, hi] 为界，模拟部署时只看得
    到前 H_max 步的视角。

    返回 (候选索引列表, 对应 Φ 列表)，均按索引升序。
    """
    values = np.asarray(profile, dtype=np.float64)
    n = values.size
    if n < 3:
        return [], []
    span_lo = max(int(lo), 0)
    span_hi = n - 1 if hi is None else min(int(hi), n - 1)
    # 候选需左右邻居各一个，故取严格内部；包围范围仍用完整 [span_lo, span_hi]。
    lower = max(span_lo, 1)
    upper = min(span_hi, n - 2)
    if lower > upper:
        return [], []

    indices: list[int] = []
    prominences: list[float] = []
    for j in range(lower, upper + 1):
        current = float(values[j])
        if not (values[j] < values[j - 1] and values[j] < values[j + 1]):
            continue
        left = j - 1
        while left >= span_lo and values[left] >= current:
            left -= 1
        left_start = left + 1 if left >= span_lo else span_lo
        right = j + 1
        while right <= span_hi and values[right] >= current:
            right += 1
        right_end = right - 1 if right <= span_hi else span_hi
        left_base = float(values[left_start : j + 1].max())
        right_base = float(values[j : right_end + 1].max())
        indices.append(j)
        prominences.append(min(left_base, right_base) - current)
    return indices, prominences


@dataclass
class PaceDecision:
    """一次 g(A_i) 的完整决策记录（供日志与离线核对）。"""

    h: int
    delta_t: float | None
    task_name: str | None
    fallback: bool
    threshold_missing: bool
    chunk_length: int
    h_max: int
    candidates: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "h": int(self.h),
            "delta_t": self.delta_t,
            "task_name": self.task_name,
            "fallback": bool(self.fallback),
            "threshold_missing": bool(self.threshold_missing),
            "chunk_length": int(self.chunk_length),
            "h_max": int(self.h_max),
            "candidates": self.candidates,
        }


def select_execution_horizon(
    chunk: np.ndarray,
    cfg: PaceConfig,
    delta_t: float | None,
    *,
    task_name: str | None = None,
) -> PaceDecision:
    """论文 §3.3 的 g：从预测 chunk 选出执行视野 h。

    多臂候选取并集后接受 Φ ≥ δ_T 者，取最早的 h；无候选时回落 h_max。
    δ_T 为 None（该任务既无逐任务标定、也无全局兜底）时不做接受判定，直接回落
    h_max —— 即退化为当前固定 horizon 行为，而不是用一个没依据的阈值去猜。
    最后把 h 夹到 [h_min, h_max]（h_min 为本仓库的成本保护，见模块 docstring）。
    """
    array = np.asarray(chunk, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"pace expects a 2-D chunk [T,D], got shape {array.shape}")
    chunk_length = int(array.shape[0])
    h_max = min(int(cfg.h_max), chunk_length)
    if h_max < 1:
        raise ValueError(f"pace has nothing to execute: chunk_length={chunk_length}")

    candidates: list[dict[str, Any]] = []
    accepted: list[tuple[int, float, str]] = []
    for arm in cfg.arms:
        profile = arm_speed_profile(array[:h_max], ARM_XYZ_IDX[arm])
        smoothed = smooth_profile(profile, cfg.smooth_window, cfg.smooth)
        indices, prominences = valley_prominences(smoothed, lo=0, hi=h_max - 1)
        for index, prominence in zip(indices, prominences):
            h = index + 1  # v[j] 是 a[j]→a[j+1] 的那一步 → 重规划点在第 j+1 步
            is_accepted = delta_t is not None and prominence >= delta_t
            candidates.append(
                {
                    "arm": arm,
                    "h": int(h),
                    "prominence": float(prominence),
                    "accepted": bool(is_accepted),
                }
            )
            if is_accepted:
                accepted.append((h, float(prominence), arm))

    if accepted:
        # d_min：候选边界最小间隔（Table 3）。按 prominence 从强到弱贪心保留，
        # 同一簇里只留下最显著的那个，抑制「同一相位边界被相邻谷值重复命中」。
        accepted.sort(key=lambda item: (-item[1], item[0]))
        kept: list[tuple[int, float, str]] = []
        kept_h: set[int] = set()
        for h, prominence, arm in accepted:
            if any(abs(h - other) < cfg.d_min for other in kept_h):
                continue
            kept.append((h, prominence, arm))
            kept_h.add(h)
        h = min(entry[0] for entry in kept)
        fallback = False
    else:
        h = h_max
        fallback = True

    h = max(cfg.h_min, min(h, h_max))
    return PaceDecision(
        h=int(h),
        delta_t=delta_t,
        task_name=task_name,
        fallback=fallback,
        threshold_missing=delta_t is None,
        chunk_length=chunk_length,
        h_max=h_max,
        candidates=candidates,
    )


class PaceSelector:
    """执行层薄封装：按任务取 δ_T 并调用 `select_execution_horizon`。

    论文的 g 是确定性的、无记忆的函数，因此本类不持有跨请求的决策状态；唯一的
    可变状态是「缺阈值任务只告警一次」的集合，由 `reset()` 在 episode 边界清空。
    """

    def __init__(self, cfg: PaceConfig) -> None:
        self.cfg = cfg
        self._warned_missing_threshold: set[str] = set()

    def reset(self) -> None:
        self._warned_missing_threshold.clear()

    def select(self, chunk: np.ndarray, *, task_name: str | None = None) -> PaceDecision:
        delta_t = self.cfg.threshold_for(task_name)
        decision = select_execution_horizon(
            chunk, self.cfg, delta_t, task_name=task_name
        )
        if decision.threshold_missing:
            key = str(task_name)
            if key not in self._warned_missing_threshold:
                self._warned_missing_threshold.add(key)
                print(
                    "[x_vla][pace] no calibrated delta_t for "
                    f"task_name={task_name!r}; falling back to h_max="
                    f"{decision.h_max} (fixed-horizon behaviour). Run "
                    "tools/pace_calibrate.py and fill pace.threshold_delta_by_task.",
                    flush=True,
                )
        return decision
