"""夹爪迟滞（X-VLA 闭环执行改造方案 3.2 节 D4，执行层后处理）。

迟滞属于执行层：策略已输出动作，本模块只对最终返回的 16 维动作
（ee-dict list 的 left/right_ee_joint_state）的夹爪维做后处理，因此放在
6D→四元数转换之后、返回客户端之前，与 20 维中间表示（rotate6d）解耦。

无跨请求状态：每次调用由调用方传入 latch 初值（取自当前 obs 的真实夹爪位置
state[..._ee_joint_state][-1]），函数是纯后处理，不需要 reset / 会话生命周期。
死区保持 / 方向钳制在单次返回的 chunk 内链式生效；下一 chunk 自动与真实夹爪
状态重新对齐。

数据约定与 model.py 一致：0=闭合、1=张开，sigmoid 后连续值 [0,1]。

两种模式（deploy.yml `hysteresis.mode`）：
- binary          ：输出离散 0/1。p<lo → 0，p>hi → 1，死区 [lo,hi] 保持上一值。
- direction_latch ：连续迟滞（锁方向、幅值透传）。closing 态 out=min(pred,上步)、
  opening 态 out=max(pred,上步)，仅当 pred 越过死区边界（<lo 触发 closing、
  >hi 触发 opening）才允许方向翻转，死区内方向锁定、值单调、保留中间开合度。
  与 gripper_mode=continuous 直通天然匹配，不依赖 gripper_threshold。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class HysteresisConfig:
    """deploy.yml 的 `hysteresis` 配置段解析结果（仅配置，无状态）。"""

    enabled: bool = False
    lo: float = 0.3
    hi: float = 0.7
    mode: str = "direction_latch"  # "binary" | "direction_latch"

    def __post_init__(self) -> None:
        if self.mode not in ("binary", "direction_latch"):
            raise ValueError(
                f"hysteresis mode must be 'binary' or 'direction_latch', got {self.mode!r}"
            )
        if not 0.0 <= self.lo < self.hi <= 1.0:
            raise ValueError(
                f"hysteresis requires 0 <= lo < hi <= 1, got lo={self.lo}, hi={self.hi}"
            )

    @classmethod
    def from_model_cfg(cls, model_cfg: dict[str, Any]) -> "HysteresisConfig":
        raw = model_cfg.get("hysteresis") or {}
        if not isinstance(raw, dict):
            raise TypeError(
                f"hysteresis config must be a dict, got {type(raw).__name__}"
            )
        enabled_raw = raw.get("enabled", False)
        if isinstance(enabled_raw, str):
            enabled = enabled_raw.strip().lower() not in (
                "",
                "0",
                "false",
                "no",
                "off",
                "null",
                "none",
            )
        else:
            enabled = bool(enabled_raw)
        return cls(
            enabled=enabled,
            lo=float(raw.get("lo", 0.3)),
            hi=float(raw.get("hi", 0.7)),
            mode=str(raw.get("mode") or "direction_latch").strip().lower(),
        )


def apply_gripper_hysteresis(
    actions,
    *,
    left_init: float,
    right_init: float,
    lo: float = 0.3,
    hi: float = 0.7,
    mode: str = "direction_latch",
):
    """对 ee-dict 动作 list 的夹爪维做迟滞，就地修改并返回原 list。

    actions: action_chunk_to_ee_dict_list 的产物，每项 dict 含
      left_ee_joint_state / right_ee_joint_state（各为 1 元素数组）；
    left_init/right_init: 当前 obs 的真实夹爪位置（state[..._ee_joint_state][-1]），
      每次调用由调用方提供，本函数不跨请求保留任何状态。
    """
    if mode not in ("binary", "direction_latch"):
        raise ValueError(
            f"hysteresis mode must be 'binary' or 'direction_latch', got {mode!r}"
        )
    left_init = _validate_init(left_init, "left_init")
    right_init = _validate_init(right_init, "right_init")

    if mode == "binary":
        left_latch = 1.0 if left_init >= 0.5 else 0.0
        right_latch = 1.0 if right_init >= 0.5 else 0.0
        for action in actions:
            left_latch = _binary_step(
                float(action["left_ee_joint_state"][0]), left_latch, lo, hi
            )
            action["left_ee_joint_state"][0] = left_latch
            right_latch = _binary_step(
                float(action["right_ee_joint_state"][0]), right_latch, lo, hi
            )
            action["right_ee_joint_state"][0] = right_latch
    else:  # mode == "direction_latch"
        left_latch, left_dir = _init_direction(left_init)
        right_latch, right_dir = _init_direction(right_init)
        for action in actions:
            left_latch, left_dir = _direction_step(
                float(action["left_ee_joint_state"][0]), left_latch, left_dir, lo, hi
            )
            action["left_ee_joint_state"][0] = left_latch
            right_latch, right_dir = _direction_step(
                float(action["right_ee_joint_state"][0]), right_latch, right_dir, lo, hi
            )
            action["right_ee_joint_state"][0] = right_latch
    return actions


def _validate_init(value: float, name: str) -> float:
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value}")
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be within [0,1], got {value}")
    return value


def _binary_step(pred: float, latch: float, lo: float, hi: float) -> float:
    if pred < lo:
        return 0.0
    if pred > hi:
        return 1.0
    return latch  # 死区 [lo, hi]：保持上一值


def _init_direction(init: float):
    v = float(np.clip(init, 0.0, 1.0))
    return v, ("opening" if v >= 0.5 else "closing")


def _direction_step(pred, latch, direction, lo, hi):
    if direction == "opening":
        if pred < lo:
            return pred, "closing"  # 强闭合信号越过死区下界 → 翻转方向
        return max(pred, latch), "opening"
    if pred > hi:
        return pred, "opening"  # 强张开信号越过死区上界 → 翻转方向
    return min(pred, latch), "closing"
