"""Quaternion helpers and deterministic EEF target interpolation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

def normalize_quat_wxyz(value: Any) -> np.ndarray:
    quat = np.asarray(value, dtype=np.float64).reshape(-1)
    if quat.shape != (4,) or not np.isfinite(quat).all():
        raise ValueError("quaternion must be four finite wxyz values")
    norm = float(np.linalg.norm(quat))
    if norm < 1e-6:
        raise ValueError("quaternion norm is degenerate")
    return quat / norm


def quat_angle_between(a: Any, b: Any) -> float:
    dot = abs(float(np.dot(normalize_quat_wxyz(a), normalize_quat_wxyz(b))))
    return 2.0 * math.acos(float(np.clip(dot, 0.0, 1.0)))


def slerp(a: Any, b: Any, t: float) -> np.ndarray:
    q0, q1 = normalize_quat_wxyz(a), normalize_quat_wxyz(b)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1, dot = -q1, -dot
    if dot > 0.9995:
        return normalize_quat_wxyz(q0 + float(t) * (q1 - q0))
    theta = math.acos(float(np.clip(dot, -1.0, 1.0)))
    sine = math.sin(theta)
    return normalize_quat_wxyz(
        math.sin((1.0 - float(t)) * theta) / sine * q0
        + math.sin(float(t) * theta) / sine * q1
    )


@dataclass(frozen=True)
class ArmState:
    pos: np.ndarray
    quat: np.ndarray
    gripper: float

    @classmethod
    def from_pose7(cls, pose: Sequence[float], gripper: Any) -> "ArmState":
        values = np.asarray(pose, dtype=np.float64).reshape(-1)
        if values.shape != (7,) or not np.isfinite(values).all():
            raise ValueError("EEF pose must be seven finite xyz+wxyz values")
        grip = float(gripper)
        if not math.isfinite(grip):
            raise ValueError("gripper must be finite")
        return cls(values[:3], normalize_quat_wxyz(values[3:]), grip)


@dataclass(frozen=True)
class ArmCommand:
    position: np.ndarray | None = None
    quat: np.ndarray | None = None
    gripper: float | None = None

    @property
    def is_keep(self) -> bool:
        return self.position is None and self.quat is None and self.gripper is None


KEEP = ArmCommand()


@dataclass(frozen=True)
class MotionConfig:
    delta_p_max_m: float = 0.005
    delta_theta_max_rad: float = 0.035
    max_target_translation_m: float = 0.05
    max_target_rotation_rad: float = 0.35
    settle_steps: int = 3
    gripper_open: float = 1.0
    gripper_close: float = 0.0

    def __post_init__(self) -> None:
        if self.delta_p_max_m <= 0 or self.delta_theta_max_rad <= 0:
            raise ValueError("interpolation increments must be positive")
        if self.max_target_translation_m <= 0 or self.max_target_rotation_rad <= 0:
            raise ValueError("per-decision target limits must be positive")
        if self.settle_steps < 0:
            raise ValueError("settle_steps must be non-negative")


@dataclass(frozen=True)
class ArmPlan:
    start_pos: np.ndarray
    start_quat: np.ndarray
    target_pos: np.ndarray
    target_quat: np.ndarray
    start_gripper: float
    target_gripper: float
    steps: int


def plan_arm(state: ArmState, command: ArmCommand, config: MotionConfig) -> ArmPlan:
    target_pos = state.pos if command.position is None else np.asarray(
        command.position, dtype=np.float64
    ).reshape(3)
    if not np.isfinite(target_pos).all():
        raise ValueError("target position must contain three finite values")
    target_quat = state.quat if command.quat is None else normalize_quat_wxyz(command.quat)
    target_gripper = state.gripper if command.gripper is None else float(command.gripper)
    if not math.isfinite(target_gripper):
        raise ValueError("target gripper must be finite")
    position_steps = math.ceil(
        float(np.linalg.norm(target_pos - state.pos)) / config.delta_p_max_m
    )
    rotation_steps = math.ceil(
        quat_angle_between(target_quat, state.quat) / config.delta_theta_max_rad
    )
    return ArmPlan(
        state.pos,
        state.quat,
        target_pos,
        target_quat,
        state.gripper,
        target_gripper,
        max(1, position_steps, rotation_steps),
    )


def target_error(state: ArmState, command: ArmCommand) -> tuple[float, float]:
    translation = 0.0 if command.position is None else float(
        np.linalg.norm(np.asarray(command.position, dtype=np.float64) - state.pos)
    )
    rotation = 0.0 if command.quat is None else quat_angle_between(state.quat, command.quat)
    return translation, rotation


def _action(left: ArmPlan, right: ArmPlan, t: float) -> dict[str, np.ndarray]:
    return {
        "left_ee_pose": np.concatenate([
            left.start_pos + (left.target_pos - left.start_pos) * t,
            slerp(left.start_quat, left.target_quat, t),
        ]).astype(np.float32),
        "left_ee_joint_state": np.asarray([left.start_gripper], dtype=np.float32),
        "right_ee_pose": np.concatenate([
            right.start_pos + (right.target_pos - right.start_pos) * t,
            slerp(right.start_quat, right.target_quat, t),
        ]).astype(np.float32),
        "right_ee_joint_state": np.asarray([right.start_gripper], dtype=np.float32),
    }


def hold_chunk(left: ArmState, right: ArmState) -> list[dict[str, np.ndarray]]:
    """Return one unchanged action frame; executing it still costs one sim step."""
    left_plan = plan_arm(left, KEEP, MotionConfig())
    right_plan = plan_arm(right, KEEP, MotionConfig())
    return [_action(left_plan, right_plan, 1.0)]


def interpolate_chunk(
    left: ArmState,
    left_command: ArmCommand,
    right: ArmState,
    right_command: ArmCommand,
    config: MotionConfig,
) -> list[dict[str, np.ndarray]]:
    """Interpolate completely to the model-provided targets with LERP and SLERP."""
    left_plan = plan_arm(left, left_command, config)
    right_plan = plan_arm(right, right_command, config)
    steps = max(left_plan.steps, right_plan.steps)
    chunk = [_action(left_plan, right_plan, index / steps) for index in range(1, steps + 1)]

    last = chunk[-1]
    last["left_ee_joint_state"] = np.asarray([left_plan.target_gripper], dtype=np.float32)
    last["right_ee_joint_state"] = np.asarray([right_plan.target_gripper], dtype=np.float32)
    for _ in range(max(0, int(config.settle_steps))):
        chunk.append({key: value.copy() for key, value in last.items()})

    return chunk
