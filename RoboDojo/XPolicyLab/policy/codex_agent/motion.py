"""Deterministic motion layer for ``codex_agent``.

Pure numpy, no torch / scipy / network / filesystem: everything here is a
function of its arguments so it can be unit-tested in isolation and reasoned
about without the simulator.

Three responsibilities, mirroring the design doc:

1. **Pose algebra** — quaternion (``wxyz``) <-> rotation matrix <-> yaw/pitch/roll.
   ``rpy`` is always *relative to the start (HOME) orientation*, because an LLM
   produces a three-angle relative description far more reliably than a
   quaternion. The absolute quaternion is reconstructed here, never by the model.
2. **Motion interpolator** — LERP on position, SLERP on rotation, with per-step
   limits that turn a target pose into a bounded ActionChunk.
3. **Guardrail** — reject / clamp targets that leave the valid workspace.

Quaternion convention is ``[qw, qx, qy, qz]`` everywhere, matching the RoboDojo
16-dim ee protocol.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

# --------------------------------------------------------------------------- #
# pose algebra
# --------------------------------------------------------------------------- #

_EPS = 1e-9


def normalize_quat_wxyz(quat: Any) -> np.ndarray:
    """Return a unit ``[qw,qx,qy,qz]`` float64 array; raise on a degenerate norm."""
    q = np.asarray(quat, dtype=np.float64).reshape(-1)
    if q.shape != (4,):
        raise ValueError(f"quaternion must have 4 components, got shape {q.shape}")
    if not np.isfinite(q).all():
        raise ValueError("quaternion contains NaN or Inf")
    norm = float(np.linalg.norm(q))
    if norm < 1e-6:
        raise ValueError(f"quaternion norm is degenerate: {norm:.3e}")
    return q / norm


def quat_wxyz_to_matrix(quat: Any) -> np.ndarray:
    w, x, y, z = normalize_quat_wxyz(quat)
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
            [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
            [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def matrix_to_quat_wxyz(matrix: Any) -> np.ndarray:
    """Rotation matrix -> ``[qw,qx,qy,qz]`` (Shepperd's branch method)."""
    m = np.asarray(matrix, dtype=np.float64)
    if m.shape != (3, 3):
        raise ValueError(f"rotation matrix must be 3x3, got shape {m.shape}")
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    return normalize_quat_wxyz([w, x, y, z])


def rpy_to_quat_wxyz(rpy: Any) -> np.ndarray:
    """``[roll, pitch, yaw]`` (extrinsic x/y/z, i.e. ``Rz*Ry*Rx``) -> quaternion."""
    rpy = np.asarray(rpy, dtype=np.float64).reshape(-1)
    if rpy.shape != (3,):
        raise ValueError(f"rpy must have 3 components, got shape {rpy.shape}")
    if not np.isfinite(rpy).all():
        raise ValueError("rpy contains NaN or Inf")
    roll, pitch, yaw = (float(v) for v in rpy)
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rot = np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )
    return matrix_to_quat_wxyz(rot)


def quat_wxyz_to_rpy(quat: Any) -> np.ndarray:
    """Quaternion -> ``[roll, pitch, yaw]`` (extrinsic x/y/z)."""
    m = quat_wxyz_to_matrix(quat)
    pitch = math.asin(float(np.clip(-m[2, 0], -1.0, 1.0)))
    if abs(math.cos(pitch)) > 1e-6:
        roll = math.atan2(m[2, 1], m[2, 2])
        yaw = math.atan2(m[1, 0], m[0, 0])
    else:  # gimbal lock: roll and yaw are degenerate, pin roll to 0
        roll = 0.0
        yaw = math.atan2(-m[0, 1], m[1, 1])
    return np.array([roll, pitch, yaw], dtype=np.float64)


def relative_rpy(quat: Any, home_quat: Any) -> np.ndarray:
    """Orientation of ``quat`` expressed in the HOME frame, as ``[r,p,y]``."""
    rel = quat_wxyz_to_matrix(home_quat).T @ quat_wxyz_to_matrix(quat)
    return quat_wxyz_to_rpy(matrix_to_quat_wxyz(rel))


def absolute_quat_from_relative_rpy(rpy: Any, home_quat: Any) -> np.ndarray:
    """Inverse of :func:`relative_rpy`.

    ``rpy`` is interpreted in the HOME frame, so the absolute rotation is
    ``R_home @ R_rpy``.
    """
    rot = quat_wxyz_to_matrix(home_quat) @ quat_wxyz_to_matrix(rpy_to_quat_wxyz(rpy))
    return matrix_to_quat_wxyz(rot)


def quat_angle_between(a: Any, b: Any) -> float:
    """Shortest-path angle (radians) between two orientations."""
    qa = normalize_quat_wxyz(a)
    qb = normalize_quat_wxyz(b)
    dot = float(np.clip(abs(float(np.dot(qa, qb))), 0.0, 1.0))
    return 2.0 * math.acos(dot)


def slerp(q0: Any, q1: Any, t: float) -> np.ndarray:
    """Spherical interpolation along the *shortest* path.

    Never linearly interpolate quaternion components: it leaves the unit sphere
    and produces non-uniform angular speed.
    """
    a = normalize_quat_wxyz(q0)
    b = normalize_quat_wxyz(q1)
    dot = float(np.dot(a, b))
    if dot < 0.0:  # q and -q are the same rotation; take the short way round
        b = -b
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:  # nearly parallel: fall back to a normalized LERP
        out = a + (b - a) * float(t)
        return normalize_quat_wxyz(out)
    theta = math.acos(dot)
    sin_theta = math.sin(theta)
    w0 = math.sin((1.0 - float(t)) * theta) / sin_theta
    w1 = math.sin(float(t) * theta) / sin_theta
    return normalize_quat_wxyz(w0 * a + w1 * b)


# --------------------------------------------------------------------------- #
# commands / state
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ArmState:
    """An arm as actually observed in the simulator."""

    pos: np.ndarray
    quat: np.ndarray  # wxyz, already normalized
    gripper: float

    @classmethod
    def from_pose7(cls, pose: Sequence[float], gripper: Any) -> "ArmState":
        pose = np.asarray(pose, dtype=np.float64).reshape(-1)
        if pose.shape != (7,):
            raise ValueError(f"ee pose must have 7 components, got shape {pose.shape}")
        return cls(
            pos=pose[:3].astype(np.float64),
            quat=normalize_quat_wxyz(pose[3:7]),
            gripper=float(np.clip(float(gripper), 0.0, 1.0)),
        )


@dataclass(frozen=True)
class ArmCommand:
    """What the model asked one arm to do. ``None`` means *keep*."""

    position: np.ndarray | None = None
    quat: np.ndarray | None = None
    gripper: float | None = None

    @property
    def is_keep(self) -> bool:
        return self.position is None and self.quat is None and self.gripper is None


KEEP = ArmCommand()


@dataclass(frozen=True)
class MotionConfig:
    delta_p_max_m: float = 0.015
    delta_theta_max_deg: float = 5.0
    settle_steps: int = 3
    gripper_open: float = 1.0
    gripper_close: float = 0.0

    @property
    def delta_theta_max_rad(self) -> float:
        return math.radians(float(self.delta_theta_max_deg))


@dataclass(frozen=True)
class Box:
    """Axis-aligned workspace box for one arm."""

    x: tuple[float, float]
    y: tuple[float, float]
    z: tuple[float, float]

    @classmethod
    def from_cfg(cls, cfg: Any) -> "Box":
        if isinstance(cfg, Box):
            return cfg
        if not isinstance(cfg, dict) or not {"x", "y", "z"} <= set(cfg):
            raise ValueError(
                "workspace box must be {x: [lo, hi], y: [lo, hi], z: [lo, hi]}, "
                f"got {cfg!r}"
            )
        return cls(
            x=(float(cfg["x"][0]), float(cfg["x"][1])),
            y=(float(cfg["y"][0]), float(cfg["y"][1])),
            z=(float(cfg["z"][0]), float(cfg["z"][1])),
        )

    @property
    def lo(self) -> np.ndarray:
        return np.array([self.x[0], self.y[0], self.z[0]], dtype=np.float64)

    @property
    def hi(self) -> np.ndarray:
        return np.array([self.x[1], self.y[1], self.z[1]], dtype=np.float64)

    def excess(self, point: Any) -> float:
        """Largest distance by which ``point`` lies outside the box (0 if inside)."""
        p = np.asarray(point, dtype=np.float64).reshape(3)
        return float(np.max(np.maximum(self.lo - p, p - self.hi)))

    def clamp(self, point: Any) -> np.ndarray:
        p = np.asarray(point, dtype=np.float64).reshape(3)
        return np.clip(p, self.lo, self.hi)


# --------------------------------------------------------------------------- #
# guardrail
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class GuardrailConfig:
    workspace: Box
    reject_margin_m: float = 0.03
    max_target_distance_m: float = 0.45
    max_target_rotation_deg: float = 170.0


@dataclass
class GuardrailOutcome:
    ok: bool
    command: ArmCommand
    clamped: bool = False
    reason: str = ""

    def feedback(self, arm: str) -> str:
        if self.ok and not self.clamped:
            return ""
        if self.clamped:
            return (
                f"The {arm} target was outside the valid workspace and was clamped "
                f"onto its boundary before execution."
            )
        return f"The {arm} target was rejected and NOT executed: {self.reason}"


def check_arm_command(
    command: ArmCommand,
    now: ArmState,
    cfg: GuardrailConfig,
) -> GuardrailOutcome:
    """Validate one arm's target. Illegal targets are reported, never silently fixed.

    Clamping only happens for a *small* excursion past the workspace boundary
    (``reject_margin_m``); anything further is rejected outright so the model has
    to replan rather than being quietly teleported somewhere it did not choose.
    """
    position = command.position
    quat = command.quat
    clamped = False

    if position is not None:
        position = np.asarray(position, dtype=np.float64).reshape(-1)
        if position.shape != (3,) or not np.isfinite(position).all():
            return GuardrailOutcome(False, KEEP, reason="position must be 3 finite numbers")
        distance = float(np.linalg.norm(position - now.pos))
        if distance > cfg.max_target_distance_m + _EPS:
            return GuardrailOutcome(
                False,
                KEEP,
                reason=(
                    f"the jump of {distance:.3f} m from the current position exceeds the "
                    f"per-decision limit of {cfg.max_target_distance_m:.3f} m; "
                    "split it into several smaller motions"
                ),
            )
        excess = cfg.workspace.excess(position)
        if excess > _EPS:
            if excess <= cfg.reject_margin_m + _EPS:
                position = cfg.workspace.clamp(position)
                clamped = True
            else:
                return GuardrailOutcome(
                    False,
                    KEEP,
                    reason=(
                        f"the target lies {excess:.3f} m outside the valid workspace "
                        f"x{list(cfg.workspace.x)} y{list(cfg.workspace.y)} "
                        f"z{list(cfg.workspace.z)}"
                    ),
                )

    if quat is not None:
        try:
            quat = normalize_quat_wxyz(quat)
        except ValueError as exc:
            return GuardrailOutcome(False, KEEP, reason=f"invalid orientation: {exc}")
        delta = quat_angle_between(now.quat, quat)
        if math.degrees(delta) > cfg.max_target_rotation_deg + _EPS:
            return GuardrailOutcome(
                False,
                KEEP,
                reason=(
                    f"the rotation of {math.degrees(delta):.1f} deg from the current "
                    "orientation is too large for a single decision"
                ),
            )

    return GuardrailOutcome(True, ArmCommand(position, quat, command.gripper), clamped=clamped)


# --------------------------------------------------------------------------- #
# interpolator
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ArmPlan:
    p0: np.ndarray
    q0: np.ndarray
    p1: np.ndarray
    q1: np.ndarray
    n_raw: int
    gripper_start: float
    gripper_end: float

    @property
    def moved(self) -> bool:
        return (
            float(np.linalg.norm(self.p1 - self.p0)) > _EPS
            or quat_angle_between(self.q0, self.q1) > _EPS
            or abs(self.gripper_end - self.gripper_start) > _EPS
        )


def plan_arm(now: ArmState, command: ArmCommand, cfg: MotionConfig) -> ArmPlan:
    """Turn a target command into per-arm interpolation parameters."""
    p0 = np.asarray(now.pos, dtype=np.float64).reshape(3)
    q0 = normalize_quat_wxyz(now.quat)
    p1 = p0 if command.position is None else np.asarray(command.position, dtype=np.float64).reshape(3)
    q1 = q0 if command.quat is None else normalize_quat_wxyz(command.quat)

    distance = float(np.linalg.norm(p1 - p0))
    theta = quat_angle_between(q0, q1)

    n_pos = math.ceil(distance / cfg.delta_p_max_m - 1e-9) if cfg.delta_p_max_m > 0 else 0
    n_rot = math.ceil(theta / cfg.delta_theta_max_rad - 1e-9) if cfg.delta_theta_max_rad > 0 else 0
    n_raw = max(1, n_pos, n_rot)

    gripper_start = float(now.gripper)
    gripper_end = gripper_start if command.gripper is None else float(command.gripper)
    return ArmPlan(p0, q0, p1, q1, n_raw, gripper_start, gripper_end)


def _step_action(
    left_plan: ArmPlan,
    right_plan: ArmPlan,
    t: float,
) -> dict[str, np.ndarray]:
    """One 16-dim ee action at interpolation parameter ``t`` (gripper = start value)."""
    left_pos = left_plan.p0 + (left_plan.p1 - left_plan.p0) * float(t)
    right_pos = right_plan.p0 + (right_plan.p1 - right_plan.p0) * float(t)
    return {
        "left_ee_pose": np.concatenate([left_pos, slerp(left_plan.q0, left_plan.q1, t)]).astype(
            np.float32
        ),
        "left_ee_joint_state": np.asarray([left_plan.gripper_start], dtype=np.float32),
        "right_ee_pose": np.concatenate(
            [right_pos, slerp(right_plan.q0, right_plan.q1, t)]
        ).astype(np.float32),
        "right_ee_joint_state": np.asarray([right_plan.gripper_start], dtype=np.float32),
    }


@dataclass
class ChunkInfo:
    """Bookkeeping about how a chunk was produced (for logging / feedback)."""

    length: int = 0
    settle_steps: int = 0
    n_raw_left: int = 1
    n_raw_right: int = 1
    truncated: bool = False
    gripper_deferred: bool = False
    moved_left: bool = False
    moved_right: bool = False
    truncated_arms: list[str] = field(default_factory=list)


def hold_chunk(
    left: ArmState,
    right: ArmState,
    steps: int,
) -> list[dict[str, np.ndarray]]:
    """``steps`` identical actions that keep both arms exactly where they are."""
    steps = max(1, int(steps))
    action = {
        "left_ee_pose": np.concatenate([left.pos, left.quat]).astype(np.float32),
        "left_ee_joint_state": np.asarray([left.gripper], dtype=np.float32),
        "right_ee_pose": np.concatenate([right.pos, right.quat]).astype(np.float32),
        "right_ee_joint_state": np.asarray([right.gripper], dtype=np.float32),
    }
    # Copy the buffers rather than the dict: a consumer that writes into one
    # step's pose would otherwise rewrite every step of the hold.
    return [{key: value.copy() for key, value in action.items()} for _ in range(steps)]


def interpolate_chunk(
    left: ArmState,
    left_command: ArmCommand,
    right: ArmState,
    right_command: ArmCommand,
    cap: int,
    cfg: MotionConfig,
) -> tuple[list[dict[str, np.ndarray]], ChunkInfo]:
    """Build a bounded ActionChunk from the observed pose to the commanded target.

    Both arms interpolate in parallel over ``n_raw = max(n_left, n_right)`` steps
    so the slower arm sets the pace. ``cap`` is the maximum total chunk length
    including the settle steps appended at the end.

    When the chunk is truncated by ``cap`` the interpolation parameter stays
    ``t = i / n_raw`` (not ``i / n``) so the chunk stops *short* of the target and
    the next decision replans from the newly observed pose. Using ``i / n`` would
    teleport the arm onto the target within ``n`` steps and cause a velocity jump.
    """
    cap = max(1, int(cap))
    settle = int(max(0, min(int(cfg.settle_steps), cap - 1)))
    n_avail = cap - settle

    left_plan = plan_arm(left, left_command, cfg)
    right_plan = plan_arm(right, right_command, cfg)
    n_raw = max(left_plan.n_raw, right_plan.n_raw)
    n = max(1, min(n_raw, n_avail))

    info = ChunkInfo(
        settle_steps=settle,
        n_raw_left=left_plan.n_raw,
        n_raw_right=right_plan.n_raw,
        moved_left=left_plan.moved,
        moved_right=right_plan.moved,
    )

    # Both arms share the global parameter t = i / n_raw, so truncation stops
    # every moving arm short of its target.
    truncated = n < n_raw
    info.truncated = truncated
    if truncated:
        if left_plan.moved:
            info.truncated_arms.append("left")
        if right_plan.moved:
            info.truncated_arms.append("right")

    # A gripper change only takes effect once the arm has actually arrived; if the
    # chunk was truncated the gripper is deferred to the next decision instead of
    # closing/opening mid-flight.
    apply_gripper = not truncated
    info.gripper_deferred = (not apply_gripper) and (
        abs(left_plan.gripper_end - left_plan.gripper_start) > _EPS
        or abs(right_plan.gripper_end - right_plan.gripper_start) > _EPS
    )

    chunk: list[dict[str, np.ndarray]] = []
    for i in range(1, n + 1):
        chunk.append(_step_action(left_plan, right_plan, i / float(n_raw)))
    last = chunk[-1]

    if apply_gripper:
        if abs(left_plan.gripper_end - left_plan.gripper_start) > _EPS:
            last["left_ee_joint_state"] = np.asarray(
                [left_plan.gripper_end], dtype=np.float32
            )
        if abs(right_plan.gripper_end - right_plan.gripper_start) > _EPS:
            last["right_ee_joint_state"] = np.asarray(
                [right_plan.gripper_end], dtype=np.float32
            )

    for _ in range(settle):
        chunk.append({k: v.copy() for k, v in last.items()})

    info.length = len(chunk)
    assert 1 <= len(chunk) <= cap, f"chunk length {len(chunk)} outside [1,{cap}]"
    return chunk, info
