"""Conservative runtime guard for repeated chunk-boundary rollback.

The guard is intentionally execution-only and opt-in.  It does not invent a
direction or waypoint.  After repeated evidence of a closed-loop limit cycle,
it translates one arm's predicted xyz trajectory so the fresh chunk starts at
the measured end-effector position, preserving every relative displacement in
the chunk.  Rotation and gripper predictions are untouched.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SelfLockGuardConfig:
    enabled: bool = False
    history_replans: int = 3
    trigger_count: int = 2
    stall_radius_m: float = 0.012
    min_start_jump_m: float = 0.002
    max_start_jump_m: float = 0.015
    min_chunk_motion_m: float = 0.003
    reverse_cosine: float = -0.8

    @classmethod
    def from_model_cfg(cls, model_cfg) -> "SelfLockGuardConfig":
        raw = model_cfg.get("self_lock_guard") or {}
        cfg = cls(
            enabled=bool(raw.get("enabled", False)),
            history_replans=int(raw.get("history_replans", 3)),
            trigger_count=int(raw.get("trigger_count", 2)),
            stall_radius_m=float(raw.get("stall_radius_m", 0.012)),
            min_start_jump_m=float(raw.get("min_start_jump_m", 0.002)),
            max_start_jump_m=float(raw.get("max_start_jump_m", 0.015)),
            min_chunk_motion_m=float(raw.get("min_chunk_motion_m", 0.003)),
            reverse_cosine=float(raw.get("reverse_cosine", -0.8)),
        )
        if cfg.history_replans < 2:
            raise ValueError("self_lock_guard.history_replans must be >= 2")
        if cfg.trigger_count < 1:
            raise ValueError("self_lock_guard.trigger_count must be >= 1")
        if not 0 < cfg.stall_radius_m:
            raise ValueError("self_lock_guard.stall_radius_m must be > 0")
        if not 0 < cfg.min_start_jump_m < cfg.max_start_jump_m:
            raise ValueError(
                "self_lock_guard requires 0 < min_start_jump_m < max_start_jump_m"
            )
        if not 0 < cfg.min_chunk_motion_m:
            raise ValueError("self_lock_guard.min_chunk_motion_m must be > 0")
        if not -1.0 <= cfg.reverse_cosine < 0.0:
            raise ValueError("self_lock_guard.reverse_cosine must be in [-1,0)")
        return cfg


class ChunkSelfLockGuard:
    """Per-environment detector and xyz re-anchoring guard."""

    # X-VLA 20-D action/proprio xyz slices, one per arm.
    _XYZ_SLICES = (slice(0, 3), slice(10, 13))

    def __init__(self, config: SelfLockGuardConfig):
        self.config = config
        self._position_history: list[np.ndarray] = []
        self._reverse_counts = np.zeros(2, dtype=np.int64)

    def reset(self) -> None:
        self._position_history.clear()
        self._reverse_counts[:] = 0

    def apply(
        self,
        current_proprio: np.ndarray,
        raw_chunk: np.ndarray,
        execute_steps: int,
    ) -> tuple[np.ndarray, dict]:
        current_proprio = np.asarray(current_proprio, dtype=np.float32)
        chunk = np.asarray(raw_chunk, dtype=np.float32)
        if current_proprio.shape != (20,):
            raise ValueError(
                f"self-lock guard expects proprio [20], got {current_proprio.shape}"
            )
        if chunk.ndim != 2 or chunk.shape[1] < 20:
            raise ValueError(
                f"self-lock guard expects action chunk [T,>=20], got {chunk.shape}"
            )
        if not 1 <= execute_steps <= chunk.shape[0]:
            raise ValueError(
                f"execute_steps must be in [1,{chunk.shape[0]}], got {execute_steps}"
            )

        current_xyz = np.stack(
            [current_proprio[index] for index in self._XYZ_SLICES], axis=0
        )
        self._position_history.append(current_xyz.copy())
        self._position_history = self._position_history[-self.config.history_replans :]

        history_ready = len(self._position_history) == self.config.history_replans
        if history_ready:
            history = np.stack(self._position_history, axis=0)
            # Per-arm maximum re-plan-to-re-plan excursion from the oldest
            # observation.  A forward/back cycle stays inside this radius.
            stall_span = np.max(
                np.linalg.norm(history - history[0:1], axis=-1), axis=0
            )
        else:
            stall_span = np.full(2, np.inf, dtype=np.float32)

        corrected = chunk.copy()
        arm_diagnostics = []
        corrected_arms = []
        last_executed_index = execute_steps - 1
        for arm, xyz_slice in enumerate(self._XYZ_SLICES):
            start_delta = chunk[0, xyz_slice] - current_xyz[arm]
            # Use the chunk's own intended motion after its first point.  This
            # catches the observed pattern "jump backwards, then crawl forwards"
            # without assuming a task-specific world direction.
            chunk_motion = (
                chunk[last_executed_index, xyz_slice] - chunk[0, xyz_slice]
            )
            start_norm = float(np.linalg.norm(start_delta))
            motion_norm = float(np.linalg.norm(chunk_motion))
            cosine = None
            if start_norm > 1e-12 and motion_norm > 1e-12:
                cosine = float(
                    np.dot(start_delta, chunk_motion) / (start_norm * motion_norm)
                )

            reverse_candidate = bool(
                history_ready
                and stall_span[arm] <= self.config.stall_radius_m
                and self.config.min_start_jump_m
                <= start_norm
                <= self.config.max_start_jump_m
                and motion_norm >= self.config.min_chunk_motion_m
                and cosine is not None
                and cosine <= self.config.reverse_cosine
            )
            if reverse_candidate:
                self._reverse_counts[arm] += 1
            else:
                self._reverse_counts[arm] = 0

            triggered = bool(
                reverse_candidate
                and self._reverse_counts[arm] >= self.config.trigger_count
            )
            shift = np.zeros(3, dtype=np.float32)
            if triggered:
                shift = current_xyz[arm] - chunk[0, xyz_slice]
                corrected[:, xyz_slice] += shift
                corrected_arms.append("left" if arm == 0 else "right")

            arm_diagnostics.append(
                {
                    "arm": "left" if arm == 0 else "right",
                    "history_ready": history_ready,
                    "stall_span_m": (
                        float(stall_span[arm]) if np.isfinite(stall_span[arm]) else None
                    ),
                    "start_jump_m": start_norm,
                    "chunk_motion_m": motion_norm,
                    "reverse_cosine": cosine,
                    "reverse_count": int(self._reverse_counts[arm]),
                    "candidate": reverse_candidate,
                    "triggered": triggered,
                    "xyz_shift_m": shift.tolist(),
                }
            )

        return corrected, {
            "enabled": True,
            "triggered": bool(corrected_arms),
            "corrected_arms": corrected_arms,
            "arms": arm_diagnostics,
        }
