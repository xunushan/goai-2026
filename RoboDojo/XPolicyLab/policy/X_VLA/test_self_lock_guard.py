"""Standalone tests for the closed-loop chunk rollback guard."""

import numpy as np

from self_lock_guard import ChunkSelfLockGuard, SelfLockGuardConfig


def _proprio(left_x=0.0, right_x=0.0):
    value = np.zeros(20, dtype=np.float32)
    value[0] = left_x
    value[10] = right_x
    return value


def _rollback_chunk(start=-0.006, end=0.001):
    chunk = np.zeros((30, 20), dtype=np.float32)
    chunk[:, 0] = np.linspace(start, end, 30)
    return chunk


def _guard(**overrides):
    values = {
        "enabled": True,
        "history_replans": 3,
        "trigger_count": 2,
        "stall_radius_m": 0.012,
        "min_start_jump_m": 0.002,
        "max_start_jump_m": 0.015,
        "min_chunk_motion_m": 0.003,
        "reverse_cosine": -0.8,
    }
    values.update(overrides)
    return ChunkSelfLockGuard(SelfLockGuardConfig(**values))


def test_repeated_stalled_reverse_boundary_is_reanchored():
    guard = _guard()
    chunk = _rollback_chunk()
    outputs = []
    diagnostics = []
    for _ in range(4):
        output, diag = guard.apply(_proprio(), chunk, execute_steps=15)
        outputs.append(output)
        diagnostics.append(diag)

    assert not diagnostics[0]["triggered"]
    assert not diagnostics[1]["triggered"]
    assert not diagnostics[2]["triggered"]  # history ready, reverse count 1
    assert diagnostics[3]["triggered"]
    assert diagnostics[3]["corrected_arms"] == ["left"]
    assert np.isclose(outputs[3][0, 0], 0.0)
    # Re-anchoring is a rigid translation: relative xyz trajectory is intact.
    assert np.allclose(np.diff(outputs[3][:, 0]), np.diff(chunk[:, 0]))


def test_normal_progress_does_not_trigger():
    guard = _guard()
    chunk = _rollback_chunk()
    for x in (0.0, 0.02, 0.04, 0.06, 0.08):
        output, diag = guard.apply(_proprio(left_x=x), chunk + _proprio(left_x=x), 15)
        assert not diag["triggered"]
        assert np.allclose(output, chunk + _proprio(left_x=x))


def test_forward_start_or_large_jump_does_not_trigger():
    for chunk in (
        _rollback_chunk(start=0.003, end=0.010),
        _rollback_chunk(start=-0.020, end=0.001),
    ):
        guard = _guard(trigger_count=1)
        for _ in range(3):
            output, diag = guard.apply(_proprio(), chunk, 15)
        assert not diag["triggered"]
        assert np.allclose(output, chunk)


def test_reset_clears_detection_history():
    guard = _guard(trigger_count=1)
    chunk = _rollback_chunk()
    for _ in range(3):
        _, diag = guard.apply(_proprio(), chunk, 15)
    assert diag["triggered"]
    guard.reset()
    _, diag = guard.apply(_proprio(), chunk, 15)
    assert not diag["triggered"]
