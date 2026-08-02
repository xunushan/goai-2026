import numpy as np
import pytest

from utils.xvla_ee import (
    ee16_to_xvla20,
    quaternion_wxyz_to_rotation6d,
    rotation6d_to_quaternion_wxyz,
    xvla20_to_ee16,
)


def _random_quaternions(count: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    quaternion = rng.normal(size=(count, 4)).astype(np.float32)
    return quaternion / np.linalg.norm(quaternion, axis=-1, keepdims=True)


def test_rotation6d_uses_xvla_interleaved_layout() -> None:
    identity = np.array([1, 0, 0, 0], dtype=np.float32)
    np.testing.assert_allclose(
        quaternion_wxyz_to_rotation6d(identity),
        np.array([1, 0, 0, 1, 0, 0], dtype=np.float32),
    )


def test_quaternion_rotation6d_batch_round_trip() -> None:
    source = _random_quaternions(128)
    restored = rotation6d_to_quaternion_wxyz(quaternion_wxyz_to_rotation6d(source))
    similarity = np.abs(np.sum(source * restored, axis=-1))
    np.testing.assert_allclose(similarity, np.ones_like(similarity), atol=1e-5)
    assert np.all(restored[:, 0] >= 0)


def test_ee_round_trip_and_gripper_direction() -> None:
    source = np.zeros((3, 16), dtype=np.float32)
    source[:, 3:7] = _random_quaternions(3, seed=1)
    source[:, 11:15] = _random_quaternions(3, seed=2)
    source[:, 0:3] = np.arange(9, dtype=np.float32).reshape(3, 3)
    source[:, 8:11] = -source[:, 0:3]
    source[:, 7] = [0.0, 0.25, 1.0]
    source[:, 15] = [1.0, 0.75, 0.0]

    converted = ee16_to_xvla20(source)
    restored = xvla20_to_ee16(converted)

    assert converted.shape == (3, 20)
    np.testing.assert_allclose(converted[:, 9], 1.0 - source[:, 7])
    np.testing.assert_allclose(converted[:, 19], 1.0 - source[:, 15])
    np.testing.assert_allclose(
        restored[:, [0, 1, 2, 7, 8, 9, 10, 15]], source[:, [0, 1, 2, 7, 8, 9, 10, 15]]
    )
    for start in (3, 11):
        similarity = np.abs(
            np.sum(
                restored[:, start : start + 4] * source[:, start : start + 4], axis=-1
            )
        )
        np.testing.assert_allclose(similarity, np.ones_like(similarity), atol=1e-5)


def test_rejects_invalid_rotations() -> None:
    with pytest.raises(ValueError, match="zero-norm"):
        quaternion_wxyz_to_rotation6d(np.zeros(4, dtype=np.float32))
    with pytest.raises(ValueError, match="degenerate first axis"):
        rotation6d_to_quaternion_wxyz(np.zeros(6, dtype=np.float32))
    with pytest.raises(ValueError, match="collinear axes"):
        rotation6d_to_quaternion_wxyz(np.array([1, 2, 0, 0, 0, 0], dtype=np.float32))
