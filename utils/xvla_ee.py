"""ARX-X5 end-effector conversions at the X-VLA integration boundary.

The RoboDojo representation is 16D::

    [left_xyz(3), left_quat_wxyz(4), left_gripper(1),
     right_xyz(3), right_quat_wxyz(4), right_gripper(1)]

The X-VLA representation is 20D::

    [left_xyz(3), left_rotation6d(6), left_gripper(1),
     right_xyz(3), right_rotation6d(6), right_gripper(1)]

X-VLA currently uses identity action normalization. If another action
preprocessor is configured, its output must first be denormalized back to the
physical 20D representation before calling :func:`xvla20_to_ee16`.
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-8


def _as_float_array(value: np.ndarray, expected_dim: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 0 or array.shape[-1] != expected_dim:
        raise ValueError(
            f"{name} must have last dimension {expected_dim}, got shape {array.shape}"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or infinite values")
    return array


def quaternion_wxyz_to_rotation6d(quaternion: np.ndarray) -> np.ndarray:
    """Convert ``(..., 4)`` WXYZ quaternions to X-VLA's interleaved rotation-6D.

    X-VLA flattens the first two matrix columns in row-major order, producing
    ``[r00, r01, r10, r11, r20, r21]``.
    """

    quaternion = _as_float_array(quaternion, 4, "quaternion")
    norm = np.linalg.norm(quaternion, axis=-1, keepdims=True)
    if np.any(norm < _EPS):
        raise ValueError("quaternion contains a zero-norm value")
    w, x, y, z = np.moveaxis(quaternion / norm, -1, 0)
    matrix = np.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ),
        axis=-1,
    ).reshape(*quaternion.shape[:-1], 3, 3)
    return matrix[..., :, :2].reshape(*quaternion.shape[:-1], 6).astype(np.float32)


def _rotation6d_to_matrix(rotation6d: np.ndarray) -> np.ndarray:
    rotation6d = _as_float_array(rotation6d, 6, "rotation6d")
    first = rotation6d[..., 0:5:2]
    second = rotation6d[..., 1:6:2]

    first_norm = np.linalg.norm(first, axis=-1, keepdims=True)
    if np.any(first_norm < _EPS):
        raise ValueError("rotation6d contains a degenerate first axis")
    axis1 = first / first_norm

    orthogonal = second - np.sum(axis1 * second, axis=-1, keepdims=True) * axis1
    second_norm = np.linalg.norm(orthogonal, axis=-1, keepdims=True)
    if np.any(second_norm < _EPS):
        raise ValueError("rotation6d contains collinear axes")
    axis2 = orthogonal / second_norm
    axis3 = np.cross(axis1, axis2)
    return np.stack((axis1, axis2, axis3), axis=-1)


def rotation6d_to_quaternion_wxyz(rotation6d: np.ndarray) -> np.ndarray:
    """Convert X-VLA rotation-6D to canonical normalized WXYZ quaternions."""

    matrix = _rotation6d_to_matrix(rotation6d)
    prefix_shape = matrix.shape[:-2]
    flat = matrix.reshape(-1, 3, 3)
    quaternion = np.empty((flat.shape[0], 4), dtype=np.float64)

    # A branch per matrix avoids the numerical instability of a single
    # trace-only formula around 180-degree rotations.
    for index, rotation in enumerate(flat):
        trace = np.trace(rotation)
        if trace > 0:
            scale = np.sqrt(trace + 1.0) * 2
            w = 0.25 * scale
            x = (rotation[2, 1] - rotation[1, 2]) / scale
            y = (rotation[0, 2] - rotation[2, 0]) / scale
            z = (rotation[1, 0] - rotation[0, 1]) / scale
        else:
            diagonal = np.diag(rotation)
            axis = int(np.argmax(diagonal))
            if axis == 0:
                scale = (
                    np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2
                )
                w = (rotation[2, 1] - rotation[1, 2]) / scale
                x = 0.25 * scale
                y = (rotation[0, 1] + rotation[1, 0]) / scale
                z = (rotation[0, 2] + rotation[2, 0]) / scale
            elif axis == 1:
                scale = (
                    np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2
                )
                w = (rotation[0, 2] - rotation[2, 0]) / scale
                x = (rotation[0, 1] + rotation[1, 0]) / scale
                y = 0.25 * scale
                z = (rotation[1, 2] + rotation[2, 1]) / scale
            else:
                scale = (
                    np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2
                )
                w = (rotation[1, 0] - rotation[0, 1]) / scale
                x = (rotation[0, 2] + rotation[2, 0]) / scale
                y = (rotation[1, 2] + rotation[2, 1]) / scale
                z = 0.25 * scale
        quaternion[index] = (w, x, y, z)

    quaternion /= np.linalg.norm(quaternion, axis=-1, keepdims=True)
    quaternion = np.where(quaternion[:, :1] < 0, -quaternion, quaternion)
    return quaternion.reshape(*prefix_shape, 4).astype(np.float32)


def ee16_to_xvla20(value: np.ndarray, *, invert_gripper: bool = True) -> np.ndarray:
    """Convert RoboDojo 16D EE state/action to X-VLA's physical 20D layout."""

    value = _as_float_array(value, 16, "ee16")
    left_gripper = value[..., 7:8]
    right_gripper = value[..., 15:16]
    if invert_gripper:
        left_gripper = 1.0 - left_gripper
        right_gripper = 1.0 - right_gripper
    return np.concatenate(
        (
            value[..., 0:3],
            quaternion_wxyz_to_rotation6d(value[..., 3:7]),
            left_gripper,
            value[..., 8:11],
            quaternion_wxyz_to_rotation6d(value[..., 11:15]),
            right_gripper,
        ),
        axis=-1,
    ).astype(np.float32)


def xvla20_to_ee16(
    value: np.ndarray,
    *,
    invert_gripper: bool = True,
    clip_gripper: bool = True,
) -> np.ndarray:
    """Convert a denormalized physical X-VLA 20D action/state to RoboDojo 16D.

    Important: this function does not undo model preprocessing. If X-VLA is
    configured with action normalization other than ``IDENTITY``, run the
    matching action unnormalizer before this conversion.
    """

    value = _as_float_array(value, 20, "xvla20")
    left_gripper = value[..., 9:10]
    right_gripper = value[..., 19:20]
    if clip_gripper:
        left_gripper = np.clip(left_gripper, 0.0, 1.0)
        right_gripper = np.clip(right_gripper, 0.0, 1.0)
    if invert_gripper:
        left_gripper = 1.0 - left_gripper
        right_gripper = 1.0 - right_gripper
    return np.concatenate(
        (
            value[..., 0:3],
            rotation6d_to_quaternion_wxyz(value[..., 3:9]),
            left_gripper,
            value[..., 10:13],
            rotation6d_to_quaternion_wxyz(value[..., 13:19]),
            right_gripper,
        ),
        axis=-1,
    ).astype(np.float32)
