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
from scipy.spatial.transform import Rotation as R


def _as_float_array(value: np.ndarray, expected_dim: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 0 or array.shape[-1] != expected_dim:
        raise ValueError(
            f"{name} must have last dimension {expected_dim}, got shape {array.shape}"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or infinite values")
    return array


def _wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    """Convert WXYZ quaternion to XYZW format for scipy."""
    return np.roll(q, -1, axis=-1)


def _xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    """Convert XYZW quaternion to WXYZ format from scipy."""
    return np.roll(q, 1, axis=-1)


def quaternion_wxyz_to_rotation6d(quaternion: np.ndarray) -> np.ndarray:
    """Convert ``(..., 4)`` WXYZ quaternions to X-VLA's interleaved rotation-6D.

    X-VLA flattens the first two matrix columns in row-major order, producing
    ``[r00, r01, r10, r11, r20, r21]``.
    """
    quaternion = _as_float_array(quaternion, 4, "quaternion")
    # scipy uses xyzw format
    quat_xyzw = _wxyz_to_xyzw(quaternion)
    return R.from_quat(quat_xyzw).as_matrix()[..., :, :2].reshape(*quaternion.shape[:-1], 6).astype(np.float32)


def rotation6d_to_quaternion_wxyz(rotation6d: np.ndarray) -> np.ndarray:
    """Convert X-VLA rotation-6D to canonical normalized WXYZ quaternions."""
    rotation6d = _as_float_array(rotation6d, 6, "rotation6d")

    a1 = rotation6d[..., 0:5:2]
    a2 = rotation6d[..., 1:6:2]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    proj = np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = a2 - proj
    b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)
    b3 = np.cross(b1, b2)
    rot_mats = np.stack((b1, b2, b3), axis=-1)

    # scipy returns xyzw format
    quat_xyzw = R.from_matrix(rot_mats).as_quat()
    return _xyzw_to_wxyz(quat_xyzw).astype(np.float32)


def ee16_to_xvla20(value: np.ndarray, *, invert_gripper: bool = False) -> np.ndarray:
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
    invert_gripper: bool = False,
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
