from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from XPolicyLab.policy.X_VLA.model import Model as _XVLABasis
from XPolicyLab.policy.X_VLA.model import rotate6d_to_quat


_POLICY_DIR = Path(__file__).resolve().parent
_DEFAULT_MODEL_DIR = _POLICY_DIR / "checkpoints" / "shared" / "X-VLA-RoboTwin2"
_ACTION_NAMES = (
    "l_x", "l_y", "l_z", "l_w", "l_wx", "l_wy", "l_wz", "l_g",
    "r_x", "r_y", "r_z", "r_w", "r_wx", "r_wy", "r_wz", "r_g",
)


def _absolute_local_path(value: str | None) -> str | None:
    if not value:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = _POLICY_DIR / path
    return str(path.resolve())


def _normalize_quaternion_wxyz(quaternion: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float32)
    norm = np.linalg.norm(quaternion, axis=-1, keepdims=True)
    if np.any(norm < 1e-8):
        raise ValueError("X-VLA produced a zero-norm quaternion.")
    return quaternion / norm


def action20_to_robodojo(
    action_chunk: np.ndarray,
    *,
    gripper_close_threshold: float = 0.7,
) -> list[dict[str, np.ndarray]]:
    """Convert X-VLA EE6D actions to RoboDojo absolute EE commands.

    X-VLA's gripper channels are close probabilities after sigmoid. RoboDojo
    expects gripper opening commands: 1=open and 0=closed.
    """
    chunk = np.asarray(action_chunk, dtype=np.float32)
    if chunk.ndim == 1:
        chunk = chunk[None, :]
    if chunk.ndim != 2 or chunk.shape[-1] != 20:
        raise ValueError(f"Expected X-VLA action chunk [T, 20], got {chunk.shape}.")

    left_quat = _normalize_quaternion_wxyz(rotate6d_to_quat(chunk[:, 3:9]))
    right_quat = _normalize_quaternion_wxyz(rotate6d_to_quat(chunk[:, 13:19]))

    # X-VLA: sigmoid output is probability of closing.
    # RoboDojo: 1 means open and 0 means closed.
    # Match the released RoboTwin2 client exactly: p > threshold means closed;
    # the boundary p == threshold remains open.
    left_open = (chunk[:, 9] <= gripper_close_threshold).astype(np.float32)
    right_open = (chunk[:, 19] <= gripper_close_threshold).astype(np.float32)

    result = []
    for index in range(chunk.shape[0]):
        result.append(
            {
                "left_ee_pose": np.concatenate(
                    [chunk[index, 0:3], left_quat[index]], axis=0
                ).astype(np.float32),
                "left_ee_joint_state": np.asarray(
                    [left_open[index]], dtype=np.float32
                ),
                "right_ee_pose": np.concatenate(
                    [chunk[index, 10:13], right_quat[index]], axis=0
                ).astype(np.float32),
                "right_ee_joint_state": np.asarray(
                    [right_open[index]], dtype=np.float32
                ),
            }
        )
    return result


def _state16(observation: dict[str, Any]) -> np.ndarray:
    state = observation["state"]
    return np.concatenate(
        [
            np.asarray(state["left_ee_pose"], dtype=np.float32),
            np.asarray(state["left_ee_joint_state"], dtype=np.float32)[-1:],
            np.asarray(state["right_ee_pose"], dtype=np.float32),
            np.asarray(state["right_ee_joint_state"], dtype=np.float32)[-1:],
        ]
    )


def _action_dict_to_16(action: dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate(
        [
            action["left_ee_pose"],
            action["left_ee_joint_state"],
            action["right_ee_pose"],
            action["right_ee_joint_state"],
        ]
    ).astype(np.float32)


def _array_summary(value: np.ndarray) -> dict[str, Any]:
    array = np.asarray(value)
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "min": float(array.min()),
        "max": float(array.max()),
        "mean": float(array.mean()),
    }


class Model(_XVLABasis):
    """RoboDojo runtime adapter for the released X-VLA-RoboTwin2 checkpoint."""

    def __init__(self, model_cfg):
        cfg = dict(model_cfg)
        cfg["action_type"] = "ee"
        cfg["domain_id"] = 6

        model_path = cfg.get("model_path") or str(_DEFAULT_MODEL_DIR)
        processor_path = cfg.get("processor_path") or model_path
        cfg["model_path"] = _absolute_local_path(model_path)
        cfg["processor_path"] = _absolute_local_path(processor_path)

        self.gripper_close_threshold = float(
            cfg.get("gripper_close_threshold", 0.7)
        )
        if not 0.0 <= self.gripper_close_threshold <= 1.0:
            raise ValueError("gripper_close_threshold must be in [0, 1].")
        self.log_io = bool(cfg.get("log_io", True))
        self.log_max_requests = int(cfg.get("log_max_requests", 10))
        self.log_full_actions = bool(cfg.get("log_full_actions", False))
        if self.log_max_requests < 0:
            raise ValueError("log_max_requests must be >= 0 (0 means unlimited).")
        self._request_index = 0
        self._raw_observations: list[dict[str, Any]] = []

        super().__init__(cfg)

        if self.model.action_mode != "ee6d":
            raise ValueError(
                "X-VLA-RoboTwin2 must use action_mode='ee6d', "
                f"got {self.model.action_mode!r}."
            )
        if int(self.model_cfg["domain_id"]) != 6:
            raise ValueError("X-VLA-RoboTwin2 requires domain_id=6.")

    def update_obs_batch(self, obs_list):
        self._raw_observations = list(obs_list)
        super().update_obs_batch(obs_list)

    def get_action_batch(self, env_idx_list=None, **kwargs):
        if self.observation_window is None:
            raise AssertionError("update_obs or update_obs_batch first!")

        if env_idx_list is None:
            env_idx_list = self._latest_env_idx_list
        if not env_idx_list:
            return []
        action_batches = []
        for batch_index, env_idx in enumerate(env_idx_list):
            encoded_obs = self.observation_window[batch_index]
            raw_action20 = self.infer(encoded_obs)
            actions = action20_to_robodojo(
                raw_action20,
                gripper_close_threshold=self.gripper_close_threshold,
            )
            action_batches.append(actions)

            self._request_index += 1
            should_log = self.log_io and (
                self.log_max_requests == 0
                or self._request_index <= self.log_max_requests
            )
            if should_log:
                raw_obs = self._raw_observations[batch_index]
                state16 = _state16(raw_obs)
                actions16 = np.stack(
                    [_action_dict_to_16(action) for action in actions], axis=0
                )
                left_quat_norm = np.linalg.norm(actions16[:, 3:7], axis=1)
                right_quat_norm = np.linalg.norm(actions16[:, 11:15], axis=1)
                observation_log = {
                    "event": "client_observation",
                    "request": self._request_index,
                    "env_idx": int(env_idx),
                    "instruction": encoded_obs["prompt"],
                    "state16": state16.tolist(),
                    "state_names": list(_ACTION_NAMES),
                    "xvla_proprio20": encoded_obs["proprio"].tolist(),
                    "images": {
                        "cam_high": _array_summary(encoded_obs["images"][0]),
                    },
                    "valid_views": 1,
                    "domain_id": 6,
                }
                action_log = {
                    "event": "server_actions",
                    "request": self._request_index,
                    "shape": list(actions16.shape),
                    "action_names": list(_ACTION_NAMES),
                    "actions": actions16.tolist(),
                    "min": actions16.min(axis=0).tolist(),
                    "max": actions16.max(axis=0).tolist(),
                    "postprocess_adjustments": {
                        "quaternion_normalized": True,
                        "left_quaternion_norm": left_quat_norm.tolist(),
                        "right_quaternion_norm": right_quat_norm.tolist(),
                        "raw_left_close_probability": np.asarray(
                            raw_action20
                        )[:, 9].tolist(),
                        "raw_right_close_probability": np.asarray(
                            raw_action20
                        )[:, 19].tolist(),
                        "gripper_output_range": [0.0, 1.0],
                    },
                    "gripper_semantics": "1=open,0=closed",
                    "gripper_close_threshold": self.gripper_close_threshold,
                }
                if self.log_full_actions:
                    action_log["raw_action20"] = np.asarray(raw_action20).tolist()
                print(
                    "[xvla_robtwin][io] "
                    + json.dumps(observation_log, ensure_ascii=False),
                    flush=True,
                )
                print(
                    "[xvla_robtwin][io] "
                    + json.dumps(action_log, ensure_ascii=False),
                    flush=True,
                )

        return action_batches

    def reset(self):
        super().reset()
        self._raw_observations = []


def get_model(deploy_cfg):
    return Model(deploy_cfg)
