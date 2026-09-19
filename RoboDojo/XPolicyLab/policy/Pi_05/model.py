#!/usr/bin/env python
# -- coding: UTF-8
"""
#!/usr/bin/python3
"""
from pathlib import Path
import json
import math
import os
import sys
from typing import Any

import cv2
import numpy as np

from openpi.policies import policy_config as _policy_config
from openpi.shared import normalize as _normalize
from openpi.training import config as _config

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.checkpoint_resolver import candidate_checkpoint_roots
from XPolicyLab.utils.process_data import (
    decode_image_bit,
    get_robot_action_dim_info,
    pack_robot_state,
    unpack_robot_state,
)


_POLICY_DIR = Path(__file__).resolve().parent
_CHECKPOINTS_DIR = _POLICY_DIR / "checkpoints"


def _finite_round(value: Any, digits: int = 4) -> float | None:
    """Round a scalar for logging; NaN/±Inf → None (mirrors X_VLA finite_list)."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return round(number, digits)


def _array_log(value: Any, max_values: int = 24) -> dict[str, Any]:
    """紧凑数组日志摘要：shape/dtype/有限值范围 + NaN|Inf 计数 + 前 max_values 个值。"""
    array = np.asarray(value)
    flat = array.reshape(-1)
    summary: dict[str, Any] = {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
    }
    if flat.size == 0:
        return summary
    finite = np.isfinite(flat.astype(np.float64, copy=False))
    summary["n_nan_inf"] = int((~finite).sum())
    if finite.any():
        finite_values = flat[finite].astype(np.float64)
        summary.update(
            {
                "min": float(finite_values.min()),
                "max": float(finite_values.max()),
                "mean": float(finite_values.mean()),
            }
        )
    summary["values"] = [_finite_round(item) for item in flat[:max_values]]
    return summary


def _action_log(action: Any) -> dict[str, Any]:
    """动作日志：dict(arm/ee 键)或 list[dict](多步)或裸数组 → 统一 JSON 摘要。"""
    if isinstance(action, dict):
        return {key: _array_log(value) for key, value in action.items()}
    if isinstance(action, (list, tuple)):
        if len(action) > 0 and isinstance(action[0], dict):
            return {"n_steps": len(action), "step0": _action_log(action[0])}
        return _array_log(action)
    return _array_log(action)


def _extract_step_number(value: Any) -> int | None:
    matches = [part for part in str(value).split("/") if part]
    if not matches:
        return None
    digits = "".join(ch for ch in matches[-1] if ch.isdigit())
    return int(digits) if digits else None


def _resolve_pi05_model_root(model_cfg: dict[str, Any]) -> Path:
    # Shared precedence: model_path/checkpoint_path keys > ckpt_name-as-path >
    # {bench}-{ckpt}-{env}-{action}-{seed} concat > checkpoints/<ckpt_name>.
    candidates = candidate_checkpoint_roots(
        model_cfg,
        _CHECKPOINTS_DIR,
        policy_dir=_POLICY_DIR,
        explicit_keys=("model_path", "checkpoint_path"),
    )
    if not candidates:
        raise ValueError("ckpt_name or model_path is required for Pi_05.")
    checkpoint_root = next((candidate for candidate in candidates if candidate.exists()), candidates[0])
    if not checkpoint_root.is_dir():
        return checkpoint_root

    candidate_dirs = []
    if (checkpoint_root / "params").exists() or (checkpoint_root / "assets").exists():
        candidate_dirs.append(checkpoint_root)
    candidate_dirs.extend(
        child
        for child in sorted(checkpoint_root.iterdir())
        if child.is_dir() and ((child / "params").exists() or (child / "assets").exists())
    )
    if not candidate_dirs:
        return checkpoint_root

    checkpoint_num = model_cfg.get("checkpoint_num")
    desired_step = _extract_step_number(checkpoint_num)
    if desired_step is not None:
        normalized = str(desired_step)
        for candidate in candidate_dirs:
            name = candidate.name.lstrip("0") or "0"
            if name == normalized:
                return candidate

        for candidate in candidate_dirs:
            candidate_step = _extract_step_number(candidate.name)
            if candidate_step is None:
                continue
            scaled_step = desired_step
            while len(str(scaled_step)) < len(str(candidate_step)):
                scaled_step *= 10
            if candidate_step in {desired_step, scaled_step}:
                return candidate

    numeric_dirs = [candidate for candidate in candidate_dirs if _extract_step_number(candidate.name) is not None]
    if numeric_dirs:
        return max(numeric_dirs, key=lambda candidate: _extract_step_number(candidate.name) or -1)
    return candidate_dirs[0]


class Model(ModelTemplate):
    def __init__(self, model_cfg: dict[str, Any]):
        self.task_name = model_cfg["task_name"]
        self.action_type = model_cfg.get("action_type", "joint")
        self.env_cfg_type = model_cfg.get("env_cfg_type")
        self.robot_action_dim_info = (
            get_robot_action_dim_info(model_cfg["env_cfg_type"]) if model_cfg.get("env_cfg_type") is not None else None
        )
        self.observation_window: dict[str, Any] | None = None
        self._latest_env_idx_list: list[int] = [0]
        # 逐请求日志（参照 X_VLA）：[pi05][io] client_observation / server_actions。
        # deploy.yml 配 log_io: false 可关。
        self.log_io = bool(model_cfg.get("log_io", True))
        self._request_index = 0

        if self.log_io:
            dim_info = self.robot_action_dim_info or {}
            action_dim = sum(dim_info.get("arm_dim", [])) + sum(dim_info.get("ee_dim", [])) or None
            print(
                "[pi05] "
                f"task_name={self.task_name} "
                f"action_type={self.action_type} "
                f"env_cfg_type={self.env_cfg_type} "
                f"train_config_name={model_cfg.get('train_config_name')} "
                f"repo_id={model_cfg.get('repo_id')} "
                f"checkpoint_num={model_cfg.get('checkpoint_num')} "
                f"action_dim={action_dim} "
                f"log_io={self.log_io}",
                flush=True,
            )

        self.policy = self.get_model(model_cfg=model_cfg)
        self.model = self.policy

    def get_model(self, model_cfg: dict[str, Any]):
        train_config_name = model_cfg.get("train_config_name", "pi05_aloha")
        repo_id = model_cfg.get("repo_id", "1118")
        model_root = _resolve_pi05_model_root(model_cfg)
        self._model_root = model_root

        if self.log_io:
            print(
                f"[pi05] loading train_config={train_config_name} repo_id={repo_id} "
                f"model_root={model_root}",
                flush=True,
            )

        config = _config.get_config(train_config_name)
        norm_stats = None
        if repo_id is not None:
            norm_stats = _normalize.load(model_root / "assets" / str(repo_id))

        return _policy_config.create_trained_policy(config, str(model_root), norm_stats=norm_stats)

    def update_obs(self, obs):
        self.update_obs_batch([obs])

    def update_obs_batch(self, obs_list):
        self._latest_env_idx_list = [obs.get("env_idx", index) for index, obs in enumerate(obs_list)]
        encoded_obs_list = [
            encode_obs(obs, self.action_type, self.robot_action_dim_info) for obs in obs_list
        ]
        self.observation_window = stack_obs(encoded_obs_list)
        # 缓存每 env 的原始/编码观测（日志用，参照 X_VLA _raw_by_env/_latest_by_env）。
        self._raw_obs_by_env = dict(zip(self._latest_env_idx_list, obs_list, strict=True))
        self._encoded_obs_by_env = dict(zip(self._latest_env_idx_list, encoded_obs_list, strict=True))

    def get_action(self, **kwargs):
        action_list = self.get_action_batch(env_idx_list=[self._latest_env_idx_list[0]], **kwargs)
        return action_list[0]

    def get_action_batch(self, env_idx_list=None, **kwargs):
        if self.observation_window is None:
            raise AssertionError("update_obs or update_obs_batch first!")

        env_idx_list = env_idx_list or self._latest_env_idx_list
        # actions = self.policy.infer(self.observation_window, **kwargs)["actions"]
        action_list = []
        request_index = self._request_index

        for batch_index, env_idx in enumerate(env_idx_list):
            single_observation = slice_stacked_obs(self.observation_window, batch_index)
            actions = self.policy.infer(single_observation, **kwargs)["actions"]

            if self.robot_action_dim_info is None:
                action = actions
            else:
                action = unpack_robot_state(
                    actions,
                    self.action_type,
                    self.robot_action_dim_info,
                    source_type="obs",
                )
            action_list.append(action)

            if self.log_io:
                self._log_request(request_index, int(env_idx), actions, action)

        if self.log_io:
            self._request_index += 1
        return action_list

    def _log_request(self, request_index: int, env_idx: int, raw_actions: Any, action: Any) -> None:
        """参照 X_VLA [x_vla][io] 打 [pi05][io] 两事件：client_observation + server_actions。"""
        encoded = self._encoded_obs_by_env.get(env_idx)
        raw = self._raw_obs_by_env.get(env_idx)

        observation_summary: dict[str, Any] = {"event": "client_observation", "request": request_index, "env_idx": env_idx}
        if isinstance(raw, dict):
            for key in ("episode_idx", "task_name"):
                if key in raw:
                    observation_summary[key] = raw[key]
        if encoded is not None:
            observation_summary["prompt"] = str(encoded.get("prompt", ""))[:200]
            state = encoded.get("state")
            if state is not None:
                observation_summary["state"] = _array_log(state)
            observation_summary["images"] = {
                name: {"shape": list(image.shape), "dtype": str(image.dtype)}
                for name, image in (encoded.get("images") or {}).items()
            }
        print("[pi05][io] " + json.dumps(observation_summary, ensure_ascii=False), flush=True)

        action_summary = {
            "event": "server_actions",
            "request": request_index,
            "env_idx": env_idx,
            "raw_actions": _array_log(raw_actions),
            "actions": _action_log(action),
        }
        print("[pi05][io] " + json.dumps(action_summary, ensure_ascii=False, default=_json_default), flush=True)


def _json_default(value: Any) -> Any:
    """JSON 兜底（np 标量等无法直接序列化时转字符串）。"""
    try:
        if isinstance(value, np.ndarray):
            return value.tolist()
        return str(value)
    except Exception:
        return repr(value)

    def reset(self):
        self.observation_window = None
        self._latest_env_idx_list = [0]

    def reset_obsrvationwindows(self):
        self.reset()


def encode_obs(observation, action_type, robot_action_dim_info):
    if "images" in observation and "state" in observation:
        state = np.asarray(observation["state"], dtype=np.float32)
        images = {
            "cam_high": ensure_chw_uint8(observation["images"]["cam_high"]),
            "cam_left_wrist": ensure_chw_uint8(observation["images"]["cam_left_wrist"]),
            "cam_right_wrist": ensure_chw_uint8(observation["images"]["cam_right_wrist"]),
        }
        prompt = observation.get("instruction")
        return {"state": state, "images": images, "prompt": prompt}

    if robot_action_dim_info is None:
        raise ValueError("env_cfg_type is required when encoding raw environment observations.")

    images = {
        "cam_high": ensure_chw_uint8(extract_image(observation, ["cam_high", "cam_head", "head_camera", "top_camera"])),
        "cam_left_wrist": ensure_chw_uint8(
            extract_image(observation, ["cam_left_wrist", "left_camera", "left_wrist", "wrist_left"])
        ),
        "cam_right_wrist": ensure_chw_uint8(
            extract_image(observation, ["cam_right_wrist", "right_camera", "right_wrist", "wrist_right"])
        ),
    }
    state = pack_robot_state(observation, action_type, robot_action_dim_info, source_type="obs").astype(np.float32)
    prompt = observation.get("instruction")
    return {"state": state, "images": images, "prompt": prompt}


def stack_obs(obs_list: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "state": np.stack([obs["state"] for obs in obs_list], axis=0),
        "images": {
            "cam_high": np.stack([obs["images"]["cam_high"] for obs in obs_list], axis=0),
            "cam_left_wrist": np.stack([obs["images"]["cam_left_wrist"] for obs in obs_list], axis=0),
            "cam_right_wrist": np.stack([obs["images"]["cam_right_wrist"] for obs in obs_list], axis=0),
        },
        "prompt": [obs["prompt"] for obs in obs_list],
    }


def slice_stacked_obs(obs: dict[str, Any], batch_index: int) -> dict[str, Any]:
    return {
        "state": obs["state"][batch_index],
        "images": {
            "cam_high": obs["images"]["cam_high"][batch_index],
            "cam_left_wrist": obs["images"]["cam_left_wrist"][batch_index],
            "cam_right_wrist": obs["images"]["cam_right_wrist"][batch_index],
        },
        "prompt": obs["prompt"][batch_index],
    }


def extract_image(observation, candidate_names):
    vision = observation.get("vision", {})
    for candidate_name in candidate_names:
        if candidate_name not in vision:
            continue
        image = vision[candidate_name]
        if isinstance(image, dict):
            for image_key in ("color", "rgb"):
                if image_key in image:
                    return image[image_key]
        else:
            return image
    raise KeyError(f"Could not find any image for candidates: {candidate_names}")


def ensure_chw_uint8(image):
    if isinstance(image, (bytes, bytearray, memoryview)):
        image = decode_compressed_image(np.frombuffer(bytes(image), dtype=np.uint8))

    image = np.asarray(image)

    if image.ndim == 1 and image.dtype == np.uint8:
        image = decode_compressed_image(image)

    if image.ndim != 3:
        raise ValueError(f"Expected image ndim=3, got shape {image.shape}")

    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image, 0.0, 1.0)
        image = (image * 255.0).astype(np.uint8)
    elif image.dtype != np.uint8:
        image = image.astype(np.uint8)

    if image.shape[-1] in (1, 3):
        image_hwc = image
    elif image.shape[0] in (1, 3):
        image_hwc = np.transpose(image, (1, 2, 0))
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")

    return np.transpose(image_hwc, (2, 0, 1))


def decode_compressed_image(image_buffer):
    return decode_image_bit(image_buffer)