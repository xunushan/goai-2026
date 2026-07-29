"""Serve a LeRobot SmolVLA policy through XPolicyLab.

This adapter intentionally imports only the LeRobot package installed in the
active Conda environment.  It accepts RoboDojo ARX-X5 observations and returns
14-dimensional absolute joint actions in this order:

    left arm joints (6), left gripper, right arm joints (6), right gripper.
"""

from __future__ import annotations

import types
from pathlib import Path
from typing import Any

import numpy as np
import torch

# Older Torch builds do not expose the optional Intel XPU namespace. LeRobot
# probes it while resolving a default device; the shim is inert on production
# CUDA servers and keeps checkpoint inspection possible on CPU-only machines.
if not hasattr(torch, "xpu"):
    torch.xpu = types.SimpleNamespace(is_available=lambda: False)

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.checkpoint_resolver import candidate_checkpoint_roots
from XPolicyLab.utils.process_data import (
    decode_image_bit,
    get_robot_action_dim_info,
    pack_robot_state,
    unpack_robot_state,
)

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.utils.constants import OBS_STATE


POLICY_DIR = Path(__file__).resolve().parent
CHECKPOINTS_DIR = POLICY_DIR / "checkpoints"
ACTION_DIM = 14
CAMERA_MAPPING = {
    "observation.images.cam_high": (
        "cam_high", "cam_head", "head_camera", "top_camera"
    ),
    "observation.images.cam_left_wrist": (
        "cam_left_wrist", "left_camera", "left_wrist", "wrist_left"
    ),
    "observation.images.cam_right_wrist": (
        "cam_right_wrist", "right_camera", "right_wrist", "wrist_right"
    ),
}
CHECKPOINT_CAMERA_KEYS = (
    "observation.images.camera1",
    "observation.images.camera2",
    "observation.images.camera3",
)


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {value}")
    return device


def _has_checkpoint(path: Path) -> bool:
    return path.is_dir() and (path / "model.safetensors").is_file()


def _resolve_checkpoint(model_cfg: dict[str, Any]) -> Path:
    roots = candidate_checkpoint_roots(
        model_cfg,
        CHECKPOINTS_DIR,
        policy_dir=POLICY_DIR,
        explicit_keys=("checkpoint_path", "pretrained_path", "model_path"),
    )
    checked: list[Path] = []
    for root in roots:
        variants = (
            root,
            root / "pretrained_model",
            root / "checkpoints" / "last" / "pretrained_model",
        )
        for candidate in variants:
            checked.append(candidate)
            if _has_checkpoint(candidate):
                return candidate.resolve()
        if root.is_dir():
            for candidate in sorted(root.rglob("model.safetensors")):
                parent = candidate.parent
                checked.append(parent)
                if _has_checkpoint(parent):
                    return parent.resolve()
    detail = "\n".join(f"  - {path}" for path in checked)
    raise FileNotFoundError(
        "Could not find a LeRobot pretrained model directory. Checked:\n" + detail
    )


def _extract_image(observation: dict[str, Any], names: tuple[str, ...]) -> Any:
    vision = observation.get("vision", {})
    for name in names:
        if name not in vision:
            continue
        value = vision[name]
        if isinstance(value, dict):
            for image_key in ("color", "rgb"):
                if image_key in value:
                    return value[image_key]
        return value
    raise KeyError(f"Missing camera; accepted names: {names}")


def _chw_uint8(value: Any) -> np.ndarray:
    if isinstance(value, (bytes, bytearray, memoryview)):
        value = np.frombuffer(bytes(value), dtype=np.uint8)
    image = np.asarray(value)
    if image.ndim == 1 and image.dtype == np.uint8:
        image = decode_image_bit(image)
    if image.ndim != 3:
        raise ValueError(f"Expected a 3D image, got {image.shape}")
    if np.issubdtype(image.dtype, np.floating):
        scale = 255.0 if float(np.nanmax(image)) <= 1.0 else 1.0
        image = np.clip(image * scale, 0, 255).astype(np.uint8)
    elif image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    if image.shape[-1] in (1, 3):
        image = np.transpose(image, (2, 0, 1))
    elif image.shape[0] not in (1, 3):
        raise ValueError(f"Cannot determine image layout for {image.shape}")
    return np.ascontiguousarray(image)


def _prompt(observation: dict[str, Any], fallback: str | None) -> str:
    for key in ("instruction", "prompt", "task", "language_instruction"):
        value = observation.get(key)
        if isinstance(value, (list, tuple)) and value:
            value = value[0]
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="ignore")
        if value is not None and str(value).strip():
            return str(value).strip()
    if fallback and str(fallback).strip():
        return str(fallback).strip()
    raise ValueError("SmolVLA requires an instruction or a configured prompt")


class Model(ModelTemplate):
    def __init__(self, model_cfg):
        self.cfg = dict(model_cfg)
        self.action_type = self.cfg.get("action_type", "joint")
        if self.action_type != "joint":
            raise ValueError("smolvla_lerobot supports only action_type='joint'")

        env_cfg = self.cfg.get("env_cfg_type") or self.cfg.get("env_cfg")
        if not env_cfg:
            raise ValueError("env_cfg_type is required")
        self.robot_action_dim_info = get_robot_action_dim_info(env_cfg)
        self.default_prompt = self.cfg.get("prompt") or self.cfg.get("task_name")
        self.device = _resolve_device(str(self.cfg.get("device", "auto")))
        self.checkpoint_path = _resolve_checkpoint(self.cfg)

        config = PreTrainedConfig.from_pretrained(str(self.checkpoint_path))
        config.device = str(self.device)
        policy_class = get_policy_class("smolvla")
        self.policy = policy_class.from_pretrained(
            str(self.checkpoint_path), config=config
        ).to(self.device)
        self.policy.eval()
        self.model = self.policy

        self._validate_config()
        self.preprocessor, self.postprocessor = self._make_processors()
        maximum = int(getattr(self.policy.config, "chunk_size", 1))
        requested = int(self.cfg.get("actions_per_chunk") or maximum)
        if not 1 <= requested <= maximum:
            raise ValueError(
                f"actions_per_chunk must be within [1,{maximum}], got {requested}"
            )
        self.actions_per_chunk = requested
        self.log_io = bool(self.cfg.get("log_io", False))
        self._payloads: dict[int, dict[str, Any]] = {}
        self._latest_env_indices = [0]
        print(
            f"[smolvla_lerobot] ready checkpoint={self.checkpoint_path} "
            f"device={self.device} execute_steps={self.actions_per_chunk}",
            flush=True,
        )

    def _validate_config(self) -> None:
        input_features = getattr(self.policy.config, "input_features", {})
        output_features = getattr(self.policy.config, "output_features", {})
        state = input_features.get(OBS_STATE)
        action = output_features.get("action")
        state_shape = tuple(getattr(state, "shape", ()))
        action_shape = tuple(getattr(action, "shape", ()))
        if state_shape != (ACTION_DIM,):
            raise ValueError(
                f"Checkpoint observation.state must be (14,), got {state_shape}"
            )
        if action_shape != (ACTION_DIM,):
            raise ValueError(f"Checkpoint action must be (14,), got {action_shape}")
        missing = set(CHECKPOINT_CAMERA_KEYS).difference(input_features)
        if missing:
            raise ValueError(f"Checkpoint is missing cameras: {sorted(missing)}")

    def _make_processors(self):
        device_override = {"device": str(self.device)}
        return make_pre_post_processors(
            self.policy.config,
            pretrained_path=str(self.checkpoint_path),
            preprocessor_overrides={
                "device_processor": device_override,
            },
            postprocessor_overrides={"device_processor": device_override},
        )

    def _encode(self, observation: dict[str, Any]) -> dict[str, Any]:
        if "images" in observation and "state" in observation:
            images = {
                key: _chw_uint8(observation["images"][key])
                for key in ("cam_high", "cam_left_wrist", "cam_right_wrist")
            }
            state = np.asarray(observation["state"], dtype=np.float32)
        else:
            images = {
                key.removeprefix("observation.images."): _chw_uint8(
                    _extract_image(observation, names)
                )
                for key, names in CAMERA_MAPPING.items()
            }
            state = pack_robot_state(
                observation,
                self.action_type,
                self.robot_action_dim_info,
                source_type="obs",
            ).astype(np.float32)
        if state.shape != (ACTION_DIM,):
            raise ValueError(f"Expected 14D joint state, got {state.shape}")
        return {
            "state": state,
            "images": images,
            "task": _prompt(observation, self.default_prompt),
        }

    @staticmethod
    def _tensor_image(image: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(image, dtype=torch.float32).unsqueeze(0) / 255.0

    def _to_lerobot(self, payload: dict[str, Any]) -> dict[str, Any]:
        images = payload["images"]
        return {
            OBS_STATE: torch.as_tensor(
                payload["state"], dtype=torch.float32
            ).unsqueeze(0),
            "observation.images.cam_high": self._tensor_image(images["cam_high"]),
            "observation.images.cam_left_wrist": self._tensor_image(
                images["cam_left_wrist"]
            ),
            "observation.images.cam_right_wrist": self._tensor_image(
                images["cam_right_wrist"]
            ),
            "task": payload["task"],
        }

    @staticmethod
    def _stack(items: list[dict[str, Any]]) -> dict[str, Any]:
        if len(items) == 1:
            return items[0]
        result: dict[str, Any] = {"task": [item["task"] for item in items]}
        for key in items[0]:
            if key != "task":
                result[key] = torch.cat([item[key] for item in items], dim=0)
        return result

    def update_obs(self, obs):
        self.update_obs_batch([obs])

    def update_obs_batch(self, obs):
        self._latest_env_indices = [
            int(item.get("env_idx", index)) for index, item in enumerate(obs)
        ]
        self._payloads = {
            env_idx: self._encode(item)
            for env_idx, item in zip(self._latest_env_indices, obs, strict=True)
        }

    @torch.inference_mode()
    def _infer(self, payloads: list[dict[str, Any]]) -> np.ndarray:
        batch = self._stack([self._to_lerobot(payload) for payload in payloads])
        batch = self.preprocessor(batch)
        actions = self.policy.predict_action_chunk(batch)
        actions = actions[:, : self.actions_per_chunk]
        batch_size, horizon, dimension = actions.shape
        actions = self.postprocessor(
            actions.reshape(batch_size * horizon, dimension)
        ).reshape(batch_size, horizon, dimension)
        result = actions.detach().cpu().float().numpy()
        if result.shape != (batch_size, self.actions_per_chunk, ACTION_DIM):
            raise ValueError(f"Unexpected action shape: {result.shape}")
        if not np.isfinite(result).all():
            raise FloatingPointError("SmolVLA returned NaN or Inf")
        if self.log_io:
            print(
                f"[smolvla_lerobot] state={payloads[0]['state'].round(4).tolist()} "
                f"action0={result[0, 0].round(4).tolist()}",
                flush=True,
            )
        return result

    def get_action(self, **kwargs):
        return self.get_action_batch([self._latest_env_indices[0]], **kwargs)[0]

    def get_action_batch(self, env_idx_list=None, **kwargs):
        indices = (
            self._latest_env_indices
            if env_idx_list is None
            else [int(index) for index in env_idx_list]
        )
        missing = [index for index in indices if index not in self._payloads]
        if missing:
            raise KeyError(f"Missing observations for env_idx={missing}")
        predictions = self._infer([self._payloads[index] for index in indices])
        return [
            unpack_robot_state(
                chunk,
                self.action_type,
                self.robot_action_dim_info,
                source_type="obs",
            )
            for chunk in predictions
        ]

    def reset(self):
        self.policy.reset()
        self._payloads = {}
        self._latest_env_indices = [0]
