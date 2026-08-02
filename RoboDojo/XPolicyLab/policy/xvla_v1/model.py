"""Serve a LeRobot X-VLA checkpoint through the XPolicyLab interface.

RoboDojo uses 16D absolute dual-arm EE values. The trained policy consumes and
predicts physical 20D values using rotation-6D. Model output is first passed
through LeRobot's postprocessor (including action denormalization), then
converted to 16D. This ordering must be preserved if checkpoint preprocessing
changes in the future.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from utils.xvla_ee import ee16_to_xvla20, xvla20_to_ee16
from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.checkpoint_resolver import candidate_checkpoint_roots

POLICY_DIR = Path(__file__).resolve().parent
CHECKPOINTS_DIR = POLICY_DIR / "checkpoints"
CHUNK_SIZE = 30
MODEL_DIM = 20
ROBOT_DIM = 16
CAMERAS = {
    "observation.images.image": ("cam_high", "cam_head", "head_camera", "top_camera"),
    "observation.images.image2": ("cam_left_wrist", "left_camera", "left_wrist", "wrist_left"),
    "observation.images.image3": ("cam_right_wrist", "right_camera", "right_wrist", "wrist_right"),
}


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {value}")
    return device


def _resolve_checkpoint(cfg: dict[str, Any]) -> Path:
    roots = candidate_checkpoint_roots(
        cfg,
        CHECKPOINTS_DIR,
        policy_dir=POLICY_DIR,
        explicit_keys=("checkpoint_path", "pretrained_path", "model_path"),
    )
    checked: list[Path] = []
    for root in roots:
        for candidate in (
            root,
            root / "pretrained_model",
            root / "checkpoints" / "last" / "pretrained_model",
        ):
            checked.append(candidate)
            if candidate.is_dir() and (candidate / "model.safetensors").is_file():
                return candidate.resolve()
    raise FileNotFoundError(
        "Could not find an X-VLA pretrained_model directory. Checked:\n" + "\n".join(f"  - {path}" for path in checked)
    )


def _extract_image(observation: dict[str, Any], names: tuple[str, ...]) -> np.ndarray:
    vision = observation.get("vision")
    if not isinstance(vision, dict):
        raise KeyError("observation must contain a 'vision' mapping")
    for name in names:
        if name not in vision:
            continue
        value = vision[name]
        if isinstance(value, dict):
            value = value.get("color", value.get("rgb"))
        if value is not None:
            return np.asarray(value)
    raise KeyError(f"Missing camera {names}; available={list(vision)}")


def _image_tensor(value: np.ndarray) -> torch.Tensor:
    image = np.asarray(value)
    if image.ndim != 3:
        raise ValueError(f"Expected a 3D camera image, got {image.shape}")
    if image.shape[-1] in (3, 4):
        image = image[..., :3].transpose(2, 0, 1)
    elif image.shape[0] != 3:
        raise ValueError(f"Cannot determine image layout for {image.shape}")
    image = image.astype(np.float32, copy=False)
    if image.size and float(np.nanmax(image)) > 1.0:
        image = image / 255.0
    image = np.clip(image, 0.0, 1.0)
    return torch.from_numpy(np.ascontiguousarray(image)).unsqueeze(0)


def _state16(observation: dict[str, Any]) -> np.ndarray:
    state = observation.get("state")
    if not isinstance(state, dict):
        raise KeyError("observation must contain a 'state' mapping")

    def part(key: str, size: int) -> np.ndarray:
        value = np.asarray(state[key], dtype=np.float32).reshape(-1)
        if value.shape != (size,):
            raise ValueError(f"state[{key!r}] must be ({size},), got {value.shape}")
        return value

    result = np.concatenate(
        (
            part("left_ee_pose", 7),
            part("left_ee_joint_state", 1),
            part("right_ee_pose", 7),
            part("right_ee_joint_state", 1),
        )
    )
    if not np.isfinite(result).all():
        raise ValueError("EE state contains NaN or Inf")
    return result


def _prompt(observation: dict[str, Any], fallback: str | None) -> str:
    for key in ("instruction", "task", "prompt", "language_instruction"):
        value = observation.get(key)
        if isinstance(value, (list, tuple)) and value:
            value = value[0]
        if value is not None and str(value).strip():
            return str(value).strip()
    if fallback and str(fallback).strip():
        return str(fallback).strip()
    raise ValueError("X-VLA requires an instruction or deploy.yml prompt")


def _unpack(chunk: np.ndarray) -> list[dict[str, np.ndarray]]:
    if chunk.shape != (CHUNK_SIZE, ROBOT_DIM):
        raise ValueError(f"Expected ({CHUNK_SIZE},{ROBOT_DIM}), got {chunk.shape}")
    return [
        {
            "left_ee_pose": row[:7].copy(),
            "left_ee_joint_state": row[7:8].copy(),
            "right_ee_pose": row[8:15].copy(),
            "right_ee_joint_state": row[15:16].copy(),
        }
        for row in chunk
    ]


class Model(ModelTemplate):
    def __init__(self, model_cfg):
        self.cfg = dict(model_cfg)
        if self.cfg.get("action_type", "ee") != "ee":
            raise ValueError("xvla_v1 supports only action_type='ee'")
        if int(self.cfg.get("action_chunk", CHUNK_SIZE)) != CHUNK_SIZE:
            raise ValueError("xvla_v1 action_chunk must be 30")
        if int(self.cfg.get("n_step_action", CHUNK_SIZE)) != CHUNK_SIZE:
            raise ValueError("xvla_v1 n_step_action must be 30")

        self.device = _resolve_device(str(self.cfg.get("device", "auto")))
        self.checkpoint_path = _resolve_checkpoint(self.cfg)
        self.default_prompt = self.cfg.get("prompt") or self.cfg.get("task_name")
        self.invert_gripper = bool(self.cfg.get("invert_gripper", True))
        self.log_io = bool(self.cfg.get("log_io", True))
        self._request = 0

        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import get_policy_class, make_pre_post_processors

        config = PreTrainedConfig.from_pretrained(str(self.checkpoint_path))
        config.device = str(self.device)
        self.policy = get_policy_class("xvla").from_pretrained(str(self.checkpoint_path), config=config).to(self.device)
        self.policy.eval()
        self.model = self.policy
        self._validate_policy()
        override = {"device": str(self.device)}
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.policy.config,
            pretrained_path=str(self.checkpoint_path),
            preprocessor_overrides={"device_processor": override},
            postprocessor_overrides={"device_processor": override},
        )
        self._payloads: dict[int, dict[str, Any]] = {}
        self._latest_env_indices = [0]
        print(
            f"[xvla_v1] ready checkpoint={self.checkpoint_path} device={self.device} "
            f"action_chunk={CHUNK_SIZE} n_step_action={CHUNK_SIZE} cameras=3",
            flush=True,
        )

    def _validate_policy(self) -> None:
        cfg = self.policy.config
        if int(getattr(cfg, "chunk_size", -1)) != CHUNK_SIZE:
            raise ValueError(f"Checkpoint chunk_size must be 30, got {cfg.chunk_size}")
        if int(getattr(cfg, "n_action_steps", -1)) != CHUNK_SIZE:
            raise ValueError(f"Checkpoint n_action_steps must be 30, got {cfg.n_action_steps}")
        inputs = getattr(cfg, "input_features", {})
        outputs = getattr(cfg, "output_features", {})
        state_shape = tuple(getattr(inputs.get("observation.state"), "shape", ()))
        action_shape = tuple(getattr(outputs.get("action"), "shape", ()))
        if state_shape != (MODEL_DIM,) or action_shape != (MODEL_DIM,):
            raise ValueError(f"Checkpoint state/action must both be 20D, got {state_shape}/{action_shape}")
        missing = set(CAMERAS).difference(inputs)
        if missing:
            raise ValueError(f"Checkpoint is missing cameras: {sorted(missing)}")

    def _encode(self, observation: dict[str, Any]) -> dict[str, Any]:
        state16 = _state16(observation)
        state20 = ee16_to_xvla20(state16, invert_gripper=self.invert_gripper)
        images = {key: _image_tensor(_extract_image(observation, names)) for key, names in CAMERAS.items()}
        return {
            "state16": state16,
            "state20": state20,
            "images": images,
            "task": _prompt(observation, self.default_prompt),
        }

    def update_obs(self, obs):
        self.update_obs_batch([obs])

    def update_obs_batch(self, obs):
        self._latest_env_indices = [int(item.get("env_idx", index)) for index, item in enumerate(obs)]
        self._payloads = {index: self._encode(item) for index, item in zip(self._latest_env_indices, obs, strict=True)}

    @staticmethod
    def _stack(payloads: list[dict[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {
            "observation.state": torch.from_numpy(np.stack([item["state20"] for item in payloads])),
            "task": [item["task"] for item in payloads],
        }
        for key in CAMERAS:
            result[key] = torch.cat([item["images"][key] for item in payloads])
        return result

    @torch.inference_mode()
    def _infer(self, payloads: list[dict[str, Any]]) -> np.ndarray:
        self._request += 1
        batch = self.preprocessor(self._stack(payloads))
        actions20 = self.policy.predict_action_chunk(batch)
        batch_size, horizon, dimension = actions20.shape
        if (horizon, dimension) != (CHUNK_SIZE, MODEL_DIM):
            raise ValueError(f"Unexpected X-VLA output shape {tuple(actions20.shape)}")
        # Always undo LeRobot preprocessing before changing representations.
        actions20 = self.postprocessor(actions20.reshape(batch_size * horizon, dimension)).reshape(
            batch_size, horizon, dimension
        )
        physical20 = actions20.detach().float().cpu().numpy()
        result = xvla20_to_ee16(
            physical20,
            invert_gripper=self.invert_gripper,
            clip_gripper=True,
        )
        if not np.isfinite(result).all():
            raise FloatingPointError("X-VLA returned NaN or Inf")
        if self.log_io:
            record = {
                "event": "policy_inference",
                "request": self._request,
                "batch_size": batch_size,
                "task": payloads[0]["task"][:200],
                "state16": payloads[0]["state16"].round(5).tolist(),
                "state20": payloads[0]["state20"].round(5).tolist(),
                "action20_shape": list(physical20.shape),
                "action16_shape": list(result.shape),
                "action16_first": result[0, 0].round(5).tolist(),
            }
            print("[xvla_v1][io] " + json.dumps(record, ensure_ascii=False), flush=True)
        return result

    def get_action(self, **kwargs):
        return self.get_action_batch([self._latest_env_indices[0]], **kwargs)[0]

    def get_action_batch(self, env_idx_list=None, **kwargs):
        indices = self._latest_env_indices if env_idx_list is None else [int(i) for i in env_idx_list]
        missing = [index for index in indices if index not in self._payloads]
        if missing:
            raise KeyError(f"Missing observations for env_idx={missing}")
        chunks = self._infer([self._payloads[index] for index in indices])
        return [_unpack(chunk) for chunk in chunks]

    def reset(self):
        reset = getattr(self.policy, "reset", None)
        if callable(reset):
            reset()
        self._payloads = {}
        self._latest_env_indices = [0]
        self._request = 0


def get_model(deploy_cfg):
    return Model(deploy_cfg)
