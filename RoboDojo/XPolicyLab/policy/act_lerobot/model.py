"""Serve a LeRobot ACT policy through the XPolicyLab model interface.

The adapter accepts the observation produced by RoboDojo's Isaac Sim
``ObsManager`` and exposes an EE action chunk consumable by
``policy/act_lerobot/deploy.py``.

The deployment target is the official LeRobot ``pretrained_model`` directory,
including its model weights, policy config, normalizer and unnormalizer.

The temporary converted ``model.ckpt`` remains supported only for backward
compatibility with earlier local smoke tests.

Both paths use the exact 16-dimensional absolute EE convention used by the
training dataset:
    L_xyz, L_quat_wxyz, L_gripper, R_xyz, R_quat_wxyz, R_gripper.
"""

from __future__ import annotations

import inspect
import json
import types
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.checkpoint_resolver import candidate_checkpoint_roots


POLICY_DIR = Path(__file__).resolve().parent
CHECKPOINTS_DIR = POLICY_DIR / "checkpoints"

CAMERA_MAPPING = {
    "observation.images.cam_high": ("cam_head", "cam_high", "head_camera", "top_camera"),
    "observation.images.cam_left_wrist": ("cam_left_wrist", "left_camera", "left_wrist"),
    "observation.images.cam_right_wrist": ("cam_right_wrist", "right_camera", "right_wrist"),
}

ACTION_DIM = 16
ACTION_NAMES = (
    "l_x", "l_y", "l_z", "l_w", "l_wx", "l_wy", "l_wz", "l_g",
    "r_x", "r_y", "r_z", "r_w", "r_wx", "r_wy", "r_wz", "r_g",
)


class _ServerTemporalEnsembler:
    """Online ACT temporal ensemble for one simulator environment.

    At simulator time t, action predictions aligned to t are combined:
    chunk_t[0], chunk_{t-1}[1], ..., chunk_{t-k}[k].
    This mirrors LeRobot's ACTTemporalEnsembler while keeping the policy-server
    response shape at exactly one action.
    """

    def __init__(self, coefficient: float, chunk_size: int):
        self.coefficient = float(coefficient)
        self.chunk_size = int(chunk_size)
        self.actions: torch.Tensor | None = None
        self.counts: torch.Tensor | None = None
        self.last_prediction_count = 0

    def reset(self) -> None:
        self.actions = None
        self.counts = None
        self.last_prediction_count = 0

    def update(self, chunk: torch.Tensor) -> torch.Tensor:
        if chunk.ndim != 3 or chunk.shape[0] != 1:
            raise ValueError(
                "Temporal ensemble expects one [1,T,D] chunk per environment, "
                f"got {tuple(chunk.shape)}"
            )
        if chunk.shape[1] != self.chunk_size:
            raise ValueError(
                f"Temporal ensemble expected chunk_size={self.chunk_size}, "
                f"got {chunk.shape[1]}"
            )

        weights = torch.exp(
            -self.coefficient
            * torch.arange(self.chunk_size, device=chunk.device, dtype=chunk.dtype)
        )
        cumulative_weights = torch.cumsum(weights, dim=0)

        if self.actions is None:
            self.actions = chunk.clone()
            self.counts = torch.ones(
                (self.chunk_size, 1), dtype=torch.long, device=chunk.device
            )
        else:
            assert self.counts is not None
            # self.actions contains predictions from older chunks for the
            # current and future simulator times. The new chunk's last item has
            # no older prediction after the queue is shifted, so update T-1
            # aligned entries and append it separately.
            previous_count = self.counts
            self.actions *= cumulative_weights[previous_count - 1]
            self.actions += chunk[:, :-1] * weights[previous_count]
            self.actions /= cumulative_weights[previous_count]
            self.counts = torch.clamp(previous_count + 1, max=self.chunk_size)
            self.actions = torch.cat((self.actions, chunk[:, -1:]), dim=1)
            self.counts = torch.cat(
                (self.counts, torch.ones_like(self.counts[-1:])), dim=0
            )

        assert self.counts is not None
        self.last_prediction_count = int(self.counts[0, 0].item())
        action = self.actions[:, 0]
        self.actions = self.actions[:, 1:]
        self.counts = self.counts[1:]
        return action


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {value}")
    return device


def _resolve_artifact(model_cfg: dict[str, Any]) -> tuple[str, Path]:
    candidates = candidate_checkpoint_roots(
        model_cfg,
        CHECKPOINTS_DIR,
        policy_dir=POLICY_DIR,
        explicit_keys=("checkpoint_path", "ckpt_path", "pretrained_path"),
    )
    checked: list[Path] = []
    for root in candidates:
        variants = (
            root,
            root / "model.ckpt",
            root / "pretrained_model",
            root / "checkpoints" / "last" / "pretrained_model",
        )
        for candidate in variants:
            checked.append(candidate)
            if candidate.is_file() and candidate.suffix == ".ckpt":
                return "converted_ckpt", candidate.resolve()
            if candidate.is_dir() and (candidate / "model.safetensors").is_file():
                return "pretrained_model", candidate.resolve()
    rendered = "\n  ".join(str(path) for path in checked) or "<none>"
    raise FileNotFoundError(
        "Could not locate a converted model.ckpt or official LeRobot "
        f"pretrained_model directory. Checked:\n  {rendered}"
    )


def _extract_image(observation: dict[str, Any], candidates: tuple[str, ...]) -> np.ndarray:
    vision = observation.get("vision")
    if not isinstance(vision, dict):
        raise KeyError("observation must contain a 'vision' mapping")
    for camera_name in candidates:
        if camera_name not in vision:
            continue
        entry = vision[camera_name]
        if isinstance(entry, dict):
            for field in ("color", "rgb"):
                if field in entry:
                    return np.asarray(entry[field])
        else:
            return np.asarray(entry)
    raise KeyError(f"Missing camera {candidates}; available={list(vision)}")


def _image_to_chw_float(image: np.ndarray) -> torch.Tensor:
    image = np.asarray(image)
    if image.ndim != 3:
        raise ValueError(f"Camera image must be 3D, got {image.shape}")
    if image.shape[-1] in (3, 4):
        image = image[..., :3]
    elif image.shape[0] == 3:
        image = np.moveaxis(image, 0, -1)
    else:
        raise ValueError(f"Camera image must be HWC or CHW RGB, got {image.shape}")

    # Isaac/RoboDojo capture output is already RGB. Do not apply a BGR swap.
    if image.shape[:2] != (480, 640):
        image = cv2.resize(image, (640, 480), interpolation=cv2.INTER_LINEAR)
    if np.issubdtype(image.dtype, np.floating):
        maximum = float(np.nanmax(image)) if image.size else 0.0
        if maximum > 1.0:
            image = image / 255.0
        image = np.clip(image, 0.0, 1.0).astype(np.float32)
    else:
        image = image.astype(np.float32) / 255.0
    return torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))


def _state16(observation: dict[str, Any]) -> np.ndarray:
    state = observation.get("state")
    if not isinstance(state, dict):
        raise KeyError("observation must contain a 'state' mapping")

    def vector(key: str, length: int) -> np.ndarray:
        if key not in state:
            raise KeyError(f"observation['state'] is missing {key!r}")
        value = np.asarray(state[key], dtype=np.float32).reshape(-1)
        if value.shape != (length,):
            raise ValueError(f"state[{key!r}] must have shape ({length},), got {value.shape}")
        if not np.isfinite(value).all():
            raise ValueError(f"state[{key!r}] contains NaN or Inf")
        return value

    left_pose = vector("left_ee_pose", 7)
    left_gripper = vector("left_ee_joint_state", 1)
    right_pose = vector("right_ee_pose", 7)
    right_gripper = vector("right_ee_joint_state", 1)

    for name, pose in (("left_ee_pose", left_pose), ("right_ee_pose", right_pose)):
        quaternion_norm = float(np.linalg.norm(pose[3:7]))
        if not 0.5 <= quaternion_norm <= 1.5:
            raise ValueError(f"{name} quaternion norm is invalid: {quaternion_norm:.6f}")

    result = np.concatenate((left_pose, left_gripper, right_pose, right_gripper))
    assert result.shape == (ACTION_DIM,)
    return result


def _unpack_action_chunk(chunk: np.ndarray) -> list[dict[str, np.ndarray]]:
    chunk = np.asarray(chunk, dtype=np.float32)
    if chunk.ndim != 2 or chunk.shape[1] != ACTION_DIM:
        raise ValueError(f"Expected action chunk [T,16], got {chunk.shape}")
    if not np.isfinite(chunk).all():
        raise ValueError("Predicted action chunk contains NaN or Inf")
    return [
        {
            "left_ee_pose": row[0:7].copy(),
            "left_ee_joint_state": row[7:8].copy(),
            "right_ee_pose": row[8:15].copy(),
            "right_ee_joint_state": row[15:16].copy(),
        }
        for row in chunk
    ]


class Model(ModelTemplate):
    def __init__(self, model_cfg: dict[str, Any]):
        self.cfg = dict(model_cfg)
        if self.cfg.get("action_type", "ee") != "ee":
            raise ValueError("act_lerobot only supports --action-type ee")

        self.device = _resolve_device(str(self.cfg.get("device", "auto")))
        self.artifact_type, self.artifact_path = _resolve_artifact(self.cfg)
        self.preprocessor = None
        self.postprocessor = None
        self.stats: dict[str, torch.Tensor] = {}
        self.log_io = bool(self.cfg.get("log_io", True))
        self._request_index = 0
        self.action_limiter_enabled = bool(
            self.cfg.get("action_limiter_enabled", False)
        )
        self.max_translation_m = float(
            self.cfg.get("max_translation_m", 0.0)
        )
        self.max_rotation_deg = float(
            self.cfg.get("max_rotation_deg", 0.0)
        )
        if self.action_limiter_enabled and (
            self.max_translation_m <= 0.0 or self.max_rotation_deg <= 0.0
        ):
            raise ValueError(
                "action_limiter_enabled requires positive max_translation_m "
                "and max_rotation_deg"
            )

        if self.artifact_type == "converted_ckpt":
            self.policy = self._load_converted_ckpt(self.artifact_path)
        else:
            self.policy = self._load_pretrained_model(self.artifact_path)

        config_steps = int(getattr(self.policy.config, "n_action_steps", 10))
        requested_steps = self.cfg.get("actions_per_chunk")
        self.actions_per_chunk = config_steps if requested_steps is None else int(requested_steps)
        chunk_size = int(getattr(self.policy.config, "chunk_size", self.actions_per_chunk))
        if not 1 <= self.actions_per_chunk <= chunk_size:
            raise ValueError(
                f"actions_per_chunk must be within [1,{chunk_size}], got {self.actions_per_chunk}"
            )
        requested_ensemble = self.cfg.get("temporal_ensemble_coeff")
        self.temporal_ensemble_coeff = (
            None if requested_ensemble is None else float(requested_ensemble)
        )
        if (
            self.temporal_ensemble_coeff is not None
            and not np.isfinite(self.temporal_ensemble_coeff)
        ):
            raise ValueError("temporal_ensemble_coeff must be finite or null")
        if self.temporal_ensemble_coeff is not None and self.actions_per_chunk != 1:
            raise ValueError(
                "server-side temporal ensemble requires actions_per_chunk=1 "
                "because it consumes exactly one aligned action per observation"
            )
        self._temporal_ensemblers: dict[int, _ServerTemporalEnsembler] = {}
        self._chunk_size = chunk_size
        self.temporal_ensemble_horizon: int | None = None
        if self.temporal_ensemble_coeff is not None:
            requested_horizon = self.cfg.get(
                "temporal_ensemble_horizon", chunk_size
            )
            self.temporal_ensemble_horizon = int(requested_horizon)
            if not 1 <= self.temporal_ensemble_horizon <= self._chunk_size:
                raise ValueError(
                    "temporal_ensemble_horizon must be within "
                    f"[1,{self._chunk_size}], got {self.temporal_ensemble_horizon}"
                )
        requested_queue_size = self.cfg.get("action_queue_size")
        self.action_queue_size = (
            None if requested_queue_size is None else int(requested_queue_size)
        )
        if self.action_queue_size is not None:
            if not 1 <= self.action_queue_size <= self._chunk_size:
                raise ValueError(
                    f"action_queue_size must be within [1,{self._chunk_size}], "
                    f"got {self.action_queue_size}"
                )
            if self.actions_per_chunk != 1:
                raise ValueError(
                    "server-side action queue requires actions_per_chunk=1 "
                    "because each client response contains one queued action"
                )
            if self.temporal_ensemble_coeff is not None:
                raise ValueError(
                    "action_queue_size and temporal_ensemble_coeff are mutually "
                    "exclusive execution modes"
                )
        self._action_queues: dict[int, dict[str, Any]] = {}

        input_features = getattr(self.policy.config, "input_features", {})
        output_features = getattr(self.policy.config, "output_features", {})
        expected_inputs = {"observation.state", *CAMERA_MAPPING}
        missing_inputs = expected_inputs.difference(input_features)
        if missing_inputs:
            raise ValueError(f"Checkpoint is missing ACT inputs: {sorted(missing_inputs)}")
        action_feature = output_features.get("action")
        action_shape = tuple(getattr(action_feature, "shape", ()))
        if action_shape != (ACTION_DIM,):
            raise ValueError(f"Checkpoint action shape must be (16,), got {action_shape}")

        self.policy.eval()
        self.model = self.policy
        self._latest_observation: dict[str, Any] | None = None
        self._latest_by_env: dict[int, dict[str, Any]] = {}
        self._latest_env_indices = [0]
        print(
            f"[act_lerobot] ready artifact={self.artifact_type} path={self.artifact_path} "
            f"device={self.device} execute_steps={self.actions_per_chunk} "
            f"temporal_ensemble_coeff={self.temporal_ensemble_coeff} "
            f"temporal_ensemble_horizon={self.temporal_ensemble_horizon} "
            f"action_queue_size={self.action_queue_size} "
            f"action_limiter={self.action_limiter_enabled} "
            f"max_translation_m={self.max_translation_m} "
            f"max_rotation_deg={self.max_rotation_deg}",
            flush=True,
        )

    @staticmethod
    def _image_summary(image: np.ndarray) -> dict[str, Any]:
        value = np.asarray(image)
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "min": float(np.min(value)),
            "max": float(np.max(value)),
            "mean": float(np.mean(value)),
        }

    def _log_observation(self, observation: dict[str, Any]) -> None:
        if not self.log_io:
            return
        state = _state16(observation)
        images = {}
        for target_key, candidates in CAMERA_MAPPING.items():
            raw = _extract_image(observation, candidates)
            model_input = _image_to_chw_float(raw).unsqueeze(0)
            images[target_key] = {
                "client": self._image_summary(raw),
                "model_input": {
                    "shape": list(model_input.shape),
                    "dtype": str(model_input.dtype),
                    "min": float(model_input.min()),
                    "max": float(model_input.max()),
                    "mean": float(model_input.mean()),
                },
            }
        summary = {
            "event": "client_observation",
            "request": self._request_index,
            "env_idx": int(observation.get("env_idx", 0)),
            "instruction": str(observation.get("instruction", ""))[:200],
            "state16": state.tolist(),
            "images": images,
        }
        print("[act_lerobot][io] " + json.dumps(summary, ensure_ascii=False), flush=True)

    def _log_actions(
        self,
        chunk: np.ndarray,
        *,
        raw_chunk: np.ndarray | None = None,
        limiter_diagnostics: dict[str, Any] | None = None,
        temporal_ensemble_diagnostics: dict[str, Any] | None = None,
        action_queue_diagnostics: dict[str, Any] | None = None,
    ) -> None:
        if not self.log_io:
            return
        summary = {
            "event": "server_actions",
            "request": self._request_index,
            "shape": list(chunk.shape),
            "action_names": list(ACTION_NAMES),
            "actions": chunk.tolist(),
            "min": np.min(chunk, axis=0).tolist(),
            "max": np.max(chunk, axis=0).tolist(),
        }
        if raw_chunk is not None:
            summary["postprocess_adjustments"] = {
                "quaternion_normalized": True,
                "gripper_clipped_to": [0.0, 1.0],
                "raw_left_quaternion_norm": np.linalg.norm(
                    raw_chunk[:, 3:7], axis=1
                ).tolist(),
                "raw_right_quaternion_norm": np.linalg.norm(
                    raw_chunk[:, 11:15], axis=1
                ).tolist(),
                "raw_left_gripper": raw_chunk[:, 7].tolist(),
                "raw_right_gripper": raw_chunk[:, 15].tolist(),
            }
        if limiter_diagnostics is not None:
            summary["action_limiter"] = limiter_diagnostics
        if temporal_ensemble_diagnostics is not None:
            summary["temporal_ensemble"] = temporal_ensemble_diagnostics
        if action_queue_diagnostics is not None:
            summary["action_queue"] = action_queue_diagnostics
        print("[act_lerobot][io] " + json.dumps(summary, ensure_ascii=False), flush=True)

    @staticmethod
    def _sanitize_action_chunk(chunk: np.ndarray) -> np.ndarray:
        """Return controller-safe EE actions without mutating raw predictions.

        ACT is a continuous regressor, so quaternion norms and gripper values
        can drift slightly outside their mathematical/physical ranges.
        """
        result = np.asarray(chunk, dtype=np.float32).copy()
        for quaternion_slice in (slice(3, 7), slice(11, 15)):
            quaternion = result[:, quaternion_slice]
            norm = np.linalg.norm(quaternion, axis=1, keepdims=True)
            if np.any(norm < 1e-8):
                raise ValueError("Predicted action contains a zero-norm quaternion")
            result[:, quaternion_slice] = quaternion / norm
        result[:, 7] = np.clip(result[:, 7], 0.0, 1.0)
        result[:, 15] = np.clip(result[:, 15], 0.0, 1.0)
        return result

    @staticmethod
    def _quaternion_angle(q0: np.ndarray, q1: np.ndarray) -> float:
        dot = float(np.clip(np.abs(np.dot(q0, q1)), 0.0, 1.0))
        return 2.0 * float(np.arccos(dot))

    @staticmethod
    def _quaternion_slerp(
        q0: np.ndarray,
        q1: np.ndarray,
        ratio: float,
    ) -> np.ndarray:
        q0 = q0 / np.linalg.norm(q0)
        q1 = q1 / np.linalg.norm(q1)
        dot = float(np.dot(q0, q1))
        if dot < 0.0:
            q1 = -q1
            dot = -dot
        dot = float(np.clip(dot, -1.0, 1.0))
        if dot > 0.9995:
            result = q0 + ratio * (q1 - q0)
            return result / np.linalg.norm(result)
        theta = float(np.arccos(dot))
        sin_theta = float(np.sin(theta))
        result = (
            np.sin((1.0 - ratio) * theta) / sin_theta * q0
            + np.sin(ratio * theta) / sin_theta * q1
        )
        return result / np.linalg.norm(result)

    def _limit_action_chunk(
        self,
        chunk: np.ndarray,
        observation: dict[str, Any],
    ) -> tuple[np.ndarray, dict[str, Any] | None]:
        """Rate-limit absolute EE targets relative to the latest real state.

        Each later action in a chunk is limited relative to the preceding
        limited action, matching the sequence the client will execute.
        """
        if not self.action_limiter_enabled:
            return chunk, None

        result = np.asarray(chunk, dtype=np.float32).copy()
        state = _state16(observation)
        previous = {
            "left": state[0:7].copy(),
            "right": state[8:15].copy(),
        }
        max_rotation_rad = float(np.deg2rad(self.max_rotation_deg))
        records = []

        for step, row in enumerate(result):
            step_record = {"step": step, "arms": {}}
            for arm, pose_slice in (
                ("left", slice(0, 7)),
                ("right", slice(8, 15)),
            ):
                target = row[pose_slice].copy()
                reference = previous[arm]

                delta = target[:3] - reference[:3]
                translation = float(np.linalg.norm(delta))
                translation_ratio = 1.0
                if translation > self.max_translation_m:
                    translation_ratio = self.max_translation_m / translation
                    target[:3] = reference[:3] + delta * translation_ratio

                rotation = self._quaternion_angle(reference[3:7], target[3:7])
                rotation_ratio = 1.0
                if rotation > max_rotation_rad:
                    rotation_ratio = max_rotation_rad / rotation
                    target[3:7] = self._quaternion_slerp(
                        reference[3:7],
                        target[3:7],
                        rotation_ratio,
                    )

                row[pose_slice] = target
                previous[arm] = target
                step_record["arms"][arm] = {
                    "raw_translation_m": translation,
                    "raw_rotation_deg": float(np.rad2deg(rotation)),
                    "translation_clipped": translation_ratio < 1.0,
                    "rotation_clipped": rotation_ratio < 1.0,
                    "translation_ratio": translation_ratio,
                    "rotation_ratio": rotation_ratio,
                }
            records.append(step_record)

        return result, {
            "enabled": True,
            "max_translation_m": self.max_translation_m,
            "max_rotation_deg": self.max_rotation_deg,
            "steps": records,
        }

    def _load_converted_ckpt(self, path: Path):
        # LeRobot 0.4.x probes torch.xpu even on Macs with older Torch builds.
        if not hasattr(torch, "xpu"):
            torch.xpu = types.SimpleNamespace(is_available=lambda: False)

        from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
        from lerobot.policies.act.configuration_act import ACTConfig
        from lerobot.policies.act.modeling_act import ACTPolicy

        saved = torch.load(path, map_location="cpu", weights_only=False)
        if "config" not in saved or "state_dict" not in saved:
            raise ValueError("Converted checkpoint must contain config and state_dict")
        raw_config = dict(saved["config"])
        valid_fields = set(inspect.signature(ACTConfig).parameters)
        config = {key: value for key, value in raw_config.items() if key in valid_fields}
        config["device"] = str(self.device)
        config["pretrained_backbone_weights"] = None
        config["input_features"] = {
            key: PolicyFeature(type=FeatureType[value["type"]], shape=tuple(value["shape"]))
            for key, value in config["input_features"].items()
        }
        config["output_features"] = {
            key: PolicyFeature(type=FeatureType[value["type"]], shape=tuple(value["shape"]))
            for key, value in config["output_features"].items()
        }
        config["normalization_mapping"] = {
            FeatureType[key]: NormalizationMode(value)
            for key, value in config["normalization_mapping"].items()
        }

        policy = ACTPolicy(ACTConfig(**config))
        model_state = {
            key: value for key, value in saved["state_dict"].items()
            if key.startswith("model.")
        }
        policy.load_state_dict(model_state, strict=True)
        policy.to(self.device)
        self.stats = {
            key: value.detach().to(self.device)
            for key, value in saved["state_dict"].items()
            if not key.startswith("model.") and isinstance(value, torch.Tensor)
        }
        for required in (
            "observation.state.mean", "observation.state.std",
            "action.mean", "action.std",
            *(
                name
                for camera in CAMERA_MAPPING
                for name in (f"{camera}.mean", f"{camera}.std")
            ),
        ):
            if required not in self.stats:
                raise KeyError(f"Converted checkpoint is missing statistic {required!r}")
        return policy

    def _load_pretrained_model(self, path: Path):
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import get_policy_class, make_pre_post_processors

        config = PreTrainedConfig.from_pretrained(path)
        config.pretrained_path = path
        policy_class = get_policy_class("act")
        policy = policy_class.from_pretrained(path, config=config)
        policy.to(self.device)
        device_override = {"device": str(self.device)}
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy.config,
            pretrained_path=path,
            preprocessor_overrides={"device_processor": device_override},
            postprocessor_overrides={"device_processor": device_override},
        )
        return policy

    def _raw_batch(self, observation: dict[str, Any]) -> dict[str, torch.Tensor]:
        batch: dict[str, torch.Tensor] = {
            "observation.state": torch.from_numpy(_state16(observation)).unsqueeze(0)
        }
        for target_key, candidates in CAMERA_MAPPING.items():
            batch[target_key] = _image_to_chw_float(
                _extract_image(observation, candidates)
            ).unsqueeze(0)
        return batch

    def _normalize_converted_batch(
        self, batch: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        result = {}
        for key, value in batch.items():
            value = value.to(self.device)
            mean = self.stats[f"{key}.mean"]
            std = self.stats[f"{key}.std"].clamp_min(1e-8)
            result[key] = (value - mean) / std
        return result

    def _postprocess_converted(self, chunk: torch.Tensor) -> torch.Tensor:
        return (
            chunk * self.stats["action.std"].clamp_min(1e-8)
            + self.stats["action.mean"]
        )

    @torch.inference_mode()
    def _infer_observation(self, observation: dict[str, Any]) -> np.ndarray:
        self._request_index += 1
        self._log_observation(observation)
        env_idx = int(observation.get("env_idx", 0))

        # Queue mode runs the model only when an environment's queue is empty.
        # Every client request still receives exactly one action, and the
        # optional limiter below always uses the latest real observation.
        queue = self._action_queues.get(env_idx)
        if self.action_queue_size is not None and queue is not None:
            queued_actions = queue["actions"]
            action_index = int(queue["next_index"])
            raw_result = queued_actions[action_index : action_index + 1]
            queue["next_index"] = action_index + 1
            remaining = len(queued_actions) - int(queue["next_index"])
            action_queue_diagnostics = {
                "enabled": True,
                "queue_size": self.action_queue_size,
                "plan_request": int(queue["plan_request"]),
                "action_index": action_index,
                "remaining_actions": remaining,
                "new_plan": False,
            }
            if remaining == 0:
                del self._action_queues[env_idx]
            result = self._sanitize_action_chunk(raw_result)
            result, limiter_diagnostics = self._limit_action_chunk(
                result, observation
            )
            self._log_actions(
                result,
                raw_chunk=raw_result,
                limiter_diagnostics=limiter_diagnostics,
                action_queue_diagnostics=action_queue_diagnostics,
            )
            return result

        batch = self._raw_batch(observation)
        if self.artifact_type == "converted_ckpt":
            processed = self._normalize_converted_batch(batch)
            predicted = self.policy.predict_action_chunk(processed)
            predicted = self._postprocess_converted(predicted)
        else:
            processed = self.preprocessor(batch)
            predicted = self.policy.predict_action_chunk(processed)
            batch_size, chunk_size, action_dim = predicted.shape
            predicted = self.postprocessor(
                predicted.reshape(batch_size * chunk_size, action_dim)
            ).reshape(batch_size, chunk_size, action_dim)

        temporal_ensemble_diagnostics = None
        action_queue_diagnostics = None
        if self.action_queue_size is not None:
            planned_actions = (
                predicted[0, : self.action_queue_size]
                .detach()
                .float()
                .cpu()
                .numpy()
            )
            raw_result = planned_actions[0:1]
            remaining = len(planned_actions) - 1
            if remaining > 0:
                self._action_queues[env_idx] = {
                    "actions": planned_actions,
                    "next_index": 1,
                    "plan_request": self._request_index,
                }
            action_queue_diagnostics = {
                "enabled": True,
                "queue_size": self.action_queue_size,
                "plan_request": self._request_index,
                "action_index": 0,
                "remaining_actions": remaining,
                "new_plan": True,
            }
        elif self.temporal_ensemble_coeff is not None:
            assert self.temporal_ensemble_horizon is not None
            pre_ensemble_action = (
                predicted[0, 0].detach().float().cpu().numpy()
            )
            ensembler = self._temporal_ensemblers.get(env_idx)
            if ensembler is None:
                ensembler = _ServerTemporalEnsembler(
                    self.temporal_ensemble_coeff,
                    self.temporal_ensemble_horizon,
                )
                self._temporal_ensemblers[env_idx] = ensembler
            predicted = ensembler.update(
                predicted[:, : self.temporal_ensemble_horizon]
            ).unsqueeze(1)
            temporal_ensemble_diagnostics = {
                "enabled": True,
                "coefficient": self.temporal_ensemble_coeff,
                "aligned_prediction_count": ensembler.last_prediction_count,
                "horizon": self.temporal_ensemble_horizon,
                "model_chunk_size": self._chunk_size,
                "pre_ensemble_action": pre_ensemble_action.tolist(),
            }
        else:
            predicted = predicted[:, : self.actions_per_chunk]
        if self.action_queue_size is None:
            predicted = predicted[0]
            raw_result = predicted.detach().float().cpu().numpy()
        result = self._sanitize_action_chunk(raw_result)
        result, limiter_diagnostics = self._limit_action_chunk(result, observation)
        self._log_actions(
            result,
            raw_chunk=raw_result,
            limiter_diagnostics=limiter_diagnostics,
            temporal_ensemble_diagnostics=temporal_ensemble_diagnostics,
            action_queue_diagnostics=action_queue_diagnostics,
        )
        return result

    def update_obs(self, obs):
        self.update_obs_batch([obs])

    def update_obs_batch(self, obs_list):
        # Persist the positional fallback into each buffered observation so
        # inference and temporal-ensemble lookup use the same environment ID.
        resolved_observations = []
        for index, obs in enumerate(obs_list):
            env_idx = int(obs.get("env_idx", index))
            if obs.get("env_idx") == env_idx:
                resolved = obs
            else:
                resolved = dict(obs)
                resolved["env_idx"] = env_idx
            resolved_observations.append(resolved)
        self._latest_env_indices = [
            int(obs["env_idx"]) for obs in resolved_observations
        ]
        self._latest_by_env = {
            env_idx: obs
            for env_idx, obs in zip(self._latest_env_indices, resolved_observations)
        }
        self._latest_observation = (
            resolved_observations[0] if resolved_observations else None
        )

    def get_action(self, **kwargs):
        if self._latest_observation is None:
            raise AssertionError("update_obs must be called before get_action")
        return _unpack_action_chunk(self._infer_observation(self._latest_observation))

    def get_action_batch(self, env_idx_list=None, **kwargs):
        if env_idx_list is None:
            env_idx_list = self._latest_env_indices
        return [
            _unpack_action_chunk(self._infer_observation(self._latest_by_env[int(env_idx)]))
            for env_idx in env_idx_list
        ]

    def reset(self):
        reset = getattr(self.policy, "reset", None)
        if callable(reset):
            reset()
        self._latest_observation = None
        self._latest_by_env = {}
        self._latest_env_indices = [0]
        self._temporal_ensemblers = {}
        self._action_queues = {}
        self._request_index = 0
