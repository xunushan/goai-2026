"""Self-contained ARX-X5 X-VLA adapter for the XPolicyLab model interface.

This module intentionally does not import ``policy.X_VLA``.  That adapter is a
legacy policy and may be removed without affecting ``xvla_lerobot``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation, Slerp

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.checkpoint_resolver import candidate_checkpoint_roots

from .xvla.models.modeling_xvla import XVLA
from .xvla.models.processing_xvla import XVLAProcessor


POLICY_DIR = Path(__file__).resolve().parent
CHECKPOINTS_DIR = POLICY_DIR / "checkpoints"


CAMERA_CANDIDATES = (
    ("cam_high", "cam_head", "head_camera", "top_camera"),
    ("cam_left_wrist", "left_camera", "left_wrist"),
    ("cam_right_wrist", "right_camera", "right_wrist"),
)
ACTION_NAMES = (
    "l_x", "l_y", "l_z", "l_w", "l_wx", "l_wy", "l_wz", "l_g",
    "r_x", "r_y", "r_z", "r_w", "r_wx", "r_wy", "r_wz", "r_g",
)


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {value}")
    return device


def _checkpoint_candidates(
    model_cfg: dict[str, Any], *, explicit_keys: tuple[str, ...]
) -> list[Path]:
    roots = candidate_checkpoint_roots(
        model_cfg,
        CHECKPOINTS_DIR,
        policy_dir=POLICY_DIR,
        explicit_keys=explicit_keys,
        resolve=True,
    )
    candidates: list[Path] = []
    for root in roots:
        for candidate in (
            root,
            root / "processor",
            root / "model",
            root / "base",
            root / "base_model",
            root / "checkpoint",
        ):
            if candidate not in candidates:
                candidates.append(candidate)
    return candidates


def _resolve_local_or_hub_reference(value: str, relative_to: Path) -> str:
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = relative_to / candidate
    if candidate.exists():
        return str(candidate.resolve())
    return value


def _find_artifact_dir(
    candidates: list[Path], marker: str, artifact_name: str
) -> Path:
    for candidate in candidates:
        if (candidate / marker).is_file():
            return candidate
    searched = ", ".join(str(path) for path in candidates) or "<none>"
    raise FileNotFoundError(
        f"Could not find {artifact_name} ({marker}). Searched: {searched}"
    )


def _first_artifact_dir(candidates: list[Path], marker: str) -> Path | None:
    return next(
        (candidate for candidate in candidates if (candidate / marker).is_file()),
        None,
    )


def _extract_image(observation: dict[str, Any], candidates: tuple[str, ...]) -> Any:
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
                    return entry[field]
        else:
            return entry
    raise KeyError(f"Missing camera {candidates}; available={list(vision)}")


def _ensure_hwc_uint8(image: Any) -> np.ndarray:
    if isinstance(image, (bytes, bytearray, memoryview)):
        image = np.frombuffer(bytes(image), dtype=np.uint8)
    image = np.asarray(image)
    if image.ndim == 1 and image.dtype == np.uint8:
        decoded = cv2.imdecode(image, cv2.IMREAD_COLOR)
        if decoded is None:
            raise ValueError("Failed to decode compressed camera image")
        image = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
    if image.ndim != 3:
        raise ValueError(f"Expected a 3D camera image, got {image.shape}")
    if image.shape[0] in (1, 3) and image.shape[-1] not in (1, 3):
        image = np.moveaxis(image, 0, -1)
    if image.shape[-1] not in (1, 3):
        raise ValueError(f"Expected an HWC RGB/gray image, got {image.shape}")
    if np.issubdtype(image.dtype, np.floating):
        maximum = float(np.nanmax(image)) if image.size else 0.0
        scale = 255.0 if maximum <= 1.0 else 1.0
        image = np.clip(image * scale, 0.0, 255.0).astype(np.uint8)
    elif image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image)


def _normalize_prompt(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="ignore")
    elif isinstance(value, np.ndarray) and value.ndim == 0:
        value = value.item()
    elif isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (list, tuple)):
        return next((item for entry in value if (item := _normalize_prompt(entry))), None)
    text = str(value).strip()
    return text or None


def _resolve_prompt(
    observation: dict[str, Any],
    default_prompt: str,
    task_prompt_map: dict[str, str] | None = None,
) -> str:
    mapping = task_prompt_map or {}
    for key in ("prompt", "instruction", "task", "language_instruction"):
        prompt = _normalize_prompt(observation.get(key))
        if prompt is not None:
            return mapping.get(prompt, prompt)
    fallback = _normalize_prompt(default_prompt)
    if fallback is None:
        raise ValueError("No valid prompt found in observation or deployment config")
    return mapping.get(fallback, fallback)


def _quat_wxyz_to_rotation6d(quaternion: np.ndarray) -> np.ndarray:
    quaternion = _normalize_wxyz(np.asarray(quaternion, dtype=np.float32))
    xyzw = quaternion[..., [1, 2, 3, 0]]
    matrix = Rotation.from_quat(xyzw).as_matrix()
    return matrix[..., :, :2].reshape(quaternion.shape[:-1] + (6,)).astype(np.float32)


def _rotation6d_to_quat_wxyz(rotation6d: np.ndarray) -> np.ndarray:
    rotation6d = np.asarray(rotation6d, dtype=np.float32)
    if rotation6d.shape[-1] != 6:
        raise ValueError(f"Expected rotation-6D last dimension 6, got {rotation6d.shape}")
    first = rotation6d[..., 0:5:2]
    second = rotation6d[..., 1:6:2]
    first_norm = np.linalg.norm(first, axis=-1, keepdims=True)
    if np.any(first_norm < 1e-8):
        raise ValueError("Predicted rotation-6D has a degenerate first axis")
    axis1 = first / first_norm
    orthogonal = second - np.sum(axis1 * second, axis=-1, keepdims=True) * axis1
    second_norm = np.linalg.norm(orthogonal, axis=-1, keepdims=True)
    if np.any(second_norm < 1e-8):
        raise ValueError("Predicted rotation-6D has collinear axes")
    axis2 = orthogonal / second_norm
    matrix = np.stack((axis1, axis2, np.cross(axis1, axis2)), axis=-1)
    xyzw = Rotation.from_matrix(matrix).as_quat()
    return xyzw[..., [3, 0, 1, 2]].astype(np.float32)


def _build_proprio20(observation: dict[str, Any]) -> np.ndarray:
    state16 = _state16(observation)
    return np.concatenate(
        (
            state16[0:3],
            _quat_wxyz_to_rotation6d(state16[3:7]),
            state16[7:8],
            state16[8:11],
            _quat_wxyz_to_rotation6d(state16[11:15]),
            state16[15:16],
        )
    ).astype(np.float32)


def _state16(observation: dict[str, Any]) -> np.ndarray:
    state = observation.get("state")
    if not isinstance(state, dict):
        raise KeyError("observation must contain a state mapping")
    def vector(key: str, size: int) -> np.ndarray:
        if key not in state:
            raise KeyError(f"observation['state'] is missing {key!r}")
        value = np.asarray(state[key], dtype=np.float32).reshape(-1)
        if value.shape != (size,):
            raise ValueError(f"state[{key!r}] must have shape ({size},), got {value.shape}")
        return value

    values = np.concatenate(
        [vector("left_ee_pose", 7), vector("left_ee_joint_state", 1),
         vector("right_ee_pose", 7), vector("right_ee_joint_state", 1)]
    )
    if values.shape != (16,) or not np.isfinite(values).all():
        raise ValueError(f"Expected finite ARX-X5 state [16], got {values.shape}.")
    return values


def _encode_observation(observation: dict[str, Any], default_prompt: str, task_prompt_map: dict) -> dict:
    images = [
        _ensure_hwc_uint8(_extract_image(observation, candidates))
        for candidates in CAMERA_CANDIDATES
    ]
    return {
        "images": images,
        "proprio": _build_proprio20(observation),
        "prompt": _resolve_prompt(observation, default_prompt, task_prompt_map),
        "output_format": "xpolicylab",
    }


def _normalize_wxyz(quaternion: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(quaternion, axis=-1, keepdims=True)
    if np.any(norm < 1e-8):
        raise ValueError("Predicted quaternion has zero norm.")
    return quaternion / norm


def _action20_to_action16(action: np.ndarray) -> np.ndarray:
    action = np.asarray(action, dtype=np.float32)
    left_quat = _normalize_wxyz(_rotation6d_to_quat_wxyz(action[:, 3:9]))
    right_quat = _normalize_wxyz(_rotation6d_to_quat_wxyz(action[:, 13:19]))
    return np.concatenate(
        [
            action[:, 0:3], left_quat, np.clip(action[:, 9:10], 0.0, 1.0),
            action[:, 10:13], right_quat, np.clip(action[:, 19:20], 0.0, 1.0),
        ],
        axis=-1,
    ).astype(np.float32)


def _slerp_wxyz(source_time: np.ndarray, quaternion: np.ndarray, target_time: np.ndarray) -> np.ndarray:
    quaternion = _normalize_wxyz(quaternion)
    xyzw = quaternion[:, [1, 2, 3, 0]]
    result = Slerp(source_time, Rotation.from_quat(xyzw))(target_time).as_quat()
    return result[:, [3, 0, 1, 2]].astype(np.float32)


def resample_one_second_chunk(
    current_state16: np.ndarray,
    predicted_action20: np.ndarray,
    *,
    control_hz: int = 25,
) -> np.ndarray:
    """Convert 30 anchors over one second into `control_hz` EE commands."""
    predicted_action20 = np.asarray(predicted_action20, dtype=np.float32)
    if predicted_action20.shape != (30, 20):
        raise ValueError(f"Expected predicted anchors [30,20], got {predicted_action20.shape}.")
    if control_hz <= 0:
        raise ValueError("control_hz must be positive.")
    anchor16 = np.concatenate(
        [np.asarray(current_state16, dtype=np.float32)[None], _action20_to_action16(predicted_action20)],
        axis=0,
    )
    source_time = np.linspace(0.0, 1.0, 31, dtype=np.float64)
    target_time = np.arange(1, control_hz + 1, dtype=np.float64) / control_hz
    output = np.empty((control_hz, 16), dtype=np.float32)
    linear_indices = (0, 1, 2, 7, 8, 9, 10, 15)
    for index in linear_indices:
        output[:, index] = np.interp(target_time, source_time, anchor16[:, index])
    output[:, 3:7] = _slerp_wxyz(source_time, anchor16[:, 3:7], target_time)
    output[:, 11:15] = _slerp_wxyz(source_time, anchor16[:, 11:15], target_time)
    output[:, (7, 15)] = np.clip(output[:, (7, 15)], 0.0, 1.0)
    return output


def _unpack_action16(chunk: np.ndarray) -> list[dict[str, np.ndarray]]:
    return [
        {
            "left_ee_pose": row[0:7].copy(),
            "left_ee_joint_state": row[7:8].copy(),
            "right_ee_pose": row[8:15].copy(),
            "right_ee_joint_state": row[15:16].copy(),
        }
        for row in np.asarray(chunk, dtype=np.float32)
    ]


def _array_summary(value: np.ndarray) -> dict[str, Any]:
    value = np.asarray(value)
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "min": float(value.min()),
        "max": float(value.max()),
        "mean": float(value.mean()),
    }


class Model(ModelTemplate):
    """Three-camera, continuous-gripper X-VLA adapter for ARX-X5."""

    def __init__(self, model_cfg):
        cfg = dict(model_cfg)
        cfg["action_type"] = "ee"
        cfg["domain_id"] = 6
        self.model_cfg = cfg
        self.task_name = str(cfg.get("task_name") or "default_task")
        self.default_prompt = str(cfg.get("prompt") or self.task_name)
        self.task_prompt_map = dict(cfg.get("task_prompt_map") or {})
        self.device = _resolve_device(str(cfg.get("device", "cuda")))
        self.denoise_steps = int(cfg.get("steps", 10))
        if self.denoise_steps < 1:
            raise ValueError("steps must be at least 1")
        self.control_hz = int(cfg.get("control_hz", 25))
        self.control_steps_per_chunk = int(cfg.get("actions_per_chunk", 5))
        if not 1 <= self.control_steps_per_chunk <= self.control_hz:
            raise ValueError("actions_per_chunk must be within [1, control_hz].")
        self.log_io = bool(cfg.get("log_io", True))
        self.log_full_actions = bool(cfg.get("log_full_actions", False))
        model_candidates = _checkpoint_candidates(
            cfg, explicit_keys=("model_path", "checkpoint_path")
        )
        adapter_dir = next(
            (
                candidate
                for candidate in model_candidates
                if (candidate / "adapter_config.json").is_file()
            ),
            None,
        )
        base_reference: str | None = None
        if adapter_dir is not None:
            adapter_config = json.loads(
                (adapter_dir / "adapter_config.json").read_text(encoding="utf-8")
            )
            configured_base = adapter_config.get("base_model_name_or_path")
            if not configured_base:
                raise ValueError(
                    f"PEFT checkpoint is missing base_model_name_or_path: {adapter_dir}"
                )
            base_reference = _resolve_local_or_hub_reference(
                str(configured_base), adapter_dir
            )

        processor_candidates = _checkpoint_candidates(
            cfg,
            explicit_keys=("processor_path", "model_path", "checkpoint_path"),
        )
        processor_dir = _first_artifact_dir(
            processor_candidates, "preprocessor_config.json"
        )
        processor_reference = str(processor_dir) if processor_dir else base_reference
        if processor_reference is None:
            _find_artifact_dir(
                processor_candidates, "preprocessor_config.json", "X-VLA processor"
            )
            raise AssertionError("unreachable")
        self.processor = XVLAProcessor.from_pretrained(processor_reference)

        if adapter_dir is None:
            model_dir = _find_artifact_dir(
                model_candidates, "config.json", "X-VLA model"
            )
            model = XVLA.from_pretrained(
                str(model_dir), trust_remote_code=True, torch_dtype=torch.float32
            )
        else:
            assert base_reference is not None
            model = XVLA.from_pretrained(
                base_reference, trust_remote_code=True, torch_dtype=torch.float32
            )
            try:
                from peft import PeftModel
            except ImportError as error:
                raise ImportError(
                    "Loading an X-VLA PEFT checkpoint requires the 'peft' package"
                ) from error
            model = PeftModel.from_pretrained(
                model, str(adapter_dir), torch_dtype=torch.float32
            )
        self.model = model.to(self.device).to(torch.float32)
        self.model.eval()
        self.model_chunk_size = int(getattr(self.model, "num_actions", 0))
        if self.model_chunk_size != 30:
            raise ValueError(
                "xvla_lerobot deployment requires exactly 30 anchors; "
                f"got num_actions={self.model_chunk_size}."
            )
        self._request_index = 0
        self._latest_env_idx_list: list[int] = [0]
        self._latest_by_env: dict[int, dict[str, Any]] = {}
        self._raw_by_env: dict[int, dict[str, Any]] = {}
        self.observation_window: list[dict[str, Any]] | None = None
        if self.model.action_mode != "arx_ee6d":
            raise ValueError(
                "xvla_lerobot deployment requires action_mode='arx_ee6d'; "
                f"got {self.model.action_mode!r}."
            )
        print(
            "[xvla_lerobot][startup] "
            + json.dumps(
                {
                    "domain_id": 6,
                    "action_mode": self.model.action_mode,
                    "model_anchors": self.model_chunk_size,
                    "anchor_horizon_s": 1.0,
                    "control_hz": self.control_hz,
                    "control_steps_per_chunk": self.control_steps_per_chunk,
                    "camera_count": 3,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    def infer(self, observation: dict[str, Any], steps: int | None = None) -> np.ndarray:
        inputs = self.processor(
            images=[Image.fromarray(image) for image in observation["images"]],
            language_instruction=observation["prompt"],
        )
        missing = {"input_ids", "image_input", "image_mask"} - set(inputs)
        if missing:
            raise ValueError(f"Processor output is missing {sorted(missing)}")

        def move(tensor: torch.Tensor) -> torch.Tensor:
            dtype = torch.float32 if tensor.is_floating_point() else tensor.dtype
            return tensor.to(device=self.device, dtype=dtype)

        model_inputs = {key: move(value) for key, value in inputs.items()}
        model_inputs["proprio"] = move(
            torch.as_tensor(observation["proprio"], dtype=torch.float32).unsqueeze(0)
        )
        model_inputs["domain_id"] = torch.tensor([6], dtype=torch.long, device=self.device)
        denoise_steps = self.denoise_steps if steps is None else int(steps)
        if denoise_steps < 1:
            raise ValueError("steps must be at least 1")
        with torch.no_grad():
            action = self.model.generate_actions(**model_inputs, steps=denoise_steps)
        result = action.squeeze(0).float().cpu().numpy()
        if result.shape != (30, 20) or not np.isfinite(result).all():
            raise ValueError(f"Expected finite X-VLA output [30,20], got {result.shape}")
        return result

    def update_obs(self, obs):
        self.update_obs_batch([obs])

    def update_obs_batch(self, obs_list):
        self._request_index += 1
        self._latest_env_idx_list = [
            int(obs.get("env_idx", index)) for index, obs in enumerate(obs_list)
        ]
        self.observation_window = [
            _encode_observation(obs, self.default_prompt, self.task_prompt_map)
            for obs in obs_list
        ]
        self._latest_by_env = dict(zip(self._latest_env_idx_list, self.observation_window, strict=True))
        self._raw_by_env = dict(zip(self._latest_env_idx_list, obs_list, strict=True))

    def get_action_batch(self, env_idx_list=None, **kwargs):
        if self.observation_window is None:
            raise AssertionError("update_obs or update_obs_batch first!")
        env_idx_list = self._latest_env_idx_list if env_idx_list is None else env_idx_list
        batches = []
        for env_idx in map(int, env_idx_list):
            encoded = self._latest_by_env[env_idx]
            raw_observation = self._raw_by_env[env_idx]
            anchors20 = self.infer(encoded)
            state16 = _state16(raw_observation)
            control16 = resample_one_second_chunk(
                state16, anchors20, control_hz=self.control_hz
            )
            executed16 = control16[: self.control_steps_per_chunk]
            batches.append(_unpack_action16(executed16))
            if self.log_io:
                record = {
                    "event": "policy_response",
                    "request": self._request_index,
                    "env_idx": env_idx,
                    "instruction": encoded["prompt"],
                    "domain_id": 6,
                    "state16": state16.tolist(),
                    "state_names": list(ACTION_NAMES),
                    "images": {
                        name: _array_summary(image)
                        for name, image in zip(
                            ("cam_high", "cam_left_wrist", "cam_right_wrist"),
                            encoded["images"],
                            strict=True,
                        )
                    },
                    "raw_anchor_shape": list(anchors20.shape),
                    "control_shape": list(control16.shape),
                    "returned_shape": list(executed16.shape),
                    "returned_min": executed16.min(axis=0).tolist(),
                    "returned_max": executed16.max(axis=0).tolist(),
                    "gripper_semantics": "continuous; identical to training data",
                }
                if self.log_full_actions:
                    record["anchors20"] = anchors20.tolist()
                    record["control16"] = control16.tolist()
                print(
                    "[xvla_lerobot][io] " + json.dumps(record, ensure_ascii=False),
                    flush=True,
                )
        return batches

    def get_action(self, **kwargs):
        if not self._latest_env_idx_list:
            raise AssertionError("update_obs or update_obs_batch first!")
        return self.get_action_batch([self._latest_env_idx_list[0]], **kwargs)[0]

    def reset(self):
        self.observation_window = None
        self._latest_env_idx_list = [0]
        self._latest_by_env = {}
        self._raw_by_env = {}
        self._request_index = 0


def get_model(deploy_cfg):
    return Model(deploy_cfg)


__all__ = ["Model", "get_model", "resample_one_second_chunk"]
