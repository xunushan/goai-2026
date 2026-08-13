from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation as R

_CUR_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _CUR_DIR.parents[2]
_XVLA_ROOT = _CUR_DIR / "xvla"
_CHECKPOINTS_DIR = _CUR_DIR / "checkpoints"

for _path in (str(_REPO_ROOT), str(_CUR_DIR), str(_XVLA_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.checkpoint_resolver import resolve_checkpoint_root
from XPolicyLab.utils.process_data import decode_image_bit, get_robot_action_dim_info

from xvla.models.modeling_xvla import XVLA
from xvla.models.processing_xvla import XVLAProcessor

from gripper_hysteresis import HysteresisConfig, apply_gripper_hysteresis


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


def extract_named_image(observation, camera_name):
    """按精确相机名从 observation['vision'] 提取单路图像（无候选回退）。

    多视角 camera_names 配置专用：找不到相机、或相机缺 color/rgb 字段直接报错，
    避免静默落到错误视角。与 extract_image 的候选回退语义互补。
    """
    vision = observation.get("vision", {})
    if camera_name not in vision:
        raise KeyError(
            f"camera {camera_name!r} not found in observation['vision'] "
            f"(available: {sorted(vision)})"
        )
    image = vision[camera_name]
    if isinstance(image, dict):
        for image_key in ("color", "rgb"):
            if image_key in image:
                return image[image_key]
        raise KeyError(
            f"camera {camera_name!r} has no color/rgb field: {sorted(image)}"
        )
    return image


def ensure_hwc_uint8(image):
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
        return image
    if image.shape[0] in (1, 3):
        return np.transpose(image, (1, 2, 0))
    raise ValueError(f"Unsupported image shape: {image.shape}")


def decode_compressed_image(image_buffer):
    return decode_image_bit(image_buffer)


def _normalize_prompt_value(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="ignore")
    elif isinstance(value, np.ndarray) and value.ndim == 0:
        value = value.item()
    elif isinstance(value, np.generic):
        value = value.item()

    if isinstance(value, (list, tuple)):
        for item in value:
            normalized = _normalize_prompt_value(item)
            if normalized is not None:
                return normalized
        return None

    if isinstance(value, str):
        value = value.strip()
        return value or None

    return str(value)


def resolve_prompt(
    observation: dict[str, Any],
    default_prompt: str,
    task_prompt_map: dict[str, str] | None = None,
) -> str:
    for key in ("prompt", "instruction", "task", "language_instruction"):
        prompt = _normalize_prompt_value(observation.get(key))
        if prompt is not None:
            return (task_prompt_map or {}).get(prompt, prompt)

    fallback = _normalize_prompt_value(default_prompt)
    if fallback is None:
        raise ValueError("No valid prompt found in observation or model config.")
    return (task_prompt_map or {}).get(fallback, fallback)


def _extract_step_number(value: Any) -> int | None:
    digits = "".join(ch for ch in str(value) if ch.isdigit())
    return int(digits) if digits else None


def _resolve_path(value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (_CUR_DIR / path).resolve()
    else:
        path = path.resolve()
    return path


def _resolve_checkpoint_root(model_cfg: dict[str, Any]) -> Path | None:
    # Shared precedence: explicit path keys > ckpt_name-as-path > 5-tuple
    # concat under checkpoints/ > checkpoints/<ckpt_name> verbatim. The
    # within-root step-dir discovery (_build_candidate_dirs) is preserved.
    # The processor is co-located with the model in the checkpoint dir, so it is
    # loaded from the resolved model path rather than a separate processor_path.
    explicit_keys = ("model_path", "checkpoint_path")
    if model_cfg.get("ckpt_name") or any(model_cfg.get(key) for key in explicit_keys):
        checkpoint_root = resolve_checkpoint_root(
            model_cfg,
            _CHECKPOINTS_DIR,
            policy_dir=_CUR_DIR,
            explicit_keys=explicit_keys,
            must_exist=False,
        )
        if not checkpoint_root.is_dir():
            return checkpoint_root

        candidate_dirs = []
        if any((checkpoint_root / marker).exists() for marker in ("config.json", "model.safetensors", "preprocessor_config.json")):
            candidate_dirs.append(checkpoint_root)
        candidate_dirs.extend(
            child
            for child in sorted(checkpoint_root.iterdir())
            if child.is_dir() and any((child / marker).exists() for marker in ("config.json", "model.safetensors", "preprocessor_config.json"))
        )
        if not candidate_dirs:
            return checkpoint_root

        checkpoint_num = model_cfg.get("checkpoint_num")
        desired_step = _extract_step_number(checkpoint_num)
        if desired_step is not None:
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

    # No explicit path / ckpt_name was given; model loading will raise its own error.
    return None


def _build_candidate_dirs(checkpoint_root: Path | None, *explicit_paths: str | None) -> list[Path]:
    candidates: list[Path] = []
    for explicit_path in explicit_paths:
        resolved = _resolve_path(explicit_path)
        if resolved is not None and resolved not in candidates:
            candidates.append(resolved)

    if checkpoint_root is not None:
        for candidate in (
            checkpoint_root,
            checkpoint_root / "processor",
            checkpoint_root / "model",
            checkpoint_root / "base",
            checkpoint_root / "base_model",
            checkpoint_root / "checkpoint",
        ):
            if candidate not in candidates:
                candidates.append(candidate)
    return candidates


def quat_to_rotate6d(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32)
    if quat.shape[-1] != 4:
        raise ValueError(f"Expected quaternion with 4 dims, got shape {quat.shape}.")
    quat = quat.copy()
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    zero_norm_mask = norm.squeeze(-1) < 1e-8
    if np.any(zero_norm_mask):
        quat[zero_norm_mask] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    quat = quat / np.clip(norm, 1e-8, None)
    xyzw = np.concatenate([quat[..., 1:], quat[..., :1]], axis=-1)
    rot = R.from_quat(xyzw).as_matrix()
    return rot[..., :, :2].reshape(quat.shape[:-1] + (6,)).astype(np.float32)


def rotate6d_to_quat(vec6: np.ndarray) -> np.ndarray:
    vec6 = np.asarray(vec6, dtype=np.float32)
    if vec6.shape[-1] != 6:
        raise ValueError(f"Expected last dim to be 6, got {vec6.shape[-1]}.")

    a1 = vec6[..., 0:5:2]
    a2 = vec6[..., 1:6:2]
    b1 = a1 / np.clip(np.linalg.norm(a1, axis=-1, keepdims=True), 1e-8, None)
    proj = np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = a2 - proj
    b2 = b2 / np.clip(np.linalg.norm(b2, axis=-1, keepdims=True), 1e-8, None)
    b3 = np.cross(b1, b2)
    rot = np.stack((b1, b2, b3), axis=-1)

    m00, m01, m02 = rot[..., 0, 0], rot[..., 0, 1], rot[..., 0, 2]
    m10, m11, m12 = rot[..., 1, 0], rot[..., 1, 1], rot[..., 1, 2]
    m20, m21, m22 = rot[..., 2, 0], rot[..., 2, 1], rot[..., 2, 2]

    trace = m00 + m11 + m22
    quat = np.empty(rot.shape[:-2] + (4,), dtype=np.float32)

    positive = trace > 0
    s = np.sqrt(np.maximum(trace[positive] + 1.0, 1e-8)) * 2
    quat[positive, 3] = 0.25 * s
    quat[positive, 0] = (m21[positive] - m12[positive]) / s
    quat[positive, 1] = (m02[positive] - m20[positive]) / s
    quat[positive, 2] = (m10[positive] - m01[positive]) / s

    cond1 = (~positive) & (m00 > m11) & (m00 > m22)
    s = np.sqrt(np.maximum(1.0 + m00[cond1] - m11[cond1] - m22[cond1], 1e-8)) * 2
    quat[cond1, 3] = (m21[cond1] - m12[cond1]) / s
    quat[cond1, 0] = 0.25 * s
    quat[cond1, 1] = (m01[cond1] + m10[cond1]) / s
    quat[cond1, 2] = (m02[cond1] + m20[cond1]) / s

    cond2 = (~positive) & (~cond1) & (m11 > m22)
    s = np.sqrt(np.maximum(1.0 + m11[cond2] - m00[cond2] - m22[cond2], 1e-8)) * 2
    quat[cond2, 3] = (m02[cond2] - m20[cond2]) / s
    quat[cond2, 0] = (m01[cond2] + m10[cond2]) / s
    quat[cond2, 1] = 0.25 * s
    quat[cond2, 2] = (m12[cond2] + m21[cond2]) / s

    cond3 = (~positive) & (~cond1) & (~cond2)
    s = np.sqrt(np.maximum(1.0 + m22[cond3] - m00[cond3] - m11[cond3], 1e-8)) * 2
    quat[cond3, 3] = (m10[cond3] - m01[cond3]) / s
    quat[cond3, 0] = (m02[cond3] + m20[cond3]) / s
    quat[cond3, 1] = (m12[cond3] + m21[cond3]) / s
    quat[cond3, 2] = 0.25 * s

    quat = quat / np.clip(np.linalg.norm(quat, axis=-1, keepdims=True), 1e-8, None)
    return np.concatenate([quat[..., 3:4], quat[..., :3]], axis=-1).astype(np.float32)




def build_xvla_proprio(observation: dict[str, Any]) -> np.ndarray:
    state = observation["state"]
    left_ee = np.asarray(state["left_ee_pose"], dtype=np.float32)
    right_ee = np.asarray(state["right_ee_pose"], dtype=np.float32)
    left_grip_joint = np.asarray(state["left_ee_joint_state"], dtype=np.float32)[-1]
    right_grip_joint = np.asarray(state["right_ee_joint_state"], dtype=np.float32)[-1]

    return np.concatenate(
        [
            left_ee[:3],
            quat_to_rotate6d(left_ee[3:]),
            np.array([left_grip_joint], dtype=np.float32),
            right_ee[:3],
            quat_to_rotate6d(right_ee[3:]),
            np.array([right_grip_joint], dtype=np.float32),
        ],
        axis=-1,
    ).astype(np.float32)


def encode_obs(observation, default_prompt, task_prompt_map=None, camera_names=None):
    if camera_names:
        # 多视角：按 camera_names 顺序逐路提取，与模型训练视角顺序严格一致。
        images = [
            ensure_hwc_uint8(extract_named_image(observation, name))
            for name in camera_names
        ]
        prompt = resolve_prompt(observation, default_prompt, task_prompt_map)
        return {
            "images": images,
            "proprio": build_xvla_proprio(observation),
            "prompt": prompt,
            "output_format": "xpolicylab",
        }
    if "images" in observation and "state" in observation:
        head = ensure_hwc_uint8(observation["images"]["cam_high"])
        prompt = resolve_prompt(observation, default_prompt, task_prompt_map)
        return {
            "images": [head],
            "proprio": build_xvla_proprio(observation),
            "prompt": prompt,
            "output_format": "xpolicylab",
        }

    images = [ensure_hwc_uint8(extract_image(observation, ["cam_high", "cam_head", "head_camera", "top_camera"]))]
    prompt = resolve_prompt(observation, default_prompt, task_prompt_map)
    return {
        "images": images,
        "proprio": build_xvla_proprio(observation),
        "prompt": prompt,
        "output_format": "xpolicylab",
    }


def action_chunk_to_ee_dict_list(
    action_chunk: np.ndarray,
    *,
    gripper_mode: str = "continuous",
    gripper_threshold: float = 0.7,
):
    action_chunk = np.asarray(action_chunk, dtype=np.float32)
    if action_chunk.ndim == 1:
        action_chunk = action_chunk[None, :]

    left_xyz = action_chunk[:, :3]
    left_rotate6d = action_chunk[:, 3:9]
    left_gripper = action_chunk[:, 9:10]
    left_quat = rotate6d_to_quat(left_rotate6d)
    # X-VLA already applies sigmoid, and RoboDojo stores continuous gripper
    # positions in [0, 1] (1=open, 0=closed). Preserve them by default.
    if gripper_mode == "continuous":
        left_grip = np.clip(left_gripper, 0.0, 1.0)
    elif gripper_mode == "threshold":
        left_grip = (left_gripper > gripper_threshold).astype(np.float32)
    else:
        raise ValueError(
            f"gripper_mode must be 'continuous' or 'threshold', got {gripper_mode!r}"
        )

    right_xyz = action_chunk[:, 10:13]
    right_rotate6d = action_chunk[:, 13:19]
    right_quat = rotate6d_to_quat(right_rotate6d)
    right_gripper = action_chunk[:, 19:20]
    if gripper_mode == "continuous":
        right_grip = np.clip(right_gripper, 0.0, 1.0)
    else:
        right_grip = (right_gripper > gripper_threshold).astype(np.float32)

    actions = []
    for idx in range(action_chunk.shape[0]):
        actions.append(
            {
                "left_ee_pose": np.concatenate([left_xyz[idx], left_quat[idx]], axis=0).astype(np.float32),
                "left_ee_joint_state": np.asarray([left_grip[idx, 0]], dtype=np.float32),
                "right_ee_pose": np.concatenate([right_xyz[idx], right_quat[idx]], axis=0).astype(np.float32),
                "right_ee_joint_state": np.asarray([right_grip[idx, 0]], dtype=np.float32),
            }
        )
    return actions


class Model(ModelTemplate):
    def __init__(self, model_cfg):
        self.model_cfg = dict(model_cfg)
        self.task_name = self.model_cfg.get("task_name", "default_task")
        self.action_type = self.model_cfg.get("action_type", "ee")
        if self.action_type != "ee":
            raise ValueError("X-VLA in XPolicyLab currently supports only action_type='ee'.")

        self.default_prompt = self.model_cfg.get("prompt", self.task_name)
        self.task_prompt_map = dict(self.model_cfg.get("task_prompt_map") or {})
        env_cfg = self.model_cfg.get("env_cfg") or self.model_cfg.get("env_cfg_type")
        self.robot_action_dim_info = get_robot_action_dim_info(env_cfg) if env_cfg is not None else None
        self._latest_env_idx_list: list[int] = [0]
        self._raw_by_env: dict[int, dict[str, Any]] = {}
        self._latest_by_env: dict[int, dict[str, Any]] = {}
        self.observation_window: list[dict[str, Any]] | None = None

        self.device = self._get_device(self.model_cfg.get("device", "cuda"))
        self.processor = self._load_processor(self.model_cfg)
        self.model = self._load_model(self.model_cfg)
        self.model.eval()

        self.model_chunk_size = int(getattr(self.model, "num_actions", 0))
        if self.model_chunk_size <= 0:
            raise ValueError(
                f"X-VLA model has invalid num_actions={self.model_chunk_size}"
            )
        requested_chunk = self.model_cfg.get("actions_per_chunk")
        self.actions_per_chunk = (
            self.model_chunk_size
            if requested_chunk is None
            else int(requested_chunk)
        )
        if not 1 <= self.actions_per_chunk <= self.model_chunk_size:
            raise ValueError(
                "actions_per_chunk must be within "
                f"[1,{self.model_chunk_size}], got {self.actions_per_chunk}"
            )
        self.gripper_mode = str(
            self.model_cfg.get("gripper_mode", "continuous")
        ).lower()
        if self.gripper_mode not in {"continuous", "threshold"}:
            raise ValueError(
                "gripper_mode must be 'continuous' or 'threshold', got "
                f"{self.gripper_mode!r}"
            )
        self.gripper_threshold = float(
            self.model_cfg.get("gripper_threshold", 0.7)
        )
        if not 0.0 <= self.gripper_threshold <= 1.0:
            raise ValueError(
                "gripper_threshold must be within [0,1], got "
                f"{self.gripper_threshold}"
            )
        # 夹爪迟滞（闭环改造方案 3.2 节 D4）：执行层后处理，作用于最终返回的
        # 16 维动作（ee-dict list）的夹爪维、6D→四元数转换之后。enabled=false 跳过。
        self._hysteresis_cfg = HysteresisConfig.from_model_cfg(self.model_cfg)
        if self._hysteresis_cfg.enabled and self.gripper_mode != "continuous":
            raise ValueError(
                "gripper hysteresis requires gripper_mode='continuous' "
                f"(got {self.gripper_mode!r}); hysteresis takes over the gripper "
                "output."
            )
        self.log_io = bool(self.model_cfg.get("log_io", True))
        self.denoise_steps = int(self.model_cfg.get("steps", 10))
        if self.denoise_steps < 1:
            raise ValueError(
                f"steps must be at least 1, got {self.denoise_steps}"
            )
        # 多视角图像：按顺序指定送入模型的相机（1~3 路，默认顺序 cam_head →
        # cam_left_wrist → cam_right_wrist）。不配置（或空）时保持原单路行为
        # （encode_obs 走 legacy 候选回退，取 cam_head）。
        raw_camera_names = self.model_cfg.get("camera_names") or []
        self.camera_names = [
            str(name).strip()
            for name in raw_camera_names
            if str(name).strip()
        ]
        if self.camera_names:
            processor_views = int(getattr(self.processor, "num_views", 3) or 3)
            if len(self.camera_names) > processor_views:
                raise ValueError(
                    f"camera_names has {len(self.camera_names)} entries but the "
                    f"model processor supports num_views={processor_views}; "
                    f"configure at most {processor_views} cameras."
                )
        self._request_index = 0
        # 评测复现用的 flow-noise seed。None / 未配置 = 关闭（保持原始随机行为，
        # 正式测评时直接删除 deploy.yml 的 policy_seed 行即可）；配置数值后每次
        # episode reset 都从固定序列起点开始，相同 (layout, policy, ckpt) 可复现。
        policy_seed_cfg = self.model_cfg.get("policy_seed")
        self.policy_seed: int | None = (
            None
            if policy_seed_cfg is None
            or str(policy_seed_cfg).strip().lower() in {"", "none", "null"}
            else int(policy_seed_cfg)
        )
        self._policy_generators: dict[int, torch.Generator] = {}
        self._policy_noise_draws: dict[int, int] = {}
        print(
            "[x_vla] "
            f"model_chunk_size={self.model_chunk_size} "
            f"execute_steps={self.actions_per_chunk} "
            f"denoise_steps={self.denoise_steps} "
            f"camera_names={self.camera_names or 'legacy(1-view)'} "
            f"gripper_mode={self.gripper_mode} "
            f"gripper_threshold={self.gripper_threshold} "
            f"hysteresis={self._hysteresis_cfg.mode if self._hysteresis_cfg.enabled else 'off'} "
            f"policy_seed={self.policy_seed}",
            flush=True,
        )

    def _get_device(self, device_arg: str):
        if device_arg == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device_arg)

    def _load_processor(self, model_cfg):
        # Processor 与模型同目录（checkpoint 目录含 preprocessor_config.json 等
        # processor 文件），随模型路径一起解析，不再有独立的 processor_path。
        checkpoint_root = _resolve_checkpoint_root(model_cfg)
        candidate_paths = _build_candidate_dirs(
            checkpoint_root,
            model_cfg.get("model_path"),
            model_cfg.get("checkpoint_path"),
        )

        processor_path = None
        for candidate in candidate_paths:
            if (candidate / "preprocessor_config.json").exists():
                processor_path = str(candidate)
                break
        if processor_path is None:
            searched = ", ".join(str(path) for path in candidate_paths) or "none"
            raise FileNotFoundError(
                "Could not find XVLA processor files (preprocessor_config.json). "
                f"Searched: {searched}"
            )
        return XVLAProcessor.from_pretrained(processor_path)

    def _load_model(self, model_cfg):
        checkpoint_root = _resolve_checkpoint_root(model_cfg)
        candidate_paths = _build_candidate_dirs(
            checkpoint_root,
            model_cfg.get("model_path"),
            model_cfg.get("checkpoint_path"),
        )
        model_path = None
        for candidate in candidate_paths:
            if (candidate / "config.json").exists():
                model_path = str(candidate)
                break
        if model_path is None:
            raise ValueError("ckpt_name, model_path, or checkpoint_path is required for X-VLA.")

        model = XVLA.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float32,
        ).to(self.device).to(torch.float32)

        lora_path = model_cfg.get("lora_path") or model_cfg.get("LoRA_path")
        if not lora_path and checkpoint_root is not None and (checkpoint_root / "adapter_config.json").exists():
            lora_path = str(checkpoint_root)
        if lora_path:
            from peft import PeftModel

            model = PeftModel.from_pretrained(
                model,
                lora_path,
                torch_dtype=torch.float32,
            ).to(self.device)
        return model

    def update_obs(self, obs):
        self.update_obs_batch([obs])

    @staticmethod
    def _array_summary(value: Any) -> dict[str, Any]:
        array = np.asarray(value)
        summary: dict[str, Any] = {
            "shape": list(array.shape),
            "dtype": str(array.dtype),
        }
        if array.size:
            summary.update(
                {
                    "min": float(np.min(array)),
                    "max": float(np.max(array)),
                    "mean": float(np.mean(array)),
                }
            )
        return summary

    def _log_observation(
        self,
        observation: dict[str, Any],
        encoded_observation: dict[str, Any],
        env_idx: int,
    ) -> None:
        if not self.log_io:
            return

        def finite_list(value: Any) -> list[float | None]:
            array = np.asarray(value, dtype=np.float32).reshape(-1)
            return [
                float(item) if np.isfinite(item) else None
                for item in array
            ]

        state = observation.get("state", {})
        state_summary = {
            key: finite_list(state[key])
            for key in (
                "left_ee_pose",
                "left_ee_joint_state",
                "right_ee_pose",
                "right_ee_joint_state",
            )
            if key in state
        }
        images = {
            f"image_{index}": self._array_summary(image)
            for index, image in enumerate(encoded_observation["images"])
        }
        summary = {
            "event": "client_observation",
            "request": self._request_index,
            "env_idx": env_idx,
            "instruction": str(observation.get("instruction", ""))[:200],
            "model_prompt": resolve_prompt(
                encoded_observation,
                self.default_prompt,
            )[:200],
            "state": state_summary,
            "proprio20": finite_list(encoded_observation["proprio"]),
            "images": images,
        }
        print(
            "[x_vla][io] " + json.dumps(summary, ensure_ascii=False),
            flush=True,
        )

    def update_obs_batch(self, obs_list):
        self._latest_env_idx_list = [
            int(obs.get("env_idx", index)) if isinstance(obs, dict) else index
            for index, obs in enumerate(obs_list)
        ]
        self.observation_window = [
            encode_obs(obs, self.default_prompt, self.task_prompt_map, self.camera_names)
            for obs in obs_list
        ]
        self._latest_by_env = dict(
            zip(self._latest_env_idx_list, self.observation_window, strict=True)
        )
        self._raw_by_env = dict(
            zip(self._latest_env_idx_list, obs_list, strict=True)
        )

    def infer(
        self,
        observation: dict[str, Any],
        steps: int | None = None,
        generator: torch.Generator | None = None,
    ):
        pil_images = [Image.fromarray(image) for image in observation["images"]]
        prompt = resolve_prompt(observation, self.default_prompt)
        inputs = self.processor(images=pil_images, language_instruction=prompt)
        missing_inputs = {"input_ids", "image_input", "image_mask"} - set(inputs)
        if missing_inputs:
            raise ValueError(
                f"Processor returned incomplete inputs: missing {sorted(missing_inputs)} for prompt={prompt!r}."
            )
        proprio = torch.as_tensor(observation["proprio"], dtype=torch.float32).unsqueeze(0)
        domain_id = torch.tensor([int(self.model_cfg.get("domain_id", 0))], dtype=torch.long)

        def to_model(tensor: torch.Tensor):
            if tensor.is_floating_point():
                return tensor.to(device=self.device, dtype=torch.float32)
            return tensor.to(device=self.device)

        inputs = {key: to_model(value) for key, value in inputs.items()}
        inputs["proprio"] = to_model(proprio)
        inputs["domain_id"] = domain_id.to(self.device)

        denoise_steps = int(steps if steps is not None else self.denoise_steps)
        if denoise_steps < 1:
            raise ValueError(f"steps must be at least 1, got {denoise_steps}")
        with torch.no_grad():
            action = self.model.generate_actions(
                **inputs,
                steps=denoise_steps,
                generator=generator,
            )
        return action.squeeze(0).float().cpu().numpy()

    def _get_policy_generator(self, env_idx: int) -> torch.Generator | None:
        """每个 env 一个独立 generator，保证 env 间噪声序列互不干扰。

        policy_seed 未配置时返回 None（infer 保持原始随机行为）。
        env_seed = policy_seed + 1_000_003 * env_idx，大质数间隔分隔不同环境；
        某环境提前结束不会改变其他环境的噪声序列。
        """
        if self.policy_seed is None:
            return None
        env_idx = int(env_idx)
        if env_idx not in self._policy_generators:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(self.policy_seed + 1_000_003 * env_idx)
            self._policy_generators[env_idx] = generator
            self._policy_noise_draws[env_idx] = 0
        return self._policy_generators[env_idx]

    def get_action(self, **kwargs):
        action_list = self.get_action_batch(env_idx_list=[self._latest_env_idx_list[0]], **kwargs)
        return action_list[0]

    def get_action_batch(self, env_idx_list=None, **kwargs):
        if self.observation_window is None:
            raise AssertionError("update_obs or update_obs_batch first!")

        env_idx_list = env_idx_list or self._latest_env_idx_list
        action_list = []
        for env_idx in env_idx_list:
            resolved_env_idx = int(env_idx)
            if resolved_env_idx not in self._latest_by_env:
                raise KeyError(
                    f"No buffered observation for env_idx={resolved_env_idx}; "
                    f"available={sorted(self._latest_by_env)}"
                )
            encoded_obs = self._latest_by_env[resolved_env_idx]
            self._log_observation(
                self._raw_by_env[resolved_env_idx],
                encoded_obs,
                resolved_env_idx,
            )
            generator = self._get_policy_generator(resolved_env_idx)
            raw_chunk = self.infer(encoded_obs, generator=generator)
            if generator is not None:
                # 记录为「本次采样后累计次数」（1-based）：env 首次重规划为 1
                self._policy_noise_draws[resolved_env_idx] += 1
            if raw_chunk.ndim != 2 or raw_chunk.shape[1] < 20:
                raise ValueError(
                    f"Expected X-VLA action chunk [T,>=20], got {raw_chunk.shape}"
                )
            if not np.isfinite(raw_chunk).all():
                raise ValueError("X-VLA action chunk contains NaN or Inf")
            if raw_chunk.shape[0] != self.model_chunk_size:
                raise ValueError(
                    "X-VLA returned an unexpected chunk length: "
                    f"expected {self.model_chunk_size}, got {raw_chunk.shape[0]}"
                )
            executed_chunk = raw_chunk[: self.actions_per_chunk]
            actions = action_chunk_to_ee_dict_list(
                executed_chunk,
                gripper_mode=self.gripper_mode,
                gripper_threshold=self.gripper_threshold,
            )
            # 夹爪迟滞（执行层后处理，D4）：作用于最终 16 维动作的夹爪维、
            # 6D→四元数转换之后。latch 每次由当前 obs 的真实夹爪位置初始化，
            # 无跨请求状态；enabled=false 跳过。
            if self._hysteresis_cfg.enabled:
                obs_state = self._raw_by_env[resolved_env_idx]["state"]
                apply_gripper_hysteresis(
                    actions,
                    left_init=float(obs_state["left_ee_joint_state"][-1]),
                    right_init=float(obs_state["right_ee_joint_state"][-1]),
                    lo=self._hysteresis_cfg.lo,
                    hi=self._hysteresis_cfg.hi,
                    mode=self._hysteresis_cfg.mode,
                )
            if self.log_io:
                summary = {
                    "event": "server_actions",
                    "request": self._request_index,
                    "env_idx": resolved_env_idx,
                    "model_chunk_size": int(raw_chunk.shape[0]),
                    "execute_steps": int(executed_chunk.shape[0]),
                    "gripper_mode": self.gripper_mode,
                    "gripper_threshold": self.gripper_threshold,
                    "hysteresis": (
                        self._hysteresis_cfg.mode
                        if self._hysteresis_cfg.enabled
                        else None
                    ),
                    "policy_seed": self.policy_seed,
                    "policy_noise_draw": (
                        self._policy_noise_draws[resolved_env_idx]
                        if self.policy_seed is not None
                        else None
                    ),
                    "actions_16d": [
                        np.concatenate(
                            [
                                action["left_ee_pose"],
                                action["left_ee_joint_state"],
                                action["right_ee_pose"],
                                action["right_ee_joint_state"],
                            ]
                        ).astype(np.float32).tolist()
                        for action in actions
                    ],
                }
                print(
                    "[x_vla][io] "
                    + json.dumps(summary, ensure_ascii=False),
                    flush=True,
                )
            self._request_index += 1
            action_list.append(actions)
        return action_list

    def reset(self):
        self.observation_window = None
        self._latest_env_idx_list = [0]
        self._raw_by_env = {}
        self._latest_by_env = {}
        self._request_index = 0
        # episode 开始：清空生成器，使每个 episode 从固定噪声序列起点重新开始
        self._policy_generators = {}
        self._policy_noise_draws = {}
