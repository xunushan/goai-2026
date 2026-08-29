"""Serve a flow_policy (DINOv2 + patch_policy transformer + X-VLA flow matching)
policy through the XPolicyLab model interface.

The flow_policy model (training code in github.com/xunushan/flow-policy.git,
``flow_policy/`` subdir) predicts a 20-dim ARX ee6d action chunk
``[l_xyz(3), l_rot6d(6), l_g(1), r_xyz(3), r_rot6d(6), r_g(1)]`` by flow-matching
denoising (continuous regression, no gripper sigmoid). It has no language /
domain concept, so instructions are ignored.

The adapter:
- converts RoboDojo ObsManager state into the 20-dim proprio (gripper NOT
  inverted, matching the training handler ``invert_gripper=False``);
- encodes 3 camera views on-the-fly (Resize 224x224 BICUBIC, ToTensor 0-1);
  DINOv2 runs inside the model (``precomputed=False`` path);
- denoises ``steps`` flow-matching iterations producing a [num_actions, 20] chunk;
- converts the 20-dim chunk back to the 16-dim ee dict list protocol
  (left_ee_pose/right_ee_pose as xyz+quat_wxyz, gripper continuous [0,1]).
"""

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

for _path in (str(_REPO_ROOT), str(_CUR_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.process_data import decode_image_bit

from flow_policy.models.model import FlowPolicy

# View order must match training configs/train.yaml camera_keys
# (cam_high -> cam_head in the RoboDojo sim). Per-view fallback candidates for
# robustness across naming conventions.
DEFAULT_CAMERA_NAMES = ("cam_head", "cam_left_wrist", "cam_right_wrist")
CAMERA_CANDIDATES = {
    "cam_head": ("cam_head", "cam_high", "head_camera", "top_camera"),
    "cam_left_wrist": ("cam_left_wrist", "left_camera", "left_wrist"),
    "cam_right_wrist": ("cam_right_wrist", "right_camera", "right_wrist"),
}
IMAGE_SIZE = 224

# Model hyperparameters used by training configs/train.yaml.
MODEL_DEFAULTS = dict(
    dim_action=20,
    dim_propio=20,
    cond_dim=384,
    n_patches=256,
    views=3,
    num_actions=30,
    n_obs_steps=1,
    n_layer=12,
    n_head=12,
    n_embd=768,
    p_drop_emb=0.0,
    p_drop_attn=0.1,
)


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {value}")
    return device


def extract_image(observation: dict[str, Any], camera_name: str) -> np.ndarray:
    """Extract a single view by exact name, then fallback candidates."""
    vision = observation.get("vision", {})
    if not isinstance(vision, dict):
        raise KeyError("observation must contain a 'vision' mapping")

    def _value(entry: Any) -> np.ndarray:
        if isinstance(entry, dict):
            for field in ("color", "rgb"):
                if field in entry:
                    return np.asarray(entry[field])
            raise KeyError(f"camera entry has no color/rgb field: {sorted(entry)}")
        return np.asarray(entry)

    for name in (camera_name, *CAMERA_CANDIDATES.get(camera_name, ())):
        if name in vision:
            return _value(vision[name])
    raise KeyError(
        f"Missing camera {camera_name!r}; available={sorted(vision)}"
    )


def ensure_hwc_uint8(image: Any) -> np.ndarray:
    """Normalize a camera image to an HWC uint8 RGB array."""
    if isinstance(image, (bytes, bytearray, memoryview)):
        image = decode_image_bit(np.frombuffer(bytes(image), dtype=np.uint8))
    image = np.asarray(image)
    if image.ndim == 1 and image.dtype == np.uint8:
        image = decode_image_bit(image)
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


def _image_to_chw_float(image: np.ndarray) -> torch.Tensor:
    """HWC uint8 -> [3,224,224] float32 in [0,1] (matches training precompute:
    Resize(224, BICUBIC) + ToTensor; NO ImageNet Normalize — DINOv2 does it)."""
    rgb = ensure_hwc_uint8(image)[..., :3]
    pil = Image.fromarray(rgb)
    if pil.size != (IMAGE_SIZE, IMAGE_SIZE):
        pil = pil.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BICUBIC)
    arr = np.asarray(pil, dtype=np.float32) / 255.0
    return torch.from_numpy(np.ascontiguousarray(np.transpose(arr, (2, 0, 1))))


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
    """Rot6d -> unit quaternion in wxyz (scalar first), matching the protocol."""
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


def build_proprio(observation: dict[str, Any]) -> np.ndarray:
    """Observation state -> 20-dim arx_ee6d proprio.

    Input state layout (RoboDojo ObsManager): left_ee_pose xyz+quat_wxyz,
    left_ee_joint_state gripper, right_ee_pose, right_ee_joint_state.
    Gripper is passed through unchanged (1=open), matching training
    (invert_gripper=False).
    """
    state = observation["state"]
    left_ee = np.asarray(state["left_ee_pose"], dtype=np.float32)
    right_ee = np.asarray(state["right_ee_pose"], dtype=np.float32)
    left_grip = np.asarray(state["left_ee_joint_state"], dtype=np.float32)[-1]
    right_grip = np.asarray(state["right_ee_joint_state"], dtype=np.float32)[-1]

    for name, pose in (("left_ee_pose", left_ee), ("right_ee_pose", right_ee)):
        quat_norm = float(np.linalg.norm(pose[3:7]))
        if not 0.5 <= quat_norm <= 1.5:
            raise ValueError(f"{name} quaternion norm is invalid: {quat_norm:.6f}")

    return np.concatenate(
        [
            left_ee[:3],
            quat_to_rotate6d(left_ee[3:7]),
            np.array([left_grip], dtype=np.float32),
            right_ee[:3],
            quat_to_rotate6d(right_ee[3:7]),
            np.array([right_grip], dtype=np.float32),
        ],
        axis=-1,
    ).astype(np.float32)


def action_chunk_to_ee_dict_list(
    action_chunk: np.ndarray,
    *,
    gripper_mode: str = "continuous",
    gripper_threshold: float = 0.7,
) -> list[dict[str, np.ndarray]]:
    """20-dim arx_ee6d chunk -> list of 16-dim ee dicts (protocol format)."""
    action_chunk = np.asarray(action_chunk, dtype=np.float32)
    if action_chunk.ndim == 1:
        action_chunk = action_chunk[None, :]

    actions = []
    for row in action_chunk:
        left_quat = rotate6d_to_quat(row[3:9])
        right_quat = rotate6d_to_quat(row[13:19])
        left_grip = float(row[9])
        right_grip = float(row[19])
        if gripper_mode == "continuous":
            left_grip = float(np.clip(left_grip, 0.0, 1.0))
            right_grip = float(np.clip(right_grip, 0.0, 1.0))
        elif gripper_mode == "threshold":
            left_grip = float(left_grip > gripper_threshold)
            right_grip = float(right_grip > gripper_threshold)
        else:
            raise ValueError(
                f"gripper_mode must be 'continuous' or 'threshold', got {gripper_mode!r}"
            )
        actions.append(
            {
                "left_ee_pose": np.concatenate([row[0:3], left_quat]).astype(np.float32),
                "left_ee_joint_state": np.asarray([left_grip], dtype=np.float32),
                "right_ee_pose": np.concatenate([row[10:13], right_quat]).astype(np.float32),
                "right_ee_joint_state": np.asarray([right_grip], dtype=np.float32),
            }
        )
    return actions


class Model(ModelTemplate):
    def __init__(self, model_cfg: dict[str, Any]):
        self.model_cfg = dict(model_cfg)
        self.task_name = self.model_cfg.get("task_name", "default_task")
        self.action_type = self.model_cfg.get("action_type", "ee")
        if self.action_type != "ee":
            raise ValueError("patch_policy (flow_policy) supports only action_type='ee'.")

        self.device = _resolve_device(str(self.model_cfg.get("device", "cuda")))
        self.log_io = bool(self.model_cfg.get("log_io", True))

        # Model hyperparameters must match training configs/train.yaml.
        model_params = dict(MODEL_DEFAULTS)
        model_params.update(dict(self.model_cfg.get("model") or {}))
        self._model_params = model_params

        self.ckpt_path = self._resolve_ckpt(self.model_cfg)
        self.model = self._load_model()
        self.model.eval()

        self.model_chunk_size = int(self.model.num_actions)
        if self.model_chunk_size <= 0:
            raise ValueError(f"flow_policy model has invalid num_actions={self.model_chunk_size}")

        requested_chunk = self.model_cfg.get("actions_per_chunk")
        self.actions_per_chunk = (
            self.model_chunk_size if requested_chunk is None else int(requested_chunk)
        )
        if not 1 <= self.actions_per_chunk <= self.model_chunk_size:
            raise ValueError(
                "actions_per_chunk must be within "
                f"[1,{self.model_chunk_size}], got {self.actions_per_chunk}"
            )

        self.denoise_steps = int(self.model_cfg.get("steps", 10))
        if self.denoise_steps < 1:
            raise ValueError(f"steps must be at least 1, got {self.denoise_steps}")

        self.gripper_mode = str(
            self.model_cfg.get("gripper_mode", "continuous")
        ).lower()
        if self.gripper_mode not in {"continuous", "threshold"}:
            raise ValueError(
                "gripper_mode must be 'continuous' or 'threshold', got "
                f"{self.gripper_mode!r}"
            )
        self.gripper_threshold = float(self.model_cfg.get("gripper_threshold", 0.7))
        if not 0.0 <= self.gripper_threshold <= 1.0:
            raise ValueError(
                "gripper_threshold must be within [0,1], got "
                f"{self.gripper_threshold}"
            )

        raw_camera_names = self.model_cfg.get("camera_names") or DEFAULT_CAMERA_NAMES
        self.camera_names = [
            str(name).strip() for name in raw_camera_names if str(name).strip()
        ]
        if not self.camera_names:
            self.camera_names = list(DEFAULT_CAMERA_NAMES)
        expected_views = int(model_params["views"])
        if len(self.camera_names) > expected_views:
            raise ValueError(
                f"camera_names has {len(self.camera_names)} entries but the model "
                f"expects {expected_views} views; configure at most {expected_views}."
            )

        self._latest_env_idx_list: list[int] = [0]
        self._latest_by_env: dict[int, dict[str, Any]] = {}
        self._request_index = 0

        print(
            "[patch_policy] ready "
            f"ckpt={self.ckpt_path} device={self.device} "
            f"model_chunk_size={self.model_chunk_size} "
            f"execute_steps={self.actions_per_chunk} "
            f"denoise_steps={self.denoise_steps} "
            f"camera_names={self.camera_names} "
            f"gripper_mode={self.gripper_mode}",
            flush=True,
        )

    # ------------------------------------------------------------------ #
    # model loading
    # ------------------------------------------------------------------ #
    def _resolve_ckpt(self, model_cfg: dict[str, Any]) -> str:
        candidates: list[str] = []
        for key in ("ckpt_path", "checkpoint_path", "model_path"):
            if model_cfg.get(key):
                candidates.append(str(model_cfg[key]))
        if model_cfg.get("ckpt_name"):
            candidates.append(str(model_cfg["ckpt_name"]))

        for candidate in candidates:
            path = Path(candidate).expanduser()
            if not path.is_absolute():
                path = (_CUR_DIR / path).resolve()
            if path.is_file():
                return str(path)

        # ckpt_name may be a bare name under checkpoints/.
        name = model_cfg.get("ckpt_name")
        if name:
            for sub in (f"{name}.pt", name):
                path = _CUR_DIR / "checkpoints" / sub
                if path.is_file():
                    return str(path)
        raise FileNotFoundError(
            "Could not locate the flow_policy checkpoint. Provide ckpt_path / "
            "checkpoint_path / model_path (or ckpt_name under checkpoints/). "
            f"Checked: {candidates or '<none>'}"
        )

    def _load_model(self) -> FlowPolicy:
        print(f"[patch_policy] loading checkpoint {self.ckpt_path}", flush=True)
        ckpt = torch.load(self.ckpt_path, map_location="cpu", weights_only=False)
        if "model_state_dict" not in ckpt:
            raise ValueError(
                "flow_policy checkpoint must contain 'model_state_dict' "
                f"(found keys: {list(ckpt)})"
            )
        model = FlowPolicy(**self._model_params)
        try:
            missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=True)
        except RuntimeError as exc:
            raise RuntimeError(
                f"flow_policy state_dict mismatch: {exc}"
            ) from exc
        if missing or unexpected:
            raise RuntimeError(
                f"flow_policy state_dict mismatch: missing={missing}, unexpected={unexpected}"
            )
        del ckpt
        model.to(self.device)
        return model

    # ------------------------------------------------------------------ #
    # observation -> model inputs
    # ------------------------------------------------------------------ #
    def _build_image_input(self, observation: dict[str, Any]) -> torch.Tensor:
        views = [
            _image_to_chw_float(extract_image(observation, name))
            for name in self.camera_names
        ]
        # [V,3,224,224] -> [B=1, T_obs=1, V, 3, 224, 224]
        x = torch.stack(views, dim=0).unsqueeze(0).unsqueeze(0)
        return x.to(self.device)

    def _infer(self, observation: dict[str, Any]) -> np.ndarray:
        image_input = self._build_image_input(observation)
        proprio = torch.from_numpy(build_proprio(observation)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.model.generate_actions(
                image_input, proprio, steps=self.denoise_steps
            )
        return action.squeeze(0).float().cpu().numpy()

    # ------------------------------------------------------------------ #
    # ModelTemplate interface
    # ------------------------------------------------------------------ #
    def update_obs(self, obs):
        self.update_obs_batch([obs])

    def update_obs_batch(self, obs_list):
        self._latest_env_idx_list = [
            int(obs.get("env_idx", index)) if isinstance(obs, dict) else index
            for index, obs in enumerate(obs_list)
        ]
        self._latest_by_env = dict(
            zip(self._latest_env_idx_list, obs_list, strict=True)
        )

    def _array_summary(self, value: Any) -> dict[str, Any]:
        array = np.asarray(value)
        summary = {"shape": list(array.shape), "dtype": str(array.dtype)}
        if array.size:
            summary.update(
                {
                    "min": float(np.min(array)),
                    "max": float(np.max(array)),
                    "mean": float(np.mean(array)),
                }
            )
        return summary

    def _log_observation(self, observation: dict[str, Any], proprio: np.ndarray, env_idx: int) -> None:
        if not self.log_io:
            return
        state = observation.get("state", {})
        summary = {
            "event": "client_observation",
            "request": self._request_index,
            "env_idx": env_idx,
            "episode_idx": observation.get("episode_idx"),
            "task_name": observation.get("task_name"),
            "instruction": str(observation.get("instruction", ""))[:200],
            "proprio20": proprio.reshape(-1).tolist(),
            "images": {
                f"view_{i}": self._array_summary(
                    extract_image(observation, name)
                )
                for i, name in enumerate(self.camera_names)
            },
        }
        print("[patch_policy][io] " + json.dumps(summary, ensure_ascii=False), flush=True)

    def _log_actions(self, executed_chunk: np.ndarray, env_idx: int) -> None:
        if not self.log_io:
            return
        summary = {
            "event": "server_actions",
            "request": self._request_index,
            "env_idx": env_idx,
            "model_chunk_size": self.model_chunk_size,
            "execute_steps": int(executed_chunk.shape[0]),
            "gripper_mode": self.gripper_mode,
            "actions_16d": [
                np.concatenate(
                    [
                        action["left_ee_pose"],
                        action["left_ee_joint_state"],
                        action["right_ee_pose"],
                        action["right_ee_joint_state"],
                    ]
                ).astype(np.float32).tolist()
                for action in action_chunk_to_ee_dict_list(
                    executed_chunk,
                    gripper_mode=self.gripper_mode,
                    gripper_threshold=self.gripper_threshold,
                )
            ],
        }
        print("[patch_policy][io] " + json.dumps(summary, ensure_ascii=False), flush=True)

    def get_action(self, **kwargs):
        action_list = self.get_action_batch(
            env_idx_list=[self._latest_env_idx_list[0]], **kwargs
        )
        return action_list[0]

    def get_action_batch(self, env_idx_list=None, **kwargs):
        if not self._latest_by_env:
            raise AssertionError("update_obs or update_obs_batch must be called first!")
        env_idx_list = env_idx_list or self._latest_env_idx_list
        action_list = []
        for env_idx in env_idx_list:
            resolved_env_idx = int(env_idx)
            if resolved_env_idx not in self._latest_by_env:
                raise KeyError(
                    f"No buffered observation for env_idx={resolved_env_idx}; "
                    f"available={sorted(self._latest_by_env)}"
                )
            observation = self._latest_by_env[resolved_env_idx]
            proprio = build_proprio(observation)
            self._log_observation(observation, proprio, resolved_env_idx)
            raw_chunk = self._infer(observation)
            if raw_chunk.ndim != 2 or raw_chunk.shape[1] < 20:
                raise ValueError(
                    f"Expected flow_policy action chunk [T,>=20], got {raw_chunk.shape}"
                )
            if not np.isfinite(raw_chunk).all():
                raise ValueError("flow_policy action chunk contains NaN or Inf")
            if raw_chunk.shape[0] != self.model_chunk_size:
                raise ValueError(
                    "flow_policy returned an unexpected chunk length: "
                    f"expected {self.model_chunk_size}, got {raw_chunk.shape[0]}"
                )
            executed_chunk = raw_chunk[: self.actions_per_chunk]
            actions = action_chunk_to_ee_dict_list(
                executed_chunk,
                gripper_mode=self.gripper_mode,
                gripper_threshold=self.gripper_threshold,
            )
            self._log_actions(executed_chunk, resolved_env_idx)
            self._request_index += 1
            action_list.append(actions)
        return action_list

    def reset(self):
        self._latest_env_idx_list = [0]
        self._latest_by_env = {}
        self._request_index = 0
