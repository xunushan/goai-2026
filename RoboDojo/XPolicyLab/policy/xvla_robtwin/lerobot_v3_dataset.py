"""LeRobot v3 EE dataset adapter for X-VLA.

RoboDojo stores dual-arm absolute EE state/action as 16 values:
left xyz + quaternion(wxyz) + gripper opening, followed by the right arm.
X-VLA EE6D consumes 20 values:
left xyz + rotation6d + gripper-close target, followed by the right arm.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

import torch
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

STATE_KEY = "observation.state"
ACTION_KEY = "action"
DEFAULT_CAMERA_KEY = "observation.images.cam_high"
EXPECTED_EE_NAMES = (
    "l_x", "l_y", "l_z", "l_w", "l_wx", "l_wy", "l_wz", "l_g",
    "r_x", "r_y", "r_z", "r_w", "r_wx", "r_wy", "r_wz", "r_g",
)


def _quat_wxyz_to_rotation6d(quat: torch.Tensor) -> torch.Tensor:
    """Return the first two rotation-matrix columns, flattened row-major."""
    w, x, y, z = quat.unbind(dim=-1)
    r00 = 1 - 2 * (y * y + z * z)
    r01 = 2 * (x * y - z * w)
    r10 = 2 * (x * y + z * w)
    r11 = 1 - 2 * (x * x + z * z)
    r20 = 2 * (x * z - y * w)
    r21 = 2 * (y * z + x * w)
    return torch.stack((r00, r01, r10, r11, r20, r21), dim=-1)


def parse_episode_list(value: str | None) -> list[int] | None:
    if value is None or not value.strip():
        return None
    candidate = Path(value).expanduser()
    if candidate.is_file():
        payload = json.loads(candidate.read_text(encoding="utf-8"))
    else:
        payload = json.loads(value)
    if isinstance(payload, dict):
        raise ValueError("Episode JSON object requires selecting train or val before parsing.")
    episodes = [int(item) for item in payload]
    if len(episodes) != len(set(episodes)):
        raise ValueError("Episode list contains duplicates.")
    return episodes


def episodes_from_split(path: str | Path, split: str) -> list[int]:
    payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if split not in payload:
        raise KeyError(f"Split file does not contain {split!r}.")
    episodes = [int(item) for item in payload[split]]
    if not episodes:
        raise ValueError(f"Split {split!r} is empty.")
    if len(episodes) != len(set(episodes)):
        raise ValueError(f"Split {split!r} contains duplicate episodes.")
    return episodes


def ee16_open_to_xvla20_close(value: torch.Tensor) -> torch.Tensor:
    """Convert [...,16] RoboDojo EE quaternion/opening to X-VLA EE6D/closing."""
    tensor = torch.as_tensor(value, dtype=torch.float32)
    if tensor.shape[-1] != 16:
        raise ValueError(f"Expected EE vector with 16 values, got {tuple(tensor.shape)}.")
    flat = tensor.reshape(-1, 16)
    if not torch.isfinite(flat).all():
        raise ValueError("EE state/action contains NaN or Inf.")

    converted_arms: list[torch.Tensor] = []
    for offset in (0, 8):
        xyz = flat[:, offset : offset + 3]
        quat = flat[:, offset + 3 : offset + 7]
        quat_norm = torch.linalg.vector_norm(quat, dim=-1)
        if torch.any(torch.abs(quat_norm - 1.0) > 0.05):
            raise ValueError("Quaternion norm differs from 1 by more than 0.05.")
        quat = quat / quat_norm[:, None].clamp_min(1e-8)
        rot6d = _quat_wxyz_to_rotation6d(quat)
        opening = flat[:, offset + 7 : offset + 8]
        if torch.any((opening < -1e-4) | (opening > 1.0001)):
            raise ValueError("RoboDojo gripper opening must be in [0, 1].")
        close_target = 1.0 - opening.clamp(0.0, 1.0)
        converted_arms.append(torch.cat((xyz, rot6d, close_target), dim=-1))

    result = torch.cat(converted_arms, dim=-1)
    return result.reshape(tensor.shape[:-1] + (20,))


def _normalize_head_image(image: torch.Tensor, *, training: bool) -> torch.Tensor:
    image = torch.as_tensor(image, dtype=torch.float32)
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError(f"Expected CHW RGB image, got {tuple(image.shape)}.")
    if image.max() > 1.5:
        image = image / 255.0
    image = image.clamp(0.0, 1.0)
    image = TF.resize(
        image,
        [224, 224],
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    )
    if training:
        # Match the original X-VLA HDF5 pipeline's modest color jitter.
        brightness = float(torch.empty(1).uniform_(0.8, 1.2))
        contrast = float(torch.empty(1).uniform_(0.8, 1.2))
        saturation = float(torch.empty(1).uniform_(0.8, 1.2))
        image = TF.adjust_brightness(image, brightness)
        image = TF.adjust_contrast(image, contrast)
        image = TF.adjust_saturation(image, saturation)
    return TF.normalize(
        image,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )


class XVLALeRobotV3EEDataset(Dataset):
    """Map-style LeRobot v3 dataset yielding native X-VLA training batches."""

    def __init__(
        self,
        *,
        root: str | Path,
        repo_id: str,
        num_actions: int,
        domain_id: int = 6,
        episodes: Sequence[int] | None = None,
        camera_key: str = DEFAULT_CAMERA_KEY,
        task_allowlist: Iterable[str] | None = None,
        training: bool = True,
        video_backend: str = "pyav",
    ) -> None:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        self.root = Path(root).expanduser().resolve()
        self.repo_id = repo_id
        self.num_actions = int(num_actions)
        self.domain_id = int(domain_id)
        self.camera_key = camera_key
        self.training = bool(training)
        if self.num_actions <= 0:
            raise ValueError("num_actions must be positive.")
        if self.domain_id < 0:
            raise ValueError("domain_id must be non-negative.")
        if self.camera_key != DEFAULT_CAMERA_KEY:
            raise ValueError(
                f"X-VLA-RoboTwin2 training requires head camera {DEFAULT_CAMERA_KEY!r}."
            )

        info_path = self.root / "meta" / "info.json"
        if not info_path.is_file():
            raise FileNotFoundError(f"LeRobot v3 metadata not found: {info_path}")
        info = json.loads(info_path.read_text(encoding="utf-8"))
        if info.get("codebase_version") != "v3.0":
            raise ValueError(
                f"Expected LeRobot codebase_version='v3.0', got {info.get('codebase_version')!r}."
            )
        features = info.get("features", {})
        for key in (STATE_KEY, ACTION_KEY, self.camera_key):
            if key not in features:
                raise KeyError(f"LeRobot dataset is missing required feature {key!r}.")
        for key in (STATE_KEY, ACTION_KEY):
            if features[key].get("shape") != [16]:
                raise ValueError(f"{key} must have shape [16], got {features[key].get('shape')}.")
            names = features[key].get("names")
            flat_names = names[0] if isinstance(names, list) and len(names) == 1 else names
            if tuple(flat_names or ()) != EXPECTED_EE_NAMES:
                raise ValueError(f"{key} names do not match RoboDojo dual-arm EE layout.")

        fps = float(info["fps"])
        delta_timestamps = {
            ACTION_KEY: [(step + 1) / fps for step in range(self.num_actions)]
        }
        self.dataset = LeRobotDataset(
            repo_id=repo_id,
            root=self.root,
            episodes=list(episodes) if episodes is not None else None,
            delta_timestamps=delta_timestamps,
            video_backend=video_backend,
        )

        allowed = set(task_allowlist or [])
        self.indices = list(range(len(self.dataset)))
        if allowed:
            task_table = self.dataset.meta.tasks
            task_to_index = {
                str(task): int(row["task_index"])
                for task, row in task_table.iterrows()
            }
            unknown = allowed - set(task_to_index)
            if unknown:
                raise ValueError(f"Unknown exact task strings: {sorted(unknown)}")
            allowed_ids = {task_to_index[task] for task in allowed}
            # Filter the parquet-backed scalar column without decoding any video.
            task_ids = self.dataset.hf_dataset["task_index"]
            self.indices = [
                index for index in self.indices
                if int(torch.as_tensor(task_ids[index]).item()) in allowed_ids
            ]
        if not self.indices:
            raise ValueError("No LeRobot samples remain after episode/task filtering.")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        sample = self.dataset[self.indices[index]]
        state = torch.as_tensor(sample[STATE_KEY], dtype=torch.float32)
        action = torch.as_tensor(sample[ACTION_KEY], dtype=torch.float32)
        if action.shape != (self.num_actions, 16):
            raise ValueError(
                f"Expected future action chunk {(self.num_actions, 16)}, got {tuple(action.shape)}."
            )
        image = _normalize_head_image(sample[self.camera_key], training=self.training)
        return {
            "language_instruction": str(sample["task"]),
            "image_input": image.unsqueeze(0),
            "image_mask": torch.ones(1, dtype=torch.bool),
            "domain_id": torch.tensor(self.domain_id, dtype=torch.long),
            "proprio": ee16_open_to_xvla20_close(state),
            "action": ee16_open_to_xvla20_close(action),
        }
