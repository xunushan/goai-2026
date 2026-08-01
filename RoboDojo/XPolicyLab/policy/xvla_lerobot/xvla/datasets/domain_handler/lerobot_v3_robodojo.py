"""RoboDojo ARX-X5 LeRobot v3 handler for X-VLA."""

from __future__ import annotations

import random
from collections.abc import Iterable

import numpy as np
import torch
from PIL import Image
from scipy.interpolate import interp1d

from ..utils import quat_wxyz_to_rotate6d
from .base import DomainHandler


STATE_KEY = "observation.state"
ACTION_KEY = "action"
DEFAULT_CAMERA_KEYS = (
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)
EXPECTED_EE_NAMES = (
    "l_x", "l_y", "l_z", "l_w", "l_wx", "l_wy", "l_wz", "l_g",
    "r_x", "r_y", "r_z", "r_w", "r_wx", "r_wy", "r_wz", "r_g",
)


def ee16_wxyz_to_xvla20(value: np.ndarray) -> np.ndarray:
    """Convert dual-arm xyz+wxyz+gripper arrays to X-VLA EE6D layout."""
    value = np.asarray(value, dtype=np.float32)
    if value.shape[-1] != 16:
        raise ValueError(f"Expected ARX-X5 EE data with 16 values, got {value.shape}.")
    if not np.isfinite(value).all():
        raise ValueError("ARX-X5 EE data contains NaN or Inf.")

    arms = []
    for offset in (0, 8):
        quaternion = value[..., offset + 3 : offset + 7]
        norm = np.linalg.norm(quaternion, axis=-1, keepdims=True)
        if np.any(np.abs(norm - 1.0) > 0.05):
            raise ValueError("ARX-X5 quaternion norm differs from 1 by more than 0.05.")
        quaternion = quaternion / np.clip(norm, 1e-8, None)
        arms.append(
            np.concatenate(
                [
                    value[..., offset : offset + 3],
                    quat_wxyz_to_rotate6d(quaternion),
                    value[..., offset + 7 : offset + 8],
                ],
                axis=-1,
            )
        )
    return np.concatenate(arms, axis=-1).astype(np.float32, copy=False)


class RoboDojoLeRobotV3EEHandler(DomainHandler):
    """Yield one-second, 30-anchor samples from selected LeRobot episodes."""

    dataset_name = "RoboDojo_LerobotV3_ARX_EE"

    def __init__(self, meta: dict, num_views: int) -> None:
        super().__init__(meta, num_views)
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        if num_views != 3:
            raise ValueError(f"ARX-X5 training requires num_views=3, got {num_views}.")
        self.camera_keys = tuple(meta.get("observation_key", DEFAULT_CAMERA_KEYS))
        if self.camera_keys != DEFAULT_CAMERA_KEYS:
            raise ValueError(
                "observation_key must be ordered as cam_high, cam_left_wrist, "
                f"cam_right_wrist; got {self.camera_keys}."
            )
        self.fps = float(meta.get("fps", 25.0))
        self.query_duration = float(meta.get("query_duration", 1.0))
        if not np.isclose(self.fps, 25.0):
            raise ValueError(f"ARX-X5 LeRobot data must use its real 25 Hz rate, got {self.fps}.")
        if not np.isclose(self.query_duration, 1.0):
            raise ValueError(
                "domain_id=6 training uses a one-second action horizon; "
                f"got query_duration={self.query_duration}."
            )

        episode_ids = [int(value) for value in meta["datalist"]]
        self.dataset = LeRobotDataset(
            repo_id=str(meta.get("repo_id", "lerobot_v30_ee")),
            root=meta["dataset_root"],
            episodes=episode_ids,
            video_backend=str(meta.get("video_backend", "pyav")),
        )
        features = self.dataset.meta.info["features"]
        for key in (STATE_KEY, ACTION_KEY, *self.camera_keys):
            if key not in features:
                raise KeyError(f"LeRobot dataset is missing required feature {key!r}.")
        for key in (STATE_KEY, ACTION_KEY):
            if tuple(features[key].get("shape", ())) != (16,):
                raise ValueError(f"{key} must have shape [16], got {features[key].get('shape')}.")
            names = features[key].get("names")
            flat_names = names[0] if isinstance(names, list) and len(names) == 1 else names
            if tuple(flat_names or ()) != EXPECTED_EE_NAMES:
                raise ValueError(f"{key} does not use the expected ARX-X5 EE layout.")

        frame_episode_ids = np.asarray(self.dataset.hf_dataset["episode_index"], dtype=np.int64)
        self.episode_rows = {
            episode_id: np.flatnonzero(frame_episode_ids == episode_id).tolist()
            for episode_id in episode_ids
        }
        missing = [episode_id for episode_id, rows in self.episode_rows.items() if not rows]
        if missing:
            raise ValueError(f"Selected episodes are absent from LeRobot data: {missing}.")

    @staticmethod
    def _to_pil(image: torch.Tensor) -> Image.Image:
        image = torch.as_tensor(image).detach().cpu()
        if image.ndim != 3 or image.shape[0] != 3:
            raise ValueError(f"Expected CHW RGB image, got {tuple(image.shape)}.")
        array = image.permute(1, 2, 0).numpy()
        if np.issubdtype(array.dtype, np.floating):
            if float(np.nanmax(array)) <= 1.5:
                array = array * 255.0
            array = np.clip(array, 0, 255).astype(np.uint8)
        return Image.fromarray(array)

    def iter_episode(
        self,
        traj_idx: int,
        *,
        num_actions: int,
        training: bool,
        image_aug,
        action_mode,
        lang_aug_map: dict | None,
        **kwargs,
    ) -> Iterable[dict]:
        if action_mode != "arx_ee6d":
            raise ValueError(f"ARX-X5 handler requires action_mode='arx_ee6d', got {action_mode!r}.")
        if num_actions != 30:
            raise ValueError(f"ARX-X5 handler requires 30 action anchors, got {num_actions}.")

        episode_id = int(self.meta["datalist"][traj_idx])
        rows = self.episode_rows[episode_id]
        raw = self.dataset.hf_dataset[rows]
        timestamps = np.asarray(raw["timestamp"], dtype=np.float64)
        if timestamps.ndim != 1 or len(timestamps) != len(rows):
            raise ValueError(f"Episode {episode_id} has invalid timestamps.")
        if np.any(np.diff(timestamps) <= 0):
            raise ValueError(f"Episode {episode_id} timestamps are not strictly increasing.")

        states = ee16_wxyz_to_xvla20(np.asarray(raw[STATE_KEY], dtype=np.float32))
        actions = ee16_wxyz_to_xvla20(np.asarray(raw[ACTION_KEY], dtype=np.float32))
        interpolator = interp1d(timestamps, actions, axis=0, bounds_error=True)
        last_start_time = float(timestamps[-1] - self.query_duration)
        candidate_indices = np.flatnonzero(timestamps <= last_start_time).tolist()
        if training:
            random.shuffle(candidate_indices)

        allowed_tasks = set(self.meta.get("tasks", []))
        image_mask = torch.ones(self.num_views, dtype=torch.bool)
        for local_index in candidate_indices:
            sample = self.dataset[rows[local_index]]
            instruction = str(sample["task"])
            if allowed_tasks and instruction not in allowed_tasks:
                continue
            if training and lang_aug_map and instruction in lang_aug_map:
                instruction = random.choice(lang_aug_map[instruction])

            current_time = float(timestamps[local_index])
            query_times = current_time + (
                np.arange(1, num_actions + 1, dtype=np.float64)
                * self.query_duration
                / num_actions
            )
            if query_times[-1] > timestamps[-1]:
                raise ValueError(
                    f"Episode {episode_id} candidate {local_index} lacks a complete "
                    f"{self.query_duration:.3f}s future window."
                )
            future_actions = np.asarray(interpolator(query_times), dtype=np.float32)
            abs_trajectory = np.concatenate(
                [states[local_index : local_index + 1], future_actions], axis=0
            )
            images = [
                image_aug(self._to_pil(sample[key]))
                for key in self.camera_keys
            ]
            yield {
                "language_instruction": instruction,
                "image_input": torch.stack(images, dim=0),
                "image_mask": image_mask.clone(),
                "abs_trajectory": torch.from_numpy(abs_trajectory),
            }


__all__ = ["RoboDojoLeRobotV3EEHandler", "ee16_wxyz_to_xvla20"]
