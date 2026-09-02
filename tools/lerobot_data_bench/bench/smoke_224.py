#!/usr/bin/env python
"""Smoke: load ds224 (224 re-encode subset) eps 46-62 with torchcodec + return_uint8."""
# Local-only subset: skip hub revision sync (repo not on hub).
# Patch the get_safe_version references in both modules that call it.
import lerobot.datasets.dataset_metadata as _dmeta
import lerobot.datasets.lerobot_dataset as _lds


def _safe_version(repo_id, version):
    return version


_dmeta.get_safe_version = _safe_version
_lds.get_safe_version = _safe_version

from lerobot.datasets.lerobot_dataset import LeRobotDataset
import torch

fps = 25
delta = {
    "action": [t / fps for t in range(50)],
    "observation.images.cam_high": [0.0],
    "observation.images.cam_left_wrist": [0.0],
    "observation.images.cam_right_wrist": [0.0],
}
ds = LeRobotDataset(
    repo_id="real_lerobot_v30_joint_224p3",
    root="/cloud/cloud-ssd1/lerobot_bench/ds224/real_lerobot_v30_joint_224p3",
    episodes=list(range(46, 63)),
    delta_timestamps=delta,
    video_backend="torchcodec",
    return_uint8=True,
)
print("len", len(ds))
for i in [0, len(ds) // 2, len(ds) - 1]:
    item = ds[i]
    shapes = {k: (tuple(v.shape), str(v.dtype)) for k, v in item.items() if isinstance(v, torch.Tensor)}
    print(f"item[{i}]", shapes)
