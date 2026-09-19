"""仿真图片落盘（策略服务端调试用，deploy.yml `save_images` 段控制）。

存什么：调用方传入的**模型输入图像**（encode_obs 产出的 images 列表，即
camera_names 顺序、ensure_hwc_uint8 已经解码成 HWC uint8 RGB 的那一份），
落在 model.py 的 _prep_env —— 因此存下来的像素与喂进模型的像素严格同源，
排查「模型到底看到了什么画面」时不需要再对流式协议里的压缩字节做二次解读。

目录结构：
    <root>/<服务启动时间戳>_<task_name>/env<env_idx>_<episode_idx>/<相机名>/<帧号>.jpg

- 最外层带服务启动时间戳：同一台机器上同 task 反复评测不会互相覆盖；
- 每个 episode 一个文件夹（episode_idx 由 eval client 逐 env 生成，多 env 并行
  评测各有各的目录），前缀 env<env_idx> 便于回溯是哪台 env；
- 每路相机一个子文件夹，名字取 deploy.yml 的 camera_names；未配置 camera_names
  时按位置退化为 view_0 / view_1…；
- 帧号是该 episode 内的第几次推理（每次重规划一张，从 000000 起）。

写盘时机与代价：每次重规划前同步写 JPEG（q=90，480x640 约 30~50KB）。批评测
（eval_batch）下客户端只把「需要重规划」的 env 观测发给服务端，所以存到的是每次
重规划时的画面，**不含 chunk 中间步**；单 env 路径才每步都有观测，但保存点仍在
推理处，行为一致。写盘失败只告警一次、不中断评测（调试功能不得拖垮评测）。

enabled=false 时本模块完全不碰磁盘（save_observation 首行即返回）。
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image

DEFAULT_ROOT = "/data/outputs/sim_images"


@dataclass(frozen=True)
class SaveImagesConfig:
    """deploy.yml 的 `save_images` 配置段解析结果（仅配置，无状态）。"""

    enabled: bool = False
    root: str = DEFAULT_ROOT
    jpeg_quality: int = 90

    def __post_init__(self) -> None:
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError(
                "save_images.jpeg_quality must be within [1,100], got "
                f"{self.jpeg_quality}"
            )
        if not str(self.root).strip():
            raise ValueError("save_images.root must not be empty")

    @classmethod
    def from_model_cfg(cls, model_cfg: dict[str, Any]) -> "SaveImagesConfig":
        raw = model_cfg.get("save_images") or {}
        if not isinstance(raw, dict):
            raise TypeError(
                f"save_images config must be a dict, got {type(raw).__name__}"
            )
        enabled_raw = raw.get("enabled", False)
        if isinstance(enabled_raw, str):
            enabled = enabled_raw.strip().lower() not in (
                "",
                "0",
                "false",
                "no",
                "off",
                "null",
                "none",
            )
        else:
            enabled = bool(enabled_raw)
        return cls(
            enabled=enabled,
            root=str(raw.get("root") or DEFAULT_ROOT),
            jpeg_quality=int(raw.get("jpeg_quality", 90)),
        )


class EpisodeImageWriter:
    """按 episode / 相机分目录落 JPEG。

    跨请求状态只有「每个 episode 已写到第几帧」的计数；reset() 清空计数，
    不与服务端其它状态耦合。
    """

    def __init__(
        self,
        config: SaveImagesConfig,
        *,
        task_name: str | None,
        camera_names: Sequence[str] | None = None,
        run_timestamp: str | None = None,
    ) -> None:
        self.config = config
        self.camera_names = [str(name) for name in (camera_names or [])]
        self.run_timestamp = run_timestamp or datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )
        self.run_dir = Path(self.config.root) / (
            f"{self.run_timestamp}_{_slug(task_name) or 'unknown_task'}"
        )
        self._frame_index: dict[str, int] = {}
        self._shape_warned = False
        self._write_failed = False

    @property
    def active(self) -> bool:
        return self.config.enabled

    def describe(self) -> str:
        """启动日志用：落盘根目录，或 'off'。"""
        return str(self.run_dir) if self.config.enabled else "off"

    def save_observation(
        self,
        *,
        episode_idx: Any,
        env_idx: int,
        images: Sequence[Any] | None,
    ) -> list[Path]:
        """落一次推理的输入图像，返回实际写出的路径列表（关闭时返回空）。"""
        if not self.config.enabled or images is None:
            return []
        images = list(images)
        if not images:
            return []
        if len(images) != len(self.camera_names) and not self._shape_warned:
            self._shape_warned = True
            print(
                "[x_vla][images] camera_names "
                f"({len(self.camera_names)}) != images ({len(images)}); "
                "多出的视角落进 view_<index>/",
                flush=True,
            )

        episode_key = f"env{int(env_idx)}_{_slug(episode_idx) or 'unknown_episode'}"
        frame = self._frame_index.get(episode_key, 0)
        episode_dir = self.run_dir / episode_key
        written: list[Path] = []
        for index, image in enumerate(images):
            path = episode_dir / self._camera_dir(index) / f"{frame:06d}.jpg"
            if self._write_jpeg(path, image):
                written.append(path)
        self._frame_index[episode_key] = frame + 1
        return written

    def reset(self) -> None:
        """episode 边界重置帧号（落盘目录与已写文件不动）。"""
        self._frame_index = {}

    def _camera_dir(self, index: int) -> str:
        name = (
            self.camera_names[index]
            if index < len(self.camera_names)
            else f"view_{index}"
        )
        return _slug(name) or f"view_{index}"

    def _write_jpeg(self, path: Path, image: Any) -> bool:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(_as_uint8_hwc(image)).save(
                path, format="JPEG", quality=self.config.jpeg_quality
            )
            return True
        except Exception as exc:  # 调试功能：写盘失败不得中断评测
            if not self._write_failed:
                self._write_failed = True
                print(
                    f"[x_vla][images] save failed ({path}): {exc!r}; "
                    "further failures will not be reported",
                    flush=True,
                )
            return False


def _as_uint8_hwc(image: Any) -> np.ndarray:
    """encode_obs 的产物已是 HWC uint8 RGB；这里只兜底单通道与非 uint8 输入。"""
    array = np.asarray(image)
    if array.ndim == 3 and array.shape[2] == 1:
        array = array[:, :, 0]
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return array


def _slug(value: Any) -> str:
    """目录名安全化：非 [0-9A-Za-z._-] 一律换成下划线。"""
    if value is None:
        return ""
    return re.sub(r"[^0-9A-Za-z._-]+", "_", str(value)).strip("_")
