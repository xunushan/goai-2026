"""策略服务端图片落盘（deploy.yml `save_images` 段控制，仿真与真机共用）。

存什么：调用方传入的**模型输入图像**（encode_obs 产出的 images 列表，即
camera_names 顺序、ensure_hwc_uint8 已经解码成 HWC uint8 RGB 的那一份），
落在 model.py 的 _prep_env —— 因此存下来的像素与喂进模型的像素严格同源，
排查「模型到底看到了什么画面」时不需要再对流式协议里的压缩字节做二次解读。

目录结构：
    <root>/<服务启动时间戳>_<task_name>/env<env_idx>_<episode>/<相机名>/<帧号>.jpg

- 最外层带服务启动时间戳：同一台机器上同 task 反复评测不会互相覆盖；
- 每个 episode 一个文件夹（episode_idx 由 eval client 逐 env 生成，多 env 并行
  评测各有各的目录），前缀 env<env_idx> 便于回溯是哪台 env；
- **客户端不发 episode_idx 时（真机测评即如此）自己编号**，见下节「episode 编号」；
- 每路相机一个子文件夹，名字取 deploy.yml 的 camera_names；未配置 camera_names
  时按位置退化为 view_0 / view_1…；
- 帧号是该 episode 内的第几次推理（每次重规划一张，从 000000 起）。

episode 编号（客户端不发 episode_idx 时）：真机客户端只发观测、不带 episode 标识，
落盘器就用**推理计数 request 归零**当 episode 边界——同一 episode 内 request 单调
递增，回到 0 即新一轮（服务端 model.py 只在 reset() 与进程启动时清 0）。每次边界
生成一个 `ep<序号>_<4 位随机>` 编号，文件夹因此按 episode 隔离、且序号可排序。

task 名（真机客户端只发指令、不发 task_name）：按 `configs/real_task_instruction.json`
把指令映射成 task_name（见 resolve_task_name），一个服务实例里跑多个任务时各任务
各占一个 `<root>/<时间戳>_<task>` 目录；映射不到就退回配置里的 task_name（服务启动
时 `--task_name` 传入的那个），行为与从前一致。

写盘时机与代价：每次重规划前同步写 JPEG（q=90，480x640 约 30~50KB）。批评测
（eval_batch）下客户端只把「需要重规划」的 env 观测发给服务端，所以存到的是每次
重规划时的画面，**不含 chunk 中间步**；单 env 路径才每步都有观测，但保存点仍在
推理处，行为一致。写盘失败只告警一次、不中断评测（调试功能不得拖垮评测）。

enabled=false 时本模块完全不碰磁盘（save_observation 首行即返回）。
"""
from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

DEFAULT_ROOT = "/data/outputs/sim_images"
# 指令 → task_name 的映射表（仓库根 configs/ 下，与训练侧 Pi_05 用的同一份）。
# save_images.py 在 <repo>/RoboDojo/XPolicyLab/policy/X_VLA/ 下：parents 依次是
# X_VLA、policy、XPolicyLab、RoboDojo，再上一层才是仓库根（configs 在仓库根下）。
DEFAULT_TASK_INSTRUCTION_JSON = str(
    Path(__file__).resolve().parents[4] / "configs" / "real_task_instruction.json"
)


@dataclass(frozen=True)
class SaveImagesConfig:
    """deploy.yml 的 `save_images` 配置段解析结果（仅配置，无状态）。"""

    enabled: bool = False
    root: str = DEFAULT_ROOT
    jpeg_quality: int = 90
    # 指令 → task_name 映射表路径；默认仓库根 configs/real_task_instruction.json。
    # 换一份 JSON（例如仿真的 configs/siml_task_instruction.json）或改路径都走这里。
    task_instruction_json: str = DEFAULT_TASK_INSTRUCTION_JSON

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
            task_instruction_json=str(
                raw.get("task_instruction_json") or DEFAULT_TASK_INSTRUCTION_JSON
            ),
        )


class EpisodeImageWriter:
    """按 episode / 相机分目录落 JPEG。

    跨请求状态只有「每个 episode 已写到第几帧」的计数，以及客户端不发 episode_idx
    时自己编的号；reset() 把这两者都推进到下一个 episode，不与服务端其它状态耦合。
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
        # 客户端没给 task_name（或指令映射不到）时的兜底任务名
        self.task_name = task_name
        self.run_timestamp = run_timestamp or datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )
        # task_name → 落盘目录，按需生成：一个服务实例里可能先后跑多个任务，
        # 用不到的 task 不会在磁盘上留下空目录。
        self._task_dirs: dict[str, Path] = {}
        self._frame_index: dict[str, int] = {}
        # 客户端不发 episode_idx 时的自编号状态：已编到第几个、当前是哪个、
        # 上一次见到的 request（用来识别 request 归零这个 episode 边界）。
        self._episode_seq = 0
        self._synth_episode: str | None = None
        self._last_request_index: int | None = None
        self._shape_warned = False
        self._write_failed = False

    @property
    def active(self) -> bool:
        return self.config.enabled

    @property
    def run_dir(self) -> Path:
        """兜底任务的落盘目录（启动日志与映射不到 task_name 时用它）。"""
        return self._dir_for(self.task_name)

    def describe(self) -> str:
        """启动日志用：落盘根目录，或 'off'。"""
        return str(self.run_dir) if self.config.enabled else "off"

    def _dir_for(self, task_name: Any) -> Path:
        """某任务的落盘目录：<root>/<服务启动时间戳>_<task>（按需建，只记路径）。"""
        key = _slug(task_name) or "unknown_task"
        path = self._task_dirs.get(key)
        if path is None:
            path = Path(self.config.root) / f"{self.run_timestamp}_{key}"
            self._task_dirs[key] = path
        return path

    def save_observation(
        self,
        *,
        episode_idx: Any,
        env_idx: int,
        images: Sequence[Any] | None,
        request_index: int | None,
        task_name: str | None = None,
    ) -> list[Path]:
        """落一次推理的输入图像，返回实际写出的路径列表（关闭时返回空）。

        request_index 是服务端 model.py 的推理计数（同一 episode 内单调递增、
        episode 边界归零），只在客户端不发 episode_idx 时用来切 episode。
        它是必传的：调用方忘了传会直接 TypeError，而不是静默把多个 episode
        的图混进同一个目录。

        task_name 是本次观测对应的任务名（由指令经 resolve_task_name 映射而来）；
        不给或给空则落到构造时的兜底任务名。
        """
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

        run_dir = self._dir_for(task_name or self.task_name)
        episode = self._resolve_episode(episode_idx, request_index, run_dir)
        episode_key = f"env{int(env_idx)}_{episode}"
        frame = self._frame_index.get(episode_key, 0)
        episode_dir = run_dir / episode_key
        written: list[Path] = []
        for index, image in enumerate(images):
            path = episode_dir / self._camera_dir(index) / f"{frame:06d}.jpg"
            if self._write_jpeg(path, image):
                written.append(path)
        self._frame_index[episode_key] = frame + 1
        return written

    def reset(self) -> None:
        """episode 边界：帧号归零、下一次落盘换一个新编号（已写文件不动）。"""
        self._frame_index = {}
        self._synth_episode = None
        self._last_request_index = None

    def _resolve_episode(
        self, episode_idx: Any, request_index: int | None, run_dir: Path
    ) -> str:
        """这一帧属于哪个 episode 目录：客户端给了 id 就用它，否则自己编号。

        自编号的触发点有两个，取「或」——两者都表示新一轮 episode，且都只在前一次
        落盘之后才可能发生，所以同一个 episode 内无论落多少帧、多少个 env，只会
        生成一个编号：

        - 本次 request_index 归零而上一次不是 0（真机：服务端 request 单调递增，
          回 0 只可能是新 episode 或进程重启）；
        - 还没有编号（进程启动后的第一帧，或刚 reset 过）。
        """
        client_episode = _slug(episode_idx)
        if client_episode:
            return client_episode
        new_episode = request_index == 0 and self._last_request_index != 0
        self._last_request_index = request_index
        if new_episode or self._synth_episode is None:
            if request_index is None:
                # 调用方拿不到 request（理论上不会发生，接口是必传的）：
                # 别编号，退回旧的兜底目录名，免得把多个 episode 混成一个编号。
                return "unknown_episode"
            self._episode_seq += 1
            self._synth_episode = (
                f"ep{self._episode_seq:03d}_{uuid.uuid4().hex[:4]}"
            )
            print(
                f"[x_vla][images] new episode #{self._episode_seq} "
                f"({self._synth_episode}) -> {run_dir}",
                flush=True,
            )
        return self._synth_episode

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


def load_task_name_map(path: str | Path | None) -> dict[str, str]:
    """读指令→task_name 映射表，返回 {归一化后的指令: task_name}。

    源文件是仓库根 `configs/real_task_instruction.json`（训练侧 Pi_05 用的同一份），
    每条任务登记 original_instruction（平台发给服务端的原文）、modified_instruction
    （训练用的改写版）与 task_id；三个都能查到同一个 task_name，客户端发哪一版都能
    映射上。文件不存在只告警、不抛异常：这只影响图片目录名，落盘会退回配置里的
    task_name，不该让服务起不来；文件存在但格式不对则直接抛，免得静默用错映射。
    """
    if not path:
        return {}
    json_path = Path(path).expanduser()
    if not json_path.is_file():
        print(
            f"[x_vla][images] 指令→task_name 映射表不存在（{json_path}）："
            "图片目录将退回 deploy.yml 的 task_name",
            flush=True,
        )
        return {}
    with json_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    tasks = data.get("tasks") if isinstance(data, dict) else None
    if not isinstance(tasks, list):
        raise ValueError(f"映射表 {json_path} 缺少 tasks 列表")
    mapping: dict[str, str] = {}
    for entry in tasks:
        if not isinstance(entry, dict):
            raise TypeError(f"映射表 {json_path} 的 tasks 项必须是对象")
        task_name = str(entry.get("task_name") or "").strip()
        if not task_name:
            raise ValueError(f"映射表 {json_path} 有任务缺少 task_name")
        for key in (
            "original_instruction",
            "modified_instruction",
            "task_id",
            "task_name",
        ):
            alias = _normalize_text(entry.get(key))
            if alias:
                mapping[alias] = task_name
    return mapping


def resolve_task_name(
    instruction: Any, task_name_map: Mapping[str, str]
) -> str | None:
    """把客户端发来的指令查成 task_name；查不到返回 None（由调用方兜底）。"""
    key = _normalize_text(instruction)
    return task_name_map.get(key) if key else None


def _normalize_text(value: Any) -> str:
    """查表用的归一化：去首尾空白、压缩内部连续空白、忽略大小写。"""
    if not isinstance(value, str):
        return ""
    return re.sub(r"\s+", " ", value).strip().casefold()


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
