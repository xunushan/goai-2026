import gc
import os

# 必须在 PIL/cv2/lerobot 之前导入 torchcodec（触发 libtorchcodec_coreN 加载）。
# 若在 PIL 之后导入：Pillow 内嵌的 libjpeg.so.8 抢占绑定，conda libtiff.so.6 的
# jpeg12_write_raw_data@LIBJPEG_8.0 符号无法解析 → "Could not load libtorchcodec"。
import torchcodec  # noqa: F401

import numpy as np
import shutil
import argparse
import cv2
import h5py
import inspect
import dataclasses
from pathlib import Path
from typing import Any, Literal
from tqdm import tqdm

from scipy.spatial.transform import Rotation

from XPolicyLab.utils.process_data import decode_image_bit

from lerobot.datasets.lerobot_dataset import LeRobotDataset

try:  # lerobot 0.4.4
    from lerobot.datasets.lerobot_dataset import HF_LEROBOT_HOME
except ImportError:  # lerobot 0.6.0：HF_LEROBOT_HOME 迁移至 lerobot.utils.constants
    from lerobot.utils.constants import HF_LEROBOT_HOME

ROOT_PATH = Path(__file__).parent.parent.parent.parent.parent

# 数据根目录可通过环境变量覆盖（默认仍为 XPolicyLab/data）
DATA_ROOT = Path(os.environ.get("XDATA_ROOT", str(ROOT_PATH / "data")))

# ============================================================
# Schema：hdf5 字段映射到 info.json 规范名（按 (bench_name, env_cfg_type) 区分）
# state_parts_*: 每臂 [ (hdf5路径, 维度int 或 变换函数名str), ... ]
#   第二元素为 str 时表示对该字段先做变换（如 eef_to_pose），输出维度见 _TRANSFORM_OUT_DIMS
# ============================================================
SCHEMAS = {
    # sim（RoboDojo, arx_x5）：joint 用 joint_states+ee_joint_states；ee 用 ee_poses(7 四元数)+ee_joint_states
    ("RoboDojo", "arx_x5"): {
        "robot_type": "dual_x5",
        "fps": 25,
        "state_parts_joint": [
            [("state/left_arm_joint_states", 6), ("state/left_ee_joint_states", 1)],
            [("state/right_arm_joint_states", 6), ("state/right_ee_joint_states", 1)],
        ],
        "state_parts_ee": [
            [("state/left_ee_poses", 7), ("state/left_ee_joint_states", 1)],
            [("state/right_ee_poses", 7), ("state/right_ee_joint_states", 1)],
        ],
        "cameras": {
            ("vision", "cam_head", "colors"): "cam_high",
            ("vision", "cam_left_wrist", "colors"): "cam_left_wrist",
            ("vision", "cam_right_wrist", "colors"): "cam_right_wrist",
        },
        "instruction": "hdf5",
    },
    # real（real, piper_x）：joint 用 joint+gripper；ee 用 eef（7 维四元数 xyzw 直接 / 6 维欧拉角按 xyz 内旋转）+gripper
    ("real", "piper_x"): {
        "robot_type": "piper_x",
        "fps": 25,
        "state_parts_joint": [
            [("left_arm/joint", 6), ("left_arm/gripper", 1)],
            [("right_arm/joint", 6), ("right_arm/gripper", 1)],
        ],
        "state_parts_ee": [
            [("left_arm/eef", "eef_to_pose"), ("left_arm/gripper", 1)],
            [("right_arm/eef", "eef_to_pose"), ("right_arm/gripper", 1)],
        ],
        "cameras": {
            ("cam_head", "color"): "cam_high",
            ("cam_left_wrist", "color"): "cam_left_wrist",
            ("cam_right_wrist", "color"): "cam_right_wrist",
        },
        "instruction": "task_desc",
    },
}

# 变换函数名 -> 输出维度
_TRANSFORM_OUT_DIMS = {"eef_to_pose": 7}

# lerobot_v30_ee 每臂命名：l_x..l_g / r_x..r_g（[x,y,z,qx,qy,qz,qw,g]，四元数 xyzw）
EE_DIM_NAMES = ["x", "y", "z", "wx", "wy", "wz", "w", "g"]

# real 数据无 instruction 字段，按任务名查描述（用户提供）
REAL_TASK_DESCRIPTIONS = {
    "fill_pen_holder": "Pick up the pen holder and place all the pens into it.",
    "put_objects_into_basket": "Place all the objects on the table into the basket.",
    "stack_and_cover_blocks": "Stack the blocks on the table, then cover them with the cup.",
    "stack_bowls": "Stack the three bowls together.",
    "stand_up_bottles": "Stand the bottle upright.",
    "insert_charger": "Insert the charger plug into the power strip, then connect the charging cable to the plug.",
}


def eef_to_pose(eef: np.ndarray) -> np.ndarray:
    """real eef 统一为 [x,y,z,qx,qy,qz,qw]（四元数 xyzw，w 在最后）。

    - 7 维：已是 xyzw 四元数，直接返回。已用「跨任务关节匹配 + 位置校验」验证：
      xyz 欧拉 × xyzw 四元数在关节对应帧对上对齐（朝向误差中位数 0.59°，
      而按 wxyz 解释则 5.64°，见 docs/real_eef四元数顺序判定.md）。
    - 6 维：[x,y,z, 欧拉角×3]。欧拉角为 xyz 内旋，scipy 转出的四元数即 xyzw。
    """
    if eef.shape[1] == 7:
        return eef
    if eef.shape[1] == 6:
        pos = eef[:, :3]
        q_xyzw = Rotation.from_euler("xyz", eef[:, 3:6], degrees=False).as_quat()
        return np.concatenate([pos, q_xyzw], axis=1)
    raise ValueError(f"eef dims {eef.shape[1]} not supported")


def _arm_dim(parts) -> int:
    total = 0
    for _, dim in parts:
        if isinstance(dim, int):
            total += dim
        elif isinstance(dim, str):
            total += _TRANSFORM_OUT_DIMS[dim]
        else:
            raise ValueError(f"Unsupported part spec: {parts!r}")
    return total


def build_motors(schema: dict, action_type: str) -> list[str]:
    if action_type == "ee":
        # 官方 lerobot_v30_ee 命名：l_x..l_g / r_x..r_g（16 维）
        return [*[f"l_{n}" for n in EE_DIM_NAMES], *[f"r_{n}" for n in EE_DIM_NAMES]]
    if action_type == "joint":
        # 官方 lerobot_v30_joint 命名：left_joint_0..6 / right_joint_0..6（14 维）
        parts = schema["state_parts_joint"]
        motors = []
        for prefix, arm_parts in zip(("left", "right"), parts):
            motors.extend(f"{prefix}_joint_{i}" for i in range(_arm_dim(arm_parts)))
        return motors
    raise ValueError(f"Unknown action_type: {action_type}")


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = False
    tolerance_s: float = 0.0001
    image_writer_processes: int = 0
    image_writer_threads: int = 1
    video_backend: str | None = None


DEFAULT_DATASET_CONFIG = DatasetConfig()


def resize_with_pad(
    images: np.ndarray,
    height: int,
    width: int,
) -> np.ndarray:
    """按 pi0.5（openpi）的 resize_with_pad 做**位级一致**的 letterbox resize。

    直接调用 openpi/shared/image_tools.py::resize_with_pad——即 pi0.5 训练时
    ResizeImages(224,224) 数据变换所用的同一个函数（见 training/config.py），
    保证转换产物与 pi0.5 训练实际看到的像素完全一致，无任何复刻偏差。

    依赖（使用 --resize 时才需要）：
      - `pip install "jax[cpu]"`（openpi 的 resize_with_pad 基于 jax.image.resize）
      - openpi 可导入：将 openpi/src 加入 PYTHONPATH，或 `pip install -e Pi_05/openpi`
      - torch / beartype / jaxtyping：openpi 导入链需要（lerobot 环境通常已含 torch）
    """
    import jax.numpy as jnp
    import openpi.shared.image_tools as _openpi_image_tools

    if images.shape[1:3] == (height, width):
        return images
    resized = _openpi_image_tools.resize_with_pad(jnp.asarray(images), height, width)
    return np.asarray(resized)


def create_empty_dataset(
    repo_id: str,
    robot_type: str,
    fps: int,
    motors: list[str],
    camera_names: list[str],
    mode: Literal["video", "image"] = "image",
    *,
    image_shape: tuple[int, int] = (480, 640),
    video_codec: str = "h264",
    crf: int | None = 25,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
    streaming_encoding: bool = True,
) -> LeRobotDataset:

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": motors,
        },
        "action": {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [motors],
        },
    }

    for camera_name in camera_names:
        features[f"observation.images.{camera_name}"] = {
            "dtype": mode,
            "shape": (3, *image_shape),
            "names": ["channels", "height", "width"],
        }

    output_path = HF_LEROBOT_HOME / repo_id
    if output_path.exists():
        shutil.rmtree(output_path)

    create_kwargs = dict(
        repo_id=repo_id,
        fps=fps,
        robot_type=robot_type,
        features=features,
        use_videos=dataset_config.use_videos or mode == "video",
        tolerance_s=dataset_config.tolerance_s,
        image_writer_processes=dataset_config.image_writer_processes,
        image_writer_threads=dataset_config.image_writer_threads,
        video_backend=dataset_config.video_backend,
        streaming_encoding=streaming_encoding and mode == "video",
    )

    # 按 lerobot 版本选择视频编码参数传入方式（GOP 两版本均保持默认 2，不做改动）：
    #   - lerobot 0.6.0：create(..., rgb_encoder=RGBEncoderConfig(vcodec=..., crf=...))，crf 可调
    #   - lerobot 0.4.4：仅 create 的 vcodec 参数可传；crf 硬编码 30、GOP=2，不可调整，保持不动
    create_params = inspect.signature(LeRobotDataset.create).parameters
    if "rgb_encoder" in create_params:
        from lerobot.configs import RGBEncoderConfig

        create_kwargs["rgb_encoder"] = RGBEncoderConfig(vcodec=video_codec, crf=crf)
    elif "vcodec" in create_params:
        create_kwargs["vcodec"] = video_codec
        if crf not in (None, 30):
            print(f"[lerobot 0.4.x] create 的 crf 硬编码为 30 无法调整，忽略 --crf={crf}")
    else:
        print(
            f"[lerobot] 当前版本不支持自定义视频编码，忽略 vcodec={video_codec} crf={crf}"
        )

    return LeRobotDataset.create(**create_kwargs)


def _make_action_from_state(state: np.ndarray) -> np.ndarray:
    action = np.empty_like(state, dtype=np.float32)
    if len(state) == 1:
        action[0] = state[0]
        return action

    action[:-1] = state[1:]
    action[-1] = state[-1]
    return action


def _get_nested(h5_group, *keys, default=None):
    cur = h5_group
    for key in keys:
        if not isinstance(cur, h5py.Group) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _extract_arm_parts(ep: h5py.File, parts) -> np.ndarray:
    """从 hdf5 读取单臂各 part 并拼接为 (T, dim)。"""
    pieces = []
    for path, dim in parts:
        value = _get_nested(ep, *path.split("/"))
        if value is None:
            raise ValueError(f"Missing hdf5 path: {path}")
        arr = np.asarray(value)
        if isinstance(dim, str):
            arr = globals()[dim](arr)
            dim = _TRANSFORM_OUT_DIMS[dim]
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[:, None]
        if arr.shape[1] != dim:
            raise ValueError(f"{path}: expected dim {dim}, got {arr.shape[1]}")
        pieces.append(arr)
    return np.concatenate(pieces, axis=1)


def load_data(ep_path: Path, schema: dict, action_type: str, block_size: int = 256) -> dict[str, Any]:
    """加载单 episode。

    图像按 block_size 帧分批解码（生成器逐个 yield），避免一次性解码整 episode
    的 ~3.5GB 图像占用内存——连续转换多 episode 时，峰值叠加会触发服务器
    cgroup OOM。数值列（state/action）小，全量驻留内存。
    """
    parts_key = "state_parts_ee" if action_type == "ee" else "state_parts_joint"
    parts = schema[parts_key]
    if parts is None:
        raise ValueError(f"action_type='{action_type}' not supported by schema {schema!r}")

    ep = h5py.File(ep_path, "r")
    try:
        left = _extract_arm_parts(ep, parts[0])
        right = _extract_arm_parts(ep, parts[1])
        state = np.concatenate([left, right], axis=1).astype(np.float32)
        action = _make_action_from_state(state)

        # 只持有压缩字节（JPEG 共 ~几十 MB），解码在生成器里分块做
        images_compressed = {}
        for source_keys, output_name in schema["cameras"].items():
            source = _get_nested(ep, *source_keys)
            if source is not None:
                images_compressed[output_name] = np.asarray(source)

        raw_instruction = None
        if schema["instruction"] == "hdf5":
            for key in ("instruction", "instructions"):
                dataset = _get_nested(ep, key)
                if dataset is not None:
                    raw_instruction = dataset[()]
                    break
            if isinstance(raw_instruction, bytes):
                raw_instruction = raw_instruction.decode("utf-8")
    except Exception:
        ep.close()
        raise

    num_frames = state.shape[0]

    def gen_blocks():
        try:
            for s in range(0, num_frames, block_size):
                e = min(s + block_size, num_frames)
                block = {
                    output_name: np.asarray(decode_image_bit(bits[s:e]))
                    for output_name, bits in images_compressed.items()
                }
                yield s, block
                del block
        finally:
            ep.close()

    return {
        "image_blocks": gen_blocks(),
        "num_frames": num_frames,
        "state": state,
        "action": action,
        "velocity": None,
        "effort": None,
        "timestamps": None,
        "instructions": raw_instruction,
    }


def main():
    parser = argparse.ArgumentParser(description="Process some episodes.")
    parser.add_argument("bench_name", type=str, help="Dataset bench name (e.g., RoboDojo)")
    parser.add_argument("ckpt_name", type=str, help="Run name; also selects raw task dir under data/<bench>/")
    parser.add_argument("env_cfg_type", type=str, help="Environment config type (e.g., arx_x5)")
    parser.add_argument("action_type", type=str, help="Action type: joint or ee")
    parser.add_argument(
        "expert_data_num",
        type=str,
        nargs="?",
        default=None,
        help="Optional number of episodes to process; non-numeric values are treated as raw_task_dirs.",
    )
    parser.add_argument(
        "raw_task_dirs",
        type=str,
        nargs="?",
        default=None,
        help="Optional raw task dir or comma-separated dirs under data/<bench>/; defaults to ckpt_name.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["video", "image"],
        default="image",
        help="Whether to store images as videos or individual image files",
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default="Do your job.",
        help="Default instruction when not present in HDF5",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default=None,
        help="Override dataset repo_id (default: {bench_name}-{ckpt_name}-{env_cfg_type}-{action_type})",
    )
    parser.add_argument(
        "--resize",
        type=str,
        default=None,
        help=(
            "源图像按 pi0.5（openpi resize_with_pad）letterbox 方式 resize 后编码成视频，"
            "只输出一份 resize 数据集（单数据集=3 编码器，无并发丢帧风险），"
            "repo_id 加后缀 '-<H>x<W>'（需要原分辨率时另行不带 --resize 转换）。取值如 '224' 或 '224x168'。"
        ),
    )
    parser.add_argument(
        "--video-codec",
        type=str,
        default="h264",
        help=(
            "视频编码格式。lerobot 0.6.0 走 rgb_encoder.vcodec；0.4.4 走 create(vcodec=)。"
            "默认 h264。"
        ),
    )
    parser.add_argument(
        "--crf",
        type=int,
        default=25,
        help=(
            "视频编码质量（仅 lerobot 0.6.0 生效；0.4.4 的 create 硬编码 crf=30 无法调整）。"
            "默认 25。"
        ),
    )
    parser.add_argument(
        "--no-streaming-encoding",
        action="store_true",
        help=(
            "关闭流式视频编码：逐帧写临时 PNG、save_episode 统一编码（不丢帧，但会短暂占用 images/ 临时目录）。"
            "默认流式（与 lerobot 0.4.4 历史行为一致；0.6.0 单数据集 3 编码器实测不丢帧）。"
            "仅 --resize 双数据集并发（6 编码器）时 0.6.0 有界队列可能静默丢帧，帧数校验不符时用此开关回退。"
        ),
    )
    args = parser.parse_args()

    bench_name = args.bench_name
    ckpt_name = args.ckpt_name
    env_cfg_type = args.env_cfg_type
    action_type = args.action_type
    repo_id = args.repo_id or f"{bench_name}-{ckpt_name}-{env_cfg_type}-{action_type}"
    mode = args.mode
    instruction = args.instruction
    expert_data_num = None
    raw_task_dirs_arg = args.raw_task_dirs
    if args.expert_data_num is not None:
        try:
            expert_data_num = int(args.expert_data_num)
        except ValueError:
            if args.raw_task_dirs is not None:
                raise ValueError("raw_task_dirs was provided twice.") from None
            raw_task_dirs_arg = args.expert_data_num
    raw_task_dirs = [item.strip() for item in (raw_task_dirs_arg or ckpt_name).split(",") if item.strip()]

    schema = SCHEMAS[(bench_name, env_cfg_type)]
    robot_type = schema["robot_type"]
    fps = schema["fps"]
    motors = build_motors(schema, action_type)

    # 解析 --resize 目标尺寸："224" -> (224,224)，"224x168" -> (224,168)
    resize_size: tuple[int, int] | None = None
    if args.resize:
        parts = [p for p in args.resize.lower().split("x") if p]
        if len(parts) == 1:
            resize_size = (int(parts[0]), int(parts[0]))
        elif len(parts) == 2:
            resize_size = (int(parts[0]), int(parts[1]))
        else:
            raise ValueError(f"Invalid --resize value: {args.resize!r} (use '224' or '224x168')")

    # --resize 语义：源图像 resize 后编码成视频，只输出一份 resize 数据集（单数据集=3 编码器）。
    if resize_size is not None:
        effective_repo_id = f"{repo_id}-{resize_size[0]}x{resize_size[1]}"
        image_shape = resize_size
    else:
        effective_repo_id = repo_id
        image_shape = (480, 640)

    dataset = create_empty_dataset(
        repo_id=effective_repo_id,
        robot_type=robot_type,
        fps=fps,
        motors=motors,
        camera_names=list(schema["cameras"].values()),
        mode=mode,
        image_shape=image_shape,
        video_codec=args.video_codec,
        crf=args.crf,
        dataset_config=DEFAULT_DATASET_CONFIG,
        streaming_encoding=not args.no_streaming_encoding,
    )

    # 收集 episode 文件，带任务名（real 的 instruction 需按任务查表）。
    # expert_data_num 按任务切片：全局切片会导致只有第一个任务被转换。
    episode_files = []  # (task_dir, path)
    for raw_task_dir in raw_task_dirs:
        load_data_dir = DATA_ROOT / str(bench_name) / raw_task_dir / str(env_cfg_type)
        task_episode_files = sorted(load_data_dir.glob("data/episode_*.hdf5"))
        if not task_episode_files:
            task_episode_files = sorted(load_data_dir.glob("*.hdf5"))
        if expert_data_num is not None:
            task_episode_files = task_episode_files[:expert_data_num]
        episode_files.extend((raw_task_dir, ep_file) for ep_file in task_episode_files)

    for raw_task_dir, ep_file in tqdm(episode_files, desc="Processing episodes", unit="episode"):
        try:
            data = load_data(ep_file, schema, action_type)
            num_frames = data["num_frames"]

            if schema["instruction"] == "task_desc":
                instruction = REAL_TASK_DESCRIPTIONS.get(raw_task_dir, instruction)
            elif data["instructions"] is not None:
                instruction = data["instructions"]

            for start, block_images in data["image_blocks"]:
                # --resize 时整块批量 resize（每相机一次 openpi resize_with_pad 调用），
                # 避免逐帧触发 jax 编译/调度开销；否则直接用原始分辨率
                images_to_feed = block_images
                if resize_size is not None:
                    r_h, r_w = resize_size
                    images_to_feed = {
                        cam: resize_with_pad(np.asarray(imgs), r_h, r_w)
                        for cam, imgs in block_images.items()
                    }

                for i in range(len(next(iter(images_to_feed.values())))):
                    idx = start + i
                    frame = {
                        "observation.state": data["state"][idx],
                        "action": data["action"][idx],
                        "task": instruction,
                    }
                    for camera_name, images in images_to_feed.items():
                        frame[f"observation.images.{camera_name}"] = images[i]
                    dataset.add_frame(frame)

            # parallel_encoding=False：避免 save_episode 内 multiprocessing fork 与
            # 已初始化的多线程 JAX（--resize 时）冲突导致死锁（RuntimeWarning 提示），
            # 顺序编码输出与并行完全一致，仅转换耗时略增
            dataset.save_episode(parallel_encoding=False)
            tqdm.write(f"Finished {ep_file.name} with {num_frames} frames")
        except Exception as e:
            tqdm.write(f"Error processing episode {ep_file}: {e}")
        finally:
            # 关闭图像块生成器（触发 hdf5 句柄关闭），释放本 episode 内存
            data = None
            gc.collect()

    # 清理 video 模式下残留的空 images/ 目录（对齐官方结构：无 images/）
    images_dir = dataset.root / "images"
    if images_dir.exists():
        for p in sorted(images_dir.rglob("*"), reverse=True):
            if p.is_dir() and not any(p.iterdir()):
                p.rmdir()
        if images_dir.exists() and not any(images_dir.iterdir()):
            images_dir.rmdir()


if __name__ == "__main__":
    main()
