import os
import numpy as np
import shutil
import argparse
import cv2
import h5py
import dataclasses
from pathlib import Path
from typing import Any, Literal
from tqdm import tqdm

from scipy.spatial.transform import Rotation

from XPolicyLab.utils.process_data import decode_image_bit

from lerobot.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.datasets.lerobot_dataset import LeRobotDataset

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
    # real（real, piper_x）：joint 用 joint+gripper；ee 用 eef（7 维四元数直接 / 6 维欧拉角按 xyz 内旋转）+gripper
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

# 官方 lerobot_v30_ee 每臂命名：l_x..l_g / r_x..r_g（[x,y,z,qw,qx,qy,qz,g]）
EE_DIM_NAMES = ["x", "y", "z", "w", "wx", "wy", "wz", "g"]

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
    """real eef 统一为 [x,y,z,qw,qx,qy,qz]。

    - 7 维：已是四元数，直接返回。
    - 6 维：[x,y,z, 欧拉角×3]。欧拉角顺序为 xyz 内旋（已用 7 维任务 home 位姿
      四元数做 ground truth 反推验证为强候选，误差 0.14 rad）。
    """
    if eef.shape[1] == 7:
        return eef
    if eef.shape[1] == 6:
        pos = eef[:, :3]
        q_xyzw = Rotation.from_euler("xyz", eef[:, 3:6], degrees=False).as_quat()
        return np.concatenate([pos, q_xyzw[:, 3:4], q_xyzw[:, :3]], axis=1)
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

def create_empty_dataset(
    repo_id: str,
    robot_type: str,
    fps: int,
    motors: list[str],
    camera_names: list[str],
    mode: Literal["video", "image"] = "image",
    *,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
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
            "shape": (3, 480, 640),
            "names": ["channels", "height", "width"],
        }

    output_path = HF_LEROBOT_HOME / repo_id
    if output_path.exists():
        shutil.rmtree(output_path)

    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        robot_type=robot_type,
        features=features,
        use_videos=dataset_config.use_videos or mode == "video",
        tolerance_s=dataset_config.tolerance_s,
        image_writer_processes=dataset_config.image_writer_processes,
        image_writer_threads=dataset_config.image_writer_threads,
        video_backend=dataset_config.video_backend,
    )


def _load_compressed_images(dataset: h5py.Dataset) -> np.ndarray:
    # h5py Dataset 不是 numpy 数组，先转成 ndarray 再交给 decode_image_bit
    # （colors 为 (T,) 的 |S* 字节串数组，转 numpy 后 dtype.kind='S'，走 sequence 分支）
    return np.asarray(decode_image_bit(np.asarray(dataset)))


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


def load_data(ep_path: Path, schema: dict, action_type: str) -> dict[str, Any]:
    parts_key = "state_parts_ee" if action_type == "ee" else "state_parts_joint"
    parts = schema[parts_key]
    if parts is None:
        raise ValueError(f"action_type='{action_type}' not supported by schema {schema!r}")

    with h5py.File(ep_path, "r") as ep:
        left = _extract_arm_parts(ep, parts[0])
        right = _extract_arm_parts(ep, parts[1])
        state = np.concatenate([left, right], axis=1).astype(np.float32)
        action = _make_action_from_state(state)

        images = {}
        for source_keys, output_name in schema["cameras"].items():
            source = _get_nested(ep, *source_keys)
            if source is not None:
                images[output_name] = _load_compressed_images(source)

        raw_instruction = None
        if schema["instruction"] == "hdf5":
            for key in ("instruction", "instructions"):
                dataset = _get_nested(ep, key)
                if dataset is not None:
                    raw_instruction = dataset[()]
                    break
            if isinstance(raw_instruction, bytes):
                raw_instruction = raw_instruction.decode("utf-8")

    return {
        "images": images,
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
        "--skip_videos",
        action="store_true",
        help="Do not add image frames (video keys become empty in parquet). "
             "Videos can be soft-linked from the joint dataset afterwards.",
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

    dataset = create_empty_dataset(
        repo_id=repo_id,
        robot_type=robot_type,
        fps=fps,
        motors=motors,
        camera_names=list(schema["cameras"].values()),
        mode=mode,
        dataset_config=DEFAULT_DATASET_CONFIG,
    )

    # 收集 episode 文件，带任务名（real 的 instruction 需按任务查表）
    episode_files = []  # (task_dir, path)
    for raw_task_dir in raw_task_dirs:
        load_data_dir = DATA_ROOT / str(bench_name) / raw_task_dir / str(env_cfg_type)
        task_episode_files = sorted(load_data_dir.glob("data/episode_*.hdf5"))
        if not task_episode_files:
            task_episode_files = sorted(load_data_dir.glob("*.hdf5"))
        episode_files.extend((raw_task_dir, ep_file) for ep_file in task_episode_files)
    if expert_data_num is not None:
        episode_files = episode_files[:expert_data_num]

    for raw_task_dir, ep_file in tqdm(episode_files, desc="Processing episodes", unit="episode"):
        try:
            data = load_data(ep_file, schema, action_type)
            num_frames = data["state"].shape[0]

            if schema["instruction"] == "task_desc":
                instruction = REAL_TASK_DESCRIPTIONS.get(raw_task_dir, instruction)
            elif data["instructions"] is not None:
                instruction = data["instructions"]

            for i in range(num_frames):
                frame = {
                    "observation.state": data["state"][i],
                    "action": data["action"][i],
                    "task": instruction,
                }
                if not args.skip_videos:
                    for camera_name, images in data["images"].items():
                        frame[f"observation.images.{camera_name}"] = images[i]

                dataset.add_frame(frame)

            dataset.save_episode()
            dataset.hf_dataset = dataset.create_hf_dataset()
            tqdm.write(f"Finished {ep_file.name} with {num_frames} frames")
        except Exception as e:
            tqdm.write(f"Error processing episode {ep_file}: {e}")

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
