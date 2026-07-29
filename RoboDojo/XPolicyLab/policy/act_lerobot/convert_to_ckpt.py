#!/usr/bin/env python3
"""将 LeRobot safetensors checkpoint 转换为 PyTorch .ckpt 格式。

用法:
    python convert_to_ckpt.py <pretrained_model_dir> [output.ckpt]

示例:
    # 推荐: 直接传 checkpoints/last 软链接，自动跟随到最新 checkpoint
    python convert_to_ckpt.py /workspace/outputs/act_xxx/checkpoints/last
    # 也可传具体的 checkpoint 步骤目录
    python convert_to_ckpt.py /workspace/outputs/act_xxx/checkpoints/010000/pretrained_model
    python convert_to_ckpt.py /workspace/outputs/act_xxx/checkpoints/010000/pretrained_model /workspace/model.ckpt
"""
import sys
import json
from pathlib import Path

from safetensors.torch import load_file


def convert(pretrained_dir: Path | str, output_path: Path | str | None = None) -> Path:
    """将 safetensors 合并为单个 .ckpt 文件。

    支持三种输入路径（自动解析）:
      - checkpoints/last 软链接目录: 自动跟随软链接到最新 checkpoint
      - checkpoint 步骤目录 (含 pretrained_model/): 如 checkpoints/000100
      - pretrained_model 目录本身

    输出位置（未显式指定 output_path 时）:
      - 输入为 checkpoints/last  → 输出到 output_dir 根目录 (last 的上一级) 的 model.ckpt
      - 其他输入 → 输出到 pretrained_model 目录下的 model.ckpt

    Args:
        pretrained_dir: checkpoints/last 目录、checkpoint 步骤目录、或 pretrained_model 目录
        output_path: 输出 .ckpt 路径，默认按上述规则自动决定

    Returns:
        实际输出的 .ckpt 路径
    """
    input_dir = Path(pretrained_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"路径不存在: {input_dir}")

    # 判断输入是否为 checkpoints/last 软链接
    is_last_link = input_dir.is_symlink() and input_dir.name == "last"

    # 解析到 pretrained_model 目录
    resolved = input_dir.resolve()
    if (resolved / "pretrained_model").is_dir():
        pretrained_dir = resolved / "pretrained_model"
    elif (input_dir / "pretrained_model").is_dir():
        pretrained_dir = input_dir / "pretrained_model"
    else:
        pretrained_dir = resolved

    if not pretrained_dir.is_dir():
        raise FileNotFoundError(f"无法定位 pretrained_model 目录: {pretrained_dir}")

    # 合并所有 safetensors 文件
    state_dict = {}
    for sf in sorted(pretrained_dir.glob("*.safetensors")):
        print(f"加载 {sf.name} ...")
        tensors = load_file(str(sf))
        state_dict.update(tensors)

    # 读取 config.json（如有）
    config_file = pretrained_dir / "config.json"
    checkpoint = {"state_dict": state_dict}
    if config_file.exists():
        with open(config_file) as f:
            checkpoint["config"] = json.load(f)
        print(f"已包含 config.json")

    # 输出路径: 若为 last 软链接则输出到上一级 (output_dir 根目录)
    if output_path is None:
        if is_last_link:
            output_path = input_dir.parent / "model.ckpt"
        else:
            output_path = pretrained_dir / "model.ckpt"
    else:
        output_path = Path(output_path)

    # 保存为 PyTorch checkpoint
    import torch
    torch.save(checkpoint, output_path)
    print(f"\n转换完成: {output_path}")
    print(f"  张量数量: {len(state_dict)}")
    print(f"  文件大小: {output_path.stat().st_size / 1024**2:.2f} MB")
    return output_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    pretrained_dir = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    convert(pretrained_dir, output_path)
