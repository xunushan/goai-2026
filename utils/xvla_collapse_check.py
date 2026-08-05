#!/usr/bin/env python3
"""
XVLA 模型语义崩塌检测脚本（定制层）

本脚本基于通用检测框架 collapse_checker，针对 XVLA 模型的 checkpoint 格式差异
（vendored 原始格式 vs native 微调格式）提供定制配置和命令行入口。

如果你要适配其它模型，只需：
  1. 修改 _orig_key_for_ft() 中的 key 映射规则
  2. 修改 _DEFAULT_SVD_LAYERS 列表（待检查的层）
  3. 修改 _PROJECTIONS 列表（投影矩阵配置，含自定义转换函数）
  4. 保留 CLI 结构和报告格式

使用示例:
  # 完整报告（全部三项检查）
  python xvla_collapse_check.py /data/checkpoints/xvla-base/model.safetensors \
      /data/outputs/.../030000/pretrained_model/model.safetensors

  # 仅检查 embedding 余弦相似度
  python xvla_collapse_check.py --check embedding \
      /data/checkpoints/xvla-base/model.safetensors \
      /data/outputs/.../030000/pretrained_model/model.safetensors

  # 仅检查 SVD 有效秩
  python xvla_collapse_check.py --check svd \
      /data/checkpoints/xvla-base/model.safetensors \
      /data/outputs/.../030000/pretrained_model/model.safetensors
"""

from __future__ import annotations

import argparse
from typing import Callable

# 导入通用检测框架
from collapse_checker import (
    check_embedding_cosine,
    check_svd_rank,
    check_condition_number,
    full_report,
)


# ---------------------------------------------------------------------------
# XVLA 特定：Key 映射（vendored 原始格式 → native 微调格式）
# ---------------------------------------------------------------------------

def _orig_key_for_ft(ft_key: str, orig_keys: set[str]) -> str | None:
    """
    将微调后模型（native 格式）的 key 映射回原始模型（vendored 格式）的 key。

    XVLA 原始基线模型使用 vendored 格式的 Florence-2 权重命名，而微调后的
    checkpoint 使用 native（HuggingFace transformers）格式命名，两者存在以下差异：
      - image_projection: nn.Parameter → nn.Linear.weight（形状需转置）
      - language_model: encoder 路径缺少 .model. 层级
      - vision_tower: attention 模块缺少 .fn. 前缀
      - multi_modal_projector 子模块命名前缀差异
      - conv / ffn / norm 的层级命名差异

    参数:
        ft_key: 微调模型中的权重 key（native 格式）
        orig_keys: 原始模型中所有 key 的集合（vendored 格式）

    返回:
        对应的原始模型 key，若无法映射则返回 None
    """
    # 直接匹配（部分 key 命名一致）
    if ft_key in orig_keys:
        return ft_key

    # image_projection: nn.Parameter → nn.Linear.weight
    if ft_key == "model.vlm.multi_modal_projector.image_projection.weight":
        return "model.vlm.image_projection"

    # language_model: 路径中缺少 .model. 层级
    if "model.vlm.language_model.encoder." in ft_key:
        c = ft_key.replace(
            "model.vlm.language_model.encoder.",
            "model.vlm.language_model.model.encoder.",
        )
        if c in orig_keys:
            return c

    # vision_tower: attention 模块缺少 .fn. 前缀
    if "channel_attn." in ft_key or "window_attn." in ft_key:
        for s in (".qkv.", ".proj.", ".q_proj.", ".k_proj.",
                  ".v_proj.", ".out_proj."):
            if s in ft_key:
                c = ft_key.replace(s, ".fn" + s)
                if c in orig_keys:
                    return c

    # multi_modal_projector 子模块命名差异
    for old, new in (
        ("multi_modal_projector.image_position_embed", "image_pos_embed"),
        ("multi_modal_projector.image_proj_norm", "image_proj_norm"),
        ("multi_modal_projector.visual_temporal_embed", "visual_temporal_embed"),
    ):
        if old in ft_key:
            c = ft_key.replace(old, new)
            if c in orig_keys:
                return c
            # 某些 key 直接去掉前缀即可匹配
            c2 = ft_key.replace(f"multi_modal_projector.", "")
            if c2 in orig_keys:
                return c2

    # conv / ffn / norm 的层级命名差异
    if "convs." in ft_key and ".conv." in ft_key:
        c = ft_key.replace(".conv.", ".proj.")
        if c in orig_keys:
            return c
    if ".ffn.fc1." in ft_key:
        c = ft_key.replace(".ffn.fc1.", ".ffn.fn.net.0.")
        if c in orig_keys:
            return c
    if ".ffn.fc2." in ft_key:
        c = ft_key.replace(".ffn.fc2.", ".ffn.fn.net.3.")
        if c in orig_keys:
            return c
    if ".norm1." in ft_key:
        for attn in ("channel_attn", "window_attn"):
            c = ft_key.replace(".norm1.", f".{attn}.norm.")
            if c in orig_keys:
                return c
    if ".norm2." in ft_key:
        c = ft_key.replace(".norm2.", ".ffn.norm.")
        if c in orig_keys:
            return c

    return None


# ---------------------------------------------------------------------------
# XVLA 特定：默认待检查层列表（SVD 有效秩检查）
# ---------------------------------------------------------------------------

_DEFAULT_SVD_LAYERS = [
    # Vision Tower：channel attention 的 QKV 和投影层（浅层 vs 深层）
    "model.vlm.vision_tower.blocks.0.0.channel_block.channel_attn.qkv.weight",
    "model.vlm.vision_tower.blocks.3.0.channel_block.channel_attn.qkv.weight",
    "model.vlm.vision_tower.blocks.0.0.spatial_block.window_attn.qkv.weight",
    "model.vlm.vision_tower.blocks.3.0.spatial_block.window_attn.qkv.weight",
    "model.vlm.vision_tower.blocks.0.0.channel_block.channel_attn.proj.weight",
    "model.vlm.vision_tower.blocks.3.0.channel_block.channel_attn.proj.weight",
    # Language Model：self-attention 和 FFN 层
    "model.vlm.language_model.encoder.layers.0.self_attn.q_proj.weight",
    "model.vlm.language_model.encoder.layers.5.self_attn.q_proj.weight",
    "model.vlm.language_model.encoder.layers.0.fc1.weight",
    # Action Decoder 和 Cross Attention（下游策略层）
    "model.transformer.action_decoder.fc.weight",
    "model.transformer.cross_attn.0.q_proj.weight",
]


# ---------------------------------------------------------------------------
# XVLA 特定：投影矩阵配置（条件数检查）
# ---------------------------------------------------------------------------

_PROJECTIONS = [
    # (原始模型key, 微调模型key, 转换函数)
    # image_projection 在原始模型中是 nn.Parameter（形状 [d_lang, d_vis]），
    # 在微调后是 nn.Linear.weight（形状 [d_vis, d_lang]），需要转置才能对齐。
    # 转换函数由调用方提供，检查函数内部不做任何硬编码假设。
    (
        "model.vlm.image_projection",
        "model.vlm.multi_modal_projector.image_projection.weight",
        lambda t: t.transpose(0, 1).contiguous(),
    ),
]


# ---------------------------------------------------------------------------
# 命令行入口
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="XVLA 模型语义崩塌检测工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("orig", help="原始模型 checkpoint 路径（safetensors 格式）")
    parser.add_argument("ft", help="微调后模型 checkpoint 路径（safetensors 格式）")
    parser.add_argument(
        "--check",
        choices=["embedding", "svd", "condition", "all"],
        default="all",
        help="选择要运行的检查项（默认: all）",
    )
    parser.add_argument(
        "--n-sample",
        type=int,
        default=200,
        help="Embedding 余弦检查中采样的 token 行数（默认: 200）",
    )
    parser.add_argument(
        "--rank-drop-threshold",
        type=int,
        default=5,
        help="SVD 有效秩下降超过此值则判定崩塌（默认: 5）",
    )

    args = parser.parse_args()

    if args.check == "embedding":
        check_embedding_cosine(args.orig, args.ft, n_sample=args.n_sample)
    elif args.check == "svd":
        check_svd_rank(
            args.orig, args.ft,
            layers=_DEFAULT_SVD_LAYERS,
            key_mapper=_orig_key_for_ft,
            rank_drop_threshold=args.rank_drop_threshold,
        )
    elif args.check == "condition":
        check_condition_number(args.orig, args.ft, projections=_PROJECTIONS)
    else:
        full_report(
            args.orig, args.ft,
            svd_layers=_DEFAULT_SVD_LAYERS,
            key_mapper=_orig_key_for_ft,
            projections=_PROJECTIONS,
            n_sample=args.n_sample,
            rank_drop_threshold=args.rank_drop_threshold,
        )


if __name__ == "__main__":
    main()
