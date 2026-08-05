#!/usr/bin/env python3
"""
模型语义崩塌通用检测框架 (Model Collapse Checker)

本模块提供与模型结构无关的通用检查函数，用于检测微调后模型是否发生
语义崩塌（Semantic Collapse）。通过三个互补的指标，从 Embedding 空间、
权重矩阵秩、投影矩阵条件数三个层面诊断"多样性丧失"问题。

使用方式：
  1. 在你的模型配置文件中定义 key_mapper 和待检查层列表
  2. 调用本模块的 check_* 函数，传入配置
  3. 查看返回的 dict 和终端输出的诊断报告

【独立检查调用示例】
    from collapse_checker import check_embedding_cosine, check_svd_rank
    results = check_embedding_cosine("orig/model.safetensors", "ft/model.safetensors")

【完整报告调用示例（推荐，只打开一次文件）】
    from collapse_checker import full_report

    def my_mapper(ft_key, orig_keys):
        if ft_key in orig_keys:
            return ft_key
        # 添加你的映射规则...
        return None

    report = full_report(
        orig_path="orig/model.safetensors",
        ft_path="ft/model.safetensors",
        svd_layers=[
            "model.encoder.layers.0.self_attn.q_proj.weight",
            "model.encoder.layers.0.fc1.weight",
        ],
        key_mapper=my_mapper,
        projections=[
            ("model.proj", "model.proj.weight", None),
        ],
    )
"""

from __future__ import annotations

import torch
from safetensors import safe_open
from typing import Callable


# ===========================================================================
# 检查 1：Token Embedding 余弦相似度
# ===========================================================================

def _check_embedding_cosine_impl(
    orig_embed: torch.Tensor,
    ft_embed: torch.Tensor,
    n_sample: int = 200,
    threshold: float = 0.1,
) -> dict:
    """
    【纯计算逻辑】检查 token embedding 是否发生语义崩塌。

    指标直觉：
    Embedding 矩阵的每一行是一个 token 的向量表示。在健康的模型中，不同 token
    应该指向嵌入空间中不同的方向，因此任意两行之间的平均余弦相似度接近 0。
    如果发生崩塌，所有 token 向量被拉向同一个主导方向，行间夹角趋近于 0，
    off-diagonal 余弦相似度均值 → 1。此时模型看到不同输入却产生几乎相同的
    内部表示，下游任务无法区分词汇语义。
    """
    print("=" * 80)
    print("检查 1：Token Embedding 行间余弦相似度")
    print("=" * 80)
    print("直觉：若崩塌，所有 token 向量指向同一方向，cosine → 1；")
    print("      健康模型中不同 token 指向不同方向，cosine ≈ 0。")
    print()

    # 逐行归一化到单位长度，计算 pairwise 余弦矩阵
    orig_norm = orig_embed / orig_embed.norm(dim=1, keepdim=True).clamp(min=1e-8)
    ft_norm = ft_embed / ft_embed.norm(dim=1, keepdim=True).clamp(min=1e-8)

    idx = torch.randperm(orig_embed.shape[0])[:n_sample]
    orig_cos = orig_norm[idx] @ orig_norm[idx].T
    ft_cos = ft_norm[idx] @ ft_norm[idx].T

    # 排除对角线（自相似恒为 1）
    mask = ~torch.eye(n_sample, dtype=torch.bool)
    orig_cos_off = orig_cos[mask]
    ft_cos_off = ft_cos[mask]

    orig_mean = orig_cos_off.mean().item()
    ft_mean = ft_cos_off.mean().item()
    diff = ft_mean - orig_mean
    collapsed = diff > threshold

    print(f"  Embedding 形状: {orig_embed.shape}")
    print(f"  采样 token 数: {n_sample}")
    print(f"  原始模型 平均余弦: {orig_mean:.4f}  标准差: {orig_cos_off.std().item():.4f}  最大: {orig_cos_off.max().item():.4f}")
    print(f"  微调模型 平均余弦: {ft_mean:.4f}  标准差: {ft_cos_off.std().item():.4f}  最大: {ft_cos_off.max().item():.4f}")
    print(f"  变化量: {diff:+.4f}")
    if collapsed:
        print(f"  *** 崩塌警报：余弦相似度上升 {diff:.4f} > 阈值 {threshold} ***")
    else:
        print(f"  正常：无显著崩塌（变化 {diff:+.4f} <= {threshold}）")

    return {
        "orig_mean": orig_mean,
        "ft_mean": ft_mean,
        "diff": diff,
        "collapsed": collapsed,
    }


def check_embedding_cosine(
    orig_path: str,
    ft_path: str,
    n_sample: int = 200,
    threshold: float = 0.1,
    orig_embed_key: str | None = None,
    ft_embed_key: str | None = None,
) -> dict:
    """
    打开 checkpoint 文件，读取 embedding 权重，调用纯计算逻辑进行检查。

    通用调用示例：
      # 自动搜索包含 "embed_tokens.weight" 的 key（适用于大多数 LLM/VLM）
      result = check_embedding_cosine("orig/model.safetensors", "ft/model.safetensors")

      # 如果模型使用非标准命名，手动指定 key
      result = check_embedding_cosine(
          "orig/model.safetensors",
          "ft/model.safetensors",
          orig_embed_key="model.embed_tokens.weight",
          ft_embed_key="model.embed_tokens.weight",
      )

    参数:
        orig_path: 原始模型 checkpoint 路径（safetensors 格式）
        ft_path: 微调后模型 checkpoint 路径
        n_sample: 采样 token 行数（默认 200，覆盖大词表时建议增大）
        threshold: 崩塌判定阈值。若训练后平均余弦相比原始上升超过此值，
                   则判定发生崩塌（默认 0.1）
        orig_embed_key: 原始模型中 embedding 权重的完整 key，None 则自动搜索
        ft_embed_key: 微调模型中 embedding 权重的完整 key，None 则自动搜索

    返回:
        dict，包含 orig_mean, ft_mean, diff, collapsed 等字段
    """
    with safe_open(orig_path, framework="pt") as orig_f, \
         safe_open(ft_path, framework="pt") as ft_f:

        # 自动搜索 embed_tokens.weight（通用模式）
        if orig_embed_key is None:
            for k in orig_f.keys():
                if "embed_tokens.weight" in k:
                    orig_embed_key = k
                    break
        if ft_embed_key is None:
            for k in ft_f.keys():
                if "embed_tokens.weight" in k:
                    ft_embed_key = k
                    break

        if not orig_embed_key or not ft_embed_key:
            print("  错误：未在 checkpoint 中找到 embed_tokens.weight，")
            print("        请通过 orig_embed_key / ft_embed_key 手动指定。")
            return {"error": "embed_tokens not found"}

        orig_embed = orig_f.get_tensor(orig_embed_key).float()
        ft_embed = ft_f.get_tensor(ft_embed_key).float()

    return _check_embedding_cosine_impl(orig_embed, ft_embed, n_sample, threshold)


# ===========================================================================
# 检查 2：权重矩阵 SVD 有效秩
# ===========================================================================

def _check_svd_rank_impl(
    orig_tensors: dict[str, torch.Tensor],
    ft_tensors: dict[str, torch.Tensor],
    layer_mapping: dict[str, str],
    rank_threshold: float = 1e-3,
    rank_drop_threshold: int = 5,
) -> dict:
    """
    【纯计算逻辑】检查权重矩阵是否因训练而发生低秩退化（有效秩下降）。

    指标直觉：
    对权重矩阵 W 做 SVD 分解：W = U Σ V^T。奇异值 σ_i 反映矩阵在各个正交方向
    上的"能量"分布。健康模型的奇异值分布相对均匀，有效秩（显著非零的奇异值
    数量）接近满秩。若发生崩塌，矩阵行/列变得线性相关，W 可被少数几个大
    奇异值近似（W ≈ σ_1 u_1 v_1^T + ...），有效秩骤降。这意味着该层的前向
    传播退化为一个低维投影，大量参数冗余，表达能力丧失。
    """
    print("=" * 80)
    print("检查 2：权重矩阵 SVD 有效秩")
    print("=" * 80)
    print("直觉：若崩塌，权重矩阵行/列线性相关，可被少数奇异值近似；")
    print("      有效秩骤降意味着该层表达能力退化，信息在特定维度丢失。")
    print()

    print(f"{'层名称':<55} {'原始秩':>10} {'微调秩':>10} {'秩变化':>8} {'原始σ₁':>10} {'微调σ₁':>10}")
    print("-" * 110)

    results = []
    any_collapsed = False

    for ft_key, orig_key in layer_mapping.items():
        orig_t = orig_tensors[orig_key]
        ft_t = ft_tensors[ft_key]

        if orig_t.shape != ft_t.shape:
            short = ".".join(ft_key.split(".")[-3:])
            print(f"{short:<55} 形状不匹配: {orig_t.shape} vs {ft_t.shape}")
            continue

        # 展平为 2D 矩阵进行 SVD
        orig_flat = orig_t.reshape(orig_t.shape[0], -1)
        ft_flat = ft_t.reshape(ft_t.shape[0], -1)

        orig_sv = torch.linalg.svdvals(orig_flat)
        ft_sv = torch.linalg.svdvals(ft_flat)

        orig_rank = (orig_sv > orig_sv[0] * rank_threshold).sum().item()
        ft_rank = (ft_sv > ft_sv[0] * rank_threshold).sum().item()
        rank_drop = orig_rank - ft_rank

        short = ".".join(ft_key.split(".")[-4:])
        flag = " *** 崩塌" if rank_drop > rank_drop_threshold else ""
        print(f"{short:<55} {orig_rank:>10} {ft_rank:>10} {rank_drop:>+8} {orig_sv[0].item():>10.4f} {ft_sv[0].item():>10.4f}{flag}")

        layer_result = {
            "layer": short,
            "orig_rank": orig_rank,
            "ft_rank": ft_rank,
            "rank_drop": rank_drop,
            "collapsed": rank_drop > rank_drop_threshold,
        }
        results.append(layer_result)
        if layer_result["collapsed"]:
            any_collapsed = True

    if not any_collapsed:
        print(f"\n  正常：所有层有效秩下降不超过 {rank_drop_threshold}。")
    else:
        collapsed_layers = [r["layer"] for r in results if r["collapsed"]]
        print(f"\n  *** 崩塌警报：以下层有效秩显著下降：{', '.join(collapsed_layers)} ***")

    return {"layers": results, "any_collapsed": any_collapsed}


def check_svd_rank(
    orig_path: str,
    ft_path: str,
    layers: list[str],
    key_mapper: Callable[[str, set[str]], str | None],
    rank_threshold: float = 1e-3,
    rank_drop_threshold: int = 5,
) -> dict:
    """
    打开 checkpoint 文件，读取指定层的权重，调用纯计算逻辑进行检查。

    通用调用示例：
      # 定义你的 key 映射函数（处理 orig/ft 命名差异）
      def my_mapper(ft_key: str, orig_keys: set[str]) -> str | None:
          if ft_key in orig_keys:
              return ft_key
          # 添加你的映射规则...
          return None

      # 指定待检查的层（fine-tuned 格式命名）
      layers_to_check = [
          "model.encoder.layers.0.self_attn.q_proj.weight",
          "model.encoder.layers.0.fc1.weight",
      ]

      result = check_svd_rank(
          "orig/model.safetensors",
          "ft/model.safetensors",
          layers=layers_to_check,
          key_mapper=my_mapper,
          rank_drop_threshold=5,
      )

    参数:
        orig_path: 原始模型 checkpoint 路径
        ft_path: 微调后模型 checkpoint 路径
        layers: 待检查的权重 key 列表（使用微调后模型的命名格式）
        key_mapper: 函数 ft_key × orig_keys_set → orig_key | None，
                    用于处理原始模型与微调模型之间的 key 命名差异
        rank_threshold: 奇异值相对阈值。σ_i < σ_max × threshold 视为零（默认 1e-3）
        rank_drop_threshold: 若某层有效秩下降超过此值，则判定该层崩塌（默认 5）

    返回:
        dict，包含 layers 列表和 any_collapsed 标志
    """
    with safe_open(orig_path, framework="pt") as orig_f, \
         safe_open(ft_path, framework="pt") as ft_f:

        orig_keys = set(orig_f.keys())
        ft_keys = set(ft_f.keys())

        # 预加载所有需要的 tensor
        orig_tensors: dict[str, torch.Tensor] = {}
        ft_tensors: dict[str, torch.Tensor] = {}
        layer_mapping: dict[str, str] = {}

        for ft_key in layers:
            orig_key = key_mapper(ft_key, orig_keys)
            if orig_key is None or orig_key not in orig_keys or ft_key not in ft_keys:
                continue
            layer_mapping[ft_key] = orig_key
            orig_tensors[orig_key] = orig_f.get_tensor(orig_key).float()
            ft_tensors[ft_key] = ft_f.get_tensor(ft_key).float()

    return _check_svd_rank_impl(
        orig_tensors, ft_tensors, layer_mapping,
        rank_threshold=rank_threshold,
        rank_drop_threshold=rank_drop_threshold,
    )


# ===========================================================================
# 检查 3：投影矩阵条件数
# ===========================================================================

def _check_condition_number_impl(
    orig_tensors: dict[str, torch.Tensor],
    ft_tensors: dict[str, torch.Tensor],
    projection_configs: list[tuple[str, str, Callable | None]],
    cond_threshold: float = 100.0,
    cond_increase_threshold: float = 3.0,
) -> dict:
    """
    【纯计算逻辑】检查投影矩阵的条件数是否恶化。

    指标直觉：
    条件数 κ = σ_max / σ_min，衡量线性变换的"各向异性"程度。把投影矩阵 P
    看作一个椭球变换：输入的单位球经过 P 后被映射为一个椭球，σ_max 和 σ_min
    分别是椭球的最长轴和最短轴半径。
    - 健康状态：κ 较小（如 10~50），椭球接近球体，各个方向信息均匀保留。
    - 崩塌状态：κ 极大（如 1000+），椭球被压成一根"针"，某些方向被极度放大，
      另一些方向被压缩到接近零。这意味着不同输入的视觉特征经过投影后几乎落到
      语言嵌入空间的同一点，模态对齐失效。
    """
    print("=" * 80)
    print("检查 3：投影矩阵条件数")
    print("=" * 80)
    print("直觉：κ = σ_max / σ_min。若崩塌，单位球被压成针状椭球；")
    print("      某些方向信息被放大，另一些被压缩到零，输出同质化。")
    print()

    results = []
    any_collapsed = False

    for orig_key, ft_key, transform in projection_configs:
        orig_t = orig_tensors[orig_key]
        ft_t = ft_tensors[ft_key]

        # 调用方自定义转换（如转置、reshape 等），而非检查函数内部硬编码
        if transform is not None:
            orig_t = transform(orig_t)

        orig_sv = torch.linalg.svdvals(orig_t)
        ft_sv = torch.linalg.svdvals(ft_t)

        orig_cond = (orig_sv[0] / orig_sv[-1]).item()
        ft_cond = (ft_sv[0] / ft_sv[-1]).item()
        cond_ratio = ft_cond / orig_cond if orig_cond > 0 else float("inf")

        name = ft_key.split(".")[-2] if "." in ft_key else ft_key
        print(f"  {name}:")
        print(f"    原始模型 top-5 奇异值: {[f'{v:.4f}' for v in orig_sv[:5].tolist()]}")
        print(f"    微调模型 top-5 奇异值: {[f'{v:.4f}' for v in ft_sv[:5].tolist()]}")
        print(f"    原始条件数: {orig_cond:.2f}")
        print(f"    微调条件数: {ft_cond:.2f}")
        print(f"    条件数变化倍数: {cond_ratio:.2f}x")

        collapsed = cond_ratio > cond_increase_threshold or ft_cond > cond_threshold
        if collapsed:
            print(f"    *** 崩塌警报：条件数增长 {cond_ratio:.2f}x 或绝对值超过 {cond_threshold} ***")
        else:
            print(f"    正常：条件数稳定")

        results.append({
            "matrix": ft_key,
            "orig_cond": orig_cond,
            "ft_cond": ft_cond,
            "cond_ratio": cond_ratio,
            "collapsed": collapsed,
        })
        if collapsed:
            any_collapsed = True

    return {"matrices": results, "any_collapsed": any_collapsed}


def check_condition_number(
    orig_path: str,
    ft_path: str,
    projections: list[tuple[str, str, Callable | None]],
    cond_threshold: float = 100.0,
    cond_increase_threshold: float = 3.0,
) -> dict:
    """
    打开 checkpoint 文件，读取投影矩阵权重，调用纯计算逻辑进行检查。

    通用调用示例：
      # 定义投影矩阵列表：(原始模型key, 微调模型key, 转换函数)
      # 转换函数用于处理原始模型与微调模型中参数形状/维度不一致的情况
      # （如原始是 nn.Parameter，微调后是 nn.Linear.weight，需要转置）
      projections = [
          ("model.image_projection", "model.mm_projector.weight", lambda t: t.transpose(0, 1)),
          ("model.visual_proj", "model.visual_proj.weight", None),  # 无需转换
      ]

      result = check_condition_number(
          "orig/model.safetensors",
          "ft/model.safetensors",
          projections=projections,
          cond_threshold=100.0,
          cond_increase_threshold=3.0,
      )

    参数:
        orig_path: 原始模型 checkpoint 路径
        ft_path: 微调后模型 checkpoint 路径
        projections: 投影矩阵配置列表，每个元素为三元组：
                     (orig_key, ft_key, transform_fn)
                     transform_fn 为 None 表示无需转换；
                     否则应为 Callable[[torch.Tensor], torch.Tensor]，
                     用于将原始模型的 tensor 转换为可与微调模型对比的形式。
        cond_threshold: 绝对条件数阈值，超过此值视为病态矩阵（默认 100）
        cond_increase_threshold: 相对阈值，若 κ 相对原始增长超过此倍数则报警（默认 3）

    返回:
        dict，包含 matrices 列表和 any_collapsed 标志
    """
    with safe_open(orig_path, framework="pt") as orig_f, \
         safe_open(ft_path, framework="pt") as ft_f:

        orig_keys = set(orig_f.keys())
        ft_keys = set(ft_f.keys())

        # 预加载所有需要的 tensor
        orig_tensors: dict[str, torch.Tensor] = {}
        ft_tensors: dict[str, torch.Tensor] = {}
        projection_configs: list[tuple[str, str, Callable | None]] = []

        for orig_key, ft_key, transform in projections:
            if orig_key not in orig_keys or ft_key not in ft_keys:
                print(f"  跳过：{orig_key} 或 {ft_key} 未在 checkpoint 中找到")
                continue
            projection_configs.append((orig_key, ft_key, transform))
            orig_tensors[orig_key] = orig_f.get_tensor(orig_key).float()
            ft_tensors[ft_key] = ft_f.get_tensor(ft_key).float()

    return _check_condition_number_impl(
        orig_tensors, ft_tensors, projection_configs,
        cond_threshold=cond_threshold,
        cond_increase_threshold=cond_increase_threshold,
    )


# ===========================================================================
# 完整报告：只打开一次文件，预加载所有 tensor
# ===========================================================================

def full_report(
    orig_path: str,
    ft_path: str,
    svd_layers: list[str],
    key_mapper: Callable[[str, set[str]], str | None],
    projections: list[tuple[str, str, Callable | None]],
    n_sample: int = 200,
    rank_drop_threshold: int = 5,
    cond_threshold: float = 100.0,
    cond_increase_threshold: float = 3.0,
    rank_threshold: float = 1e-3,
) -> dict:
    """
    运行全部三项检查并输出汇总报告。

    本函数只打开一次 checkpoint 文件，预加载所有需要的 tensor，然后分别调用
    三个纯计算函数（_check_*_impl），避免重复 IO。

    通用调用示例：
      from collapse_checker import full_report

      def my_mapper(ft_key, orig_keys):
          if ft_key in orig_keys:
              return ft_key
          return None

      report = full_report(
          orig_path="orig/model.safetensors",
          ft_path="ft/model.safetensors",
          svd_layers=[
              "model.encoder.layers.0.self_attn.q_proj.weight",
              "model.encoder.layers.0.fc1.weight",
          ],
          key_mapper=my_mapper,
          projections=[
              ("model.proj", "model.proj.weight", None),
          ],
      )

    参数:
        orig_path: 原始模型 checkpoint 路径
        ft_path: 微调后模型 checkpoint 路径
        svd_layers: SVD 有效秩检查中待检查的层 key 列表（微调模型命名格式）
        key_mapper: ft_key × orig_keys_set → orig_key | None 的映射函数
        projections: 条件数检查中的投影矩阵配置列表，
                     每个元素为 (orig_key, ft_key, transform_fn)
        n_sample: Embedding 余弦检查中采样 token 行数（默认 200）
        rank_drop_threshold: SVD 有效秩下降报警阈值（默认 5）
        cond_threshold: 条件数绝对报警阈值（默认 100）
        cond_increase_threshold: 条件数相对增长报警阈值（默认 3）
        rank_threshold: SVD 奇异值相对阈值（默认 1e-3）

    返回:
        dict，包含 embedding / svd_rank / condition_number 三个子结果
    """
    print("=" * 80)
    print("模型语义崩塌验证报告")
    print("=" * 80)
    print(f"原始模型:   {orig_path}")
    print(f"微调模型: {ft_path}")
    print()

    with safe_open(orig_path, framework="pt") as orig_f, \
         safe_open(ft_path, framework="pt") as ft_f:

        orig_keys = set(orig_f.keys())
        ft_keys = set(ft_f.keys())

        # -------------------------------------------------------------------
        # 预加载所有需要的 tensor
        # -------------------------------------------------------------------
        orig_tensors: dict[str, torch.Tensor] = {}
        ft_tensors: dict[str, torch.Tensor] = {}

        # 1. embed_tokens
        orig_embed_key = None
        for k in orig_f.keys():
            if "embed_tokens.weight" in k:
                orig_embed_key = k
                break
        ft_embed_key = None
        for k in ft_f.keys():
            if "embed_tokens.weight" in k:
                ft_embed_key = k
                break

        if orig_embed_key and ft_embed_key:
            orig_tensors[orig_embed_key] = orig_f.get_tensor(orig_embed_key).float()
            ft_tensors[ft_embed_key] = ft_f.get_tensor(ft_embed_key).float()

        # 2. svd layers
        svd_layer_mapping: dict[str, str] = {}
        for ft_key in svd_layers:
            orig_key = key_mapper(ft_key, orig_keys)
            if orig_key and orig_key in orig_keys and ft_key in ft_keys:
                svd_layer_mapping[ft_key] = orig_key
                orig_tensors[orig_key] = orig_f.get_tensor(orig_key).float()
                ft_tensors[ft_key] = ft_f.get_tensor(ft_key).float()

        # 3. projections
        projection_configs: list[tuple[str, str, Callable | None]] = []
        for orig_key, ft_key, transform in projections:
            if orig_key in orig_keys and ft_key in ft_keys:
                projection_configs.append((orig_key, ft_key, transform))
                orig_tensors[orig_key] = orig_f.get_tensor(orig_key).float()
                ft_tensors[ft_key] = ft_f.get_tensor(ft_key).float()

    # -----------------------------------------------------------------------
    # 调用纯计算逻辑（此时文件已关闭，所有数据在内存中）
    # -----------------------------------------------------------------------
    r1 = {"error": "embed_tokens not found"}
    if orig_embed_key and ft_embed_key:
        r1 = _check_embedding_cosine_impl(
            orig_tensors[orig_embed_key],
            ft_tensors[ft_embed_key],
            n_sample=n_sample,
        )
    print()

    r2 = _check_svd_rank_impl(
        orig_tensors, ft_tensors, svd_layer_mapping,
        rank_threshold=rank_threshold,
        rank_drop_threshold=rank_drop_threshold,
    )
    print()

    r3 = _check_condition_number_impl(
        orig_tensors, ft_tensors, projection_configs,
        cond_threshold=cond_threshold,
        cond_increase_threshold=cond_increase_threshold,
    )

    # -----------------------------------------------------------------------
    # 汇总
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("汇总")
    print("=" * 80)
    checks = [
        ("Token Embedding 余弦相似度", r1.get("collapsed", None) if "collapsed" in r1 else None),
        ("权重矩阵 SVD 有效秩", r2.get("any_collapsed", None)),
        ("投影矩阵条件数", r3.get("any_collapsed", None)),
    ]
    for name, collapsed in checks:
        if collapsed is None:
            status = "错误（无法检查）"
        elif collapsed:
            status = "*** 检测到崩塌 ***"
        else:
            status = "正常"
        print(f"  {name}: {status}")

    return {
        "embedding": r1,
        "svd_rank": r2,
        "condition_number": r3,
    }
