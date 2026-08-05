#!/usr/bin/env python3
"""Verify VLM semantic collapse after fine-tuning.

Semantic collapse means the visual encoder outputs converge to similar
representations for different inputs, making the downstream policy unable
to distinguish different observations.  This tool checks three signals:

1. **Embedding cosine similarity** – If token embeddings collapse, rows of
   the embedding matrix become nearly parallel, raising the average
   off-diagonal cosine similarity.  A healthy model has diverse embeddings
   (mean cosine ≈ 0); a collapsed model has mean cosine → 1.

2. **Weight SVD rank** – If a weight matrix collapses, its effective rank
   (number of singular values above a threshold) drops because rows/columns
   become linearly dependent.  Comparing rank before/after training reveals
   whether the layer lost expressiveness.

3. **Condition number** – The ratio of largest to smallest singular value.
   A collapsed projection matrix has a very high condition number (one
   direction dominates).  Comparing before/after training shows whether
   the projection degenerated.

Usage examples::

  # Full report (all three checks)
  python -m utils.vlm_collapse_check /data/checkpoints/xvla-base/model.safetensors \
      /data/outputs/.../030000/pretrained_model/model.safetensors

  # Only embedding cosine similarity
  python -m utils.vlm_collapse_check --check embedding \
      /data/checkpoints/xvla-base/model.safetensors \
      /data/outputs/.../030000/pretrained_model/model.safetensors

  # Only SVD rank check
  python -m utils.vlm_collapse_check --check svd \
      /data/checkpoints/xvla-base/model.safetensors \
      /data/outputs/.../030000/pretrained_model/model.safetensors
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from safetensors import safe_open


# ---------------------------------------------------------------------------
# Key remapping (vendored → native Florence-2 layout)
# ---------------------------------------------------------------------------

def _orig_key_for_ft(ft_key: str, orig_keys: set[str]) -> str | None:
    """Find the original (vendored-format) key corresponding to a fine-tuned
    (native-format) key.  Returns *None* if no mapping is found."""

    if ft_key in orig_keys:
        return ft_key

    # image_projection: nn.Parameter → nn.Linear.weight (needs transpose)
    if ft_key == "model.vlm.multi_modal_projector.image_projection.weight":
        return "model.vlm.image_projection"

    # language_model: drop .model.
    if "model.vlm.language_model.encoder." in ft_key:
        c = ft_key.replace(
            "model.vlm.language_model.encoder.",
            "model.vlm.language_model.model.encoder.",
        )
        if c in orig_keys:
            return c

    # vision_tower: drop .fn.
    if "channel_attn." in ft_key or "window_attn." in ft_key:
        for s in (".qkv.", ".proj.", ".q_proj.", ".k_proj.",
                  ".v_proj.", ".out_proj."):
            if s in ft_key:
                c = ft_key.replace(s, ".fn" + s)
                if c in orig_keys:
                    return c

    # multi_modal_projector sub-modules
    for old, new in (
        ("multi_modal_projector.image_position_embed", "image_pos_embed"),
        ("multi_modal_projector.image_proj_norm", "image_proj_norm"),
        ("multi_modal_projector.visual_temporal_embed", "visual_temporal_embed"),
    ):
        if old in ft_key:
            c = ft_key.replace(old, new)
            if c in orig_keys:
                return c
            # some keys just drop the prefix
            c2 = ft_key.replace(f"multi_modal_projector.", "")
            if c2 in orig_keys:
                return c2

    # conv / ffn / norm renames
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
# Check 1: Embedding cosine similarity
# ---------------------------------------------------------------------------

def check_embedding_cosine(
    orig_path: str,
    ft_path: str,
    n_sample: int = 200,
    threshold: float = 0.1,
) -> dict:
    """Check whether token embeddings have collapsed by comparing the
    average off-diagonal cosine similarity between rows.

    **Principle**: In a healthy embedding matrix, different token embeddings
    point in diverse directions, so the average pairwise cosine similarity
    is low (near 0 for random init, modestly positive for trained models).
    If the model collapses, embeddings converge to similar directions and
    the mean cosine similarity rises toward 1.

    Args:
        orig_path: Path to the original model.safetensors.
        ft_path: Path to the fine-tuned model.safetensors.
        n_sample: Number of token rows to sample for the comparison.
        threshold: If the fine-tuned mean cosine exceeds the original by
            more than this value, collapse is flagged.

    Returns:
        Dict with orig_mean, ft_mean, diff, collapsed flag.
    """
    print("=" * 80)
    print("Check 1: Embedding cosine similarity")
    print("=" * 80)
    print("Principle: If embeddings collapse, rows become nearly parallel,")
    print("raising the average off-diagonal cosine similarity toward 1.")
    print()

    with safe_open(orig_path, framework="pt") as orig_f, \
         safe_open(ft_path, framework="pt") as ft_f:

        # Find embed_tokens keys (may differ between vendored/native formats)
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

        if not orig_embed_key or not ft_embed_key:
            print("  ERROR: embed_tokens.weight not found in one or both checkpoints")
            return {"error": "embed_tokens not found"}

        orig_embed = orig_f.get_tensor(orig_embed_key).float()
        ft_embed = ft_f.get_tensor(ft_embed_key).float()

    # Normalize each row to unit length, then compute pairwise cosine matrix
    orig_norm = orig_embed / orig_embed.norm(dim=1, keepdim=True).clamp(min=1e-8)
    ft_norm = ft_embed / ft_embed.norm(dim=1, keepdim=True).clamp(min=1e-8)

    idx = torch.randperm(orig_embed.shape[0])[:n_sample]
    orig_cos = orig_norm[idx] @ orig_norm[idx].T
    ft_cos = ft_norm[idx] @ ft_norm[idx].T

    # Exclude diagonal (self-similarity is always 1)
    mask = ~torch.eye(n_sample, dtype=torch.bool)
    orig_cos_off = orig_cos[mask]
    ft_cos_off = ft_cos[mask]

    orig_mean = orig_cos_off.mean().item()
    ft_mean = ft_cos_off.mean().item()
    diff = ft_mean - orig_mean
    collapsed = diff > threshold

    print(f"  embed_tokens shape: {orig_embed.shape}")
    print(f"  Sampled tokens: {n_sample}")
    print(f"  Original  mean cosine: {orig_mean:.4f}  std: {orig_cos_off.std().item():.4f}  max: {orig_cos_off.max().item():.4f}")
    print(f"  Fine-tuned mean cosine: {ft_mean:.4f}  std: {ft_cos_off.std().item():.4f}  max: {ft_cos_off.max().item():.4f}")
    print(f"  Change: {diff:+.4f}")
    if collapsed:
        print(f"  *** COLLAPSE DETECTED: cosine similarity increased by {diff:.4f} > {threshold} ***")
    else:
        print(f"  OK: No significant collapse (change {diff:+.4f} <= {threshold})")

    return {
        "orig_mean": orig_mean,
        "ft_mean": ft_mean,
        "diff": diff,
        "collapsed": collapsed,
    }


# ---------------------------------------------------------------------------
# Check 2: Weight SVD rank
# ---------------------------------------------------------------------------

# Default layers to sample (fine-tuned / native-format keys)
_DEFAULT_SVD_LAYERS = [
    "model.vlm.vision_tower.blocks.0.0.channel_block.channel_attn.qkv.weight",
    "model.vlm.vision_tower.blocks.3.0.channel_block.channel_attn.qkv.weight",
    "model.vlm.vision_tower.blocks.0.0.spatial_block.window_attn.qkv.weight",
    "model.vlm.vision_tower.blocks.3.0.spatial_block.window_attn.qkv.weight",
    "model.vlm.vision_tower.blocks.0.0.channel_block.channel_attn.proj.weight",
    "model.vlm.vision_tower.blocks.3.0.channel_block.channel_attn.proj.weight",
    "model.vlm.language_model.encoder.layers.0.self_attn.q_proj.weight",
    "model.vlm.language_model.encoder.layers.5.self_attn.q_proj.weight",
    "model.vlm.language_model.encoder.layers.0.fc1.weight",
    "model.transformer.action_decoder.fc.weight",
    "model.transformer.cross_attn.0.q_proj.weight",
]


def check_svd_rank(
    orig_path: str,
    ft_path: str,
    layers: list[str] | None = None,
    rank_threshold: float = 1e-3,
    rank_drop_threshold: int = 5,
) -> dict:
    """Check whether weight matrices have lost effective rank, which
    indicates that rows/columns have become linearly dependent (collapse).

    **Principle**: The effective rank of a matrix is the number of singular
    values above a relative threshold (e.g., 1e-3 × σ_max).  If training
    causes rows to converge, the matrix becomes low-rank and the effective
    rank drops.  Comparing rank before/after training per layer reveals
    which layers lost expressiveness.

    Args:
        orig_path: Path to the original model.safetensors.
        ft_path: Path to the fine-tuned model.safetensors.
        layers: List of fine-tuned format keys to check.
        rank_threshold: Singular values below this fraction of σ_max are
            considered zero for rank counting.
        rank_drop_threshold: If a layer's rank drops by more than this
            many, it is flagged as collapsed.

    Returns:
        Dict with per-layer results and overall collapsed flag.
    """
    if layers is None:
        layers = _DEFAULT_SVD_LAYERS

    print("=" * 80)
    print("Check 2: Weight SVD effective rank")
    print("=" * 80)
    print("Principle: If a weight matrix collapses, its rows become linearly")
    print("dependent, reducing the effective rank (number of significant")
    print("singular values). A large rank drop signals collapse.")
    print()

    orig_keys: set[str] = set()
    with safe_open(orig_path, framework="pt") as f:
        orig_keys = set(f.keys())

    print(f"{'Layer':<55} {'Orig rank':>10} {'FT rank':>10} {'Δ rank':>8} {'Orig σ₁':>10} {'FT σ₁':>10}")
    print("-" * 110)

    results = []
    any_collapsed = False

    with safe_open(orig_path, framework="pt") as orig_f, \
         safe_open(ft_path, framework="pt") as ft_f:

        for ft_key in layers:
            orig_key = _orig_key_for_ft(ft_key, orig_keys)
            if orig_key is None or orig_key not in orig_keys or ft_key not in ft_f.keys():
                short = ".".join(ft_key.split(".")[-3:])
                print(f"{short:<55} {'N/A':>10}")
                continue

            orig_t = orig_f.get_tensor(orig_key).float()
            ft_t = ft_f.get_tensor(ft_key).float()

            # image_projection needs transpose
            if orig_key == "model.vlm.image_projection":
                orig_t = orig_t.transpose(0, 1).contiguous()

            if orig_t.shape != ft_t.shape:
                short = ".".join(ft_key.split(".")[-3:])
                print(f"{short:<55} shape mismatch: {orig_t.shape} vs {ft_t.shape}")
                continue

            # Flatten to 2D for SVD
            orig_flat = orig_t.reshape(orig_t.shape[0], -1)
            ft_flat = ft_t.reshape(ft_t.shape[0], -1)

            orig_sv = torch.linalg.svdvals(orig_flat)
            ft_sv = torch.linalg.svdvals(ft_flat)

            orig_rank = (orig_sv > orig_sv[0] * rank_threshold).sum().item()
            ft_rank = (ft_sv > ft_sv[0] * rank_threshold).sum().item()
            rank_drop = orig_rank - ft_rank

            short = ".".join(ft_key.split(".")[-4:])
            flag = " *** COLLAPSE" if rank_drop > rank_drop_threshold else ""
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
        print(f"\n  OK: No layer lost more than {rank_drop_threshold} in effective rank.")
    else:
        collapsed_layers = [r["layer"] for r in results if r["collapsed"]]
        print(f"\n  *** COLLAPSE DETECTED in layers: {', '.join(collapsed_layers)} ***")

    return {"layers": results, "any_collapsed": any_collapsed}


# ---------------------------------------------------------------------------
# Check 3: Condition number of projection matrices
# ---------------------------------------------------------------------------

def check_condition_number(
    orig_path: str,
    ft_path: str,
    cond_threshold: float = 100.0,
    cond_increase_threshold: float = 3.0,
) -> dict:
    """Check whether projection matrices have degenerated by comparing
    their condition numbers (ratio of largest to smallest singular value).

    **Principle**: The condition number κ = σ_max / σ_min measures how
    "well-behaved" a linear transformation is.  A low κ means the
    projection preserves information in all directions; a high κ means
    one direction dominates and information in other directions is
    suppressed (collapse).  If κ increases significantly after training,
    the projection is degenerating.

    Args:
        orig_path: Path to the original model.safetensors.
        ft_path: Path to the fine-tuned model.safetensors.
        cond_threshold: Absolute condition number above which a matrix
            is flagged as ill-conditioned.
        cond_increase_threshold: If the condition number increases by
            more than this factor after training, it is flagged.

    Returns:
        Dict with per-matrix results and overall collapsed flag.
    """
    print("=" * 80)
    print("Check 3: Projection matrix condition number")
    print("=" * 80)
    print("Principle: The condition number κ = σ_max/σ_min measures how much")
    print("one direction dominates. A large increase in κ after training")
    print("means the projection is degenerating toward collapse.")
    print()

    # Projection matrices to check (orig_key, ft_key, needs_transpose)
    projections = [
        ("model.vlm.image_projection",
         "model.vlm.multi_modal_projector.image_projection.weight", True),
    ]

    results = []
    any_collapsed = False

    with safe_open(orig_path, framework="pt") as orig_f, \
         safe_open(ft_path, framework="pt") as ft_f:

        for orig_key, ft_key, needs_transpose in projections:
            if orig_key not in orig_f.keys() or ft_key not in ft_f.keys():
                print(f"  {orig_key} or {ft_key} not found, skipping")
                continue

            orig_t = orig_f.get_tensor(orig_key).float()
            ft_t = ft_f.get_tensor(ft_key).float()
            if needs_transpose:
                orig_t = orig_t.transpose(0, 1).contiguous()

            orig_sv = torch.linalg.svdvals(orig_t)
            ft_sv = torch.linalg.svdvals(ft_t)

            orig_cond = (orig_sv[0] / orig_sv[-1]).item()
            ft_cond = (ft_sv[0] / ft_sv[-1]).item()
            cond_ratio = ft_cond / orig_cond if orig_cond > 0 else float("inf")

            print(f"  {ft_key.split('.')[-2]}:")
            print(f"    Original  top-5 sv: {[f'{v:.4f}' for v in orig_sv[:5].tolist()]}")
            print(f"    Fine-tuned top-5 sv: {[f'{v:.4f}' for v in ft_sv[:5].tolist()]}")
            print(f"    Original  condition number: {orig_cond:.2f}")
            print(f"    Fine-tuned condition number: {ft_cond:.2f}")
            print(f"    Condition number ratio (ft/orig): {cond_ratio:.2f}x")

            collapsed = cond_ratio > cond_increase_threshold or ft_cond > cond_threshold
            if collapsed:
                print(f"    *** COLLAPSE: condition number increased by {cond_ratio:.2f}x or exceeds {cond_threshold} ***")
            else:
                print(f"    OK: condition number stable")

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


# ---------------------------------------------------------------------------
# Full report
# ---------------------------------------------------------------------------

def full_report(orig_path: str, ft_path: str) -> None:
    """Run all three collapse checks and print a summary."""
    print("=" * 80)
    print("VLM Semantic Collapse Verification Report")
    print("=" * 80)
    print(f"Original:   {orig_path}")
    print(f"Fine-tuned: {ft_path}")
    print()

    r1 = check_embedding_cosine(orig_path, ft_path)
    print()
    r2 = check_svd_rank(orig_path, ft_path)
    print()
    r3 = check_condition_number(orig_path, ft_path)

    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    checks = [
        ("Embedding cosine similarity", r1.get("collapsed", None)),
        ("Weight SVD rank", r2.get("any_collapsed", None)),
        ("Condition number", r3.get("any_collapsed", None)),
    ]
    for name, collapsed in checks:
        if collapsed is None:
            status = "ERROR (could not check)"
        elif collapsed:
            status = "*** COLLAPSE DETECTED ***"
        else:
            status = "OK"
        print(f"  {name}: {status}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify VLM semantic collapse after fine-tuning.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("orig", help="Path to the original model.safetensors")
    parser.add_argument("ft", help="Path to the fine-tuned model.safetensors")
    parser.add_argument(
        "--check",
        choices=["embedding", "svd", "condition", "all"],
        default="all",
        help="Which check to run (default: all)",
    )
    parser.add_argument(
        "--n-sample",
        type=int,
        default=200,
        help="Number of token rows to sample for embedding cosine check (default: 200)",
    )
    parser.add_argument(
        "--rank-drop-threshold",
        type=int,
        default=5,
        help="Flag a layer as collapsed if its SVD rank drops by more than this (default: 5)",
    )

    args = parser.parse_args()

    if args.check == "embedding":
        check_embedding_cosine(args.orig, args.ft, n_sample=args.n_sample)
    elif args.check == "svd":
        check_svd_rank(args.orig, args.ft, rank_drop_threshold=args.rank_drop_threshold)
    elif args.check == "condition":
        check_condition_number(args.orig, args.ft)
    else:
        full_report(args.orig, args.ft)


if __name__ == "__main__":
    main()
