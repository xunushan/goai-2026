#!/usr/bin/env python3
"""Evaluate a LeRobot ACT checkpoint on validation data with visualization.

Evaluation pipeline (3 stages)
------------------------------
1. Inference:  Run model on sampled validation frames → save raw results (raw_results.npz).
2. Metrics:    Compute all metrics from raw inference results → save metrics.json.
3. Visualize:  Generate time-series and bar charts from metrics + raw results → save plots.

Output metrics
--------------
- eval_loss: normalized L1 loss (may be absent if checkpoint lacks training loss)
- physical_mae.first_step / execution_window / full_chunk
- physical_mae.per_dimension (16 dims)
- physical_mae.groups (6 functional groups)

Visualization
-------------
- 6 group time-series plots (expert vs predicted action values)
- per-dimension MAE bar chart
- grouped MAE bar chart
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm

from lerobot.configs.default import DatasetConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.factory import make_dataset
from lerobot.policies import make_policy, make_pre_post_processors
from lerobot.utils.collate import lerobot_collate_fn


ACTION_NAMES = (
    "l_x",
    "l_y",
    "l_z",
    "l_w",
    "l_wx",
    "l_wy",
    "l_wz",
    "l_g",
    "r_x",
    "r_y",
    "r_z",
    "r_w",
    "r_wx",
    "r_wy",
    "r_wz",
    "r_g",
)

ACTION_GROUPS = {
    "left_position": (0, 1, 2),
    "left_rotation": (3, 4, 5, 6),
    "left_gripper": (7,),
    "right_position": (8, 9, 10),
    "right_rotation": (11, 12, 13, 14),
    "right_gripper": (15,),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate LeRobot ACT checkpoint on validation split"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="LeRobot checkpoint pretrained_model directory.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Path to lerobot_v3 format dataset root directory.",
    )
    parser.add_argument(
        "--split-path",
        type=Path,
        required=True,
        help="JSON file with train/val episode splits.",
    )
    parser.add_argument(
        "--split",
        choices=("train", "val"),
        default="val",
        help="Which split to evaluate (default: val).",
    )
    parser.add_argument("--repo-id", default="lerobot_v30_ee")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--video-backend", default="pyav")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Maximum validation frames; 0 evaluates all.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=25,
        help="Sampling stride for evaluation frames and visualization (default: 25).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/eval_val"),
        help="Output directory for metrics and plots.",
    )
    parser.add_argument(
        "--convert-20d-to-16d",
        action="store_true",
        help="Convert 20D X-VLA predictions to 16D EE before computing physical MAE. "
        "Uses utils.xvla_ee.xvla20_to_ee16 (rotation6d→quaternion + gripper invert).",
    )
    parser.add_argument(
        "--rename-map",
        type=str,
        default=None,
        help="JSON mapping of dataset feature keys to policy feature keys, "
        'e.g. \'{"observation.images.cam_high":"observation.images.image"}\'.',
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Stage 1: Inference — collect raw predictions, no metric computation
# ---------------------------------------------------------------------------


@dataclass
class InferenceResult:
    """Raw inference outputs for metric computation and visualization."""

    # Per-frame first-step actions in physical units
    frame_indices: np.ndarray  # [N] global frame index
    expert_first: np.ndarray  # [N, action_dim] expert first-step action
    predicted_first: np.ndarray  # [N, action_dim] predicted first-step action

    # Per-batch data for metric computation
    physical_errors: list = field(default_factory=list)  # list of [B, chunk, D] arrays
    valid_masks: list = field(default_factory=list)  # list of [B, chunk] bool arrays
    normalized_error_sums: list = field(default_factory=list)  # per-batch sum
    normalized_error_counts: list = field(default_factory=list)  # per-batch count
    execution_steps: int = 0


def run_inference(
    policy,
    preprocessor,
    postprocessor,
    dataloader,
    dataset,
    device: torch.device,
    convert_20d_to_16d: bool = False,
) -> InferenceResult:
    """Run batch inference and collect raw data (no metric computation).

    Returns an InferenceResult containing per-frame expert/predicted actions
    and per-batch error tensors for downstream metric computation.
    """
    from utils.xvla_ee import xvla20_to_ee16

    first_expert: list[np.ndarray] = []
    first_predicted: list[np.ndarray] = []
    frame_indices: list[int] = []
    physical_errors: list[np.ndarray] = []
    valid_masks: list[np.ndarray] = []
    normalized_error_sums: list[float] = []
    normalized_error_counts: list[int] = []

    global_idx = 0

    with torch.inference_mode():
        for batch in tqdm(dataloader, desc="Inference", unit="batch"):
            expert_action_physical = (
                batch["action"].to(device=device, dtype=torch.float32).clone()
            )
            for camera_key in dataset.meta.camera_keys:
                if camera_key in batch and batch[camera_key].dtype == torch.uint8:
                    batch[camera_key] = batch[camera_key].float() / 255.0
            batch = preprocessor(batch)

            predicted_action_normalized = policy.predict_action_chunk(batch)
            expert_action_normalized = batch["action"]
            valid_step_mask = ~batch["action_is_pad"]

            # Normalized L1 (collect raw sums for later aggregation)
            valid_element_mask = valid_step_mask.unsqueeze(-1).expand_as(
                predicted_action_normalized
            )
            normalized_error = torch.abs(
                predicted_action_normalized - expert_action_normalized
            )
            normalized_error_sums.append(
                normalized_error[valid_element_mask].sum().item()
            )
            normalized_error_counts.append(int(valid_element_mask.sum().item()))

            # Physical-space prediction
            batch_size, horizon, action_dim = predicted_action_normalized.shape
            predicted_action_physical = (
                postprocessor(
                    predicted_action_normalized.reshape(
                        batch_size * horizon, action_dim
                    )
                )
                .reshape(batch_size, horizon, action_dim)
                .to(device)
            )
            if convert_20d_to_16d:
                pred_20d = predicted_action_physical.float().cpu().numpy()
                pred_16d = xvla20_to_ee16(pred_20d.reshape(-1, 20)).reshape(
                    pred_20d.shape[:-1] + (16,)
                )
                predicted_action_physical = torch.as_tensor(
                    pred_16d, dtype=torch.float32, device=device
                )
                expert_20d = expert_action_physical.float().cpu().numpy()
                expert_16d = xvla20_to_ee16(expert_20d.reshape(-1, 20)).reshape(
                    expert_20d.shape[:-1] + (16,)
                )
                expert_action_physical = torch.as_tensor(
                    expert_16d, dtype=torch.float32, device=device
                )

            # Physical error (collect for later metric computation)
            physical_error = torch.abs(
                predicted_action_physical - expert_action_physical
            )
            physical_valid_mask = valid_step_mask.unsqueeze(-1).expand_as(
                physical_error
            )
            physical_errors.append(physical_error.cpu().numpy())
            valid_masks.append(physical_valid_mask.cpu().numpy())

            # Collect per-frame first-step actions
            for b in range(batch_size):
                frame_indices.append(global_idx)
                first_expert.append(
                    expert_action_physical[b, 0].float().cpu().numpy()
                )
                first_predicted.append(
                    predicted_action_physical[b, 0].float().cpu().numpy()
                )
                global_idx += 1

    return InferenceResult(
        frame_indices=np.asarray(frame_indices),
        expert_first=np.stack(first_expert),
        predicted_first=np.stack(first_predicted),
        physical_errors=physical_errors,
        valid_masks=valid_masks,
        normalized_error_sums=normalized_error_sums,
        normalized_error_counts=normalized_error_counts,
        execution_steps=int(policy.config.n_action_steps),
    )


# ---------------------------------------------------------------------------
# Stage 2: Metric computation — pure calculation, no model involved
# ---------------------------------------------------------------------------


def compute_metrics(result: InferenceResult) -> dict:
    """Compute all metrics from InferenceResult. No model involved."""
    # Aggregate normalized L1
    normalized_l1 = sum(result.normalized_error_sums) / max(
        sum(result.normalized_error_counts), 1
    )

    # Aggregate physical errors
    all_physical = np.concatenate(result.physical_errors, axis=0)  # [N, chunk, D]
    all_valid = np.concatenate(result.valid_masks, axis=0)  # [N, chunk, D]

    # Full chunk MAE
    physical_mae_full = float(all_physical[all_valid].mean()) if all_valid.any() else 0.0

    # First step MAE
    first_valid = all_valid[:, :1]
    first_errors = all_physical[:, :1]
    physical_mae_first = float(first_errors[first_valid].mean()) if first_valid.any() else 0.0

    # Execution window MAE
    exec_steps = min(result.execution_steps, all_physical.shape[1])
    exec_valid = all_valid[:, :exec_steps]
    exec_errors = all_physical[:, :exec_steps]
    physical_mae_exec = float(exec_errors[exec_valid].mean()) if exec_valid.any() else 0.0

    # Per-dimension MAE
    per_dim_sum = (all_physical * all_valid).sum(axis=(0, 1))  # [D]
    per_dim_count = all_valid.sum(axis=(0, 1))  # [D]
    per_dim_count = np.maximum(per_dim_count, 1)
    per_dim_mae = per_dim_sum / per_dim_count
    per_dimension = {
        name: float(value)
        for name, value in zip(ACTION_NAMES, per_dim_mae.tolist(), strict=True)
    }

    # Grouped MAE
    grouped = {}
    for name, indices in ACTION_GROUPS.items():
        idx = list(indices)
        group_sum = per_dim_sum[idx].sum()
        group_count = per_dim_count[idx].sum()
        grouped[name] = float(group_sum / max(group_count, 1))

    return {
        "eval_loss": normalized_l1,
        "physical_mae": {
            "first_step": physical_mae_first,
            "execution_window": physical_mae_exec,
            "execution_steps": exec_steps,
            "full_chunk": physical_mae_full,
            "per_dimension": per_dimension,
            "groups": grouped,
        },
    }


# ---------------------------------------------------------------------------
# Stage 3: Visualization
# ---------------------------------------------------------------------------


def save_timeseries_plots(
    output_dir: Path,
    frame_indices: np.ndarray,
    expert: np.ndarray,
    predicted: np.ndarray,
) -> None:
    """Save 6 group time-series plots: expert vs predicted action values."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for group_name, indices in ACTION_GROUPS.items():
        indices = list(indices)
        fig, axes = plt.subplots(
            len(indices), 1, figsize=(12, 2.6 * len(indices)), sharex=True
        )
        axes = np.atleast_1d(axes)
        for ax, dim in zip(axes, indices, strict=True):
            ax.plot(frame_indices, expert[:, dim], label="expert", linewidth=1.8)
            ax.plot(
                frame_indices,
                predicted[:, dim],
                label="predicted",
                linestyle="--",
                linewidth=1.5,
            )
            ax.set_ylabel(ACTION_NAMES[dim])
            ax.grid(alpha=0.25)
        axes[0].legend()
        axes[-1].set_xlabel("frame index")
        fig.tight_layout()
        fig.savefig(output_dir / f"{group_name}_timeseries.png", dpi=150)
        plt.close(fig)


def save_mae_bar_charts(
    output_dir: Path,
    per_dimension_mae: dict[str, float],
    grouped_mae: dict[str, float],
) -> None:
    """Save per-dimension and grouped MAE bar charts."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Per-dimension MAE
    names = list(per_dimension_mae.keys())
    values = list(per_dimension_mae.values())
    fig, ax = plt.subplots(figsize=(14, 4))
    colors = []
    for name in names:
        if name.startswith("l_"):
            colors.append("#4C72B0")
        else:
            colors.append("#DD8452")
    ax.bar(names, values, color=colors)
    ax.set_ylabel("MAE (physical units)")
    ax.set_title("Per-Dimension MAE")
    ax.tick_params(axis="x", rotation=45)
    for i, v in enumerate(values):
        ax.text(i, v, f"{v:.4f}", ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    fig.savefig(output_dir / "per_dimension_mae.png", dpi=150)
    plt.close(fig)

    # Grouped MAE
    g_names = list(grouped_mae.keys())
    g_values = list(grouped_mae.values())
    fig, ax = plt.subplots(figsize=(8, 4))
    g_colors = ["#4C72B0" if "left" in n else "#DD8452" for n in g_names]
    ax.bar(g_names, g_values, color=g_colors)
    ax.set_ylabel("MAE (physical units)")
    ax.set_title("Grouped MAE")
    ax.tick_params(axis="x", rotation=45)
    for i, v in enumerate(g_values):
        ax.text(i, v, f"{v:.4f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "grouped_mae.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def validate_local_dataset(
    repo_id: str,
    root: Path,
    episodes: list[int],
) -> None:
    """Fail clearly before LeRobot silently falls back to Hugging Face Hub."""

    required_metadata = (
        root / "meta/info.json",
        root / "meta/stats.json",
        root / "meta/tasks.parquet",
    )
    missing = [path for path in required_metadata if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Local LeRobot dataset metadata is incomplete:\n"
            + "\n".join(f"  - {path}" for path in missing)
        )
    episode_metadata = list((root / "meta/episodes").rglob("*.parquet"))
    if not episode_metadata:
        raise FileNotFoundError(
            f"Local LeRobot dataset has no episode metadata under {root / 'meta/episodes'}"
        )

    metadata = LeRobotDatasetMetadata(repo_id, root=root)
    if any(index < 0 or index >= metadata.total_episodes for index in episodes):
        invalid = [
            index for index in episodes if index < 0 or index >= metadata.total_episodes
        ]
        raise ValueError(
            f"Split contains out-of-range episodes {invalid[:20]}; "
            f"dataset has {metadata.total_episodes} episodes"
        )

    required_paths: set[Path] = set()
    for episode in episodes:
        required_paths.add(root / metadata.get_data_file_path(episode))
        for camera_key in metadata.video_keys:
            required_paths.add(root / metadata.get_video_file_path(episode, camera_key))
    missing = sorted(path for path in required_paths if not path.exists())
    if missing:
        broken_links = [path for path in missing if path.is_symlink()]
        detail = "\n".join(f"  - {path}" for path in missing[:50])
        hint = (
            "\nBroken video symlinks were detected. Re-run the conversion script "
            "on the server, or recreate links so they target the server's original dataset."
            if broken_links
            else ""
        )
        raise FileNotFoundError(
            f"Local dataset is missing {len(missing)} required data/video files. "
            "LeRobot would otherwise fall back to the Hub:\n"
            f"{detail}{hint}"
        )


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint}")

    # Load split
    with args.split_path.resolve().open(encoding="utf-8") as f:
        split_data = json.load(f)
    episodes = [int(v) for v in split_data[args.split]]
    if not episodes:
        raise ValueError(f"Split {args.split!r} is empty in {args.split_path}")
    dataset_root = args.dataset_root.resolve()
    validate_local_dataset(args.repo_id, dataset_root, episodes)

    # Build policy config from checkpoint
    policy_cfg = PreTrainedConfig.from_pretrained(checkpoint)
    policy_cfg.pretrained_path = checkpoint

    # Build dataset
    rename_map = json.loads(args.rename_map) if args.rename_map else {}
    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(
            repo_id=args.repo_id,
            root=str(dataset_root),
            episodes=episodes,
            video_backend=args.video_backend,
            eval_split=0.0,
        ),
        policy=policy_cfg,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        eval_steps=0,
        rename_map=rename_map,
    )
    dataset = make_dataset(cfg)

    eval_dataset = dataset
    if args.max_samples > 0:
        count = min(args.max_samples, len(dataset))
        indices = torch.linspace(0, len(dataset) - 1, steps=count).long().tolist()
        eval_dataset = Subset(dataset, indices)
    elif args.stride > 1:
        indices = list(range(0, len(dataset), args.stride))
        eval_dataset = Subset(dataset, indices)
        print(f"Stride={args.stride}: evaluating {len(indices)}/{len(dataset)} frames")

    collate_fn = lerobot_collate_fn if dataset.meta.has_language_columns else None
    dataloader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        collate_fn=collate_fn,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )

    # Build policy and preprocessor
    policy = make_policy(cfg=policy_cfg, ds_meta=dataset.meta, rename_map=rename_map or None)
    device = next(policy.parameters()).device
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=checkpoint,
        dataset_stats=dataset.meta.stats,
    )
    policy.eval()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Stage 1: Inference ---
    print(
        f"Evaluating {args.split} split: {len(episodes)} episodes, {len(eval_dataset)} frames"
    )
    if args.convert_20d_to_16d:
        print("20D→16D conversion enabled for physical MAE")
    inference_result = run_inference(
        policy,
        preprocessor,
        postprocessor,
        dataloader,
        dataset,
        device,
        convert_20d_to_16d=args.convert_20d_to_16d,
    )

    # Save raw inference results
    np.savez(
        output_dir / "raw_results.npz",
        frame_indices=inference_result.frame_indices,
        expert_first=inference_result.expert_first,
        predicted_first=inference_result.predicted_first,
    )
    print(f"Saved raw results: {output_dir / 'raw_results.npz'}")

    # --- Stage 2: Metric computation ---
    metrics = compute_metrics(inference_result)
    result = {
        "checkpoint": str(checkpoint),
        "split": args.split,
        "split_path": str(args.split_path.resolve()),
        "val_episodes": len(episodes),
        "val_frames": len(eval_dataset),
        "batch_size": args.batch_size,
        "device": str(device),
        "convert_20d_to_16d": args.convert_20d_to_16d,
        **metrics,
    }
    text = json.dumps(result, ensure_ascii=False, indent=2)
    (output_dir / "metrics.json").write_text(text + "\n", encoding="utf-8")
    print(text)
    print(f"Saved metrics: {output_dir / 'metrics.json'}")

    # --- Stage 3: Visualization ---
    # Downsample for time-series plots using stride
    vis_stride = args.stride
    vis_indices = inference_result.frame_indices[::vis_stride]
    vis_expert = inference_result.expert_first[::vis_stride]
    vis_predicted = inference_result.predicted_first[::vis_stride]

    save_timeseries_plots(output_dir, vis_indices, vis_expert, vis_predicted)
    save_mae_bar_charts(
        output_dir,
        metrics["physical_mae"]["per_dimension"],
        metrics["physical_mae"]["groups"],
    )
    print(f"Saved plots to: {output_dir}")


if __name__ == "__main__":
    main()
