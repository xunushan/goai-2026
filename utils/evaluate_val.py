#!/usr/bin/env python3
"""Evaluate a LeRobot ACT checkpoint on validation data with visualization.

Evaluation pipeline
-------------------
1. Load checkpoint via LeRobot official pipeline (PreTrainedConfig + make_policy).
2. Build validation dataset from split JSON + lerobot_v3 data.
3. Batch inference over all val frames → compute metrics.
4. Sparse sampling for time-series visualization.

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
        help="Sampling stride for visualization (default: 25).",
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
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Batch evaluation: compute all metrics over the full validation set
# ---------------------------------------------------------------------------


def run_batch_evaluation(
    policy,
    preprocessor,
    postprocessor,
    dataloader,
    dataset,
    device: torch.device,
    convert_20d_to_16d: bool = False,
) -> dict:
    """Run batch inference and compute normalized + physical-unit metrics.

    When *convert_20d_to_16d* is True, predicted actions are unnormalized to
    physical 20D, then converted to 16D via ``xvla20_to_ee16`` before comparing
    with the 16D expert actions.  The normalized L1 (eval_loss) is still computed
    in the model's native 20D space.
    """
    from utils.xvla_ee import xvla20_to_ee16

    normalized_error_sum = 0.0
    normalized_error_count = 0
    physical_error_sum = 0.0
    physical_error_count = 0
    first_error_sum = 0.0
    first_error_count = 0
    execution_error_sum = 0.0
    execution_error_count = 0
    per_dimension_sum = torch.zeros(len(ACTION_NAMES), dtype=torch.float64)
    per_dimension_count = torch.zeros(len(ACTION_NAMES), dtype=torch.float64)

    with torch.inference_mode():
        for batch in tqdm(dataloader, desc="Validation", unit="batch"):
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
            valid_element_mask = valid_step_mask.unsqueeze(-1).expand_as(
                predicted_action_normalized
            )

            # Normalized L1
            normalized_error = torch.abs(
                predicted_action_normalized - expert_action_normalized
            )
            normalized_error_sum += normalized_error[valid_element_mask].sum().item()
            normalized_error_count += int(valid_element_mask.sum().item())

            # Physical MAE
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
                # The converted dataset stores both expert and prediction in
                # physical 20D. Convert both sides only after prediction
                # postprocessing/denormalization.
                pred_20d = predicted_action_physical.cpu().numpy()
                pred_16d = xvla20_to_ee16(pred_20d.reshape(-1, 20)).reshape(
                    pred_20d.shape[:-1] + (16,)
                )
                predicted_action_physical = torch.as_tensor(
                    pred_16d, dtype=torch.float32, device=device
                )
                expert_20d = expert_action_physical.cpu().numpy()
                expert_16d = xvla20_to_ee16(expert_20d.reshape(-1, 20)).reshape(
                    expert_20d.shape[:-1] + (16,)
                )
                expert_action_physical = torch.as_tensor(
                    expert_16d, dtype=torch.float32, device=device
                )
            physical_error = torch.abs(
                predicted_action_physical - expert_action_physical
            )
            physical_valid_mask = valid_step_mask.unsqueeze(-1).expand_as(
                physical_error
            )
            physical_error_sum += physical_error[physical_valid_mask].sum().item()
            physical_error_count += int(physical_valid_mask.sum().item())

            # First step
            first_mask = physical_valid_mask[:, :1]
            first_error_sum += physical_error[:, :1][first_mask].sum().item()
            first_error_count += int(first_mask.sum().item())

            # Execution window
            execution_steps = min(
                int(policy.config.n_action_steps), physical_error.shape[1]
            )
            execution_mask = physical_valid_mask[:, :execution_steps]
            execution_error_sum += (
                physical_error[:, :execution_steps][execution_mask].sum().item()
            )
            execution_error_count += int(execution_mask.sum().item())

            # Per dimension
            per_dimension_sum += (
                (physical_error * physical_valid_mask)
                .sum(dim=(0, 1))
                .detach()
                .double()
                .cpu()
            )
            per_dimension_count += (
                physical_valid_mask.sum(dim=(0, 1)).detach().double().cpu()
            )

    per_dimension_mae = per_dimension_sum / per_dimension_count.clamp_min(1)
    grouped_mae = {
        name: float(
            per_dimension_sum[list(indices)].sum()
            / per_dimension_count[list(indices)].sum().clamp_min(1)
        )
        for name, indices in ACTION_GROUPS.items()
    }
    normalized_l1 = normalized_error_sum / max(normalized_error_count, 1)

    return {
        "eval_loss": normalized_l1,
        "physical_mae": {
            "first_step": first_error_sum / max(first_error_count, 1),
            "execution_window": execution_error_sum / max(execution_error_count, 1),
            "execution_steps": int(policy.config.n_action_steps),
            "full_chunk": physical_error_sum / max(physical_error_count, 1),
            "per_dimension": {
                name: float(value)
                for name, value in zip(
                    ACTION_NAMES, per_dimension_mae.tolist(), strict=True
                )
            },
            "groups": grouped_mae,
        },
    }


# ---------------------------------------------------------------------------
# Sparse sampling for visualization
# ---------------------------------------------------------------------------


def run_sparse_sampling(
    policy,
    preprocessor,
    postprocessor,
    dataset,
    device: torch.device,
    stride: int,
    max_samples: int = 0,
    convert_20d_to_16d: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample observations at stride intervals and collect expert/predicted actions.

    Uses the same collate + preprocessor pipeline as batch evaluation to ensure
    consistency. Samples are processed one-by-one to keep memory low.

    Returns:
        frame_indices: 1-D int array of sampled frame indices.
        expert_array:  [N, 16] float array of expert first-step actions (physical units).
        predicted_array: [N, 16] float array of predicted first-step actions (physical units).
    """
    from utils.xvla_ee import xvla20_to_ee16

    sample_indices = list(range(0, len(dataset), stride))
    if max_samples > 0:
        sample_indices = sample_indices[:max_samples]

    collate_fn = lerobot_collate_fn if dataset.meta.has_language_columns else None

    first_expert: list[np.ndarray] = []
    first_predicted: list[np.ndarray] = []
    frame_indices: list[int] = []

    for idx in tqdm(sample_indices, desc="Sampling for visualization", unit="frame"):
        sample = dataset[idx]
        # Collate single sample into batch format (same as DataLoader)
        if collate_fn is not None:
            batch = collate_fn([sample])
        else:
            batch = default_collate([sample])

        # Save expert action in physical units before preprocessor normalizes it
        expert_first = batch["action"].to(device=device, dtype=torch.float32).clone()

        # Preprocess: normalize images and state (same as batch evaluation)
        for camera_key in dataset.meta.camera_keys:
            if camera_key in batch and batch[camera_key].dtype == torch.uint8:
                batch[camera_key] = batch[camera_key].float() / 255.0
        batch = preprocessor(batch)

        with torch.inference_mode():
            predicted_chunk = policy.predict_action_chunk(batch)  # [1, chunk, D]
        predicted_first_normalized = predicted_chunk[0, 0]  # [D]
        predicted_first_physical = postprocessor(
            predicted_first_normalized.unsqueeze(0)
        )[0].to(device)

        if convert_20d_to_16d:
            pred_20d = predicted_first_physical.cpu().numpy().reshape(-1, 20)
            pred_16d = xvla20_to_ee16(pred_20d).reshape(-1)
            predicted_first_physical = torch.as_tensor(
                pred_16d, dtype=torch.float32, device=device
            )
            expert_20d = expert_first[0].cpu().numpy().reshape(-1, 20)
            expert_16d = xvla20_to_ee16(expert_20d).reshape(-1)
            expert_first = torch.as_tensor(
                expert_16d, dtype=torch.float32, device=device
            ).unsqueeze(0)

        first_expert.append(expert_first[0].cpu().numpy())
        first_predicted.append(predicted_first_physical.cpu().numpy())
        frame_indices.append(idx)

    return (
        np.asarray(frame_indices),
        np.stack(first_expert),
        np.stack(first_predicted),
    )


# ---------------------------------------------------------------------------
# Visualization
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
    )
    dataset = make_dataset(cfg)

    eval_dataset = dataset
    if args.max_samples > 0:
        count = min(args.max_samples, len(dataset))
        indices = torch.linspace(0, len(dataset) - 1, steps=count).long().tolist()
        eval_dataset = Subset(dataset, indices)

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
    policy = make_policy(cfg=policy_cfg, ds_meta=dataset.meta)
    device = next(policy.parameters()).device
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=checkpoint,
        dataset_stats=dataset.meta.stats,
    )
    policy.eval()

    # --- Batch evaluation ---
    print(
        f"Evaluating {args.split} split: {len(episodes)} episodes, {len(eval_dataset)} frames"
    )
    if args.convert_20d_to_16d:
        print("20D→16D conversion enabled for physical MAE")
    metrics = run_batch_evaluation(
        policy,
        preprocessor,
        postprocessor,
        dataloader,
        dataset,
        device,
        convert_20d_to_16d=args.convert_20d_to_16d,
    )

    # Build result dict
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

    # --- Sparse sampling for visualization ---
    print(f"Sampling for visualization (stride={args.stride})...")
    frame_indices, expert_array, predicted_array = run_sparse_sampling(
        policy,
        preprocessor,
        postprocessor,
        dataset,
        device,
        args.stride,
        convert_20d_to_16d=args.convert_20d_to_16d,
    )

    # --- Output ---
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save metrics JSON
    text = json.dumps(result, ensure_ascii=False, indent=2)
    (output_dir / "metrics.json").write_text(text + "\n", encoding="utf-8")
    print(text)
    print(f"Saved metrics: {output_dir / 'metrics.json'}")

    # Save visualization
    save_timeseries_plots(output_dir, frame_indices, expert_array, predicted_array)
    save_mae_bar_charts(
        output_dir,
        metrics["physical_mae"]["per_dimension"],
        metrics["physical_mae"]["groups"],
    )
    print(f"Saved plots to: {output_dir}")


if __name__ == "__main__":
    main()
