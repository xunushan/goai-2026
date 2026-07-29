#!/usr/bin/env python3
"""Evaluate one LeRobot v0.6 ACT checkpoint on the fixed validation episodes.

This is intentionally separate from training: validation data is read once,
after training, and never participates in optimizer updates or checkpoint
selection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from lerobot.configs.default import DatasetConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.policies import make_policy, make_pre_post_processors
from lerobot.utils.collate import lerobot_collate_fn


ACTION_NAMES = (
    "l_x", "l_y", "l_z", "l_w", "l_wx", "l_wy", "l_wz", "l_g",
    "r_x", "r_y", "r_z", "r_w", "r_wx", "r_wy", "r_wz", "r_g",
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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="LeRobot checkpoint pretrained_model directory.",
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--repo-id", default="lerobot_v30_ee")
    parser.add_argument("--split-path", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--video-backend", default="pyav")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Maximum validation frames; 0 evaluates all validation frames.",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint}")

    with args.split_path.resolve().open(encoding="utf-8") as file:
        split = json.load(file)
    val_episodes = [int(value) for value in split["val"]]
    if not val_episodes:
        raise ValueError("Validation split is empty")

    # Reconstruct the policy config from the official pretrained_model output.
    policy_cfg = PreTrainedConfig.from_pretrained(checkpoint)
    policy_cfg.pretrained_path = checkpoint

    # Reuse LeRobot's factory so ACT delta timestamps and video handling match
    # training. eval_split remains zero because episode membership is explicit.
    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(
            repo_id=args.repo_id,
            root=str(args.dataset_root.resolve()),
            episodes=val_episodes,
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
        # Deterministic, evenly spaced coverage is more representative than
        # taking only the first contiguous frames.
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

    policy = make_policy(cfg=policy_cfg, ds_meta=dataset.meta)
    device = next(policy.parameters()).device
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=checkpoint,
        dataset_stats=dataset.meta.stats,
    )
    policy.eval()

    # The trained ACT checkpoint uses per-dimension mean/std normalization.
    # A scalar normalized L1 cannot be unnormalized after aggregation, so keep
    # errors per step and per dimension until physical-unit metrics are summed.
    action_stats = dataset.meta.stats["action"]
    action_mean = torch.as_tensor(action_stats["mean"], dtype=torch.float32, device=device).reshape(1, 1, -1)
    action_std = torch.as_tensor(action_stats["std"], dtype=torch.float32, device=device).reshape(1, 1, -1)

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
            # Preserve expert actions in original dataset units before the
            # official preprocessor normalizes them.
            expert_action_physical = batch["action"].to(device=device, dtype=torch.float32).clone()
            for camera_key in dataset.meta.camera_keys:
                if camera_key in batch and batch[camera_key].dtype == torch.uint8:
                    batch[camera_key] = batch[camera_key].float() / 255.0
            batch = preprocessor(batch)

            # One model forward provides both the official normalized L1 loss
            # and all physical MAE diagnostics. This avoids evaluating val twice.
            predicted_action_normalized = policy.predict_action_chunk(batch)
            expert_action_normalized = batch["action"]
            valid_step_mask = ~batch["action_is_pad"]
            valid_element_mask = valid_step_mask.unsqueeze(-1).expand_as(predicted_action_normalized)

            normalized_error = torch.abs(predicted_action_normalized - expert_action_normalized)
            normalized_error_sum += normalized_error[valid_element_mask].sum().item()
            normalized_error_count += int(valid_element_mask.sum().item())

            predicted_action_physical = predicted_action_normalized * action_std + action_mean
            physical_error = torch.abs(predicted_action_physical - expert_action_physical)
            physical_error_sum += physical_error[valid_element_mask].sum().item()
            physical_error_count += int(valid_element_mask.sum().item())

            first_mask = valid_element_mask[:, :1]
            first_error_sum += physical_error[:, :1][first_mask].sum().item()
            first_error_count += int(first_mask.sum().item())

            execution_steps = min(int(policy.config.n_action_steps), physical_error.shape[1])
            execution_mask = valid_element_mask[:, :execution_steps]
            execution_error_sum += physical_error[:, :execution_steps][execution_mask].sum().item()
            execution_error_count += int(execution_mask.sum().item())

            per_dimension_sum += (
                (physical_error * valid_element_mask).sum(dim=(0, 1)).detach().double().cpu()
            )
            per_dimension_count += valid_element_mask.sum(dim=(0, 1)).detach().double().cpu()

    per_dimension_mae = per_dimension_sum / per_dimension_count.clamp_min(1)
    grouped_mae = {
        name: float(per_dimension_sum[list(indices)].sum() / per_dimension_count[list(indices)].sum().clamp_min(1))
        for name, indices in ACTION_GROUPS.items()
    }
    normalized_l1 = normalized_error_sum / max(normalized_error_count, 1)

    result = {
        "checkpoint": str(checkpoint),
        "split_path": str(args.split_path.resolve()),
        "val_episodes": len(val_episodes),
        "val_frames": len(eval_dataset),
        "batch_size": args.batch_size,
        "device": str(device),
        # ACT is in eval mode, so the VAE latent is zero and official eval_loss
        # equals the valid-element normalized L1 loss.
        "eval_loss": normalized_l1,
        "val_l1_loss": normalized_l1,
        "physical_mae": {
            "first_step": first_error_sum / max(first_error_count, 1),
            "execution_window": execution_error_sum / max(execution_error_count, 1),
            "execution_steps": int(policy.config.n_action_steps),
            "full_chunk": physical_error_sum / max(physical_error_count, 1),
            "per_dimension": {
                name: float(value)
                for name, value in zip(ACTION_NAMES, per_dimension_mae.tolist(), strict=True)
            },
            "groups": grouped_mae,
        },
    }
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)

    output = args.output or checkpoint.parent.parent / "val_metrics.json"
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text + "\n", encoding="utf-8")
    print(f"Saved validation metrics: {output}")


if __name__ == "__main__":
    main()
