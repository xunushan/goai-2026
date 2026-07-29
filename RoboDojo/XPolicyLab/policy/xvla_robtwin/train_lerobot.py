#!/usr/bin/env python3
"""LoRA post-train X-VLA-RoboTwin2 on RoboDojo LeRobot v3 EE data."""

from __future__ import annotations

import argparse
import json
import logging
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from accelerate import Accelerator
from peft import LoraConfig, get_peft_model
from torch.optim import AdamW
from torch.utils.data import DataLoader

from xvla.models.modeling_xvla import XVLA
from xvla.models.processing_xvla import XVLAProcessor
from lerobot_v3_dataset import (
    XVLALeRobotV3EEDataset,
    episodes_from_split,
    parse_episode_list,
)


LOGGER = logging.getLogger("xvla_lerobot")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--repo-id", default="lerobot_v30_ee")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split-path")
    parser.add_argument("--episodes", help="JSON list or path to a JSON list")
    parser.add_argument("--tasks-json", default="[]", help="Exact task strings as a JSON list")
    parser.add_argument("--allow-all-episodes", action="store_true")
    parser.add_argument("--allow-all-tasks", action="store_true")
    parser.add_argument("--domain-id", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--save-interval", type=int, default=1000)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--mixed-precision", choices=("no", "fp16", "bf16"), default="fp16")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--video-backend", default="pyav")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def lr_multiplier(step: int, warmup_steps: int) -> float:
    if warmup_steps <= 0:
        return 1.0
    return min(1.0, (step + 1) / warmup_steps)


def copy_processor_files(model_path: Path, output_path: Path) -> None:
    names = (
        "preprocessor_config.json", "tokenizer_config.json", "tokenizer.json",
        "vocab.json", "merges.txt", "special_tokens_map.json", "added_tokens.json",
    )
    if not model_path.is_dir():
        return
    for name in names:
        source = model_path / name
        if source.is_file():
            shutil.copy2(source, output_path / name)


def main(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    if args.domain_id != 6:
        raise ValueError(
            "RoboDojo post-training from X-VLA-RoboTwin2 must use domain_id=6; "
            "a different ID requires an explicit new-domain experiment."
        )
    if args.steps <= 0 or args.batch_size <= 0:
        raise ValueError("steps and batch-size must be positive.")
    tasks = json.loads(args.tasks_json)
    if not isinstance(tasks, list) or not all(isinstance(item, str) for item in tasks):
        raise ValueError("--tasks-json must be a JSON list of exact task strings.")
    if args.split_path and args.episodes:
        raise ValueError("Use only one of --split-path and --episodes.")
    if not args.split_path and not args.episodes and not args.allow_all_episodes:
        raise ValueError(
            "A fixed --split-path/--episodes selection is required. "
            "Use --allow-all-episodes only for an intentional all-data run."
        )
    if not tasks and not args.allow_all_tasks:
        raise ValueError(
            "--tasks-json must select exact tasks. "
            "Use --allow-all-tasks only for an intentional all-task run."
        )
    episodes = (
        episodes_from_split(args.split_path, "train")
        if args.split_path
        else parse_episode_list(args.episodes)
    )

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=args.output_dir,
    )
    set_seed(args.seed + accelerator.process_index)

    model = XVLA.from_pretrained(args.model_path, trust_remote_code=True)
    if model.action_mode != "ee6d":
        raise ValueError(f"Base model action_mode must be 'ee6d', got {model.action_mode!r}.")
    processor = XVLAProcessor.from_pretrained(args.model_path)
    dataset = XVLALeRobotV3EEDataset(
        root=args.dataset_root,
        repo_id=args.repo_id,
        num_actions=model.num_actions,
        domain_id=args.domain_id,
        episodes=episodes,
        task_allowlist=tasks,
        training=True,
        video_backend=args.video_backend,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        drop_last=True,
    )
    if len(loader) == 0:
        raise ValueError("Dataset is smaller than one batch; reduce --batch-size.")

    sample = dataset[0]
    LOGGER.info(
        "dataset=%s samples=%d episodes=%s action=%s image=%s domain_id=%d",
        args.dataset_root,
        len(dataset),
        "all" if episodes is None else len(episodes),
        tuple(sample["action"].shape),
        tuple(sample["image_input"].shape),
        int(sample["domain_id"]),
    )
    if args.dry_run:
        LOGGER.info("Dry run complete; model and one transformed sample are valid.")
        return

    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        bias="none",
        target_modules="all-linear",
        modules_to_save=[
            "transformer.soft_prompt_hub",
            "transformer.action_encoder",
            "transformer.action_decoder",
        ],
    )
    model = get_peft_model(model, lora)
    if accelerator.is_main_process:
        model.print_trainable_parameters()

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = AdamW(
        trainable,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
    accelerator.init_trackers(
        "XVLA-RoboDojo-LeRobot-v3",
        config={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    )

    output_dir = Path(args.output_dir)
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "train_config.json").write_text(
            json.dumps(vars(args), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        if args.split_path:
            shutil.copy2(args.split_path, output_dir / "dataset_split.json")

    model.train()
    data_iterator = iter(loader)
    started = time.time()
    optimizer_step = 0
    micro_step = 0
    while optimizer_step < args.steps:
        try:
            batch = next(data_iterator)
        except StopIteration:
            data_iterator = iter(loader)
            batch = next(data_iterator)

        language = processor.encode_language(batch.pop("language_instruction"))
        batch.update({
            key: value.to(accelerator.device, non_blocking=True)
            if isinstance(value, torch.Tensor) else value
            for key, value in language.items()
        })
        with accelerator.accumulate(model):
            with accelerator.autocast():
                loss_dict = model(**batch)
                loss = sum(loss_dict.values())
            accelerator.backward(loss)
            if accelerator.sync_gradients and args.max_grad_norm > 0:
                accelerator.clip_grad_norm_(trainable, args.max_grad_norm)
            multiplier = lr_multiplier(optimizer_step, args.warmup_steps)
            for group in optimizer.param_groups:
                group["lr"] = args.learning_rate * multiplier
            optimizer.step()
            optimizer.zero_grad()
        micro_step += 1
        if not accelerator.sync_gradients:
            continue

        optimizer_step += 1
        completed = optimizer_step
        if completed % args.log_interval == 0 and accelerator.is_main_process:
            metrics = {
                key: float(value.detach().float()) for key, value in loss_dict.items()
            }
            metrics["loss_total"] = float(loss.detach().float())
            metrics["learning_rate"] = optimizer.param_groups[0]["lr"]
            metrics["steps_per_second"] = completed / max(time.time() - started, 1e-6)
            metrics["micro_step"] = micro_step
            LOGGER.info("step=%d/%d %s", completed, args.steps, metrics)
            accelerator.log(metrics, step=completed)

        should_save = completed % args.save_interval == 0 or completed == args.steps
        if should_save:
            accelerator.wait_for_everyone()
            save_dir = output_dir / f"ckpt-{completed}"
            if accelerator.is_main_process:
                save_dir.mkdir(parents=True, exist_ok=True)
                unwrapped = accelerator.unwrap_model(model)
                unwrapped.save_pretrained(
                    save_dir,
                    safe_serialization=True,
                    save_function=accelerator.save,
                )
                copy_processor_files(Path(args.model_path), save_dir)
                (save_dir / "state.json").write_text(
                    json.dumps({"global_step": completed}),
                    encoding="utf-8",
                )
            accelerator.wait_for_everyone()

    accelerator.end_training()


if __name__ == "__main__":
    main(parse_args())
