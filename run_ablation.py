#!/usr/bin/env python3
"""
Unified ablation experiment runner for NEST (fine-tuning stage).

Supports all 7 ablation variants:
  Group A (no retraining):
    temporal_shuffle  — A1: shuffle encounter order
    mean_pool_tokens  — A2: CLS → mean pool tokens
    freeze_swe        — A3: freeze SWE, tune CSE + head
    freeze_cse        — A3: freeze CSE, tune SWE + head

  Group B (need pretrained ablation checkpoint):
    swe_only    — B1: SWE-only encoder + Bi-GRU
    no_swe      — B2: mean pool → CSE only
    swe_with_pe — B3: SWE + learnable positional encoding

Usage:
    # Single task, single seed
    python run_ablation.py --variant temporal_shuffle \\
        --pretrained checkpoints/best.pt --task mortality --seed 42

    # All 3 binary tasks, multiple seeds, aggregated results
    python run_ablation.py --variant temporal_shuffle \\
        --pretrained checkpoints/best.pt \\
        --tasks mortality readmission_30d prolonged_los \\
        --seeds 42 123 456 --output-dir checkpoints/ablation
"""

import argparse
import json
import sys
import os
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
import pandas as pd
from tokenizers import Tokenizer
from torch.utils.data import DataLoader

from src.finetune.data_utils import (
    FinetuneDataset, DownstreamTask, collate_finetune,
)
from src.finetune.model import create_finetune_model
from src.finetune.trainer import FinetuneTrainer


# Group A variants use standard NEST checkpoint
GROUP_A = {"temporal_shuffle", "mean_pool_tokens", "freeze_swe", "freeze_cse"}
# Group B variants need their own pretrained checkpoint
GROUP_B = {"swe_only", "no_swe", "swe_with_pe"}

BINARY_TASKS = ["mortality", "readmission_30d", "prolonged_los"]


def parse_args():
    parser = argparse.ArgumentParser(description="NEST ablation experiments")

    parser.add_argument("--variant", required=True,
                        choices=sorted(GROUP_A | GROUP_B),
                        help="Ablation variant to run")
    parser.add_argument("--pretrained", required=True,
                        help="Path to pre-trained checkpoint")

    # Task and seed selection
    parser.add_argument("--task", type=str, default=None,
                        choices=BINARY_TASKS,
                        help="Single task to run")
    parser.add_argument("--tasks", nargs="+", default=None,
                        choices=BINARY_TASKS,
                        help="Multiple tasks to run sequentially")
    parser.add_argument("--seed", type=int, default=None,
                        help="Single random seed")
    parser.add_argument("--seeds", nargs="+", type=int, default=None,
                        help="Multiple random seeds")

    # Data paths
    parser.add_argument("--data-path", type=str,
                        default="dataset/mimic4/data/mimic4_tokens.parquet")
    parser.add_argument("--labels-path", type=str,
                        default="dataset/mimic4/data/downstream_labels.csv")
    parser.add_argument("--tokenizer-path", type=str,
                        default="dataset/mimic4/data/mapping/mimic4_tokenizer.json")
    parser.add_argument("--output-dir", type=str,
                        default="checkpoints/ablation")

    # Training hyperparameters
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.001)
    parser.add_argument("--max-seg", type=int, default=8)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--metric", type=str, default="auprc",
                        choices=["auroc", "auprc"])
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--pooling", type=str, default="last_cls",
                        choices=["last_cls", "mean_cls", "attention"])
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--no-class-weights", action="store_true")
    parser.add_argument("--resume", action="store_true",
                        help="Resume each task/seed run from its existing output directory")

    # Logging
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="nest-ablation")

    return parser.parse_args()


def create_model_for_variant(variant, pretrained_path, num_classes, args):
    """Create the appropriate model for the given ablation variant."""

    if variant in GROUP_A:
        # Group A: use standard NEST checkpoint with modifications
        encounter_repr = "mean_pool_tokens" if variant == "mean_pool_tokens" else "cls"
        model = create_finetune_model(
            pretrained_path=pretrained_path,
            num_classes=num_classes,
            dropout=args.dropout,
            pooling=args.pooling,
            freeze_encoder=False,
            encounter_repr=encounter_repr,
        )
        # A3: selective freezing
        if variant == "freeze_swe":
            model.freeze_swe()
            print("  Frozen: SWE parameters")
        elif variant == "freeze_cse":
            model.freeze_cse()
            print("  Frozen: CSE parameters")

        return model

    else:
        # Group B: use ablation-specific checkpoint
        from src.ablation.models import create_ablation_finetune_model
        model = create_ablation_finetune_model(
            variant=variant,
            pretrained_path=pretrained_path,
            num_classes=num_classes,
            dropout=args.dropout,
            pooling=args.pooling,
            max_seq_len=args.max_seq_len,
        )
        return model


def run_single_experiment(variant, task, seed, args):
    """Run a single (variant, task, seed) experiment. Returns test metrics dict."""
    print(f"\n{'='*60}")
    print(f"Ablation: {variant} | Task: {task} | Seed: {seed}")
    print(f"{'='*60}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    tokenizer = Tokenizer.from_file(args.tokenizer_path)
    labels_df = pd.read_csv(args.labels_path)
    task_enum = DownstreamTask(task)

    # Create datasets
    temporal_shuffle = (variant == "temporal_shuffle")
    train_dataset = FinetuneDataset(
        data_path=args.data_path, labels_df=labels_df, tokenizer=tokenizer,
        task=task_enum, max_seg=args.max_seg, max_seq_len=args.max_seq_len,
        split="train", seed=seed, temporal_shuffle=temporal_shuffle,
    )
    val_dataset = FinetuneDataset(
        data_path=args.data_path, labels_df=labels_df, tokenizer=tokenizer,
        task=task_enum, max_seg=args.max_seg, max_seq_len=args.max_seq_len,
        split="val", seed=seed, temporal_shuffle=temporal_shuffle,
    )
    test_dataset = FinetuneDataset(
        data_path=args.data_path, labels_df=labels_df, tokenizer=tokenizer,
        task=task_enum, max_seg=args.max_seg, max_seq_len=args.max_seq_len,
        split="test", seed=seed, temporal_shuffle=temporal_shuffle,
    )

    print(f"  Train: {len(train_dataset):,}  Val: {len(val_dataset):,}  "
          f"Test: {len(test_dataset):,}  Classes: {train_dataset.num_classes}")

    # Create model
    model = create_model_for_variant(
        variant, args.pretrained, train_dataset.num_classes, args
    )

    # Count trainable params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total:,} total, {trainable:,} trainable")

    # Output directory
    run_name = f"{variant}/{task}_seed{seed}"
    output_dir = Path(args.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save run config
    run_config = {
        "variant": variant, "task": task, "seed": seed,
        "pretrained": args.pretrained,
        "lr": args.lr, "batch_size": args.batch_size,
        "epochs": args.epochs, "max_seg": args.max_seg,
        "max_seq_len": args.max_seq_len,
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(run_config, f, indent=2)

    # Create trainer
    resume_from = str(output_dir) if args.resume else None
    if resume_from:
        print(f"  Resume from: {resume_from}")

    trainer = FinetuneTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        task=task_enum,
        output_dir=str(output_dir),
        learning_rate=args.lr,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        use_class_weights=not args.no_class_weights,
        gradient_accumulation_steps=args.gradient_accumulation,
        patience=args.patience,
        metric_for_best_model=args.metric,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=f"{variant}_{task}_s{seed}",
        num_workers=args.num_workers,
        resume_from=resume_from,
    )

    # Train
    best_metric = trainer.train()

    # Test
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size * 2,
        shuffle=False,
        collate_fn=collate_finetune,
        num_workers=args.num_workers,
    )
    test_metrics = trainer.evaluate(test_loader)

    print(f"\n  Test Results: AUROC={test_metrics['auroc']:.4f}  "
          f"AP={test_metrics['auprc']:.4f}")

    # Save results
    results = {
        "variant": variant, "task": task, "seed": seed,
        "best_val_metric": best_metric,
        "test_metrics": test_metrics,
    }
    with open(output_dir / "test_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return test_metrics


def aggregate_results(all_results, output_dir):
    """Aggregate results across seeds: mean +/- std per task.

    Args:
        all_results: dict of {task: [metrics_dict_per_seed]}
        output_dir: where to save summary
    """
    summary = {}
    print(f"\n{'='*60}")
    print("Aggregated Results (mean +/- std)")
    print(f"{'='*60}")

    for task, seed_metrics_list in all_results.items():
        aurocs = [m["auroc"] for m in seed_metrics_list]
        auprcs = [m["auprc"] for m in seed_metrics_list]

        summary[task] = {
            "auroc_mean": float(np.mean(aurocs)),
            "auroc_std": float(np.std(aurocs)),
            "auprc_mean": float(np.mean(auprcs)),
            "auprc_std": float(np.std(auprcs)),
            "n_seeds": len(seed_metrics_list),
            "per_seed": seed_metrics_list,
        }

        print(f"  {task:20s}: AUROC={np.mean(aurocs):.4f} +/- {np.std(aurocs):.4f}  "
              f"AP={np.mean(auprcs):.4f} +/- {np.std(auprcs):.4f}")

    output_path = Path(output_dir) / "summary.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {output_path}")

    return summary


def main():
    args = parse_args()

    # Resolve tasks and seeds
    if args.tasks:
        tasks = args.tasks
    elif args.task:
        tasks = [args.task]
    else:
        tasks = BINARY_TASKS

    if args.seeds:
        seeds = args.seeds
    elif args.seed is not None:
        seeds = [args.seed]
    else:
        seeds = [42, 123, 456]

    variant = args.variant
    variant_output_dir = Path(args.output_dir) / variant

    print(f"Ablation experiment: {variant}")
    print(f"Tasks: {tasks}")
    print(f"Seeds: {seeds}")
    print(f"Output: {variant_output_dir}")
    print(f"Pretrained: {args.pretrained}")

    # Run all (task, seed) combinations
    all_results = {task: [] for task in tasks}

    for task in tasks:
        for seed in seeds:
            metrics = run_single_experiment(variant, task, seed, args)
            all_results[task].append(metrics)

    # Aggregate
    if len(seeds) > 1:
        aggregate_results(all_results, variant_output_dir)
    else:
        # Single seed — just print results
        print(f"\n{'='*60}")
        print("Results")
        print(f"{'='*60}")
        for task, metrics_list in all_results.items():
            m = metrics_list[0]
            print(f"  {task:20s}: AUROC={m['auroc']:.4f}  AP={m['auprc']:.4f}")


if __name__ == "__main__":
    main()
