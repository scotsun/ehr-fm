#!/usr/bin/env python3
"""
Run fine-tuning for HAT downstream tasks.

Usage:
    # Single task
    python run_finetune.py --task mortality --pretrained checkpoints/best.pt

    # All tasks
    python run_finetune.py --all --pretrained checkpoints/best.pt

    # With wandb logging
    python run_finetune.py --task mortality --pretrained checkpoints/best.pt --wandb
"""

import argparse
import json
from pathlib import Path
from datetime import datetime

import torch
import pandas as pd
from tokenizers import Tokenizer
from torch.utils.data import DataLoader

from src.finetune.data_utils import FinetuneDataset, DownstreamTask, collate_finetune
from src.finetune.model import create_finetune_model
from src.finetune.trainer import FinetuneTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune HAT on downstream tasks")

    # Required arguments
    parser.add_argument(
        "--pretrained",
        type=str,
        required=True,
        help="Path to pre-trained checkpoint",
    )

    # Task selection
    parser.add_argument(
        "--task",
        type=str,
        choices=["mortality", "readmission_30d", "prolonged_los", "icd_chapter", "icd_category_multilabel"],
        help="Task to fine-tune on",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all tasks sequentially",
    )

    # Data paths
    parser.add_argument(
        "--data-path",
        type=str,
        default="dataset/mimic4/data/mimic4_tokens.parquet",
        help="Path to parquet data directory",
    )
    parser.add_argument(
        "--labels-path",
        type=str,
        default="dataset/mimic4/data/downstream_labels.csv",
        help="Path to labels CSV file",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default="dataset/mimic4/data/mapping/mimic4_tokenizer.json",
        help="Path to tokenizer file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="checkpoints/finetune",
        help="Output directory for checkpoints",
    )

    # Model arguments
    parser.add_argument(
        "--pooling",
        type=str,
        default="last_cls",
        choices=["last_cls", "mean_cls", "attention"],
        help="Pooling strategy for classification",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout rate for classifier",
    )
    parser.add_argument(
        "--freeze-encoder",
        action="store_true",
        help="Freeze encoder weights (feature extraction mode)",
    )

    # Training arguments
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--warmup-ratio", type=float, default=0.1, help="Warmup ratio")
    parser.add_argument("--weight-decay", type=float, default=0.001, help="Weight decay")
    parser.add_argument("--max-seg", type=int, default=8, help="Max segments")
    parser.add_argument("--max-seq-len", type=int, default=512, help="Max sequence length")
    parser.add_argument(
        "--gradient-accumulation", type=int, default=1, help="Gradient accumulation steps"
    )

    # Early stopping
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience")
    parser.add_argument(
        "--metric",
        type=str,
        default="auprc",
        choices=["auroc", "auprc", "accuracy"],
        help="Metric for best model selection",
    )

    # Logging
    parser.add_argument("--wandb", action="store_true", help="Use wandb logging")
    parser.add_argument("--wandb-project", type=str, default="hat-finetune", help="Wandb project")

    # Other
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num-workers", type=int, default=4, help="Data loader workers")
    parser.add_argument(
        "--no-class-weights",
        action="store_true",
        help="Disable class weighting for imbalanced data",
    )

    return parser.parse_args()


def run_single_task(args, task: str):
    """Run fine-tuning for a single task."""
    print(f"\n{'='*60}")
    print(f"Fine-tuning HAT for task: {task}")
    print(f"{'='*60}")

    # Set seed
    torch.manual_seed(args.seed)

    # Load tokenizer
    print(f"Loading tokenizer from {args.tokenizer_path}")
    tokenizer = Tokenizer.from_file(args.tokenizer_path)

    # Load labels
    print(f"Loading labels from {args.labels_path}")
    labels_df = pd.read_csv(args.labels_path)

    # Create task enum
    task_enum = DownstreamTask(task)

    # Create datasets
    print("Creating datasets...")
    train_dataset = FinetuneDataset(
        data_path=args.data_path,
        labels_df=labels_df,
        tokenizer=tokenizer,
        task=task_enum,
        max_seg=args.max_seg,
        max_seq_len=args.max_seq_len,
        split="train",
        seed=args.seed,
    )
    val_dataset = FinetuneDataset(
        data_path=args.data_path,
        labels_df=labels_df,
        tokenizer=tokenizer,
        task=task_enum,
        max_seg=args.max_seg,
        max_seq_len=args.max_seq_len,
        split="val",
        seed=args.seed,
    )

    print(f"  Train samples: {len(train_dataset):,}")
    print(f"  Val samples:   {len(val_dataset):,}")
    print(f"  Num classes:   {train_dataset.num_classes}")

    # Create model
    print(f"Loading pre-trained model from {args.pretrained}")
    model = create_finetune_model(
        pretrained_path=args.pretrained,
        num_classes=train_dataset.num_classes,
        dropout=args.dropout,
        pooling=args.pooling,
        freeze_encoder=args.freeze_encoder,
    )

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"{task}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save args
    with open(output_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # Create trainer
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
        wandb_run_name=f"{task}_{timestamp}",
        num_workers=args.num_workers,
    )

    # Train
    best_metric = trainer.train()

    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_dataset = FinetuneDataset(
        data_path=args.data_path,
        labels_df=labels_df,
        tokenizer=tokenizer,
        task=task_enum,
        max_seg=args.max_seg,
        max_seq_len=args.max_seq_len,
        split="test",
        seed=args.seed,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size * 2,
        shuffle=False,
        collate_fn=collate_finetune,
        num_workers=args.num_workers,
    )

    test_metrics = trainer.evaluate(test_loader)

    # Print results
    print(f"\n{'='*60}")
    print(f"Test Results for {task}")
    print(f"{'='*60}")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")

    # Save results
    results = {
        "task": task,
        "best_val_metric": best_metric,
        "test_metrics": test_metrics,
        "args": vars(args),
    }
    with open(output_dir / "test_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return test_metrics


def main():
    args = parse_args()

    if args.all:
        # Run all tasks
        tasks = ["mortality", "readmission_30d", "prolonged_los", "icd_chapter", "icd_category_multilabel"]
        all_results = {}

        for task in tasks:
            results = run_single_task(args, task)
            all_results[task] = results

        # Print summary
        print(f"\n{'='*60}")
        print("Summary of All Tasks")
        print(f"{'='*60}")
        for task, metrics in all_results.items():
            auroc = metrics.get("auroc", "N/A")
            auprc = metrics.get("auprc", "N/A")
            if isinstance(auroc, float):
                auroc = f"{auroc:.4f}"
            if isinstance(auprc, float):
                auprc = f"{auprc:.4f}"
            print(f"  {task:20s}: AUROC={auroc}, AUPRC={auprc}")

        # Save summary
        summary_path = Path(args.output_dir) / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nSummary saved to {summary_path}")

    elif args.task:
        run_single_task(args, args.task)
    else:
        print("Error: Please specify --task or --all")
        exit(1)


if __name__ == "__main__":
    main()
