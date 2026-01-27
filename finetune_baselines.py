#!/usr/bin/env python3
"""
Baseline Models Fine-tuning Script

Uses FlatFinetuneDataset for consistent flat (512,) format with pretrain.

Usage:
    python finetune_baselines.py --model core-behrt --task mortality \
        --pretrained checkpoints/core-behrt/best_model.pt
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import json
from datetime import datetime
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score

from tokenizers import Tokenizer

# Use flat dataset for baselines (consistent with pretrain)
from src.baselines.data_utils import (
    FlatFinetuneDataset, NextVisitFlatDataset,
    DownstreamTask, TASK_CONFIGS,
    collate_flat_finetune, collate_next_visit_flat
)
from src.baselines.core_behrt import BEHRTConfig, BEHRTForSequenceClassification
from src.baselines.heart import HEARTConfig, HEARTForSequenceClassification


def compute_recall_at_k(preds: np.ndarray, targets: np.ndarray, k: int) -> float:
    """Compute Recall@k for set prediction task.

    Args:
        preds: Predicted probabilities, shape (batch_size, num_classes)
        targets: Binary target vectors, shape (batch_size, num_classes)
        k: Number of top predictions to consider

    Returns:
        Average recall@k across all samples
    """
    recalls = []
    for pred, target in zip(preds, targets):
        # Get top-k predicted indices
        top_k_indices = np.argsort(pred)[-k:]
        # Get true positive indices
        true_indices = np.where(target > 0)[0]
        if len(true_indices) == 0:
            continue
        # Count hits
        hits = len(set(top_k_indices) & set(true_indices))
        recalls.append(hits / len(true_indices))
    return np.mean(recalls) if recalls else 0.0


def compute_ndcg_at_k(preds: np.ndarray, targets: np.ndarray, k: int) -> float:
    """Compute NDCG@k for set prediction task.

    Args:
        preds: Predicted probabilities, shape (batch_size, num_classes)
        targets: Binary target vectors, shape (batch_size, num_classes)
        k: Number of top predictions to consider

    Returns:
        Average NDCG@k across all samples
    """
    ndcgs = []
    for pred, target in zip(preds, targets):
        # Get top-k predicted indices
        top_k_indices = np.argsort(pred)[-k:][::-1]  # Descending order
        # Compute DCG
        dcg = 0.0
        for i, idx in enumerate(top_k_indices):
            if target[idx] > 0:
                dcg += 1.0 / np.log2(i + 2)  # i+2 because position starts from 1
        # Compute ideal DCG
        num_relevant = int(target.sum())
        if num_relevant == 0:
            continue
        ideal_dcg = sum(1.0 / np.log2(i + 2) for i in range(min(k, num_relevant)))
        ndcgs.append(dcg / ideal_dcg if ideal_dcg > 0 else 0.0)
    return np.mean(ndcgs) if ndcgs else 0.0


def create_finetune_model(model_type, pretrained_path, num_classes, vocab_size, args, is_multilabel=False):
    """Create fine-tune model from pretrained checkpoint."""

    checkpoint = torch.load(pretrained_path, map_location="cpu", weights_only=False)

    # Use max_seq_len directly (flat format, same as pretrain)
    max_seq_len = args.max_seq_len

    if model_type == "core-behrt":
        config = BEHRTConfig(
            vocab_size=vocab_size,
            d_model=args.d_model,
            n_blocks=args.n_blocks,
            n_heads=args.n_heads,
            d_ff=args.d_ff,
            max_seq_len=max_seq_len,
            dropout=0.0,
        )
        model = BEHRTForSequenceClassification(
            config=config,
            num_classes=num_classes,
            dropout=args.dropout,
            is_multilabel=is_multilabel,
        )

    elif model_type == "heart":
        config = HEARTConfig(
            vocab_size=vocab_size,
            d_model=args.d_model,
            n_blocks=args.n_blocks,
            n_heads=args.n_heads,
            d_ff=args.d_ff,
            max_seq_len=max_seq_len,
            dropout=0.0,
            n_token_types=8,
            edge_hidden_size=64,
            max_visits=50,  # Must match pretrained model
        )
        model = HEARTForSequenceClassification(
            config=config,
            num_classes=num_classes,
            dropout=args.dropout,
            is_multilabel=is_multilabel,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Load pretrained encoder weights
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    encoder_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("encoder."):
            encoder_state_dict[k] = v
        elif not k.startswith("classifier.") and not k.startswith("mlm_head."):
            encoder_state_dict[f"encoder.{k}"] = v

    missing, unexpected = model.load_state_dict(encoder_state_dict, strict=False)
    missing = [k for k in missing if not k.startswith("classifier.")]
    if missing:
        print(f"  Note: Missing encoder keys (expected for new classifier): {len(missing)}")

    if args.freeze_encoder:
        model.freeze_encoder()
        print("  Encoder weights frozen")

    return model


class BaselineFinetuneTrainer:
    """Trainer for baseline model fine-tuning."""

    def __init__(
        self,
        model,
        model_type: str,
        train_dataset,
        val_dataset,
        task: DownstreamTask,
        output_dir: str,
        learning_rate: float = 1e-5,
        batch_size: int = 32,
        num_epochs: int = 30,
        warmup_ratio: float = 0.1,
        weight_decay: float = 0.01,
        use_class_weights: bool = True,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        patience: int = 5,
        use_amp: bool = False,
        device: str = "cuda",
        num_workers: int = 4,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.model_type = model_type
        self.task = task
        self.task_config = TASK_CONFIGS[task]
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.num_epochs = num_epochs
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.patience = patience
        self.use_amp = use_amp and self.device.type == "cuda"

        self.is_multilabel = self.task_config.get("is_multilabel", False)
        self.is_set_prediction = self.task_config.get("is_set_prediction", False)
        num_classes = model.num_classes

        # Determine best metric for model selection
        if self.is_set_prediction:
            self.metric_for_best_model = "recall@20"
        elif num_classes > 2 or self.is_multilabel:
            self.metric_for_best_model = "auroc"
        else:
            self.metric_for_best_model = "auprc"

        # Select appropriate collate function
        if self.is_set_prediction:
            collate_fn = collate_next_visit_flat
        else:
            collate_fn = collate_flat_finetune

        # Data loaders
        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            collate_fn=collate_fn, num_workers=num_workers,
            pin_memory=(self.device.type == "cuda")
        )
        self.val_loader = DataLoader(
            val_dataset, batch_size=batch_size * 2, shuffle=False,
            collate_fn=collate_fn, num_workers=num_workers,
            pin_memory=(self.device.type == "cuda")
        )

        # Class weights
        self.class_weights = None
        if use_class_weights and hasattr(train_dataset, 'get_class_weights'):
            weights = train_dataset.get_class_weights()
            self.class_weights = weights.to(self.device)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        # Scheduler
        total_steps = len(self.train_loader) * num_epochs // gradient_accumulation_steps
        warmup_steps = int(total_steps * warmup_ratio)

        warmup_scheduler = LinearLR(
            self.optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps
        )
        main_scheduler = CosineAnnealingLR(
            self.optimizer, T_max=max(1, total_steps - warmup_steps), eta_min=learning_rate * 0.01
        )
        self.scheduler = SequentialLR(
            self.optimizer, schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_steps]
        )

        self.scaler = GradScaler("cuda") if self.use_amp else None
        self.global_step = 0
        self.best_metric = 0.0
        self.patience_counter = 0

    def _forward_step(self, batch):
        """Forward pass with flat batch (already in correct format)."""
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        t = batch["t"].to(self.device)
        labels = batch["label"].to(self.device)

        if self.model_type == "core-behrt":
            output = self.model(input_ids, attention_mask, t, labels=labels)
        else:  # heart
            token_types = batch["token_types"].to(self.device)
            visit_ids = batch["visit_ids"].to(self.device)
            output = self.model(input_ids, attention_mask, token_types, visit_ids, t, labels=labels)

        return output, labels

    def train_epoch(self, epoch):
        """Train one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = 0

        self.optimizer.zero_grad()
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}")

        for step, batch in enumerate(pbar):
            if self.use_amp:
                with autocast("cuda"):
                    output, labels = self._forward_step(batch)
                    loss = output["loss"] / self.gradient_accumulation_steps

                self.scaler.scale(loss).backward()

                if (step + 1) % self.gradient_accumulation_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1
            else:
                output, labels = self._forward_step(batch)
                loss = output["loss"] / self.gradient_accumulation_steps
                loss.backward()

                if (step + 1) % self.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1

            total_loss += loss.item() * self.gradient_accumulation_steps
            num_batches += 1
            pbar.set_postfix({"loss": f"{total_loss / num_batches:.4f}"})

        return total_loss / num_batches

    @torch.no_grad()
    def evaluate(self, dataloader=None):
        """Evaluate on validation/test set."""
        if dataloader is None:
            dataloader = self.val_loader

        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []

        for batch in tqdm(dataloader, desc="Evaluating"):
            if self.use_amp:
                with autocast("cuda"):
                    output, labels = self._forward_step(batch)
            else:
                output, labels = self._forward_step(batch)

            logits = output["logits"].cpu()
            labels = labels.cpu()

            num_classes = self.model.num_classes

            # Handle different task types
            if self.is_set_prediction or self.is_multilabel:
                # Multilabel/set prediction: use sigmoid
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).long()
                all_probs.append(probs.numpy())
                all_preds.append(preds.numpy())
                all_labels.append(labels.numpy())
            elif num_classes == 1:
                probs = torch.sigmoid(logits.squeeze(-1))
                preds = (probs > 0.5).long()
                all_probs.extend(probs.numpy())
                all_preds.extend(preds.numpy())
                all_labels.extend(labels.numpy())
            elif num_classes == 2:
                probs = F.softmax(logits, dim=-1)[:, 1]
                preds = (probs > 0.5).long()
                all_probs.extend(probs.numpy())
                all_preds.extend(preds.numpy())
                all_labels.extend(labels.numpy())
            else:
                probs = F.softmax(logits, dim=-1)
                preds = logits.argmax(dim=-1)
                all_probs.append(probs.numpy())
                all_preds.extend(preds.numpy())
                all_labels.extend(labels.numpy())

        # Compute metrics based on task type
        if self.is_set_prediction:
            # Set prediction metrics: recall@k, NDCG@k
            all_probs = np.vstack(all_probs)
            all_labels = np.vstack(all_labels)

            metrics = {}
            for k in [10, 20, 30]:
                metrics[f"recall@{k}"] = compute_recall_at_k(all_probs, all_labels, k)
                metrics[f"ndcg@{k}"] = compute_ndcg_at_k(all_probs, all_labels, k)
            return metrics

        elif self.is_multilabel:
            # Multilabel classification metrics
            all_probs = np.vstack(all_probs)
            all_labels = np.vstack(all_labels)
            all_preds = np.vstack(all_preds)

            metrics = {}
            try:
                # Macro-averaged AUROC (average over classes)
                aurocs = []
                for c in range(all_labels.shape[1]):
                    if all_labels[:, c].sum() > 0 and all_labels[:, c].sum() < len(all_labels):
                        aurocs.append(roc_auc_score(all_labels[:, c], all_probs[:, c]))
                metrics["auroc"] = np.mean(aurocs) if aurocs else 0.0

                # Macro-averaged AUPRC
                auprcs = []
                for c in range(all_labels.shape[1]):
                    if all_labels[:, c].sum() > 0:
                        auprcs.append(average_precision_score(all_labels[:, c], all_probs[:, c]))
                metrics["auprc"] = np.mean(auprcs) if auprcs else 0.0
            except ValueError as e:
                print(f"  Warning: Could not compute metrics: {e}")
                metrics["auroc"] = 0.0
                metrics["auprc"] = 0.0

            # F1 scores
            metrics["f1_macro"] = f1_score(all_labels, all_preds, average="macro", zero_division=0)
            metrics["f1_micro"] = f1_score(all_labels, all_preds, average="micro", zero_division=0)
            return metrics

        else:
            # Standard classification
            all_preds = np.array(all_preds)
            all_labels = np.array(all_labels)
            num_classes = self.model.num_classes

            metrics = {"accuracy": accuracy_score(all_labels, all_preds)}

            if num_classes <= 2:
                all_probs = np.array(all_probs)
                try:
                    metrics["auroc"] = roc_auc_score(all_labels, all_probs)
                    metrics["auprc"] = average_precision_score(all_labels, all_probs)
                except ValueError:
                    metrics["auroc"] = 0.0
                    metrics["auprc"] = 0.0
                metrics["f1"] = f1_score(all_labels, all_preds)
            else:
                all_probs = np.vstack(all_probs)
                try:
                    # Multi-class AUROC: compute per-class OVR AUROC and average
                    class_aurocs = []
                    for c in range(num_classes):
                        binary_labels = (all_labels == c).astype(int)
                        if binary_labels.sum() > 0 and binary_labels.sum() < len(binary_labels):
                            class_auroc = roc_auc_score(binary_labels, all_probs[:, c])
                            class_aurocs.append(class_auroc)
                    metrics["auroc"] = np.mean(class_aurocs) if class_aurocs else 0.0
                except ValueError as e:
                    print(f"  Warning: Could not compute AUROC: {e}")
                    metrics["auroc"] = 0.0
                metrics["f1_macro"] = f1_score(all_labels, all_preds, average="macro")

            return metrics

    def train(self):
        """Full training loop."""
        print(f"\n{'='*60}")
        print(f"Training {self.model_type.upper()} on {self.task.value}")
        print(f"{'='*60}")
        print(f"  Train samples: {len(self.train_loader.dataset):,}")
        print(f"  Val samples:   {len(self.val_loader.dataset):,}")
        print(f"{'='*60}\n")

        for epoch in range(self.num_epochs):
            train_loss = self.train_epoch(epoch)
            val_metrics = self.evaluate()

            print(f"\nEpoch {epoch + 1}/{self.num_epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            for k, v in val_metrics.items():
                print(f"  Val {k}: {v:.4f}")

            current_metric = val_metrics.get(self.metric_for_best_model, val_metrics.get("accuracy", 0))
            if current_metric > self.best_metric:
                self.best_metric = current_metric
                self.patience_counter = 0

                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "best_metric": self.best_metric,
                }, self.output_dir / "best_model.pt")
                print(f"  -> Saved best model ({self.metric_for_best_model}: {self.best_metric:.4f})")
            else:
                self.patience_counter += 1

            if self.patience_counter >= self.patience:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break

        return self.best_metric


def parse_args():
    parser = argparse.ArgumentParser(description="Baseline Models Fine-tuning")

    parser.add_argument("--model", type=str, required=True,
                       choices=["core-behrt", "heart"])
    parser.add_argument("--task", type=str, required=True,
                       choices=["mortality", "readmission_30d", "prolonged_los",
                                "icd_chapter", "icd_category_multilabel", "next_visit"])

    parser.add_argument("--pretrained", type=str, required=True)
    parser.add_argument("--data_path", type=str,
                       default="dataset/mimic4/data/mimic4_tokens.parquet")
    parser.add_argument("--labels_path", type=str,
                       default="dataset/mimic4/data/downstream_labels.csv")
    parser.add_argument("--tokenizer_path", type=str, default="tokenizer.json")
    parser.add_argument("--output_dir", type=str, default="checkpoints/finetune")

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=30)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)

    # Model config (should match pretrained)
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--n_blocks", type=int, default=6)
    parser.add_argument("--n_heads", type=int, default=12)
    parser.add_argument("--d_ff", type=int, default=2048)
    parser.add_argument("--max_seq_len", type=int, default=2048,
                        help="Sequence length (same as pretrain, default 512)")
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--freeze_encoder", action="store_true")
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)

    return parser.parse_args()


def main():
    args = parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    task = DownstreamTask(args.task)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"{args.model}_{args.task}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"  {args.model.upper()} Fine-tuning on {args.task}")
    print("=" * 60)

    # Load tokenizer
    tokenizer = Tokenizer.from_file(args.tokenizer_path)
    print(f"Vocab size: {tokenizer.get_vocab_size()}")

    # Load labels
    print(f"Loading labels from {args.labels_path}")
    labels_df = pd.read_csv(args.labels_path)

    # Create datasets
    is_heart = (args.model == "heart")
    task_config = TASK_CONFIGS[task]
    is_set_prediction = task_config.get("is_set_prediction", False)

    if is_set_prediction:
        # Use NextVisitFlatDataset for NEXT_VISIT task
        print("\nCreating datasets (using NextVisitFlatDataset)...")
        train_dataset = NextVisitFlatDataset(
            data_path=args.data_path,
            labels_df=labels_df,
            tokenizer=tokenizer,
            max_seq_len=args.max_seq_len,
            split="train",
            seed=args.seed,
            include_token_types=is_heart,
            include_visit_ids=is_heart,
        )
        val_dataset = NextVisitFlatDataset(
            data_path=args.data_path,
            labels_df=labels_df,
            tokenizer=tokenizer,
            max_seq_len=args.max_seq_len,
            split="val",
            seed=args.seed,
            include_token_types=is_heart,
            include_visit_ids=is_heart,
        )
        test_dataset = NextVisitFlatDataset(
            data_path=args.data_path,
            labels_df=labels_df,
            tokenizer=tokenizer,
            max_seq_len=args.max_seq_len,
            split="test",
            seed=args.seed,
            include_token_types=is_heart,
            include_visit_ids=is_heart,
        )
    else:
        # Use FlatFinetuneDataset for classification tasks
        print("\nCreating datasets (using FlatFinetuneDataset)...")
        train_dataset = FlatFinetuneDataset(
            data_path=args.data_path,
            labels_df=labels_df,
            tokenizer=tokenizer,
            task=task,
            max_seq_len=args.max_seq_len,
            split="train",
            seed=args.seed,
            include_token_types=is_heart,
            include_visit_ids=is_heart,
        )
        val_dataset = FlatFinetuneDataset(
            data_path=args.data_path,
            labels_df=labels_df,
            tokenizer=tokenizer,
            task=task,
            max_seq_len=args.max_seq_len,
            split="val",
            seed=args.seed,
            include_token_types=is_heart,
            include_visit_ids=is_heart,
        )
        test_dataset = FlatFinetuneDataset(
            data_path=args.data_path,
            labels_df=labels_df,
            tokenizer=tokenizer,
            task=task,
            max_seq_len=args.max_seq_len,
            split="test",
            seed=args.seed,
            include_token_types=is_heart,
            include_visit_ids=is_heart,
        )

    num_classes = train_dataset.num_classes
    print(f"  Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    print(f"  Num classes: {num_classes}")

    # Determine if multilabel task (includes set prediction tasks like next_visit)
    is_multilabel = is_set_prediction or task_config.get("is_multilabel", False)

    # Create model
    print(f"\nLoading pre-trained model from {args.pretrained}")
    vocab_size = tokenizer.get_vocab_size()
    model = create_finetune_model(args.model, args.pretrained, num_classes, vocab_size, args, is_multilabel=is_multilabel)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {n_params:,}")

    # Save config
    with open(output_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # Create trainer
    trainer = BaselineFinetuneTrainer(
        model=model,
        model_type=args.model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        task=task,
        output_dir=output_dir,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        patience=args.patience,
        use_amp=args.use_amp,
        device=args.device,
        num_workers=args.num_workers,
    )

    # Train
    best_metric = trainer.train()

    # Test
    print("\n" + "=" * 60)
    print("Testing...")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    best_ckpt = torch.load(output_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])

    # Use appropriate collate function
    test_collate_fn = collate_next_visit_flat if is_set_prediction else collate_flat_finetune
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size * 2, shuffle=False,
        collate_fn=test_collate_fn, num_workers=args.num_workers
    )
    test_metrics = trainer.evaluate(test_loader)

    print("\nTest Results:")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")

    # Save results
    results = {
        "model": args.model,
        "task": args.task,
        "best_val_metric": best_metric,
        "test_metrics": test_metrics,
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
