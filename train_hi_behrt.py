#!/usr/bin/env python3
"""
Hi-BEHRT Training Script with BYOL Pre-trained Weights

Hi-BEHRT uses BYOL (Bootstrap Your Own Latent) for self-supervised pre-training.
This script supports:
1. Training from scratch (no pre-training)
2. Fine-tuning from BYOL pre-trained weights

Time Encoding:
- Uses Time2Vec to encode cumulative time (cumsum of days_since_prior_admission)
- This replaces the original age embedding from the Hi-BEHRT paper

Usage:
    # Train from scratch
    python train_hi_behrt.py --task mortality

    # Fine-tune from BYOL pre-trained weights
    python train_hi_behrt.py --task mortality --pretrained_path checkpoints/hi-behrt-byol/best_model.pt
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
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

# Reuse HAT's FinetuneDataset
from src.finetune.data_utils import (
    FinetuneDataset, NextVisitDataset, DownstreamTask, TASK_CONFIGS, collate_finetune
)
from src.baselines.hi_behrt import HiBEHRTConfig, HiBEHRTSimpleForClassification


def flatten_batch_for_hibehrt(batch):
    """
    Convert HAT's hierarchical batch format to flat format for Hi-BEHRT.

    HAT format:
        input_ids: (B, max_seg, max_seq_len)
        attention_mask: (B, max_seg, max_seq_len)
        segment_time: (B, max_seg) - optional, cumsum time per segment

    Hi-BEHRT format:
        input_ids: (B, max_seg * max_seq_len)
        attention_mask: (B, max_seg * max_seq_len)
        time_values: (B, max_seg * max_seq_len) - cumsum time expanded to tokens
    """
    B, S, L = batch["input_ids"].shape

    # Flatten input_ids and attention_mask
    input_ids = batch["input_ids"].view(B, S * L)
    attention_mask = batch["attention_mask"].view(B, S * L)

    # Expand segment_time to token level if available
    time_values = None
    if "segment_time" in batch and batch["segment_time"] is not None:
        # segment_time: (B, S) -> expand to (B, S, L) -> flatten to (B, S*L)
        segment_time = batch["segment_time"]  # (B, S)
        time_values = segment_time.unsqueeze(-1).expand(B, S, L).reshape(B, S * L)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "time_values": time_values,
        "label": batch["label"],
    }


def compute_recall_at_k(preds: np.ndarray, targets: np.ndarray, k: int) -> float:
    """Compute Recall@k for set prediction task."""
    recalls = []
    for pred, target in zip(preds, targets):
        top_k_indices = np.argsort(pred)[-k:]
        true_indices = np.where(target > 0)[0]
        if len(true_indices) == 0:
            continue
        hits = len(set(top_k_indices) & set(true_indices))
        recalls.append(hits / len(true_indices))
    return np.mean(recalls) if recalls else 0.0


def compute_ndcg_at_k(preds: np.ndarray, targets: np.ndarray, k: int) -> float:
    """Compute NDCG@k for set prediction task."""
    ndcgs = []
    for pred, target in zip(preds, targets):
        top_k_indices = np.argsort(pred)[-k:][::-1]
        dcg = 0.0
        for i, idx in enumerate(top_k_indices):
            if target[idx] > 0:
                dcg += 1.0 / np.log2(i + 2)
        num_relevant = int(target.sum())
        if num_relevant == 0:
            continue
        ideal_dcg = sum(1.0 / np.log2(i + 2) for i in range(min(k, num_relevant)))
        ndcgs.append(dcg / ideal_dcg if ideal_dcg > 0 else 0.0)
    return np.mean(ndcgs) if ndcgs else 0.0


def create_hi_behrt_model(num_classes, args, pretrained_path=None):
    """
    Create Hi-BEHRT model for downstream tasks.

    Args:
        num_classes: Number of output classes
        args: Training arguments
        pretrained_path: Path to BYOL pretrained checkpoint (optional)
    """
    config = HiBEHRTConfig(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_extractor_layers=args.n_extractor_layers,
        n_aggregator_layers=args.n_aggregator_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        max_seq_len=args.window_size,  # Local window size
        max_segments=args.max_segments,
        dropout=args.dropout,
        attention_dropout=args.attention_dropout,
        t2v_dim=args.t2v_dim,  # Time2Vec dimension
    )

    model = HiBEHRTSimpleForClassification(
        config=config,
        num_classes=num_classes,
        window_size=args.window_size,
        stride=args.stride,
        dropout=args.classifier_dropout,
    )

    # Load BYOL pretrained weights if provided
    if pretrained_path is not None:
        print(f"\nLoading BYOL pretrained weights from {pretrained_path}")
        model.load_byol_pretrained(pretrained_path)

    return model


class HiBEHRTTrainer:
    """Trainer for Hi-BEHRT end-to-end training."""

    def __init__(
        self,
        model,
        train_dataset,
        val_dataset,
        task: DownstreamTask,
        output_dir: str,
        learning_rate: float = 5e-5,
        batch_size: int = 16,
        num_epochs: int = 30,
        warmup_ratio: float = 0.1,
        weight_decay: float = 0.01,
        use_class_weights: bool = True,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        patience: int = 10,
        use_amp: bool = False,
        device: str = "cuda",
        num_workers: int = 4,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
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

        # Data loaders (use HAT's collate_finetune)
        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            collate_fn=collate_finetune, num_workers=num_workers,
            pin_memory=(self.device.type == "cuda")
        )
        self.val_loader = DataLoader(
            val_dataset, batch_size=batch_size * 2, shuffle=False,
            collate_fn=collate_finetune, num_workers=num_workers,
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

        # Scheduler: warmup + cosine annealing
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
        """Forward pass with flattened batch."""
        # Flatten HAT's hierarchical format to flat format
        flat_batch = flatten_batch_for_hibehrt(batch)

        input_ids = flat_batch["input_ids"].to(self.device)
        attention_mask = flat_batch["attention_mask"].to(self.device)
        labels = flat_batch["label"].to(self.device)

        # Pass time_values if available (for Time2Vec encoding)
        time_values = flat_batch.get("time_values")
        if time_values is not None:
            time_values = time_values.to(self.device)

        output = self.model(input_ids, attention_mask, time_values=time_values, labels=labels)

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
            all_probs = np.vstack(all_probs)
            all_labels = np.vstack(all_labels)
            metrics = {}
            for k in [10, 20, 30]:
                metrics[f"recall@{k}"] = compute_recall_at_k(all_probs, all_labels, k)
                metrics[f"ndcg@{k}"] = compute_ndcg_at_k(all_probs, all_labels, k)
            return metrics

        elif self.is_multilabel:
            all_probs = np.vstack(all_probs)
            all_labels = np.vstack(all_labels)
            all_preds = np.vstack(all_preds)

            metrics = {}
            try:
                aurocs = []
                for c in range(all_labels.shape[1]):
                    if all_labels[:, c].sum() > 0 and all_labels[:, c].sum() < len(all_labels):
                        aurocs.append(roc_auc_score(all_labels[:, c], all_probs[:, c]))
                metrics["auroc"] = np.mean(aurocs) if aurocs else 0.0

                auprcs = []
                for c in range(all_labels.shape[1]):
                    if all_labels[:, c].sum() > 0:
                        auprcs.append(average_precision_score(all_labels[:, c], all_probs[:, c]))
                metrics["auprc"] = np.mean(auprcs) if auprcs else 0.0
            except ValueError as e:
                print(f"  Warning: Could not compute metrics: {e}")
                metrics["auroc"] = 0.0
                metrics["auprc"] = 0.0

            metrics["f1_macro"] = f1_score(all_labels, all_preds, average="macro", zero_division=0)
            metrics["f1_micro"] = f1_score(all_labels, all_preds, average="micro", zero_division=0)
            return metrics

        else:
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
        print(f"Hi-BEHRT End-to-End Training on {self.task.value}")
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
    parser = argparse.ArgumentParser(description="Hi-BEHRT End-to-End Training")

    parser.add_argument("--task", type=str, required=True,
                       choices=["mortality", "readmission_30d", "prolonged_los",
                                "icd_chapter", "icd_category_multilabel", "next_visit"])

    parser.add_argument("--data_path", type=str,
                       default="dataset/mimic4/data/mimic4_tokens.parquet")
    parser.add_argument("--labels_path", type=str,
                       default="dataset/mimic4/data/downstream_labels.csv")
    parser.add_argument("--tokenizer_path", type=str, default="tokenizer.json")
    parser.add_argument("--output_dir", type=str, default="checkpoints/hi-behrt")
    parser.add_argument("--pretrained_path", type=str, default=None,
                       help="Path to BYOL pretrained checkpoint (optional)")

    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=30)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)

    # Model architecture (unified with other baselines)
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--n_extractor_layers", type=int, default=6)
    parser.add_argument("--n_aggregator_layers", type=int, default=6)
    parser.add_argument("--n_heads", type=int, default=12)
    parser.add_argument("--d_ff", type=int, default=2048)

    # Hi-BEHRT specific
    parser.add_argument("--window_size", type=int, default=50)
    parser.add_argument("--stride", type=int, default=30)
    parser.add_argument("--max_segments", type=int, default=40)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--attention_dropout", type=float, default=0.1)
    parser.add_argument("--classifier_dropout", type=float, default=0.1)
    parser.add_argument("--t2v_dim", type=int, default=64,
                       help="Time2Vec output dimension")

    # Dataset config (matching HAT's FinetuneDataset)
    parser.add_argument("--max_seg", type=int, default=8)
    parser.add_argument("--max_seq_len", type=int, default=512)

    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)

    return parser.parse_args()


def main():
    args = parse_args()

    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    task = DownstreamTask(args.task)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"{args.task}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"  Hi-BEHRT End-to-End Training on {args.task}")
    print("=" * 60)

    # Load or train tokenizer
    tokenizer_path = Path(args.tokenizer_path)
    if tokenizer_path.exists():
        print(f"Loading tokenizer from {args.tokenizer_path}")
        tokenizer = Tokenizer.from_file(args.tokenizer_path)
    else:
        print(f"Tokenizer not found at {args.tokenizer_path}, training new tokenizer...")
        from src.tokenizer import get_tokenizer
        import pyarrow.parquet as pq
        from concurrent.futures import ThreadPoolExecutor, as_completed

        # Use ALL patients for tokenizer training
        data_path = Path(args.data_path)
        patient_dirs = sorted(data_path.glob("subject_id=*"))
        print(f"Loading data from {len(patient_dirs)} patients for tokenizer training...")

        def read_patient(d):
            try:
                return pq.read_table(d).to_pandas()
            except Exception:
                return None

        dfs = []
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(read_patient, d) for d in patient_dirs]
            for f in tqdm(as_completed(futures), total=len(futures), desc="Loading tokenizer data"):
                result = f.result()
                if result is not None:
                    dfs.append(result)

        df_sample = pd.concat(dfs, ignore_index=True)
        tokenizer = get_tokenizer([df_sample], {
            "tokenizer_path": str(output_dir / "tokenizer.json"),
            "patient_id_col": "subject_id",
            "token_col": "code",
            "min_frequency": 5,
        })
        print(f"Tokenizer trained and saved to {output_dir / 'tokenizer.json'}")

    args.vocab_size = tokenizer.get_vocab_size()
    print(f"Vocab size: {args.vocab_size}")

    # Load labels
    print(f"Loading labels from {args.labels_path}")
    labels_df = pd.read_csv(args.labels_path)

    # Create datasets
    task_config = TASK_CONFIGS[task]
    is_set_prediction = task_config.get("is_set_prediction", False)

    if is_set_prediction:
        # Use NextVisitDataset for NEXT_VISIT task
        print("\nCreating datasets (using NextVisitDataset)...")
        train_dataset = NextVisitDataset(
            data_path=args.data_path,
            labels_df=labels_df,
            tokenizer=tokenizer,
            max_seg=args.max_seg,
            max_seq_len=args.max_seq_len,
            split="train",
            seed=args.seed,
        )
        val_dataset = NextVisitDataset(
            data_path=args.data_path,
            labels_df=labels_df,
            tokenizer=tokenizer,
            max_seg=args.max_seg,
            max_seq_len=args.max_seq_len,
            split="val",
            seed=args.seed,
        )
        test_dataset = NextVisitDataset(
            data_path=args.data_path,
            labels_df=labels_df,
            tokenizer=tokenizer,
            max_seg=args.max_seg,
            max_seq_len=args.max_seq_len,
            split="test",
            seed=args.seed,
        )
    else:
        # Use FinetuneDataset for classification tasks
        print("\nCreating datasets (using FinetuneDataset)...")
        train_dataset = FinetuneDataset(
            data_path=args.data_path,
            labels_df=labels_df,
            tokenizer=tokenizer,
            task=task,
            max_seg=args.max_seg,
            max_seq_len=args.max_seq_len,
            split="train",
            seed=args.seed,
        )
        val_dataset = FinetuneDataset(
            data_path=args.data_path,
            labels_df=labels_df,
            tokenizer=tokenizer,
            task=task,
            max_seg=args.max_seg,
            max_seq_len=args.max_seq_len,
            split="val",
            seed=args.seed,
        )
        test_dataset = FinetuneDataset(
            data_path=args.data_path,
            labels_df=labels_df,
            tokenizer=tokenizer,
            task=task,
            max_seg=args.max_seg,
            max_seq_len=args.max_seq_len,
            split="test",
            seed=args.seed,
        )

    num_classes = train_dataset.num_classes if hasattr(train_dataset, 'num_classes') else train_dataset.vocab_size
    print(f"  Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    print(f"  Num classes: {num_classes}")

    # Create model
    if args.pretrained_path:
        print(f"\nCreating Hi-BEHRT model (with BYOL pretrained weights)...")
    else:
        print("\nCreating Hi-BEHRT model (training from scratch)...")
    model = create_hi_behrt_model(num_classes, args, pretrained_path=args.pretrained_path)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {n_params:,}")
    print(f"  d_model: {args.d_model}")
    print(f"  n_extractor_layers: {args.n_extractor_layers}")
    print(f"  n_aggregator_layers: {args.n_aggregator_layers}")
    print(f"  window_size: {args.window_size}, stride: {args.stride}")
    print(f"  t2v_dim: {args.t2v_dim}")

    # Save config
    with open(output_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # Create trainer
    trainer = HiBEHRTTrainer(
        model=model,
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

    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size * 2, shuffle=False,
        collate_fn=collate_finetune, num_workers=args.num_workers
    )
    test_metrics = trainer.evaluate(test_loader)

    print("\nTest Results:")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")

    # Save results
    results = {
        "model": "hi-behrt",
        "task": args.task,
        "best_val_metric": best_metric,
        "test_metrics": test_metrics,
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
