"""
Fine-tuning Trainer for HAT downstream tasks.

Supports:
- AUROC and AUPRC metrics (important for imbalanced data)
- Class weighting for imbalanced classes
- Early stopping
- Learning rate scheduling
- Mixed precision training
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast  # PyTorch 2.0+ style
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import json
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    f1_score,
    classification_report,
)
from typing import Optional, Dict, Any
import wandb

from src.finetune.model import HATForSequenceClassification
from src.finetune.data_utils import FinetuneDataset, collate_finetune, DownstreamTask


class FinetuneTrainer:
    """Trainer for HAT fine-tuning on downstream tasks."""

    def __init__(
        self,
        model: HATForSequenceClassification,
        train_dataset: FinetuneDataset,
        val_dataset: FinetuneDataset,
        task: DownstreamTask,
        output_dir: str,
        # Training hyperparameters
        learning_rate: float = 2e-5,
        batch_size: int = 32,
        num_epochs: int = 10,
        warmup_ratio: float = 0.1,
        weight_decay: float = 0.01,
        # Training options
        use_class_weights: bool = True,
        use_amp: bool = True,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        # Early stopping
        patience: int = 3,
        metric_for_best_model: str = "auprc",  # "auroc" or "auprc"
        # Logging
        log_interval: int = 100,
        eval_interval: int = 500,
        use_wandb: bool = False,
        wandb_project: str = "hat-finetune",
        wandb_run_name: Optional[str] = None,
        # Hardware
        device: str = "cuda",
        num_workers: int = 4,
    ):
        self.model = model.to(device)
        self.device = device
        self.task = task
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Hyperparameters
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.warmup_ratio = warmup_ratio
        self.weight_decay = weight_decay
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm

        # Data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_finetune,
            num_workers=num_workers,
            pin_memory=True,
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size * 2,  # Larger batch for eval
            shuffle=False,
            collate_fn=collate_finetune,
            num_workers=num_workers,
            pin_memory=True,
        )

        # Class weights for imbalanced data
        self.class_weights = None
        if use_class_weights:
            weights = train_dataset.get_class_weights()
            self.class_weights = weights.to(device)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        # Learning rate scheduler
        total_steps = len(self.train_loader) * num_epochs // gradient_accumulation_steps
        warmup_steps = int(total_steps * warmup_ratio)

        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        main_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps - warmup_steps,
            eta_min=learning_rate * 0.01,
        )
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_steps],
        )

        # Mixed precision
        self.use_amp = use_amp
        self.scaler = GradScaler() if use_amp else None

        # Early stopping
        self.patience = patience
        self.metric_for_best_model = metric_for_best_model
        self.best_metric = 0.0
        self.patience_counter = 0

        # Logging
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.use_wandb = use_wandb
        self.global_step = 0

        # Initialize wandb
        if use_wandb:
            wandb.init(
                project=wandb_project,
                name=wandb_run_name or f"{task.value}",
                config={
                    "task": task.value,
                    "learning_rate": learning_rate,
                    "batch_size": batch_size,
                    "num_epochs": num_epochs,
                    "warmup_ratio": warmup_ratio,
                    "weight_decay": weight_decay,
                    "num_classes": model.num_classes,
                    "pooling": model.classifier.pooling,
                },
            )

        # Track metrics
        self.train_losses = []
        self.val_metrics_history = []

    def train(self):
        """Run the full training loop."""
        print(f"\n{'='*60}")
        print(f"Starting fine-tuning for task: {self.task.value}")
        print(f"{'='*60}")
        print(f"  Train samples: {len(self.train_loader.dataset):,}")
        print(f"  Val samples:   {len(self.val_loader.dataset):,}")
        print(f"  Num classes:   {self.model.num_classes}")
        print(f"  Batch size:    {self.batch_size}")
        print(f"  Num epochs:    {self.num_epochs}")
        print(f"  Learning rate: {self.learning_rate}")
        print(f"{'='*60}\n")

        for epoch in range(self.num_epochs):
            # Train one epoch
            train_loss = self._train_epoch(epoch)

            # Evaluate
            val_metrics = self.evaluate()

            # Log epoch results
            print(f"\nEpoch {epoch + 1}/{self.num_epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val AUROC:  {val_metrics['auroc']:.4f}")
            print(f"  Val AUPRC:  {val_metrics['auprc']:.4f}")
            print(f"  Val Acc:    {val_metrics['accuracy']:.4f}")

            if self.use_wandb:
                wandb.log({
                    "epoch": epoch + 1,
                    "train/loss_epoch": train_loss,
                    **{f"val/{k}": v for k, v in val_metrics.items()},
                })

            # Early stopping check
            current_metric = val_metrics[self.metric_for_best_model]
            if current_metric > self.best_metric:
                self.best_metric = current_metric
                self.patience_counter = 0
                self._save_checkpoint("best")
                print(f"  New best {self.metric_for_best_model}: {self.best_metric:.4f}")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                    break

            # Save checkpoint
            self._save_checkpoint(f"epoch_{epoch + 1}")

        # Save final checkpoint
        self._save_checkpoint("final")

        # Load best model for final evaluation
        self._load_checkpoint("best")

        print(f"\nTraining complete! Best {self.metric_for_best_model}: {self.best_metric:.4f}")

        return self.best_metric

    def _train_epoch(self, epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}")

        for batch_idx, batch in enumerate(pbar):
            # Move to device
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            segment_attention_mask = batch["segment_attention_mask"].to(self.device)
            labels = batch["label"].to(self.device)

            segment_time = batch.get("segment_time")
            if segment_time is not None:
                segment_time = segment_time.to(self.device)

            token_time = batch.get("token_time")
            if token_time is not None:
                token_time = token_time.to(self.device)

            # Forward pass
            with autocast(device_type='cuda', enabled=self.use_amp):
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    segment_attention_mask=segment_attention_mask,
                    segment_time=segment_time,
                    token_time=token_time,
                    labels=labels,
                )
                loss = outputs["loss"]

                # Apply class weights if available
                if self.class_weights is not None:
                    # Recompute loss with class weights
                    logits = outputs["logits"]
                    loss = nn.functional.cross_entropy(
                        logits, labels, weight=self.class_weights
                    )

                loss = loss / self.gradient_accumulation_steps

            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Gradient accumulation
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.max_grad_norm
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.max_grad_norm
                    )
                    self.optimizer.step()

                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1

            total_loss += loss.item() * self.gradient_accumulation_steps
            num_batches += 1

            # Update progress bar
            pbar.set_postfix({
                "loss": f"{loss.item() * self.gradient_accumulation_steps:.4f}",
                "lr": f"{self.scheduler.get_last_lr()[0]:.2e}",
            })

            # Log to wandb
            if self.use_wandb and self.global_step % self.log_interval == 0:
                wandb.log({
                    "train/loss": loss.item() * self.gradient_accumulation_steps,
                    "train/learning_rate": self.scheduler.get_last_lr()[0],
                    "global_step": self.global_step,
                })

        # Handle remaining accumulated gradients if batch count is not divisible
        # by gradient_accumulation_steps (important fix for incomplete final batch)
        if num_batches % self.gradient_accumulation_steps != 0:
            if self.use_amp:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )
                self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
            self.global_step += 1

        return total_loss / num_batches

    @torch.no_grad()
    def evaluate(self, loader: Optional[DataLoader] = None) -> Dict[str, float]:
        """Evaluate the model."""
        if loader is None:
            loader = self.val_loader

        self.model.eval()

        all_preds = []
        all_labels = []
        all_probs = []

        for batch in tqdm(loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            segment_attention_mask = batch["segment_attention_mask"].to(self.device)
            labels = batch["label"]

            segment_time = batch.get("segment_time")
            if segment_time is not None:
                segment_time = segment_time.to(self.device)

            token_time = batch.get("token_time")
            if token_time is not None:
                token_time = token_time.to(self.device)

            with autocast(device_type='cuda', enabled=self.use_amp):
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    segment_attention_mask=segment_attention_mask,
                    segment_time=segment_time,
                    token_time=token_time,
                )

            logits = outputs["logits"].cpu()
            probs = torch.softmax(logits, dim=-1)

            if self.model.num_classes == 2:
                preds = (probs[:, 1] > 0.5).long()
                all_probs.extend(probs[:, 1].numpy())
            else:
                preds = logits.argmax(dim=-1)
                all_probs.append(probs.numpy())

            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        # Compute metrics
        metrics = {
            "accuracy": accuracy_score(all_labels, all_preds),
        }

        if self.model.num_classes == 2:
            # Binary classification metrics
            all_probs = np.array(all_probs)
            metrics["auroc"] = roc_auc_score(all_labels, all_probs)
            metrics["auprc"] = average_precision_score(all_labels, all_probs)
            metrics["f1"] = f1_score(all_labels, all_preds)

            # Compute positive rate for calibration check
            metrics["pred_positive_rate"] = all_preds.mean()
            metrics["true_positive_rate"] = all_labels.mean()
        else:
            # Multi-class metrics
            all_probs = np.vstack(all_probs)
            try:
                metrics["auroc"] = roc_auc_score(
                    all_labels, all_probs, multi_class="ovr", average="macro"
                )
            except ValueError:
                metrics["auroc"] = 0.0

            try:
                # For AUPRC in multi-class, compute weighted average by class frequency
                # This is more stable than macro average when some classes have few samples
                auprc_scores = []
                class_weights = []
                for i in range(self.model.num_classes):
                    binary_labels = (all_labels == i).astype(int)
                    class_count = binary_labels.sum()
                    if class_count > 0:
                        auprc_scores.append(
                            average_precision_score(binary_labels, all_probs[:, i])
                        )
                        class_weights.append(class_count)
                if auprc_scores:
                    # Weighted average AUPRC
                    total_weight = sum(class_weights)
                    metrics["auprc"] = sum(
                        s * w for s, w in zip(auprc_scores, class_weights)
                    ) / total_weight
                    # Also report macro for comparison
                    metrics["auprc_macro"] = np.mean(auprc_scores)
                else:
                    metrics["auprc"] = 0.0
                    metrics["auprc_macro"] = 0.0
            except ValueError:
                metrics["auprc"] = 0.0
                metrics["auprc_macro"] = 0.0

            metrics["f1_macro"] = f1_score(all_labels, all_preds, average="macro")
            metrics["f1_weighted"] = f1_score(all_labels, all_preds, average="weighted")

        self.val_metrics_history.append(metrics)

        return metrics

    def _save_checkpoint(self, name: str):
        """Save a checkpoint."""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "best_metric": self.best_metric,
            "task": self.task.value,
            "num_classes": self.model.num_classes,
        }

        path = self.output_dir / f"checkpoint_{name}.pt"
        torch.save(checkpoint, path)

    def _load_checkpoint(self, name: str):
        """Load a checkpoint."""
        path = self.output_dir / f"checkpoint_{name}.pt"
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])


def run_finetune(
    task: str,
    pretrained_path: str,
    data_path: str,
    labels_path: str,
    output_dir: str,
    tokenizer_path: str,
    # Model args
    pooling: str = "last_cls",
    dropout: float = 0.1,
    freeze_encoder: bool = False,
    # Training args
    learning_rate: float = 2e-5,
    batch_size: int = 32,
    num_epochs: int = 10,
    warmup_ratio: float = 0.1,
    # Options
    use_wandb: bool = False,
    seed: int = 42,
):
    """
    Main function to run fine-tuning.

    Args:
        task: Task name (mortality, readmission_30d, prolonged_los, icd_chapter)
        pretrained_path: Path to pre-trained checkpoint
        data_path: Path to parquet data directory
        labels_path: Path to labels CSV file
        output_dir: Output directory for checkpoints and logs
        tokenizer_path: Path to tokenizer file
        ... (other args)
    """
    from tokenizers import Tokenizer

    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load tokenizer
    tokenizer = Tokenizer.from_file(tokenizer_path)

    # Load labels
    labels_df = pd.read_csv(labels_path)

    # Create task enum
    task_enum = DownstreamTask(task)

    # Create datasets
    print(f"Creating datasets for task: {task}")
    train_dataset = FinetuneDataset(
        data_path=data_path,
        labels_df=labels_df,
        tokenizer=tokenizer,
        task=task_enum,
        split="train",
        seed=seed,
    )
    val_dataset = FinetuneDataset(
        data_path=data_path,
        labels_df=labels_df,
        tokenizer=tokenizer,
        task=task_enum,
        split="val",
        seed=seed,
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Num classes: {train_dataset.num_classes}")

    # Create model
    from src.finetune.model import create_finetune_model

    model = create_finetune_model(
        pretrained_path=pretrained_path,
        num_classes=train_dataset.num_classes,
        dropout=dropout,
        pooling=pooling,
        freeze_encoder=freeze_encoder,
    )

    # Create trainer
    trainer = FinetuneTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        task=task_enum,
        output_dir=output_dir,
        learning_rate=learning_rate,
        batch_size=batch_size,
        num_epochs=num_epochs,
        warmup_ratio=warmup_ratio,
        use_wandb=use_wandb,
    )

    # Train
    best_metric = trainer.train()

    # Final evaluation on test set
    print("\nEvaluating on test set...")
    test_dataset = FinetuneDataset(
        data_path=data_path,
        labels_df=labels_df,
        tokenizer=tokenizer,
        task=task_enum,
        split="test",
        seed=seed,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        collate_fn=collate_finetune,
        num_workers=4,
    )

    test_metrics = trainer.evaluate(test_loader)

    print(f"\nTest Results:")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")

    # Save test results
    results = {
        "task": task,
        "best_val_metric": best_metric,
        "test_metrics": test_metrics,
    }

    with open(Path(output_dir) / "test_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return test_metrics
