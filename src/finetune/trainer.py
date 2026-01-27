"""Fine-tuning Trainer for HAT downstream tasks."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
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
)
from typing import Optional, Dict
import wandb

from src.finetune.model import HATForSequenceClassification, HATForNextVisit
from src.finetune.data_utils import FinetuneDataset, NextVisitDataset, collate_finetune, DownstreamTask, TASK_CONFIGS


class FinetuneTrainer:
    """Trainer for HAT fine-tuning on downstream tasks."""

    def __init__(
        self,
        model: HATForSequenceClassification,
        train_dataset: FinetuneDataset,
        val_dataset: FinetuneDataset,
        task: DownstreamTask,
        output_dir: str,
        learning_rate: float = 2e-5,
        batch_size: int = 32,
        num_epochs: int = 10,
        warmup_ratio: float = 0.1,
        weight_decay: float = 0.01,
        use_class_weights: bool = True,
        use_amp: bool = True,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        patience: int = 3,
        metric_for_best_model: str = "auprc",
        log_interval: int = 100,
        eval_interval: int = 500,
        use_wandb: bool = False,
        wandb_project: str = "hat-finetune",
        wandb_run_name: Optional[str] = None,
        device: str = "auto",
        num_workers: int = 4,
    ):
        # Auto-detect device
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
            print(f"Using device: {device}")

        self.model = model.to(device)
        self.device = device
        self.device_type = device.split(':')[0]
        self.task = task
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.warmup_ratio = warmup_ratio
        self.weight_decay = weight_decay
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm

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
            batch_size=batch_size * 2,
            shuffle=False,
            collate_fn=collate_finetune,
            num_workers=num_workers,
            pin_memory=True,
        )

        self.is_multilabel = TASK_CONFIGS[task].get("is_multilabel", False)

        self.class_weights = None
        if use_class_weights:
            weights = train_dataset.get_class_weights()
            self.class_weights = weights.to(device)

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

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

        # GradScaler only works with CUDA
        self.use_amp = use_amp and self.device_type == "cuda"
        self.scaler = GradScaler() if self.use_amp else None
        if use_amp and not self.use_amp:
            print(f"Warning: AMP disabled on {self.device_type} (GradScaler requires CUDA)")

        self.patience = patience
        self.metric_for_best_model = metric_for_best_model
        self.best_metric = 0.0
        self.patience_counter = 0

        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.use_wandb = use_wandb
        self.global_step = 0

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
            train_loss = self._train_epoch(epoch)
            val_metrics = self.evaluate()

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

            # Save periodic checkpoint and cleanup old ones (keep last 3)
            self._save_checkpoint(f"epoch_{epoch + 1:04d}")
            self._cleanup_old_checkpoints()

        self._save_checkpoint("final")
        self._load_checkpoint("best")

        print(f"\nTraining complete! Best {self.metric_for_best_model}: {self.best_metric:.4f}")
        return self.best_metric

    def _cleanup_old_checkpoints(self):
        """Keep only the last 3 epoch checkpoints."""
        checkpoints = sorted(self.output_dir.glob("checkpoint_epoch_*.pt"))
        if len(checkpoints) > 3:
            for old_ckpt in checkpoints[:-3]:
                old_ckpt.unlink()

    def _train_epoch(self, epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}")

        for batch_idx, batch in enumerate(pbar):
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

            with autocast(device_type=self.device_type, enabled=self.use_amp):
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    segment_attention_mask=segment_attention_mask,
                    segment_time=segment_time,
                    token_time=token_time,
                    labels=None,
                )

                logits = outputs["logits"]

                if self.is_multilabel:
                    if self.class_weights is not None:
                        loss = nn.functional.binary_cross_entropy_with_logits(
                            logits, labels, pos_weight=self.class_weights
                        )
                    else:
                        loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
                else:
                    if self.class_weights is not None:
                        loss = nn.functional.cross_entropy(logits, labels, weight=self.class_weights)
                    else:
                        loss = nn.functional.cross_entropy(logits, labels)

                loss = loss / self.gradient_accumulation_steps

            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()

                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1

            total_loss += loss.item() * self.gradient_accumulation_steps
            num_batches += 1

            pbar.set_postfix({
                "loss": f"{loss.item() * self.gradient_accumulation_steps:.4f}",
                "lr": f"{self.scheduler.get_last_lr()[0]:.2e}",
            })

            if self.use_wandb and self.global_step % self.log_interval == 0:
                wandb.log({
                    "train/loss": loss.item() * self.gradient_accumulation_steps,
                    "train/learning_rate": self.scheduler.get_last_lr()[0],
                    "global_step": self.global_step,
                })

        # Handle remaining accumulated gradients
        if num_batches % self.gradient_accumulation_steps != 0:
            if self.use_amp:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
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

            with autocast(device_type=self.device_type, enabled=self.use_amp):
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    segment_attention_mask=segment_attention_mask,
                    segment_time=segment_time,
                    token_time=token_time,
                )

            logits = outputs["logits"].cpu()

            if self.is_multilabel:
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()
                all_probs.append(probs.numpy())
                all_preds.append(preds.numpy())
                all_labels.append(labels.numpy())
            else:
                probs = torch.softmax(logits, dim=-1)
                if self.model.num_classes == 2:
                    preds = (probs[:, 1] > 0.5).long()
                    all_probs.extend(probs[:, 1].numpy())
                else:
                    preds = logits.argmax(dim=-1)
                    all_probs.append(probs.numpy())
                all_preds.extend(preds.numpy())
                all_labels.extend(labels.numpy())

        if self.is_multilabel:
            all_probs = np.vstack(all_probs)
            all_preds = np.vstack(all_preds)
            all_labels = np.vstack(all_labels)

            auroc_scores = []
            auprc_scores = []
            for i in range(self.model.num_classes):
                class_labels = all_labels[:, i]
                class_probs = all_probs[:, i]
                if class_labels.sum() > 0 and class_labels.sum() < len(class_labels):
                    auroc_scores.append(roc_auc_score(class_labels, class_probs))
                    auprc_scores.append(average_precision_score(class_labels, class_probs))

            if auroc_scores:
                metrics = {
                    "auroc": np.mean(auroc_scores),
                    "auprc": np.mean(auprc_scores),
                    "auroc_micro": roc_auc_score(all_labels.ravel(), all_probs.ravel()),
                }
            else:
                metrics = {"auroc": 0.0, "auprc": 0.0, "auroc_micro": 0.0}

            metrics["accuracy"] = (all_preds == all_labels).mean()
            metrics["f1_micro"] = f1_score(all_labels, all_preds, average="micro")
            metrics["f1_macro"] = f1_score(all_labels, all_preds, average="macro")
            metrics["avg_labels_per_sample"] = all_labels.sum(axis=1).mean()
        else:
            all_preds = np.array(all_preds)
            all_labels = np.array(all_labels)

            metrics = {"accuracy": accuracy_score(all_labels, all_preds)}

            if self.model.num_classes == 2:
                all_probs = np.array(all_probs)
                metrics["auroc"] = roc_auc_score(all_labels, all_probs)
                metrics["auprc"] = average_precision_score(all_labels, all_probs)
                metrics["f1"] = f1_score(all_labels, all_preds)
                metrics["pred_positive_rate"] = all_preds.mean()
                metrics["true_positive_rate"] = all_labels.mean()
            else:
                all_probs = np.vstack(all_probs)
                auroc_scores = []
                auprc_scores = []
                class_weights = []
                for i in range(self.model.num_classes):
                    binary_labels = (all_labels == i).astype(int)
                    class_count = binary_labels.sum()
                    if class_count > 0 and class_count < len(all_labels):
                        auroc_scores.append(roc_auc_score(binary_labels, all_probs[:, i]))
                        auprc_scores.append(average_precision_score(binary_labels, all_probs[:, i]))
                        class_weights.append(class_count)

                if auroc_scores:
                    total_weight = sum(class_weights)
                    metrics["auroc"] = np.mean(auroc_scores)  # macro average
                    metrics["auprc"] = sum(s * w for s, w in zip(auprc_scores, class_weights)) / total_weight
                    metrics["auprc_macro"] = np.mean(auprc_scores)
                else:
                    metrics["auroc"] = 0.0
                    metrics["auprc"] = 0.0
                    metrics["auprc_macro"] = 0.0

                metrics["f1_macro"] = f1_score(all_labels, all_preds, average="macro")
                metrics["f1_weighted"] = f1_score(all_labels, all_preds, average="weighted")

        self.val_metrics_history.append(metrics)
        return metrics

    def _save_checkpoint(self, name: str):
        """Save a checkpoint."""
        config_dict = {
            "vocab_size": self.model.config.vocab_size,
            "d_model": self.model.config.d_model,
            "d_ff": self.model.config.d_ff,
            "n_blocks": self.model.config.n_blocks,
            "n_heads": self.model.config.n_heads,
        }

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "best_metric": self.best_metric,
            "task": self.task.value,
            "num_classes": self.model.num_classes,
            "config": config_dict,
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
    pooling: str = "last_cls",
    dropout: float = 0.1,
    freeze_encoder: bool = False,
    learning_rate: float = 2e-5,
    batch_size: int = 32,
    num_epochs: int = 10,
    warmup_ratio: float = 0.1,
    use_wandb: bool = False,
    seed: int = 42,
):
    """Main function to run fine-tuning."""
    from tokenizers import Tokenizer

    torch.manual_seed(seed)
    np.random.seed(seed)

    tokenizer = Tokenizer.from_file(tokenizer_path)
    labels_df = pd.read_csv(labels_path)
    task_enum = DownstreamTask(task)

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

    from src.finetune.model import create_finetune_model

    model = create_finetune_model(
        pretrained_path=pretrained_path,
        num_classes=train_dataset.num_classes,
        dropout=dropout,
        pooling=pooling,
        freeze_encoder=freeze_encoder,
    )

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

    best_metric = trainer.train()

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

    results = {
        "task": task,
        "best_val_metric": best_metric,
        "test_metrics": test_metrics,
    }

    with open(Path(output_dir) / "test_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return test_metrics


# ============== Next Visit Prediction ==============

def recall_at_k(pred_scores: torch.Tensor, target_ids: list, k: int = 10) -> float:
    """Calculate Recall@K for a single sample.

    Args:
        pred_scores: (vocab_size,) tensor of prediction scores
        target_ids: list of true token IDs
        k: number of top predictions to consider
    """
    if len(target_ids) == 0:
        return 0.0

    top_k_preds = set(pred_scores.topk(k).indices.tolist())
    hits = len(top_k_preds & set(target_ids))
    return hits / len(target_ids)


def ndcg_at_k(pred_scores: torch.Tensor, target_ids: list, k: int = 10) -> float:
    """Calculate NDCG@K for a single sample.

    Args:
        pred_scores: (vocab_size,) tensor of prediction scores
        target_ids: list of true token IDs
        k: number of top predictions to consider
    """
    if len(target_ids) == 0:
        return 0.0

    target_set = set(target_ids)
    top_k_indices = pred_scores.topk(k).indices.tolist()

    # DCG: sum of 1/log2(rank+1) for each hit
    dcg = 0.0
    for rank, idx in enumerate(top_k_indices):
        if idx in target_set:
            dcg += 1.0 / np.log2(rank + 2)  # rank+2 because rank is 0-indexed

    # Ideal DCG: all targets at top positions
    ideal_num = min(len(target_ids), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_num))

    if idcg == 0:
        return 0.0
    return dcg / idcg


class NextVisitTrainer:
    """Trainer for Next Visit Prediction task."""

    def __init__(
        self,
        model: HATForNextVisit,
        train_dataset: NextVisitDataset,
        val_dataset: NextVisitDataset,
        output_dir: str,
        learning_rate: float = 2e-5,
        batch_size: int = 32,
        num_epochs: int = 10,
        warmup_ratio: float = 0.1,
        weight_decay: float = 0.01,
        use_amp: bool = True,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        patience: int = 3,
        k_values: list = [10, 20, 30],
        metric_for_best_model: str = "recall@20",
        log_interval: int = 100,
        use_wandb: bool = False,
        wandb_project: str = "hat-finetune",
        wandb_run_name: Optional[str] = None,
        device: str = "auto",
        num_workers: int = 4,
    ):
        # Auto-detect device
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
            print(f"Using device: {device}")

        self.model = model.to(device)
        self.device = device
        self.device_type = device.split(':')[0]
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.warmup_ratio = warmup_ratio
        self.weight_decay = weight_decay
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.k_values = k_values
        self.metric_for_best_model = metric_for_best_model

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
            batch_size=batch_size * 2,
            shuffle=False,
            collate_fn=collate_finetune,
            num_workers=num_workers,
            pin_memory=True,
        )

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

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

        self.use_amp = use_amp and self.device_type == "cuda"
        self.scaler = GradScaler() if self.use_amp else None

        self.patience = patience
        self.best_metric = 0.0
        self.patience_counter = 0

        self.log_interval = log_interval
        self.use_wandb = use_wandb
        self.global_step = 0

        if use_wandb:
            wandb.init(
                project=wandb_project,
                name=wandb_run_name or "next_visit",
                config={
                    "task": "next_visit",
                    "learning_rate": learning_rate,
                    "batch_size": batch_size,
                    "num_epochs": num_epochs,
                    "vocab_size": model.vocab_size,
                    "pooling": model.predictor.pooling,
                },
            )

    def train(self):
        """Run the full training loop."""
        print(f"\n{'='*60}")
        print("Starting fine-tuning for task: next_visit")
        print(f"{'='*60}")
        print(f"  Train samples: {len(self.train_loader.dataset):,}")
        print(f"  Val samples:   {len(self.val_loader.dataset):,}")
        print(f"  Vocab size:    {self.model.vocab_size}")
        print(f"  Batch size:    {self.batch_size}")
        print(f"  Num epochs:    {self.num_epochs}")
        print(f"{'='*60}\n")

        for epoch in range(self.num_epochs):
            train_loss = self._train_epoch(epoch)
            val_metrics = self.evaluate()

            print(f"\nEpoch {epoch + 1}/{self.num_epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            for k in self.k_values:
                print(f"  Recall@{k}: {val_metrics[f'recall@{k}']:.4f}  NDCG@{k}: {val_metrics[f'ndcg@{k}']:.4f}")

            if self.use_wandb:
                wandb.log({
                    "epoch": epoch + 1,
                    "train/loss_epoch": train_loss,
                    **{f"val/{k}": v for k, v in val_metrics.items()},
                })

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

            # Save periodic checkpoint and cleanup old ones (keep last 3)
            self._save_checkpoint(f"epoch_{epoch + 1:04d}")
            self._cleanup_old_checkpoints()

        self._save_checkpoint("final")
        self._load_checkpoint("best")

        print(f"\nTraining complete! Best {self.metric_for_best_model}: {self.best_metric:.4f}")
        return self.best_metric

    def _cleanup_old_checkpoints(self):
        """Keep only the last 3 epoch checkpoints."""
        checkpoints = sorted(self.output_dir.glob("checkpoint_epoch_*.pt"))
        if len(checkpoints) > 3:
            for old_ckpt in checkpoints[:-3]:
                old_ckpt.unlink()

    def _train_epoch(self, epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}")

        for batch_idx, batch in enumerate(pbar):
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

            with autocast(device_type=self.device_type, enabled=self.use_amp):
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    segment_attention_mask=segment_attention_mask,
                    segment_time=segment_time,
                    token_time=token_time,
                    labels=labels,
                )

                loss = outputs["loss"] / self.gradient_accumulation_steps

            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()

                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1

            total_loss += loss.item() * self.gradient_accumulation_steps
            num_batches += 1

            pbar.set_postfix({
                "loss": f"{loss.item() * self.gradient_accumulation_steps:.4f}",
                "lr": f"{self.scheduler.get_last_lr()[0]:.2e}",
            })

        return total_loss / num_batches

    @torch.no_grad()
    def evaluate(self, loader: Optional[DataLoader] = None) -> Dict[str, float]:
        """Evaluate the model using Recall@K and NDCG@K."""
        if loader is None:
            loader = self.val_loader

        self.model.eval()

        all_recalls = {k: [] for k in self.k_values}
        all_ndcgs = {k: [] for k in self.k_values}

        for batch in tqdm(loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            segment_attention_mask = batch["segment_attention_mask"].to(self.device)
            target_token_ids_batch = batch["target_token_ids"]

            segment_time = batch.get("segment_time")
            if segment_time is not None:
                segment_time = segment_time.to(self.device)

            token_time = batch.get("token_time")
            if token_time is not None:
                token_time = token_time.to(self.device)

            with autocast(device_type=self.device_type, enabled=self.use_amp):
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    segment_attention_mask=segment_attention_mask,
                    segment_time=segment_time,
                    token_time=token_time,
                )

            logits = outputs["logits"].cpu()

            for i in range(logits.shape[0]):
                target_ids = target_token_ids_batch[i]
                pred_scores = logits[i]

                for k in self.k_values:
                    all_recalls[k].append(recall_at_k(pred_scores, target_ids, k))
                    all_ndcgs[k].append(ndcg_at_k(pred_scores, target_ids, k))

        metrics = {}
        for k in self.k_values:
            metrics[f"recall@{k}"] = np.mean(all_recalls[k])
            metrics[f"ndcg@{k}"] = np.mean(all_ndcgs[k])

        return metrics

    def _save_checkpoint(self, name: str):
        """Save a checkpoint."""
        config_dict = {
            "vocab_size": self.model.config.vocab_size,
            "d_model": self.model.config.d_model,
            "d_ff": self.model.config.d_ff,
            "n_blocks": self.model.config.n_blocks,
            "n_heads": self.model.config.n_heads,
        }

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "best_metric": self.best_metric,
            "task": "next_visit",
            "vocab_size": self.model.vocab_size,
            "config": config_dict,
        }

        path = self.output_dir / f"checkpoint_{name}.pt"
        torch.save(checkpoint, path)

    def _load_checkpoint(self, name: str):
        """Load a checkpoint."""
        path = self.output_dir / f"checkpoint_{name}.pt"
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])


def run_next_visit(
    pretrained_path: str,
    data_path: str,
    labels_path: str,
    output_dir: str,
    tokenizer_path: str,
    pooling: str = "last_cls",
    dropout: float = 0.1,
    freeze_encoder: bool = False,
    learning_rate: float = 2e-5,
    batch_size: int = 32,
    num_epochs: int = 10,
    warmup_ratio: float = 0.1,
    k_values: list = [10, 20, 30],
    use_wandb: bool = False,
    seed: int = 42,
):
    """Main function to run Next Visit Prediction fine-tuning."""
    from tokenizers import Tokenizer
    from src.finetune.model import create_next_visit_model

    torch.manual_seed(seed)
    np.random.seed(seed)

    tokenizer = Tokenizer.from_file(tokenizer_path)
    labels_df = pd.read_csv(labels_path)

    print("Creating datasets for task: next_visit")
    train_dataset = NextVisitDataset(
        data_path=data_path,
        labels_df=labels_df,
        tokenizer=tokenizer,
        split="train",
        seed=seed,
    )
    val_dataset = NextVisitDataset(
        data_path=data_path,
        labels_df=labels_df,
        tokenizer=tokenizer,
        split="val",
        seed=seed,
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Vocab size: {tokenizer.get_vocab_size()}")

    model = create_next_visit_model(
        pretrained_path=pretrained_path,
        vocab_size=tokenizer.get_vocab_size(),
        dropout=dropout,
        pooling=pooling,
        freeze_encoder=freeze_encoder,
    )

    trainer = NextVisitTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        output_dir=output_dir,
        learning_rate=learning_rate,
        batch_size=batch_size,
        num_epochs=num_epochs,
        warmup_ratio=warmup_ratio,
        k_values=k_values,
        use_wandb=use_wandb,
    )

    best_metric = trainer.train()

    print("\nEvaluating on test set...")
    test_dataset = NextVisitDataset(
        data_path=data_path,
        labels_df=labels_df,
        tokenizer=tokenizer,
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

    results = {
        "task": "next_visit",
        "best_val_metric": best_metric,
        "test_metrics": test_metrics,
    }

    with open(Path(output_dir) / "test_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return test_metrics
