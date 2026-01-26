#!/usr/bin/env python3
"""
Baseline Models Training Script
Pre-training CORE-BEHRT, HEART, and Hi-BEHRT on MIMIC-IV data
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import json

from src.tokenizer import get_tokenizer
from src.baselines.data_utils import FlatEHRDataset, get_sample_patient_ids
from src.baselines.core_behrt import BEHRT, BEHRTConfig
from src.baselines.heart import HEART, HEARTConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Baseline Models Training")

    # ========================================================================
    # MODEL SELECTION
    # ========================================================================
    parser.add_argument("--model", type=str, required=True,
                       choices=["core-behrt", "heart"],
                       help="Which baseline model to train (MLM pre-training)")

    # ========================================================================
    # DATA PARAMETERS
    # ========================================================================
    parser.add_argument("--data_path", type=str,
                       default="dataset/mimic4/data/mimic4_tokens.parquet",
                       help="Path to MIMIC-IV parquet data")
    parser.add_argument("--tokenizer_path", type=str, default=None,
                       help="Path to existing tokenizer.json (if None, will train new)")
    parser.add_argument("--output_dir", type=str, default="checkpoints",
                       help="Output directory for models and logs")

    # ========================================================================
    # TRAINING PARAMETERS
    # ========================================================================
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=50,
                       help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--mask_prob", type=float, default=0.15,
                       help="MLM masking probability")
    parser.add_argument("--patience", type=int, default=10,
                       help="Early stopping patience")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                       help="Gradient accumulation steps")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                       help="Max gradient norm for clipping")

    # ========================================================================
    # MODEL PARAMETERS
    # ========================================================================
    parser.add_argument("--d_model", type=int, default=256,
                       help="Model dimension")
    parser.add_argument("--n_blocks", type=int, default=4,
                       help="Number of transformer blocks")
    parser.add_argument("--n_heads", type=int, default=8,
                       help="Number of attention heads")
    parser.add_argument("--d_ff", type=int, default=512,
                       help="Feed-forward dimension")
    parser.add_argument("--max_seq_len", type=int, default=2048,
                       help="Maximum sequence length")
    parser.add_argument("--dropout", type=float, default=0.1)

    # ========================================================================
    # OPTIONAL
    # ========================================================================
    parser.add_argument("--max_patients", type=int, default=None,
                       help="Max patients for debugging")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_amp", action="store_true",
                       help="Enable Automatic Mixed Precision")
    parser.add_argument("--num_workers", type=int, default=4)

    # Checkpoint
    parser.add_argument("--save_every", type=int, default=5,
                       help="Save checkpoint every N epochs")
    parser.add_argument("--resume_from", type=str, default=None,
                       help="Path to checkpoint to resume from")

    return parser.parse_args()


class BaselineTrainer:
    """Trainer for baseline models with MLM pre-training."""

    def __init__(
        self,
        model,
        model_type: str,
        tokenizer,
        optimizer,
        device,
        mask_prob: float = 0.15,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        use_amp: bool = False,
    ):
        self.model = model
        self.model_type = model_type
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.device = device
        self.mask_prob = mask_prob
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.use_amp = use_amp and device.type == "cuda"

        self.scaler = GradScaler('cuda') if self.use_amp else None
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)

        # Special token IDs
        self.mask_id = tokenizer.token_to_id("[MASK]")
        self.pad_id = tokenizer.token_to_id("[PAD]")
        self.cls_id = tokenizer.token_to_id("[CLS]")
        self.sep_id = tokenizer.token_to_id("[SEP]")

    def _create_mlm_labels(self, input_ids, attention_mask):
        """Create MLM labels with random masking."""
        labels = input_ids.clone()

        # Probability matrix for masking
        prob_matrix = torch.rand(input_ids.shape, device=self.device)

        # Don't mask special tokens and padding
        special_tokens_mask = (
            (input_ids == self.pad_id) |
            (input_ids == self.cls_id) |
            (input_ids == self.sep_id)
        )
        prob_matrix.masked_fill_(special_tokens_mask, 1.0)
        prob_matrix.masked_fill_(~attention_mask.bool(), 1.0)

        # Select tokens to mask
        masked_indices = prob_matrix < self.mask_prob
        labels[~masked_indices] = -100  # Only compute loss on masked tokens

        # 80% -> [MASK], 10% -> random, 10% -> original
        indices_replaced = torch.bernoulli(torch.full(input_ids.shape, 0.8, device=self.device)).bool() & masked_indices
        input_ids_masked = input_ids.clone()
        input_ids_masked[indices_replaced] = self.mask_id

        indices_random = torch.bernoulli(torch.full(input_ids.shape, 0.5, device=self.device)).bool() & masked_indices & ~indices_replaced
        random_tokens = torch.randint(self.tokenizer.get_vocab_size(), input_ids.shape, device=self.device)
        input_ids_masked[indices_random] = random_tokens[indices_random]

        return input_ids_masked, labels

    def _forward_step(self, batch):
        """Forward pass for different model types."""
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)

        # Create MLM inputs
        input_ids_masked, labels = self._create_mlm_labels(input_ids, attention_mask)

        if self.model_type == "core-behrt":
            t = batch["t"].to(self.device)
            logits, _ = self.model(input_ids_masked, attention_mask, t)

        elif self.model_type == "heart":
            t = batch["t"].to(self.device)
            token_types = batch["token_types"].to(self.device)
            visit_ids = batch["visit_ids"].to(self.device)
            logits, _ = self.model(
                input_ids=input_ids_masked,
                attention_mask=attention_mask,
                token_types=token_types,
                visit_ids=visit_ids,
                time_offsets=t,
            )

        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        # Compute loss
        loss = self.criterion(logits.view(-1, logits.size(-1)), labels.view(-1))

        return loss, logits

    def train_epoch(self, dataloader, epoch_id=0):
        """Train one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = 0
        nan_count = 0

        self.optimizer.zero_grad()

        pbar = tqdm(dataloader, desc=f"Epoch {epoch_id} [Train]")
        for step, batch in enumerate(pbar):
            if self.use_amp:
                with autocast('cuda'):
                    loss, _ = self._forward_step(batch)
                    if loss is None:
                        continue
                    loss = loss / self.gradient_accumulation_steps

                # Check for NaN loss
                if torch.isnan(loss) or torch.isinf(loss):
                    nan_count += 1
                    if nan_count <= 3:
                        print(f"\nWarning: NaN/Inf loss detected at step {step}. Skipping batch.")
                    self.optimizer.zero_grad()
                    continue

                self.scaler.scale(loss).backward()

                if (step + 1) % self.gradient_accumulation_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    # Check for NaN gradients
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                        nan_count += 1
                        if nan_count <= 3:
                            print(f"\nWarning: NaN/Inf gradient at step {step}. Skipping update.")
                        self.optimizer.zero_grad()
                        continue
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                loss, _ = self._forward_step(batch)
                if loss is None:
                    continue
                loss = loss / self.gradient_accumulation_steps

                # Check for NaN loss
                if torch.isnan(loss) or torch.isinf(loss):
                    nan_count += 1
                    if nan_count <= 3:
                        print(f"\nWarning: NaN/Inf loss detected at step {step}. Skipping batch.")
                    self.optimizer.zero_grad()
                    continue

                loss.backward()

                if (step + 1) % self.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            total_loss += loss.item() * self.gradient_accumulation_steps
            num_batches += 1
            pbar.set_postfix({"loss": total_loss / num_batches})

        if nan_count > 0:
            print(f"\nEpoch {epoch_id}: {nan_count} NaN/Inf batches skipped")

        return total_loss / max(num_batches, 1)

    @torch.no_grad()
    def evaluate(self, dataloader, epoch_id=0):
        """Evaluate on validation set."""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        total_correct = 0
        total_tokens = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch_id} [Val]")
        for batch in pbar:
            if self.use_amp:
                with autocast('cuda'):
                    loss, logits = self._forward_step(batch)
            else:
                loss, logits = self._forward_step(batch)

            if loss is None:
                continue

            total_loss += loss.item()
            num_batches += 1

            # Calculate accuracy on masked tokens
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            _, labels = self._create_mlm_labels(input_ids, attention_mask)

            mask = labels != -100
            if mask.any():
                preds = logits.argmax(dim=-1)
                correct = (preds == labels) & mask
                total_correct += correct.sum().item()
                total_tokens += mask.sum().item()

            pbar.set_postfix({"loss": total_loss / num_batches})

        avg_loss = total_loss / max(num_batches, 1)
        accuracy = total_correct / max(total_tokens, 1)

        return {"loss": avg_loss, "accuracy": accuracy}


def create_model(model_type, vocab_size, args):
    """Create model based on type."""
    if model_type == "core-behrt":
        config = BEHRTConfig(
            vocab_size=vocab_size,
            d_model=args.d_model,
            n_blocks=args.n_blocks,
            n_heads=args.n_heads,
            d_ff=args.d_ff,
            max_seq_len=args.max_seq_len,
            dropout=args.dropout,
        )
        return BEHRT(config)

    elif model_type == "heart":
        config = HEARTConfig(
            vocab_size=vocab_size,
            d_model=args.d_model,
            n_blocks=args.n_blocks,
            n_heads=args.n_heads,
            d_ff=args.d_ff,
            max_seq_len=args.max_seq_len,
            dropout=args.dropout,
            n_token_types=8,
            edge_hidden_size=64,
            max_visits=50,
        )
        return HEART(config)

    else:
        raise ValueError(f"Unknown model type: {model_type}")


def main():
    args = parse_args()

    # Set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Create output directory
    output_dir = Path(args.output_dir) / args.model
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"  {args.model.upper()} Pre-training")
    print("=" * 60)

    # Load data
    print("\nLoading data...")
    data_path = Path(args.data_path)

    # Get patient directories
    subject_dirs = sorted(data_path.glob("subject_id=*"))
    if args.max_patients:
        subject_dirs = subject_dirs[:args.max_patients]

    patient_ids = [d.name.split('=')[1] for d in subject_dirs]
    print(f"Total patients: {len(patient_ids)}")

    # Load or train tokenizer
    if args.tokenizer_path and Path(args.tokenizer_path).exists():
        print(f"Loading tokenizer from {args.tokenizer_path}")
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(args.tokenizer_path)
    else:
        print("Training tokenizer...")
        import pyarrow.parquet as pq
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def read_codes(d):
            try:
                table = pq.read_table(d, columns=['subject_id', 'code'])
                return table.to_pandas()
            except:
                return None

        dfs = []
        with ThreadPoolExecutor(max_workers=8) as ex:
            futures = {ex.submit(read_codes, d): d for d in subject_dirs}
            for f in tqdm(as_completed(futures), total=len(futures), desc="Loading"):
                r = f.result()
                if r is not None:
                    dfs.append(r)

        df_sample = pd.concat(dfs, ignore_index=True)
        tokenizer = get_tokenizer([df_sample], {
            "tokenizer_path": str(output_dir / "tokenizer.json"),
            "patient_id_col": "subject_id",
            "token_col": "code",
            "min_frequency": 5,
        })

    vocab_size = tokenizer.get_vocab_size()
    print(f"Vocab size: {vocab_size}")

    # Split patients
    np.random.shuffle(patient_ids)
    n_train = int(len(patient_ids) * args.train_ratio)
    n_val = int(len(patient_ids) * args.val_ratio)

    train_ids = patient_ids[:n_train]
    val_ids = patient_ids[n_train:n_train+n_val]
    test_ids = patient_ids[n_train+n_val:]

    print(f"Split: Train={len(train_ids)}, Val={len(val_ids)}, Test={len(test_ids)}")

    # Create datasets
    print("\nCreating datasets...")

    # Dataset parameters based on model type
    include_token_types = (args.model == "heart")
    include_visit_ids = (args.model == "heart")

    common_cfg = {
        "data_path": str(data_path),
        "tokenizer": tokenizer,
        "max_seq_len": args.max_seq_len,
        "patient_id_col": "subject_id",
        "enc_id_col": "hadm_id",
        "token_col": "code",
        "code_type_col": "code_type",
        "sort_col": "visit_seq",
        "token_time_col": "time_offset_hours",
        "visit_time_col": "days_since_prior_admission",
        "include_token_types": include_token_types,
        "include_visit_ids": include_visit_ids,
    }

    train_cohort = pd.DataFrame({"subject_id": train_ids})
    val_cohort = pd.DataFrame({"subject_id": val_ids})
    test_cohort = pd.DataFrame({"subject_id": test_ids})

    train_dataset = FlatEHRDataset(supervised_task_cohort=train_cohort, **common_cfg)
    val_dataset = FlatEHRDataset(supervised_task_cohort=val_cohort, **common_cfg)
    test_dataset = FlatEHRDataset(supervised_task_cohort=test_cohort, **common_cfg)

    use_gpu = args.device == "cuda" and torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=use_gpu
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=use_gpu
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=use_gpu
    )

    # Create model
    print("\nCreating model...")
    device = torch.device(args.device)
    model = create_model(args.model, vocab_size, args)
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    print(f"Device: {device}")
    print(f"AMP: {'Enabled' if args.use_amp else 'Disabled'}")

    # Save config
    config_path = output_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)

    # Trainer
    trainer = BaselineTrainer(
        model=model,
        model_type=args.model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        device=device,
        mask_prob=args.mask_prob,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        use_amp=args.use_amp,
    )

    # Resume from checkpoint
    start_epoch = 0
    best_val_loss = float('inf')

    if args.resume_from and Path(args.resume_from).exists():
        print(f"\nResuming from {args.resume_from}")
        ckpt = torch.load(args.resume_from, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_val_loss = ckpt.get('best_val_loss', float('inf'))
        if trainer.scaler and 'scaler_state_dict' in ckpt:
            trainer.scaler.load_state_dict(ckpt['scaler_state_dict'])
        print(f"Resumed from epoch {start_epoch}")

    # Training
    print("\n" + "=" * 60)
    print("Training...")
    print("=" * 60)

    patience_counter = 0

    for epoch in range(start_epoch, args.num_epochs):
        # Train
        train_loss = trainer.train_epoch(train_loader, epoch_id=epoch)

        # Validate
        val_metrics = trainer.evaluate(val_loader, epoch_id=epoch)
        val_loss = val_metrics['loss']
        val_acc = val_metrics['accuracy']

        print(f"Epoch {epoch} | train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f} | val_acc: {val_acc:.4f}")

        # Save checkpoint
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % args.save_every == 0 or is_best:
            ckpt_path = output_dir / f"checkpoint_epoch{epoch}.pt"
            ckpt = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'best_val_loss': best_val_loss,
            }
            if trainer.scaler:
                ckpt['scaler_state_dict'] = trainer.scaler.state_dict()
            torch.save(ckpt, ckpt_path)

            if is_best:
                best_path = output_dir / "best_model.pt"
                torch.save(ckpt, best_path)
                print(f"  -> Saved best model (val_loss: {val_loss:.4f})")

        # Latest checkpoint for resume
        latest_path = output_dir / "latest_checkpoint.pt"
        torch.save(ckpt, latest_path)

        # Early stopping
        if patience_counter >= args.patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break

    # Test
    print("\n" + "=" * 60)
    print("Testing...")
    print("=" * 60)

    # Load best model
    best_path = output_dir / "best_model.pt"
    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])

    test_metrics = trainer.evaluate(test_loader, epoch_id=0)
    print(f"Test Loss: {test_metrics['loss']:.4f}")
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")

    # Save final results
    results = {
        "model": args.model,
        "best_val_loss": best_val_loss,
        "test_loss": test_metrics['loss'],
        "test_accuracy": test_metrics['accuracy'],
        "n_params": n_params,
    }
    with open(output_dir / "results.json", 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print(f"Training complete! Model saved to {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
