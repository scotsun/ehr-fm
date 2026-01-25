#!/usr/bin/env python3
"""
Hi-BEHRT BYOL Pre-training Script

Bootstrap Your Own Latent (BYOL) self-supervised pre-training for Hi-BEHRT.

BYOL is a self-supervised learning method that uses:
- Dual network structure: online network + target network (EMA updated)
- Projector + Predictor MLP heads on the online network
- Cosine similarity loss between online predictions and target projections
- Bernoulli masking for augmentation

Time Encoding:
- Uses Time2Vec to encode cumulative time (cumsum of days_since_prior_admission)
- This replaces the original age embedding from the Hi-BEHRT paper

Usage:
    python train_hi_behrt_byol.py --data_path dataset/mimic4/data/mimic4_tokens.parquet
    python train_hi_behrt_byol.py --data_path dataset/mimic4/data/mimic4_tokens.parquet --use_amp
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.distributions import Bernoulli
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import json
from datetime import datetime
import pyarrow.parquet as pq

from tokenizers import Tokenizer

from src.baselines.hi_behrt import (
    HiBEHRTConfig,
    HiBEHRTForBYOL,
    create_hi_behrt_config,
)


class BYOLPretrainDataset(Dataset):
    """
    Dataset for Hi-BEHRT BYOL pre-training.

    Loads patient sequences with cumsum time for Time2Vec encoding.
    Returns flat sequences that will be segmented by the model.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: Tokenizer,
        max_total_len: int = 2000,  # max total sequence length
        patient_id_col: str = "subject_id",
        enc_id_col: str = "hadm_id",
        token_col: str = "code",
        time_col: str = "days_since_prior_admission",
        sort_col: str = "visit_seq",
        token_time_col: str = "time_offset_hours",
    ):
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_total_len = max_total_len
        self.patient_id_col = patient_id_col
        self.enc_id_col = enc_id_col
        self.token_col = token_col
        self.time_col = time_col
        self.sort_col = sort_col
        self.token_time_col = token_time_col

        # Get pad and CLS token IDs
        self.pad_token_id = tokenizer.token_to_id("[PAD]")
        self.cls_token_id = tokenizer.token_to_id("[CLS]")

        # Scan patient directories (fast, only reads directory structure)
        self.subject_dirs = sorted(self.data_path.glob(f"{patient_id_col}=*"))
        if not self.subject_dirs:
            self.subject_dirs = sorted(self.data_path.glob("subject_id=*"))
        if not self.subject_dirs:
            raise ValueError(f"No patient directories found in {data_path}")

        self.patient_ids = [d.name.split('=')[1] for d in self.subject_dirs]
        print(f"Found {len(self.patient_ids)} patients for BYOL pretraining")

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, index):
        """
        Load patient data and return flat sequence with time values.

        Returns:
            dict with:
                - input_ids: (total_seq_len,) flat token sequence
                - attention_mask: (total_seq_len,) 1 for valid tokens
                - time_values: (total_seq_len,) cumulative time in days
        """
        subject_dir = self.subject_dirs[index]

        # Read patient data
        needed_cols = [self.enc_id_col, self.token_col, self.sort_col]
        if self.time_col:
            needed_cols.append(self.time_col)
        if self.token_time_col:
            needed_cols.append(self.token_time_col)

        table = pq.read_table(subject_dir, columns=needed_cols)
        patient_data = table.to_pandas()

        # Group by encounter and sort
        grouped = patient_data.groupby(self.enc_id_col)

        encounter_data = []
        for enc_id, group in grouped:
            tokens = group[self.token_col].tolist()
            time_val = group[self.time_col].iloc[0] if self.time_col and self.time_col in group.columns else 0.0
            sort_val = group[self.sort_col].iloc[0] if self.sort_col and self.sort_col in group.columns else 0

            # Token-level time (hours within encounter)
            if self.token_time_col and self.token_time_col in group.columns:
                token_times = group[self.token_time_col].tolist()
            else:
                token_times = [0.0] * len(tokens)

            encounter_data.append({
                'enc_id': enc_id,
                'tokens': tokens,
                'time': time_val if not (time_val is None or (isinstance(time_val, float) and np.isnan(time_val))) else 0.0,
                'sort_key': sort_val,
                'token_times': token_times,
            })

        # Sort by visit order
        encounter_data.sort(key=lambda x: x['sort_key'] if x['sort_key'] is not None else 0)

        # Compute cumulative time (segment level)
        cumsum_time = 0.0
        for enc in encounter_data:
            cumsum_time += enc['time']
            enc['cumsum_time'] = cumsum_time

        # Flatten tokens and create time values (token-level)
        all_tokens = []
        all_times = []

        for enc in encounter_data:
            # Add [CLS] token at start of each encounter
            all_tokens.append("[CLS]")
            all_times.append(enc['cumsum_time'])

            # Add tokens with their cumsum time + token-level offset (in days)
            for tok, tok_time in zip(enc['tokens'], enc['token_times']):
                all_tokens.append(tok)
                # Token time is cumsum days + offset hours converted to days
                tok_time_clean = tok_time if not (tok_time is None or (isinstance(tok_time, float) and np.isnan(tok_time))) else 0.0
                all_times.append(enc['cumsum_time'] + tok_time_clean / 24.0)

        # Truncate if too long (keep most recent)
        if len(all_tokens) > self.max_total_len:
            all_tokens = all_tokens[-self.max_total_len:]
            all_times = all_times[-self.max_total_len:]

        # Tokenize
        encoding = self.tokenizer.encode(all_tokens, is_pretokenized=True)
        input_ids = encoding.ids

        # Create attention mask
        attention_mask = [1 if tid != self.pad_token_id else 0 for tid in input_ids]

        # Pad time values to match input_ids length
        if len(all_times) < len(input_ids):
            last_time = all_times[-1] if all_times else 0.0
            all_times = all_times + [last_time] * (len(input_ids) - len(all_times))
        else:
            all_times = all_times[:len(input_ids)]

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'time_values': torch.tensor(all_times, dtype=torch.float),
        }


def collate_byol(batch):
    """Collate function for BYOL pretraining - pads to max length in batch."""
    max_len = max(item['input_ids'].shape[0] for item in batch)

    input_ids = []
    attention_mask = []
    time_values = []

    for item in batch:
        seq_len = item['input_ids'].shape[0]
        pad_len = max_len - seq_len

        if pad_len > 0:
            input_ids.append(F.pad(item['input_ids'], (0, pad_len), value=0))
            attention_mask.append(F.pad(item['attention_mask'], (0, pad_len), value=0))
            # Pad time with last value
            last_time = item['time_values'][-1].item() if seq_len > 0 else 0.0
            time_values.append(F.pad(item['time_values'], (0, pad_len), value=last_time))
        else:
            input_ids.append(item['input_ids'])
            attention_mask.append(item['attention_mask'])
            time_values.append(item['time_values'])

    return {
        'input_ids': torch.stack(input_ids),
        'attention_mask': torch.stack(attention_mask),
        'time_values': torch.stack(time_values),
    }


class BYOLPretrainer:
    """Trainer for Hi-BEHRT BYOL pre-training."""

    def __init__(
        self,
        model: HiBEHRTForBYOL,
        train_dataset: Dataset,
        val_dataset: Dataset = None,
        output_dir: str = "checkpoints/hi-behrt-byol",
        learning_rate: float = 1e-4,
        batch_size: int = 32,
        num_epochs: int = 100,
        warmup_ratio: float = 0.1,
        weight_decay: float = 0.01,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        byol_momentum: float = 0.99,
        mask_probability: float = 0.15,
        use_amp: bool = False,
        device: str = "cuda",
        num_workers: int = 4,
        save_every: int = 5,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.num_epochs = num_epochs
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.byol_momentum = byol_momentum
        self.mask_probability = mask_probability
        self.use_amp = use_amp and self.device.type == "cuda"
        self.save_every = save_every

        # Data loaders
        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            collate_fn=collate_byol, num_workers=num_workers,
            pin_memory=(self.device.type == "cuda"),
            drop_last=True,
        )
        self.val_loader = None
        if val_dataset is not None:
            self.val_loader = DataLoader(
                val_dataset, batch_size=batch_size * 2, shuffle=False,
                collate_fn=collate_byol, num_workers=num_workers,
                pin_memory=(self.device.type == "cuda"),
            )

        # Optimizer - exclude target network
        trainable_params = [p for n, p in model.named_parameters()
                          if not n.startswith('target_') and p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable_params, lr=learning_rate, weight_decay=weight_decay
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

    def train_step(self, batch):
        """
        Single training step for BYOL.

        The model (HiBEHRTForBYOL) handles segmentation internally.
        We pass flat sequences and the model returns segment-level representations.
        """
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        time_values = batch['time_values'].to(self.device)

        # First forward pass (without masking) to get global_mask shape
        # This is needed to create bernoulli_mask with correct dimensions
        with torch.no_grad():
            _, _, _, global_mask = self.model(
                input_ids, attention_mask, time_values,
                bernoulli_mask=None, apply_mask=False
            )

        # Create Bernoulli mask for BYOL augmentation
        # mask_prob = probability of KEEPING a segment (1 = keep, 0 = mask)
        bernoulli_mask = Bernoulli(
            torch.ones_like(global_mask.float()) * (1 - self.mask_probability)
        ).sample().to(self.device)

        # View 1: Online network with masking
        y1, z1, h1, global_mask = self.model(
            input_ids, attention_mask, time_values,
            bernoulli_mask=bernoulli_mask, apply_mask=True
        )

        # View 2: Target network without masking
        with torch.no_grad():
            y2, z2, _ = self.model.forward_target(
                input_ids, attention_mask, time_values
            )

        # Loss A: predict target from masked online
        loss_a = self.model.byol_loss(h1, z2, global_mask, bernoulli_mask)

        # View 3: Online network without masking (for symmetric loss)
        y3, z3, h3, _ = self.model(
            input_ids, attention_mask, time_values,
            bernoulli_mask=bernoulli_mask, apply_mask=False
        )

        # View 4: Target network with masking (use same data, different augmentation)
        # For symmetric loss, we use online network with masking as target
        with torch.no_grad():
            y4, z4, _, _ = self.model(
                input_ids, attention_mask, time_values,
                bernoulli_mask=bernoulli_mask, apply_mask=True
            )

        # Loss B: symmetric (predict masked from unmasked)
        loss_b = self.model.byol_loss(h3, z4.detach(), global_mask, bernoulli_mask)

        total_loss = loss_a + loss_b

        return total_loss, loss_a.item(), loss_b.item()

    def train_epoch(self, epoch):
        """Train one epoch."""
        self.model.train()
        total_loss = 0
        total_loss_a = 0
        total_loss_b = 0
        num_batches = 0

        self.optimizer.zero_grad()
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}")

        for step, batch in enumerate(pbar):
            if self.use_amp:
                with autocast("cuda"):
                    loss, loss_a, loss_b = self.train_step(batch)
                    loss = loss / self.gradient_accumulation_steps

                self.scaler.scale(loss).backward()

                if (step + 1) % self.gradient_accumulation_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self.model.parameters() if p.requires_grad],
                        self.max_grad_norm
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                    # Update target network with EMA
                    self.model.update_target_network(self.byol_momentum)
                    self.global_step += 1
            else:
                loss, loss_a, loss_b = self.train_step(batch)
                loss = loss / self.gradient_accumulation_steps
                loss.backward()

                if (step + 1) % self.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self.model.parameters() if p.requires_grad],
                        self.max_grad_norm
                    )
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                    # Update target network with EMA
                    self.model.update_target_network(self.byol_momentum)
                    self.global_step += 1

            total_loss += loss.item() * self.gradient_accumulation_steps
            total_loss_a += loss_a
            total_loss_b += loss_b
            num_batches += 1

            pbar.set_postfix({
                "loss": f"{total_loss / num_batches:.4f}",
                "loss_a": f"{total_loss_a / num_batches:.4f}",
                "loss_b": f"{total_loss_b / num_batches:.4f}",
            })

        return total_loss / num_batches

    @torch.no_grad()
    def validate(self):
        """Validate on validation set."""
        if self.val_loader is None:
            return None

        self.model.eval()
        total_loss = 0
        num_batches = 0

        for batch in tqdm(self.val_loader, desc="Validating"):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            time_values = batch['time_values'].to(self.device)

            # First get global_mask shape
            _, _, _, global_mask = self.model(
                input_ids, attention_mask, time_values,
                bernoulli_mask=None, apply_mask=False
            )

            # Create mask
            bernoulli_mask = Bernoulli(
                torch.ones_like(global_mask.float()) * (1 - self.mask_probability)
            ).sample().to(self.device)

            # Forward with masking
            y1, z1, h1, global_mask = self.model(
                input_ids, attention_mask, time_values,
                bernoulli_mask=bernoulli_mask, apply_mask=True
            )

            # Target forward
            y2, z2, _ = self.model.forward_target(
                input_ids, attention_mask, time_values
            )

            loss = self.model.byol_loss(h1, z2, global_mask, bernoulli_mask)
            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches

    def save_checkpoint(self, epoch, val_loss=None):
        """Save checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'config': self.model.config,
        }
        if val_loss is not None:
            checkpoint['val_loss'] = val_loss

        # Save latest
        torch.save(checkpoint, self.output_dir / "latest_checkpoint.pt")

        # Save epoch checkpoint
        if (epoch + 1) % self.save_every == 0:
            torch.save(checkpoint, self.output_dir / f"checkpoint_epoch_{epoch + 1}.pt")

    def train(self):
        """Full training loop."""
        print(f"\n{'='*60}")
        print(f"Hi-BEHRT BYOL Pre-training")
        print(f"{'='*60}")
        print(f"  Train samples: {len(self.train_loader.dataset):,}")
        if self.val_loader:
            print(f"  Val samples:   {len(self.val_loader.dataset):,}")
        print(f"  Epochs: {self.num_epochs}")
        print(f"  BYOL momentum: {self.byol_momentum}")
        print(f"  Mask probability: {self.mask_probability}")
        print(f"{'='*60}\n")

        best_val_loss = float('inf')

        for epoch in range(self.num_epochs):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate()

            print(f"\nEpoch {epoch + 1}/{self.num_epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            if val_loss is not None:
                print(f"  Val Loss:   {val_loss:.4f}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'config': self.model.config,
                        'val_loss': val_loss,
                    }, self.output_dir / "best_model.pt")
                    print(f"  -> Saved best model (val_loss: {val_loss:.4f})")

            self.save_checkpoint(epoch, val_loss)

        # Save final encoder for downstream tasks
        torch.save({
            'encoder_state_dict': self.model.encoder.state_dict(),
            'config': self.model.config,
        }, self.output_dir / "encoder_final.pt")
        print(f"\nTraining complete. Encoder saved to {self.output_dir / 'encoder_final.pt'}")

        return best_val_loss if best_val_loss != float('inf') else train_loss


def parse_args():
    parser = argparse.ArgumentParser(description="Hi-BEHRT BYOL Pre-training")

    # Data
    parser.add_argument("--data_path", type=str, required=True,
                       help="Path to Hive-partitioned parquet data")
    parser.add_argument("--tokenizer_path", type=str, default="tokenizer.json")
    parser.add_argument("--output_dir", type=str, default="checkpoints/hi-behrt-byol")
    parser.add_argument("--val_split", type=float, default=0.05,
                       help="Validation split ratio")

    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    # BYOL specific
    parser.add_argument("--byol_momentum", type=float, default=0.99,
                       help="EMA momentum for target network update")
    parser.add_argument("--mask_probability", type=float, default=0.15,
                       help="Probability of masking segments")
    parser.add_argument("--projector_hidden_size", type=int, default=256)
    parser.add_argument("--projector_output_size", type=int, default=128)

    # Model architecture
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--n_extractor_layers", type=int, default=4)
    parser.add_argument("--n_aggregator_layers", type=int, default=4)
    parser.add_argument("--n_heads", type=int, default=12)
    parser.add_argument("--d_ff", type=int, default=2048)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--attention_dropout", type=float, default=0.1)

    # Hi-BEHRT specific
    parser.add_argument("--window_size", type=int, default=50)
    parser.add_argument("--stride", type=int, default=30)
    parser.add_argument("--max_segments", type=int, default=40)
    parser.add_argument("--max_total_len", type=int, default=2000,
                       help="Maximum total sequence length")

    # Time2Vec
    parser.add_argument("--t2v_dim", type=int, default=64,
                       help="Time2Vec output dimension")

    # Misc
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_every", type=int, default=10,
                       help="Save checkpoint every N epochs")

    return parser.parse_args()


def main():
    args = parse_args()

    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Hi-BEHRT BYOL Pre-training")
    print("=" * 60)

    # Load tokenizer
    tokenizer = Tokenizer.from_file(args.tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()
    print(f"Vocab size: {vocab_size}")

    # Create dataset
    print(f"\nLoading data from {args.data_path}")
    full_dataset = BYOLPretrainDataset(
        data_path=args.data_path,
        tokenizer=tokenizer,
        max_total_len=args.max_total_len,
    )

    # Split into train/val
    n_total = len(full_dataset)
    n_val = int(n_total * args.val_split)
    n_train = n_total - n_val

    # Use random split
    indices = np.random.permutation(n_total)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices) if n_val > 0 else None

    print(f"  Train: {len(train_dataset)}, Val: {len(val_dataset) if val_dataset else 0}")

    # Create model config
    config = HiBEHRTConfig(
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_extractor_layers=args.n_extractor_layers,
        n_aggregator_layers=args.n_aggregator_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        max_seq_len=args.window_size,
        max_segments=args.max_segments,
        dropout=args.dropout,
        attention_dropout=args.attention_dropout,
        t2v_dim=args.t2v_dim,
        projector_hidden_size=args.projector_hidden_size,
        projector_output_size=args.projector_output_size,
        byol_momentum=args.byol_momentum,
    )

    # Create model
    print("\nCreating Hi-BEHRT for BYOL...")
    model = HiBEHRTForBYOL(config)

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {n_params:,}")
    print(f"  Trainable parameters: {n_trainable:,}")
    print(f"  d_model: {args.d_model}")
    print(f"  n_extractor_layers: {args.n_extractor_layers}")
    print(f"  n_aggregator_layers: {args.n_aggregator_layers}")
    print(f"  window_size: {args.window_size}, stride: {args.stride}")
    print(f"  t2v_dim: {args.t2v_dim}")

    # Save config
    with open(output_dir / "config.json", "w") as f:
        config_dict = vars(args)
        config_dict['vocab_size'] = vocab_size
        json.dump(config_dict, f, indent=2)

    # Create trainer
    trainer = BYOLPretrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        output_dir=output_dir,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        byol_momentum=args.byol_momentum,
        mask_probability=args.mask_probability,
        use_amp=args.use_amp,
        device=args.device,
        num_workers=args.num_workers,
        save_every=args.save_every,
    )

    # Train
    final_loss = trainer.train()

    # Save final results
    results = {
        "model": "hi-behrt-byol",
        "final_loss": final_loss,
        "epochs": args.num_epochs,
        "config": vars(args),
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
