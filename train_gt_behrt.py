#!/usr/bin/env python3
"""
GT-BEHRT Training Script

Pre-training GT-BEHRT on MIMIC-IV data using:
- Step 1: Node Attribute Masking (NAM) - trains Graph Transformer only
- Step 2: Missing Node Prediction (MNP) + Visit Type Prediction (VTP) - trains full model
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
import signal
import random

from src.tokenizer import get_tokenizer
from src.baselines.data_utils import GTBEHRTDataset, collate_gtbehrt
from src.baselines.gt_behrt import (
    GTBEHRT,
    GTBEHRTConfig,
    GTBEHRTForPretraining,
    create_gtbehrt_config,
)


def parse_args():
    parser = argparse.ArgumentParser(description="GT-BEHRT Pre-training")

    # ========================================================================
    # DATA PARAMETERS
    # ========================================================================
    parser.add_argument("--data_path", type=str,
                       default="dataset/mimic4/data/mimic4_tokens.parquet",
                       help="Path to MIMIC-IV parquet data")
    parser.add_argument("--tokenizer_path", type=str, default=None,
                       help="Path to existing tokenizer.json")
    parser.add_argument("--output_dir", type=str, default="checkpoints/gt-behrt",
                       help="Output directory for models")

    # ========================================================================
    # TRAINING PARAMETERS
    # ========================================================================
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=50,
                       help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=3e-5,
                       help="Learning rate")
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                       help="Warmup ratio")
    parser.add_argument("--patience", type=int, default=5,
                       help="Early stopping patience")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                       help="Gradient accumulation steps")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                       help="Max gradient norm for clipping")

    # ========================================================================
    # MODEL PARAMETERS
    # ========================================================================
    parser.add_argument("--hidden_size", type=int, default=540,
                       help="Hidden size (must be 5 * d_stream)")
    parser.add_argument("--n_graph_layers", type=int, default=3,
                       help="Number of Graph Transformer layers")
    parser.add_argument("--n_bert_layers", type=int, default=6,
                       help="Number of BERT layers")
    parser.add_argument("--n_heads", type=int, default=12,
                       help="Number of attention heads")
    parser.add_argument("--dropout", type=float, default=0.2,
                       help="Dropout rate")

    # ========================================================================
    # SEQUENCE PARAMETERS
    # ========================================================================
    parser.add_argument("--max_visits", type=int, default=50,
                       help="Maximum number of visits per patient")
    parser.add_argument("--max_codes_per_visit", type=int, default=100,
                       help="Maximum codes per visit")

    # ========================================================================
    # PRE-TRAINING PARAMETERS
    # ========================================================================
    parser.add_argument("--nam_mask_prob", type=float, default=0.15,
                       help="NAM masking probability")
    parser.add_argument("--vtp_mask_prob", type=float, default=0.5,
                       help="VTP masking probability")
    parser.add_argument("--nam_epochs", type=int, default=10,
                       help="Epochs for NAM pre-training (Step 1)")

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
    parser.add_argument("--save_every", type=int, default=5,
                       help="Save checkpoint every N epochs")
    parser.add_argument("--resume_from", type=str, default=None,
                       help="Path to checkpoint to resume from")

    return parser.parse_args()


class GTBEHRTTrainer:
    """Trainer for GT-BEHRT pre-training."""

    def __init__(
        self,
        model: GTBEHRTForPretraining,
        tokenizer,
        optimizer,
        scheduler,
        device,
        nam_mask_prob: float = 0.15,
        vtp_mask_prob: float = 0.5,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        use_amp: bool = False,
        output_dir: Path = None,
        patience: int = 5,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.nam_mask_prob = nam_mask_prob
        self.vtp_mask_prob = vtp_mask_prob
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.use_amp = use_amp and device.type == "cuda"
        self.output_dir = output_dir
        self.patience = patience

        self.scaler = GradScaler('cuda') if self.use_amp else None

        # Special token IDs
        self.mask_id = tokenizer.token_to_id("[MASK]")
        self.pad_id = tokenizer.token_to_id("[PAD]")
        self.vocab_size = tokenizer.get_vocab_size()

        # Graceful shutdown
        self.should_stop = False
        signal.signal(signal.SIGTERM, self._handle_signal)

    def _handle_signal(self, signum, frame):
        """Handle termination signal for graceful shutdown."""
        print("\nReceived shutdown signal, saving checkpoint...")
        self.should_stop = True

    def _move_to_device(self, batch):
        """Move batch data to device."""
        graph_data = {
            'node_ids': batch['graph_data']['node_ids'].to(self.device),
            'edge_index': batch['graph_data']['edge_index'].to(self.device),
            'edge_type': batch['graph_data']['edge_type'].to(self.device),
            'vst_indices': batch['graph_data']['vst_indices'].to(self.device),
            'batch_visit_counts': batch['graph_data']['batch_visit_counts'].to(self.device),
        }
        return {
            'graph_data': graph_data,
            'visit_types': batch['visit_types'].to(self.device),
            'positions': batch['positions'].to(self.device),
            'ages': batch['ages'].to(self.device),
            'days': batch['days'].to(self.device),
            'attention_mask': batch['attention_mask'].to(self.device),
        }

    def _create_nam_inputs(self, batch):
        """Create inputs for Node Attribute Masking (NAM)."""
        graph_data = batch['graph_data']
        node_ids = graph_data['node_ids']
        vst_indices = graph_data['vst_indices']

        # Create mask for non-VST nodes
        all_indices = torch.arange(node_ids.size(0), device=self.device)
        vst_mask = torch.zeros(node_ids.size(0), dtype=torch.bool, device=self.device)
        vst_mask[vst_indices] = True

        # Get non-VST node indices
        non_vst_mask = ~vst_mask

        # Random masking
        prob_matrix = torch.rand(node_ids.size(0), device=self.device)
        prob_matrix[vst_mask] = 1.0  # Don't mask VST nodes
        prob_matrix[node_ids == self.pad_id] = 1.0  # Don't mask PAD

        masked_indices = (prob_matrix < self.nam_mask_prob) & non_vst_mask
        labels = node_ids[masked_indices].clone()

        # Replace masked nodes with [MASK] token
        node_ids_masked = node_ids.clone()
        node_ids_masked[masked_indices] = self.mask_id

        # Update graph data with masked node IDs
        graph_data_masked = {
            'node_ids': node_ids_masked,
            'edge_index': graph_data['edge_index'],
            'edge_type': graph_data['edge_type'],
            'vst_indices': vst_indices,
            'batch_visit_counts': graph_data['batch_visit_counts'],
        }

        return graph_data_masked, masked_indices.nonzero(as_tuple=True)[0], labels

    def _create_mnp_vtp_inputs(self, batch):
        """Create inputs for MNP and VTP pre-training.

        MNP (Missing Node Prediction):
        - Select an actual medical code from each patient's graph
        - Use the token ID as the label for prediction
        - The model uses CLS token to predict this "missing" code

        VTP (Visit Type Prediction):
        - Mask some visit types and predict them
        """
        graph_data = batch['graph_data']
        visit_types = batch['visit_types']

        node_ids = graph_data['node_ids']  # (total_nodes,)
        vst_indices = graph_data['vst_indices']  # (total_visits,)
        batch_visit_counts = graph_data['batch_visit_counts']  # (batch_size,)

        batch_size = visit_types.size(0)

        # MNP: Select an actual code from each patient's graph
        mnp_labels = []

        # Create a mask for VST nodes (these should not be selected for MNP)
        vst_mask = torch.zeros(node_ids.size(0), dtype=torch.bool, device=self.device)
        vst_mask[vst_indices] = True

        # Also exclude PAD tokens (token_id == 0)
        non_pad_mask = node_ids != self.pad_id

        # Valid nodes for MNP: not VST and not PAD
        valid_for_mnp = (~vst_mask) & non_pad_mask

        # Track node boundaries per patient
        node_offset = 0
        visit_offset = 0

        for i in range(batch_size):
            n_visits = batch_visit_counts[i].item()

            # Find nodes belonging to this patient
            # Each visit has 1 VST + some code nodes
            # We need to find where this patient's nodes start and end
            patient_vst_indices = vst_indices[visit_offset:visit_offset + n_visits]

            if len(patient_vst_indices) > 0:
                # Patient's nodes span from first VST to before next patient's first VST
                patient_start = patient_vst_indices[0].item()

                # Find end: either next patient's start or end of all nodes
                if i + 1 < batch_size:
                    next_visit_offset = visit_offset + n_visits
                    if next_visit_offset < len(vst_indices):
                        patient_end = vst_indices[next_visit_offset].item()
                    else:
                        patient_end = node_ids.size(0)
                else:
                    patient_end = node_ids.size(0)

                # Get valid (non-VST, non-PAD) nodes for this patient
                patient_valid_mask = valid_for_mnp[patient_start:patient_end]
                patient_node_ids = node_ids[patient_start:patient_end]

                valid_indices = patient_valid_mask.nonzero(as_tuple=True)[0]

                if len(valid_indices) > 0:
                    # Randomly select one code node
                    rand_idx = torch.randint(0, len(valid_indices), (1,), device=self.device)
                    selected_local_idx = valid_indices[rand_idx[0]]
                    selected_token_id = patient_node_ids[selected_local_idx]
                    mnp_labels.append(selected_token_id)
                else:
                    # Fallback: use a random valid token from vocabulary
                    # This shouldn't happen often if data is correct
                    mnp_labels.append(torch.tensor(1, device=self.device))  # Use [CLS] as fallback
            else:
                # No visits for this patient (shouldn't happen)
                mnp_labels.append(torch.tensor(1, device=self.device))

            visit_offset += n_visits

        mnp_labels = torch.stack(mnp_labels)

        # VTP: Mask visit types
        vtp_labels = visit_types.clone()
        vtp_mask = torch.rand_like(visit_types.float()) < self.vtp_mask_prob
        vtp_mask = vtp_mask & batch['attention_mask']  # Only mask valid visits

        # Replace masked visit types with 0 (mask token)
        visit_types_masked = visit_types.clone()
        visit_types_masked[vtp_mask] = 0

        return mnp_labels, vtp_labels, vtp_mask, visit_types_masked

    def train_epoch_nam(self, dataloader, epoch_id=0):
        """Train one epoch of NAM (Step 1)."""
        self.model.train()
        total_loss = 0
        num_batches = 0

        self.optimizer.zero_grad()

        pbar = tqdm(dataloader, desc=f"Epoch {epoch_id} [NAM]")
        for step, batch in enumerate(pbar):
            if self.should_stop:
                break

            batch = self._move_to_device(batch)
            graph_data_masked, masked_indices, labels = self._create_nam_inputs(batch)

            if len(labels) == 0:
                continue

            if self.use_amp:
                with autocast('cuda'):
                    loss, _ = self.model.forward_nam(graph_data_masked, masked_indices, labels)
                    loss = loss / self.gradient_accumulation_steps

                self.scaler.scale(loss).backward()

                if (step + 1) % self.gradient_accumulation_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    if self.scheduler:
                        self.scheduler.step()
            else:
                loss, _ = self.model.forward_nam(graph_data_masked, masked_indices, labels)
                loss = loss / self.gradient_accumulation_steps
                loss.backward()

                if (step + 1) % self.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    if self.scheduler:
                        self.scheduler.step()

            total_loss += loss.item() * self.gradient_accumulation_steps
            num_batches += 1
            pbar.set_postfix({"loss": total_loss / num_batches})

        return total_loss / max(num_batches, 1)

    def train_epoch_mnp_vtp(self, dataloader, epoch_id=0):
        """Train one epoch of MNP + VTP (Step 2)."""
        self.model.train()
        total_loss = 0
        total_mnp_loss = 0
        total_vtp_loss = 0
        num_batches = 0

        self.optimizer.zero_grad()

        pbar = tqdm(dataloader, desc=f"Epoch {epoch_id} [MNP+VTP]")
        for step, batch in enumerate(pbar):
            if self.should_stop:
                break

            batch = self._move_to_device(batch)
            mnp_labels, vtp_labels, vtp_mask, visit_types_masked = self._create_mnp_vtp_inputs(batch)

            # Update batch with masked visit types
            batch['visit_types'] = visit_types_masked

            if self.use_amp:
                with autocast('cuda'):
                    outputs = self.model.forward_mnp_vtp(
                        batch['graph_data'],
                        batch['visit_types'],
                        batch['positions'],
                        batch['ages'],
                        batch['days'],
                        batch['attention_mask'],
                        mnp_labels=mnp_labels,
                        vtp_labels=vtp_labels,
                        vtp_mask=vtp_mask,
                    )

                    loss = 0
                    if 'mnp_loss' in outputs:
                        loss = loss + outputs['mnp_loss']
                        total_mnp_loss += outputs['mnp_loss'].item()
                    if 'vtp_loss' in outputs:
                        loss = loss + outputs['vtp_loss']
                        total_vtp_loss += outputs['vtp_loss'].item()

                    if loss == 0:
                        continue

                    loss = loss / self.gradient_accumulation_steps

                self.scaler.scale(loss).backward()

                if (step + 1) % self.gradient_accumulation_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    if self.scheduler:
                        self.scheduler.step()
            else:
                outputs = self.model.forward_mnp_vtp(
                    batch['graph_data'],
                    batch['visit_types'],
                    batch['positions'],
                    batch['ages'],
                    batch['days'],
                    batch['attention_mask'],
                    mnp_labels=mnp_labels,
                    vtp_labels=vtp_labels,
                    vtp_mask=vtp_mask,
                )

                loss = 0
                if 'mnp_loss' in outputs:
                    loss = loss + outputs['mnp_loss']
                    total_mnp_loss += outputs['mnp_loss'].item()
                if 'vtp_loss' in outputs:
                    loss = loss + outputs['vtp_loss']
                    total_vtp_loss += outputs['vtp_loss'].item()

                if loss == 0:
                    continue

                loss = loss / self.gradient_accumulation_steps
                loss.backward()

                if (step + 1) % self.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    if self.scheduler:
                        self.scheduler.step()

            total_loss += loss.item() * self.gradient_accumulation_steps
            num_batches += 1
            pbar.set_postfix({
                "loss": total_loss / num_batches,
                "mnp": total_mnp_loss / num_batches,
                "vtp": total_vtp_loss / num_batches,
            })

        return {
            "loss": total_loss / max(num_batches, 1),
            "mnp_loss": total_mnp_loss / max(num_batches, 1),
            "vtp_loss": total_vtp_loss / max(num_batches, 1),
        }

    @torch.no_grad()
    def evaluate(self, dataloader, epoch_id=0, mode="mnp_vtp"):
        """Evaluate on validation set."""
        self.model.eval()
        total_loss = 0
        num_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch_id} [Val]")
        for batch in pbar:
            batch = self._move_to_device(batch)

            if mode == "nam":
                graph_data_masked, masked_indices, labels = self._create_nam_inputs(batch)
                if len(labels) == 0:
                    continue
                if self.use_amp:
                    with autocast('cuda'):
                        loss, _ = self.model.forward_nam(graph_data_masked, masked_indices, labels)
                else:
                    loss, _ = self.model.forward_nam(graph_data_masked, masked_indices, labels)
            else:
                mnp_labels, vtp_labels, vtp_mask, visit_types_masked = self._create_mnp_vtp_inputs(batch)
                batch['visit_types'] = visit_types_masked

                if self.use_amp:
                    with autocast('cuda'):
                        outputs = self.model.forward_mnp_vtp(
                            batch['graph_data'],
                            batch['visit_types'],
                            batch['positions'],
                            batch['ages'],
                            batch['days'],
                            batch['attention_mask'],
                            mnp_labels=mnp_labels,
                            vtp_labels=vtp_labels,
                            vtp_mask=vtp_mask,
                        )
                else:
                    outputs = self.model.forward_mnp_vtp(
                        batch['graph_data'],
                        batch['visit_types'],
                        batch['positions'],
                        batch['ages'],
                        batch['days'],
                        batch['attention_mask'],
                        mnp_labels=mnp_labels,
                        vtp_labels=vtp_labels,
                        vtp_mask=vtp_mask,
                    )

                loss = 0
                if 'mnp_loss' in outputs:
                    loss = loss + outputs['mnp_loss']
                if 'vtp_loss' in outputs:
                    loss = loss + outputs['vtp_loss']

                if loss == 0:
                    continue

            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({"loss": total_loss / num_batches})

        return {"loss": total_loss / max(num_batches, 1)}

    def save_checkpoint(self, epoch, val_loss, is_best=False, filename="checkpoint.pt"):
        """Save training checkpoint."""
        ckpt = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
        }
        if self.scheduler:
            ckpt['scheduler_state_dict'] = self.scheduler.state_dict()
        if self.scaler:
            ckpt['scaler_state_dict'] = self.scaler.state_dict()

        torch.save(ckpt, self.output_dir / filename)

        if is_best:
            torch.save(ckpt, self.output_dir / "best_model.pt")
            print(f"  -> Saved best model (val_loss: {val_loss:.4f})")


def main():
    args = parse_args()

    # Set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  GT-BEHRT Pre-training")
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
            except Exception:
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

    # Save tokenizer
    tokenizer.save(str(output_dir / "tokenizer.json"))

    # Split patients
    np.random.shuffle(patient_ids)
    n_train = int(len(patient_ids) * args.train_ratio)
    n_val = int(len(patient_ids) * args.val_ratio)

    train_ids = patient_ids[:n_train]
    val_ids = patient_ids[n_train:n_train + n_val]
    test_ids = patient_ids[n_train + n_val:]

    print(f"Split: Train={len(train_ids)}, Val={len(val_ids)}, Test={len(test_ids)}")

    # Create datasets
    print("\nCreating datasets...")

    common_cfg = {
        "data_path": str(data_path),
        "tokenizer": tokenizer,
        "max_visits": args.max_visits,
        "max_codes_per_visit": args.max_codes_per_visit,
        "patient_id_col": "subject_id",
        "enc_id_col": "hadm_id",
        "token_col": "code",
        "code_type_col": "code_type",
        "sort_col": "visit_seq",
        "visit_time_col": "days_since_prior_admission",
    }

    train_cohort = pd.DataFrame({"subject_id": train_ids})
    val_cohort = pd.DataFrame({"subject_id": val_ids})

    train_dataset = GTBEHRTDataset(supervised_task_cohort=train_cohort, **common_cfg)
    val_dataset = GTBEHRTDataset(supervised_task_cohort=val_cohort, **common_cfg)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    use_gpu = args.device == "cuda" and torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=use_gpu,
        collate_fn=collate_gtbehrt,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=use_gpu,
        collate_fn=collate_gtbehrt,
    )

    # Create model
    print("\nCreating model...")
    device = torch.device(args.device)

    config = create_gtbehrt_config(
        vocab_size=vocab_size,
        hidden_size=args.hidden_size,
        n_graph_layers=args.n_graph_layers,
        n_bert_layers=args.n_bert_layers,
        n_heads=args.n_heads,
        graph_dropout=args.dropout,
        bert_dropout=args.dropout,
        attention_dropout=args.dropout,
        max_visits=args.max_visits,
        max_codes_per_visit=args.max_codes_per_visit,
    )

    model = GTBEHRTForPretraining(config)
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    print(f"Device: {device}")
    print(f"AMP: {'Enabled' if args.use_amp else 'Disabled'}")

    # Save config
    with open(output_dir / "config.json", 'w') as f:
        config_dict = {
            'vocab_size': config.vocab_size,
            'hidden_size': config.hidden_size,
            'd_stream': config.d_stream,
            'n_graph_layers': config.n_graph_layers,
            'n_bert_layers': config.n_bert_layers,
            'n_bert_heads': config.n_bert_heads,
            'max_visits': config.max_visits,
            'max_codes_per_visit': config.max_codes_per_visit,
            **vars(args),
        }
        json.dump(config_dict, f, indent=2)

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)

    total_steps = len(train_loader) * args.num_epochs // args.gradient_accumulation_steps
    warmup_steps = int(total_steps * args.warmup_ratio)

    from transformers import get_linear_schedule_with_warmup
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # Trainer
    trainer = GTBEHRTTrainer(
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        nam_mask_prob=args.nam_mask_prob,
        vtp_mask_prob=args.vtp_mask_prob,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        use_amp=args.use_amp,
        output_dir=output_dir,
        patience=args.patience,
    )

    # Resume from checkpoint
    start_epoch = 0
    best_val_loss = float('inf')
    nam_complete = False

    if args.resume_from and Path(args.resume_from).exists():
        print(f"\nResuming from {args.resume_from}")
        ckpt = torch.load(args.resume_from, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_val_loss = ckpt.get('best_val_loss', float('inf'))
        nam_complete = ckpt.get('nam_complete', False)
        if scheduler and 'scheduler_state_dict' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        if trainer.scaler and 'scaler_state_dict' in ckpt:
            trainer.scaler.load_state_dict(ckpt['scaler_state_dict'])
        print(f"Resumed from epoch {start_epoch}")

    # ========================================================================
    # Step 1: NAM Pre-training (Graph Transformer only)
    # ========================================================================
    if not nam_complete and args.nam_epochs > 0:
        print("\n" + "=" * 60)
        print("Step 1: Node Attribute Masking (NAM) Pre-training")
        print("=" * 60)

        patience_counter = 0
        best_nam_loss = float('inf')

        for epoch in range(args.nam_epochs):
            if trainer.should_stop:
                break

            train_loss = trainer.train_epoch_nam(train_loader, epoch_id=epoch)
            val_metrics = trainer.evaluate(val_loader, epoch_id=epoch, mode="nam")
            val_loss = val_metrics['loss']

            print(f"NAM Epoch {epoch} | train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f}")

            is_best = val_loss < best_nam_loss
            if is_best:
                best_nam_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            trainer.save_checkpoint(epoch, val_loss, is_best, f"nam_checkpoint_{epoch}.pt")

            if patience_counter >= args.patience:
                print(f"NAM early stopping at epoch {epoch}")
                break

        # Save NAM completion state
        nam_complete = True
        print(f"\nNAM pre-training complete. Best loss: {best_nam_loss:.4f}")

    # ========================================================================
    # Step 2: MNP + VTP Pre-training (Full model)
    # ========================================================================
    print("\n" + "=" * 60)
    print("Step 2: MNP + VTP Pre-training")
    print("=" * 60)

    patience_counter = 0

    for epoch in range(start_epoch, args.num_epochs):
        if trainer.should_stop:
            break

        train_metrics = trainer.train_epoch_mnp_vtp(train_loader, epoch_id=epoch)
        val_metrics = trainer.evaluate(val_loader, epoch_id=epoch, mode="mnp_vtp")

        train_loss = train_metrics['loss']
        val_loss = val_metrics['loss']

        print(f"Epoch {epoch} | train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f}")

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % args.save_every == 0 or is_best:
            trainer.save_checkpoint(epoch, val_loss, is_best, f"checkpoint_epoch{epoch}.pt")

        # Latest checkpoint for resume
        ckpt = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'best_val_loss': best_val_loss,
            'nam_complete': True,
        }
        if scheduler:
            ckpt['scheduler_state_dict'] = scheduler.state_dict()
        if trainer.scaler:
            ckpt['scaler_state_dict'] = trainer.scaler.state_dict()
        torch.save(ckpt, output_dir / "latest_checkpoint.pt")

        if patience_counter >= args.patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break

    # Save final results
    results = {
        "model": "gt-behrt",
        "best_val_loss": best_val_loss,
        "n_params": n_params,
    }
    with open(output_dir / "results.json", 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print(f"Training complete! Model saved to {output_dir}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
