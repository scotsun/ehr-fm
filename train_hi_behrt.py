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
from torch.utils.data import DataLoader, Dataset
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
import pyarrow.parquet as pq

from tokenizers import Tokenizer

from src.finetune.data_utils import DownstreamTask, TASK_CONFIGS
from src.baselines.hi_behrt import HiBEHRTConfig, HiBEHRTSimpleForClassification


# ============================================================================
# Hi-BEHRT Native Finetune Dataset
# ============================================================================

class HiBEHRTFinetuneDataset(Dataset):
    """
    Native flat-format dataset for Hi-BEHRT finetuning.

    Produces compact flat token sequences with per-token cumulative time values,
    matching the format used during BYOL pretraining. This avoids the data format
    mismatch caused by flattening HAT's hierarchical format (which introduces
    inter-segment padding).
    """

    def __init__(
        self,
        data_path,
        labels_df: pd.DataFrame,
        tokenizer: Tokenizer,
        task: DownstreamTask,
        max_total_len: int = 2048,
        split: str = "train",
        split_ratios: tuple = (0.7, 0.15, 0.15),
        seed: int = 42,
        patient_id_col: str = "subject_id",
        enc_id_col: str = "hadm_id",
        token_col: str = "code",
        time_col: str = "days_since_prior_admission",
        sort_col: str = "admittime",
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

        self.task = task
        self.task_config = TASK_CONFIGS[task]
        self.is_multilabel = self.task_config.get("is_multilabel", False)

        self.pad_token_id = tokenizer.token_to_id("[PAD]")
        self.cls_token_id = tokenizer.token_to_id("[CLS]")

        # Label setup (same logic as FinetuneDataset)
        if self.is_multilabel:
            valid_labels = labels_df[labels_df['icd_categories'].notna() & (labels_df['icd_categories'] != '')].copy()
            self.hadm_to_categories = {}
            for _, row in valid_labels.iterrows():
                cats = [int(x) for x in row['icd_categories'].split(',')]
                self.hadm_to_categories[row[enc_id_col]] = cats
            all_indices = [idx for cats in self.hadm_to_categories.values() for idx in cats]
            self.num_classes = max(all_indices) + 1 if all_indices else 0
            self.label_mapping = None
        else:
            self.label_col = task.value
            valid_labels = labels_df[labels_df[self.label_col] >= 0].copy()

            if task == DownstreamTask.ICD_CHAPTER:
                self.num_classes = valid_labels[self.label_col].nunique()
                unique_labels = sorted(valid_labels[self.label_col].unique())
                self.label_mapping = {old: new for new, old in enumerate(unique_labels)}
            else:
                self.num_classes = 2
                self.label_mapping = None

        # Patient-level split
        all_patients = valid_labels[patient_id_col].unique()
        rng = np.random.default_rng(seed)
        all_patients = rng.permutation(all_patients)

        n_train = int(len(all_patients) * split_ratios[0])
        n_val = int(len(all_patients) * split_ratios[1])

        if split == "train":
            split_patients = set(all_patients[:n_train])
        elif split == "val":
            split_patients = set(all_patients[n_train:n_train + n_val])
        else:
            split_patients = set(all_patients[n_train + n_val:])

        self.labels = valid_labels[
            valid_labels[patient_id_col].isin(split_patients)
        ].reset_index(drop=True)

        # Build label dict and patient admissions
        self.label_dict = {}
        pid_idx = self.labels.columns.get_loc(patient_id_col)
        enc_idx = self.labels.columns.get_loc(enc_id_col)

        if self.is_multilabel:
            for row in self.labels.itertuples(index=False):
                key = (row[pid_idx], row[enc_idx])
                hadm_id = row[enc_idx]
                if hadm_id in self.hadm_to_categories:
                    self.label_dict[key] = self.hadm_to_categories[hadm_id]
        else:
            label_idx = self.labels.columns.get_loc(self.label_col)
            for row in self.labels.itertuples(index=False):
                key = (row[pid_idx], row[enc_idx])
                label = row[label_idx]
                if self.label_mapping:
                    label = self.label_mapping[label]
                self.label_dict[key] = label

        self.samples = list(self.label_dict.keys())

        self.patient_admissions = {}
        for (pid, hadm_id) in self.samples:
            if pid not in self.patient_admissions:
                self.patient_admissions[pid] = []
            self.patient_admissions[pid].append(hadm_id)

        for pid in self.patient_admissions:
            patient_labels = self.labels[self.labels[patient_id_col] == pid]
            sorted_hadms = patient_labels.sort_values(sort_col)[enc_id_col].tolist()
            self.patient_admissions[pid] = sorted_hadms

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        subject_id, target_hadm_id = self.samples[index]
        label = self.label_dict[(subject_id, target_hadm_id)]

        all_hadms = self.patient_admissions[subject_id]
        target_idx = all_hadms.index(target_hadm_id)
        history_hadms = list(all_hadms[:target_idx + 1])

        # Read patient data
        subject_dir = self.data_path / f"{self.patient_id_col}={subject_id}"
        table = pq.read_table(subject_dir, filters=[(self.enc_id_col, 'in', history_hadms)])
        history_data = table.to_pandas()

        grouped = history_data.groupby(self.enc_id_col)
        max_hours = self.task_config.get("max_hours")
        exclude_target_dx = self.task_config.get("exclude_target_dx", False)

        encounter_data = []
        for enc_id, group in grouped:
            if enc_id == target_hadm_id:
                if exclude_target_dx:
                    group = group[~group[self.token_col].str.startswith('DX:', na=False)]
                if max_hours is not None and self.token_time_col in group.columns:
                    group = group[group[self.token_time_col] <= max_hours]

            tokens = group[self.token_col].tolist()
            if len(tokens) == 0:
                continue

            time_val = group[self.time_col].iloc[0] if self.time_col and self.time_col in group.columns else 0.0
            sort_val = group[self.sort_col].iloc[0] if self.sort_col and self.sort_col in group.columns else 0

            token_times = group[self.token_time_col].tolist() if self.token_time_col and self.token_time_col in group.columns else [0.0] * len(tokens)

            encounter_data.append({
                'tokens': tokens,
                'time': time_val if not (time_val is None or (isinstance(time_val, float) and np.isnan(time_val))) else 0.0,
                'sort_key': sort_val,
                'token_times': token_times,
            })

        encounter_data.sort(key=lambda x: x['sort_key'] if x['sort_key'] is not None else 0)

        # Compute cumulative time
        cumsum_time = 0.0
        for enc in encounter_data:
            cumsum_time += enc['time']
            enc['cumsum_time'] = cumsum_time

        # Build flat token sequence with per-token time (same as BYOLPretrainDataset)
        all_tokens = []
        all_times = []

        for enc in encounter_data:
            all_tokens.append("[CLS]")
            all_times.append(enc['cumsum_time'])

            for tok, tok_time in zip(enc['tokens'], enc['token_times']):
                all_tokens.append(tok)
                tok_time_clean = tok_time if not (tok_time is None or (isinstance(tok_time, float) and np.isnan(tok_time))) else 0.0
                all_times.append(enc['cumsum_time'] + tok_time_clean / 24.0)

        # Truncate (keep most recent)
        if len(all_tokens) > self.max_total_len:
            all_tokens = all_tokens[-self.max_total_len:]
            all_times = all_times[-self.max_total_len:]

        # Tokenize
        encoding = self.tokenizer.encode(all_tokens, is_pretokenized=True)
        input_ids = encoding.ids
        attention_mask = [1 if tid != self.pad_token_id else 0 for tid in input_ids]

        # Align time values
        if len(all_times) < len(input_ids):
            last_time = all_times[-1] if all_times else 0.0
            all_times = all_times + [last_time] * (len(input_ids) - len(all_times))
        else:
            all_times = all_times[:len(input_ids)]

        item = {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'time_values': torch.tensor(all_times, dtype=torch.float),
        }

        if self.is_multilabel:
            multi_hot = torch.zeros(self.num_classes, dtype=torch.float)
            for idx in label:
                multi_hot[idx] = 1.0
            item["label"] = multi_hot
        else:
            item["label"] = torch.tensor(label, dtype=torch.long)

        return item

    def get_class_weights(self):
        """Get inverse frequency weights for imbalanced data."""
        if self.is_multilabel:
            class_counts = torch.zeros(self.num_classes)
            for s in self.samples:
                for idx in self.label_dict[s]:
                    class_counts[idx] += 1
            neg_counts = len(self.samples) - class_counts
            return (neg_counts / (class_counts + 1e-6)).float()
        else:
            labels = torch.tensor([self.label_dict[s] for s in self.samples])
            class_counts = torch.bincount(labels, minlength=self.num_classes).float()
            weights = 1.0 / (class_counts + 1e-6)
            weights = weights / weights.sum() * self.num_classes
            return weights


class HiBEHRTNextVisitDataset(Dataset):
    """
    Native flat-format dataset for Hi-BEHRT next visit prediction.

    Input: history admissions (NOT including target)
    Label: multi-hot vector of token IDs from target admission
    """

    def __init__(
        self,
        data_path,
        labels_df: pd.DataFrame,
        tokenizer: Tokenizer,
        max_total_len: int = 2048,
        split: str = "train",
        split_ratios: tuple = (0.7, 0.15, 0.15),
        seed: int = 42,
        patient_id_col: str = "subject_id",
        enc_id_col: str = "hadm_id",
        token_col: str = "code",
        time_col: str = "days_since_prior_admission",
        sort_col: str = "admittime",
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

        self.pad_token_id = tokenizer.token_to_id("[PAD]")
        self.vocab_size = tokenizer.get_vocab_size()
        self.num_classes = self.vocab_size

        # Patient-level split
        all_patients = labels_df[patient_id_col].unique()
        rng = np.random.default_rng(seed)
        all_patients = rng.permutation(all_patients)

        n_train = int(len(all_patients) * split_ratios[0])
        n_val = int(len(all_patients) * split_ratios[1])

        if split == "train":
            split_patients = set(all_patients[:n_train])
        elif split == "val":
            split_patients = set(all_patients[n_train:n_train + n_val])
        else:
            split_patients = set(all_patients[n_train + n_val:])

        labels = labels_df[labels_df[patient_id_col].isin(split_patients)].reset_index(drop=True)

        self.patient_admissions = {}
        for pid in labels[patient_id_col].unique():
            patient_data = labels[labels[patient_id_col] == pid]
            sorted_hadms = patient_data.sort_values(sort_col)[enc_id_col].tolist()
            if len(sorted_hadms) >= 2:
                self.patient_admissions[pid] = sorted_hadms

        self.samples = []
        for pid, hadms in self.patient_admissions.items():
            for i in range(1, len(hadms)):
                self.samples.append((pid, hadms[i]))

        print(f"HiBEHRTNextVisitDataset [{split}]: {len(self.samples)} samples from {len(self.patient_admissions)} patients")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        subject_id, target_hadm_id = self.samples[index]

        all_hadms = self.patient_admissions[subject_id]
        target_idx = all_hadms.index(target_hadm_id)
        history_hadms = list(all_hadms[:target_idx])

        subject_dir = self.data_path / f"{self.patient_id_col}={subject_id}"

        # Read history
        history_table = pq.read_table(subject_dir, filters=[(self.enc_id_col, 'in', history_hadms)])
        history_data = history_table.to_pandas()

        # Read target for labels
        target_table = pq.read_table(subject_dir, filters=[(self.enc_id_col, '==', target_hadm_id)])
        target_data = target_table.to_pandas()

        # Process history encounters
        grouped = history_data.groupby(self.enc_id_col)
        encounter_data = []
        for enc_id, group in grouped:
            tokens = group[self.token_col].tolist()
            if len(tokens) == 0:
                continue
            time_val = group[self.time_col].iloc[0] if self.time_col and self.time_col in group.columns else 0.0
            sort_val = group[self.sort_col].iloc[0] if self.sort_col and self.sort_col in group.columns else 0
            token_times = group[self.token_time_col].tolist() if self.token_time_col and self.token_time_col in group.columns else [0.0] * len(tokens)
            encounter_data.append({
                'tokens': tokens,
                'time': time_val if not (time_val is None or (isinstance(time_val, float) and np.isnan(time_val))) else 0.0,
                'sort_key': sort_val,
                'token_times': token_times,
            })

        encounter_data.sort(key=lambda x: x['sort_key'] if x['sort_key'] is not None else 0)

        # Compute cumulative time
        cumsum_time = 0.0
        for enc in encounter_data:
            cumsum_time += enc['time']
            enc['cumsum_time'] = cumsum_time

        # Build flat token sequence
        all_tokens = []
        all_times = []
        for enc in encounter_data:
            all_tokens.append("[CLS]")
            all_times.append(enc['cumsum_time'])
            for tok, tok_time in zip(enc['tokens'], enc['token_times']):
                all_tokens.append(tok)
                tok_time_clean = tok_time if not (tok_time is None or (isinstance(tok_time, float) and np.isnan(tok_time))) else 0.0
                all_times.append(enc['cumsum_time'] + tok_time_clean / 24.0)

        if len(all_tokens) > self.max_total_len:
            all_tokens = all_tokens[-self.max_total_len:]
            all_times = all_times[-self.max_total_len:]

        # Handle empty history
        if len(all_tokens) == 0:
            all_tokens = ["[CLS]"]
            all_times = [0.0]

        encoding = self.tokenizer.encode(all_tokens, is_pretokenized=True)
        input_ids = encoding.ids
        attention_mask = [1 if tid != self.pad_token_id else 0 for tid in input_ids]

        if len(all_times) < len(input_ids):
            last_time = all_times[-1] if all_times else 0.0
            all_times = all_times + [last_time] * (len(input_ids) - len(all_times))
        else:
            all_times = all_times[:len(input_ids)]

        # Label: multi-hot vector of target admission tokens
        target_tokens = target_data[self.token_col].tolist()
        target_token_ids = set()
        for token in target_tokens:
            token_id = self.tokenizer.token_to_id(str(token))
            if token_id is not None and token_id >= 4:  # Skip special tokens
                target_token_ids.add(token_id)

        multi_hot = torch.zeros(self.vocab_size, dtype=torch.float)
        for tid in target_token_ids:
            multi_hot[tid] = 1.0

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'time_values': torch.tensor(all_times, dtype=torch.float),
            'label': multi_hot,
        }


def collate_hibehrt(batch):
    """Collate function for Hi-BEHRT - pads variable-length flat sequences to max length in batch."""
    max_len = max(item['input_ids'].shape[0] for item in batch)

    input_ids = []
    attention_mask = []
    time_values = []
    labels = []

    for item in batch:
        seq_len = item['input_ids'].shape[0]
        pad_len = max_len - seq_len

        if pad_len > 0:
            input_ids.append(F.pad(item['input_ids'], (0, pad_len), value=0))
            attention_mask.append(F.pad(item['attention_mask'], (0, pad_len), value=0))
            last_time = item['time_values'][-1].item() if seq_len > 0 else 0.0
            time_values.append(F.pad(item['time_values'], (0, pad_len), value=last_time))
        else:
            input_ids.append(item['input_ids'])
            attention_mask.append(item['attention_mask'])
            time_values.append(item['time_values'])

        labels.append(item['label'])

    return {
        'input_ids': torch.stack(input_ids),
        'attention_mask': torch.stack(attention_mask),
        'time_values': torch.stack(time_values),
        'label': torch.stack(labels),
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

        # Data loaders (use native flat collate)
        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            collate_fn=collate_hibehrt, num_workers=num_workers,
            pin_memory=(self.device.type == "cuda")
        )
        self.val_loader = DataLoader(
            val_dataset, batch_size=batch_size * 2, shuffle=False,
            collate_fn=collate_hibehrt, num_workers=num_workers,
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
        """Forward pass with native flat batch."""
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = batch["label"].to(self.device)
        time_values = batch["time_values"].to(self.device)

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

                    class_auprcs = []
                    for c in range(num_classes):
                        binary_labels = (all_labels == c).astype(int)
                        if binary_labels.sum() > 0 and binary_labels.sum() < len(binary_labels):
                            class_auprcs.append(average_precision_score(binary_labels, all_probs[:, c]))
                    metrics["auprc_macro"] = np.mean(class_auprcs) if class_auprcs else 0.0
                except ValueError as e:
                    print(f"  Warning: Could not compute AUROC/AUPRC: {e}")
                    metrics["auroc"] = 0.0
                    metrics["auprc_macro"] = 0.0
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

    # Dataset config (native flat format, matching BYOL pretrain)
    parser.add_argument("--max_total_len", type=int, default=2048,
                       help="Max flat sequence length (same as BYOL pretrain)")

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

    # Create datasets (native flat format matching BYOL pretrain)
    task_config = TASK_CONFIGS[task]
    is_set_prediction = task_config.get("is_set_prediction", False)

    if is_set_prediction:
        print("\nCreating datasets (using HiBEHRTNextVisitDataset)...")
        train_dataset = HiBEHRTNextVisitDataset(
            data_path=args.data_path,
            labels_df=labels_df,
            tokenizer=tokenizer,
            max_total_len=args.max_total_len,
            split="train",
            seed=args.seed,
        )
        val_dataset = HiBEHRTNextVisitDataset(
            data_path=args.data_path,
            labels_df=labels_df,
            tokenizer=tokenizer,
            max_total_len=args.max_total_len,
            split="val",
            seed=args.seed,
        )
        test_dataset = HiBEHRTNextVisitDataset(
            data_path=args.data_path,
            labels_df=labels_df,
            tokenizer=tokenizer,
            max_total_len=args.max_total_len,
            split="test",
            seed=args.seed,
        )
    else:
        print("\nCreating datasets (using HiBEHRTFinetuneDataset)...")
        train_dataset = HiBEHRTFinetuneDataset(
            data_path=args.data_path,
            labels_df=labels_df,
            tokenizer=tokenizer,
            task=task,
            max_total_len=args.max_total_len,
            split="train",
            seed=args.seed,
        )
        val_dataset = HiBEHRTFinetuneDataset(
            data_path=args.data_path,
            labels_df=labels_df,
            tokenizer=tokenizer,
            task=task,
            max_total_len=args.max_total_len,
            split="val",
            seed=args.seed,
        )
        test_dataset = HiBEHRTFinetuneDataset(
            data_path=args.data_path,
            labels_df=labels_df,
            tokenizer=tokenizer,
            task=task,
            max_total_len=args.max_total_len,
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
        collate_fn=collate_hibehrt, num_workers=args.num_workers
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
