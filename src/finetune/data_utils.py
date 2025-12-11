"""
Fine-tuning Dataset for HAT downstream tasks.

Key differences from pre-training dataset:
- Admission-level (not patient-level) for per-admission predictions
- Returns labels along with features
- Supports multiple tasks with task-specific filtering
"""

import torch
from torch.utils.data import Dataset
from tokenizers import Tokenizer
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq
import numpy as np
from enum import Enum

from src.tokenizer import get_batch_encoding, SPECIAL_TOKEN_IDS


class DownstreamTask(Enum):
    MORTALITY = "mortality"
    READMISSION_30D = "readmission_30d"
    PROLONGED_LOS = "prolonged_los"
    ICD_CHAPTER = "icd_chapter"
    ABNORMAL_LAB = "abnormal_lab"


# Task-specific configuration to avoid data leakage
TASK_CONFIGS = {
    # Mortality: predict at admission time, use ONLY previous admissions
    # (current admission data would leak outcome)
    DownstreamTask.MORTALITY: {
        "prediction_time": "admission",  # Only use data BEFORE target admission
        "include_current": False,        # Exclude current admission entirely
    },
    # Prolonged LoS: same as mortality - predict at admission
    DownstreamTask.PROLONGED_LOS: {
        "prediction_time": "admission",
        "include_current": False,
    },
    # Readmission: predict at discharge, use full admission history
    DownstreamTask.READMISSION_30D: {
        "prediction_time": "discharge",  # Use all data up to discharge
        "include_current": True,         # Include current admission
    },
    # ICD Chapter: predict diagnosis at discharge, use full history
    DownstreamTask.ICD_CHAPTER: {
        "prediction_time": "discharge",
        "include_current": True,
    },
    # Abnormal Lab: depends on use case
    DownstreamTask.ABNORMAL_LAB: {
        "prediction_time": "admission",
        "include_current": False,
    },
}


class FinetuneDataset(Dataset):
    """
    Fine-tuning dataset for HAT downstream tasks.

    Key features:
    - Admission-level: each sample is one admission (hadm_id)
    - Uses patient history up to the target admission
    - Returns labels for the specified task

    Args:
        data_path: Path to Hive-partitioned parquet directory
        labels_df: DataFrame with columns [subject_id, hadm_id, task_label_columns...]
        tokenizer: Tokenizer instance
        task: DownstreamTask enum specifying which task
        max_seg: Maximum number of segments (admissions)
        max_seq_len: Maximum sequence length per segment
        split: 'train', 'val', or 'test'
        split_ratios: (train, val, test) ratios, default (0.7, 0.15, 0.15)
        seed: Random seed for reproducible splits
    """

    def __init__(
        self,
        data_path,
        labels_df: pd.DataFrame,
        tokenizer: Tokenizer,
        task: DownstreamTask,
        max_seg: int = 32,
        max_seq_len: int = 512,
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
        self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]", length=max_seq_len)
        self.tokenizer.enable_truncation(max_length=max_seq_len)

        self.task = task
        self.max_seg = max_seg
        self.max_seq_len = max_seq_len
        self.patient_id_col = patient_id_col
        self.enc_id_col = enc_id_col
        self.token_col = token_col
        self.time_col = time_col
        self.sort_col = sort_col
        self.token_time_col = token_time_col

        # Get task-specific label column
        self.label_col = task.value

        # Filter labels for this task (remove invalid labels)
        valid_labels = labels_df[labels_df[self.label_col] >= 0].copy()

        # For ICD chapter, we have multi-class; for others, binary
        if task == DownstreamTask.ICD_CHAPTER:
            self.num_classes = valid_labels[self.label_col].nunique()
            # Create label mapping (some chapters might be missing)
            unique_labels = sorted(valid_labels[self.label_col].unique())
            self.label_mapping = {old: new for new, old in enumerate(unique_labels)}
        else:
            self.num_classes = 2
            self.label_mapping = None

        # Patient-level split (not admission-level)
        # Use local RNG to avoid affecting global random state
        all_patients = valid_labels[patient_id_col].unique()
        rng = np.random.default_rng(seed)
        all_patients = rng.permutation(all_patients)

        n_train = int(len(all_patients) * split_ratios[0])
        n_val = int(len(all_patients) * split_ratios[1])

        if split == "train":
            split_patients = set(all_patients[:n_train])
        elif split == "val":
            split_patients = set(all_patients[n_train:n_train + n_val])
        else:  # test
            split_patients = set(all_patients[n_train + n_val:])

        # Filter to split
        self.labels = valid_labels[
            valid_labels[patient_id_col].isin(split_patients)
        ].reset_index(drop=True)

        # Build index: (subject_id, hadm_id) -> label
        self.label_dict = {}
        for _, row in self.labels.iterrows():
            key = (row[patient_id_col], row[enc_id_col])
            label = row[self.label_col]
            if self.label_mapping:
                label = self.label_mapping[label]
            self.label_dict[key] = label

        # Build sample index (each sample = one admission)
        self.samples = list(self.label_dict.keys())

        # Group labels by patient for efficient lookup
        self.patient_admissions = {}
        for (pid, hadm_id) in self.samples:
            if pid not in self.patient_admissions:
                self.patient_admissions[pid] = []
            self.patient_admissions[pid].append(hadm_id)

        # Sort admissions by admittime for each patient
        for pid in self.patient_admissions:
            patient_labels = self.labels[self.labels[patient_id_col] == pid]
            sorted_hadms = patient_labels.sort_values('admittime')[enc_id_col].tolist()
            self.patient_admissions[pid] = sorted_hadms

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        """
        Get one admission's data and label.

        Prediction timing depends on task:
        - MORTALITY/PROLONGED_LOS: use only PREVIOUS admissions (predict at admission)
        - READMISSION/ICD_CHAPTER: use all history including current (predict at discharge)
        """
        subject_id, target_hadm_id = self.samples[index]
        label = self.label_dict[(subject_id, target_hadm_id)]

        # Get task-specific configuration
        task_config = TASK_CONFIGS[self.task]
        include_current = task_config["include_current"]

        # Load patient data
        subject_dir = self.data_path / f"{self.patient_id_col}={subject_id}"
        table = pq.read_table(subject_dir)
        patient_data = table.to_pandas()

        # Get admissions based on prediction timing
        all_hadms = self.patient_admissions[subject_id]
        target_idx = all_hadms.index(target_hadm_id)

        if include_current:
            # Include current admission (for readmission/ICD prediction at discharge)
            history_hadms = set(all_hadms[:target_idx + 1])
        else:
            # Exclude current admission (for mortality/los prediction at admission)
            # This prevents data leakage from current admission's labs/meds
            history_hadms = set(all_hadms[:target_idx])

            # If no previous admissions, use empty history
            # The model will need to handle this case gracefully
            if not history_hadms:
                # For patients with only one admission, we have no history
                # Return minimal data (just padding)
                return self._get_empty_item(label)

        # Filter to history admissions
        history_data = patient_data[patient_data[self.enc_id_col].isin(history_hadms)]

        # Group by admission
        grouped = history_data.groupby(self.enc_id_col)

        # Extract encounter data
        encounter_data = []
        for enc_id, group in grouped:
            tokens = group[self.token_col].tolist()
            time_val = group[self.time_col].iloc[0] if self.time_col and self.time_col in group.columns else None
            sort_val = group[self.sort_col].iloc[0] if self.sort_col and self.sort_col in group.columns else None
            token_times = group[self.token_time_col].tolist() if self.token_time_col and self.token_time_col in group.columns else None
            encounter_data.append({
                'enc_id': enc_id,
                'tokens': tokens,
                'time': time_val,
                'sort_key': sort_val,
                'token_times': token_times,
            })

        # Sort by admission time
        encounter_data.sort(key=lambda x: x['sort_key'] if x['sort_key'] is not None else 0)

        # Take last max_seg encounters (most recent history)
        if len(encounter_data) > self.max_seg:
            encounter_data = encounter_data[-self.max_seg:]

        # Extract tokens and times
        _tokens = [enc['tokens'] for enc in encounter_data]

        if self.time_col:
            _times = [enc['time'] for enc in encounter_data]
            _times, _ = self._pad_segment_time(_times, do_cumsum=True)
        else:
            _times = None

        if self.token_time_col:
            _token_times = [enc['token_times'] for enc in encounter_data]
        else:
            _token_times = None

        _tokens, _seg_attn = self._pad_segment(_tokens)
        _tokens = [["[CLS]"] + elem for elem in _tokens]

        item = get_batch_encoding(self.tokenizer, _tokens)
        item["segment_attention_mask"] = _seg_attn

        if _times is not None:
            item["segment_time"] = _times

        if _token_times is not None:
            item["token_time"] = self._pad_token_time(_token_times)

        # Add label
        item["label"] = torch.tensor(label, dtype=torch.long)

        return item

    def _get_empty_item(self, label):
        """
        Return an item with empty/padded data for patients with no history.
        This happens for mortality/los tasks when it's the patient's first admission.
        """
        # Create all-padding tensors
        _tokens = [["[CLS]", "[PAD]"]] * 1  # Single segment with just CLS
        _tokens, _seg_attn = self._pad_segment(_tokens)

        item = get_batch_encoding(self.tokenizer, _tokens)
        item["segment_attention_mask"] = _seg_attn

        # Add zero times
        item["segment_time"] = torch.zeros(self.max_seg, dtype=torch.float32)
        item["token_time"] = torch.zeros(self.max_seg, self.max_seq_len, dtype=torch.float32)

        # Add label
        item["label"] = torch.tensor(label, dtype=torch.long)

        return item

    def _pad_segment(self, tokens):
        if len(tokens) > self.max_seg:
            seg_attn = torch.ones(self.max_seg)
            tokens = tokens[-self.max_seg:]  # Take most recent
        else:
            seg_attn = torch.cat([
                torch.ones(len(tokens)),
                torch.zeros(self.max_seg - len(tokens))
            ])
            tokens = tokens + [["[PAD]"]] * (self.max_seg - len(tokens))
        return tokens, seg_attn

    def _pad_segment_time(self, times, do_cumsum=True):
        times_clean = [0.0 if (t is None or (isinstance(t, float) and np.isnan(t))) else float(t) for t in times]
        if do_cumsum:
            times_cumsum = np.cumsum(times_clean).tolist()
        else:
            times_cumsum = times_clean

        if len(times_cumsum) > self.max_seg:
            times_final = times_cumsum[-self.max_seg:]
        else:
            last_time = times_cumsum[-1] if times_cumsum else 0.0
            times_final = times_cumsum + [last_time] * (self.max_seg - len(times_cumsum))

        return torch.tensor(times_final, dtype=torch.float32), None

    def _pad_token_time(self, token_times_list):
        if len(token_times_list) > self.max_seg:
            token_times_list = token_times_list[-self.max_seg:]
        else:
            token_times_list = token_times_list + [[]] * (self.max_seg - len(token_times_list))

        padded_times = []
        for token_times in token_times_list:
            times_clean = [
                0.0 if (t is None or (isinstance(t, float) and np.isnan(t))) else float(t)
                for t in (token_times or [])
            ]

            if times_clean:
                times_clean = [0.0] + times_clean  # [CLS] token
            else:
                times_clean = [0.0]

            if len(times_clean) > self.max_seq_len:
                times_clean = times_clean[:self.max_seq_len]
            else:
                pad_val = times_clean[-1] if times_clean else 0.0
                times_clean = times_clean + [pad_val] * (self.max_seq_len - len(times_clean))

            padded_times.append(times_clean)

        return torch.tensor(padded_times, dtype=torch.float32)

    def get_class_weights(self):
        """Get class weights for handling imbalanced data."""
        labels = torch.tensor([self.label_dict[s] for s in self.samples])
        class_counts = torch.bincount(labels, minlength=self.num_classes).float()
        weights = 1.0 / (class_counts + 1e-6)
        weights = weights / weights.sum() * self.num_classes
        return weights


def collate_finetune(batch):
    """Collate function for fine-tuning dataset."""
    # Stack all tensors
    result = {}
    for key in batch[0].keys():
        if key == "label":
            result[key] = torch.stack([item[key] for item in batch])
        elif isinstance(batch[0][key], torch.Tensor):
            result[key] = torch.stack([item[key] for item in batch])
        else:
            result[key] = [item[key] for item in batch]
    return result


def create_patient_splits(labels_path: str, output_dir: str, seed: int = 42):
    """
    Create and save patient-level train/val/test splits.

    This ensures the same patients are used for all tasks.
    """
    labels = pd.read_csv(labels_path)
    patients = labels['subject_id'].unique()

    # Use local RNG to avoid affecting global random state
    rng = np.random.default_rng(seed)
    patients = rng.permutation(patients)

    n_train = int(len(patients) * 0.7)
    n_val = int(len(patients) * 0.15)

    splits = {
        'train': patients[:n_train],
        'val': patients[n_train:n_train + n_val],
        'test': patients[n_train + n_val:]
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, patient_ids in splits.items():
        split_df = pd.DataFrame({'subject_id': patient_ids})
        split_df.to_csv(output_dir / f"{split_name}_patients.csv", index=False)
        print(f"{split_name}: {len(patient_ids)} patients")

    return splits
