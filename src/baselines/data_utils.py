"""
Data utilities for baseline models.

CORE-BEHRT and other flat models need (batch, seq_len) format,
unlike HAT which uses (batch, max_seg, max_seq_len) hierarchical format.
"""

import torch
import numpy as np
from torch.utils.data import Dataset
from tokenizers import Tokenizer
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq

from src.tokenizer import SPECIAL_TOKEN_IDS


# Token type mapping for HEART
# 0=PAD, 1=CLS, 2=SEP, 3=MASK, 4=DX, 5=PR, 6=LAB, 7=MED
CODE_TYPE_TO_TOKEN_TYPE = {
    "diagnosis": 4,
    "procedure": 5,
    "lab": 6,
    "medication": 7,
}


class FlatEHRDataset(Dataset):
    """
    Flat EHR Dataset for baseline models like CORE-BEHRT and HEART.

    Concatenates all visits/encounters into a single sequence,
    with [CLS] token at the beginning of each visit.

    Time handling (aligned with HAT):
    - days_since_prior_admission: cumsum to get absolute visit time (days)
    - time_offset_hours: token-level relative time within visit
    - Final t = visit_abs_time (days) + time_offset_hours / 24

    Input: Hive-partitioned parquet directory
    Output: (seq_len,) flat sequence per patient
    """
    def __init__(
        self,
        data_path: str | Path,
        tokenizer: Tokenizer,
        supervised_task_cohort=None,
        max_seq_len: int = 2048,
        patient_id_col: str = "subject_id",
        enc_id_col: str = "hadm_id",
        token_col: str = "code",
        code_type_col: str = "code_type",  # for HEART token types
        sort_col: str = "visit_seq",
        token_time_col: str = "time_offset_hours",
        visit_time_col: str = "days_since_prior_admission",
        include_token_types: bool = False,  # for HEART
        include_visit_ids: bool = False,  # for HEART
    ):
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.patient_id_col = patient_id_col
        self.enc_id_col = enc_id_col
        self.token_col = token_col
        self.code_type_col = code_type_col
        self.sort_col = sort_col
        self.token_time_col = token_time_col
        self.visit_time_col = visit_time_col
        self.include_token_types = include_token_types
        self.include_visit_ids = include_visit_ids

        # Scan patient directories
        self.subject_dirs = sorted(self.data_path.glob(f"{patient_id_col}=*"))
        if not self.subject_dirs:
            self.subject_dirs = sorted(self.data_path.glob("subject_id=*"))

        self.patient_ids = [d.name.split('=')[1] for d in self.subject_dirs]

        # Filter by cohort if provided
        if supervised_task_cohort is not None:
            cohort_ids = set(supervised_task_cohort[patient_id_col].astype(str).tolist())
            filtered = [(d, pid) for d, pid in zip(self.subject_dirs, self.patient_ids) if pid in cohort_ids]
            self.subject_dirs = [d for d, _ in filtered]
            self.patient_ids = [pid for _, pid in filtered]

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, index):
        """
        Get a patient's EHR as a flat sequence.

        Returns:
            dict with keys:
                - input_ids: (seq_len,)
                - attention_mask: (seq_len,)
                - t: (seq_len,) absolute time values (days)
                - token_types: (seq_len,) token type IDs (if include_token_types=True)
                - visit_ids: (seq_len,) visit indices (if include_visit_ids=True)
        """
        subject_dir = self.subject_dirs[index]

        # Read needed columns
        needed_cols = [self.enc_id_col, self.token_col]
        if self.sort_col:
            needed_cols.append(self.sort_col)
        if self.token_time_col:
            needed_cols.append(self.token_time_col)
        if self.visit_time_col:
            needed_cols.append(self.visit_time_col)
        if self.include_token_types and self.code_type_col:
            needed_cols.append(self.code_type_col)

        table = pq.read_table(subject_dir, columns=needed_cols)
        df = table.to_pandas()

        # Sort by encounter/visit
        if self.sort_col and self.sort_col in df.columns:
            df = df.sort_values(by=self.sort_col)

        # Group by encounter and concatenate
        grouped = df.groupby(self.enc_id_col, sort=False)

        all_tokens = []
        all_times = []
        all_token_types = []  # for HEART
        all_visit_ids = []  # for HEART

        # Compute visit absolute times via cumsum
        visit_relative_times = []
        for enc_id, group in grouped:
            # Get days_since_prior_admission for this visit (take first value)
            if self.visit_time_col and self.visit_time_col in group.columns:
                rel_time = group[self.visit_time_col].iloc[0]
                rel_time = 0.0 if pd.isna(rel_time) else float(rel_time)
            else:
                rel_time = 0.0
            visit_relative_times.append(rel_time)

        # Cumsum to get absolute visit times (days since first visit)
        visit_abs_times = np.cumsum(visit_relative_times).tolist()

        # Now iterate again to build tokens and times
        for visit_idx, (enc_id, group) in enumerate(grouped):
            visit_abs_time = visit_abs_times[visit_idx]  # days

            # Each encounter starts with [CLS]
            tokens = group[self.token_col].tolist()
            all_tokens.append("[CLS]")
            all_tokens.extend(tokens)

            # Token-level times: visit_abs_time + token_offset_hours / 24
            if self.token_time_col and self.token_time_col in group.columns:
                token_offsets = group[self.token_time_col].fillna(0.0).tolist()
                # Convert hours to days and add to visit absolute time
                token_abs_times = [visit_abs_time + (h / 24.0) for h in token_offsets]
                all_times.append(visit_abs_time)  # [CLS] gets visit time
                all_times.extend(token_abs_times)
            else:
                # No token time, use visit time for all tokens
                all_times.append(visit_abs_time)
                all_times.extend([visit_abs_time] * len(tokens))

            # Token types for HEART (0=PAD, 1=CLS, 4=DX, 5=PR, 6=LAB, 7=MED)
            if self.include_token_types:
                all_token_types.append(1)  # [CLS] token type
                if self.code_type_col and self.code_type_col in group.columns:
                    code_types = group[self.code_type_col].tolist()
                    token_type_ids = [CODE_TYPE_TO_TOKEN_TYPE.get(ct, 0) for ct in code_types]
                    all_token_types.extend(token_type_ids)
                else:
                    all_token_types.extend([0] * len(tokens))

            # Visit IDs for HEART
            if self.include_visit_ids:
                all_visit_ids.append(visit_idx)  # [CLS] visit id
                all_visit_ids.extend([visit_idx] * len(tokens))

        # Truncate to max_seq_len
        all_tokens = all_tokens[:self.max_seq_len]
        all_times = all_times[:self.max_seq_len]
        if self.include_token_types:
            all_token_types = all_token_types[:self.max_seq_len]
        if self.include_visit_ids:
            all_visit_ids = all_visit_ids[:self.max_seq_len]

        # Encode tokens
        encoding = self.tokenizer.encode(all_tokens, is_pretokenized=True)
        input_ids = encoding.ids[:self.max_seq_len]

        # Pad to max_seq_len
        seq_len = len(input_ids)
        if seq_len < self.max_seq_len:
            pad_len = self.max_seq_len - seq_len
            input_ids = input_ids + [0] * pad_len
            all_times = all_times + [0.0] * pad_len
            if self.include_token_types:
                all_token_types = all_token_types + [0] * pad_len  # 0 = PAD type
            if self.include_visit_ids:
                all_visit_ids = all_visit_ids + [0] * pad_len

        # Create attention mask
        attention_mask = [1] * seq_len + [0] * (self.max_seq_len - seq_len)

        result = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.bool),
            "t": torch.tensor(all_times, dtype=torch.float32),
        }

        if self.include_token_types:
            result["token_types"] = torch.tensor(all_token_types, dtype=torch.long)
        if self.include_visit_ids:
            result["visit_ids"] = torch.tensor(all_visit_ids, dtype=torch.long)

        return result


def get_sample_patient_ids(data_folder: str | Path, n_patients: int = 30):
    """Get a sample of patient IDs from the parquet directory."""
    patient_dirs = list(Path(data_folder).glob("subject_id=*"))[:n_patients]
    patient_ids = [int(d.name.split("=")[1]) for d in patient_dirs]
    return patient_ids


# ============================================================================
# Downstream Task Configuration (same as HAT)
# ============================================================================

from enum import Enum


class DownstreamTask(Enum):
    MORTALITY = "mortality"
    READMISSION_30D = "readmission_30d"
    PROLONGED_LOS = "prolonged_los"
    ICD_CHAPTER = "icd_chapter"
    ICD_CATEGORY_MULTILABEL = "icd_category_multilabel"
    NEXT_VISIT = "next_visit"


# max_hours: time window for target admission; exclude_target_dx: prevent data leakage
TASK_CONFIGS = {
    DownstreamTask.MORTALITY: {"prediction_time": "admission_24h", "max_hours": 24, "exclude_target_dx": True},
    DownstreamTask.PROLONGED_LOS: {"prediction_time": "admission_48h", "max_hours": 48, "exclude_target_dx": True},
    DownstreamTask.READMISSION_30D: {"prediction_time": "discharge", "max_hours": None, "exclude_target_dx": False},
    DownstreamTask.ICD_CHAPTER: {"prediction_time": "discharge", "max_hours": None, "exclude_target_dx": True},
    DownstreamTask.ICD_CATEGORY_MULTILABEL: {"prediction_time": "discharge", "max_hours": None, "exclude_target_dx": True, "is_multilabel": True},
    DownstreamTask.NEXT_VISIT: {"prediction_time": "discharge", "include_target": False, "is_set_prediction": True},
}


class FlatFinetuneDataset(Dataset):
    """
    Flat EHR Dataset for baseline model fine-tuning.

    Same flat (seq_len,) format as FlatEHRDataset for pretrain,
    but with downstream task labels and task-specific data filtering.

    Key differences from HAT's FinetuneDataset:
    - Outputs flat (max_seq_len,) sequences instead of hierarchical (max_seg, max_seq_len)
    - Consistent with baseline pretrain format
    """

    def __init__(
        self,
        data_path: str | Path,
        labels_df: pd.DataFrame,
        tokenizer: Tokenizer,
        task: DownstreamTask,
        max_seq_len: int = 2048,
        split: str = "train",
        split_ratios: tuple = (0.7, 0.15, 0.15),
        seed: int = 42,
        patient_id_col: str = "subject_id",
        enc_id_col: str = "hadm_id",
        token_col: str = "code",
        code_type_col: str = "code_type",
        sort_col: str = "admittime",
        token_time_col: str = "time_offset_hours",
        visit_time_col: str = "days_since_prior_admission",
        include_token_types: bool = False,  # for HEART
        include_visit_ids: bool = False,  # for HEART
    ):
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.task = task
        self.task_config = TASK_CONFIGS[task]
        self.max_seq_len = max_seq_len
        self.patient_id_col = patient_id_col
        self.enc_id_col = enc_id_col
        self.token_col = token_col
        self.code_type_col = code_type_col
        self.sort_col = sort_col
        self.token_time_col = token_time_col
        self.visit_time_col = visit_time_col
        self.include_token_types = include_token_types
        self.include_visit_ids = include_visit_ids

        # Setup label handling
        self.is_multilabel = self.task_config.get("is_multilabel", False)

        if task == DownstreamTask.ICD_CATEGORY_MULTILABEL:
            # Multilabel ICD category prediction
            self.label_col = "icd_categories"
            valid_labels = labels_df[labels_df['icd_categories'].notna() & (labels_df['icd_categories'] != '')].copy()
            # Parse icd_categories to find all unique category IDs
            all_categories = set()
            self.hadm_to_categories = {}
            for _, row in valid_labels.iterrows():
                cats = [int(x) for x in str(row['icd_categories']).split(',')]
                self.hadm_to_categories[row[enc_id_col]] = cats
                all_categories.update(cats)
            self.num_classes = max(all_categories) + 1  # Category IDs are 0-indexed
            self.label_mapping = None
        elif task == DownstreamTask.NEXT_VISIT:
            # Next visit prediction - handled separately
            raise NotImplementedError("NEXT_VISIT task requires NextVisitFlatDataset")
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

        # Patient-level split (same as HAT)
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
            # For multilabel tasks, use hadm_to_categories mapping
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

        # Build patient admission order
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
        """
        Get a patient's EHR history as a flat sequence with label.

        Returns:
            dict with keys:
                - input_ids: (seq_len,)
                - attention_mask: (seq_len,)
                - t: (seq_len,) absolute time values (days)
                - token_types: (seq_len,) if include_token_types
                - visit_ids: (seq_len,) if include_visit_ids
                - label: scalar or tensor
        """
        subject_id, target_hadm_id = self.samples[index]
        label = self.label_dict[(subject_id, target_hadm_id)]

        # Get history up to and including target admission
        all_hadms = self.patient_admissions[subject_id]
        target_idx = all_hadms.index(target_hadm_id)
        history_hadms = list(all_hadms[:target_idx + 1])

        # Read data
        subject_dir = self.data_path / f"{self.patient_id_col}={subject_id}"
        table = pq.read_table(subject_dir, filters=[(self.enc_id_col, 'in', history_hadms)])
        history_data = table.to_pandas()

        # Group by encounter
        grouped = history_data.groupby(self.enc_id_col, sort=False)
        max_hours = self.task_config.get("max_hours")
        exclude_target_dx = self.task_config.get("exclude_target_dx", False)

        # Build encounter data with task-specific filtering
        encounter_data = []
        for enc_id, group in grouped:
            # Apply task-specific filtering for target admission
            if enc_id == target_hadm_id:
                if exclude_target_dx:
                    group = group[~group[self.token_col].str.startswith('DX:', na=False)]
                if max_hours is not None and self.token_time_col in group.columns:
                    group = group[group[self.token_time_col] <= max_hours]

            tokens = group[self.token_col].tolist()
            if len(tokens) == 0:
                continue

            # Get visit time
            visit_time = group[self.visit_time_col].iloc[0] if self.visit_time_col in group.columns else 0.0
            visit_time = 0.0 if pd.isna(visit_time) else float(visit_time)

            # Get sort key
            sort_val = group[self.sort_col].iloc[0] if self.sort_col in group.columns else None

            # Get token times
            token_times = group[self.token_time_col].fillna(0.0).tolist() if self.token_time_col in group.columns else None

            # Get code types for HEART
            code_types = group[self.code_type_col].tolist() if self.code_type_col in group.columns else None

            encounter_data.append({
                'enc_id': enc_id,
                'tokens': tokens,
                'visit_time': visit_time,
                'sort_key': sort_val,
                'token_times': token_times,
                'code_types': code_types,
            })

        # Sort by admission time
        encounter_data.sort(key=lambda x: x['sort_key'] if x['sort_key'] is not None else 0)

        # Compute cumulative visit times
        visit_rel_times = [enc['visit_time'] for enc in encounter_data]
        visit_abs_times = np.cumsum(visit_rel_times).tolist()

        # Build flat sequences
        all_tokens = []
        all_times = []
        all_token_types = []
        all_visit_ids = []

        for visit_idx, enc in enumerate(encounter_data):
            visit_abs_time = visit_abs_times[visit_idx]
            tokens = enc['tokens']

            # Add [CLS] and tokens
            all_tokens.append("[CLS]")
            all_tokens.extend(tokens)

            # Add times: [CLS] gets visit time, tokens get visit_time + offset/24
            all_times.append(visit_abs_time)
            if enc['token_times']:
                token_abs_times = [visit_abs_time + (h / 24.0) for h in enc['token_times']]
                all_times.extend(token_abs_times)
            else:
                all_times.extend([visit_abs_time] * len(tokens))

            # Token types for HEART
            if self.include_token_types:
                all_token_types.append(1)  # [CLS] type
                if enc['code_types']:
                    type_ids = [CODE_TYPE_TO_TOKEN_TYPE.get(ct, 0) for ct in enc['code_types']]
                    all_token_types.extend(type_ids)
                else:
                    all_token_types.extend([0] * len(tokens))

            # Visit IDs for HEART
            if self.include_visit_ids:
                all_visit_ids.append(visit_idx)
                all_visit_ids.extend([visit_idx] * len(tokens))

        # Truncate to max_seq_len
        all_tokens = all_tokens[:self.max_seq_len]
        all_times = all_times[:self.max_seq_len]
        if self.include_token_types:
            all_token_types = all_token_types[:self.max_seq_len]
        if self.include_visit_ids:
            all_visit_ids = all_visit_ids[:self.max_seq_len]

        # Encode tokens
        encoding = self.tokenizer.encode(all_tokens, is_pretokenized=True)
        input_ids = encoding.ids[:self.max_seq_len]

        # Pad to max_seq_len
        seq_len = len(input_ids)
        pad_len = self.max_seq_len - seq_len
        if pad_len > 0:
            input_ids = input_ids + [0] * pad_len
            all_times = all_times + [0.0] * pad_len
            if self.include_token_types:
                all_token_types = all_token_types + [0] * pad_len
            if self.include_visit_ids:
                all_visit_ids = all_visit_ids + [0] * pad_len

        # Create attention mask
        attention_mask = [1] * seq_len + [0] * pad_len

        result = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.bool),
            "t": torch.tensor(all_times, dtype=torch.float32),
        }

        # Handle label differently for multilabel vs single-label tasks
        if self.is_multilabel:
            multi_hot = torch.zeros(self.num_classes, dtype=torch.float)
            for idx in label:
                multi_hot[idx] = 1.0
            result["label"] = multi_hot
        else:
            result["label"] = torch.tensor(label, dtype=torch.long)

        if self.include_token_types:
            result["token_types"] = torch.tensor(all_token_types, dtype=torch.long)
        if self.include_visit_ids:
            result["visit_ids"] = torch.tensor(all_visit_ids, dtype=torch.long)

        return result


def collate_flat_finetune(batch):
    """Collate function for FlatFinetuneDataset."""
    return {
        "input_ids": torch.stack([x["input_ids"] for x in batch]),
        "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
        "t": torch.stack([x["t"] for x in batch]),
        "label": torch.stack([x["label"] for x in batch]),
        # Optional fields for HEART
        "token_types": torch.stack([x["token_types"] for x in batch]) if "token_types" in batch[0] else None,
        "visit_ids": torch.stack([x["visit_ids"] for x in batch]) if "visit_ids" in batch[0] else None,
    }


class NextVisitFlatDataset(Dataset):
    """
    Next Visit Prediction Dataset for flat baseline models (CORE-BEHRT, HEART).

    Key differences from FlatFinetuneDataset:
    - Input: historical admissions (NOT including target)
    - Label: multi-hot vector of token IDs from the target admission
    - Only includes samples where there IS a next admission (needs ≥2 admissions)
    """

    def __init__(
        self,
        data_path: str | Path,
        labels_df: pd.DataFrame,
        tokenizer: Tokenizer,
        max_seq_len: int = 2048,
        split: str = "train",
        split_ratios: tuple = (0.7, 0.15, 0.15),
        seed: int = 42,
        patient_id_col: str = "subject_id",
        enc_id_col: str = "hadm_id",
        token_col: str = "code",
        code_type_col: str = "code_type",
        sort_col: str = "admittime",
        token_time_col: str = "time_offset_hours",
        visit_time_col: str = "days_since_prior_admission",
        include_token_types: bool = False,  # for HEART
        include_visit_ids: bool = False,  # for HEART
    ):
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.patient_id_col = patient_id_col
        self.enc_id_col = enc_id_col
        self.token_col = token_col
        self.code_type_col = code_type_col
        self.sort_col = sort_col
        self.token_time_col = token_time_col
        self.visit_time_col = visit_time_col
        self.include_token_types = include_token_types
        self.include_visit_ids = include_visit_ids

        self.vocab_size = tokenizer.get_vocab_size()

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

        # Build patient_admissions: {patient_id: [sorted list of hadm_ids]}
        self.patient_admissions = {}
        for pid in labels[patient_id_col].unique():
            patient_data = labels[labels[patient_id_col] == pid]
            sorted_hadms = patient_data.sort_values(sort_col)[enc_id_col].tolist()
            if len(sorted_hadms) >= 2:  # Need at least 2 admissions for next visit prediction
                self.patient_admissions[pid] = sorted_hadms

        # Build samples: (patient_id, target_hadm_id) where target is NOT the first admission
        self.samples = []
        for pid, hadms in self.patient_admissions.items():
            for i in range(1, len(hadms)):
                self.samples.append((pid, hadms[i]))

        print(f"NextVisitFlatDataset [{split}]: {len(self.samples)} samples from {len(self.patient_admissions)} patients")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        subject_id, target_hadm_id = self.samples[index]

        all_hadms = self.patient_admissions[subject_id]
        target_idx = all_hadms.index(target_hadm_id)

        # Input: all admissions BEFORE target (not including target)
        history_hadms = list(all_hadms[:target_idx])

        # Read data
        subject_dir = self.data_path / f"{self.patient_id_col}={subject_id}"

        # Read history data
        try:
            history_table = pq.read_table(subject_dir, filters=[(self.enc_id_col, 'in', history_hadms)])
            history_data = history_table.to_pandas()
        except Exception:
            history_table = pq.read_table(subject_dir)
            history_data = history_table.to_pandas()
            history_data = history_data[history_data[self.enc_id_col].isin(history_hadms)]

        # Read target data for labels
        try:
            target_table = pq.read_table(subject_dir, filters=[(self.enc_id_col, '==', target_hadm_id)])
            target_data = target_table.to_pandas()
        except Exception:
            target_table = pq.read_table(subject_dir)
            target_data = target_table.to_pandas()
            target_data = target_data[target_data[self.enc_id_col] == target_hadm_id]

        # Build encounter data from history
        grouped = history_data.groupby(self.enc_id_col, sort=False)
        encounter_data = []

        for enc_id, group in grouped:
            tokens = group[self.token_col].tolist()
            if len(tokens) == 0:
                continue

            visit_time = group[self.visit_time_col].iloc[0] if self.visit_time_col in group.columns else 0.0
            visit_time = 0.0 if pd.isna(visit_time) else float(visit_time)
            sort_val = group[self.sort_col].iloc[0] if self.sort_col in group.columns else None
            token_times = group[self.token_time_col].fillna(0.0).tolist() if self.token_time_col in group.columns else None
            code_types = group[self.code_type_col].tolist() if self.code_type_col in group.columns else None

            encounter_data.append({
                'enc_id': enc_id,
                'tokens': tokens,
                'visit_time': visit_time,
                'sort_key': sort_val,
                'token_times': token_times,
                'code_types': code_types,
            })

        encounter_data.sort(key=lambda x: x['sort_key'] if x['sort_key'] is not None else 0)

        # Compute cumulative visit times
        visit_rel_times = [enc['visit_time'] for enc in encounter_data]
        visit_abs_times = np.cumsum(visit_rel_times).tolist() if visit_rel_times else []

        # Build flat sequences
        all_tokens = []
        all_times = []
        all_token_types = []
        all_visit_ids = []

        for visit_idx, enc in enumerate(encounter_data):
            visit_abs_time = visit_abs_times[visit_idx] if visit_idx < len(visit_abs_times) else 0.0
            tokens = enc['tokens']

            all_tokens.append("[CLS]")
            all_tokens.extend(tokens)

            all_times.append(visit_abs_time)
            if enc['token_times']:
                token_abs_times = [visit_abs_time + (h / 24.0) for h in enc['token_times']]
                all_times.extend(token_abs_times)
            else:
                all_times.extend([visit_abs_time] * len(tokens))

            if self.include_token_types:
                all_token_types.append(1)
                if enc['code_types']:
                    type_ids = [CODE_TYPE_TO_TOKEN_TYPE.get(ct, 0) for ct in enc['code_types']]
                    all_token_types.extend(type_ids)
                else:
                    all_token_types.extend([0] * len(tokens))

            if self.include_visit_ids:
                all_visit_ids.append(visit_idx)
                all_visit_ids.extend([visit_idx] * len(tokens))

        # Truncate and pad
        all_tokens = all_tokens[:self.max_seq_len]
        all_times = all_times[:self.max_seq_len]
        if self.include_token_types:
            all_token_types = all_token_types[:self.max_seq_len]
        if self.include_visit_ids:
            all_visit_ids = all_visit_ids[:self.max_seq_len]

        # Encode tokens
        encoding = self.tokenizer.encode(all_tokens, is_pretokenized=True)
        input_ids = encoding.ids[:self.max_seq_len]

        seq_len = len(input_ids)
        pad_len = self.max_seq_len - seq_len
        if pad_len > 0:
            input_ids = input_ids + [0] * pad_len
            all_times = all_times + [0.0] * pad_len
            if self.include_token_types:
                all_token_types = all_token_types + [0] * pad_len
            if self.include_visit_ids:
                all_visit_ids = all_visit_ids + [0] * pad_len

        attention_mask = [1] * seq_len + [0] * pad_len

        # Label: multi-hot vector of target admission's token IDs
        target_tokens = target_data[self.token_col].tolist()
        target_token_ids = set()
        for token in target_tokens:
            token_id = self.tokenizer.token_to_id(str(token))
            if token_id is not None and token_id >= 4:  # Skip special tokens
                target_token_ids.add(token_id)

        multi_hot = torch.zeros(self.vocab_size, dtype=torch.float)
        for token_id in target_token_ids:
            multi_hot[token_id] = 1.0

        result = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.bool),
            "t": torch.tensor(all_times, dtype=torch.float32),
            "label": multi_hot,
            "target_token_ids": list(target_token_ids),  # For evaluation
        }

        if self.include_token_types:
            result["token_types"] = torch.tensor(all_token_types, dtype=torch.long)
        if self.include_visit_ids:
            result["visit_ids"] = torch.tensor(all_visit_ids, dtype=torch.long)

        return result


def collate_next_visit_flat(batch):
    """Collate function for NextVisitFlatDataset."""
    return {
        "input_ids": torch.stack([x["input_ids"] for x in batch]),
        "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
        "t": torch.stack([x["t"] for x in batch]),
        "label": torch.stack([x["label"] for x in batch]),
        "target_token_ids": [x["target_token_ids"] for x in batch],
        "token_types": torch.stack([x["token_types"] for x in batch]) if "token_types" in batch[0] else None,
        "visit_ids": torch.stack([x["visit_ids"] for x in batch]) if "visit_ids" in batch[0] else None,
    }


# ============================================================================
# GT-BEHRT Data Utilities
# ============================================================================

# Edge type mapping for GT-BEHRT
# Edge types based on code type pairs (symmetric)
EDGE_TYPE_MAP = {
    ("diagnosis", "diagnosis"): 0,
    ("diagnosis", "procedure"): 1,
    ("procedure", "diagnosis"): 1,
    ("diagnosis", "medication"): 2,
    ("medication", "diagnosis"): 2,
    ("diagnosis", "lab"): 3,
    ("lab", "diagnosis"): 3,
    ("procedure", "procedure"): 4,
    ("procedure", "medication"): 5,
    ("medication", "procedure"): 5,
    ("procedure", "lab"): 6,
    ("lab", "procedure"): 6,
    ("medication", "medication"): 7,
    ("medication", "lab"): 8,
    ("lab", "medication"): 8,
    ("lab", "lab"): 9,
}
# Default edge type for unknown pairs
DEFAULT_EDGE_TYPE = 0
# Edge type for VST node connections
VST_EDGE_TYPE = 0  # Same as diagnosis-diagnosis


class GTBEHRTDataset(Dataset):
    """
    GT-BEHRT Dataset for graph-based EHR processing.

    Converts parquet data into graph format:
    - Each visit becomes a fully-connected graph of medical codes
    - A virtual <VST> node is added to each visit for graph-level readout
    - Edge types are determined by code type pairs
    - Temporal features (visit type, age, day-of-year) are extracted

    Input: Hive-partitioned parquet directory
    Output: Graph data + temporal features per patient
    """

    def __init__(
        self,
        data_path: str | Path,
        tokenizer,
        supervised_task_cohort=None,
        max_visits: int = 50,
        max_codes_per_visit: int = 100,
        patient_id_col: str = "subject_id",
        enc_id_col: str = "hadm_id",
        token_col: str = "code",
        code_type_col: str = "code_type",
        sort_col: str = "visit_seq",
        visit_time_col: str = "days_since_prior_admission",
        # GT-BEHRT specific columns (optional, use defaults if not available)
        visit_type_col: str = None,  # Visit type (derived from days_since_prior_admission)
        age_col: str = "anchor_age",  # Patient age at visit
        los_col: str = "los",  # Length of stay
    ):
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_visits = max_visits
        self.max_codes_per_visit = max_codes_per_visit
        self.patient_id_col = patient_id_col
        self.enc_id_col = enc_id_col
        self.token_col = token_col
        self.code_type_col = code_type_col
        self.sort_col = sort_col
        self.visit_time_col = visit_time_col
        self.visit_type_col = visit_type_col
        self.age_col = age_col
        self.los_col = los_col

        # Get special token IDs
        self.vst_token_id = tokenizer.token_to_id("[PAD]")  # Use PAD as VST placeholder

        # Scan patient directories
        self.subject_dirs = sorted(self.data_path.glob(f"{patient_id_col}=*"))
        if not self.subject_dirs:
            self.subject_dirs = sorted(self.data_path.glob("subject_id=*"))

        self.patient_ids = [d.name.split('=')[1] for d in self.subject_dirs]

        # Filter by cohort if provided
        if supervised_task_cohort is not None:
            cohort_ids = set(supervised_task_cohort[patient_id_col].astype(str).tolist())
            filtered = [(d, pid) for d, pid in zip(self.subject_dirs, self.patient_ids) if pid in cohort_ids]
            self.subject_dirs = [d for d, _ in filtered]
            self.patient_ids = [pid for _, pid in filtered]

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, index):
        """
        Get a patient's EHR as graph data.

        Returns:
            dict with:
                - node_ids: (n_nodes,) token IDs including VST nodes
                - edge_index: (2, n_edges) graph edges
                - edge_type: (n_edges,) edge type IDs
                - vst_indices: (n_visits,) indices of VST nodes
                - code_types: (n_nodes,) code type for each node
                - visit_types: (n_visits,) visit type IDs
                - positions: (n_visits,) visit positions (0, 1, 2, ...)
                - ages: (n_visits,) patient age at each visit
                - days: (n_visits,) day of year for each visit
                - attention_mask: (n_visits,) valid visit mask
        """
        subject_dir = self.subject_dirs[index]

        # Read needed columns
        needed_cols = [self.enc_id_col, self.token_col]
        if self.sort_col:
            needed_cols.append(self.sort_col)
        if self.code_type_col:
            needed_cols.append(self.code_type_col)
        if self.visit_time_col:
            needed_cols.append(self.visit_time_col)

        # Add optional columns if they might exist
        optional_cols = [self.visit_type_col, self.age_col, self.los_col]
        for col in optional_cols:
            if col:
                needed_cols.append(col)

        # Read and deduplicate columns
        needed_cols = list(set(needed_cols))

        try:
            table = pq.read_table(subject_dir, columns=needed_cols)
            df = table.to_pandas()
        except Exception:
            # Fallback: read all and select
            table = pq.read_table(subject_dir)
            df = table.to_pandas()
            df = df[[c for c in needed_cols if c in df.columns]]

        # Sort by visit
        if self.sort_col and self.sort_col in df.columns:
            df = df.sort_values(by=self.sort_col)

        # Group by encounter
        grouped = df.groupby(self.enc_id_col, sort=False)

        # Collect visit data
        visits = []
        for enc_id, group in grouped:
            tokens = group[self.token_col].tolist()[:self.max_codes_per_visit]
            if len(tokens) == 0:
                continue

            code_types = (
                group[self.code_type_col].tolist()[:self.max_codes_per_visit]
                if self.code_type_col and self.code_type_col in group.columns
                else ["diagnosis"] * len(tokens)
            )

            # Get visit-level features
            # Derive visit_type from days_since_prior_admission:
            #   Type 1: First visit (NULL or no prior)
            #   Type 2: Acute readmission (<30 days)
            #   Type 3: Short-term readmission (30-90 days)
            #   Type 4: Long-term follow-up (>90 days)
            visit_type = 1  # Default: first visit
            if self.visit_time_col and self.visit_time_col in group.columns:
                days = group[self.visit_time_col].iloc[0]
                if pd.notna(days):
                    days = float(days)
                    if days < 30:
                        visit_type = 2  # Acute readmission
                    elif days < 90:
                        visit_type = 3  # Short-term readmission
                    else:
                        visit_type = 4  # Long-term follow-up

            age = 50  # Default
            if self.age_col and self.age_col in group.columns:
                a = group[self.age_col].iloc[0]
                age = int(a) if pd.notna(a) else 50
                age = min(max(age, 0), 102)  # Clip to valid range

            # Day of year (use visit order as proxy if not available)
            day = 1
            if self.visit_time_col and self.visit_time_col in group.columns:
                t = group[self.visit_time_col].iloc[0]
                day = int(abs(t) % 366) + 1 if pd.notna(t) else 1

            visits.append({
                'tokens': tokens,
                'code_types': code_types,
                'visit_type': visit_type,
                'age': age,
                'day': day,
            })

        # Limit to max_visits
        visits = visits[:self.max_visits]

        # Build graph data
        all_node_ids = []
        all_code_types = []
        all_edges_src = []
        all_edges_dst = []
        all_edge_types = []
        vst_indices = []

        node_offset = 0
        for visit in visits:
            tokens = visit['tokens']
            code_types = visit['code_types']
            n_codes = len(tokens)

            # Add VST node first
            vst_idx = node_offset
            vst_indices.append(vst_idx)
            all_node_ids.append(self.vst_token_id)  # VST uses PAD token ID
            all_code_types.append("vst")

            # Add code nodes
            for token, ctype in zip(tokens, code_types):
                encoding = self.tokenizer.encode(token, add_special_tokens=False)
                token_id = encoding.ids[0] if encoding.ids else 0
                all_node_ids.append(token_id)
                all_code_types.append(ctype)

            # Build fully-connected graph within this visit
            visit_nodes = list(range(node_offset, node_offset + n_codes + 1))  # +1 for VST

            for i, src_node in enumerate(visit_nodes):
                for j, dst_node in enumerate(visit_nodes):
                    if i != j:  # No self-loops
                        all_edges_src.append(src_node)
                        all_edges_dst.append(dst_node)

                        # Determine edge type
                        if i == 0 or j == 0:  # VST node connection
                            edge_type = VST_EDGE_TYPE
                        else:
                            src_type = code_types[i - 1] if i > 0 else "vst"
                            dst_type = code_types[j - 1] if j > 0 else "vst"
                            edge_type = EDGE_TYPE_MAP.get(
                                (src_type, dst_type), DEFAULT_EDGE_TYPE
                            )
                        all_edge_types.append(edge_type)

            node_offset += n_codes + 1  # +1 for VST

        # Pad visits to max_visits
        n_visits = len(visits)
        visit_types = [v['visit_type'] for v in visits]
        ages = [v['age'] for v in visits]
        days = [v['day'] for v in visits]
        positions = list(range(n_visits))

        # Pad temporal features
        if n_visits < self.max_visits:
            pad_len = self.max_visits - n_visits
            visit_types += [0] * pad_len
            ages += [0] * pad_len
            days += [0] * pad_len
            positions += list(range(n_visits, self.max_visits))

        # Create attention mask
        attention_mask = [1] * n_visits + [0] * (self.max_visits - n_visits)

        # Convert to tensors
        result = {
            "node_ids": torch.tensor(all_node_ids, dtype=torch.long),
            "edge_index": torch.tensor([all_edges_src, all_edges_dst], dtype=torch.long)
            if all_edges_src else torch.zeros((2, 0), dtype=torch.long),
            "edge_type": torch.tensor(all_edge_types, dtype=torch.long)
            if all_edge_types else torch.zeros(0, dtype=torch.long),
            "vst_indices": torch.tensor(vst_indices, dtype=torch.long),
            "n_visits": n_visits,
            "visit_types": torch.tensor(visit_types[:self.max_visits], dtype=torch.long),
            "positions": torch.tensor(positions[:self.max_visits], dtype=torch.long),
            "ages": torch.tensor(ages[:self.max_visits], dtype=torch.long),
            "days": torch.tensor(days[:self.max_visits], dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask[:self.max_visits], dtype=torch.bool),
        }

        return result


def collate_gtbehrt(batch):
    """
    Collate function for GT-BEHRT dataset.

    Combines multiple patient graphs into a single batched graph.
    """
    # Batch graph data
    all_node_ids = []
    all_edges_src = []
    all_edges_dst = []
    all_edge_types = []
    all_vst_indices = []
    batch_visit_counts = []

    node_offset = 0
    vst_offset = 0

    for sample in batch:
        n_nodes = sample['node_ids'].size(0)
        n_edges = sample['edge_index'].size(1)
        n_visits = sample['n_visits']

        # Add node IDs
        all_node_ids.append(sample['node_ids'])

        # Add edges with offset
        if n_edges > 0:
            all_edges_src.append(sample['edge_index'][0] + node_offset)
            all_edges_dst.append(sample['edge_index'][1] + node_offset)
            all_edge_types.append(sample['edge_type'])

        # Add VST indices with offset
        all_vst_indices.append(sample['vst_indices'] + node_offset)

        batch_visit_counts.append(n_visits)
        node_offset += n_nodes

    # Concatenate graph data
    graph_data = {
        'node_ids': torch.cat(all_node_ids) if all_node_ids else torch.zeros(0, dtype=torch.long),
        'edge_index': torch.cat([
            torch.cat(all_edges_src).unsqueeze(0),
            torch.cat(all_edges_dst).unsqueeze(0),
        ], dim=0) if all_edges_src else torch.zeros((2, 0), dtype=torch.long),
        'edge_type': torch.cat(all_edge_types) if all_edge_types else torch.zeros(0, dtype=torch.long),
        'vst_indices': torch.cat(all_vst_indices) if all_vst_indices else torch.zeros(0, dtype=torch.long),
        'batch_visit_counts': torch.tensor(batch_visit_counts, dtype=torch.long),
    }

    # Stack temporal features
    result = {
        'graph_data': graph_data,
        'visit_types': torch.stack([x['visit_types'] for x in batch]),
        'positions': torch.stack([x['positions'] for x in batch]),
        'ages': torch.stack([x['ages'] for x in batch]),
        'days': torch.stack([x['days'] for x in batch]),
        'attention_mask': torch.stack([x['attention_mask'] for x in batch]),
    }

    return result


class GTBEHRTFinetuneDataset(Dataset):
    """
    GT-BEHRT Dataset for fine-tuning on downstream tasks.

    Similar to GTBEHRTDataset but with task-specific labels and filtering.
    """

    def __init__(
        self,
        data_path: str | Path,
        labels_df: pd.DataFrame,
        tokenizer,
        task: DownstreamTask,
        max_visits: int = 50,
        max_codes_per_visit: int = 100,
        split: str = "train",
        split_ratios: tuple = (0.7, 0.15, 0.15),
        seed: int = 42,
        patient_id_col: str = "subject_id",
        enc_id_col: str = "hadm_id",
        token_col: str = "code",
        code_type_col: str = "code_type",
        sort_col: str = "admittime",
        visit_time_col: str = "days_since_prior_admission",
        visit_type_col: str = None,  # Derived from days_since_prior_admission
        age_col: str = "anchor_age",
        token_time_col: str = "time_offset_hours",
    ):
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.task = task
        self.task_config = TASK_CONFIGS[task]
        self.max_visits = max_visits
        self.max_codes_per_visit = max_codes_per_visit
        self.patient_id_col = patient_id_col
        self.enc_id_col = enc_id_col
        self.token_col = token_col
        self.code_type_col = code_type_col
        self.sort_col = sort_col
        self.visit_time_col = visit_time_col
        self.visit_type_col = visit_type_col
        self.age_col = age_col
        self.token_time_col = token_time_col

        self.vst_token_id = tokenizer.token_to_id("[PAD]")

        # Setup label handling
        self.is_multilabel = self.task_config.get("is_multilabel", False)

        if task == DownstreamTask.ICD_CATEGORY_MULTILABEL:
            # Multilabel ICD category prediction
            self.label_col = "icd_categories"
            valid_labels = labels_df[labels_df['icd_categories'].notna() & (labels_df['icd_categories'] != '')].copy()
            # Parse icd_categories to find all unique category IDs
            all_categories = set()
            self.hadm_to_categories = {}
            for _, row in valid_labels.iterrows():
                cats = [int(x) for x in str(row['icd_categories']).split(',')]
                self.hadm_to_categories[row[enc_id_col]] = cats
                all_categories.update(cats)
            self.num_classes = max(all_categories) + 1  # Category IDs are 0-indexed
            self.label_mapping = None
        elif task == DownstreamTask.NEXT_VISIT:
            # Next visit prediction - handled separately
            raise NotImplementedError("NEXT_VISIT task requires NextVisitGTBEHRTDataset")
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
            # For multilabel tasks, use hadm_to_categories mapping
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

        # Build patient admission order
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
        """Get a patient's EHR as graph data with label."""
        subject_id, target_hadm_id = self.samples[index]
        label = self.label_dict[(subject_id, target_hadm_id)]

        # Get history up to and including target admission
        all_hadms = self.patient_admissions[subject_id]
        target_idx = all_hadms.index(target_hadm_id)
        history_hadms = list(all_hadms[:target_idx + 1])

        # Read data
        subject_dir = self.data_path / f"{self.patient_id_col}={subject_id}"
        try:
            table = pq.read_table(subject_dir, filters=[(self.enc_id_col, 'in', history_hadms)])
            history_data = table.to_pandas()
        except Exception:
            table = pq.read_table(subject_dir)
            history_data = table.to_pandas()
            history_data = history_data[history_data[self.enc_id_col].isin(history_hadms)]

        # Group by encounter
        grouped = history_data.groupby(self.enc_id_col, sort=False)
        max_hours = self.task_config.get("max_hours")
        exclude_target_dx = self.task_config.get("exclude_target_dx", False)

        # Build visit data with task-specific filtering
        visits = []
        for enc_id, group in grouped:
            # Apply task-specific filtering for target admission
            if enc_id == target_hadm_id:
                if exclude_target_dx:
                    group = group[~group[self.token_col].str.startswith('DX:', na=False)]
                if max_hours is not None and self.token_time_col in group.columns:
                    group = group[group[self.token_time_col] <= max_hours]

            tokens = group[self.token_col].tolist()[:self.max_codes_per_visit]
            if len(tokens) == 0:
                continue

            code_types = (
                group[self.code_type_col].tolist()[:self.max_codes_per_visit]
                if self.code_type_col in group.columns
                else ["diagnosis"] * len(tokens)
            )

            # Visit-level features
            # Derive visit_type from days_since_prior_admission:
            #   Type 1: First visit (NULL or no prior)
            #   Type 2: Acute readmission (<30 days)
            #   Type 3: Short-term readmission (30-90 days)
            #   Type 4: Long-term follow-up (>90 days)
            visit_type = 1  # Default: first visit
            if self.visit_time_col in group.columns:
                days_val = group[self.visit_time_col].iloc[0]
                if pd.notna(days_val):
                    days_val = float(days_val)
                    if days_val < 30:
                        visit_type = 2  # Acute readmission
                    elif days_val < 90:
                        visit_type = 3  # Short-term readmission
                    else:
                        visit_type = 4  # Long-term follow-up

            age = 50
            if self.age_col in group.columns:
                a = group[self.age_col].iloc[0]
                age = int(a) if pd.notna(a) else 50
                age = min(max(age, 0), 102)

            day = 1
            if self.visit_time_col in group.columns:
                t = group[self.visit_time_col].iloc[0]
                day = int(abs(t) % 366) + 1 if pd.notna(t) else 1

            # Sort key for ordering
            sort_val = group[self.sort_col].iloc[0] if self.sort_col in group.columns else None

            visits.append({
                'tokens': tokens,
                'code_types': code_types,
                'visit_type': visit_type,
                'age': age,
                'day': day,
                'sort_key': sort_val,
            })

        # Sort visits by admission time
        visits.sort(key=lambda x: x['sort_key'] if x['sort_key'] is not None else 0)
        visits = visits[:self.max_visits]

        # Build graph data (same as GTBEHRTDataset)
        all_node_ids = []
        all_edges_src = []
        all_edges_dst = []
        all_edge_types = []
        vst_indices = []

        node_offset = 0
        for visit in visits:
            tokens = visit['tokens']
            code_types = visit['code_types']
            n_codes = len(tokens)

            # Add VST node
            vst_idx = node_offset
            vst_indices.append(vst_idx)
            all_node_ids.append(self.vst_token_id)

            # Add code nodes
            for token, ctype in zip(tokens, code_types):
                encoding = self.tokenizer.encode(token, add_special_tokens=False)
                token_id = encoding.ids[0] if encoding.ids else 0
                all_node_ids.append(token_id)

            # Build fully-connected graph
            visit_nodes = list(range(node_offset, node_offset + n_codes + 1))

            for i, src_node in enumerate(visit_nodes):
                for j, dst_node in enumerate(visit_nodes):
                    if i != j:
                        all_edges_src.append(src_node)
                        all_edges_dst.append(dst_node)

                        if i == 0 or j == 0:
                            edge_type = VST_EDGE_TYPE
                        else:
                            src_type = code_types[i - 1] if i > 0 else "vst"
                            dst_type = code_types[j - 1] if j > 0 else "vst"
                            edge_type = EDGE_TYPE_MAP.get(
                                (src_type, dst_type), DEFAULT_EDGE_TYPE
                            )
                        all_edge_types.append(edge_type)

            node_offset += n_codes + 1

        # Pad temporal features
        n_visits = len(visits)
        visit_types = [v['visit_type'] for v in visits]
        ages = [v['age'] for v in visits]
        days = [v['day'] for v in visits]
        positions = list(range(n_visits))

        if n_visits < self.max_visits:
            pad_len = self.max_visits - n_visits
            visit_types += [0] * pad_len
            ages += [0] * pad_len
            days += [0] * pad_len
            positions += list(range(n_visits, self.max_visits))

        attention_mask = [1] * n_visits + [0] * (self.max_visits - n_visits)

        result = {
            "node_ids": torch.tensor(all_node_ids, dtype=torch.long),
            "edge_index": torch.tensor([all_edges_src, all_edges_dst], dtype=torch.long)
            if all_edges_src else torch.zeros((2, 0), dtype=torch.long),
            "edge_type": torch.tensor(all_edge_types, dtype=torch.long)
            if all_edge_types else torch.zeros(0, dtype=torch.long),
            "vst_indices": torch.tensor(vst_indices, dtype=torch.long),
            "n_visits": n_visits,
            "visit_types": torch.tensor(visit_types[:self.max_visits], dtype=torch.long),
            "positions": torch.tensor(positions[:self.max_visits], dtype=torch.long),
            "ages": torch.tensor(ages[:self.max_visits], dtype=torch.long),
            "days": torch.tensor(days[:self.max_visits], dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask[:self.max_visits], dtype=torch.bool),
        }

        # Handle label differently for multilabel vs single-label tasks
        if self.is_multilabel:
            multi_hot = torch.zeros(self.num_classes, dtype=torch.float)
            for idx in label:
                multi_hot[idx] = 1.0
            result["label"] = multi_hot
        else:
            result["label"] = torch.tensor(label, dtype=torch.long)

        return result


def collate_gtbehrt_finetune(batch):
    """Collate function for GT-BEHRT fine-tuning dataset."""
    base_collate = collate_gtbehrt(batch)
    base_collate['label'] = torch.stack([x['label'] for x in batch])
    return base_collate


class NextVisitGTBEHRTDataset(Dataset):
    """
    Next Visit Prediction Dataset for GT-BEHRT.

    Key differences from GTBEHRTFinetuneDataset:
    - Input: historical admissions (NOT including target) as graph data
    - Label: multi-hot vector of token IDs from the target admission
    - Only includes samples where there IS a next admission (needs ≥2 admissions)
    """

    def __init__(
        self,
        data_path: str | Path,
        labels_df: pd.DataFrame,
        tokenizer,
        max_visits: int = 50,
        max_codes_per_visit: int = 100,
        split: str = "train",
        split_ratios: tuple = (0.7, 0.15, 0.15),
        seed: int = 42,
        patient_id_col: str = "subject_id",
        enc_id_col: str = "hadm_id",
        token_col: str = "code",
        code_type_col: str = "code_type",
        sort_col: str = "admittime",
        visit_time_col: str = "days_since_prior_admission",
        age_col: str = "anchor_age",
    ):
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_visits = max_visits
        self.max_codes_per_visit = max_codes_per_visit
        self.patient_id_col = patient_id_col
        self.enc_id_col = enc_id_col
        self.token_col = token_col
        self.code_type_col = code_type_col
        self.sort_col = sort_col
        self.visit_time_col = visit_time_col
        self.age_col = age_col

        self.vst_token_id = tokenizer.token_to_id("[PAD]")
        self.vocab_size = tokenizer.get_vocab_size()

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

        # Build patient_admissions: {patient_id: [sorted list of hadm_ids]}
        self.patient_admissions = {}
        for pid in labels[patient_id_col].unique():
            patient_data = labels[labels[patient_id_col] == pid]
            sorted_hadms = patient_data.sort_values(sort_col)[enc_id_col].tolist()
            if len(sorted_hadms) >= 2:  # Need at least 2 admissions for next visit prediction
                self.patient_admissions[pid] = sorted_hadms

        # Build samples: (patient_id, target_hadm_id) where target is NOT the first admission
        self.samples = []
        for pid, hadms in self.patient_admissions.items():
            for i in range(1, len(hadms)):
                self.samples.append((pid, hadms[i]))

        print(f"NextVisitGTBEHRTDataset [{split}]: {len(self.samples)} samples from {len(self.patient_admissions)} patients")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        subject_id, target_hadm_id = self.samples[index]

        all_hadms = self.patient_admissions[subject_id]
        target_idx = all_hadms.index(target_hadm_id)

        # Input: all admissions BEFORE target (not including target)
        history_hadms = list(all_hadms[:target_idx])

        # Read data
        subject_dir = self.data_path / f"{self.patient_id_col}={subject_id}"

        # Read history data
        try:
            history_table = pq.read_table(subject_dir, filters=[(self.enc_id_col, 'in', history_hadms)])
            history_data = history_table.to_pandas()
        except Exception:
            history_table = pq.read_table(subject_dir)
            history_data = history_table.to_pandas()
            history_data = history_data[history_data[self.enc_id_col].isin(history_hadms)]

        # Read target data for labels
        try:
            target_table = pq.read_table(subject_dir, filters=[(self.enc_id_col, '==', target_hadm_id)])
            target_data = target_table.to_pandas()
        except Exception:
            target_table = pq.read_table(subject_dir)
            target_data = target_table.to_pandas()
            target_data = target_data[target_data[self.enc_id_col] == target_hadm_id]

        # Build visit data from history
        grouped = history_data.groupby(self.enc_id_col, sort=False)
        visits = []

        for enc_id, group in grouped:
            tokens = group[self.token_col].tolist()[:self.max_codes_per_visit]
            if len(tokens) == 0:
                continue

            code_types = (
                group[self.code_type_col].tolist()[:self.max_codes_per_visit]
                if self.code_type_col in group.columns
                else ["diagnosis"] * len(tokens)
            )

            # Visit-level features
            visit_type = 1  # Default: first visit
            if self.visit_time_col in group.columns:
                days_val = group[self.visit_time_col].iloc[0]
                if pd.notna(days_val):
                    days_val = float(days_val)
                    if days_val < 30:
                        visit_type = 2
                    elif days_val < 90:
                        visit_type = 3
                    else:
                        visit_type = 4

            age = 50
            if self.age_col in group.columns:
                a = group[self.age_col].iloc[0]
                age = int(a) if pd.notna(a) else 50
                age = min(max(age, 0), 102)

            day = 1
            if self.visit_time_col in group.columns:
                t = group[self.visit_time_col].iloc[0]
                day = int(abs(t) % 366) + 1 if pd.notna(t) else 1

            sort_val = group[self.sort_col].iloc[0] if self.sort_col in group.columns else None

            visits.append({
                'tokens': tokens,
                'code_types': code_types,
                'visit_type': visit_type,
                'age': age,
                'day': day,
                'sort_key': sort_val,
            })

        # Sort visits by admission time
        visits.sort(key=lambda x: x['sort_key'] if x['sort_key'] is not None else 0)
        visits = visits[:self.max_visits]

        # Build graph data
        all_node_ids = []
        all_edges_src = []
        all_edges_dst = []
        all_edge_types = []
        vst_indices = []

        node_offset = 0
        for visit in visits:
            tokens = visit['tokens']
            code_types = visit['code_types']
            n_codes = len(tokens)

            # Add VST node
            vst_idx = node_offset
            vst_indices.append(vst_idx)
            all_node_ids.append(self.vst_token_id)

            # Add code nodes
            for token, ctype in zip(tokens, code_types):
                encoding = self.tokenizer.encode(token, add_special_tokens=False)
                token_id = encoding.ids[0] if encoding.ids else 0
                all_node_ids.append(token_id)

            # Build fully-connected graph
            visit_nodes = list(range(node_offset, node_offset + n_codes + 1))

            for i, src_node in enumerate(visit_nodes):
                for j, dst_node in enumerate(visit_nodes):
                    if i != j:
                        all_edges_src.append(src_node)
                        all_edges_dst.append(dst_node)

                        if i == 0 or j == 0:
                            edge_type = VST_EDGE_TYPE
                        else:
                            src_type = code_types[i - 1] if i > 0 else "vst"
                            dst_type = code_types[j - 1] if j > 0 else "vst"
                            edge_type = EDGE_TYPE_MAP.get(
                                (src_type, dst_type), DEFAULT_EDGE_TYPE
                            )
                        all_edge_types.append(edge_type)

            node_offset += n_codes + 1

        # Pad temporal features
        n_visits = len(visits)
        visit_types = [v['visit_type'] for v in visits]
        ages = [v['age'] for v in visits]
        days = [v['day'] for v in visits]
        positions = list(range(n_visits))

        if n_visits < self.max_visits:
            pad_len = self.max_visits - n_visits
            visit_types += [0] * pad_len
            ages += [0] * pad_len
            days += [0] * pad_len
            positions += list(range(n_visits, self.max_visits))

        attention_mask = [1] * n_visits + [0] * (self.max_visits - n_visits)

        # Label: multi-hot vector of target admission's token IDs
        target_tokens = target_data[self.token_col].tolist()
        target_token_ids = set()
        for token in target_tokens:
            token_id = self.tokenizer.token_to_id(str(token))
            if token_id is not None and token_id >= 4:  # Skip special tokens
                target_token_ids.add(token_id)

        multi_hot = torch.zeros(self.vocab_size, dtype=torch.float)
        for token_id in target_token_ids:
            multi_hot[token_id] = 1.0

        result = {
            "node_ids": torch.tensor(all_node_ids, dtype=torch.long) if all_node_ids else torch.zeros(1, dtype=torch.long),
            "edge_index": torch.tensor([all_edges_src, all_edges_dst], dtype=torch.long)
            if all_edges_src else torch.zeros((2, 0), dtype=torch.long),
            "edge_type": torch.tensor(all_edge_types, dtype=torch.long)
            if all_edge_types else torch.zeros(0, dtype=torch.long),
            "vst_indices": torch.tensor(vst_indices, dtype=torch.long) if vst_indices else torch.zeros(1, dtype=torch.long),
            "n_visits": max(n_visits, 1),  # Ensure at least 1 visit for batching
            "visit_types": torch.tensor(visit_types[:self.max_visits], dtype=torch.long),
            "positions": torch.tensor(positions[:self.max_visits], dtype=torch.long),
            "ages": torch.tensor(ages[:self.max_visits], dtype=torch.long),
            "days": torch.tensor(days[:self.max_visits], dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask[:self.max_visits], dtype=torch.bool),
            "label": multi_hot,
            "target_token_ids": list(target_token_ids),  # For evaluation
        }

        return result


def collate_next_visit_gtbehrt(batch):
    """Collate function for NextVisitGTBEHRTDataset."""
    # Batch graph data
    all_node_ids = []
    all_edges_src = []
    all_edges_dst = []
    all_edge_types = []
    all_vst_indices = []
    batch_visit_counts = []

    node_offset = 0

    for sample in batch:
        n_nodes = sample['node_ids'].size(0)
        n_edges = sample['edge_index'].size(1)
        n_visits = sample['n_visits']

        # Add node IDs
        all_node_ids.append(sample['node_ids'])

        # Add edges with offset
        if n_edges > 0:
            all_edges_src.append(sample['edge_index'][0] + node_offset)
            all_edges_dst.append(sample['edge_index'][1] + node_offset)
            all_edge_types.append(sample['edge_type'])

        # Add VST indices with offset
        all_vst_indices.append(sample['vst_indices'] + node_offset)

        batch_visit_counts.append(n_visits)
        node_offset += n_nodes

    # Concatenate graph data
    graph_data = {
        'node_ids': torch.cat(all_node_ids) if all_node_ids else torch.zeros(0, dtype=torch.long),
        'edge_index': torch.cat([
            torch.cat(all_edges_src).unsqueeze(0),
            torch.cat(all_edges_dst).unsqueeze(0),
        ], dim=0) if all_edges_src else torch.zeros((2, 0), dtype=torch.long),
        'edge_type': torch.cat(all_edge_types) if all_edge_types else torch.zeros(0, dtype=torch.long),
        'vst_indices': torch.cat(all_vst_indices) if all_vst_indices else torch.zeros(0, dtype=torch.long),
        'batch_visit_counts': torch.tensor(batch_visit_counts, dtype=torch.long),
    }

    # Stack temporal features and labels
    result = {
        'graph_data': graph_data,
        'visit_types': torch.stack([x['visit_types'] for x in batch]),
        'positions': torch.stack([x['positions'] for x in batch]),
        'ages': torch.stack([x['ages'] for x in batch]),
        'days': torch.stack([x['days'] for x in batch]),
        'attention_mask': torch.stack([x['attention_mask'] for x in batch]),
        'label': torch.stack([x['label'] for x in batch]),
        'target_token_ids': [x['target_token_ids'] for x in batch],
    }

    return result
