"""Fine-tuning Dataset for HAT downstream tasks."""

import torch
from torch.utils.data import Dataset
from tokenizers import Tokenizer
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq
import numpy as np
from enum import Enum

from src.tokenizer import get_batch_encoding


class DownstreamTask(Enum):
    MORTALITY = "mortality"
    READMISSION_30D = "readmission_30d"
    PROLONGED_LOS = "prolonged_los"
    ICD_CHAPTER = "icd_chapter"
    ABNORMAL_LAB = "abnormal_lab"


# Task configurations for prediction time points
# - "admission_24h": Use only first 24 hours of current admission (for predicting future outcomes)
# - "admission_48h": Use only first 48 hours of current admission
# - "discharge": Use entire admission (for summarizing or predicting post-discharge events)
# - "exclude_target_dx": Exclude diagnosis codes from target admission (prevent data leakage)
TASK_CONFIGS = {
    DownstreamTask.MORTALITY: {"prediction_time": "admission_24h", "max_hours": 24, "exclude_target_dx": True},
    DownstreamTask.PROLONGED_LOS: {"prediction_time": "admission_48h", "max_hours": 48, "exclude_target_dx": True},
    DownstreamTask.READMISSION_30D: {"prediction_time": "discharge", "max_hours": None, "exclude_target_dx": False},
    DownstreamTask.ICD_CHAPTER: {"prediction_time": "discharge", "max_hours": None, "exclude_target_dx": False},
    DownstreamTask.ABNORMAL_LAB: {"prediction_time": "discharge", "max_hours": None, "exclude_target_dx": False},
}


class FinetuneDataset(Dataset):
    """Admission-level dataset for downstream tasks."""

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
        self.task_config = TASK_CONFIGS[task]
        self.max_seg = max_seg
        self.max_seq_len = max_seq_len
        self.patient_id_col = patient_id_col
        self.enc_id_col = enc_id_col
        self.token_col = token_col
        self.time_col = time_col
        self.sort_col = sort_col
        self.token_time_col = token_time_col

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

        # Build label dict using itertuples (faster than iterrows)
        self.label_dict = {}
        pid_idx = self.labels.columns.get_loc(patient_id_col)
        enc_idx = self.labels.columns.get_loc(enc_id_col)
        label_idx = self.labels.columns.get_loc(self.label_col)
        for row in self.labels.itertuples(index=False):
            key = (row[pid_idx], row[enc_idx])
            label = row[label_idx]
            if self.label_mapping:
                label = self.label_mapping[label]
            self.label_dict[key] = label

        self.samples = list(self.label_dict.keys())

        # Group admissions by patient
        self.patient_admissions = {}
        for (pid, hadm_id) in self.samples:
            if pid not in self.patient_admissions:
                self.patient_admissions[pid] = []
            self.patient_admissions[pid].append(hadm_id)

        # Sort by admittime
        for pid in self.patient_admissions:
            patient_labels = self.labels[self.labels[patient_id_col] == pid]
            sorted_hadms = patient_labels.sort_values('admittime')[enc_id_col].tolist()
            self.patient_admissions[pid] = sorted_hadms

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        subject_id, target_hadm_id = self.samples[index]
        label = self.label_dict[(subject_id, target_hadm_id)]

        all_hadms = self.patient_admissions[subject_id]
        target_idx = all_hadms.index(target_hadm_id)
        history_hadms = list(all_hadms[:target_idx + 1])

        # Load with pyarrow filter
        subject_dir = self.data_path / f"{self.patient_id_col}={subject_id}"
        table = pq.read_table(subject_dir, filters=[(self.enc_id_col, 'in', history_hadms)])
        history_data = table.to_pandas()

        grouped = history_data.groupby(self.enc_id_col)

        # Get task config for filtering
        max_hours = self.task_config.get("max_hours")
        exclude_target_dx = self.task_config.get("exclude_target_dx", False)

        encounter_data = []
        for enc_id, group in grouped:
            # For target admission, apply task-specific filtering
            if enc_id == target_hadm_id:
                # Exclude diagnosis codes if configured (prevent data leakage)
                if exclude_target_dx:
                    group = group[~group[self.token_col].str.startswith('DX:', na=False)]

                # Apply time-based filtering
                if max_hours is not None and self.token_time_col in group.columns:
                    group = group[group[self.token_time_col] <= max_hours]

            tokens = group[self.token_col].tolist()
            if len(tokens) == 0:
                continue  # Skip empty encounters

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

        encounter_data.sort(key=lambda x: x['sort_key'] if x['sort_key'] is not None else 0)

        if len(encounter_data) > self.max_seg:
            encounter_data = encounter_data[-self.max_seg:]

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

        item["label"] = torch.tensor(label, dtype=torch.long)
        return item

    def _pad_segment(self, tokens):
        if len(tokens) > self.max_seg:
            seg_attn = torch.ones(self.max_seg)
            tokens = tokens[-self.max_seg:]
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

            times_clean = [0.0] + times_clean if times_clean else [0.0]

            if len(times_clean) > self.max_seq_len:
                times_clean = times_clean[:self.max_seq_len]
            else:
                pad_val = times_clean[-1] if times_clean else 0.0
                times_clean = times_clean + [pad_val] * (self.max_seq_len - len(times_clean))

            padded_times.append(times_clean)

        return torch.tensor(padded_times, dtype=torch.float32)

    def get_class_weights(self):
        """Get inverse frequency weights for imbalanced data."""
        labels = torch.tensor([self.label_dict[s] for s in self.samples])
        class_counts = torch.bincount(labels, minlength=self.num_classes).float()
        weights = 1.0 / (class_counts + 1e-6)
        weights = weights / weights.sum() * self.num_classes
        return weights


def collate_finetune(batch):
    result = {}
    for key in batch[0].keys():
        if isinstance(batch[0][key], torch.Tensor):
            result[key] = torch.stack([item[key] for item in batch])
        else:
            result[key] = [item[key] for item in batch]
    return result


def create_patient_splits(labels_path: str, output_dir: str, seed: int = 42):
    """Create and save patient-level train/val/test splits."""
    labels = pd.read_csv(labels_path)
    patients = labels['subject_id'].unique()

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
