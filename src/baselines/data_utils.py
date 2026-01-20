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
        max_seq_len: int = 512,
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
