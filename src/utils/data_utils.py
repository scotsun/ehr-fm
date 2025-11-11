import torch
from torch.utils.data import Dataset
from tokenizers import Tokenizer
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq

from src.tokenizer import get_batch_encoding, SPECIAL_TOKEN_IDS


class EHRDataset(Dataset):
    """
    EHR Dataset supporting both eager and lazy loading modes.
    
    Modes:
    - Eager (legacy): Pass DataFrame, all data loaded in memory
    - Lazy: Pass path to Hive-partitioned parquet, load on-demand per patient
    """
    def __init__(
        self,
        data,  # DataFrame (eager) or str/Path (lazy)
        tokenizer: Tokenizer,
        supervised_task_cohort=None,
        max_seg: int = 32,
        max_seq_len: int = 512,
        patient_id_col: str = "patient_id",
        enc_id_col: str = "visit_id",
        token_col: str = "code",
        time_col: str = None,
        sort_col: str = None,
        token_time_col: str = None,
    ):
        self.tokenizer = tokenizer
        self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]", length=max_seq_len)
        self.tokenizer.enable_truncation(max_length=max_seq_len)

        self.max_seg = max_seg
        self.max_seq_len = max_seq_len
        self.patient_id_col = patient_id_col
        self.enc_id_col = enc_id_col
        self.token_col = token_col
        self.time_col = time_col
        self.sort_col = sort_col if sort_col else time_col
        self.token_time_col = token_time_col
        self.supervised_task_cohort = supervised_task_cohort
        
        # Detect mode: lazy (path) or eager (DataFrame)
        if isinstance(data, (str, Path)):
            self._lazy_mode = True
            self.data_path = Path(data)
            
            # Scan patient directories (fast, only reads directory structure)
            partition_col = patient_id_col if patient_id_col.startswith("subject_id") or patient_id_col.startswith("user_id") else patient_id_col
            # Handle both subject_id and patient_id partition names
            self.subject_dirs = sorted(self.data_path.glob(f"{partition_col}=*"))
            if not self.subject_dirs:
                # Try alternative partition name
                self.subject_dirs = sorted(self.data_path.glob("subject_id=*"))
            if not self.subject_dirs:
                self.subject_dirs = sorted(self.data_path.glob("user_id=*"))
                
            self.patient_ids = [d.name.split('=')[1] for d in self.subject_dirs]
            
            # Filter by cohort if provided
            if supervised_task_cohort is not None:
                cohort_ids = set(supervised_task_cohort[patient_id_col].astype(str).tolist())
                filtered = [(d, pid) for d, pid in zip(self.subject_dirs, self.patient_ids) if pid in cohort_ids]
                self.subject_dirs = [d for d, _ in filtered]
                self.patient_ids = [pid for _, pid in filtered]
                
            self.data = None
        else:
            self._lazy_mode = False
            
            # Eager mode: original behavior
            if supervised_task_cohort is not None:
                data = data[data[patient_id_col].isin(supervised_task_cohort[patient_id_col])]
            
            self.data = data.groupby(patient_id_col)
            self.patient_ids = list(self.data.groups.keys())
            self.data_path = None
            self.subject_dirs = None

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, index):
        """
        Get a patient's EHR data.
        
        Returns:
            dict: Patient's tokenized EHR data with keys:
                - input_ids, attention_mask, segment_attention_mask
                - segment_time (optional), token_time (optional)
        """
        patient_id = self.patient_ids[index]
        
        # Load patient data based on mode
        if self._lazy_mode:
            # Lazy: read from disk on-demand using PyArrow (fast!)
            subject_dir = self.subject_dirs[index]
            
            # Only read needed columns for better performance
            needed_cols = [self.enc_id_col, self.token_col]
            if self.time_col:
                needed_cols.append(self.time_col)
            if self.sort_col and self.sort_col != self.time_col:
                needed_cols.append(self.sort_col)
            if self.token_time_col:
                needed_cols.append(self.token_time_col)
            
            # PyArrow reads entire directory efficiently (handles multiple parquet files)
            table = pq.read_table(subject_dir, columns=needed_cols)
            _patid_data = table.to_pandas()
        else:
            # Eager: get from pre-loaded groupby
            _patid_data = self.data.get_group(patient_id)
        
        # group by visit/order and extract tokens and times
        grouped = _patid_data.groupby(self.enc_id_col)
        
        # extract information for each encounter
        encounter_data = []
        for enc_id, group in grouped:
            tokens = group[self.token_col].to_list()
            time_val = group[self.time_col].iloc[0] if self.time_col else None
            sort_val = group[self.sort_col].iloc[0] if self.sort_col else None
            # extract token-level time (e.g., time_offset_hours)
            token_times = group[self.token_time_col].to_list() if self.token_time_col else None
            encounter_data.append({
                'enc_id': enc_id,
                'tokens': tokens,
                'time': time_val,
                'sort_key': sort_val,
                'token_times': token_times  # token-level time
            })
        
        # sort by sort_col (e.g., order_number, visit_seq, admittime)
        if self.sort_col:
            encounter_data.sort(key=lambda x: x['sort_key'] if x['sort_key'] is not None else 0)
        
        # extract sorted tokens and times
        _tokens = [enc['tokens'] for enc in encounter_data]
        
        # segment-level time (cumsum of days_since_prior_admission)
        if self.time_col:
            _times = [enc['time'] for enc in encounter_data]
            _times, _ = self.pad_segment_time(_times, do_cumsum=True)  # cumsum converts to absolute time
        else:
            _times = None
        
        # token-level time (time_offset_hours within each segment)
        if self.token_time_col:
            _token_times = [enc['token_times'] for enc in encounter_data]
        else:
            _token_times = None
        
        _tokens, _seg_attn = self.pad_segment(_tokens)
        _tokens = [["[CLS]"] + elem for elem in _tokens]

        item = get_batch_encoding(self.tokenizer, _tokens)
        item["segment_attention_mask"] = _seg_attn
        
        if _times is not None:
            item["segment_time"] = _times  # segment time (for CSE RoPE)
        
        if _token_times is not None:
            item["token_time"] = self.pad_token_time(_token_times)  # token time (for SWE RoPE)

        # TODO: if supervised task cohort is provided, add labels to the item

        return item

    def pad_segment(self, tokens):
        if len(tokens) > self.max_seg:
            seg_attn = torch.ones(self.max_seg)
            tokens = tokens[: self.max_seg]
        else:
            seg_attn = torch.cat(
                [torch.ones(len(tokens)), torch.zeros(self.max_seg - len(tokens))]
            )
            tokens = tokens + [["[PAD]"]] * (self.max_seg - len(tokens))
        return tokens, seg_attn
    
    def pad_segment_time(self, times, do_cumsum=True):
        """
        Pad/truncate time series to max_seg
        
        Args:
            times: Relative time interval list (e.g., days_since_prior_order)
            do_cumsum: Whether to use cumsum to convert to absolute time (time since first visit)
        
        Returns:
            torch.Tensor: Time feature tensor
        """
        import torch
        import numpy as np
        
        # handle NaN/None (first visit/order)
        times_clean = [0.0 if (t is None or np.isnan(t)) else float(t) for t in times]
        
        # cumsum converts to absolute time
        if do_cumsum:
            times_cumsum = np.cumsum(times_clean).tolist()
        else:
            times_cumsum = times_clean
        
        # Pad/truncate
        if len(times_cumsum) > self.max_seg:
            times_final = times_cumsum[: self.max_seg]
        else:
            # pad with last time value (representing no future visits)
            last_time = times_cumsum[-1] if times_cumsum else 0.0
            times_final = times_cumsum + [last_time] * (self.max_seg - len(times_cumsum))
        
        return torch.tensor(times_final, dtype=torch.float32), None
    
    def pad_token_time(self, token_times_list):
        """
        Pad/truncate token-level time (time_offset_hours) to (max_seg, max_seq_len)
        
        Args:
            token_times_list: List of lists, each inner list contains token times for one segment
        
        Returns:
            torch.Tensor: shape (max_seg, max_seq_len)
        """
        import torch
        import numpy as np
        
        # Truncate/pad segments
        if len(token_times_list) > self.max_seg:
            token_times_list = token_times_list[:self.max_seg]
        else:
            # Pad with empty lists
            token_times_list = token_times_list + [[]] * (self.max_seg - len(token_times_list))
        
        # Pad each segment's tokens
        padded_times = []
        for token_times in token_times_list:
            # Clean NaN/None
            times_clean = [0.0 if (t is None or (isinstance(t, float) and np.isnan(t))) else float(t) for t in token_times]
            
            # Add [CLS] token time (use 0.0 or the first token's time)
            if times_clean:
                cls_time = 0.0  # [CLS] token has no actual time, use 0
                times_clean = [cls_time] + times_clean
            else:
                times_clean = [0.0]  # Empty segment
            
            # Truncate/pad to max_seq_len
            if len(times_clean) > self.max_seq_len:
                times_clean = times_clean[:self.max_seq_len]
            else:
                # Pad with last time or 0
                pad_val = times_clean[-1] if times_clean else 0.0
                times_clean = times_clean + [pad_val] * (self.max_seq_len - len(times_clean))
            
            padded_times.append(times_clean)
        
        return torch.tensor(padded_times, dtype=torch.float32)


def random_masking(input_ids: torch.Tensor, tokenizer: Tokenizer, mlm_probability=0.15):
    labels = input_ids.clone()
    device = input_ids.device

    # Step 1: Pick mask positions (ignore <bos>, <pad>, <eos>)
    probability_matrix = torch.full(labels.shape, mlm_probability, device=device)
    special_mask = torch.isin(labels, torch.tensor(SPECIAL_TOKEN_IDS, device=device))
    probability_matrix.masked_fill_(special_mask, value=0.0)

    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # only masked positions contribute to loss

    # Step 2: Apply 80/10/10
    # 80% -> [MASK]
    indices_replaced = (
        torch.bernoulli(torch.full(labels.shape, 0.8, device=device)).bool()
        & masked_indices
    )
    input_ids[indices_replaced] = tokenizer.token_to_id("[MASK]")

    # 10% -> random token
    indices_random = (
        torch.bernoulli(torch.full(labels.shape, 0.5, device=device)).bool()
        & masked_indices
        & ~indices_replaced
    )
    random_words = torch.randint(
        tokenizer.get_vocab_size(), labels.shape, dtype=torch.int64, device=device
    )
    input_ids[indices_random] = random_words[indices_random]

    # 10% -> unchanged

    return input_ids, labels
