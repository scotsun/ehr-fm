import torch
from torch.utils.data import Dataset
from tokenizers import Tokenizer

from src.tokenizer import get_batch_encoding, SPECIAL_TOKEN_IDS


class EHRDataset(Dataset):
    def __init__(
        self,
        data,
        tokenizer: Tokenizer,
        supervised_task_cohort=None,
        max_seg: int = 32,
        max_seq_len: int = 512,  # 调整为512以适应MIMIC-IV数据（平均236个事件/就诊）
        patient_id_col: str = "patient_id",
        enc_id_col: str = "visit_id",
        token_col: str = "code",
        time_col: str = None,  # 时间间隔列（如 days_since_prior_order）需要 cumsum
        sort_col: str = None,  # 排序列（如 order_number, visit_seq, admittime）
        token_time_col: str = None,  # token级别时间列（如 time_offset_hours）用于 SWE 的 RoPE
    ):
        if supervised_task_cohort is not None:
            data = data[
                data[patient_id_col].isin(supervised_task_cohort[patient_id_col])
            ]

        self.data = data.groupby(patient_id_col)
        self.patient_id = list(self.data.groups.keys())

        self.tokenizer = tokenizer
        self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]", length=max_seq_len)
        self.tokenizer.enable_truncation(max_length=max_seq_len)

        self.max_seg = max_seg
        self.max_seq_len = max_seq_len
        self.enc_id_col = enc_id_col
        self.token_col = token_col
        self.time_col = time_col
        self.sort_col = sort_col if sort_col else time_col  # 默认用 time_col 排序
        self.token_time_col = token_time_col  # token级别时间（time_offset_hours）

    def __len__(self):
        return self.data.ngroups

    def __getitem__(self, index):
        """
        Get a patient's EHR data.

        Args:
            index (int): The index of the patient.

        Returns:
            dict: A dictionary containing the patient's EHR data.
            Keys:
                input_ids (torch.Tensor): The input IDs of the patient's EHR data.
                attention_mask (torch.Tensor): The attention mask of the patient's EHR data.
                segment_attention_mask (torch.Tensor): The segment attention mask of the patient's EHR data.
        """
        _patid = self.patient_id[index]
        _patid_data = self.data.get_group(_patid)
        
        # 按就诊/订单分组并提取 tokens 和时间
        grouped = _patid_data.groupby(self.enc_id_col)
        
        # 提取每个 encounter 的信息
        encounter_data = []
        for enc_id, group in grouped:
            tokens = group[self.token_col].to_list()
            time_val = group[self.time_col].iloc[0] if self.time_col else None
            sort_val = group[self.sort_col].iloc[0] if self.sort_col else None
            # 提取 token 级别的时间（如 time_offset_hours）
            token_times = group[self.token_time_col].to_list() if self.token_time_col else None
            encounter_data.append({
                'enc_id': enc_id,
                'tokens': tokens,
                'time': time_val,
                'sort_key': sort_val,
                'token_times': token_times  # token级别时间
            })
        
        # 按 sort_col 排序（如 order_number, visit_seq, admittime）
        if self.sort_col:
            encounter_data.sort(key=lambda x: x['sort_key'] if x['sort_key'] is not None else 0)
        
        # 提取排序后的 tokens 和 times
        _tokens = [enc['tokens'] for enc in encounter_data]
        
        # Segment级别的时间（cumsum of days_since_prior_admission）
        if self.time_col:
            _times = [enc['time'] for enc in encounter_data]
            _times, _ = self.pad_segment_time(_times, do_cumsum=True)  # cumsum 转换为绝对时间
        else:
            _times = None
        
        # Token级别的时间（time_offset_hours within each segment）
        if self.token_time_col:
            _token_times = [enc['token_times'] for enc in encounter_data]
        else:
            _token_times = None
        
        _tokens, _seg_attn = self.pad_segment(_tokens)
        _tokens = [["[CLS]"] + elem for elem in _tokens]

        item = get_batch_encoding(self.tokenizer, _tokens)
        item["segment_attention_mask"] = _seg_attn
        
        if _times is not None:
            item["segment_time"] = _times  # Segment时间（用于CSE的RoPE）
        
        if _token_times is not None:
            item["token_time"] = self.pad_token_time(_token_times)  # Token时间（用于SWE的RoPE）

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
        Pad/truncate 时间序列到 max_seg
        
        Args:
            times: 相对时间间隔列表（如 days_since_prior_order）
            do_cumsum: 是否使用 cumsum 转换为绝对时间（time since first visit）
        
        Returns:
            torch.Tensor: 时间特征张量
        """
        import torch
        import numpy as np
        
        # 处理 NaN/None（第一次就诊/订单）
        times_clean = [0.0 if (t is None or np.isnan(t)) else float(t) for t in times]
        
        # cumsum 转换为绝对时间
        if do_cumsum:
            times_cumsum = np.cumsum(times_clean).tolist()
        else:
            times_cumsum = times_clean
        
        # Pad/truncate
        if len(times_cumsum) > self.max_seg:
            times_final = times_cumsum[: self.max_seg]
        else:
            # 用最后一个时间值填充（表示未来没有新的就诊）
            last_time = times_cumsum[-1] if times_cumsum else 0.0
            times_final = times_cumsum + [last_time] * (self.max_seg - len(times_cumsum))
        
        return torch.tensor(times_final, dtype=torch.float32), None
    
    def pad_token_time(self, token_times_list):
        """
        Pad/truncate token级别的时间（time_offset_hours）到 (max_seg, max_seq_len)
        
        Args:
            token_times_list: List of lists, 每个内层list是一个segment的token时间
        
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
                cls_time = 0.0  # [CLS] token 没有实际时间，用0
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

    # 10% -> unchanged (do nothing)

    return input_ids, labels
