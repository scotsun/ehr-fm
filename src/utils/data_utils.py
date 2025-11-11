import torch
import pandas as pd
import pyarrow.parquet as pq

from typing import Callable
from torch.utils.data import Dataset
from tokenizers import Tokenizer

from src.tokenizer import get_batch_encoding, SPECIAL_TOKEN_IDS


class SeqSet(Dataset):
    def __init__(
        self,
        tokenizer: Tokenizer,
        data_folder: str,
        max_seq: int,
        max_set_size: int,
        downstream_task_cohort: None | pd.DataFrame,
        outcome_vars: None | list[str],
        time_operation: Callable[[pd.DataFrame], pd.Series],
        seq_id_col: str,  # "patient_id", "user_id"
        set_id_col: str,  # "enc_id", "order_id"
        token_col: str,  # "code", "product_id"
        additional_cols: list[str] = [],
    ):
        self.data_folder = data_folder
        self.downstream_task_cohort = downstream_task_cohort
        if self.downstream_task_cohort is not None:
            self.seq_ids = downstream_task_cohort[seq_id_col]
        else:
            self.seq_ids = pd.read_csv(f"{self.data_folder}/metadata.csv")
        self.tokenizer = tokenizer
        self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]", length=max_set_size)
        self.tokenizer.enable_truncation(max_length=max_set_size)

        self.max_seq = max_seq
        self.max_set_size = max_set_size
        self.outcome_vars = outcome_vars
        self.time_operation = time_operation
        self.seq_id_col = seq_id_col
        self.set_id_col = set_id_col
        self.token_col = token_col
        self.addtional_cols = additional_cols

    def __len__(self):
        return self.data.ngroups

    def __getitem__(self, index):
        """
        Get a Sequence of Sets of Tokens (i.e., patient's EHR) data.

        Args:
            index (int): The index of the patient.

        Returns:
            dict: A dictionary containing the SeqSet data.
            Keys:
                input_ids (torch.Tensor): The input token IDs of the SeqSet data.
                attention_mask (torch.Tensor): The token-level attention mask of the SeqSet data.
                set_attention_mask (torch.Tensor): The set-level attention mask of the SeqSet data.
        """
        if self.downstream_task_cohort is not None:
            _seq_id_dir = f"{self.seq_id_col}={self.seq_ids[index]}"
        else:
            _seq_id_dir = self.seq_ids[index]

        _seq_id_data = pq.read_table(
            f"{self.data_folder}/{_seq_id_dir}",
            columns=[self.set_id_col, self.token_col] + self.additional_cols,
        ).to_pandas()

        if self.downstream_task_cohort is not None and self.outcome_vars:
            _set_id = self.downstream_task_cohort.iloc[index][self.set_id_col]
            _stop_idx = _seq_id_data.loc[
                _seq_id_data[self.set_id_col] == _set_id
            ].index[0]
            _seq_id_data = _seq_id_data.iloc[:_stop_idx]

        _seq_id_data["t"] = self.time_operation(_seq_id_data)

        _grouped = _seq_id_data.groupby(self.set_id_col)
        _tokens = [_set_tokens.to_list() for _, _set_tokens in _grouped[self.token_col]]
        _ts = [_set_ts.to_list() for _, _set_ts in _grouped["t"]]

        _tokens, _seg_attn, _ts = self.pad_seq(_tokens, _ts)
        _tokens = [["[CLS]"] + elem for elem in _tokens]

        item = get_batch_encoding(self.tokenizer, _tokens)
        # item: dict
        # keys: input_ids, attention_mask, set_attention_mask, t
        item["set_attention_mask"] = _seg_attn
        item["t"] = pad_set(self.max_seq, _ts)

        return item

    def pad_seq(self, tokens, times):
        """pad/trunc the seq[set] to max_seq"""
        n_sets = len(tokens)

        if n_sets > self.max_seq:  # truncation
            set_attn = torch.ones(self.max_seq, dtype=torch.bool)
            tokens = tokens[: self.max_seq]
            times = times[: self.max_seq]
        else:  # pad
            set_attn = torch.zeros(self.max_seq, dtype=torch.bool)
            set_attn[:n_sets] = True

            pad_length = self.max_seq - n_sets
            tokens = tokens + [["[PAD]"]] * pad_length
            times = times + [[-1]] * pad_length

        # tokens: 2d list
        # set_attn: 1d list
        # times: 2d list
        return tokens, set_attn, times


def pad_set(max_set_size: int, sets, padding_value=-1):
    padded_list = [(s + [padding_value] * max_set_size)[:max_set_size] for s in sets]
    return torch.tensor(padded_list, dtype=torch.int64)


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
