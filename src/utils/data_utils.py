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
        data_root: str,
        data_folder: str,
        max_seq: int,
        max_set_size: int,
        downstream_task_cohort: None | pd.DataFrame,
        outcome_vars: None | list[str],
        time_operation: Callable[[pd.DataFrame], pd.Series],
        seq_id_col: str,  # "patient_id", "user_id"
        set_id_col: str,  # "enc_id", "order_id"
        token_col: str,  # "code", "product_id"
        val_col: str | None = None,
        additional_cols: list[str] = [],
    ):
        self.data_root = data_root
        self.data_folder = data_folder
        self.downstream_task_cohort = downstream_task_cohort
        if self.downstream_task_cohort is not None:
            self.seq_ids = downstream_task_cohort[seq_id_col].values
        else:
            self.seq_ids = pd.read_csv(f"{self.data_root}/metadata.csv")[
                "subdir"
            ].values
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
        self.val_col = val_col
        self.additional_cols = additional_cols

    def __len__(self):
        return len(self.seq_ids)

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
        _seq_id_data[self.token_col] = _seq_id_data[self.token_col].astype(str)

        if self.downstream_task_cohort is not None and len(self.outcome_vars) > 0:
            _set_id = self.downstream_task_cohort.iloc[index][self.set_id_col]
            _stop_idx = _seq_id_data.loc[
                _seq_id_data[self.set_id_col] == _set_id
            ].index[0]
            _seq_id_data = _seq_id_data.iloc[:_stop_idx]

        _seq_id_data["t"] = self.time_operation(_seq_id_data)

        _grouped = _seq_id_data.groupby("t")

        _tokens = [
            ["[CLS]"] + _set_tokens.to_list()
            for _, _set_tokens in _grouped[self.token_col]
        ]
        if self.val_col is not None:
            _vs = [
                [-999] + _set_vs.fillna(-999).to_list()
                for _, _set_vs in _grouped[self.val_col]
            ]
        else:
            _vs = None
        _ts = [[t[0], *t] for t in (_set_ts.to_list() for _, _set_ts in _grouped["t"])]
        _tokens, _set_attn, _ts, _vs = self.pad_seq(_tokens, _ts, _vs)

        item = get_batch_encoding(self.tokenizer, _tokens)

        # item: dict
        # keys: input_ids, attention_mask, set_attention_mask, t, v
        item["set_attention_mask"] = _set_attn
        item["t"] = self.pad_set(self.max_set_size, _ts, type="t")
        if _vs is not None:
            item["v"] = self.pad_set(self.max_set_size, _vs, type="v")

        if self.downstream_task_cohort is not None and self.outcome_vars:
            _outrow = self.downstream_task_cohort.iloc[index][
                self.outcome_vars
            ].to_dict()
            for _outvar, _value in _outrow.items():
                item[_outvar] = torch.tensor(_value, dtype=torch.float32)

        return item

    def pad_seq(self, tokens, times, values):
        """pad/trunc the seq[set] to max_seq"""
        # tokens: 2d list
        # set_attn: 1d list
        # times: 2d list
        # values: 2d list
        n_sets = len(tokens)

        if n_sets > self.max_seq:  # truncation
            set_attn = torch.ones(self.max_seq, dtype=torch.bool)
            tokens = tokens[: self.max_seq]
            times = times[: self.max_seq]
            if values is not None:
                values = values[: self.max_seq]
        else:  # pad
            set_attn = torch.zeros(self.max_seq, dtype=torch.bool)
            set_attn[:n_sets] = True

            pad_length = self.max_seq - n_sets
            tokens = tokens + [["[PAD]"]] * pad_length
            times = times + [[-1.0]] * pad_length
            if values is not None:
                values = values + [[-99.0]] * pad_length

        return tokens, set_attn, times, values

    def pad_set(self, max_set_size: int, sets: list[list], type: str = "t"):
        match type:
            case "t":
                padding_value = -1.0
            case "v":
                padding_value = -99.0
            case _:
                raise ValueError("incorrect type")
        padded_list = [
            (s + [padding_value] * max_set_size)[:max_set_size] for s in sets
        ]
        return torch.tensor(padded_list, dtype=torch.float32)


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


class Seq(Dataset):
    def __init__(
        self,
        tokenizer: Tokenizer,
        data_root: str,
        data_folder: str,
        max_seq: int,
        downstream_task_cohort: None | pd.DataFrame,
        outcome_vars: None | list[str],
        time_operation: Callable[[pd.DataFrame], pd.Series],
        seq_id_col: str,  # "patient_id", "user_id"
        set_id_col: str,  # "enc_id", "order_id"
        token_col: str,  # "code", "product_id"
        val_col: str | None = None,
        additional_cols: list[str] = [],
    ):
        self.data_root = data_root
        self.data_folder = data_folder
        self.downstream_task_cohort = downstream_task_cohort
        if self.downstream_task_cohort is not None:
            self.seq_ids = downstream_task_cohort[seq_id_col].values
        else:
            self.seq_ids = pd.read_csv(f"{self.data_root}/metadata.csv")[
                "subdir"
            ].values
        self.tokenizer = tokenizer
        self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]", length=max_seq)
        self.tokenizer.enable_truncation(max_length=max_seq)

        self.max_seq = max_seq
        self.outcome_vars = outcome_vars
        self.time_operation = time_operation
        self.seq_id_col = seq_id_col
        self.set_id_col = set_id_col
        self.token_col = token_col
        self.val_col = val_col
        self.additional_cols = additional_cols

    def __len__(self):
        return len(self.seq_ids)

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
        _seq_id_data[self.token_col] = _seq_id_data[self.token_col].astype(str)

        if self.downstream_task_cohort is not None and len(self.outcome_vars) > 0:
            _set_id = self.downstream_task_cohort.iloc[index][self.set_id_col]
            _stop_idx = _seq_id_data.loc[
                _seq_id_data[self.set_id_col] == _set_id
            ].index[0]
            _seq_id_data = _seq_id_data.iloc[:_stop_idx]

        _seq_id_data["t"] = self.time_operation(_seq_id_data)

        _grouped = _seq_id_data.groupby("t")
        _tokens = sum(
            (["[CLS]"] + _set.to_list() for _, _set in _grouped[self.token_col]), []
        )
        if self.val_col is not None:
            _vs = sum(
                ([-999] + _set_vs.to_list() for _, _set_vs in _grouped[self.val_col]),
                [],
            )
        else:
            _vs = None
        _ts = sum(
            ([t[0], *t] for t in (_set_ts.to_list() for _, _set_ts in _grouped["t"])),
            [],
        )

        encoding = self.tokenizer.encode(_tokens, is_pretokenized=True)

        item = {
            "input_ids": torch.tensor(encoding.ids[: self.max_seq], dtype=torch.int64),
            "attention_mask": torch.tensor(
                encoding.attention_mask[: self.max_seq], dtype=torch.bool
            ),
            "t": torch.tensor(_ts[: self.max_seq], dtype=torch.float32),
        }
        if _vs is not None:
            item["v"] = torch.tensor(_vs[: self.max_seq], dtype=torch.float32)

        return item
