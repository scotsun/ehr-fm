import torch
from torch.utils.data import Dataset
from tokenizers import Tokenizer

from src.tokenizer import get_batch_encoding


class EHRDataset(Dataset):
    def __init__(
        self,
        data,
        tokenizer: Tokenizer,
        max_seg: int,
        max_seq_len: int,
        patient_id_col: str,
        enc_id_col: str,
    ):
        self.data = data.groupby(patient_id_col)
        self.patient_id = list(self.data.groups.keys())
        self.tokenizer = tokenizer
        self.max_seg = max_seg
        self.max_seq_len = max_seq_len
        self.enc_id_col = enc_id_col

    def __len__(self):
        return self.data.ngroups

    def __getitem__(self, index):
        _patid = self.patient_id[index]
        _patid_data = self.data.get_group(_patid)
        _tokens = list(
            map(lambda x: x[1].to_list(), _patid_data.groupby(self.enc_id_col)["code"])
        )
        _tokens, _seg_attn = self.pad_segment(_tokens)
        _tokens = [["[CLS]"] + elem for elem in _tokens]

        item = get_batch_encoding(self.tokenizer, _tokens)
        item["segment_attention_mask"] = _seg_attn

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


class Collator:
    def __init__(self):
        pass
