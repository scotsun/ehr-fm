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
        max_seq_len: int = 64,
        patient_id_col: str = "patient_id",
        enc_id_col: str = "visit_id",
        token_col: str = "code",
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
        _tokens = list(
            map(
                lambda x: x[1].to_list(),
                _patid_data.groupby(self.enc_id_col)[self.token_col],
            )
        )
        _tokens, _seg_attn = self.pad_segment(_tokens)
        _tokens = [["[CLS]"] + elem for elem in _tokens]

        item = get_batch_encoding(self.tokenizer, _tokens)
        item["segment_attention_mask"] = _seg_attn

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


def random_masking(input_ids: torch.Tensor, tokenizer: Tokenizer, mlm_probability=0.15):
    labels = input_ids.clone()

    # Step 1: Pick mask positions (ignore <bos>, <pad>, <eos>)
    probability_matrix = torch.full(labels.shape, mlm_probability)
    special_mask = torch.isin(labels, torch.tensor(SPECIAL_TOKEN_IDS))
    probability_matrix.masked_fill_(special_mask, value=0.0)

    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # only masked positions contribute to loss

    # Step 2: Apply 80/10/10
    # 80% -> [MASK]
    indices_replaced = (
        torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    )
    input_ids[indices_replaced] = tokenizer.token_to_id("[MASK]")

    # 10% -> random token
    indices_random = (
        torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
        & masked_indices
        & ~indices_replaced
    )
    random_words = torch.randint(
        tokenizer.get_vocab_size(), labels.shape, dtype=torch.int64
    )
    input_ids[indices_random] = random_words[indices_random]

    # 10% -> unchanged (do nothing)

    return input_ids, labels
