from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from pathlib import Path
from itertools import chain
import torch
from pandas.core.groupby.generic import DataFrameGroupBy

SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[MASK]"]


def get_all_records(grouped_dataset: DataFrameGroupBy, token_col: str):
    for _, group in grouped_dataset:
        yield from group[token_col].tolist()


def get_tokenizer(datasets, config) -> Tokenizer:
    grouped_datasets = [
        dataset.groupby(config["patient_id_col"]) for dataset in datasets
    ]
    tokenizer_path = Path(config["tokenizer_path"])

    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
        trainer = WordLevelTrainer(
            special_tokens=SPECIAL_TOKENS,
            min_frequency=config["min_frequency"],
            show_progress=True,
        )
        tokenizer.train_from_iterator(
            trainer=trainer,
            iterator=chain(
                *[
                    get_all_records(grouped, config["token_col"])
                    for grouped in grouped_datasets
                ]
            ),
        )
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))

    return tokenizer


def get_batch_encoding(tk: Tokenizer, records: list[list[str]]) -> torch.Tensor:
    encoding = tk.encode_batch_fast(records, is_pretokenized=True)
    input_ids = torch.tensor(
        list(map(lambda elem: elem.ids, encoding)),
        dtype=torch.int64,
    )
    attention_mask = (input_ids != 0).to(torch.int64)
    return {"input_ids": input_ids, "attention_mask": attention_mask}
