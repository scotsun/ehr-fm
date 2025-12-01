import torch
import pandas as pd

from concurrent.futures import ThreadPoolExecutor
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Split
from pathlib import Path


SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[MASK]"]
SPECIAL_TOKEN_IDS = [0, 1, 2, 3]

PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"

PAD_ID = 0
UNK_ID = 1


def _process_subdir(args):
    datafolder, subdir, token_col = args
    record = pd.read_parquet(
        f"{datafolder}/{subdir}", engine="pyarrow", dtype_backend="pyarrow"
    ).astype({token_col: str})  # make a raw token is a string
    return ";".join(record[token_col].tolist())


def get_all_records(
    datafolder: str, metadata_file_path: str, max_worker: int, token_col: str
):
    datasets_subdirs = pd.read_csv(metadata_file_path)["subdir"].values
    with ThreadPoolExecutor(max_workers=max_worker) as executor:
        for codes_str in executor.map(
            _process_subdir,
            [(datafolder, subdir, token_col) for subdir in datasets_subdirs],
        ):
            yield codes_str


def get_tokenizer(config) -> Tokenizer:
    tokenizer_path = Path(config["tokenizer_path"])
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token=UNK_TOKEN))
        tokenizer.pre_tokenizer = Split(pattern=";", behavior="removed")
        tokenizer.enable_padding(pad_id=PAD_ID, pad_token=PAD_TOKEN)
        trainer = WordLevelTrainer(
            vocab_size=config["vocab_size"],
            special_tokens=SPECIAL_TOKENS,
            min_frequency=config["min_frequency"],
            show_progress=True,
        )
        tokenizer.train_from_iterator(
            get_all_records(
                config["data_folder"],
                config["metadata_file_path"],
                config["max_worker"],
                config["token_col"],
            ),
            trainer=trainer,
            length=config["length"],
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
    attention_mask = (input_ids != 0).to(torch.bool)
    return {"input_ids": input_ids, "attention_mask": attention_mask}
