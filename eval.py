import torch
import mlflow

from torch.utils.data import DataLoader
from tokenizers import Tokenizer

from src.utils.data_utils import SeqSet
from src.models import FMConfig
from src.utils.model_utils import build_model
from src.utils.train_utils import (
    load_cfg,
    build_trainer,
)

device = torch.device("cuda")

tk = Tokenizer.from_file("./dataset/instacart/data/tk.json")

cfg_dict = load_cfg("./config/instacart_base.yaml")
tk = Tokenizer.from_file(f"./{cfg_dict['model']['tokenizer']}")

cfg = FMConfig(
    vocab_size=tk.get_vocab_size(),
    dataset=cfg_dict["dataset"],
    trainer=cfg_dict["trainer"],
    **cfg_dict["model"],
)
model = build_model(cfg, "FMBase", device)
trainer = build_trainer(cfg, model, tk, device)


def time_operation(x):
    return x["t"]


instacart = SeqSet(
    tokenizer=tk,
    data_root="./dataset/instacart/data",
    data_folder="./dataset/instacart/data/instacart.parquet",
    split="test",
    max_seq=64,
    max_set_size=32,
    downstream_task_cohort=None,
    outcome_vars=None,
    time_operation=time_operation,
    seq_id_col="user_id",
    set_id_col="order_number",
    token_col="product_id",
    additional_cols=["t"],
)
dataloader = DataLoader(instacart, batch_size=8, shuffle=True)


run_id = "aa2d82f3a03a4cf496524f1be053772e"
best_model = mlflow.pytorch.load_model(
    f"runs:/{run_id}/best_model", map_location=torch.device("cuda")
)
trainer.model = best_model
rlt = trainer.evaluate(dataloader, verbose=True)
print(rlt)
