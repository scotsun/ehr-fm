import os
import yaml
import numpy as np
import torch
import torch.distributed as dist
import mlflow

from mlflow.types import TensorSpec, Schema
from mlflow.models import ModelSignature
from datetime import timedelta

from src.models import FMConfig
from src.trainer import Trainer


def load_cfg(cfg_path) -> dict:
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def setup_training():
    # check if in ddp env
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        rank = int(os.environ["RANK"])

        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", timeout=timedelta(seconds=600))

        return local_rank, world_size, rank, device, True
    else:
        local_rank = 0
        world_size = 1
        rank = 0

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.cuda.set_device(local_rank)

        return local_rank, world_size, rank, device, False


def setup_mlflow_tracked_fit(
    mlflow_uri: str,
    experiment_name: str,
    run_name: str | None,
    rank: int,
    is_distributed: bool,
    cfg: FMConfig,
    trainer: Trainer,
    train_loader: torch.utils.data.DataLoader,
    valid_loader: torch.utils.data.DataLoader,
):
    if not is_distributed or rank == 0:
        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params(cfg.to_diff_dict())
            trainer.fit(
                epochs=cfg.trainer["epochs"],
                train_loader=train_loader,
                valid_loader=valid_loader,
            )
    else:
        trainer.fit(
            epochs=cfg.trainer["epochs"],
            train_loader=train_loader,
            valid_loader=valid_loader,
        )


def make_fmbase_signature(cfg: FMConfig) -> ModelSignature:
    max_seq = cfg.dataset["max_seq"]
    max_set_size = cfg.dataset["max_set_size"]
    input_scheme = Schema(
        [
            TensorSpec(
                np.dtype(np.int64), [-1, max_seq, max_set_size], name="input_ids"
            ),
            TensorSpec(
                np.dtype(np.bool), [-1, max_seq, max_set_size], name="attention_mask"
            ),
            TensorSpec(np.dtype(np.bool), [-1, max_seq], name="set_attention_mask"),
            TensorSpec(np.dtype(np.float32), [-1, max_seq, max_set_size], name="t"),
        ]
    )
    output_scheme = Schema(
        [
            TensorSpec(
                np.dtype(np.float32),
                [-1, max_seq, max_set_size, cfg.vocab_size],
                name="logits",
            ),
            TensorSpec(
                np.dtype(np.float32),
                [-1, max_seq, max_set_size, cfg.d_model],
                name="last_hidden_state",
            ),
        ]
    )
    signature = ModelSignature(inputs=input_scheme, outputs=output_scheme)
    return signature
