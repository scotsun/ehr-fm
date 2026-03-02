import numpy as np
import torch
import mlflow

from torch.nn import CrossEntropyLoss, KLDivLoss
from mlflow.types import TensorSpec, Schema
from mlflow.models import ModelSignature
from transformers import PreTrainedModel
from tokenizers import Tokenizer

from src.models import FMConfig
from src.loss.softclt import SoftCLT
from src.trainers import EarlyStopping, Trainer, pt_trainers


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


def make_bert_signature(cfg: FMConfig) -> ModelSignature:
    max_seq = cfg.dataset["max_seq"]
    input_scheme = Schema(
        [
            TensorSpec(np.dtype(np.int64), [-1, max_seq], name="input_ids"),
            TensorSpec(np.dtype(np.int64), [-1, max_seq], name="attention_mask"),
            TensorSpec(np.dtype(np.bool), [-1, max_seq], name="t"),
        ]
    )
    output_scheme = Schema(
        [
            TensorSpec(
                np.dtype(np.float32),
                [-1, max_seq, cfg.vocab_size],
                name="logits",
            ),
            TensorSpec(
                np.dtype(np.float32),
                [-1, max_seq, cfg.d_model],
                name="last_hidden_state",
            ),
        ]
    )
    signature = ModelSignature(inputs=input_scheme, outputs=output_scheme)
    return signature


def make_fm_signature(cfg: FMConfig) -> ModelSignature:
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


def build_trainer(
    cfg: FMConfig, fm: PreTrainedModel, tk: Tokenizer, device: torch.device
):
    model_type = fm.model_type

    trainer_args = {
        "model": fm,
        "tokenizer": tk,
        "optimizer": torch.optim.AdamW(fm.parameters(), lr=cfg.trainer["lr"]),
        "early_stopping": EarlyStopping(
            patience=cfg.trainer["early_stopping_patience"],
            mode=cfg.trainer["early_stopping_mode"],
        ),
        "verbose_period": 1,
        "device": device,
        "trainer_args": cfg.trainer,
    }

    match model_type:
        case "fm-bert":
            trainer_class = pt_trainers.BertTrainer
            signature = make_bert_signature(cfg)
            criterions = {"cross_entropy": CrossEntropyLoss(ignore_index=-100)}
        case "fm-base":
            trainer_class = pt_trainers.BaseTrainer
            signature = make_fm_signature(cfg)
            criterions = {"cross_entropy": CrossEntropyLoss(ignore_index=-100)}
        case "fm-longformer":
            trainer_class = pt_trainers.LongformerTrainer
            signature = make_bert_signature(cfg)
            criterions = {"cross_entropy": CrossEntropyLoss(ignore_index=-100)}
        case "fm-base-with_heads":
            trainer_class = pt_trainers.BaseWithHeadsTrainer
            signature = make_fm_signature(cfg)
            criterions = {
                "cross_entropy": CrossEntropyLoss(ignore_index=-100),
                "kl_div": KLDivLoss(reduction="batchmean"),
            }
        case "fm-base-with_softclt":
            trainer_class = pt_trainers.BaseWithSoftCLTTrainer
            signature = make_fm_signature(cfg)
            criterions = {
                "cross_entropy": CrossEntropyLoss(ignore_index=-100),
                "softclt": SoftCLT(**cfg.trainer["softclt"]),
            }
        case _:
            raise ValueError(f"Unknown model type: {model_type}")

    trainer = trainer_class(
        criterions=criterions,
        model_signature=signature,
        **trainer_args,
    )

    return trainer
