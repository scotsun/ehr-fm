import os
import sys
from pathlib import Path
import torch
import torch.distributed as dist
import argparse

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, random_split, DistributedSampler

from tokenizers import Tokenizer
from setproctitle import setproctitle

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.data_utils import SeqSet
from src.models import FMConfig
from src.utils.model_utils import build_model
from src.utils.train_utils import (
    load_cfg,
    setup_training,
    build_trainer,
    setup_mlflow_tracked_fit,
)


def get_args():
    parser = argparse.ArgumentParser(description="Instacart FM")
    parser.add_argument("--backend-uri", type=str, default="./mlruns")
    parser.add_argument("--experiment-name", type=str)
    parser.add_argument("--run-name", type=str, default=None)
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    MLFLOW_URI = args.backend_uri
    EXPERIMENT_NAME = args.experiment_name
    RUN_NAME = args.run_name

    os.environ["OMP_NUM_THREADS"] = "1"
    local_rank, world_size, rank, device, is_distributed = setup_training()
    setproctitle(f"instacart-fm-pretrain-{local_rank}")
    print(f"local_rank: {local_rank}")
    print(f"process {rank}/{world_size} using device {device}\n")

    cfg_dict = load_cfg("./config/instacart_fm.yaml")
    tk = Tokenizer.from_file(f"./{cfg_dict['model']['tokenizer']}")

    cfg = FMConfig(
        vocab_size=tk.get_vocab_size(),
        dataset=cfg_dict["dataset"],
        trainer=cfg_dict["trainer"],
        **cfg_dict["model"],
    )
    model = build_model(cfg, "FMBase", device)

    trainer = build_trainer(cfg, model, tk, device)

    if is_distributed:
        trainer.model = DDP(model, device_ids=[rank])

    kwargs = cfg_dict["dataset"]
    kwargs["time_operation"] = lambda x: x["t"]
    instacart = SeqSet(
        tokenizer=tk,
        data_folder="/".join([cfg.dataset["data_root"], "instacart.parquet"]),
        downstream_task_cohort=None,
        outcome_vars=None,
        **kwargs,
    )
    train, valid = random_split(
        dataset=instacart,
        lengths=cfg.trainer["split"],
        generator=torch.Generator().manual_seed(42),
    )
    train_loader = DataLoader(
        dataset=train,
        batch_size=cfg.trainer["batch_size"],
        sampler=DistributedSampler(train),
        num_workers=8,
    )
    valid_loader = DataLoader(
        dataset=valid,
        batch_size=cfg.trainer["batch_size"],
        sampler=DistributedSampler(valid),
        num_workers=8,
    )

    setup_mlflow_tracked_fit(
        mlflow_uri=MLFLOW_URI,
        experiment_name=EXPERIMENT_NAME,
        run_name=RUN_NAME,
        rank=rank,
        is_distributed=is_distributed,
        cfg=cfg,
        trainer=trainer,
        train_loader=train_loader,
        valid_loader=valid_loader,
    )

    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
