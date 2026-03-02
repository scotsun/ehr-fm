import os
import torch
import torch.nn as nn
import torch.distributed as dist

from torch.nn.parallel import DistributedDataParallel as DDP
from datetime import timedelta


def _dist_is_initialized():
    return dist.is_available() and dist.is_initialized()


def is_main_process():
    return not _dist_is_initialized() or dist.get_rank() == 0


def _get_module(model: nn.Module | DDP) -> nn.Module:
    return model.module if isinstance(model, DDP) else model


def _broadcast_bool(flag: bool, device: torch.device) -> bool:
    """
    single gpu: return the flag as is
    ddp: broadcast the flag from rank 0 to all other ranks, and return the broadcasted flag
    """
    if not _dist_is_initialized():
        return flag
    flag_tensor = torch.tensor([1 if flag else 0], device=device, dtype=torch.int32)
    dist.broadcast(flag_tensor, src=0)
    return bool(flag_tensor.item())


def _broadcast_float(value: float, device: torch.device) -> float:
    """
    single gpu: return the value as is
    ddp: broadcast the value from rank 0 to all other ranks, and return the broadcasted value
    """
    if not _dist_is_initialized():
        return value
    value_tensor = torch.tensor([value], device=device, dtype=torch.float32)
    dist.broadcast(value_tensor, src=0)
    return float(value_tensor.item())


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
