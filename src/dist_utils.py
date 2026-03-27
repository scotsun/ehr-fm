"""Distributed training utilities for multi-GPU support via PyTorch DDP."""

import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from datetime import timedelta


def _dist_is_initialized():
    """Check if distributed training is active."""
    return dist.is_available() and dist.is_initialized()


def is_main_process():
    """Return True if this is rank 0 or non-distributed."""
    return not _dist_is_initialized() or dist.get_rank() == 0


def get_module(model: nn.Module) -> nn.Module:
    """Extract underlying module from DDP wrapper."""
    return model.module if isinstance(model, DDP) else model


def _broadcast_bool(flag: bool, device: torch.device) -> bool:
    """Broadcast boolean from rank 0 to all other ranks."""
    if not _dist_is_initialized():
        return flag
    flag_tensor = torch.tensor([1 if flag else 0], device=device, dtype=torch.int32)
    dist.broadcast(flag_tensor, src=0)
    return bool(flag_tensor.item())


def setup_training():
    """Initialize distributed training if launched via torchrun.

    Returns:
        (local_rank, world_size, rank, device, is_distributed)
    """
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
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)

        return local_rank, world_size, rank, device, False


def cleanup():
    """Destroy process group if distributed training is active."""
    if _dist_is_initialized():
        dist.destroy_process_group()