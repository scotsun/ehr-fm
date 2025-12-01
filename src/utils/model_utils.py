import torch

from transformers import PreTrainedModel, PretrainedConfig
from typing import Type


def is_model_half(model: torch.nn.Module):
    """Check if the model is half precision"""
    for param in model.parameters():
        if param.dtype != torch.half:
            return False
    return True


def build_model(
    cfg: PretrainedConfig, model_class: Type[PreTrainedModel], device: str
) -> PreTrainedModel:
    """Build the model"""
    model = model_class(cfg)
    model.to(device)
    return model
