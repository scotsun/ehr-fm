import torch

from transformers import PreTrainedModel, PretrainedConfig

from src.models.base import FMBase, FMBaseWithHeads


def is_model_half(model: torch.nn.Module):
    """Check if the model is half precision"""
    for param in model.parameters():
        if param.dtype != torch.half:
            return False
    return True


def build_model(
    cfg: PretrainedConfig, model_class: str, device: str
) -> PreTrainedModel:
    """Build the model"""
    match model_class:
        case "FMBase":
            model = FMBase(cfg)
        case "FMBaseWithHeads":
            model = FMBaseWithHeads(cfg)
        case _:
            raise ValueError(f"Unknown model class: {model_class}")
    model.to(device)
    return model
