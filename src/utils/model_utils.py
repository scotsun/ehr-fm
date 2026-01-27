import torch

from transformers import PreTrainedModel, PretrainedConfig

from src.models.base import FMBase, FMBaseWithHeads
from src.models.bert import FMBert
from src.models.longformer import FMLongformer


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
        case "FMBert":
            model = FMBert(cfg)
        case "FMLongformer":
            model = FMLongformer(cfg)
        case "FMBase":
            model = FMBase(cfg)
        case "FMBaseWithHeads":
            model = FMBaseWithHeads(cfg)
        case "FMBaseWithSoftCLT":
            model = FMBase(cfg)
            model.model_type = "fm-base-with_softclt"
        case _:
            raise ValueError(f"Unknown model class: {model_class}")
    model.to(device)
    return model
