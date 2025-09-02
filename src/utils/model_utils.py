import torch


def is_model_half(model: torch.nn.Module):
    """Check if the model is half precision"""
    for param in model.parameters():
        if param.dtype != torch.half:
            return False
    return True
