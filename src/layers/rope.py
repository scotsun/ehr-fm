import torch
import torch.nn as nn
from torch import einsum


class RoPE(nn.Module):
    """
    Rotary Positional Embedding (RoPE)
    Input shape: (batch, h, seq_len, d)
    Output shape: (batch, h, seq_len, d)
    """

    def __init__(self, d: int, base: float = 1e4):
        super().__init__()
        self.base = base
        self.d = d  # d = r * d_model, r is typically 1
        self.cos_cached = None
        self.sin_cached = None

    def _build_cache(self, seq_len: int, device: torch.device):
        # (seq_len, d/2)
        theta = 1.0 / (self.base ** (torch.arange(0, self.d, 2).float() / self.d)).to(
            device
        )
        # (seq_len, d/2)
        position = torch.arange(seq_len, device=device).float()

        angle = einsum("i,j -> ij", position, theta)
        angle = torch.cat([angle, angle], dim=-1)  # (seq_len, d)

        # (1, 1, seq_len, d)
        self.cos_cached = torch.cos(angle)[None, None, :, :]
        self.sin_cached = torch.sin(angle)[None, None, :, :]

    def forward(self, x: torch.Tensor):
        # x: (batch, h, seq_len, d)
        self._build_cache(seq_len=x.shape[-2], device=x.device)
        x_rope, x_pass = x[..., : self.d], x[..., self.d :]
        neg_half = torch.cat(
            [-x_rope[..., self.d // 2 :], x_rope[..., : self.d // 2]], dim=-1
        )
        x_rope = x_rope * self.cos_cached + neg_half * self.sin_cached
        return torch.cat([x_rope, x_pass], dim=-1)
