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

    def encode_time(self, time: torch.Tensor):
        # time: (b, seq_len)
        device = time.device
        # (seq_len, d/2)
        theta = 1.0 / (self.base ** (torch.arange(0, self.d, 2).float() / self.d)).to(
            device
        )

        # Compute angle in FP32 to avoid precision issues with large time values
        # Then convert back to input dtype for consistency
        original_dtype = time.dtype
        time_fp32 = time.float()
        theta_fp32 = theta.float()

        angle = einsum("bi,j -> bij", time_fp32, theta_fp32)  # shape: (b, seq_len, d/2)
        angle = torch.cat([angle, angle], dim=-1)  # shape: (b, seq_len, d)

        # Compute cos/sin in FP32 for numerical stability
        cos_angle = torch.cos(angle).to(original_dtype)
        sin_angle = torch.sin(angle).to(original_dtype)

        # (b, 1, seq_len, d)
        return cos_angle[:, None, :, :], sin_angle[:, None, :, :]

    def forward(self, x: torch.Tensor, time: torch.Tensor | None = None):
        # x: (batch, h, seq_len, d)
        if time is None:
            time = torch.arange(x.shape[2], device=x.device).float()
            time = time.repeat(x.shape[0], 1)

        cos_rotery, sin_rotery = self.encode_time(time=time)
        x_rope, x_pass = x[..., : self.d], x[..., self.d :]
        neg_half = torch.cat(
            [-x_rope[..., self.d // 2 :], x_rope[..., : self.d // 2]], dim=-1
        )

        # Perform RoPE rotation in FP32 to prevent cumulative precision errors
        # FP16 multiplication accumulates errors over many training steps
        original_dtype = x_rope.dtype
        x_rope_rotated = (
            x_rope.float() * cos_rotery.float() + neg_half.float() * sin_rotery.float()
        ).to(original_dtype)

        return torch.cat([x_rope_rotated, x_pass], dim=-1)


class T2V(nn.Module):
    """
    Time2Vec: learnable time encoding.
    Input: (batch, max_seg, max_seq_len) -> Output: (batch, max_seg, max_seq_len, d_model)
    """

    def __init__(self, d_model: int, scale: float = 1.0):
        super().__init__()
        self.d_model = d_model
        self.scale = scale
        self.w0 = nn.Parameter(torch.randn(1, 1) * 0.02)
        self.b0 = nn.Parameter(torch.zeros(1, 1))
        self.w = nn.Parameter(torch.randn(1, d_model - 1) * 0.02)
        self.b = nn.Parameter(torch.zeros(1, d_model - 1))

    def forward(self, t):
        original_shape = t.shape
        t = t.reshape(-1, 1)
        v0 = self.scale * t @ self.w0 + self.b0
        v = torch.sin(self.scale * t @ self.w + self.b)
        out = torch.cat([v0, v], dim=-1)
        return out.reshape(*original_shape, self.d_model)
