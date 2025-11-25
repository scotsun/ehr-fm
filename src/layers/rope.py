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
        x_rope = x_rope * cos_rotery + neg_half * sin_rotery
        return torch.cat([x_rope, x_pass], dim=-1)


class T2V(nn.Module):
    """
    Time2Vec (T2V)
    Input shape: (batch, seq_len)
    Output shape: (batch, seq_len, d)
    """

    def __init__(self, d, scale, f=torch.sin):
        super().__init__()
        self.scale = scale
        self.f = f
        self.w0 = nn.Parameter(torch.rand(1, 1))
        self.b0 = nn.Parameter(torch.rand(1, 1))
        self.w = nn.Parameter(torch.rand(1, d - 1))
        self.b = nn.Parameter(torch.rand(1, d - 1))
        self.register_parameter("w0", self.w0)
        self.register_parameter("b0", self.b0)
        self.register_parameter("w", self.w)
        self.register_parameter("b", self.b)

    def forward(self, t):
        # x: (batch, 1)
        v0 = self.scale * t @ self.w0 + self.b0
        v = self.f(self.scale * t @ self.w + self.b)
        return torch.cat([v0, v], dim=-1)


def main():
    t2v = T2V(5, 1.0)
    x = torch.rand(2, 1)
    print(t2v(x).shape)
    print(t2v(x))


if __name__ == "__main__":
    main()
