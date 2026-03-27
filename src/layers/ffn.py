import torch.nn as nn
import torch.nn.functional as F


class FFNSwiGLUBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        d_out: int | None = None,
        dropout: float = 0,
    ):
        super().__init__()
        if d_out is None:
            d_out = d_model
        self.linear_gate = nn.Linear(d_model, d_ff)
        self.linear_up = nn.Linear(d_model, d_ff)
        self.linear_down = nn.Linear(d_ff, d_out)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        gate_output = self.linear_gate(x)
        up_output = self.linear_up(x)
        activated_gate = F.silu(gate_output)
        if self.dropout.p > 0:
            activated_gate = self.dropout(activated_gate)
        return self.linear_down(activated_gate * up_output)  # SwiGLU


class FFNLUBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        ffn_type: str = "relu",
        d_out: int | None = None,
        dropout: float = 0,
    ):
        super().__init__()
        if d_out is None:
            d_out = d_model
        self.linear_up = nn.Linear(d_model, d_ff)
        self.linear_down = nn.Linear(d_ff, d_out)
        self.dropout = nn.Dropout(dropout)
        if ffn_type == "relu":
            self.activation = nn.ReLU()
        elif ffn_type == "gelu":
            self.activation = nn.GELU()
        else:
            raise ValueError(f"ffn_type must be relu or gelu, got {ffn_type}")

    def forward(self, x):
        x = self.linear_up(x)
        if self.dropout.p > 0:
            x = self.dropout(x)
        return self.linear_down(self.activation(x))
