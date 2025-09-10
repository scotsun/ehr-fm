import torch.nn as nn
import torch.nn.functional as F


class FFNSwiGLUBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.linear_gate = nn.Linear(d_model, d_ff)
        self.linear_up = nn.Linear(d_model, d_ff)
        self.linear_down = nn.Linear(d_ff, d_model)

    def forward(self, x):
        gate_output = self.linear_gate(x)
        up_output = self.linear_up(x)
        activated_gate = F.silu(gate_output)
        return self.linear_down(activated_gate * up_output)


class FFNLUBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, activation: str = "relu"):
        super().__init__()
        self.linear_up = nn.Linear(d_model, d_ff)
        self.linear_down = nn.Linear(d_ff, d_model)
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        else:
            raise ValueError(f"activation must be relu or gelu, got {activation}")

    def forward(self, x):
        return self.linear_down(self.activation(self.linear_up(x)))
