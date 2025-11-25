import torch
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
        # Compute multiplication in FP32 to avoid FP16 overflow
        # Two large FP16 values multiplied can exceed FP16 max (~65504)
        original_dtype = x.dtype
        hidden = (activated_gate.float() * up_output.float()).to(original_dtype)
        return self.linear_down(hidden)


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
