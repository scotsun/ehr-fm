import torch.nn as nn


class ResidualConnection(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0, norm_type: str = "layer"):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        if norm_type == "layer":
            self.norm = nn.LayerNorm(d_model)
        elif norm_type == "rms":
            self.norm = nn.RMSNorm(d_model)
        else:
            raise ValueError(f"norm_type must be layer or rms, got {norm_type}")

    def forward(self, x, sublayer):
        "sublayer: either MHA (or its variants) or FFN"
        # examples:
        # x = self.residual_connections(x, lambda x: self.self_attention_block(x, x, x, src_mask))
        # x = self.residual_connections(x, self.feed_forward_block)
        return x + self.dropout(sublayer(self.norm(x)))
