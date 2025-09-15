import torch.nn as nn

from .attention import MultiHeadAttentionBlock
from .ffn import FFNSwiGLUBlock, FFNLUBlock
from .residual import ResidualConnection


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        h: int,
        with_rope: bool,
        dropout: float = 0,
        norm_type: str = "layer",
        ffn_type: str = "swiglu",
        attn_backend: str = "base",
    ):
        """
        d_ff = 4 * d_model
        but *GLU typically scale down by 2/3 to previous parameter size
        """
        super().__init__()
        self.self_attn_block = MultiHeadAttentionBlock(
            d_model, h, with_rope, attn_backend
        )
        self.ffn_block = (
            FFNSwiGLUBlock(d_model, d_ff)
            if ffn_type == "swiglu"
            else FFNLUBlock(d_model, d_ff, ffn_type)
        )
        self.residual_connection = ResidualConnection(d_model, dropout, norm_type)

    def forward(self, x, time, mask):
        x = self.residual_connection(x, lambda x: self.self_attn_block(x, time, mask))
        x = self.residual_connection(x, self.ffn_block)
        return x
