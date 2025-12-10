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
        # Use separate residual connections for attention and FFN (standard practice)
        self.residual_connection_1 = ResidualConnection(d_model, dropout, norm_type)
        self.residual_connection_2 = ResidualConnection(d_model, dropout, norm_type)

    def forward(self, x, mask, time):
        x = self.residual_connection_1(
            x, lambda x: self.self_attn_block(x, x, x, mask, time)
        )
        x = self.residual_connection_2(x, self.ffn_block)
        return x
