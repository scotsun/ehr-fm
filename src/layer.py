import math
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, h: int, output_only: bool):
        super().__init__()
        self.d_model = d_model
        self.h = h
        self.output_only = output_only
        assert d_model % h == 0, "d_model must be divisible by h"
        self.d_k = d_model // h  # dimension of each head
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

    @staticmethod
    def attention(query, key, value, mask):
        # this can be replaced
        d_k = query.shape[-1]
        # transpose matmul
        attention_scores = einsum("bhqd, bhkd->bhqk", query, key) / math.sqrt(d_k)
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1)
        # matmul
        x = einsum("bhqk, bhkd->bhqd", attention_scores, value)
        return x, attention_scores

    def forward(self, q, k, v, mask):
        query = self.w_q(q)  # (batch, seq_len, d_model)
        key = self.w_k(k)  # (batch, seq_len, d_model)
        value = self.w_v(v)  # (batch, seq_len, d_model)
        # chunk d_model from to h * d_k
        # (batch, seq_len, d_model) -> (batch, seq_len, h, d_k) -> (batch, h, seq_len, d_k)
        query = self._reshape(query)
        key = self._reshape(key)
        value = self._reshape(value)
        # (batch, 1, 1, seq_len) which will broadcast to all `heads` and `queries`
        mask = mask[:, None, None, :]

        x, attention_scores = self.attention(query, key, value, mask)
        # (batch, h, seq_len, d_k) -> (batch, seq_len, h, d_k) -> (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.d_model)
        o = self.w_o(x)
        if self.output_only:
            return o
        else:
            return o, attention_scores

    def _reshape(self, tensor):
        return tensor.view(
            tensor.shape[0], tensor.shape[1], self.h, self.d_k
        ).transpose(1, 2)


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


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        h: int,
        dropout: float = 0,
        norm_type: str = "layer",
        ffn_type: str = "swiglu",
    ):
        """
        d_ff = 4 * d_model
        but *GLU typically scale down by 2/3 to previous parameter size
        """
        super().__init__()
        self.self_attn_block = MultiHeadAttentionBlock(d_model, h, output_only=True)
        self.ffn_block = (
            FFNSwiGLUBlock(d_model, d_ff)
            if ffn_type == "swiglu"
            else FFNLUBlock(d_model, d_ff, ffn_type)
        )
        self.residual_connection = ResidualConnection(d_model, dropout, norm_type)

    def forward(self, x, mask):
        x = self.residual_connection(x, lambda x: self.self_attn_block(x, x, x, mask))
        x = self.residual_connection(x, self.ffn_block)
        return x


class HierarchicalTransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        h: int,
        dropout=0,
        norm_type="layer",
        ffn_type="swiglu",
    ):
        super().__init__()
        # segment-wise encoder
        self.swe = TransformerBlock(d_model, d_ff, h, dropout, norm_type, ffn_type)
        # cross-segment encoder
        self.cse = TransformerBlock(d_model, d_ff, h, dropout, norm_type, ffn_type)

    def forward(self, x, token_mask, seg_mask):
        # input x'shape: (batch, max_seg, max_seq_len, d_model)
        # token_mask'shape: (batch, max_seg, max_seq_len)
        # seg_mask'shape: (batch, max_seg)

        # therefore, x is truncated at both seg and seq_len level
        # segment-wise encoding
        seg_hidden_state = self.swe(
            x.reshape(-1, x.shape[2], x.shape[3]).contiguous(),
            token_mask.reshape(-1, token_mask.shape[2]).contiguous(),
        )
        seg_hidden_state = seg_hidden_state.reshape(
            x.shape[0], x.shape[1], x.shape[2], x.shape[3]
        ).contiguous()

        # cross-segment encoding
        # cls embeddings from all seg: (batch, max_seg, d_model)
        seg_cls_hidden_state = seg_hidden_state[:, :, 0, :].clone()
        seg_cls_hidden_state = self.cse(seg_cls_hidden_state, seg_mask)

        # combine
        seg_hidden_state[:, :, 0, :] = seg_cls_hidden_state

        return seg_hidden_state
