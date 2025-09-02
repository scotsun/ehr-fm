import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from torch.nn.attention import sdpa_kernel, SDPBackend


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


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, h: int, with_rope: bool):
        super().__init__()
        self.d_model = d_model
        self.h = h
        self.with_rope = with_rope

        assert d_model % h == 0, "d_model must be divisible by h"
        self.d_k = d_model // h  # dimension of each head
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        if self.with_rope:
            self.rope_q = RoPE(self.d_k)
            self.rope_k = RoPE(self.d_k)

    @staticmethod
    def attention(query, key, value, mask):
        # this can be replaced
        d_k = query.shape[-1]
        # transpose matmul
        attention_scores = einsum("bhqd, bhkd->bhqk", query, key) / math.sqrt(d_k)
        if mask is not None:
            # (batch, 1, 1, seq_len) which will broadcast to all `heads` and `queries`
            mask = mask[:, None, None, :]
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1)
        # matmul
        mh_out = einsum("bhqk, bhkd->bhqd", attention_scores, value)
        return mh_out

    @staticmethod
    def flash_attention(query, key, value, mask):
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
            mh_out = F.scaled_dot_product_attention(
                query, key, value, attn_mask=mask, dropout_p=0.0, is_causal=False
            )
        return mh_out

    def forward(self, q, k, v, mask):
        query = self.w_q(q)  # (batch, seq_len, d_model)
        key = self.w_k(k)  # (batch, seq_len, d_model)
        value = self.w_v(v)  # (batch, seq_len, d_model)
        # chunk d_model from to h * d_k
        # (batch, seq_len, d_model) -> (batch, seq_len, h, d_k) -> (batch, h, seq_len, d_k)
        query = self._reshape(query)
        key = self._reshape(key)
        value = self._reshape(value)

        if self.with_rope:
            query, key = self.rope_k(key), self.rope_q(query)

        mh_out = self.attention(query, key, value, mask)
        # (batch, h, seq_len, d_k) -> (batch, seq_len, h, d_k) -> (batch, seq_len, d_model)
        mh_out = (
            mh_out.transpose(1, 2).contiguous().view(mh_out.shape[0], -1, self.d_model)
        )
        o = self.w_o(mh_out)
        return o

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
        with_rope: bool,
        dropout: float = 0,
        norm_type: str = "layer",
        ffn_type: str = "swiglu",
    ):
        """
        d_ff = 4 * d_model
        but *GLU typically scale down by 2/3 to previous parameter size
        """
        super().__init__()
        self.self_attn_block = MultiHeadAttentionBlock(d_model, h, with_rope)
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
        self.swe = TransformerBlock(
            d_model, d_ff, h, False, dropout, norm_type, ffn_type
        )
        # cross-segment encoder
        self.cse = TransformerBlock(
            d_model, d_ff, h, True, dropout, norm_type, ffn_type
        )

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
