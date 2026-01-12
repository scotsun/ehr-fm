import math
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from torch.nn.attention import sdpa_kernel, SDPBackend

from .pe import RoPE
from .ffn import FFNSwiGLUBlock, FFNLUBlock


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, h: int, with_rope: bool, attn_backend: str):
        super().__init__()
        self.d_model = d_model
        self.h = h
        self.with_rope = with_rope

        assert d_model % h == 0, f"d_model {d_model} must be divisible by h {h}"
        self.d_k = d_model // h  # dimension of each head
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        if self.with_rope:
            self.rope_q = RoPE(self.d_k)
            self.rope_k = RoPE(self.d_k)
        match attn_backend:
            case "base":
                self._attn_func = self.attention
            case "efficient_attention":
                self._attn_func = self.efficient_attention
            case _:
                raise ValueError(
                    f"attn_backend must be base or efficient_attention, got {attn_backend}"
                )

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
    def efficient_attention(query, key, value, attn_mask):
        attn_mask = einsum("bq,bk -> bqk", attn_mask, attn_mask)[:, None, :, :]
        with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
            mh_out = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attn_mask, dropout_p=0.0, is_causal=False
            )
        return mh_out

    def forward(self, q, k, v, mask, t=None):
        query = self.w_q(q)  # (-1, seq_len, d_model)
        key = self.w_k(k)  # (-1, seq_len, d_model)
        value = self.w_v(v)  # (-1, seq_len, d_model)
        # chunk d_model from to h * d_k
        # (-1, seq_len, d_model) -> (-1, seq_len, h, d_k) -> (-1, h, seq_len, d_k)
        query = self._reshape(query)
        key = self._reshape(key)
        value = self._reshape(value)

        if self.with_rope:
            query, key = self.rope_q(query, t), self.rope_k(key, t)

        print(query.shape)

        mh_out = self._attn_func(query, key, value, mask)
        # (-1, h, seq_len, d_k) -> (-1, seq_len, h, d_k) -> (-1, seq_len, d_model)
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
        self.residual_connection0 = ResidualConnection(d_model, dropout, norm_type)
        self.residual_connection1 = ResidualConnection(d_model, dropout, norm_type)

    def forward(self, x, mask, t=None):
        x = self.residual_connection0(
            x, lambda x: self.self_attn_block(x, x, x, mask, t)
        )
        x = self.residual_connection1(x, self.ffn_block)
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
        attn_backend: str = "base",
        cs_pe: str = "pos",  # or time
    ):
        super().__init__()
        self.cs_pe = cs_pe
        # segment-wise encoder
        self.swe = TransformerBlock(
            d_model, d_ff, h, True, dropout, norm_type, ffn_type, attn_backend
        )
        # cross-segment encoder
        self.cse = TransformerBlock(
            d_model, d_ff, h, True, dropout, norm_type, ffn_type, attn_backend
        )

    def forward(self, x, token_mask, seg_mask, t):
        # input x'shape: (batch, max_seg, max_seq_len, d_model)
        # token_mask'shape: (batch, max_seg, max_seq_len)
        # seg_mask'shape: (batch, max_seg)
        batch, max_seg, max_seq_len, d_model = x.shape
        x = x.reshape(-1, max_seq_len, d_model).contiguous()
        token_mask = token_mask.reshape(-1, max_seq_len).contiguous()
        t = t.reshape(-1, max_seq_len).contiguous()

        # therefore, x is truncated at both seg and seq_len level
        # segment-wise encoding
        seg_hidden_state = self.swe(x, token_mask)
        seg_hidden_state = seg_hidden_state.reshape(
            batch, max_seg, max_seq_len, d_model
        ).contiguous()

        # cross-segment encoding
        # cls embeddings from all seg: (batch, max_seg, d_model)
        seg_cls_hidden_state = seg_hidden_state[:, :, 0, :].clone()
        if self.cs_pe == "pos":
            seg_cls_hidden_state = self.cse(seg_cls_hidden_state, seg_mask)
        elif self.cs_pe == "time":
            seg_cls_hidden_state = self.cse(seg_cls_hidden_state, seg_mask, t)

        # combine
        seg_hidden_state[:, :, 0, :] = seg_cls_hidden_state

        return seg_hidden_state
