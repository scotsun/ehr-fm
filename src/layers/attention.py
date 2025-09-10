import math
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from torch.nn.attention import sdpa_kernel, SDPBackend

from .rope import RoPE


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, h: int, with_rope: bool, attn_backend: str):
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
        match attn_backend:
            case "base":
                self._attn_func = self.attention
            case "flash_attention":
                self._attn_func = self.flash_attention

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

        mh_out = self._attn_func(query, key, value, mask)
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
