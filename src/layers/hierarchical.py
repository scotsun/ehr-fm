import torch.nn as nn

from .transformer import TransformerBlock


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
    ):
        super().__init__()
        # segment-wise encoder - 现在也使用 RoPE (因为有 TIME_BIN tokens)
        self.swe = TransformerBlock(
            d_model, d_ff, h, True, dropout, norm_type, ffn_type, attn_backend
        )
        # cross-segment encoder - 使用 RoPE 处理跨 segment 的时间
        self.cse = TransformerBlock(
            d_model, d_ff, h, True, dropout, norm_type, ffn_type, attn_backend
        )

    def forward(self, x, token_mask, seg_mask, seg_time=None, token_time=None):
        # input x'shape: (batch, max_seg, max_seq_len, d_model)
        # token_mask'shape: (batch, max_seg, max_seq_len)
        # seg_mask'shape: (batch, max_seg)
        # seg_time'shape: (batch, max_seg) - segment 级别的时间（如 days_since_first_visit）
        # token_time'shape: (batch, max_seg, max_seq_len) - token 级别的时间（如 time_offset_hours）

        # therefore, x is truncated at both seg and seq_len level
        # segment-wise encoding - 使用 token 级别的时间偏移
        if token_time is not None:
            # Reshape token_time: (batch, max_seg, max_seq_len) -> (batch*max_seg, max_seq_len)
            token_time_reshaped = token_time.reshape(-1, token_time.shape[2]).contiguous()
        else:
            token_time_reshaped = None
            
        seg_hidden_state = self.swe(
            x.reshape(-1, x.shape[2], x.shape[3]).contiguous(),
            token_mask.reshape(-1, token_mask.shape[2]).contiguous(),
            time=token_time_reshaped  # 使用 event-level 的时间偏移（time_offset_hours）
        )
        seg_hidden_state = seg_hidden_state.reshape(
            x.shape[0], x.shape[1], x.shape[2], x.shape[3]
        ).contiguous()

        # cross-segment encoding 
        # cls embeddings from all seg: (batch, max_seg, d_model)
        seg_cls_hidden_state = seg_hidden_state[:, :, 0, :].clone()
        seg_cls_hidden_state = self.cse(
            seg_cls_hidden_state, 
            seg_mask, 
            time=seg_time  # 使用 segment 级别的时间（cumsum of days_since_prior_admission）
        )

        # combine
        seg_hidden_state[:, :, 0, :] = seg_cls_hidden_state

        return seg_hidden_state
