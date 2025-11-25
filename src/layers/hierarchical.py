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
        swe_rope: bool = True,
    ):
        super().__init__()
        # segment-wise encoder - RoPE optional
        self.swe = TransformerBlock(
            d_model, d_ff, h, swe_rope, dropout, norm_type, ffn_type, attn_backend
        )
        # cross-segment encoder - always uses RoPE
        self.cse = TransformerBlock(
            d_model, d_ff, h, True, dropout, norm_type, ffn_type, attn_backend
        )

    def forward(self, x, token_mask, seg_mask, seg_time=None, token_time=None):
        # input x'shape: (batch, max_seg, max_seq_len, d_model)
        # token_mask'shape: (batch, max_seg, max_seq_len)
        # seg_mask'shape: (batch, max_seg)
        # seg_time'shape: (batch, max_seg) - segment-level time (e.g., days_since_first_visit)
        # token_time'shape: (batch, max_seg, max_seq_len) - token-level time (e.g., time_offset_hours)

        # CRITICAL: Ensure time tensors stay in FP32 to avoid precision loss
        # Large time values (e.g., 2533 days) lose significant precision in FP16
        # which causes RoPE to compute incorrect cos/sin values
        if seg_time is not None:
            seg_time = seg_time.float()
        if token_time is not None:
            token_time = token_time.float()

        # therefore, x is truncated at both seg and seq_len level
        # segment-wise encoding - uses token-level time offset
        if token_time is not None:
            # Reshape token_time: (batch, max_seg, max_seq_len) -> (batch*max_seg, max_seq_len)
            token_time_reshaped = token_time.reshape(-1, token_time.shape[2]).contiguous()
        else:
            token_time_reshaped = None

        seg_hidden_state = self.swe(
            x.reshape(-1, x.shape[2], x.shape[3]).contiguous(),
            token_mask.reshape(-1, token_mask.shape[2]).contiguous(),
            time=token_time_reshaped  # uses event-level time offset (time_offset_hours)
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
            time=seg_time  # uses segment-level time (cumsum)
        )

        # combine
        seg_hidden_state[:, :, 0, :] = seg_cls_hidden_state

        return seg_hidden_state
