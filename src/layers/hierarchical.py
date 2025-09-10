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
        # segment-wise encoder
        self.swe = TransformerBlock(
            d_model, d_ff, h, False, dropout, norm_type, ffn_type, attn_backend
        )
        # cross-segment encoder
        self.cse = TransformerBlock(
            d_model, d_ff, h, True, dropout, norm_type, ffn_type, attn_backend
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
