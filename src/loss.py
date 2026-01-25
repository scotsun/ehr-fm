import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftCLT(nn.Module):
    """adapted from softclt github repo"""

    def __init__(self, model, temperature):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1, z2, mask1, mask2):
        # the mask is the set_attention_mask
        pass

    def masked_max_pool1d(self, z, set_attention_mask, kernel_size):
        # z (batch_size, seq_len, d_model)
        # set_attention_mask (batch_size, seq_len)
        B, C, T = z.shape
        z = z.transpose(1, 2)  # (B, T, C)
        set_attention_mask = set_attention_mask.unsqueeze(1).expand(
            -1, C, -1
        )  # (B, C, T)
        z_masked = z.masked_fill(~set_attention_mask, -float("inf"))
        z_pooled = F.max_pool1d(z_masked, kernel_size=kernel_size)
        z_pooled = z_pooled.transpose(1, 2)  # (B, T // kernel_size, C)
        z_pooled = z_pooled.masked_fill(torch.isinf(z_pooled), 0.0)
        return z_pooled
