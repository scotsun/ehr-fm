import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftCLT(nn.Module):
    def __init__(self, model, temperature):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1, z2, set_attention_mask):
        pass

    def masked_max_pool1d(self, z, set_attention_mask):
        # z (batch_size, seq_len, d_model)
        # set_attention_mask (batch_size, seq_len)
        pass
