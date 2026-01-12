import torch
import torch.nn as nn

from transformers.modeling_utils import PreTrainedModel
from transformers.models.longformer.modeling_longformer import LongformerSelfAttention
from transformers import LongformerConfig

from . import FMConfig, FMEmbeddings
from src.layers import T2V, RoPE, FFNSwiGLUBlock, FFNLUBlock


class LongformerMHABlock(LongformerSelfAttention):
    pass


class LongformerBlock(nn.Module):
    pass


class FMLongformer(PreTrainedModel):
    config_class = FMConfig
    model_type = "fm-longformer"

    def __init__(self, config: FMConfig):
        super().__init__(config)
        self.embeddings = FMEmbeddings(config)
        self.t2v = T2V(config.d_model, config.t2v_scale)

    def _merge_to_attention_mask(
        self, attention_mask: torch.Tensor, global_attention_mask: torch.Tensor
    ):
        # longformer self attention expects attention mask to have 0 (no attn), 1 (local attn), 2 (global attn)
        # (global_attention_mask + 1) => 1 for local attention, 2 for global attention
        # => final attention_mask => 0 for no attention, 1 for local attention 2 for global attention
        if attention_mask is not None:
            attention_mask = attention_mask * (global_attention_mask + 1)
        else:
            # simply use `global_attention_mask` as `attention_mask`
            # if no `attention_mask` is given
            attention_mask = global_attention_mask + 1
        return attention_mask
