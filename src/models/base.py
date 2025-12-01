import torch.nn as nn
from transformers.modeling_utils import PreTrainedModel

from . import FMConfig, FMEmbeddings
from src.layers import HierarchicalTransformerBlock, T2V


class FMTransformerEncoder(nn.Module):
    config_class = FMConfig
    base_model_prefix = "fm"

    def __init__(self, config: FMConfig):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                HierarchicalTransformerBlock(
                    d_model=config.d_model,
                    d_ff=config.d_ff,
                    h=config.n_heads,
                    dropout=config.dropout,
                    norm_type=config.norm_type,
                    ffn_type=config.ffn_type,
                    attn_backend=config.attn_backend,
                )
                for _ in range(config.n_blocks)
            ]
        )

    def forward(self, h, attention_mask, set_attention_mask, t):
        for block in self.blocks:
            h = block(h, attention_mask, set_attention_mask, t)
        return h


class FMBase(PreTrainedModel):
    config_class = FMConfig
    base_model_prefix = "fm-base"

    def __init__(self, config: FMConfig):
        super().__init__(config)
        self.embeddings = FMEmbeddings(config)
        self.t2v = T2V(config.d_model, config.t2v_scale)
        self.transformer_encoder = FMTransformerEncoder(config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size)
        if config.weight_tying:
            self.lm_head.weight = self.embeddings.embeddings.weight

    def forward(self, input_ids, attention_mask, set_attention_mask, t):
        h = self.encode(input_ids, attention_mask, set_attention_mask, t)
        logits = self.lm_head(h)
        # (batch, max_seq, max_set_size, d_model)
        return logits, h

    def encode(self, input_ids, attention_mask, set_attention_mask, t):
        h = self.embeddings(input_ids)
        h = h + self.t2v(t)
        h = self.transformer_encoder(h, attention_mask, set_attention_mask, t)
        # (batch, max_seq, max_set_size, d_model)
        return h
