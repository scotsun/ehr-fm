import torch.nn as nn
from torch import cumsum
from transformers.modeling_utils import PreTrainedModel

from . import FMConfig, FMEmbeddings
from src.layers import TransformerBlock, T2V


class FMBert(PreTrainedModel):
    config_class = FMConfig
    model_type = "fm-bert"

    def __init__(self, config: FMConfig):
        super().__init__(config)
        self.embeddings = FMEmbeddings(config)
        self.t2v = T2V(config.d_model, config.t2v_scale)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=config.d_model,
                    d_ff=config.d_ff,
                    h=config.n_heads,
                    with_rope=True,
                    dropout=config.dropout,
                    norm_type=config.norm_type,
                    ffn_type=config.ffn_type,
                    attn_backend=config.attn_backend,
                )
                for _ in range(config.n_blocks)
            ]
        )
        match config.norm_type:
            case "layer":
                self.last_norm = nn.LayerNorm(config.d_model)
            case "rms":
                self.last_norm = nn.RMSNorm(config.d_model)
            case _:
                raise ValueError(f"{config.norm_type} not implemented")

        self.lm_head = nn.Linear(config.d_model, config.vocab_size)
        if config.weight_tying:
            self.lm_head.weight = self.embeddings.embeddings.weight

    def forward(self, input_ids, attention_mask, t):
        h = self.encode(input_ids, attention_mask, t)
        logits = self.lm_head(h)
        return logits, h

    def encode(self, input_ids, attention_mask, t):
        # shapes: (batch, L)
        set_pos = attention_mask * cumsum(input_ids == 2, dim=1)
        h = self.embeddings(input_ids)
        h = h + self.t2v(t)
        for block in self.blocks:
            h = block(h, attention_mask, time=set_pos)
        h = self.last_norm(h)
        return h
