import torch.nn as nn
from transformers.modeling_utils import PreTrainedModel

from . import FMConfig, FMEmbeddings
from src.layers import HierarchicalTransformerBlock, T2V, FFNSwiGLUBlock


class FMBase(PreTrainedModel):
    config_class = FMConfig
    model_type = "fm-base"

    def __init__(self, config: FMConfig):
        super().__init__(config)
        self.embeddings = FMEmbeddings(config)
        self.t2v = T2V(config.d_model, config.t2v_scale)
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
        match config.norm_type:
            case "layer":
                self.last_norm = nn.LayerNorm(config.d_model, bias=False)
            case "rms":
                self.last_norm = nn.RMSNorm(config.d_model)
            case _:
                raise ValueError(f"{config.norm_type} not implemented")
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
        for block in self.blocks:
            h = block(h, attention_mask, set_attention_mask, t)
        h = self.last_norm(h)
        # (batch, max_seq, max_set_size, d_model)
        return h


class FMBaseWithHeads(PreTrainedModel):
    config_class = FMConfig
    model_type = "fm-base-with_heads"

    def __init__(self, config: FMConfig):
        super().__init__(config)
        self.transformer = FMBase(config)
        self.mlp_dm = FFNSwiGLUBlock(
            d_model=config.d_model, d_ff=config.d_ff, d_out=config.vocab_size
        )
        # TODO: self.mlp_tte

    def forward(self, input_ids, attention_mask, set_attention_mask, t, set_mask=None):
        # logits_mlm: (batch, max_seq, max_set_size, vocab_size)
        # h: (batch, max_seq, max_set_size, d_model)
        if set_mask is None:
            set_mask = set_attention_mask
        logits_mlm, h = self.transformer(
            input_ids, attention_mask, set_attention_mask, t
        )
        # M := # masked/selected sets
        # h_cls: (M, d_model) where M is the total n of observed sets
        h_cls = h[set_mask][:, 0, :]
        # logits_dm: (M, vocab_size)
        logits_dm = self.mlp_dm(h_cls)
        # TODO: logits_tte = self.mlp_tte(t)

        return logits_mlm, logits_dm, h

    def encode(self, input_ids, attention_mask, set_attention_mask, t):
        return self.transformer.encode(input_ids, attention_mask, set_attention_mask, t)
