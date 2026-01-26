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
        if config.dropout > 0.0:
            _half_n = config.n_blocks // 2
            dropout_ps = [0.0] * _half_n + [config.dropout] * _half_n
        else:
            dropout_ps = [0.0] * config.n_blocks
        self.blocks = nn.ModuleList(
            [
                HierarchicalTransformerBlock(
                    d_model=config.d_model,
                    d_ff=config.d_ff,
                    h=config.n_heads,
                    dropout=dropout_ps[i],
                    norm_type=config.norm_type,
                    ffn_type=config.ffn_type,
                    attn_backend=config.attn_backend,
                )
                for i in range(config.n_blocks)
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
        h, mid_h = self.encode(input_ids, attention_mask, set_attention_mask, t)
        logits = self.lm_head(h)
        # (batch, max_seq, max_set_size, d_model)
        return logits, (h, mid_h)

    def encode(self, input_ids, attention_mask, set_attention_mask, t):
        h = self.embeddings(input_ids)
        h = h + self.t2v(t)
        mid_h = None
        for l_id, block in enumerate(self.blocks):  # DONE: get mid representation
            h = block(h, attention_mask, set_attention_mask, t)
            if l_id == len(self.blocks) // 2:
                mid_h = h
        h = self.last_norm(h)
        # (batch, max_seq, max_set_size, d_model)
        return h, mid_h


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
        # set_mask: (batch, max_seq) is used to select the sets for DM (e.g., MSM sets)
        if set_mask is None:
            set_mask = set_attention_mask
        logits_mlm, (h, mid_h) = self.transformer(
            input_ids, attention_mask, set_attention_mask, t
        )
        # M := # masked/selected sets
        # h_cls: (M, d_model) where M is the total n of observed sets
        h_cls = h[set_mask][:, 0, :]
        # logits_dm: (M, vocab_size)
        logits_dm = self.mlp_dm(h_cls)
        # TODO: logits_tte = self.mlp_tte(t)

        return logits_mlm, logits_dm, (h, mid_h)

    def encode(self, input_ids, attention_mask, set_attention_mask, t):
        return self.transformer.encode(input_ids, attention_mask, set_attention_mask, t)
