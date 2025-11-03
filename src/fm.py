import torch.nn as nn
from transformers.modeling_utils import PreTrainedModel
from transformers import PretrainedConfig

from src.layers.hierarchical import HierarchicalTransformerBlock


class FMConfig(PretrainedConfig):
    model_type = "fm"

    def __init__(
        self,
        vocab_size: int = 15000,
        d_model: int = 768,
        n_blocks: int = 6,
        n_heads: int = 12,
        d_ff: int = 2048,
        dropout: float = 0.0,
        norm_type: str = "layer",
        ffn_type: str = "swiglu",
        pad_token_id: int = 0,
        weight_tying: bool = False,
        attn_backend: str = "base",
        swe_rope: bool = True,  # Whether SWE uses RoPE
        **kwargs,
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.dropout = dropout
        self.norm_type = norm_type
        self.ffn_type = ffn_type
        self.weight_tying = weight_tying
        self.attn_backend = attn_backend
        self.swe_rope = swe_rope
        for k, v in kwargs.items():
            setattr(self, k, v)


class FMEmbeddings(nn.Module):
    def __init__(self, config: FMConfig):
        super().__init__()
        self.config = config
        self.embeddings = nn.Embedding(
            config.vocab_size, config.d_model, padding_idx=config.pad_token_id
        )

    def forward(self, input_ids):
        return self.embeddings(input_ids)


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
                    swe_rope=config.swe_rope,
                )
                for _ in range(config.n_blocks)
            ]
        )

    def forward(self, h, attention_mask, segment_attention_mask, segment_time=None, token_time=None):
        for block in self.blocks:
            h = block(h, attention_mask, segment_attention_mask, segment_time, token_time)
        return h


class FMBase(PreTrainedModel):
    config_class = FMConfig
    base_model_prefix = "fm-base"

    def __init__(self, config: PretrainedConfig):
        super().__init__(config)
        self.embeddings = FMEmbeddings(config)
        self.transformer_encoder = FMTransformerEncoder(config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size)
        if config.weight_tying:
            self.lm_head.weight = self.embeddings.embeddings.weight

    def forward(self, input_ids, attention_mask, segment_attention_mask, segment_time=None, token_time=None):
        h = self.encode(input_ids, attention_mask, segment_attention_mask, segment_time, token_time)
        logits = self.lm_head(h)
        # (batch, max_seg, max_seq_len, vocab)
        return logits

    def encode(self, input_ids, attention_mask, segment_attention_mask, segment_time=None, token_time=None):
        h = self.embeddings(input_ids)
        h = self.transformer_encoder(h, attention_mask, segment_attention_mask, segment_time, token_time)
        # (batch, max_seg, max_seq_len, d_model)
        return h
