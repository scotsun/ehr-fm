import torch.nn as nn
from transformers.modeling_utils import PreTrainedModel
from transformers import PretrainedConfig

from src.layer import HierarchicalTransformerBlock


class FMConfig(PretrainedConfig):
    model_type = "fm"

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 768,
        n_blocks: int = 6,
        n_heads: int = 12,
        d_ff: int = 2048,
        dropout: float = 0.1,
        norm_type: str = "layer",
        ffn_type: str = "swiglu",
        pad_token_id: int = 0,
        weight_tying: bool = False,
        **kwargs,
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.dropout = dropout
        self.norm_type = norm_type
        self.ffn_type = ffn_type
        self.weight_tying = weight_tying


class FMEmbeddings(nn.Module):
    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.config = config
        self.embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )

    def forward(self, input_ids):
        return self.embeddings(input_ids)


class FMTransformer(PreTrainedModel):
    config_class = FMConfig
    base_model_prefix = "fm"

    def __init__(self, config: PretrainedConfig):
        super().__init__(config)
        self.n_blocks = config.n_blocks
        self.embeddings = FMEmbeddings(config)
        self.blocks = nn.ModuleList(
            [
                HierarchicalTransformerBlock(
                    d_model=config.hidden_size,
                    d_ff=config.d_ff,
                    h=config.n_heads,
                    dropout=config.dropout,
                    norm_type=config.norm_type,
                    ffn_type=config.ffn_type,
                )
                for _ in range(self.n_blocks)
            ]
        )
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)
        if config.weight_tying:
            self.lm_head.weight = self.embeddings.embeddings.weight

    def forward(self, input_ids, attention_mask, segment_attention_mask):
        h = self.embeddings(input_ids)
        # pos embedding
        for block in self.blocks:
            h = block(h, attention_mask, segment_attention_mask)
        logits = self.lm_head(h)
        return logits
