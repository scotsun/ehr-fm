import torch.nn as nn
from transformers import PretrainedConfig


class FMConfig(PretrainedConfig):
    model_type = "fm"

    def __init__(
        self,
        vocab_size: int = 30000,
        dataset: dict = {},
        trainer: dict = {},
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
        **kwargs,
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)
        self.vocab_size = vocab_size
        self.dataset = dataset
        self.trainer = trainer
        self.d_model = d_model
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.dropout = dropout
        self.norm_type = norm_type
        self.ffn_type = ffn_type
        self.weight_tying = weight_tying
        self.attn_backend = attn_backend
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
