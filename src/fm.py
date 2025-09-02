import torch.nn as nn
from transformers.modeling_utils import PreTrainedModel
from transformers import PretrainedConfig

from src.layer import HierarchicalTransformerBlock


class FMEmbeddings(nn.Module):
    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.config = config
        self.embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )


class FMTransformer(PreTrainedModel):
    def __init__(self, config: PretrainedConfig):
        super().__init__(config)
        self.embeddings = FMEmbeddings(config)
        self.n_blocks = config.n_blocks
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
