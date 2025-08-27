import torch.nn as nn
from transformers.modeling_utils import PreTrainedModel
from transformers import PretrainedConfig


class FMEmbeddings(nn.Module):
    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.config = config
        self.embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )
