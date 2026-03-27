import torch
import torch.nn as nn
from transformers import PreTrainedModel
from src.layers import T2V, FFNSwiGLUBlock, FFNLUBlock, ResidualConnection


class PerformerSelfAttention(nn.Module):
    """
    Performer Self-Attention Module as a replacement for LongformerSelfAttention.
    Implements FAVOR+ mechanism for efficient attention computation.
    """

    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.embed_dim = config.hidden_size
        self.epsilon = 1e-6

        self.query = nn.Linear(self.embed_dim, self.embed_dim)
        self.key = nn.Linear(self.embed_dim, self.embed_dim)
        self.value = nn.Linear(self.embed_dim, self.embed_dim)

        self.output = nn.Linear(self.embed_dim, self.embed_dim)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def favor_positive_features(self, x):
        """
        Applies the softmax kernel approximation (FAVOR+)
        """
        projected = torch.exp(x - torch.max(x, dim=-1, keepdim=True).values)
        return projected / (torch.sum(projected, dim=-1, keepdim=True) + self.epsilon)

    def forward(self, hidden_states, attention_mask=None):
        batch_size, seq_len, hidden_dim = hidden_states.size()

        # Linear projections for query, key, and value
        queries = self.query(hidden_states)
        keys = self.key(hidden_states)
        values = self.value(hidden_states)

        # Reshape for multi-head attention
        queries = queries.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        keys = keys.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(
            1, 2
        )
        values = values.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)

        # Normalize queries and keys with FAVOR+
        queries = self.favor_positive_features(queries)
        keys = self.favor_positive_features(keys)

        # Compute Performer attention weights
        numerator = torch.einsum("bhld,bhmd->bhlm", queries, keys)  # Scaled dot product
        denominator = torch.einsum("bhld,bhmd->bhlm", queries, torch.ones_like(keys))
        attention_probs = numerator / (
            denominator + self.epsilon
        )  # Normalize attention weights

        attention_probs = self.dropout(attention_probs)
        context_layer = torch.einsum("bhlm,bhmd->bhld", attention_probs, values)

        # Combine heads and project back to hidden_dim
        context_layer = context_layer.transpose(1, 2).reshape(
            batch_size, seq_len, hidden_dim
        )
        output = self.output(context_layer)
        return output


class PerformerBlock(nn.Module):
    """
    A single block with Performer Self-Attention and Feed-Forward
    """

    def __init__(self, config):
        super().__init__()
        self.attention = PerformerSelfAttention(config)
        self.ffn = (
            FFNSwiGLUBlock(config.hidden_size, config.intermediate_size)
            if config.ffn_type == "swiglu"
            else FFNLUBlock(
                config.hidden_size, config.intermediate_size, config.ffn_type
            )
        )
        self.residual_connection1 = ResidualConnection(
            config.hidden_size, config.hidden_dropout_prob, config.norm_type
        )
        self.residual_connection2 = ResidualConnection(
            config.hidden_size, config.hidden_dropout_prob, config.norm_type
        )

    def forward(self, x, attention_mask=None):
        # Attention output with residual connection
        x = self.residual_connection1(x, lambda x: self.attention(x, attention_mask))
        # Feed-forward output with residual connection
        x = self.residual_connection2(x, self.ffn)
        return x


class FMPerformer(PreTrainedModel):
    """
    Feature Mapping (FM) Model with Performer Attention
    """

    config_class = None  # Replace with your FMConfig
    model_type = "fm-performer"

    def __init__(self, config):
        super().__init__(config)

        # Embedding layer
        self.embeddings = T2V(config.d_model, config.t2v_scale)

        # Initialize Performer blocks
        self.blocks = nn.ModuleList(
            [PerformerBlock(config) for _ in range(config.n_blocks)]
        )

        # Normalization layer
        if config.norm_type == "layer":
            self.last_norm = nn.LayerNorm(config.d_model, bias=False)
        elif config.norm_type == "rms":
            self.last_norm = nn.RMSNorm(config.d_model)
        else:
            raise ValueError(f"Unsupported norm_type: {config.norm_type}")

        # Output layer
        self.lm_head = nn.Linear(config.d_model, config.vocab_size)
        if config.weight_tying:
            self.lm_head.weight = self.embeddings.embeddings.weight

    def forward(self, input_ids, attention_mask, t):
        # Encode the inputs with the Performer Layers
        h = self.embeddings(input_ids)
        for block in self.blocks:
            h = block(h, attention_mask)
        h = self.last_norm(h)

        # Compute logits
        logits = self.lm_head(h)
        return logits, h
