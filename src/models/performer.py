import math
import torch
import torch.nn as nn
from transformers import PreTrainedModel
from . import FMConfig, FMEmbeddings
from src.layers import T2V, FFNSwiGLUBlock, FFNLUBlock, ResidualConnection


def _create_projection_matrix(
    nb_features: int, d_k: int, device: torch.device
) -> torch.Tensor:
    """
    Creates a random matrix with orthogonal rows scaled by chi-distributed norms,
    as required by FAVOR+ (Choromanski et al., 2020).
    Returns shape: (nb_features, d_k)
    """
    nb_full_blocks = nb_features // d_k
    blocks = []
    for _ in range(nb_full_blocks):
        q, _ = torch.linalg.qr(torch.randn(d_k, d_k, device=device))
        blocks.append(q.T)
    remainder = nb_features - nb_full_blocks * d_k
    if remainder > 0:
        q, _ = torch.linalg.qr(torch.randn(d_k, d_k, device=device))
        blocks.append(q.T[:remainder])
    matrix = torch.cat(blocks, dim=0)  # (nb_features, d_k)
    # Scale rows by the norms of d_k-dimensional Gaussian vectors (chi distribution)
    norms = torch.randn(nb_features, d_k, device=device).norm(dim=1)
    return matrix * norms.unsqueeze(1)


class PerformerSelfAttention(nn.Module):
    """
    Performer Self-Attention with FAVOR+.
    Approximates softmax attention in O(L * nb_features * d_k) via the kernel trick,
    avoiding materializing the L x L attention matrix.
    """

    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_heads
        self.d_k = config.d_model // config.n_heads
        self.d_model = config.d_model
        self.epsilon = 1e-6

        nb_features = getattr(config, "nb_features", None)
        self.nb_features = (
            nb_features
            if nb_features is not None
            else max(1, int(self.d_k * math.log(self.d_k)))
        )
        self.feature_redraw_interval = getattr(config, "feature_redraw_interval", 1000)
        self._calls_since_last_redraw = 0

        self.query = nn.Linear(self.d_model, self.d_model, bias=False)
        self.key = nn.Linear(self.d_model, self.d_model, bias=False)
        self.value = nn.Linear(self.d_model, self.d_model, bias=False)
        self.output = nn.Linear(self.d_model, self.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)

        # Projection matrix is a non-trainable buffer that moves with the model
        self.register_buffer(
            "projection_matrix",
            _create_projection_matrix(
                self.nb_features, self.d_k, device=torch.device("cpu")
            ),
        )

    def _redraw_projection_matrix(self):
        self.projection_matrix.copy_(
            _create_projection_matrix(
                self.nb_features, self.d_k, device=self.projection_matrix.device
            )
        )

    def _favor_positive_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        FAVOR+ positive random feature map that approximates the softmax kernel:
            phi(x) = ratio * exp(W * x / d_k^{1/4}  -  ||x||^2 / (2 * sqrt(d_k)))
        Shape: (..., d_k) -> (..., nb_features)
        """
        data_normalizer = self.d_k**-0.25
        ratio = self.nb_features**-0.5
        # Project scaled input through the random orthogonal matrix
        projected = torch.einsum(
            "...d,md->...m", data_normalizer * x, self.projection_matrix
        )
        # Subtract half the squared norm (consistently scaled) for the softmax approximation
        norm_sq = (x**2).sum(dim=-1, keepdim=True) * (data_normalizer**2) / 2
        # Detach max for numerical stability (lucidrains trick)
        projected_max = projected.amax(dim=-1, keepdim=True).detach()
        return ratio * (torch.exp(projected - projected_max - norm_sq) + self.epsilon)

    def forward(self, hidden_states, attention_mask=None):
        batch_size, seq_len, _ = hidden_states.size()

        # Periodically redraw the projection matrix during training
        if self.training:
            self._calls_since_last_redraw += 1
            if self._calls_since_last_redraw >= self.feature_redraw_interval:
                self._redraw_projection_matrix()
                self._calls_since_last_redraw = 0

        # Linear projections -> (B, H, L, d_k)
        queries = (
            self.query(hidden_states)
            .view(batch_size, seq_len, self.n_head, self.d_k)
            .transpose(1, 2)
        )
        keys = (
            self.key(hidden_states)
            .view(batch_size, seq_len, self.n_head, self.d_k)
            .transpose(1, 2)
        )
        values = (
            self.value(hidden_states)
            .view(batch_size, seq_len, self.n_head, self.d_k)
            .transpose(1, 2)
        )

        # FAVOR+ feature maps -> (B, H, L, nb_features)
        phi_q = self._favor_positive_features(queries)
        phi_k = self._favor_positive_features(keys)

        # Zero out padding key positions before accumulation
        # attention_mask: (B, L), 1=keep, 0=pad
        if attention_mask is not None:
            phi_k = phi_k * attention_mask[:, None, :, None].float()

        # Linear attention — O(L) — never forms the L x L matrix
        # kv: (B, H, nb_features, d_k)
        kv = torch.einsum("bhlm,bhld->bhmd", phi_k, values)
        # out: (B, H, L, d_k)
        out = torch.einsum("bhlm,bhmd->bhld", phi_q, kv)

        # Normalizer: row-wise denominator for each query
        k_sum = phi_k.sum(dim=2)  # (B, H, nb_features)
        denom = (
            torch.einsum("bhlm,bhm->bhl", phi_q, k_sum)
            .unsqueeze(-1)
            .clamp(min=self.epsilon)
        )
        out = out / denom

        out = self.dropout(out)

        # Merge heads -> (B, L, d_model)
        out = out.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)
        return self.output(out)


class PerformerBlock(nn.Module):
    """
    A single block with Performer Self-Attention and Feed-Forward
    """

    def __init__(self, config):
        super().__init__()
        self.attention = PerformerSelfAttention(config)
        self.ffn = (
            FFNSwiGLUBlock(config.d_model, config.d_ff)
            if config.ffn_type == "swiglu"
            else FFNLUBlock(config.d_model, config.d_ff, config.ffn_type)
        )
        self.residual_connection1 = ResidualConnection(
            config.d_model, config.dropout, config.norm_type
        )
        self.residual_connection2 = ResidualConnection(
            config.d_model, config.dropout, config.norm_type
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

    config_class = FMConfig
    model_type = "fm-performer"

    def __init__(self, config):
        super().__init__(config)

        # Embedding layer
        self.embeddings = FMEmbeddings(config)
        self.t2v = T2V(config.d_model, config.t2v_scale)

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
        h = self.embeddings(input_ids)
        h = h + self.t2v(t)
        # Encode the inputs with the Performer Layers
        for block in self.blocks:
            h = block(h, attention_mask)
        h = self.last_norm(h)

        # Compute logits
        logits = self.lm_head(h)
        return logits, h
