"""
Hi-BEHRT Baseline Model with BYOL Pre-training

Hierarchical BERT for EHR: two-level Transformer architecture for processing long EHR sequences.

Reference: Hi-BEHRT: Hierarchical Transformer-based model for accurate prediction of clinical
events using multimodal longitudinal electronic health records (Li et al., 2021)

Core Architecture:
1. Embedding Layer: Token + Time (T2V) + Segment + Position (sinusoidal)
2. Local Feature Extractor: Transformer operating on segments within sliding windows
3. Feature Aggregator: Transformer globally summarizing segment representations

BYOL Pre-training:
- Bootstrap Your Own Latent (BYOL) self-supervised learning
- Dual network structure: online network + target network (EMA updated)
- Projector + Predictor MLP heads
- Cosine similarity loss between online predictions and target projections

Paper hyperparameters:
- Hidden size: 150
- Attention heads: 6
- Intermediate size: 108
- Local extractor layers: 4
- Aggregator layers: 4
- Window size: 50, Stride: 30
- Max sequence length: 1220
"""

import math
import copy
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class HiBEHRTConfig:
    """Configuration for Hi-BEHRT baseline model."""
    vocab_size: int = 15000
    d_model: int = 150
    n_extractor_layers: int = 4
    n_aggregator_layers: int = 4
    n_heads: int = 6
    d_ff: int = 108
    max_seq_len: int = 50  # window size for local extractor
    max_segments: int = 40  # max number of segments for aggregator
    dropout: float = 0.2
    attention_dropout: float = 0.3
    pad_token_id: int = 0
    t2v_dim: int = 64  # Time2Vec output dimension (replaces age embedding)
    seg_vocab_size: int = 2  # alternating 0/1 for segment embedding
    initializer_range: float = 0.02
    hidden_act: str = "gelu"
    # BYOL specific
    projector_hidden_size: int = 256
    projector_output_size: int = 128
    byol_momentum: float = 0.99  # EMA momentum for target network


def get_activation(name: str):
    """Get activation function by name."""
    if name == "gelu":
        return F.gelu
    elif name == "relu":
        return F.relu
    elif name == "tanh":
        return torch.tanh
    else:
        raise ValueError(f"Unknown activation: {name}")


# ============================================================================
# Time2Vec - Replaces Age Embedding
# ============================================================================

class Time2Vec(nn.Module):
    """
    Time2Vec: Learnable time encoding.
    From: Time2Vec: Learning Vector Representation of Time (Kazemi et al., 2019)

    Used to encode cumulative time (cumsum of days_since_prior_admission)
    as a replacement for age embedding in original Hi-BEHRT.
    """

    def __init__(self, d_out: int):
        super().__init__()
        self.d_out = d_out
        # Linear component
        self.w0 = nn.Parameter(torch.randn(1) * 0.01)
        self.b0 = nn.Parameter(torch.zeros(1))
        # Periodic components
        self.w = nn.Parameter(torch.randn(d_out - 1) * 0.01)
        self.b = nn.Parameter(torch.zeros(d_out - 1))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (..., ) or (..., 1) - time values (continuous, e.g., days)
        Returns:
            (..., d_out) - time embeddings
        """
        original_shape = t.shape
        if t.dim() >= 1 and t.shape[-1] != 1:
            t = t.unsqueeze(-1)  # (..., 1)

        # Linear component: w0 * t + b0
        v0 = self.w0 * t + self.b0  # (..., 1)

        # Periodic components: sin(w * t + b)
        v = torch.sin(self.w * t + self.b)  # (..., d_out-1)

        return torch.cat([v0, v], dim=-1)  # (..., d_out)


# ============================================================================
# Position Embedding
# ============================================================================

class SinusoidalPositionEmbedding(nn.Module):
    """Sinusoidal position embedding (fixed, not learned)."""

    def __init__(self, max_len: int, d_model: int):
        super().__init__()

        # Create sinusoidal position encoding
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])

        self.register_buffer('pe', pe)

    def forward(self, position_ids):
        """
        Args:
            position_ids: (..., seq_len) position indices
        Returns:
            position_embeddings: (..., seq_len, d_model)
        """
        return self.pe[position_ids]


# ============================================================================
# Attention Components
# ============================================================================

class HiBEHRTSelfAttention(nn.Module):
    """
    Multi-head self-attention for Hi-BEHRT.

    Handles both:
    - encounter=True: 5D input [batch, num_segments, seq_len, d_model] (local extractor)
    - encounter=False: 4D input [batch, num_segments, d_model] (aggregator)
    """

    def __init__(self, config: HiBEHRTConfig):
        super().__init__()

        if config.d_model % config.n_heads != 0:
            raise ValueError(
                f"d_model ({config.d_model}) must be divisible by n_heads ({config.n_heads})"
            )

        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.all_head_size = self.n_heads * self.head_dim

        self.query = nn.Linear(config.d_model, self.all_head_size)
        self.key = nn.Linear(config.d_model, self.all_head_size)
        self.value = nn.Linear(config.d_model, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_dropout)

    def transpose_for_scores(self, x, encounter=True):
        """Reshape for multi-head attention."""
        new_shape = x.size()[:-1] + (self.n_heads, self.head_dim)
        x = x.view(*new_shape)

        if encounter:
            # 5D: [batch, num_segments, seq_len, n_heads, head_dim]
            # -> [batch, num_segments, n_heads, seq_len, head_dim]
            return x.permute(0, 1, 3, 2, 4)
        else:
            # 4D: [batch, seq_len, n_heads, head_dim]
            # -> [batch, n_heads, seq_len, head_dim]
            return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask, encounter=True):
        """
        Args:
            hidden_states: (batch, num_segments, seq_len, d_model) if encounter=True
                          (batch, num_segments, d_model) if encounter=False
            attention_mask: pre-computed attention mask with -10000 for masked positions
            encounter: whether processing encounters (local) or aggregating (global)

        Returns:
            context: same shape as hidden_states
        """
        q = self.transpose_for_scores(self.query(hidden_states), encounter)
        k = self.transpose_for_scores(self.key(hidden_states), encounter)
        v = self.transpose_for_scores(self.value(hidden_states), encounter)

        # Attention scores
        scores = torch.matmul(q, k.transpose(-1, -2))
        scores = scores / math.sqrt(self.head_dim)
        scores = scores + attention_mask

        # Attention probs - compute in FP32 for numerical stability
        probs = F.softmax(scores.float(), dim=-1).to(scores.dtype)
        probs = self.dropout(probs)

        # Context
        context = torch.matmul(probs, v)

        if encounter:
            # [batch, num_segments, n_heads, seq_len, head_dim]
            # -> [batch, num_segments, seq_len, n_heads * head_dim]
            context = context.permute(0, 1, 3, 2, 4).contiguous()
        else:
            # [batch, n_heads, seq_len, head_dim]
            # -> [batch, seq_len, n_heads * head_dim]
            context = context.permute(0, 2, 1, 3).contiguous()

        new_shape = context.size()[:-2] + (self.all_head_size,)
        context = context.view(*new_shape)

        return context


class HiBEHRTSelfOutput(nn.Module):
    """Output projection with residual connection."""

    def __init__(self, config: HiBEHRTConfig):
        super().__init__()
        self.dense = nn.Linear(config.d_model, config.d_model)
        self.layer_norm = nn.LayerNorm(config.d_model, eps=1e-12)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layer_norm(hidden_states + input_tensor)
        return hidden_states


class HiBEHRTAttention(nn.Module):
    """Complete attention block with self-attention and output projection."""

    def __init__(self, config: HiBEHRTConfig):
        super().__init__()
        self.self_attn = HiBEHRTSelfAttention(config)
        self.output = HiBEHRTSelfOutput(config)

    def forward(self, hidden_states, attention_mask, encounter=True):
        attn_output = self.self_attn(hidden_states, attention_mask, encounter)
        output = self.output(attn_output, hidden_states)
        return output


class HiBEHRTIntermediate(nn.Module):
    """Intermediate (FFN up-projection) layer."""

    def __init__(self, config: HiBEHRTConfig):
        super().__init__()
        self.dense = nn.Linear(config.d_model, config.d_ff)
        self.act_fn = get_activation(config.hidden_act)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        return hidden_states


class HiBEHRTOutput(nn.Module):
    """FFN down-projection with residual connection."""

    def __init__(self, config: HiBEHRTConfig):
        super().__init__()
        self.dense = nn.Linear(config.d_ff, config.d_model)
        self.layer_norm = nn.LayerNorm(config.d_model, eps=1e-12)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layer_norm(hidden_states + input_tensor)
        return hidden_states


class HiBEHRTLayer(nn.Module):
    """Single Transformer layer for Hi-BEHRT."""

    def __init__(self, config: HiBEHRTConfig):
        super().__init__()
        self.attention = HiBEHRTAttention(config)
        self.intermediate = HiBEHRTIntermediate(config)
        self.output = HiBEHRTOutput(config)

    def forward(self, hidden_states, attention_mask, encounter=True):
        attention_output = self.attention(hidden_states, attention_mask, encounter)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class HiBEHRTEncoder(nn.Module):
    """Stack of Transformer layers."""

    def __init__(self, config: HiBEHRTConfig, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([HiBEHRTLayer(config) for _ in range(num_layers)])

    def forward(self, hidden_states, attention_mask, encounter=True):
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask, encounter)
        return hidden_states


class HiBEHRTPooler(nn.Module):
    """
    Pool the encoder output by taking the first token ([CLS]).

    For encounter=True: Takes first token of each segment
    For encounter=False: Takes first segment's representation
    """

    def __init__(self, config: HiBEHRTConfig):
        super().__init__()
        self.dense = nn.Linear(config.d_model, config.d_model)
        self.activation = nn.Tanh()

    def forward(self, hidden_states, encounter=True):
        """
        Args:
            hidden_states: (batch, num_segments, seq_len, d_model) if encounter=True
                          (batch, num_segments, d_model) if encounter=False

        Returns:
            pooled: (batch, num_segments, d_model) if encounter=True
                   (batch, d_model) if encounter=False
        """
        if encounter:
            # Take first token of each segment: [batch, num_segments, d_model]
            first_token = hidden_states[:, :, 0]
        else:
            # Take first segment: [batch, d_model]
            first_token = hidden_states[:, 0]

        pooled = self.dense(first_token)
        pooled = self.activation(pooled)
        return pooled


# ============================================================================
# BYOL Components
# ============================================================================

class MLP(nn.Module):
    """MLP for BYOL Projector and Predictor."""

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size, eps=1e-12),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        return self.net(x)


# ============================================================================
# Hi-BEHRT Encoder (shared by BYOL and Classification)
# ============================================================================

class HiBEHRTEncoder_Full(nn.Module):
    """
    Full Hi-BEHRT encoder with sliding window segmentation.

    This is the SHARED encoder used by both:
    - HiBEHRTForBYOL (pretraining)
    - HiBEHRTForClassification (finetuning)

    This ensures weights can be correctly loaded from BYOL to classification.
    """

    def __init__(self, config: HiBEHRTConfig, window_size: int = 50, stride: int = 30):
        super().__init__()
        self.config = config
        self.window_size = window_size
        self.stride = stride

        # Embeddings
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.d_model, padding_idx=config.pad_token_id
        )
        self.position_embeddings = SinusoidalPositionEmbedding(window_size, config.d_model)

        # Time2Vec for time encoding (replaces age embedding)
        self.time2vec = Time2Vec(config.t2v_dim)
        self.time_proj = nn.Linear(config.t2v_dim, config.d_model)

        # Segment embedding (alternating 0/1)
        self.segment_embeddings = nn.Embedding(config.seg_vocab_size, config.d_model)

        self.embed_layer_norm = nn.LayerNorm(config.d_model, eps=1e-12)
        self.embed_dropout = nn.Dropout(config.dropout)

        # Local feature extractor
        self.extractor = HiBEHRTEncoder(config, config.n_extractor_layers)
        self.extractor_pooler = HiBEHRTPooler(config)

        # Feature aggregator
        self.aggregator = HiBEHRTEncoder(config, config.n_aggregator_layers)

    def segment_sequence(self, input_ids, attention_mask, time_values=None):
        """
        Segment flat sequence using sliding window.

        Args:
            input_ids: (batch, total_seq_len)
            attention_mask: (batch, total_seq_len)
            time_values: (batch, total_seq_len) optional time values

        Returns:
            segmented_ids: (batch, num_segments, window_size)
            segmented_mask: (batch, num_segments, window_size)
            segment_mask: (batch, num_segments) - which segments are valid
            segmented_time: (batch, num_segments, window_size) if time_values provided
        """
        batch_size, total_len = input_ids.shape
        device = input_ids.device

        # Calculate number of segments
        if total_len <= self.window_size:
            num_segments = 1
        else:
            num_segments = (total_len - self.window_size) // self.stride + 1
            if (total_len - self.window_size) % self.stride != 0:
                num_segments += 1

        # Limit to max_segments
        num_segments = min(num_segments, self.config.max_segments)

        # Create segments
        segmented_ids = torch.zeros(batch_size, num_segments, self.window_size, dtype=torch.long, device=device)
        segmented_mask = torch.zeros(batch_size, num_segments, self.window_size, dtype=torch.long, device=device)
        segment_mask = torch.zeros(batch_size, num_segments, dtype=torch.long, device=device)

        if time_values is not None:
            segmented_time = torch.zeros(batch_size, num_segments, self.window_size, dtype=torch.float, device=device)
        else:
            segmented_time = None

        for i in range(num_segments):
            start = i * self.stride
            end = min(start + self.window_size, total_len)
            length = end - start

            segmented_ids[:, i, :length] = input_ids[:, start:end]
            segmented_mask[:, i, :length] = attention_mask[:, start:end]

            if time_values is not None:
                segmented_time[:, i, :length] = time_values[:, start:end]

            # Mark segment as valid if any token is non-padding
            segment_mask[:, i] = (segmented_mask[:, i].sum(dim=-1) > 0).long()

        return segmented_ids, segmented_mask, segment_mask, segmented_time

    def forward(self, input_ids, attention_mask, time_values=None):
        """
        Args:
            input_ids: (batch, total_seq_len) flat token indices
            attention_mask: (batch, total_seq_len) attention mask
            time_values: (batch, total_seq_len) optional cumulative time values

        Returns:
            aggregated: (batch, num_segments, d_model) - aggregated segment representations
            global_mask: (batch, num_segments) - valid segment mask
        """
        # Segment the sequence
        segmented_ids, local_mask, global_mask, segmented_time = self.segment_sequence(
            input_ids, attention_mask, time_values
        )
        batch_size, num_segments, seq_len = segmented_ids.shape

        # Create position ids (0 to window_size-1 for each segment)
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).unsqueeze(0)
        position_ids = position_ids.expand(batch_size, num_segments, -1)

        # Create segment ids (alternating 0/1)
        segment_ids = torch.zeros_like(segmented_ids)
        segment_ids[:, 1::2, :] = 1  # Odd segments get 1

        # Embedding
        embedded = self.word_embeddings(segmented_ids)
        embedded = embedded + self.position_embeddings(position_ids)
        embedded = embedded + self.segment_embeddings(segment_ids)

        # Add time embedding if available
        if segmented_time is not None:
            time_emb = self.time2vec(segmented_time.float())
            time_emb = self.time_proj(time_emb)
            embedded = embedded + time_emb

        embedded = self.embed_layer_norm(embedded)
        embedded = self.embed_dropout(embedded)

        # Local extraction: create attention mask for local extractor
        local_attn_mask = local_mask.to(dtype=embedded.dtype)
        local_attn_mask = local_attn_mask.unsqueeze(2).unsqueeze(3)  # (batch, num_seg, 1, 1, seq_len)
        local_attn_mask = (1.0 - local_attn_mask) * -10000.0

        # Encode within segments
        encoded = self.extractor(embedded, local_attn_mask, encounter=True)

        # Pool: get segment representation
        segment_repr = self.extractor_pooler(encoded, encounter=True)  # (batch, num_segments, d_model)

        # Global aggregation: create attention mask
        global_attn_mask = global_mask.to(dtype=segment_repr.dtype)
        global_attn_mask = global_attn_mask.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, num_seg)
        global_attn_mask = (1.0 - global_attn_mask) * -10000.0

        # Aggregate across segments
        aggregated = self.aggregator(segment_repr, global_attn_mask, encounter=False)

        return aggregated, global_mask


# ============================================================================
# Hi-BEHRT for BYOL Pre-training
# ============================================================================

class HiBEHRTForBYOL(nn.Module):
    """
    Hi-BEHRT with BYOL pre-training support.

    BYOL (Bootstrap Your Own Latent) self-supervised learning:
    - Online network: encoder + projector + predictor
    - Target network: encoder + projector (EMA updated)
    - Loss: cosine similarity between online predictions and target projections
    """

    def __init__(self, config: HiBEHRTConfig, window_size: int = 50, stride: int = 30):
        super().__init__()
        self.config = config

        # Online network
        self.encoder = HiBEHRTEncoder_Full(config, window_size, stride)
        self.projector = MLP(
            config.d_model,
            config.projector_hidden_size,
            config.projector_output_size,
        )
        self.predictor = MLP(
            config.projector_output_size,
            config.projector_hidden_size,
            config.projector_output_size,
        )

        # Target network (EMA of online encoder + projector)
        self.target_encoder = copy.deepcopy(self.encoder)
        self.target_projector = copy.deepcopy(self.projector)

        # Freeze target network
        for param in self.target_encoder.parameters():
            param.requires_grad = False
        for param in self.target_projector.parameters():
            param.requires_grad = False

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.Embedding):
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)

    @torch.no_grad()
    def update_target_network(self, momentum: float = None):
        """Update target network with EMA of online network."""
        if momentum is None:
            momentum = self.config.byol_momentum

        for online_params, target_params in zip(
            self.encoder.parameters(), self.target_encoder.parameters()
        ):
            target_params.data = momentum * target_params.data + (1 - momentum) * online_params.data

        for online_params, target_params in zip(
            self.projector.parameters(), self.target_projector.parameters()
        ):
            target_params.data = momentum * target_params.data + (1 - momentum) * online_params.data

    def forward(
        self,
        input_ids,
        attention_mask,
        time_values=None,
        bernoulli_mask=None,
        apply_mask=False,
    ):
        """
        Forward pass for BYOL training.

        Args:
            input_ids: (batch, total_seq_len)
            attention_mask: (batch, total_seq_len)
            time_values: (batch, total_seq_len)
            bernoulli_mask: (batch, num_segments) - mask for BYOL augmentation
            apply_mask: whether to apply masking (for online view)

        Returns:
            y: (batch, num_segments, d_model) - encoder output
            z: (batch, num_segments, projector_output_size) - projected
            h: (batch, num_segments, projector_output_size) - predicted (online only)
            global_mask: (batch, num_segments) - valid segment mask
        """
        # Encode
        y, global_mask = self.encoder(input_ids, attention_mask, time_values)

        # Apply BYOL masking augmentation
        if apply_mask and bernoulli_mask is not None:
            prob = random.random()
            if prob < 0.85:
                # 85% of time: mask segment representations to 0
                mask = bernoulli_mask.unsqueeze(-1).to(y.dtype)  # (batch, num_seg, 1)
                y = y * mask
            else:
                # 15% of time: add random noise to masked segments
                noise_mask = torch.randn_like(y) * (1 - bernoulli_mask.unsqueeze(-1).to(y.dtype))
                y = y + noise_mask

        # Project
        z = self.projector(y)

        # Predict (online network only)
        h = self.predictor(z)

        return y, z, h, global_mask

    @torch.no_grad()
    def forward_target(self, input_ids, attention_mask, time_values=None):
        """Forward pass through target network (no gradient)."""
        y, global_mask = self.target_encoder(input_ids, attention_mask, time_values)
        z = self.target_projector(y)
        return y, z, global_mask

    def byol_loss(self, h_online, z_target, global_attention_mask, bernoulli_mask):
        """
        Compute BYOL loss (cosine similarity).

        Args:
            h_online: (batch, num_segments, d) - online predictions
            z_target: (batch, num_segments, d) - target projections
            global_attention_mask: (batch, num_segments) - valid segment mask
            bernoulli_mask: (batch, num_segments) - which segments were masked

        Returns:
            loss: scalar
        """
        # Normalize
        h_online = F.normalize(h_online, dim=-1)
        z_target = F.normalize(z_target, dim=-1)

        # Cosine similarity -> loss
        sim = (h_online * z_target).sum(dim=-1)  # (batch, num_segments)
        loss = 2 - 2 * sim

        # Mask padding
        loss = loss * global_attention_mask.to(loss.dtype)

        # Only compute loss on masked segments (BYOL augmentation)
        # bernoulli_mask: 1 = keep, 0 = masked -> we want loss on masked (inverted)
        loss = loss * (1 - bernoulli_mask.to(loss.dtype))

        # Average over valid positions
        num_masked = ((1 - bernoulli_mask) * global_attention_mask).sum()
        if num_masked > 0:
            loss = loss.sum() / num_masked
        else:
            loss = loss.sum() * 0  # No loss if nothing masked

        return loss

    def get_encoder_state_dict(self):
        """Get encoder state dict for loading into classification model."""
        return self.encoder.state_dict()


# ============================================================================
# Hi-BEHRT for Classification (with BYOL pretrained weights)
# ============================================================================

class HiBEHRTForClassification(nn.Module):
    """
    Hi-BEHRT with classification head.

    Uses the SAME encoder architecture as HiBEHRTForBYOL,
    ensuring BYOL pretrained weights can be correctly loaded.
    """

    def __init__(
        self,
        config: HiBEHRTConfig,
        num_classes: int,
        window_size: int = 50,
        stride: int = 30,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.config = config
        self.num_classes = num_classes

        # Use the same encoder as BYOL
        self.encoder = HiBEHRTEncoder_Full(config, window_size, stride)

        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(config.d_model, num_classes),
        )

        # Initialize classifier (encoder will be loaded from pretrained)
        self._init_classifier()

    def _init_classifier(self):
        """Initialize classifier weights."""
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
                if module.bias is not None:
                    module.bias.data.zero_()

    def forward(self, input_ids, attention_mask, time_values=None, labels=None):
        """
        Args:
            input_ids: (batch, total_seq_len) flat token indices
            attention_mask: (batch, total_seq_len)
            time_values: (batch, total_seq_len) optional cumulative time values
            labels: (batch,) optional

        Returns:
            dict with 'logits' and optionally 'loss'
        """
        # Encode
        aggregated, global_mask = self.encoder(input_ids, attention_mask, time_values)

        # Pool: take LAST valid segment (most recent information)
        batch_size = aggregated.shape[0]
        last_valid_idx = (global_mask.cumsum(dim=1) * global_mask).argmax(dim=1)  # (batch,)
        batch_indices = torch.arange(batch_size, device=input_ids.device)
        pooled = aggregated[batch_indices, last_valid_idx]  # (batch, d_model)

        # Classify
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)

        output = {"logits": logits, "pooled_output": pooled}

        if labels is not None:
            if self.num_classes == 1:
                loss = F.binary_cross_entropy_with_logits(logits.squeeze(-1), labels.float())
            else:
                loss = F.cross_entropy(logits, labels)
            output["loss"] = loss

        return output

    def load_byol_pretrained(self, checkpoint_path: str):
        """Load BYOL pretrained encoder weights."""
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'encoder_state_dict' in checkpoint:
            # Direct encoder state dict
            state_dict = {'encoder.' + k: v for k, v in checkpoint['encoder_state_dict'].items()}
        else:
            state_dict = checkpoint

        # Extract encoder weights from BYOL model
        encoder_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('encoder.'):
                encoder_state_dict[k] = v

        # Load encoder weights
        missing, unexpected = self.load_state_dict(encoder_state_dict, strict=False)

        # Filter out classifier keys from missing (they're expected to be missing)
        missing = [k for k in missing if not k.startswith('classifier.') and not k.startswith('dropout.')]

        print(f"Loaded BYOL pretrained weights from {checkpoint_path}")
        if missing:
            print(f"  Missing encoder keys: {len(missing)}")
            for k in missing[:5]:
                print(f"    - {k}")
            if len(missing) > 5:
                print(f"    ... and {len(missing) - 5} more")
        if unexpected:
            print(f"  Unexpected keys (ignored): {len(unexpected)}")

        return missing, unexpected


# ============================================================================
# Backward Compatibility Aliases
# ============================================================================

# Keep old names for backward compatibility
HiBEHRTSimple = HiBEHRTEncoder_Full
HiBEHRTSimpleForClassification = HiBEHRTForClassification


# ============================================================================
# Factory Functions
# ============================================================================

def create_hi_behrt_config(
    vocab_size: int = 15000,
    d_model: int = 768,
    n_extractor_layers: int = 4,
    n_aggregator_layers: int = 4,
    n_heads: int = 12,
    d_ff: int = 2048,
    max_seq_len: int = 50,
    max_segments: int = 40,
    dropout: float = 0.1,
    **kwargs,
) -> HiBEHRTConfig:
    """Create HiBEHRTConfig with given parameters."""
    return HiBEHRTConfig(
        vocab_size=vocab_size,
        d_model=d_model,
        n_extractor_layers=n_extractor_layers,
        n_aggregator_layers=n_aggregator_layers,
        n_heads=n_heads,
        d_ff=d_ff,
        max_seq_len=max_seq_len,
        max_segments=max_segments,
        dropout=dropout,
        **kwargs,
    )


def create_hi_behrt_for_byol(
    config: HiBEHRTConfig,
    window_size: int = 50,
    stride: int = 30,
) -> HiBEHRTForBYOL:
    """Create Hi-BEHRT model for BYOL pretraining."""
    return HiBEHRTForBYOL(config, window_size, stride)


def create_hi_behrt_for_classification(
    config: HiBEHRTConfig,
    num_classes: int,
    window_size: int = 50,
    stride: int = 30,
    dropout: float = 0.1,
) -> HiBEHRTForClassification:
    """Create Hi-BEHRT model for classification."""
    return HiBEHRTForClassification(
        config=config,
        num_classes=num_classes,
        window_size=window_size,
        stride=stride,
        dropout=dropout,
    )
