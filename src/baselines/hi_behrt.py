"""
Hi-BEHRT Baseline Model

Hierarchical BERT for EHR: two-level Transformer architecture for processing long EHR sequences.

Reference: Hi-BEHRT: Hierarchical Transformer-based model for accurate prediction of clinical
events using multimodal longitudinal electronic health records (Li et al., 2021)

Core Architecture:
1. Embedding Layer: Token + Age + Segment + Position (sinusoidal)
2. Local Feature Extractor: Transformer operating on segments within sliding windows
3. Feature Aggregator: Transformer globally summarizing segment representations

Key differences from HAT:
- HAT: hierarchical attention with explicit visit/encounter structure
- Hi-BEHRT: sliding window segmentation with local-to-global two-stage Transformer

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
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional


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
    age_vocab_size: int = 200  # max age in years
    seg_vocab_size: int = 2  # alternating 0/1 for segment embedding
    initializer_range: float = 0.02
    hidden_act: str = "gelu"


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


class SinusoidalPositionEmbedding(nn.Module):
    """Sinusoidal position embedding (fixed, not learned)."""

    def __init__(self, max_len: int, d_model: int):
        super().__init__()

        # Create sinusoidal position encoding
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, position_ids):
        """
        Args:
            position_ids: (batch, ..., seq_len) position indices

        Returns:
            position_embeddings: (batch, ..., seq_len, d_model)
        """
        return self.pe[position_ids]


class HiBEHRTEmbedding(nn.Module):
    """
    Hi-BEHRT Embedding Layer.

    Combines: Token + Age + Segment + Position embeddings
    Position uses sinusoidal encoding (as in original paper).
    """

    def __init__(self, config: HiBEHRTConfig):
        super().__init__()

        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.d_model, padding_idx=config.pad_token_id
        )
        self.age_embeddings = nn.Embedding(config.age_vocab_size, config.d_model)
        self.segment_embeddings = nn.Embedding(config.seg_vocab_size, config.d_model)
        self.position_embeddings = SinusoidalPositionEmbedding(config.max_seq_len, config.d_model)

        self.layer_norm = nn.LayerNorm(config.d_model, eps=1e-12)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, token_ids, age_ids, segment_ids, position_ids):
        """
        Args:
            token_ids: (batch, num_segments, seq_len)
            age_ids: (batch, num_segments, seq_len)
            segment_ids: (batch, num_segments, seq_len)
            position_ids: (batch, num_segments, seq_len)

        Returns:
            embeddings: (batch, num_segments, seq_len, d_model)
        """
        word_emb = self.word_embeddings(token_ids)
        age_emb = self.age_embeddings(age_ids)
        seg_emb = self.segment_embeddings(segment_ids)
        pos_emb = self.position_embeddings(position_ids)

        embeddings = word_emb + age_emb + seg_emb + pos_emb
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


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

        # Attention probs
        probs = F.softmax(scores, dim=-1)
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


class LocalFeatureExtractor(nn.Module):
    """
    Local Feature Extractor: Processes segments with sliding window.

    Applies Transformer within each segment and pools to get segment representation.
    """

    def __init__(self, config: HiBEHRTConfig):
        super().__init__()
        self.encoder = HiBEHRTEncoder(config, config.n_extractor_layers)
        self.pooler = HiBEHRTPooler(config)

    def forward(self, hidden_states, attention_mask):
        """
        Args:
            hidden_states: (batch, num_segments, seq_len, d_model)
            attention_mask: (batch, num_segments, seq_len) - 1 for valid, 0 for padding

        Returns:
            segment_repr: (batch, num_segments, d_model)
        """
        # Create attention mask: (batch, num_segments, 1, 1, seq_len)
        mask = attention_mask.to(dtype=hidden_states.dtype)
        extended_mask = mask.unsqueeze(2).unsqueeze(3)
        extended_mask = (1.0 - extended_mask) * -10000.0

        # Encode within each segment
        encoded = self.encoder(hidden_states, extended_mask, encounter=True)

        # Pool: get segment representation from first token
        segment_repr = self.pooler(encoded, encounter=True)

        return segment_repr


class FeatureAggregator(nn.Module):
    """
    Feature Aggregator: Globally summarizes segment representations.

    Applies Transformer across all segment representations.
    """

    def __init__(self, config: HiBEHRTConfig):
        super().__init__()
        self.encoder = HiBEHRTEncoder(config, config.n_aggregator_layers)

    def forward(self, hidden_states, attention_mask):
        """
        Args:
            hidden_states: (batch, num_segments, d_model)
            attention_mask: (batch, num_segments) - 1 for valid, 0 for padding

        Returns:
            aggregated: (batch, num_segments, d_model)
        """
        # Create attention mask: (batch, 1, 1, num_segments)
        mask = attention_mask.to(dtype=hidden_states.dtype)
        extended_mask = mask.unsqueeze(1).unsqueeze(2)
        extended_mask = (1.0 - extended_mask) * -10000.0

        # Aggregate across segments
        aggregated = self.encoder(hidden_states, extended_mask, encounter=False)

        return aggregated


class HiBEHRT(nn.Module):
    """
    Hi-BEHRT: Hierarchical BERT for Electronic Health Records.

    Two-level Transformer architecture:
    1. Local Feature Extractor: Processes segments (sliding windows)
    2. Feature Aggregator: Globally summarizes segment representations

    Input shape: (batch, num_segments, seq_len)
    Output shape: (batch, num_segments, d_model)
    """

    def __init__(self, config: HiBEHRTConfig):
        super().__init__()
        self.config = config

        # Embedding layer
        self.embedding = HiBEHRTEmbedding(config)

        # Local feature extractor (within segments)
        self.extractor = LocalFeatureExtractor(config)

        # Feature aggregator (across segments)
        self.aggregator = FeatureAggregator(config)

        # Final pooler
        self.pooler = HiBEHRTPooler(config)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights."""
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

    def forward(
        self,
        token_ids,
        age_ids,
        segment_ids,
        position_ids,
        local_attention_mask,
        global_attention_mask,
    ):
        """
        Args:
            token_ids: (batch, num_segments, seq_len) token indices
            age_ids: (batch, num_segments, seq_len) age indices
            segment_ids: (batch, num_segments, seq_len) segment indices (alternating 0/1)
            position_ids: (batch, num_segments, seq_len) position indices within segment
            local_attention_mask: (batch, num_segments, seq_len) mask for local extractor
            global_attention_mask: (batch, num_segments) mask for aggregator

        Returns:
            aggregated_output: (batch, num_segments, d_model)
        """
        # Embedding
        embedded = self.embedding(token_ids, age_ids, segment_ids, position_ids)

        # Local feature extraction (within segments)
        segment_repr = self.extractor(embedded, local_attention_mask)

        # Global aggregation (across segments)
        aggregated = self.aggregator(segment_repr, global_attention_mask)

        return aggregated

    def get_pooled_output(
        self,
        token_ids,
        age_ids,
        segment_ids,
        position_ids,
        local_attention_mask,
        global_attention_mask,
    ):
        """
        Get pooled output for classification.

        Returns:
            pooled: (batch, d_model)
        """
        aggregated = self.forward(
            token_ids, age_ids, segment_ids, position_ids,
            local_attention_mask, global_attention_mask
        )
        pooled = self.pooler(aggregated, encounter=False)
        return pooled


class HiBEHRTForSequenceClassification(nn.Module):
    """
    Hi-BEHRT with classification head for downstream tasks.

    Uses pooled output (first segment) for classification.
    """

    def __init__(
        self,
        config: HiBEHRTConfig,
        num_classes: int,
        dropout: float = 0.1,
        freeze_encoder: bool = False,
    ):
        super().__init__()
        self.config = config
        self.num_classes = num_classes

        # Hi-BEHRT encoder
        self.encoder = HiBEHRT(config)

        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(config.d_model, num_classes)

        if freeze_encoder:
            self.freeze_encoder()

    def freeze_encoder(self):
        """Freeze encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        """Unfreeze encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = True

    def forward(
        self,
        token_ids,
        age_ids,
        segment_ids,
        position_ids,
        local_attention_mask,
        global_attention_mask,
        labels=None,
    ):
        """
        Args:
            token_ids: (batch, num_segments, seq_len)
            age_ids: (batch, num_segments, seq_len)
            segment_ids: (batch, num_segments, seq_len)
            position_ids: (batch, num_segments, seq_len)
            local_attention_mask: (batch, num_segments, seq_len)
            global_attention_mask: (batch, num_segments)
            labels: (batch,) optional

        Returns:
            dict with 'logits' and optionally 'loss'
        """
        # Get pooled output
        pooled = self.encoder.get_pooled_output(
            token_ids, age_ids, segment_ids, position_ids,
            local_attention_mask, global_attention_mask
        )

        # Classify
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)

        output = {"logits": logits, "pooled_output": pooled}

        if labels is not None:
            if self.num_classes == 1:
                # Binary classification
                loss = F.binary_cross_entropy_with_logits(logits.squeeze(-1), labels.float())
            else:
                # Multi-class classification
                loss = F.cross_entropy(logits, labels)
            output["loss"] = loss

        return output

    def load_pretrained(self, checkpoint_path: str, strict: bool = True):
        """Load pre-trained Hi-BEHRT weights."""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        # Load encoder weights
        encoder_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('encoder.'):
                encoder_state_dict[k] = v
            else:
                encoder_state_dict[f'encoder.{k}'] = v

        missing, unexpected = self.load_state_dict(encoder_state_dict, strict=False)
        missing = [k for k in missing if not k.startswith('classifier.')]

        if missing and strict:
            raise RuntimeError(f"Missing keys: {missing}")

        return missing, unexpected


class HiBEHRTSimple(nn.Module):
    """
    Simplified Hi-BEHRT that works with flat input format.

    This version internally handles the segmentation using sliding window,
    making it easier to use with existing data pipelines.

    Input: flat sequence (batch, total_seq_len)
    Internally: segments using sliding window, then applies hierarchical processing
    """

    def __init__(self, config: HiBEHRTConfig, window_size: int = 50, stride: int = 30):
        super().__init__()
        self.config = config
        self.window_size = window_size
        self.stride = stride

        # Core Hi-BEHRT components
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.d_model, padding_idx=config.pad_token_id
        )
        self.position_embeddings = SinusoidalPositionEmbedding(window_size, config.d_model)
        self.layer_norm = nn.LayerNorm(config.d_model, eps=1e-12)
        self.dropout = nn.Dropout(config.dropout)

        # Local feature extractor
        self.extractor = HiBEHRTEncoder(config, config.n_extractor_layers)
        self.extractor_pooler = HiBEHRTPooler(config)

        # Feature aggregator
        self.aggregator = HiBEHRTEncoder(config, config.n_aggregator_layers)

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

    def segment_sequence(self, input_ids, attention_mask):
        """
        Segment flat sequence using sliding window.

        Args:
            input_ids: (batch, total_seq_len)
            attention_mask: (batch, total_seq_len)

        Returns:
            segmented_ids: (batch, num_segments, window_size)
            segmented_mask: (batch, num_segments, window_size)
            segment_mask: (batch, num_segments) - which segments are valid
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

        for i in range(num_segments):
            start = i * self.stride
            end = min(start + self.window_size, total_len)
            length = end - start

            segmented_ids[:, i, :length] = input_ids[:, start:end]
            segmented_mask[:, i, :length] = attention_mask[:, start:end]

            # Mark segment as valid if any token is non-padding
            segment_mask[:, i] = (segmented_mask[:, i].sum(dim=-1) > 0).long()

        return segmented_ids, segmented_mask, segment_mask

    def forward(self, input_ids, attention_mask):
        """
        Args:
            input_ids: (batch, total_seq_len) flat token indices
            attention_mask: (batch, total_seq_len) attention mask

        Returns:
            pooled_output: (batch, d_model)
            aggregated: (batch, num_segments, d_model)
        """
        # Segment the sequence
        segmented_ids, local_mask, global_mask = self.segment_sequence(input_ids, attention_mask)
        batch_size, num_segments, seq_len = segmented_ids.shape

        # Create position ids (0 to window_size-1 for each segment)
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).unsqueeze(0)
        position_ids = position_ids.expand(batch_size, num_segments, -1)

        # Embedding
        embedded = self.word_embeddings(segmented_ids)
        embedded = embedded + self.position_embeddings(position_ids)
        embedded = self.layer_norm(embedded)
        embedded = self.dropout(embedded)

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

        # Pool LAST valid segment for classification (most recent information)
        # Find the last valid segment for each sample
        # global_mask: (batch, num_segments) - 1 for valid, 0 for padding
        last_valid_idx = (global_mask.cumsum(dim=1) * global_mask).argmax(dim=1)  # (batch,)
        batch_indices = torch.arange(batch_size, device=input_ids.device)
        pooled = aggregated[batch_indices, last_valid_idx]  # (batch, d_model)

        return pooled, aggregated


class HiBEHRTSimpleForClassification(nn.Module):
    """
    Simplified Hi-BEHRT with classification head.

    Works with flat input format, internally handles segmentation.
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

        # Hi-BEHRT encoder
        self.encoder = HiBEHRTSimple(config, window_size, stride)

        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(config.d_model, num_classes),
        )

    def forward(self, input_ids, attention_mask, labels=None):
        """
        Args:
            input_ids: (batch, total_seq_len) flat token indices
            attention_mask: (batch, total_seq_len)
            labels: (batch,) optional

        Returns:
            dict with 'logits' and optionally 'loss'
        """
        pooled, _ = self.encoder(input_ids, attention_mask)
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


def create_hi_behrt_model(
    vocab_size: int = 15000,
    d_model: int = 150,
    n_extractor_layers: int = 4,
    n_aggregator_layers: int = 4,
    n_heads: int = 6,
    d_ff: int = 108,
    max_seq_len: int = 50,
    max_segments: int = 40,
    dropout: float = 0.2,
    **kwargs,
) -> HiBEHRT:
    """Create a Hi-BEHRT model with given configuration."""
    config = HiBEHRTConfig(
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
    return HiBEHRT(config)
