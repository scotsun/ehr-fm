"""
HEART Baseline Model

Heterogeneous Relation-Aware Transformer for EHR data.

Reference: HEART: Learning better representation of EHR data with a
heterogeneous relation-aware transformer (Huang et al., 2024)

Core Ideas from Paper:
1. Heterogeneous Relation Embedding: Type-specific transformations to capture
   pairwise relations between different entity types (diagnosis, medication, etc.)
2. Multi-Level Attention: Entity-level (within visit) + Encounter-level (across visits)
3. Biased Self-Attention: Attention scores biased by relation embeddings

Adapted for our data format:
- code_type: diagnosis, procedure, lab, medication (4 types)
- visit_seq: visit boundary identifier
- time_offset_hours: temporal information within visit
- days_since_prior_admission: temporal information across visits
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers.modeling_utils import PreTrainedModel
from transformers import PretrainedConfig


class HEARTConfig(PretrainedConfig):
    """Configuration for HEART baseline model."""
    model_type = "heart"

    def __init__(
        self,
        vocab_size: int = 15000,
        d_model: int = 768,
        n_blocks: int = 6,
        n_heads: int = 12,
        d_ff: int = 2048,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        pad_token_id: int = 0,
        # Token types: 0=PAD, 1=CLS, 2=SEP, 3=MASK, 4=DX, 5=PR, 6=LAB, 7=MED
        n_token_types: int = 8,
        edge_hidden_size: int = 64,
        max_visits: int = 50,
        # Time encoding
        use_time_encoding: bool = True,
        t2v_dim: int = 64,
        **kwargs,
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.n_token_types = n_token_types
        self.edge_hidden_size = edge_hidden_size
        self.max_visits = max_visits
        self.use_time_encoding = use_time_encoding
        self.t2v_dim = t2v_dim


# ============================================================================
# Time Encoding (Time2Vec)
# ============================================================================

class Time2Vec(nn.Module):
    """
    Time2Vec: Learnable time encoding.
    From: Time2Vec: Learning Vector Representation of Time (Kazemi et al., 2019)
    """

    def __init__(self, d_out: int):
        super().__init__()
        self.d_out = d_out
        # Linear component
        self.w0 = nn.Parameter(torch.randn(1))
        self.b0 = nn.Parameter(torch.randn(1))
        # Periodic components
        self.w = nn.Parameter(torch.randn(d_out - 1))
        self.b = nn.Parameter(torch.randn(d_out - 1))

    def forward(self, t: Tensor) -> Tensor:
        """
        Args:
            t: (batch, seq_len) or (batch, seq_len, 1) - time values
        Returns:
            (batch, seq_len, d_out) - time embeddings
        """
        if t.dim() == 2:
            t = t.unsqueeze(-1)  # (B, L, 1)

        # Linear component: w0 * t + b0
        v0 = self.w0 * t + self.b0  # (B, L, 1)

        # Periodic components: sin(w * t + b)
        v = torch.sin(self.w * t + self.b)  # (B, L, d_out-1)

        return torch.cat([v0, v], dim=-1)  # (B, L, d_out)


# ============================================================================
# Basic Components
# ============================================================================

def gelu(x):
    """Gaussian Error Linear Unit activation."""
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class FeedForward(nn.Module):
    """Position-wise Feed-Forward Network."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.dropout(gelu(self.fc1(x))))


# ============================================================================
# HEART Core: Heterogeneous Relation Embedding
# ============================================================================

class EdgeModule(nn.Module):
    """
    Generates edge embeddings based on token type pairs.

    Paper Equations (5-6):
    r_n = Linear_τ(V_n)(v_n)  -- type-specific transformation
    r_m = Linear_τ(V_m)(v_m)
    r_{n←m} = Linear(r_n || r_m)  -- combine to get relation embedding

    This captures heterogeneous relations: how a diagnosis relates to a medication
    is different from how two diagnoses relate to each other.
    """

    def __init__(self, d_model: int, edge_hidden_size: int, n_token_types: int = 8):
        super().__init__()
        self.n_types = n_token_types
        self.d_model = d_model

        # Type-specific transformations (Paper Eq. 5)
        self.left_transform = nn.Parameter(torch.zeros(n_token_types, d_model, d_model))
        self.right_transform = nn.Parameter(torch.zeros(n_token_types, d_model, d_model))
        # Combine embeddings (Paper Eq. 6)
        self.output = nn.Linear(d_model * 2, edge_hidden_size)

        self._init_parameters()

    def _init_parameters(self):
        nn.init.xavier_uniform_(self.left_transform)
        nn.init.xavier_uniform_(self.right_transform)

    def forward(self, token_embs: Tensor, token_types: Tensor) -> Tensor:
        """
        Args:
            token_embs: (batch, seq_len, d_model)
            token_types: (batch, seq_len) - integer token type IDs

        Returns:
            edge_embs: (batch, seq_len, seq_len, edge_hidden_size)
        """
        batch_size, seq_len, _ = token_embs.size()

        # Get type-specific transformations
        left_trans = self.left_transform[token_types]  # (B, L, D, D)
        right_trans = self.right_transform[token_types]  # (B, L, D, D)

        # Apply transformations: r_n = Linear_τ(v_n)
        # einsum: 'bld,blmd->blm' means output[b,l,m] = sum_d(input[b,l,d] * weight[b,l,m,d])
        left_embs = torch.einsum('bld,blmd->blm', token_embs, left_trans)  # (B, L, D)
        right_embs = torch.einsum('bld,blmd->blm', token_embs, right_trans)  # (B, L, D)

        # Create pairwise edge embeddings: r_{n←m} = Linear(r_n || r_m)
        left_expanded = left_embs.unsqueeze(2).expand(-1, -1, seq_len, -1)  # (B, L, L, D)
        right_expanded = right_embs.unsqueeze(1).expand(-1, seq_len, -1, -1)  # (B, L, L, D)

        edge_embs = torch.cat([left_expanded, right_expanded], dim=-1)  # (B, L, L, 2D)
        return self.output(edge_embs)  # (B, L, L, edge_hidden_size)


# ============================================================================
# HEART Core: Biased Self-Attention with Edge Embeddings
# ============================================================================

class MultiHeadEdgeAttention(nn.Module):
    """
    Multi-Head Attention with Edge Representation bias.

    Paper Equation (7):
    a_nm = Softmax_n(1/√d * v_n^qry^T * v_m^key + b_nm)
    where b_nm = Linear(LN(r_{n←m})) is the edge bias

    Paper Equation (8):
    v'_n = Linear(LN(Σ a_nm * v_m^val || Σ a_nm * r_{n←m})) + v_n
    Aggregates both value embeddings and relation embeddings.
    """

    def __init__(self, d_model: int, n_heads: int, edge_hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.d_k = d_model // n_heads
        self.d_edge = edge_hidden_size
        self.n_heads = n_heads

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)

        # Edge-related projections
        self.W_K_edge = nn.Linear(edge_hidden_size, edge_hidden_size)
        self.W_edge_bias = nn.Linear(edge_hidden_size, 1)  # For attention bias b_nm
        self.W_edge_output = nn.Linear(edge_hidden_size * n_heads, d_model)

        # Output combines value context and edge context (Paper Eq. 8)
        self.W_output = nn.Linear(d_model * 2, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q: Tensor, K: Tensor, V: Tensor, mask: Tensor, edge_embs: Tensor) -> Tensor:
        """
        Args:
            Q, K, V: (batch, seq_len, d_model)
            mask: (batch, seq_len, seq_len) - True for positions to mask
            edge_embs: (batch, seq_len, seq_len, edge_hidden_size)
        """
        batch_size, n_tokens = Q.size(0), Q.size(1)

        # Standard QKV projections
        q = self.W_Q(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.W_K(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.W_V(V).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # Edge attention bias: b_nm = Linear(edge_embs)
        k_edge = self.W_K_edge(edge_embs)  # (B, L, L, edge_hidden_size)
        edge_bias = self.W_edge_bias(edge_embs).view(batch_size, 1, n_tokens, n_tokens)
        edge_bias = edge_bias * (2 ** -0.5)

        # Compute attention with edge bias (Paper Eq. 7)
        scores = torch.matmul(q, k.transpose(-1, -2)) * ((2 * self.d_k) ** -0.5)
        scores = scores + edge_bias

        if mask is not None:
            mask = mask.unsqueeze(1)  # (B, 1, L, L)
            scores = scores.masked_fill(mask, float('-inf'))

        attn = self.dropout(F.softmax(scores, dim=-1))

        # Aggregate value embeddings
        context = torch.matmul(attn, v)  # (B, H, L, D_k)

        # Aggregate relation embeddings (Paper Eq. 8)
        # einsum: 'bhnm,bnmd->bhnd' means weighted sum of edge embeddings
        edge_context = torch.einsum('bhnm,bnmd->bhnd', attn, k_edge)  # (B, H, L, edge_hidden_size)

        # Combine contexts
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_k)
        edge_context = edge_context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_edge)
        edge_context = self.W_edge_output(edge_context)

        return self.W_output(torch.cat([context, edge_context], dim=-1))


class EdgeTransformerBlock(nn.Module):
    """
    Transformer Block with Edge Representation (Entity-Level Attention).
    Used for attention within a visit.
    """

    def __init__(self, d_model: int, d_ff: int, n_heads: int, edge_hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadEdgeAttention(d_model, n_heads, edge_hidden_size, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm_attn = nn.LayerNorm(d_model)
        self.norm_ffn = nn.LayerNorm(d_model)
        self.norm_edge = nn.LayerNorm(edge_hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, edge_embs: Tensor, mask: Tensor = None) -> Tensor:
        # Pre-norm attention with edge bias
        norm_x = self.norm_attn(x)
        norm_edge = self.norm_edge(edge_embs)
        x = x + self.dropout(self.self_attn(norm_x, norm_x, norm_x, mask, norm_edge))

        # Pre-norm FFN
        norm_x = self.norm_ffn(x)
        x = x + self.dropout(self.ffn(norm_x))
        return x


# ============================================================================
# HEART Core: Encounter-Level Attention (Cross-Visit)
# ============================================================================

class EncounterLevelAttention(nn.Module):
    """
    Encounter-Level Self-Attention Module.

    Paper Equation (9):
    d_i^qry = Linear(d_i + t_i)
    d_i^key = Linear(d_i + t_i)

    This operates on the [CLS] tokens (demography tokens in paper)
    across different visits, incorporating visit order as position encoding.
    """

    def __init__(self, d_model: int, n_heads: int, max_visits: int = 50, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # Visit position encoding (Paper Eq. 9: t_i)
        self.visit_pos_encoding = nn.Embedding(max_visits, d_model)

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, cls_tokens: Tensor, visit_indices: Tensor, visit_mask: Tensor) -> Tensor:
        """
        Args:
            cls_tokens: (batch, n_visits, d_model) - [CLS] token for each visit
            visit_indices: (batch, n_visits) - visit index (0, 1, 2, ...)
            visit_mask: (batch, n_visits) - True for valid visits

        Returns:
            Updated [CLS] tokens: (batch, n_visits, d_model)
        """
        batch_size, n_visits, _ = cls_tokens.size()

        # Add visit position encoding (Paper Eq. 9)
        pos_enc = self.visit_pos_encoding(visit_indices)
        x = cls_tokens + pos_enc

        # Pre-norm
        x = self.norm(x)

        # QKV projections
        q = self.W_Q(x).view(batch_size, n_visits, self.n_heads, self.d_k).transpose(1, 2)
        k = self.W_K(x).view(batch_size, n_visits, self.n_heads, self.d_k).transpose(1, 2)
        v = self.W_V(cls_tokens).view(batch_size, n_visits, self.n_heads, self.d_k).transpose(1, 2)

        # Attention scores
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.d_k)

        # Mask invalid visits
        if visit_mask is not None:
            attn_mask = ~visit_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, V)
            scores = scores.masked_fill(attn_mask, float('-inf'))

        attn = self.dropout(F.softmax(scores, dim=-1))
        context = torch.matmul(attn, v)

        # Reshape and project
        context = context.transpose(1, 2).contiguous().view(batch_size, n_visits, -1)
        out = self.W_O(context) + cls_tokens  # Residual

        return out


# ============================================================================
# Standard Transformer Block (fallback without edge module)
# ============================================================================

class TransformerBlock(nn.Module):
    """Standard Transformer Block with Pre-Norm."""

    def __init__(self, d_model: int, d_ff: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_k = d_model // n_heads
        self.n_heads = n_heads

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm_attn = nn.LayerNorm(d_model)
        self.norm_ffn = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        batch_size = x.size(0)
        norm_x = self.norm_attn(x)

        q = self.W_Q(norm_x).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.W_K(norm_x).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.W_V(norm_x).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1), float('-inf'))

        attn = self.dropout(F.softmax(scores, dim=-1))
        context = torch.matmul(attn, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_k)

        x = x + self.dropout(self.W_O(context))
        norm_x = self.norm_ffn(x)
        x = x + self.dropout(self.ffn(norm_x))
        return x


# ============================================================================
# HEART Model
# ============================================================================

class HEART(PreTrainedModel):
    """
    HEART: Heterogeneous Relation-Aware Transformer.

    Architecture (following paper):
    1. Token Embeddings + Time Encoding (T2V)
    2. Edge Module: Generate pairwise relation embeddings
    3. N x (Entity-Level Attention + Encounter-Level Attention)
       - Entity-Level: EdgeTransformerBlock within each visit
       - Encounter-Level: Cross-visit attention on [CLS] tokens
    4. Final LayerNorm + LM Head

    Simplified for our data format:
    - Uses code_type for heterogeneous token types
    - Uses visit_seq for multi-level attention
    - Uses time_offset_hours for temporal encoding
    """
    config_class = HEARTConfig
    base_model_prefix = "heart"

    def __init__(self, config: HEARTConfig):
        super().__init__(config)
        self.config = config

        # Token embeddings
        self.embeddings = nn.Embedding(
            config.vocab_size, config.d_model, padding_idx=config.pad_token_id
        )

        # Time encoding (for time_offset_hours)
        if config.use_time_encoding:
            self.t2v = Time2Vec(config.t2v_dim)
            self.time_proj = nn.Linear(config.t2v_dim, config.d_model)
        else:
            self.t2v = None

        self.emb_dropout = nn.Dropout(config.dropout)

        # Edge module for heterogeneous relations
        self.edge_module = EdgeModule(
            config.d_model, config.edge_hidden_size, config.n_token_types
        )

        # Entity-level attention blocks (within visit)
        self.entity_blocks = nn.ModuleList([
            EdgeTransformerBlock(
                config.d_model, config.d_ff, config.n_heads,
                config.edge_hidden_size, config.dropout
            )
            for _ in range(config.n_blocks)
        ])

        # Encounter-level attention blocks (across visits)
        self.encounter_blocks = nn.ModuleList([
            EncounterLevelAttention(
                config.d_model, config.n_heads, config.max_visits, config.dropout
            )
            for _ in range(config.n_blocks)
        ])

        # Final norm and LM head
        self.final_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        token_types: Tensor = None,
        visit_ids: Tensor = None,
        time_offsets: Tensor = None,
    ):
        """
        Args:
            input_ids: (batch, seq_len) - Token IDs
            attention_mask: (batch, seq_len) - 1 for valid, 0 for padding
            token_types: (batch, seq_len) - Token type IDs (0=PAD, 4=DX, 5=PR, 6=LAB, 7=MED)
            visit_ids: (batch, seq_len) - Visit index for each token (0, 1, 2, ...)
            time_offsets: (batch, seq_len) - time_offset_hours for each token

        Returns:
            logits: (batch, seq_len, vocab_size)
            hidden_states: (batch, seq_len, d_model)
        """
        h = self.encode(input_ids, attention_mask, token_types, visit_ids, time_offsets)
        logits = self.lm_head(h)
        return logits, h

    def encode(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        token_types: Tensor = None,
        visit_ids: Tensor = None,
        time_offsets: Tensor = None,
    ) -> Tensor:
        """Encode input to hidden representations."""
        batch_size, seq_len = input_ids.size()
        device = input_ids.device

        # Default values
        if token_types is None:
            token_types = torch.zeros_like(input_ids)
        if visit_ids is None:
            visit_ids = torch.zeros_like(input_ids)

        # Token embeddings
        h = self.embeddings(input_ids)

        # Add time encoding if available
        if self.t2v is not None and time_offsets is not None:
            time_emb = self.t2v(time_offsets)
            time_emb = self.time_proj(time_emb)
            h = h + time_emb

        h = self.emb_dropout(h)

        # Compute edge embeddings (heterogeneous relations)
        edge_embs = self.edge_module(h, token_types)

        # Create attention mask: True for positions to mask
        pad_mask = ~attention_mask.bool()
        pair_pad_mask = pad_mask.unsqueeze(1).expand(-1, seq_len, -1)

        # Multi-level attention: Entity + Encounter
        for i in range(self.config.n_blocks):
            # Entity-level attention (with edge bias)
            h = self.entity_blocks[i](h, edge_embs, pair_pad_mask)

            # Encounter-level attention (across visits on [CLS] tokens)
            if visit_ids is not None:
                h = self._apply_encounter_attention(h, i, input_ids, attention_mask, visit_ids)

        h = self.final_norm(h)
        return h

    def _apply_encounter_attention(
        self,
        h: Tensor,
        layer_idx: int,
        input_ids: Tensor,
        attention_mask: Tensor,
        visit_ids: Tensor,
    ) -> Tensor:
        """
        Apply encounter-level attention on [CLS] tokens.

        Strategy: Find the first token of each visit (acts as visit [CLS]),
        apply cross-visit attention, then scatter back.
        """
        batch_size, seq_len, d_model = h.size()
        device = h.device

        # Find unique visits and their first token positions
        max_visits = self.config.max_visits

        # Collect [CLS] tokens (first token of each visit)
        cls_tokens = torch.zeros(batch_size, max_visits, d_model, device=device)
        visit_mask = torch.zeros(batch_size, max_visits, dtype=torch.bool, device=device)
        visit_indices = torch.arange(max_visits, device=device).unsqueeze(0).expand(batch_size, -1)
        cls_positions = torch.zeros(batch_size, max_visits, dtype=torch.long, device=device)

        for b in range(batch_size):
            valid_mask = attention_mask[b].bool()
            visits = visit_ids[b][valid_mask].unique()

            for i, v in enumerate(visits):
                if i >= max_visits:
                    break
                # Find first position of this visit
                visit_token_mask = (visit_ids[b] == v) & valid_mask
                first_pos = visit_token_mask.nonzero(as_tuple=True)[0][0]

                cls_tokens[b, i] = h[b, first_pos]
                visit_mask[b, i] = True
                cls_positions[b, i] = first_pos

        # Apply encounter-level attention
        updated_cls = self.encounter_blocks[layer_idx](cls_tokens, visit_indices, visit_mask)

        # Scatter updated [CLS] back to original positions (avoid in-place modification)
        h_new = h.clone()
        for b in range(batch_size):
            for i in range(max_visits):
                if visit_mask[b, i]:
                    h_new[b, cls_positions[b, i]] = updated_cls[b, i]

        return h_new


# ============================================================================
# Downstream Task Wrapper
# ============================================================================

class HEARTForSequenceClassification(nn.Module):
    """HEART with classification head for downstream tasks."""

    def __init__(
        self,
        config: HEARTConfig,
        num_classes: int,
        dropout: float = 0.1,
        freeze_encoder: bool = False,
    ):
        super().__init__()
        self.config = config
        self.num_classes = num_classes

        self.encoder = HEART(config)

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(config.d_model, num_classes),
        )

        if freeze_encoder:
            self.freeze_encoder()

    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = True

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        token_types: Tensor = None,
        visit_ids: Tensor = None,
        time_offsets: Tensor = None,
        labels: Tensor = None,
    ):
        """
        Args:
            input_ids, attention_mask, token_types, visit_ids, time_offsets: See HEART.forward
            labels: (batch,) - Optional labels for loss computation

        Returns:
            dict with 'logits' and optionally 'loss'
        """
        hidden_states = self.encoder.encode(
            input_ids, attention_mask, token_types, visit_ids, time_offsets
        )

        # Use first token ([CLS]) for classification
        cls_hidden = hidden_states[:, 0, :]
        cls_hidden = self.dropout(cls_hidden)
        logits = self.classifier(cls_hidden)

        output = {"logits": logits, "hidden_states": hidden_states}

        if labels is not None:
            if self.num_classes == 1:
                loss = F.binary_cross_entropy_with_logits(logits.squeeze(-1), labels.float())
            else:
                loss = F.cross_entropy(logits, labels)
            output["loss"] = loss

        return output


class HEARTForMultiLabelClassification(nn.Module):
    """HEART for multi-label classification (e.g., next diagnosis prediction)."""

    def __init__(
        self,
        config: HEARTConfig,
        num_labels: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.config = config
        self.num_labels = num_labels

        self.encoder = HEART(config)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(config.d_model, num_labels)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        token_types: Tensor = None,
        visit_ids: Tensor = None,
        time_offsets: Tensor = None,
        labels: Tensor = None,
    ):
        hidden_states = self.encoder.encode(
            input_ids, attention_mask, token_types, visit_ids, time_offsets
        )

        cls_hidden = hidden_states[:, 0, :]
        cls_hidden = self.dropout(cls_hidden)
        logits = self.classifier(cls_hidden)

        output = {"logits": logits}

        if labels is not None:
            loss = F.binary_cross_entropy_with_logits(logits, labels.float())
            output["loss"] = loss

        return output


# ============================================================================
# Factory Function
# ============================================================================

def create_heart_model(
    vocab_size: int = 15000,
    d_model: int = 768,
    n_blocks: int = 6,
    n_heads: int = 12,
    d_ff: int = 2048,
    max_seq_len: int = 512,
    dropout: float = 0.1,
    n_token_types: int = 8,
    edge_hidden_size: int = 64,
    max_visits: int = 50,
    use_time_encoding: bool = True,
    **kwargs,
) -> HEART:
    """Create a HEART model with given configuration."""
    config = HEARTConfig(
        vocab_size=vocab_size,
        d_model=d_model,
        n_blocks=n_blocks,
        n_heads=n_heads,
        d_ff=d_ff,
        max_seq_len=max_seq_len,
        dropout=dropout,
        n_token_types=n_token_types,
        edge_hidden_size=edge_hidden_size,
        max_visits=max_visits,
        use_time_encoding=use_time_encoding,
        **kwargs,
    )
    return HEART(config)


# Token type mapping for our data format
TOKEN_TYPE_MAP = {
    'PAD': 0,
    'CLS': 1,
    'SEP': 2,
    'MASK': 3,
    'diagnosis': 4,  # DX:
    'procedure': 5,  # PR:
    'lab': 6,        # LAB:
    'medication': 7, # MED:
}
