"""
HEART Baseline Model

Heterogeneous Relation-Aware Transformer for EHR data.

Reference: HEART: Learning better representation of EHR data with a
heterogeneous relation-aware transformer (Huang et al., 2024)

Core Ideas from Paper (Eq. 5-9):
1. Heterogeneous Relation Embedding (Eq. 5-6): Type-specific transforms to capture
   pairwise relations between different entity types (diagnosis, medication, etc.)
   r_n = Linear_τ(v_n), r_{n←m} = Linear(r_n || r_m) → edge_embs (B, L, L, E)
2. Biased Self-Attention (Eq. 7): Attention scores biased by relation embeddings
   a_nm = Softmax(q^T k / sqrt(d) + b_nm) where b_nm = Linear(LN(r_{n←m}))
3. Edge Context Aggregation (Eq. 8): Edge embeddings aggregated in output
   v'_n = Linear(Σ a_nm * v_m || Σ a_nm * r_{n←m}) + v_n
4. Multi-Level Attention (Eq. 9): Entity-level (within visit) + Encounter-level (across visits)

Memory-Efficient Modifications (3 minimal changes to original HEART code):
1. Low-rank type transforms: (n_types, D, D) → (n_types, D, r) where r << D
   Reduces per-token transform from O(D²) to O(Dr) parameters
2. Decomposed pairwise output: Linear(cat(a,b)) = W_l@a + W_r@b + bias
   Avoids materializing (B, L, L, 2r) intermediate tensor
3. Softcap on edge bias scalar: cap * tanh(b/cap) prevents unbounded growth
   Root cause of original training instability (loss divergence after 2 epochs)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint as grad_checkpoint
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
        max_seq_len: int = 2048,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        pad_token_id: int = 0,
        # Token types: 0=PAD, 1=CLS, 2=SEP, 3=MASK, 4=DX, 5=PR, 6=LAB, 7=MED
        n_token_types: int = 8,
        edge_hidden_size: int = 64,  # Dimension of edge embeddings (E in paper)
        edge_rank: int = 32,  # Low-rank approximation rank (r << D)
        edge_softcap: float = 5.0,  # Soft-cap for edge bias scalar (stability fix)
        max_visits: int = 50,
        # Time encoding
        use_time_encoding: bool = True,
        t2v_dim: int = 64,
        # Debug options
        disable_encounter_attention: bool = False,
        # Memory optimization
        use_gradient_checkpointing: bool = False,
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
        self.edge_rank = edge_rank
        self.edge_softcap = edge_softcap
        self.max_visits = max_visits
        self.use_time_encoding = use_time_encoding
        self.t2v_dim = t2v_dim
        self.disable_encounter_attention = disable_encounter_attention
        self.use_gradient_checkpointing = use_gradient_checkpointing


# ============================================================================
# Time Encoding (Time2Vec)
# ============================================================================

class Time2Vec(nn.Module):
    """
    Time2Vec: Learnable time encoding.
    From: Time2Vec: Learning Vector Representation of Time (Kazemi et al., 2019)

    Note: All computations forced to FP32 for numerical stability with AMP.
    Large time values (thousands of hours) can overflow FP16.
    """

    def __init__(self, d_out: int):
        super().__init__()
        self.d_out = d_out
        self.w0 = nn.Parameter(torch.randn(1) * 0.01)
        self.b0 = nn.Parameter(torch.zeros(1))
        self.w = nn.Parameter(torch.randn(d_out - 1) * 0.01)
        self.b = nn.Parameter(torch.zeros(d_out - 1))

    def forward(self, t: Tensor) -> Tensor:
        original_dtype = t.dtype
        t = t.float()
        if t.dim() == 2:
            t = t.unsqueeze(-1)
        t_normalized = torch.sign(t) * torch.log1p(torch.abs(t))
        v0 = self.w0.float() * t_normalized + self.b0.float()
        v = torch.sin(self.w.float() * t_normalized + self.b.float())
        result = torch.cat([v0, v], dim=-1)
        return result.to(original_dtype)


# ============================================================================
# Basic Components
# ============================================================================

class FeedForward(nn.Module):
    """Position-wise Feed-Forward Network."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.dropout(F.gelu(self.fc1(x))))


# ============================================================================
# HEART Core: Heterogeneous Relation Embedding (Paper Eq. 5-6)
# ============================================================================

class EdgeModule(nn.Module):
    """
    Heterogeneous Relation Embedding Module (Paper Eq. 5-6).

    Faithful to original HEART implementation with 3 memory-efficient modifications:

    Original (O(D²) per type, O(L²·2D) intermediate):
        left_transform: (n_types, D, D)  → token → (D,) per type
        pairwise: cat(left_i, right_j)   → (B, L, L, 2D)
        output: Linear(2D, E)            → (B, L, L, E)

    This implementation (O(Dr) per type, no L²·2r intermediate):
        left_transform: (n_types, r, D)  → token → (r,) per type  [Mod 1: low-rank]
        output_left/right: Linear(r, E)  → (B, L, E) each         [Mod 2: decomposed]
        pairwise: broadcast addition     → (B, L, L, E)

    Output shape matches original: (B, L, L, edge_hidden_size)
    """

    def __init__(self, d_model: int, n_token_types: int, edge_hidden_size: int, edge_rank: int):
        super().__init__()
        self.n_types = n_token_types
        self.edge_rank = edge_rank

        # Type-specific transforms: (n_types, r, D) — low-rank version of original (n_types, D, D)
        self.left_transform = nn.Parameter(torch.zeros(n_token_types, edge_rank, d_model))
        self.right_transform = nn.Parameter(torch.zeros(n_token_types, edge_rank, d_model))

        # Decomposed output: Linear(cat(a,b)) = W_l@a + W_r@b + bias
        # Avoids materializing (B, L, L, 2r) intermediate
        self.output_left = nn.Linear(edge_rank, edge_hidden_size)
        self.output_right = nn.Linear(edge_rank, edge_hidden_size, bias=False)

        self._init_parameters()

    def _init_parameters(self):
        nn.init.xavier_uniform_(self.left_transform)
        nn.init.xavier_uniform_(self.right_transform)

    def forward(self, token_embs: Tensor, token_types: Tensor) -> Tensor:
        """
        Args:
            token_embs: (B, L, D)
            token_types: (B, L) integer token type IDs

        Returns:
            edge_embs: (B, L, L, edge_hidden_size) — pairwise relation embeddings
        """
        # Force FP32 for edge computation — FP16 einsum over D=768 dims can overflow
        with torch.amp.autocast('cuda', enabled=False):
            token_embs = token_embs.float()

            # Type-specific linear transform (Paper Eq. 5): r_n = Linear_τ(v_n)
            types_safe = token_types.clamp(0, self.n_types - 1)
            left_trans = self.left_transform[types_safe]    # (B, L, r, D)
            right_trans = self.right_transform[types_safe]  # (B, L, r, D)

            # einsum: for each (b,l), compute transform[b,l] @ token[b,l] → (r,)
            left_embs = torch.einsum('bld,blrd->blr', token_embs, left_trans)   # (B, L, r)
            right_embs = torch.einsum('bld,blrd->blr', token_embs, right_trans)  # (B, L, r)

            # Decomposed pairwise combination (Paper Eq. 6): r_{n←m} = Linear(r_n || r_m)
            # Linear(cat(a,b)) = W_l@a + W_r@b + bias — avoids (B, L, L, 2r) intermediate
            left_out = self.output_left(left_embs)    # (B, L, E) — includes bias
            right_out = self.output_right(right_embs)  # (B, L, E) — no bias

            # Broadcast addition produces (B, L, L, E) without materializing concat
            edge_embs = left_out.unsqueeze(2) + right_out.unsqueeze(1)  # (B, L, L, E)

        return edge_embs


# ============================================================================
# HEART Core: Biased Self-Attention with Edge Context (Paper Eq. 7-8)
# ============================================================================

class MultiHeadEdgeAttention(nn.Module):
    """
    Multi-Head Attention with heterogeneous edge bias and edge context aggregation.

    Paper Eq. 7: a_nm = Softmax(q^T k / sqrt(2d) + b_nm)
        where b_nm = Linear(LN(r_{n←m})) * 2^{-0.5}

    Paper Eq. 8: v'_n = Linear(Σ a_nm * v_m^val || Σ a_nm * k_edge(r_{n←m})) + v_n
        Two-stream aggregation: standard value context + edge context

    Modification [3]: softcap on edge_bias scalar to prevent unbounded growth.
    """

    def __init__(self, d_model: int, n_heads: int, edge_hidden_size: int,
                 dropout: float = 0.1, edge_softcap: float = 5.0):
        super().__init__()
        self.d_k = d_model // n_heads
        self.d_edge = edge_hidden_size
        self.n_heads = n_heads
        self.edge_softcap = edge_softcap

        # Standard QKV projections
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)

        # Edge-specific projections (from original transformer_rel.py)
        self.W_K_edge = nn.Linear(edge_hidden_size, edge_hidden_size)
        self.W_edge = nn.Linear(edge_hidden_size, 1)  # edge_embs → scalar bias
        self.W_edge_output = nn.Linear(self.d_edge * n_heads, d_model)

        # Output combines standard context and edge context (Eq. 8)
        self.W_output = nn.Linear(d_model * 2, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, Q: Tensor, K: Tensor, V: Tensor, mask: Tensor, edge_embs: Tensor) -> Tensor:
        """
        Args:
            Q, K, V: (B, L, D)
            mask: (B, L, L) - True for positions to mask
            edge_embs: (B, L, L, edge_hidden_size) - pairwise relation embeddings
        """
        B, L = Q.size(0), Q.size(1)
        original_dtype = Q.dtype

        # QKV projections → (B, H, L, d_k)
        q_s = self.W_Q(Q).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)
        k_s = self.W_K(K).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)
        v_s = self.W_V(V).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)

        # Edge projections
        k_s_edge = self.W_K_edge(edge_embs)  # (B, L, L, E)
        edge_bias = self.W_edge(edge_embs).view(B, 1, L, L)  # (B, 1, L, L) — broadcast over heads
        edge_bias = edge_bias * (2 ** -0.5)  # Original scaling from paper

        # [Mod 3] Softcap on edge bias scalar — prevents unbounded growth
        cap = self.edge_softcap
        edge_bias = cap * torch.tanh(edge_bias / cap)

        # Attention mask: (B, L, L) → (B, H, L, L)
        if mask is not None:
            attn_mask = mask.unsqueeze(1).expand(-1, self.n_heads, -1, -1)

        # Biased attention scores (Eq. 7)
        # Original uses 1/sqrt(2*d_k) scaling to account for edge_bias addition
        scores = torch.matmul(q_s.float(), k_s.float().transpose(-1, -2)) * ((2 * self.d_k) ** -0.5)
        scores = scores + edge_bias.float()
        if mask is not None:
            scores.masked_fill_(attn_mask, -1e9)
        attn = self.dropout(F.softmax(scores, dim=-1))

        # Two-stream aggregation (Eq. 8)
        # Stream 1: standard value context
        context = torch.matmul(attn, v_s.float())
        # Stream 2: edge context — weighted sum of edge embeddings
        edge_context = torch.einsum('bhnm,bnmd->bhnd', attn, k_s_edge.float())  # (B, H, L, E)

        # Reshape and combine — stay in FP32 to prevent overflow on FP16 cast
        context = context.transpose(1, 2).contiguous().view(B, -1, self.n_heads * self.d_k)
        edge_context = edge_context.transpose(1, 2).contiguous().view(B, -1, self.n_heads * self.d_edge)

        with torch.amp.autocast('cuda', enabled=False):
            edge_context = self.W_edge_output(edge_context)  # (B, L, D) in FP32
            combined = torch.cat([context, edge_context], dim=-1)  # (B, L, 2D) in FP32
            return self.W_output(combined)  # (B, L, D) in FP32


class EdgeTransformerBlock(nn.Module):
    """
    Transformer Block with Edge Embeddings (Entity-Level Attention).
    Matches original EdgeTransformerBlock from transformer_rel.py:
    - Pre-norm on both token embeddings and edge embeddings
    - Edge-biased self-attention with edge context aggregation
    """

    def __init__(self, d_model: int, d_ff: int, n_heads: int,
                 edge_hidden_size: int, dropout: float = 0.1, edge_softcap: float = 5.0):
        super().__init__()
        self.self_attn = MultiHeadEdgeAttention(
            d_model, n_heads, edge_hidden_size, dropout, edge_softcap
        )
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm_attn = nn.LayerNorm(d_model)
        self.norm_edge = nn.LayerNorm(edge_hidden_size)  # Restored from original
        self.norm_ffn = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, edge_embs: Tensor, mask: Tensor = None) -> Tensor:
        norm_x = self.norm_attn(x)
        norm_edge_embs = self.norm_edge(edge_embs)  # LayerNorm on edge embeddings
        x = x + self.dropout(self.self_attn(norm_x, norm_x, norm_x, mask, norm_edge_embs))
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

        self.visit_pos_encoding = nn.Embedding(max_visits, d_model)

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, cls_tokens: Tensor, visit_indices: Tensor, visit_mask: Tensor) -> Tensor:
        batch_size, n_visits, _ = cls_tokens.size()
        original_dtype = cls_tokens.dtype

        pos_enc = self.visit_pos_encoding(visit_indices)
        x = cls_tokens + pos_enc
        x = self.norm(x)

        q = self.W_Q(x).view(batch_size, n_visits, self.n_heads, self.d_k).transpose(1, 2)
        k = self.W_K(x).view(batch_size, n_visits, self.n_heads, self.d_k).transpose(1, 2)
        v = self.W_V(cls_tokens).view(batch_size, n_visits, self.n_heads, self.d_k).transpose(1, 2)

        # FP32 for stability
        scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) / math.sqrt(self.d_k)

        if visit_mask is not None:
            attn_mask = ~visit_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(attn_mask, -1e4)

        attn = self.dropout(F.softmax(scores, dim=-1))
        context = torch.matmul(attn, v.float())

        context = context.to(original_dtype)
        context = context.transpose(1, 2).contiguous().view(batch_size, n_visits, -1)
        out = self.W_O(context) + cls_tokens
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
        original_dtype = x.dtype
        norm_x = self.norm_attn(x)

        q = self.W_Q(norm_x).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.W_K(norm_x).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.W_V(norm_x).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1), -1e4)

        attn = self.dropout(F.softmax(scores, dim=-1))
        context = torch.matmul(attn, v.float())

        context = context.to(original_dtype)
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
    2. Edge Module: Generate pairwise relation embeddings (B, L, L, E)
       — computed once, shared across all layers (same as original HiEdgeTransformer)
    3. N x (Entity-Level Attention + Encounter-Level Attention)
       - Entity-Level: EdgeTransformerBlock with edge embeddings (Eq. 7-8)
       - Encounter-Level: Cross-visit attention on first tokens (Eq. 9)
    4. Final LayerNorm + LM Head
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

        # Edge module: low-rank type transforms → (B, L, L, E) edge embeddings
        self.edge_module = EdgeModule(
            config.d_model, config.n_token_types,
            config.edge_hidden_size, config.edge_rank
        )

        # Entity-level attention blocks (within visit)
        self.entity_blocks = nn.ModuleList([
            EdgeTransformerBlock(
                config.d_model, config.d_ff, config.n_heads,
                config.edge_hidden_size, config.dropout, config.edge_softcap
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

        if token_types is None:
            token_types = torch.zeros_like(input_ids)
        if visit_ids is None:
            visit_ids = torch.zeros_like(input_ids)

        if time_offsets is not None:
            time_offsets = time_offsets.float().clamp(-1e6, 1e6)

        # Token embeddings
        h = self.embeddings(input_ids)

        # Add time encoding
        if self.t2v is not None and time_offsets is not None:
            time_emb = self.t2v(time_offsets)
            time_emb = self.time_proj(time_emb)
            h = h + time_emb

        h = self.emb_dropout(h)

        # Compute edge embeddings once, shared across all layers (same as original)
        # edge_embs: (B, L, L, edge_hidden_size)
        edge_embs = self.edge_module(h, token_types)

        # Create attention mask: True for positions to mask
        pad_mask = ~attention_mask.bool()
        pair_pad_mask = pad_mask.unsqueeze(1).expand(-1, seq_len, -1)

        # Multi-level attention: Entity + Encounter
        use_ckpt = self.config.use_gradient_checkpointing and self.training
        for i in range(self.config.n_blocks):
            # Entity-level attention with edge embeddings (Eq. 7-8)
            if use_ckpt:
                h = grad_checkpoint(
                    self.entity_blocks[i], h, edge_embs, pair_pad_mask,
                    use_reentrant=False
                )
            else:
                h = self.entity_blocks[i](h, edge_embs, pair_pad_mask)

            # Encounter-level attention (across visits on first tokens)
            if visit_ids is not None and not self.config.disable_encounter_attention:
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
        Apply encounter-level attention on first tokens of each visit.

        Vectorized implementation: detects visit boundaries via shifted comparison
        instead of per-visit Python loops.
        """
        batch_size, seq_len, d_model = h.size()
        device = h.device
        dtype = h.dtype
        max_visits = self.config.max_visits

        cls_tokens = torch.zeros(batch_size, max_visits, d_model, device=device, dtype=dtype)
        visit_mask = torch.zeros(batch_size, max_visits, dtype=torch.bool, device=device)
        cls_positions = torch.zeros(batch_size, max_visits, dtype=torch.long, device=device)
        visit_indices = torch.arange(max_visits, device=device).unsqueeze(0).expand(batch_size, -1)

        # Detect visit boundaries: where visit_id changes
        valid = attention_mask.bool()
        shifted_visits = torch.cat([
            torch.full((batch_size, 1), -1, device=device, dtype=visit_ids.dtype),
            visit_ids[:, :-1]
        ], dim=1)
        is_boundary = (visit_ids != shifted_visits) & valid  # (B, L)

        # Gather first token of each visit (loop over batch only, no inner visit loop)
        for b in range(batch_size):
            positions = is_boundary[b].nonzero(as_tuple=True)[0]
            n = min(len(positions), max_visits)
            if n > 0:
                positions = positions[:n]
                cls_tokens[b, :n] = h[b, positions]
                visit_mask[b, :n] = True
                cls_positions[b, :n] = positions

        # Apply encounter-level attention
        updated_cls = self.encounter_blocks[layer_idx](cls_tokens, visit_indices, visit_mask)

        # Scatter back (vectorized)
        h_new = h.clone()
        if visit_mask.any():
            batch_idx = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, max_visits)
            h_new[batch_idx[visit_mask], cls_positions[visit_mask]] = updated_cls[visit_mask]

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
        is_multilabel: bool = False,
    ):
        super().__init__()
        self.config = config
        self.num_classes = num_classes
        self.is_multilabel = is_multilabel

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
        hidden_states = self.encoder.encode(
            input_ids, attention_mask, token_types, visit_ids, time_offsets
        )

        # Find the last [CLS] token position for each sample (token_type == 1 is [CLS])
        cls_mask = (token_types == 1) if token_types is not None else (input_ids == 2)
        cls_positions = cls_mask.long().cumsum(dim=1) * cls_mask.long()
        last_cls_idx = cls_positions.argmax(dim=1)

        batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
        cls_hidden = hidden_states[batch_indices, last_cls_idx, :]
        cls_hidden = self.dropout(cls_hidden)
        logits = self.classifier(cls_hidden)

        output = {"logits": logits, "hidden_states": hidden_states}

        if labels is not None:
            if self.is_multilabel:
                loss = F.binary_cross_entropy_with_logits(logits, labels.float())
            elif self.num_classes == 1:
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

        cls_mask = (token_types == 1) if token_types is not None else (input_ids == 2)
        cls_positions = cls_mask.long().cumsum(dim=1) * cls_mask.long()
        last_cls_idx = cls_positions.argmax(dim=1)

        batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
        cls_hidden = hidden_states[batch_indices, last_cls_idx, :]
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
    max_seq_len: int = 2048,
    dropout: float = 0.1,
    n_token_types: int = 8,
    edge_hidden_size: int = 64,
    max_visits: int = 50,
    use_time_encoding: bool = True,
    edge_rank: int = 16,
    edge_softcap: float = 5.0,
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
        edge_rank=edge_rank,
        edge_softcap=edge_softcap,
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