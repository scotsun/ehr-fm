"""
GT-BEHRT: Graph Transformer BERT for Electronic Health Records

A hybrid model that combines:
1. Graph Transformer: Processes medical codes within each visit as a fully-connected graph
2. BERT Encoder: Processes the sequence of visits temporally

Reference: "Graph Transformers on EHRs: Better Representation Improves Downstream Performance"
           Poulain & Beheshti, ICLR 2024

Key components:
- TransformerConv: Graph attention layer with edge features
- GraphTransformer: Stacks TransformerConv layers for visit-level embeddings
- Multi-stream embeddings: graph, visit type, position, age, day-of-year
- BERT encoder for temporal relationships

This implementation adapts GT-BEHRT to work with our parquet data format.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass

# Try to import PyTorch Geometric, with fallback
try:
    import torch_geometric
    from torch_geometric.nn import MessagePassing
    from torch_geometric.utils import softmax as pyg_softmax
    from torch_geometric.data import Data, Batch
    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    MessagePassing = nn.Module  # Fallback


@dataclass
class GTBEHRTConfig:
    """Configuration for GT-BEHRT model."""
    # Vocabulary
    vocab_size: int = 15000
    pad_token_id: int = 0
    cls_token_id: int = 2

    # Model dimensions
    hidden_size: int = 540  # 5 * 108 for 5 embedding streams
    d_stream: int = 108  # hidden_size // 5

    # Graph Transformer
    n_graph_layers: int = 3
    n_graph_heads: int = 2
    n_edge_types: int = 10  # Different edge types for code relationships
    graph_dropout: float = 0.0  # Disabled by default

    # BERT Transformer
    n_bert_layers: int = 6
    n_bert_heads: int = 12
    intermediate_size: int = 512
    bert_dropout: float = 0.0  # Disabled by default
    attention_dropout: float = 0.0  # Disabled by default

    # Sequence limits
    max_visits: int = 50  # Maximum number of visits
    max_codes_per_visit: int = 100  # Maximum codes per visit

    # Additional embeddings
    n_visit_types: int = 11  # Number of visit types
    max_age: int = 103  # Maximum age vocabulary
    max_day_of_year: int = 367  # 1-366 + padding
    max_delta: int = 144  # Time delta vocabulary
    max_los: int = 1192  # Length of stay vocabulary

    def __post_init__(self):
        # Ensure hidden_size is 5 * d_stream
        assert self.hidden_size == 5 * self.d_stream, \
            f"hidden_size ({self.hidden_size}) must be 5 * d_stream ({self.d_stream})"


class TransformerConv(nn.Module):
    """
    Graph Transformer Convolution Layer.

    Implements graph attention with edge features following:
    A_ij = (Q_i)^T (K_j + E_ij) / sqrt(d)
    Attn_i = sum_j softmax(A_ij) * (V_j + E_ij)

    This is a simplified version that doesn't require PyTorch Geometric.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 2,
        edge_dim: Optional[int] = None,
        dropout: float = 0.0,
        concat: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.dropout = dropout
        self.concat = concat
        self.edge_dim = edge_dim

        # Linear projections
        self.lin_query = nn.Linear(in_channels, heads * out_channels)
        self.lin_key = nn.Linear(in_channels, heads * out_channels)
        self.lin_value = nn.Linear(in_channels, heads * out_channels)

        if edge_dim is not None:
            self.lin_edge = nn.Linear(edge_dim, heads * out_channels, bias=False)
        else:
            self.lin_edge = None

        # Output projection
        if concat:
            self.proj = nn.Linear(heads * out_channels, out_channels)

        # LayerNorm and FFN (Pre-LN architecture)
        self.layernorm1 = nn.LayerNorm(in_channels)
        self.layernorm2 = nn.LayerNorm(out_channels)
        self.ffn = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_channels, out_channels),
        )

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.lin_query.weight)
        nn.init.xavier_uniform_(self.lin_key.weight)
        nn.init.xavier_uniform_(self.lin_value.weight)
        if self.lin_edge is not None:
            nn.init.xavier_uniform_(self.lin_edge.weight)

    def forward(
        self,
        x: Tensor,  # (n_nodes, in_channels)
        edge_index: Tensor,  # (2, n_edges)
        edge_attr: Optional[Tensor] = None,  # (n_edges, edge_dim)
        batch: Optional[Tensor] = None,  # (n_nodes,) for batching
    ) -> Tensor:
        """
        Args:
            x: Node features (n_nodes, in_channels)
            edge_index: Edge indices (2, n_edges) - [source, target]
            edge_attr: Edge features (n_edges, edge_dim)
            batch: Batch assignment for nodes (n_nodes,)

        Returns:
            Updated node features (n_nodes, out_channels)
        """
        n_nodes = x.size(0)
        H, C = self.heads, self.out_channels

        # Pre-LN
        residual = x
        x = self.layernorm1(x)

        # Project to Q, K, V
        query = self.lin_query(x).view(n_nodes, H, C)  # (n_nodes, H, C)
        key = self.lin_key(x).view(n_nodes, H, C)
        value = self.lin_value(x).view(n_nodes, H, C)

        # Get source and target indices
        source, target = edge_index  # source -> target edges
        n_edges = source.size(0)

        if n_edges == 0:
            # No edges, just return projected input
            if self.concat:
                out = self.proj(query.view(n_nodes, H * C))
            else:
                out = query.mean(dim=1)
            return out + residual[:, :out.size(-1)] if residual.size(-1) == out.size(-1) else out

        # Get Q for targets, K and V for sources
        query_i = query[target]  # (n_edges, H, C)
        key_j = key[source]  # (n_edges, H, C)
        value_j = value[source]  # (n_edges, H, C)

        # Add edge features to K and V
        if self.lin_edge is not None and edge_attr is not None:
            edge_embed = self.lin_edge(edge_attr).view(n_edges, H, C)
            key_j = key_j + edge_embed
            value_j = value_j + edge_embed

        # Compute attention scores
        alpha = (query_i * key_j).sum(dim=-1) / math.sqrt(C)  # (n_edges, H)

        # Softmax per target node
        alpha = self._scatter_softmax(alpha, target, n_nodes)  # (n_edges, H)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        # Weighted sum of values
        out = alpha.unsqueeze(-1) * value_j  # (n_edges, H, C)

        # Scatter add to target nodes
        out = self._scatter_add(out, target, n_nodes)  # (n_nodes, H, C)

        # Project output
        if self.concat:
            out = self.proj(out.view(n_nodes, H * C))  # (n_nodes, out_channels)
        else:
            out = out.mean(dim=1)  # (n_nodes, out_channels)

        # Residual connection (handle dimension mismatch)
        if residual.size(-1) == out.size(-1):
            out = out + residual

        # FFN with Pre-LN
        residual = out
        out = self.layernorm2(out)
        out = self.ffn(out)
        out = F.dropout(out, p=self.dropout, training=self.training)
        out = out + residual

        return out

    def _scatter_softmax(self, src: Tensor, index: Tensor, num_nodes: int) -> Tensor:
        """Compute softmax over edges grouped by target node."""
        # src: (n_edges, H), index: (n_edges,)
        src_max = self._scatter_max(src, index, num_nodes)  # (num_nodes, H)
        src = src - src_max[index]  # Subtract max for numerical stability
        src_exp = src.exp()
        src_sum = self._scatter_add(src_exp, index, num_nodes)  # (num_nodes, H)
        return src_exp / (src_sum[index] + 1e-8)

    def _scatter_max(self, src: Tensor, index: Tensor, num_nodes: int) -> Tensor:
        """Scatter max operation."""
        out = torch.full((num_nodes, src.size(-1)), float('-inf'), device=src.device, dtype=src.dtype)
        out.scatter_reduce_(0, index.unsqueeze(-1).expand_as(src), src, reduce='amax')
        return out

    def _scatter_add(self, src: Tensor, index: Tensor, num_nodes: int) -> Tensor:
        """Scatter add operation."""
        if src.dim() == 2:
            out = torch.zeros(num_nodes, src.size(-1), device=src.device, dtype=src.dtype)
            out.scatter_add_(0, index.unsqueeze(-1).expand_as(src), src)
        else:  # dim == 3
            out = torch.zeros(num_nodes, src.size(1), src.size(2), device=src.device, dtype=src.dtype)
            index_expanded = index.view(-1, 1, 1).expand_as(src)
            out.scatter_add_(0, index_expanded, src)
        return out


class GraphTransformer(nn.Module):
    """
    Graph Transformer for extracting visit-level embeddings.

    Processes medical codes in a visit as a fully-connected graph
    and uses a virtual <VST> node for graph-level readout.
    """

    def __init__(self, config: GTBEHRTConfig):
        super().__init__()
        self.config = config
        d = config.d_stream

        # Node embeddings (medical codes)
        self.node_embed = nn.Embedding(config.vocab_size, d, padding_idx=config.pad_token_id)

        # Edge type embeddings
        self.edge_embed = nn.Embedding(config.n_edge_types, d)

        # Virtual <VST> node embedding (learnable)
        self.vst_embed = nn.Parameter(torch.randn(1, d))

        # Graph Transformer layers
        self.layers = nn.ModuleList([
            TransformerConv(
                in_channels=d,
                out_channels=d,
                heads=config.n_graph_heads,
                edge_dim=d,
                dropout=config.graph_dropout,
                concat=(i < config.n_graph_layers - 1),  # Last layer averages heads
            )
            for i in range(config.n_graph_layers)
        ])

        # Activation between layers
        self.activations = nn.ModuleList([
            nn.GELU() for _ in range(config.n_graph_layers - 1)
        ])

    def forward(
        self,
        node_ids: Tensor,  # (n_total_codes,) all codes across visits
        edge_index: Tensor,  # (2, n_edges) fully-connected within visits
        edge_type: Tensor,  # (n_edges,) edge types
        vst_indices: Tensor,  # (n_visits,) indices of <VST> nodes
        batch: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            node_ids: Code token IDs for all nodes including <VST>
            edge_index: Graph edges (fully connected within visits)
            edge_type: Edge type IDs for edge embeddings
            vst_indices: Indices of <VST> nodes to extract as visit embeddings
            batch: Batch assignment (optional)

        Returns:
            visit_embeddings: (n_visits, d_stream)
        """
        # Embed nodes
        h = self.node_embed(node_ids)  # (n_nodes, d)

        # Replace <VST> node embeddings with learnable embedding
        h[vst_indices] = self.vst_embed.expand(len(vst_indices), -1)

        # Embed edges
        edge_attr = self.edge_embed(edge_type)  # (n_edges, d)

        # Apply Graph Transformer layers
        for i, layer in enumerate(self.layers):
            h = layer(h, edge_index, edge_attr, batch)
            if i < len(self.activations):
                h = self.activations[i](h)

        # Extract <VST> node representations as visit embeddings
        visit_embeddings = h[vst_indices]  # (n_visits, d)

        return visit_embeddings

    def forward_all_nodes(
        self,
        node_ids: Tensor,
        edge_index: Tensor,
        edge_type: Tensor,
        vst_indices: Tensor,
        batch: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass that returns ALL node embeddings (for NAM pretraining).

        Args:
            Same as forward()

        Returns:
            all_node_embeddings: (n_nodes, d_stream)
        """
        # Embed nodes
        h = self.node_embed(node_ids)  # (n_nodes, d)

        # Replace <VST> node embeddings with learnable embedding
        h[vst_indices] = self.vst_embed.expand(len(vst_indices), -1)

        # Embed edges
        edge_attr = self.edge_embed(edge_type)  # (n_edges, d)

        # Apply Graph Transformer layers
        for i, layer in enumerate(self.layers):
            h = layer(h, edge_index, edge_attr, batch)
            if i < len(self.activations):
                h = self.activations[i](h)

        return h  # All node embeddings


class SinusoidalPositionalEmbedding(nn.Module):
    """Sinusoidal positional embedding (non-learnable)."""

    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()

        # Create position encodings
        pe = torch.zeros(vocab_size, d_model)
        position = torch.arange(0, vocab_size, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model > 1:
            pe[:, 1::2] = torch.cos(position * div_term[:d_model//2])

        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (batch, seq_len) position indices
        Returns:
            (batch, seq_len, d_model) positional embeddings
        """
        return self.pe[x]


class GTBEHRTEmbeddings(nn.Module):
    """
    GT-BEHRT Embedding Layer.

    Combines multiple embedding streams:
    1. Graph embedding (from GraphTransformer) - d_stream
    2. Visit type embedding - d_stream
    3. Position embedding (sinusoidal) - d_stream
    4. Age embedding (sinusoidal) - d_stream
    5. Day-of-year embedding (sinusoidal) - d_stream

    Total: 5 * d_stream = hidden_size
    """

    def __init__(self, config: GTBEHRTConfig):
        super().__init__()
        self.config = config
        d = config.d_stream

        # Graph Transformer for code-level -> visit-level embeddings
        self.graph_transformer = GraphTransformer(config)

        # Additional embeddings (all d_stream dimensional)
        self.visit_type_embed = nn.Embedding(config.n_visit_types, d, padding_idx=0)
        self.position_embed = SinusoidalPositionalEmbedding(config.max_visits, d)
        self.age_embed = SinusoidalPositionalEmbedding(config.max_age, d)
        self.day_embed = SinusoidalPositionalEmbedding(config.max_day_of_year, d)

        # CLS token embedding (learnable)
        self.cls_embed = nn.Parameter(torch.randn(1, 1, config.hidden_size))

        # Projection layers after concatenation
        self.proj = nn.Sequential(
            nn.LayerNorm(config.hidden_size),
            nn.Dropout(config.bert_dropout),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
        )
        self.final_norm = nn.LayerNorm(config.hidden_size)

    def forward(
        self,
        graph_data: Dict[str, Tensor],
        visit_types: Tensor,  # (batch, n_visits)
        positions: Tensor,  # (batch, n_visits)
        ages: Tensor,  # (batch, n_visits)
        days: Tensor,  # (batch, n_visits)
        attention_mask: Tensor,  # (batch, n_visits)
    ) -> Tensor:
        """
        Args:
            graph_data: Dictionary containing:
                - node_ids: (n_total_codes,)
                - edge_index: (2, n_edges)
                - edge_type: (n_edges,)
                - vst_indices: (total_visits,)
                - batch_visit_counts: (batch_size,) number of visits per sample
            visit_types: (batch, n_visits) visit type IDs
            positions: (batch, n_visits) visit positions (0, 1, 2, ...)
            ages: (batch, n_visits) patient age at visit
            days: (batch, n_visits) day of year (1-366)
            attention_mask: (batch, n_visits) valid visit mask

        Returns:
            embeddings: (batch, n_visits + 1, hidden_size) with CLS token prepended
        """
        batch_size = visit_types.size(0)
        n_visits = visit_types.size(1)

        # Get graph embeddings for all visits
        graph_embeds = self.graph_transformer(
            graph_data['node_ids'],
            graph_data['edge_index'],
            graph_data['edge_type'],
            graph_data['vst_indices'],
        )  # (total_visits, d_stream)

        # Reshape to (batch, n_visits, d_stream)
        # Use batch_visit_counts to split
        batch_visit_counts = graph_data['batch_visit_counts']
        graph_embeds_list = torch.split(graph_embeds, batch_visit_counts.tolist())

        # Pad to same length
        graph_embeds_padded = torch.zeros(
            batch_size, n_visits, self.config.d_stream,
            device=graph_embeds.device, dtype=graph_embeds.dtype
        )
        for i, emb in enumerate(graph_embeds_list):
            graph_embeds_padded[i, :len(emb)] = emb

        # Get other embeddings
        type_embeds = self.visit_type_embed(visit_types)  # (batch, n_visits, d)
        pos_embeds = self.position_embed(positions)  # (batch, n_visits, d)
        age_embeds = self.age_embed(ages)  # (batch, n_visits, d)
        day_embeds = self.day_embed(days)  # (batch, n_visits, d)

        # Concatenate all embeddings: [graph, type, pos, age, day]
        embeddings = torch.cat([
            graph_embeds_padded,
            type_embeds,
            pos_embeds,
            age_embeds,
            day_embeds,
        ], dim=-1)  # (batch, n_visits, hidden_size)

        # Add CLS token
        cls_tokens = self.cls_embed.expand(batch_size, -1, -1)  # (batch, 1, hidden_size)
        embeddings = torch.cat([cls_tokens, embeddings], dim=1)  # (batch, n_visits + 1, hidden_size)

        # Project and normalize
        embeddings = self.proj(embeddings)
        embeddings = self.final_norm(embeddings)

        return embeddings


class BertEncoder(nn.Module):
    """Standard BERT Transformer Encoder."""

    def __init__(self, config: GTBEHRTConfig):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.n_bert_heads,
            dim_feedforward=config.intermediate_size,
            dropout=config.bert_dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,  # Pre-LN
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.n_bert_layers,
        )

    def forward(
        self,
        hidden_states: Tensor,  # (batch, seq_len, hidden_size)
        attention_mask: Tensor,  # (batch, seq_len)
    ) -> Tensor:
        """
        Args:
            hidden_states: (batch, seq_len, hidden_size)
            attention_mask: (batch, seq_len) - 1 for valid, 0 for padding

        Returns:
            encoded: (batch, seq_len, hidden_size)
        """
        # Convert attention mask to transformer format (True = ignore)
        # PyTorch expects: True for positions to mask (ignore)
        src_key_padding_mask = ~attention_mask.bool()  # (batch, seq_len)

        return self.encoder(hidden_states, src_key_padding_mask=src_key_padding_mask)


class BertPooler(nn.Module):
    """BERT Pooler - extracts CLS token representation."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: Tensor) -> Tensor:
        """
        Args:
            hidden_states: (batch, seq_len, hidden_size)

        Returns:
            pooled: (batch, hidden_size) - from CLS token
        """
        cls_token = hidden_states[:, 0]  # (batch, hidden_size)
        return self.activation(self.dense(cls_token))


class GTBEHRT(nn.Module):
    """
    GT-BEHRT: Graph Transformer BERT for EHR.

    Combines:
    1. Graph Transformer for visit-level embeddings (code relationships)
    2. BERT encoder for temporal relationships between visits

    Pre-training objectives:
    - Node Attribute Masking (NAM): Mask codes, predict original
    - Missing Node Prediction (MNP): Remove node, predict removed code
    - Visit Type Prediction (VTP): Mask visit type, predict original
    """

    def __init__(self, config: GTBEHRTConfig):
        super().__init__()
        self.config = config

        # Embeddings
        self.embeddings = GTBEHRTEmbeddings(config)

        # BERT Encoder
        self.encoder = BertEncoder(config)

        # Pooler
        self.pooler = BertPooler(config.hidden_size)

        # Pre-training heads
        # NAM head (node-level, on graph output)
        self.nam_head = nn.Linear(config.d_stream, config.vocab_size)

        # MNP head (sequence-level)
        self.mnp_head = nn.Linear(config.hidden_size, config.vocab_size)

        # VTP head (sequence-level)
        self.vtp_head = nn.Linear(config.hidden_size, config.n_visit_types)

        # LM head for downstream
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()

    def forward(
        self,
        graph_data: Dict[str, Tensor],
        visit_types: Tensor,
        positions: Tensor,
        ages: Tensor,
        days: Tensor,
        attention_mask: Tensor,
        output_all_encoded_layers: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward pass.

        Args:
            graph_data: Graph data dictionary
            visit_types: (batch, n_visits)
            positions: (batch, n_visits)
            ages: (batch, n_visits)
            days: (batch, n_visits)
            attention_mask: (batch, n_visits) - 1 for valid, 0 for padding
            output_all_encoded_layers: Whether to output all layers

        Returns:
            sequence_output: (batch, n_visits + 1, hidden_size)
            pooled_output: (batch, hidden_size)
        """
        # Get embeddings with CLS token
        embeddings = self.embeddings(
            graph_data, visit_types, positions, ages, days, attention_mask
        )  # (batch, n_visits + 1, hidden_size)

        # Extend attention mask for CLS token
        cls_mask = torch.ones(attention_mask.size(0), 1, device=attention_mask.device)
        extended_mask = torch.cat([cls_mask, attention_mask.float()], dim=1)

        # BERT encoding
        sequence_output = self.encoder(embeddings, extended_mask)

        # Pooling
        pooled_output = self.pooler(sequence_output)

        return sequence_output, pooled_output

    def get_graph_embeddings(
        self,
        graph_data: Dict[str, Tensor],
    ) -> Tensor:
        """Get graph-level embeddings for NAM pre-training."""
        return self.embeddings.graph_transformer(
            graph_data['node_ids'],
            graph_data['edge_index'],
            graph_data['edge_type'],
            graph_data['vst_indices'],
        )


def remove_nodes_from_graph(
    node_ids: Tensor,
    edge_index: Tensor,
    edge_type: Tensor,
    vst_indices: Tensor,
    nodes_to_remove: Tensor,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """
    Remove nodes from graph and update edge indices.

    This is used for true MNP (Missing Node Prediction) where nodes are
    physically removed from the graph, not just masked.

    Args:
        node_ids: (n_nodes,) node token IDs
        edge_index: (2, n_edges) edge indices [src, dst]
        edge_type: (n_edges,) edge types
        vst_indices: (n_vst,) indices of VST nodes
        nodes_to_remove: (n_remove,) indices of nodes to remove

    Returns:
        new_node_ids: Updated node IDs
        new_edge_index: Updated edge indices
        new_edge_type: Updated edge types
        new_vst_indices: Updated VST indices
        old_to_new: Mapping from old indices to new indices (-1 for removed)
    """
    n_nodes = node_ids.size(0)
    device = node_ids.device

    # Create mask for nodes to keep
    keep_mask = torch.ones(n_nodes, dtype=torch.bool, device=device)
    keep_mask[nodes_to_remove] = False

    # Create mapping from old to new indices
    old_to_new = torch.full((n_nodes,), -1, dtype=torch.long, device=device)
    new_indices = torch.arange(keep_mask.sum(), device=device)
    old_to_new[keep_mask] = new_indices

    # Filter node IDs
    new_node_ids = node_ids[keep_mask]

    # Filter and remap edges
    src, dst = edge_index
    # Keep only edges where both endpoints are kept
    edge_keep_mask = keep_mask[src] & keep_mask[dst]

    if edge_keep_mask.any():
        new_src = old_to_new[src[edge_keep_mask]]
        new_dst = old_to_new[dst[edge_keep_mask]]
        new_edge_index = torch.stack([new_src, new_dst], dim=0)
        new_edge_type = edge_type[edge_keep_mask]
    else:
        new_edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
        new_edge_type = torch.zeros(0, dtype=torch.long, device=device)

    # Remap VST indices
    new_vst_indices = old_to_new[vst_indices]
    # VST indices should never be removed, but filter just in case
    new_vst_indices = new_vst_indices[new_vst_indices >= 0]

    return new_node_ids, new_edge_index, new_edge_type, new_vst_indices, old_to_new


class GTBEHRTForPretraining(nn.Module):
    """
    GT-BEHRT with pre-training heads.

    Two-step pre-training:
    1. NAM (Node Attribute Masking): Graph-level, mask codes and predict
    2. MNP + VTP: Sequence-level, missing node + visit type prediction

    MNP is implemented as true node removal (not MLM-style masking).
    """

    def __init__(self, config: GTBEHRTConfig):
        super().__init__()
        self.config = config
        self.gtbehrt = GTBEHRT(config)

        # Loss functions
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-100)

    def forward_nam(
        self,
        graph_data: Dict[str, Tensor],
        masked_indices: Tensor,  # Indices of masked nodes
        labels: Tensor,  # Original node IDs for masked positions
    ) -> Tuple[Tensor, Tensor]:
        """
        Node Attribute Masking forward pass (Step 1 pre-training).

        Only trains the Graph Transformer.
        Nodes are MASKED (embedding replaced), not removed from graph.
        """
        # Get ALL node embeddings after graph transformer (not just VST)
        all_node_embeds = self.gtbehrt.embeddings.graph_transformer.forward_all_nodes(
            graph_data['node_ids'],
            graph_data['edge_index'],
            graph_data['edge_type'],
            graph_data['vst_indices'],
        )

        # Get embeddings for masked nodes
        # Note: masked_indices should point to non-VST nodes
        masked_embeds = all_node_embeds[masked_indices]  # (n_masked, d_stream)

        # Predict original codes
        logits = self.gtbehrt.nam_head(masked_embeds)  # (n_masked, vocab_size)

        loss = self.ce_loss(logits, labels)
        return loss, logits

    def forward_mnp(
        self,
        graph_data: Dict[str, Tensor],
        removed_node_indices: Tensor,  # (n_visits,) indices of removed nodes (one per visit)
        removed_node_labels: Tensor,  # (n_visits,) original token IDs of removed nodes
        visit_to_vst: Tensor,  # (n_visits,) mapping: visit index -> VST node index
    ) -> Tuple[Tensor, Tensor]:
        """
        Missing Node Prediction forward pass (true MNP).

        Nodes are PHYSICALLY REMOVED from the graph, then we predict
        the removed node using the corresponding VST embedding.

        Args:
            graph_data: Graph data dict with node_ids, edge_index, etc.
            removed_node_indices: Indices of nodes that were removed
            removed_node_labels: Original token IDs of removed nodes
            visit_to_vst: Maps each removed node to its visit's VST index

        Returns:
            loss: MNP loss
            logits: Prediction logits (n_visits, vocab_size)
        """
        # Remove nodes from graph
        new_node_ids, new_edge_index, new_edge_type, new_vst_indices, old_to_new = \
            remove_nodes_from_graph(
                graph_data['node_ids'],
                graph_data['edge_index'],
                graph_data['edge_type'],
                graph_data['vst_indices'],
                removed_node_indices,
            )

        # Run Graph Transformer on modified graph
        all_node_embeds = self.gtbehrt.embeddings.graph_transformer.forward_all_nodes(
            new_node_ids,
            new_edge_index,
            new_edge_type,
            new_vst_indices,
        )

        # Get VST embeddings for prediction
        # Map original VST indices to new indices
        new_vst_for_removed = old_to_new[visit_to_vst]
        vst_embeds = all_node_embeds[new_vst_for_removed]  # (n_visits, d_stream)

        # Predict removed nodes
        logits = self.gtbehrt.nam_head(vst_embeds)  # (n_visits, vocab_size)

        loss = self.ce_loss(logits, removed_node_labels)
        return loss, logits

    def forward_mnp_vtp(
        self,
        graph_data: Dict[str, Tensor],
        visit_types: Tensor,
        positions: Tensor,
        ages: Tensor,
        days: Tensor,
        attention_mask: Tensor,
        mnp_labels: Optional[Tensor] = None,  # (batch,) removed node IDs
        vtp_labels: Optional[Tensor] = None,  # (batch, n_visits) masked visit types
        vtp_mask: Optional[Tensor] = None,  # (batch, n_visits) which types are masked
    ) -> Dict[str, Tensor]:
        """
        MNP + VTP forward pass (Step 2 pre-training).

        Trains the full model.
        """
        # Full forward pass
        sequence_output, pooled_output = self.gtbehrt(
            graph_data, visit_types, positions, ages, days, attention_mask
        )

        outputs = {'sequence_output': sequence_output, 'pooled_output': pooled_output}

        # MNP loss: predict removed node using CLS token
        if mnp_labels is not None:
            mnp_logits = self.gtbehrt.mnp_head(pooled_output)  # (batch, vocab_size)
            outputs['mnp_loss'] = self.ce_loss(mnp_logits, mnp_labels)
            outputs['mnp_logits'] = mnp_logits

        # VTP loss: predict masked visit types
        if vtp_labels is not None and vtp_mask is not None:
            # sequence_output[:, 1:] corresponds to visits (skip CLS)
            visit_outputs = sequence_output[:, 1:]  # (batch, n_visits, hidden)
            vtp_logits = self.gtbehrt.vtp_head(visit_outputs)  # (batch, n_visits, n_types)

            # Flatten and compute loss only on masked positions
            vtp_logits_flat = vtp_logits[vtp_mask]  # (n_masked, n_types)
            vtp_labels_flat = vtp_labels[vtp_mask]  # (n_masked,)

            if vtp_logits_flat.numel() > 0:
                outputs['vtp_loss'] = self.ce_loss(vtp_logits_flat, vtp_labels_flat)
            outputs['vtp_logits'] = vtp_logits

        return outputs


class GTBEHRTForSequenceClassification(nn.Module):
    """
    GT-BEHRT for downstream sequence classification tasks.

    Uses the pooled CLS representation for classification.
    """

    def __init__(self, config: GTBEHRTConfig, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.config = config
        self.num_classes = num_classes

        self.gtbehrt = GTBEHRT(config)

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(config.hidden_size, num_classes),
        )

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(
        self,
        graph_data: Dict[str, Tensor],
        visit_types: Tensor,
        positions: Tensor,
        ages: Tensor,
        days: Tensor,
        attention_mask: Tensor,
        labels: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        Args:
            ... (same as GTBEHRT.forward)
            labels: (batch,) classification labels

        Returns:
            dict with 'logits' and optionally 'loss'
        """
        _, pooled_output = self.gtbehrt(
            graph_data, visit_types, positions, ages, days, attention_mask
        )

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        output = {'logits': logits}

        if labels is not None:
            output['loss'] = self.loss_fn(logits, labels)

        return output

    def load_pretrained(self, checkpoint_path: str, strict: bool = False):
        """Load pre-trained GT-BEHRT weights."""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        # Handle different prefixes
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('gtbehrt.'):
                new_state_dict[k] = v
            else:
                new_state_dict[f'gtbehrt.{k}'] = v

        missing, unexpected = self.load_state_dict(new_state_dict, strict=False)

        # Filter out classifier keys from missing (expected)
        missing = [k for k in missing if not k.startswith('classifier.')]

        if missing and strict:
            raise RuntimeError(f"Missing keys: {missing}")

        return missing, unexpected


def create_gtbehrt_config(
    vocab_size: int = 15000,
    hidden_size: int = 540,
    n_graph_layers: int = 3,
    n_bert_layers: int = 6,
    n_heads: int = 12,
    **kwargs,
) -> GTBEHRTConfig:
    """Create GT-BEHRT configuration."""
    d_stream = hidden_size // 5
    return GTBEHRTConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        d_stream=d_stream,
        n_graph_layers=n_graph_layers,
        n_bert_layers=n_bert_layers,
        n_bert_heads=n_heads,
        **kwargs,
    )
