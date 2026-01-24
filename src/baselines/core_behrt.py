"""
BEHRT Baseline Model

A flat Transformer baseline (no hierarchical structure).
All visits are concatenated into a single sequence.

Reference: BEHRT: Transformer for Electronic Health Records (Li et al., 2020)

Key differences from HAT:
- HAT: (batch, max_seg, max_seq_len) hierarchical input
- BEHRT: (batch, seq_len) flat input, all tokens concatenated

This implementation aligns with FMBert (ehr-fm/ehr-fm/src/models/bert.py):
- Uses T2V (Time2Vec) for time encoding
- Uses set_pos (based on [CLS] token cumsum) for RoPE
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import cumsum
from transformers.modeling_utils import PreTrainedModel
from transformers import PretrainedConfig

from src.layers import TransformerBlock, T2V


class BEHRTConfig(PretrainedConfig):
    """Configuration for BEHRT baseline model."""
    model_type = "behrt"

    def __init__(
        self,
        vocab_size: int = 15000,
        d_model: int = 768,
        n_blocks: int = 6,
        n_heads: int = 12,
        d_ff: int = 2048,
        max_seq_len: int = 2048,
        dropout: float = 0.0,
        norm_type: str = "layer",
        ffn_type: str = "swiglu",
        pad_token_id: int = 0,
        cls_token_id: int = 2,  # [CLS] token id for set_pos calculation
        weight_tying: bool = False,
        attn_backend: str = "base",
        t2v_scale: float = 1.0,  # T2V scale factor
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
        self.norm_type = norm_type
        self.ffn_type = ffn_type
        self.cls_token_id = cls_token_id
        self.weight_tying = weight_tying
        self.attn_backend = attn_backend
        self.t2v_scale = t2v_scale


class BEHRT(PreTrainedModel):
    """
    BEHRT: Flat Transformer for EHR.

    Unlike HAT which processes segments hierarchically,
    BEHRT concatenates all tokens into a single long sequence.

    Aligned with FMBert implementation:
    - T2V for time encoding
    - set_pos (cumsum of [SEP] tokens) for RoPE position

    Input shape: (batch, seq_len)
    Output shape: (batch, seq_len, vocab_size), (batch, seq_len, d_model)
    """
    config_class = BEHRTConfig
    base_model_prefix = "behrt"

    def __init__(self, config: BEHRTConfig):
        super().__init__(config)
        self.config = config

        # Token embeddings
        self.embeddings = nn.Embedding(
            config.vocab_size, config.d_model, padding_idx=config.pad_token_id
        )

        # T2V time encoding (aligned with FMBert)
        self.t2v = T2V(config.d_model, config.t2v_scale)

        # Transformer blocks (with RoPE)
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=config.d_model,
                d_ff=config.d_ff,
                h=config.n_heads,
                with_rope=True,
                dropout=config.dropout,
                norm_type=config.norm_type,
                ffn_type=config.ffn_type,
                attn_backend=config.attn_backend,
            )
            for _ in range(config.n_blocks)
        ])

        # Final layer norm (Pre-Norm architecture)
        if config.norm_type == "layer":
            self.final_norm = nn.LayerNorm(config.d_model, bias=False)
        else:
            self.final_norm = nn.RMSNorm(config.d_model)

        # LM head for MLM pre-training
        self.lm_head = nn.Linear(config.d_model, config.vocab_size)
        if config.weight_tying:
            self.lm_head.weight = self.embeddings.weight

    def forward(self, input_ids, attention_mask, t):
        """
        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)
            t: (batch, seq_len) time values for T2V

        Returns:
            logits: (batch, seq_len, vocab_size)
            hidden_states: (batch, seq_len, d_model)
        """
        h = self.encode(input_ids, attention_mask, t)
        logits = self.lm_head(h)
        return logits, h

    def encode(self, input_ids, attention_mask, t):
        """
        Encode input tokens to hidden representations.

        Aligned with FMBert:
        - set_pos: cumsum of [CLS] tokens for RoPE position (encounter id)
        - T2V: time encoding added to embeddings

        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)
            t: (batch, seq_len) time values

        Returns:
            hidden_states: (batch, seq_len, d_model)
        """
        # CRITICAL: Ensure time stays in FP32 for numerical stability with AMP
        # Large time values lose precision in FP16
        t = t.float()

        # Compute set position based on [CLS] token (encounter id for RoPE)
        set_pos = attention_mask * cumsum(input_ids == self.config.cls_token_id, dim=1)

        # Embeddings + T2V time encoding
        h = self.embeddings(input_ids)
        h = h + self.t2v(t)

        # Transformer blocks with RoPE using set_pos
        for block in self.blocks:
            h = block(h, attention_mask, time=set_pos)

        h = self.final_norm(h)
        return h


class BEHRTForSequenceClassification(nn.Module):
    """
    BEHRT with classification head for downstream tasks.

    Uses [CLS] token (first token) for classification.
    """

    def __init__(
        self,
        config: BEHRTConfig,
        num_classes: int,
        dropout: float = 0.1,
        freeze_encoder: bool = False,
    ):
        super().__init__()
        self.config = config
        self.num_classes = num_classes

        # BEHRT encoder
        self.encoder = BEHRT(config)

        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(config.d_model, num_classes),
        )

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
        input_ids,
        attention_mask,
        t,
        labels=None,
    ):
        """
        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)
            t: (batch, seq_len) time values
            labels: (batch,) optional

        Returns:
            dict with 'logits' and optionally 'loss'
        """
        # Encode
        hidden_states = self.encoder.encode(input_ids, attention_mask, t)

        # Find the last [CLS] token position for each sample
        # Each visit starts with [CLS], we want the last visit's [CLS] for prediction
        cls_mask = (input_ids == self.config.cls_token_id)  # (batch, seq_len)
        # Get the last [CLS] position by finding the highest index with [CLS]
        # Use cumsum to count [CLS] tokens, then find max position
        cls_positions = cls_mask.long().cumsum(dim=1) * cls_mask.long()  # positions: 1, 2, 3, ...
        last_cls_idx = cls_positions.argmax(dim=1)  # index of last [CLS]

        # Gather last [CLS] hidden states
        batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
        cls_hidden = hidden_states[batch_indices, last_cls_idx, :]  # (batch, d_model)
        cls_hidden = self.dropout(cls_hidden)

        # Classify
        logits = self.classifier(cls_hidden)  # (batch, num_classes)

        output = {"logits": logits}

        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            output["loss"] = loss

        return output

    def load_pretrained(self, checkpoint_path: str, strict: bool = True):
        """Load pre-trained BEHRT weights."""
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


def create_behrt_model(
    vocab_size: int = 15000,
    d_model: int = 768,
    n_blocks: int = 6,
    n_heads: int = 12,
    d_ff: int = 2048,
    max_seq_len: int = 2048,
    dropout: float = 0.0,
    t2v_scale: float = 1.0,
    **kwargs,
) -> BEHRT:
    """Create a BEHRT model with given configuration."""
    config = BEHRTConfig(
        vocab_size=vocab_size,
        d_model=d_model,
        n_blocks=n_blocks,
        n_heads=n_heads,
        d_ff=d_ff,
        max_seq_len=max_seq_len,
        dropout=dropout,
        t2v_scale=t2v_scale,
        **kwargs,
    )
    return BEHRT(config)
