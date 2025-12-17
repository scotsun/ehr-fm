"""
BEHRT Baseline Model

A flat Transformer baseline (no hierarchical structure).
All visits are concatenated into a single sequence.

Reference: BEHRT: Transformer for Electronic Health Records (Li et al., 2020)

Key differences from HAT:
- HAT: (batch, max_seg, max_seq_len) hierarchical input
- BEHRT: (batch, seq_len) flat input, all tokens concatenated
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_utils import PreTrainedModel
from transformers import PretrainedConfig

from src.layers import TransformerBlock


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
        max_seq_len: int = 512,
        dropout: float = 0.0,
        norm_type: str = "layer",
        ffn_type: str = "swiglu",
        pad_token_id: int = 0,
        weight_tying: bool = False,
        attn_backend: str = "base",
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
        self.weight_tying = weight_tying
        self.attn_backend = attn_backend


class BEHRT(PreTrainedModel):
    """
    BEHRT: Flat Transformer for EHR.

    Unlike HAT which processes segments hierarchically,
    BEHRT concatenates all tokens into a single long sequence.

    Input shape: (batch, seq_len)
    Output shape: (batch, seq_len, vocab_size)
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

        # Transformer blocks (with RoPE)
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=config.d_model,
                d_ff=config.d_ff,
                h=config.n_heads,
                with_rope=True,  # Use RoPE for positional encoding
                dropout=config.dropout,
                norm_type=config.norm_type,
                ffn_type=config.ffn_type,
                attn_backend=config.attn_backend,
            )
            for _ in range(config.n_blocks)
        ])

        # Final layer norm (Pre-Norm architecture)
        if config.norm_type == "layer":
            self.final_norm = nn.LayerNorm(config.d_model)
        else:
            self.final_norm = nn.RMSNorm(config.d_model)

        # LM head for MLM pre-training
        self.lm_head = nn.Linear(config.d_model, config.vocab_size)
        if config.weight_tying:
            self.lm_head.weight = self.embeddings.weight

    def forward(self, input_ids, attention_mask, time=None):
        """
        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)
            time: (batch, seq_len) optional time values for RoPE

        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        h = self.encode(input_ids, attention_mask, time)
        logits = self.lm_head(h)
        return logits

    def encode(self, input_ids, attention_mask, time=None):
        """
        Encode input tokens to hidden representations.

        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)
            time: (batch, seq_len) optional

        Returns:
            hidden_states: (batch, seq_len, d_model)
        """
        h = self.embeddings(input_ids)

        for block in self.blocks:
            h = block(h, attention_mask, time)

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
        time=None,
        labels=None,
    ):
        """
        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)
            time: (batch, seq_len) optional
            labels: (batch,) optional

        Returns:
            dict with 'logits' and optionally 'loss'
        """
        # Encode
        hidden_states = self.encoder.encode(input_ids, attention_mask, time)

        # Use [CLS] token (first token) for classification
        cls_hidden = hidden_states[:, 0, :]  # (batch, d_model)
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
    max_seq_len: int = 512,
    dropout: float = 0.0,
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
        **kwargs,
    )
    return BEHRT(config)
