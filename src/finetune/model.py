"""
Fine-tuning Model for HAT downstream tasks.

Adds a classification head on top of pre-trained HAT encoder.
Uses the [CLS] token from the last valid segment for classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from src.fm import FMBase, FMConfig


class ClassificationHead(nn.Module):
    """Classification head with optional pooling and dropout."""

    def __init__(
        self,
        d_model: int,
        num_classes: int,
        dropout: float = 0.1,
        pooling: str = "last_cls",  # "last_cls", "mean_cls", "attention"
    ):
        super().__init__()
        self.pooling = pooling
        self.dropout = nn.Dropout(dropout)

        if pooling == "attention":
            self.attention = nn.Sequential(
                nn.Linear(d_model, d_model // 4),
                nn.Tanh(),
                nn.Linear(d_model // 4, 1),
            )

        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, hidden_states, segment_attention_mask):
        """
        Args:
            hidden_states: (batch, max_seg, max_seq_len, d_model)
            segment_attention_mask: (batch, max_seg) - 1 for valid segments

        Returns:
            logits: (batch, num_classes)
        """
        # Extract [CLS] tokens from all segments: (batch, max_seg, d_model)
        cls_tokens = hidden_states[:, :, 0, :]

        if self.pooling == "last_cls":
            # Get the last valid segment's [CLS] token
            # segment_attention_mask: (batch, max_seg)
            # We want the last 1 in each row
            # Sum along dim=1 gives the count of valid segments
            valid_counts = segment_attention_mask.sum(dim=1).long()  # (batch,)
            batch_size = cls_tokens.shape[0]

            # Get indices of last valid segment
            last_indices = (valid_counts - 1).clamp(min=0)  # (batch,)

            # Gather the [CLS] token from last valid segment
            batch_indices = torch.arange(batch_size, device=cls_tokens.device)
            pooled = cls_tokens[batch_indices, last_indices]  # (batch, d_model)

        elif self.pooling == "mean_cls":
            # Mean pooling over valid [CLS] tokens
            mask = segment_attention_mask.unsqueeze(-1)  # (batch, max_seg, 1)
            pooled = (cls_tokens * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        elif self.pooling == "attention":
            # Attention-weighted pooling
            attn_scores = self.attention(cls_tokens).squeeze(-1)  # (batch, max_seg)
            # Mask invalid segments
            attn_scores = attn_scores.masked_fill(
                segment_attention_mask == 0, float('-inf')
            )
            attn_weights = F.softmax(attn_scores, dim=1)  # (batch, max_seg)
            pooled = (cls_tokens * attn_weights.unsqueeze(-1)).sum(dim=1)

        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)

        return logits


class HATForSequenceClassification(nn.Module):
    """HAT model with classification head for downstream tasks."""

    def __init__(
        self,
        config: FMConfig,
        num_classes: int,
        dropout: float = 0.1,
        pooling: str = "last_cls",
        freeze_encoder: bool = False,
    ):
        super().__init__()
        self.config = config
        self.num_classes = num_classes

        # Pre-trained encoder (will load weights separately)
        self.encoder = FMBase(config)

        # Classification head
        self.classifier = ClassificationHead(
            d_model=config.d_model,
            num_classes=num_classes,
            dropout=dropout,
            pooling=pooling,
        )

        # Optionally freeze encoder
        if freeze_encoder:
            self.freeze_encoder()

    def freeze_encoder(self):
        """Freeze encoder parameters for feature extraction mode."""
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        """Unfreeze encoder parameters for full fine-tuning."""
        for param in self.encoder.parameters():
            param.requires_grad = True

    def load_pretrained(self, checkpoint_path: str, strict: bool = True):
        """Load pre-trained encoder weights from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        # Remove 'encoder.' prefix if present (from HF format)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('encoder.'):
                new_state_dict[k] = v
            else:
                new_state_dict[f'encoder.{k}'] = v

        # Load encoder weights (ignore classifier weights)
        encoder_state_dict = {
            k: v for k, v in new_state_dict.items()
            if k.startswith('encoder.')
        }

        missing, unexpected = self.load_state_dict(encoder_state_dict, strict=False)

        # Filter out expected missing keys (classifier)
        missing = [k for k in missing if not k.startswith('classifier.')]

        if missing and strict:
            raise RuntimeError(f"Missing keys in checkpoint: {missing}")

        return missing, unexpected

    def forward(
        self,
        input_ids,
        attention_mask,
        segment_attention_mask,
        segment_time=None,
        token_time=None,
        labels=None,
    ):
        """
        Args:
            input_ids: (batch, max_seg, max_seq_len)
            attention_mask: (batch, max_seg, max_seq_len)
            segment_attention_mask: (batch, max_seg)
            segment_time: (batch, max_seg) - optional
            token_time: (batch, max_seg, max_seq_len) - optional
            labels: (batch,) - optional, for computing loss

        Returns:
            dict with 'loss' (if labels provided) and 'logits'
        """
        # Encode
        hidden_states = self.encoder.encode(
            input_ids,
            attention_mask,
            segment_attention_mask,
            segment_time,
            token_time,
        )

        # Classify
        logits = self.classifier(hidden_states, segment_attention_mask)

        output = {"logits": logits}

        if labels is not None:
            # Cross-entropy works for both binary and multi-class classification
            loss = F.cross_entropy(logits, labels)
            output["loss"] = loss

        return output

    def predict_proba(self, logits):
        """Convert logits to probabilities."""
        if self.num_classes == 2:
            return F.softmax(logits, dim=-1)[:, 1]  # Return positive class prob
        else:
            return F.softmax(logits, dim=-1)


def create_finetune_model(
    pretrained_path: str,
    num_classes: int,
    dropout: float = 0.1,
    pooling: str = "last_cls",
    freeze_encoder: bool = False,
) -> HATForSequenceClassification:
    """
    Create a fine-tuning model from pre-trained checkpoint.

    Args:
        pretrained_path: Path to pre-trained checkpoint
        num_classes: Number of output classes
        dropout: Dropout rate for classifier
        pooling: Pooling strategy ("last_cls", "mean_cls", "attention")
        freeze_encoder: Whether to freeze encoder weights

    Returns:
        HATForSequenceClassification model with loaded weights
    """
    # Load config from checkpoint
    checkpoint = torch.load(pretrained_path, map_location='cpu')

    if 'config' in checkpoint:
        config = checkpoint['config']
        if isinstance(config, dict):
            config = FMConfig(**config)
    else:
        # Use default config (you may need to adjust this)
        config = FMConfig()

    # Create model
    model = HATForSequenceClassification(
        config=config,
        num_classes=num_classes,
        dropout=dropout,
        pooling=pooling,
        freeze_encoder=freeze_encoder,
    )

    # Load pre-trained weights
    model.load_pretrained(pretrained_path, strict=False)

    return model
