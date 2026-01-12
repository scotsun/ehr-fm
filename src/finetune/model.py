"""Fine-tuning Model for HAT downstream tasks."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.fm import FMBase, FMConfig


class ClassificationHead(nn.Module):
    """Classification head with optional pooling and dropout."""

    def __init__(
        self,
        d_model: int,
        num_classes: int,
        dropout: float = 0.1,
        pooling: str = "last_cls",
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
            segment_attention_mask: (batch, max_seg)
        Returns:
            logits: (batch, num_classes)
        """
        cls_tokens = hidden_states[:, :, 0, :]  # (batch, max_seg, d_model)

        if self.pooling == "last_cls":
            valid_counts = segment_attention_mask.sum(dim=1).long()
            batch_size = cls_tokens.shape[0]
            last_indices = (valid_counts - 1).clamp(min=0)
            batch_indices = torch.arange(batch_size, device=cls_tokens.device)
            pooled = cls_tokens[batch_indices, last_indices]

        elif self.pooling == "mean_cls":
            mask = segment_attention_mask.unsqueeze(-1)
            pooled = (cls_tokens * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        elif self.pooling == "attention":
            attn_scores = self.attention(cls_tokens).squeeze(-1)
            attn_scores = attn_scores.masked_fill(segment_attention_mask == 0, float('-inf'))
            attn_weights = F.softmax(attn_scores, dim=1)
            pooled = (cls_tokens * attn_weights.unsqueeze(-1)).sum(dim=1)

        pooled = self.dropout(pooled)
        return self.classifier(pooled)


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

        self.encoder = FMBase(config)
        self.classifier = ClassificationHead(
            d_model=config.d_model,
            num_classes=num_classes,
            dropout=dropout,
            pooling=pooling,
        )

        if freeze_encoder:
            self.freeze_encoder()

    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = True

    def load_pretrained(self, checkpoint_path: str, strict: bool = True):
        """Load pre-trained encoder weights from checkpoint.

        Supports both FMBase and FMBaseWithHeads checkpoints:
        - FMBase: keys start without prefix or with 'encoder.'
        - FMBaseWithHeads: keys start with 'transformer.' (for the inner FMBase)
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        # Detect checkpoint type and normalize keys
        # Check if this is a FMBaseWithHeads checkpoint (keys have 'transformer.' prefix)
        is_with_heads = any(k.startswith('transformer.') for k in state_dict.keys())

        new_state_dict = {}
        for k, v in state_dict.items():
            # Skip non-encoder keys from FMBaseWithHeads (mlp_dm, dm_head)
            if k.startswith('mlp_dm.') or k.startswith('dm_head.'):
                continue

            if is_with_heads:
                # FMBaseWithHeads: remove 'transformer.' prefix, then add 'encoder.'
                if k.startswith('transformer.'):
                    new_key = f'encoder.{k[len("transformer."):]}'
                    new_state_dict[new_key] = v
            else:
                # FMBase: add 'encoder.' prefix if not present
                if k.startswith('encoder.'):
                    new_state_dict[k] = v
                else:
                    new_state_dict[f'encoder.{k}'] = v

        encoder_state_dict = {k: v for k, v in new_state_dict.items() if k.startswith('encoder.')}
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
            labels: (batch,) - optional
        Returns:
            dict with 'loss' (if labels provided) and 'logits'
        """
        hidden_states = self.encoder.encode(
            input_ids,
            attention_mask,
            segment_attention_mask,
            segment_time,
            token_time,
        )

        logits = self.classifier(hidden_states, segment_attention_mask)
        output = {"logits": logits}

        if labels is not None:
            output["loss"] = F.cross_entropy(logits, labels)

        return output

    def predict_proba(self, logits):
        """Convert logits to probabilities."""
        if self.num_classes == 2:
            return F.softmax(logits, dim=-1)[:, 1]
        return F.softmax(logits, dim=-1)


def create_finetune_model(
    pretrained_path: str,
    num_classes: int,
    dropout: float = 0.1,
    pooling: str = "last_cls",
    freeze_encoder: bool = False,
) -> HATForSequenceClassification:
    """Create a fine-tuning model from pre-trained checkpoint.

    Supports both FMBase and FMBaseWithHeads checkpoints.
    """
    checkpoint = torch.load(pretrained_path, map_location='cpu')

    if 'config' in checkpoint:
        config = checkpoint['config']
        if isinstance(config, dict):
            config = FMConfig(**config)
    else:
        # Infer config from state_dict
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        # Detect if this is a FMBaseWithHeads checkpoint
        is_with_heads = any(k.startswith('transformer.') for k in state_dict.keys())

        # Get vocab_size and d_model from embedding weight
        embed_key = None
        for k in state_dict.keys():
            if 'embeddings.embeddings.weight' in k:
                embed_key = k
                break

        if embed_key:
            vocab_size, d_model = state_dict[embed_key].shape
        else:
            vocab_size, d_model = 15000, 768

        # Get d_ff from FFN layer (handle both checkpoint types)
        d_ff = 2048
        for k, v in state_dict.items():
            if 'ffn_block.linear_gate.weight' in k:
                d_ff = v.shape[0]
                break

        # Infer n_heads (assume head_dim=64)
        n_heads = d_model // 64

        # Count transformer blocks (handle both checkpoint types)
        n_blocks = 0
        block_pattern = 'transformer.transformer_encoder.blocks.' if is_with_heads else 'transformer_encoder.blocks.'
        for k in state_dict.keys():
            if block_pattern in k:
                parts = k.split(block_pattern)
                if len(parts) > 1:
                    block_idx = int(parts[1].split('.')[0])
                    n_blocks = max(n_blocks, block_idx + 1)
        if n_blocks == 0:
            n_blocks = 6

        config = FMConfig(
            vocab_size=vocab_size,
            d_model=d_model,
            d_ff=d_ff,
            n_blocks=n_blocks,
            n_heads=n_heads,
        )
        ckpt_type = "FMBaseWithHeads" if is_with_heads else "FMBase"
        print(f"Detected checkpoint type: {ckpt_type}")
        print(f"Inferred config: vocab_size={vocab_size}, d_model={d_model}, d_ff={d_ff}, n_blocks={n_blocks}, n_heads={n_heads}")

    model = HATForSequenceClassification(
        config=config,
        num_classes=num_classes,
        dropout=dropout,
        pooling=pooling,
        freeze_encoder=freeze_encoder,
    )

    model.load_pretrained(pretrained_path, strict=False)
    return model
