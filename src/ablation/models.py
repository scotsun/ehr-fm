"""Ablation model variants for NEST architecture study.

B1: SWE-only (no CSE) — pretrain MLM only, finetune with Bi-GRU aggregation
B2: No SWE (mean pool → CSE) — pretrain MLM only
B3: SWE + learnable positional encoding — pretrain MLM + MSM (full recipe)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.fm import FMConfig, FMEmbeddings, FMTransformerEncoder
from src.layers.transformer import TransformerBlock
from src.layers.rope import T2V
from src.layers.ffn import FFNSwiGLUBlock


# ============================================================
# Block variants
# ============================================================

class SWEOnlyBlock(nn.Module):
    """B1: SWE only, no CSE. Each encounter processed independently."""

    def __init__(self, d_model, d_ff, h, swe_rope=True, dropout=0,
                 norm_type="layer", ffn_type="swiglu", attn_backend="base"):
        super().__init__()
        self.swe = TransformerBlock(
            d_model, d_ff, h, swe_rope, dropout, norm_type, ffn_type, attn_backend
        )

    def forward(self, x, token_mask, seg_mask, seg_time=None, token_time=None):
        # x: (B, S, L, D)
        if token_time is not None:
            token_time = token_time.float()
            token_time_reshaped = token_time.reshape(-1, token_time.shape[2]).contiguous()
        else:
            token_time_reshaped = None

        seg_hidden = self.swe(
            x.reshape(-1, x.shape[2], x.shape[3]).contiguous(),
            token_mask.reshape(-1, token_mask.shape[2]).contiguous(),
            time=token_time_reshaped,
        )
        return seg_hidden.reshape(x.shape[0], x.shape[1], x.shape[2], x.shape[3]).contiguous()


class MeanPoolCSEBlock(nn.Module):
    """B2: No SWE attention. Mean pool tokens → CSE only."""

    def __init__(self, d_model, d_ff, h, dropout=0,
                 norm_type="layer", ffn_type="swiglu", attn_backend="base"):
        super().__init__()
        self.cse = TransformerBlock(
            d_model, d_ff, h, True, dropout, norm_type, ffn_type, attn_backend
        )

    def forward(self, x, token_mask, seg_mask, seg_time=None, token_time=None):
        # x: (B, S, L, D)
        if seg_time is not None:
            seg_time = seg_time.float()

        # Mean pool non-CLS tokens to get set representation
        token_mask_no_cls = token_mask.clone()
        token_mask_no_cls[:, :, 0] = 0  # exclude CLS
        mask_exp = token_mask_no_cls.unsqueeze(-1)  # (B, S, L, 1)
        mean_pooled = (x * mask_exp).sum(dim=2) / mask_exp.sum(dim=2).clamp(min=1)  # (B, S, D)

        # CSE on mean-pooled set representations
        cse_out = self.cse(mean_pooled, seg_mask, time=seg_time)  # (B, S, D)

        # Broadcast CSE output to ALL token positions as residual.
        # Without SWE, this is the only way tokens receive cross-segment context.
        # cse_out.unsqueeze(2): (B, S, 1, D) → broadcast to (B, S, L, D)
        out = x + cse_out.unsqueeze(2)
        return out


# ============================================================
# Encoder variants
# ============================================================

class AblationTransformerEncoder(nn.Module):
    """Transformer encoder using ablation block variants."""

    def __init__(self, config: FMConfig, block_cls, block_kwargs):
        super().__init__()
        self.blocks = nn.ModuleList([
            block_cls(**block_kwargs) for _ in range(config.n_blocks)
        ])
        if config.norm_type == "layer":
            self.final_norm = nn.LayerNorm(config.d_model)
        else:
            self.final_norm = nn.RMSNorm(config.d_model)

    def forward(self, h, attention_mask, segment_attention_mask,
                segment_time=None, token_time=None):
        for block in self.blocks:
            h = block(h, attention_mask, segment_attention_mask, segment_time, token_time)
        h = self.final_norm(h)
        return h


# ============================================================
# Pretrain models
# ============================================================

class AblationFMBase(nn.Module):
    """Base pretrain model for B1/B2 ablations. Supports MLM only."""

    def __init__(self, config: FMConfig, block_cls, block_kwargs):
        super().__init__()
        self.config = config
        self.embeddings = FMEmbeddings(config)
        if config.use_t2v:
            self.t2v = T2V(config.d_model, config.t2v_scale)
        self.transformer_encoder = AblationTransformerEncoder(config, block_cls, block_kwargs)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size)
        if config.weight_tying:
            self.lm_head.weight = self.embeddings.embeddings.weight

    def encode(self, input_ids, attention_mask, segment_attention_mask,
               segment_time=None, token_time=None):
        h = self.embeddings(input_ids)
        if hasattr(self, 't2v') and token_time is not None:
            h = h + self.t2v(token_time)
        h = self.transformer_encoder(h, attention_mask, segment_attention_mask,
                                     segment_time, token_time)
        return h

    def forward(self, input_ids, attention_mask, segment_attention_mask,
                segment_time=None, token_time=None):
        h = self.encode(input_ids, attention_mask, segment_attention_mask,
                        segment_time, token_time)
        logits = self.lm_head(h)
        return logits, h


class SWEWithPEFMBase(nn.Module):
    """B3: Standard NEST architecture + learnable positional embeddings in SWE.

    Adds position 0, 1, ..., max_seq_len-1 embeddings to token representations
    before entering the transformer, breaking permutation invariance within sets.
    """

    def __init__(self, config: FMConfig, max_seq_len: int = 512):
        super().__init__()
        self.config = config
        self.embeddings = FMEmbeddings(config)
        if config.use_t2v:
            self.t2v = T2V(config.d_model, config.t2v_scale)
        self.position_embeddings = nn.Embedding(max_seq_len, config.d_model)
        self.transformer_encoder = FMTransformerEncoder(config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size)
        if config.weight_tying:
            self.lm_head.weight = self.embeddings.embeddings.weight

    def encode(self, input_ids, attention_mask, segment_attention_mask,
               segment_time=None, token_time=None):
        h = self.embeddings(input_ids)
        if hasattr(self, 't2v') and token_time is not None:
            h = h + self.t2v(token_time)
        # Add learnable positional embeddings (breaks permutation invariance)
        positions = torch.arange(h.shape[2], device=h.device)  # (max_seq_len,)
        h = h + self.position_embeddings(positions)  # broadcast over (B, S)
        h = self.transformer_encoder(h, attention_mask, segment_attention_mask,
                                     segment_time, token_time)
        return h

    def forward(self, input_ids, attention_mask, segment_attention_mask,
                segment_time=None, token_time=None):
        h = self.encode(input_ids, attention_mask, segment_attention_mask,
                        segment_time, token_time)
        logits = self.lm_head(h)
        return logits, h


class SWEWithPEFMBaseWithHeads(nn.Module):
    """B3 with MSM head: for pretraining with MLM + MSM (full NEST recipe)."""

    def __init__(self, config: FMConfig, max_seq_len: int = 512):
        super().__init__()
        self.config = config
        self.transformer = SWEWithPEFMBase(config, max_seq_len)
        self.mlp_dm = FFNSwiGLUBlock(d_model=config.d_model, d_ff=config.d_ff)
        self.dm_head = nn.Linear(config.d_model, config.vocab_size)

    def forward(self, input_ids, attention_mask, segment_attention_mask,
                segment_time=None, token_time=None, segment_mask=None):
        if segment_mask is None:
            segment_mask = segment_attention_mask
        logits_mlm, h = self.transformer(
            input_ids, attention_mask, segment_attention_mask, segment_time, token_time
        )
        h_cls = h[segment_mask][:, 0, :]
        h_dm = self.mlp_dm(h_cls)
        logits_dm = self.dm_head(h_dm)
        return logits_mlm, logits_dm, h

    def encode(self, input_ids, attention_mask, segment_attention_mask,
               segment_time=None, token_time=None):
        return self.transformer.encode(
            input_ids, attention_mask, segment_attention_mask, segment_time, token_time
        )


# ============================================================
# Finetune models
# ============================================================

class SWEOnlyForClassification(nn.Module):
    """B1 finetune: SWE-only encoder + Bi-GRU aggregation + classification head.

    Since there is no CSE, encounter CLS embeddings are independent.
    Bi-GRU provides temporal aggregation across encounters.
    """

    def __init__(self, config: FMConfig, num_classes: int, dropout: float = 0.1,
                 pooling: str = "last_cls"):
        super().__init__()
        self.config = config
        self.num_classes = num_classes

        block_kwargs = dict(
            d_model=config.d_model, d_ff=config.d_ff, h=config.n_heads,
            swe_rope=config.swe_rope, dropout=config.dropout,
            norm_type=config.norm_type, ffn_type=config.ffn_type,
        )
        self.encoder = AblationFMBase(config, SWEOnlyBlock, block_kwargs)

        # Bi-GRU for temporal aggregation across encounters
        self.gru = nn.GRU(
            config.d_model, config.d_model // 2,
            bidirectional=True, batch_first=True,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(config.d_model, num_classes),
        )

    def load_pretrained(self, checkpoint_path: str, strict: bool = True):
        """Load pre-trained SWE-only encoder weights."""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        # Map pretrained keys to encoder.* namespace
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('lm_head.'):
                continue  # skip MLM head
            new_state_dict[f'encoder.{k}'] = v

        encoder_keys = {k: v for k, v in new_state_dict.items() if k.startswith('encoder.')}
        missing, unexpected = self.load_state_dict(encoder_keys, strict=False)
        missing = [k for k in missing if not k.startswith('gru.') and not k.startswith('classifier.')]
        if missing and strict:
            raise RuntimeError(f"Missing keys: {missing}")
        return missing, unexpected

    def forward(self, input_ids, attention_mask, segment_attention_mask,
                segment_time=None, token_time=None, labels=None):
        hidden_states = self.encoder.encode(
            input_ids, attention_mask, segment_attention_mask, segment_time, token_time
        )
        # Extract CLS tokens: (B, S, D)
        cls_tokens = hidden_states[:, :, 0, :]

        # Bi-GRU aggregation
        lengths = segment_attention_mask.sum(dim=1).long().cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            cls_tokens, lengths, batch_first=True, enforce_sorted=False
        )
        gru_out, _ = self.gru(packed)
        gru_out, _ = nn.utils.rnn.pad_packed_sequence(gru_out, batch_first=True)

        # Take last valid position
        batch_indices = torch.arange(cls_tokens.shape[0], device=cls_tokens.device)
        last_indices = (lengths - 1).clamp(min=0).to(cls_tokens.device)
        pooled = gru_out[batch_indices, last_indices]

        logits = self.classifier(pooled)
        output = {"logits": logits}
        if labels is not None:
            output["loss"] = F.cross_entropy(logits, labels)
        return output

    def predict_proba(self, logits):
        if self.num_classes == 2:
            return F.softmax(logits, dim=-1)[:, 1]
        return F.softmax(logits, dim=-1)

    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = True


class AblationForClassification(nn.Module):
    """B2/B3 finetune: Ablation encoder + standard classification head.

    Uses the same classification head as HATForSequenceClassification.
    """

    def __init__(self, config: FMConfig, num_classes: int, dropout: float = 0.1,
                 pooling: str = "last_cls"):
        super().__init__()
        self.config = config
        self.num_classes = num_classes
        self.pooling = pooling

        # Encoder will be set by load_pretrained or set_encoder
        self.encoder = None
        self._build_classifier(config.d_model, num_classes, dropout, pooling)

    def _build_classifier(self, d_model, num_classes, dropout, pooling):
        from src.finetune.model import ClassificationHead
        self.classifier = ClassificationHead(
            d_model=d_model,
            num_classes=num_classes,
            dropout=dropout,
            pooling=pooling,
        )

    def set_encoder(self, encoder):
        """Set the encoder (AblationFMBase or SWEWithPEFMBase)."""
        self.encoder = encoder

    def load_pretrained(self, checkpoint_path: str, variant: str, strict: bool = True,
                        max_seq_len: int = 512):
        """Load pre-trained ablation encoder weights."""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        if 'config' in checkpoint:
            config = checkpoint['config']
            if isinstance(config, dict):
                config = FMConfig(**config)
            self.config = config

        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        # Create encoder based on variant
        if variant == "no_swe":
            block_kwargs = dict(
                d_model=self.config.d_model, d_ff=self.config.d_ff,
                h=self.config.n_heads, dropout=self.config.dropout,
                norm_type=self.config.norm_type, ffn_type=self.config.ffn_type,
            )
            self.encoder = AblationFMBase(self.config, MeanPoolCSEBlock, block_kwargs)
        elif variant == "swe_with_pe":
            self.encoder = SWEWithPEFMBase(self.config, max_seq_len)
        else:
            raise ValueError(f"Unknown variant: {variant}")

        # Load encoder weights (skip lm_head, dm heads)
        encoder_state = {}
        for k, v in state_dict.items():
            if k.startswith('lm_head.') or k.startswith('mlp_dm.') or k.startswith('dm_head.'):
                continue
            # Handle FMBaseWithHeads wrapping (B3)
            if k.startswith('transformer.'):
                encoder_state[f'encoder.{k[len("transformer."):]}'] = v
            else:
                encoder_state[f'encoder.{k}'] = v

        missing, unexpected = self.load_state_dict(encoder_state, strict=False)
        missing = [k for k in missing if not k.startswith('classifier.')]
        if missing and strict:
            raise RuntimeError(f"Missing keys: {missing}")
        return missing, unexpected

    def forward(self, input_ids, attention_mask, segment_attention_mask,
                segment_time=None, token_time=None, labels=None):
        hidden_states = self.encoder.encode(
            input_ids, attention_mask, segment_attention_mask, segment_time, token_time
        )
        logits = self.classifier(hidden_states, segment_attention_mask)
        output = {"logits": logits}
        if labels is not None:
            output["loss"] = F.cross_entropy(logits, labels)
        return output

    def predict_proba(self, logits):
        if self.num_classes == 2:
            return F.softmax(logits, dim=-1)[:, 1]
        return F.softmax(logits, dim=-1)

    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = True


# ============================================================
# Factory functions
# ============================================================

def create_ablation_pretrain_model(config: FMConfig, variant: str, max_seq_len: int = 512):
    """Create pretrain model for a given ablation variant.

    Args:
        config: FMConfig
        variant: "swe_only", "no_swe", or "swe_with_pe"
        max_seq_len: max sequence length (for B3 positional embeddings)

    Returns:
        model: nn.Module with forward() returning (logits, h) or (logits_mlm, logits_dm, h)
        needs_heads: bool — whether this variant uses MLM+MSM (needs BaseWithHeadsTrainer)
    """
    if variant == "swe_only":
        block_kwargs = dict(
            d_model=config.d_model, d_ff=config.d_ff, h=config.n_heads,
            swe_rope=config.swe_rope, dropout=config.dropout,
            norm_type=config.norm_type, ffn_type=config.ffn_type,
        )
        return AblationFMBase(config, SWEOnlyBlock, block_kwargs), False

    elif variant == "no_swe":
        block_kwargs = dict(
            d_model=config.d_model, d_ff=config.d_ff, h=config.n_heads,
            dropout=config.dropout, norm_type=config.norm_type,
            ffn_type=config.ffn_type,
        )
        return AblationFMBase(config, MeanPoolCSEBlock, block_kwargs), False

    elif variant == "swe_with_pe":
        # Full NEST recipe with PE — uses MLM + MSM
        return SWEWithPEFMBaseWithHeads(config, max_seq_len), True

    else:
        raise ValueError(f"Unknown pretrain variant: {variant}")


def create_ablation_finetune_model(
    variant: str,
    pretrained_path: str,
    num_classes: int,
    dropout: float = 0.1,
    pooling: str = "last_cls",
    max_seq_len: int = 512,
    freeze_encoder: bool = False,
):
    """Create finetune model for a given ablation variant.

    Args:
        variant: "swe_only", "no_swe", or "swe_with_pe"
        pretrained_path: path to pretrained ablation checkpoint
        num_classes: number of output classes
        dropout: classifier dropout
        pooling: pooling strategy for classification
        max_seq_len: max sequence length
        freeze_encoder: whether to freeze encoder weights

    Returns:
        model: nn.Module with forward() returning {"logits": ...}
    """
    checkpoint = torch.load(pretrained_path, map_location='cpu')
    if 'config' in checkpoint:
        config = checkpoint['config']
        if isinstance(config, dict):
            config = FMConfig(**config)
    else:
        # Infer config from state_dict
        sd = checkpoint.get('model_state_dict', checkpoint)
        embed_key = next((k for k in sd if 'embeddings.embeddings.weight' in k), None)
        if embed_key:
            vocab_size, d_model = sd[embed_key].shape
        else:
            vocab_size, d_model = 15000, 768
        d_ff = 2048
        for k, v in sd.items():
            if 'ffn_block.linear_gate.weight' in k:
                d_ff = v.shape[0]
                break
        n_blocks = 0
        for k in sd.keys():
            if 'blocks.' in k:
                idx = int(k.split('blocks.')[1].split('.')[0])
                n_blocks = max(n_blocks, idx + 1)
        config = FMConfig(
            vocab_size=vocab_size, d_model=d_model, d_ff=d_ff,
            n_blocks=n_blocks or 6, n_heads=d_model // 64,
        )

    if variant == "swe_only":
        model = SWEOnlyForClassification(
            config=config, num_classes=num_classes,
            dropout=dropout, pooling=pooling,
        )
        model.load_pretrained(pretrained_path, strict=False)
    else:
        model = AblationForClassification(
            config=config, num_classes=num_classes,
            dropout=dropout, pooling=pooling,
        )
        model.load_pretrained(pretrained_path, variant=variant,
                              strict=False, max_seq_len=max_seq_len)

    if freeze_encoder:
        model.freeze_encoder()

    return model
