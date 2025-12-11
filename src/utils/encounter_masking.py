"""
Encounter-level Masking (Vectorized Version)
Mask entire encounters and predict all event types (DX, MED, LAB, PR, etc.)
"""

import torch
from tokenizers import Tokenizer


def encounter_masking(
    input_ids: torch.Tensor,
    segment_attention_mask: torch.Tensor,
    tokenizer: Tokenizer,
    mask_probability: float = 0.2
):
    """
    Vectorized encounter-level masking: randomly select encounters and mask all their tokens.

    Args:
        input_ids: (batch, max_seg, max_seq_len)
        segment_attention_mask: (batch, max_seg) - which segments are valid
        tokenizer: Tokenizer instance
        mask_probability: probability of masking each encounter (default: 0.2)

    Returns:
        masked_input_ids: (batch, max_seg, max_seq_len) - masked input
        labels: (batch, max_seg, max_seq_len) - prediction targets, -100 for ignore
        encounter_mask: (batch, max_seg) - which encounters were masked
    """
    device = input_ids.device
    batch_size, max_seg, max_seq_len = input_ids.shape

    # Get special token IDs
    pad_id = tokenizer.token_to_id("[PAD]")
    cls_id = tokenizer.token_to_id("[CLS]")
    mask_id = tokenizer.token_to_id("[MASK]")
    vocab_size = tokenizer.get_vocab_size()

    # Clone input for modification
    masked_input_ids = input_ids.clone()

    # Step 1: Randomly select encounters to mask
    # Only mask valid segments (where segment_attention_mask == 1)
    encounter_probs = torch.rand(batch_size, max_seg, device=device)
    encounter_mask = (encounter_probs < mask_probability) & (segment_attention_mask.bool())
    # Shape: (batch, max_seg)

    # Step 2: Expand encounter_mask to token level
    # (batch, max_seg) -> (batch, max_seg, max_seq_len)
    token_in_masked_encounter = encounter_mask.unsqueeze(-1).expand(-1, -1, max_seq_len)

    # Step 3: Identify special tokens (should not be masked)
    is_pad = (input_ids == pad_id)
    is_cls = (input_ids == cls_id)
    is_special = is_pad | is_cls

    # Step 4: Tokens to mask = in masked encounter AND not special
    tokens_to_mask = token_in_masked_encounter & ~is_special

    # Step 5: Create labels (-100 for positions we don't predict)
    labels = torch.full_like(input_ids, -100)
    labels[tokens_to_mask] = input_ids[tokens_to_mask]

    # Step 6: Apply 80/10/10 strategy
    # Generate random values for all positions
    rand_vals = torch.rand(batch_size, max_seg, max_seq_len, device=device)

    # 80% -> [MASK]
    mask_token_indices = tokens_to_mask & (rand_vals < 0.8)
    masked_input_ids[mask_token_indices] = mask_id

    # 10% -> random token (exclude special tokens 0-3)
    random_token_indices = tokens_to_mask & (rand_vals >= 0.8) & (rand_vals < 0.9)
    random_tokens = torch.randint(
        4, vocab_size,
        (batch_size, max_seg, max_seq_len),
        device=device,
        dtype=input_ids.dtype
    )
    masked_input_ids[random_token_indices] = random_tokens[random_token_indices]

    # 10% -> unchanged (rand >= 0.9), nothing to do

    return masked_input_ids, labels, encounter_mask


def evaluate_encounter_prediction(
    logits: torch.Tensor,
    labels: torch.Tensor,
    encounter_mask: torch.Tensor,
    tokenizer: Tokenizer
):
    """
    Evaluate encounter-level prediction performance (vectorized).

    Args:
        logits: (batch, max_seg, max_seq_len, vocab_size)
        labels: (batch, max_seg, max_seq_len)
        encounter_mask: (batch, max_seg)
        tokenizer: Tokenizer instance

    Returns:
        metrics: dict with evaluation metrics
    """
    # Get predictions
    predictions = logits.argmax(dim=-1)  # (batch, max_seg, max_seq_len)

    # Valid positions = where we need to predict (labels != -100)
    valid_mask = (labels != -100)

    # Token-level accuracy
    if valid_mask.sum() > 0:
        correct = (predictions == labels) & valid_mask
        token_accuracy = correct.sum().float() / valid_mask.sum().float()
    else:
        token_accuracy = torch.tensor(0.0, device=logits.device)

    # Encounter-level accuracy (all tokens in encounter must be correct)
    batch_size, max_seg, max_seq_len = labels.shape

    # Reshape to (batch * max_seg, max_seq_len)
    valid_flat = valid_mask.view(batch_size * max_seg, max_seq_len)
    correct_flat = ((predictions == labels) & valid_mask).view(batch_size * max_seg, max_seq_len)
    encounter_mask_flat = encounter_mask.view(-1)  # (batch * max_seg,)

    # For each encounter: count correct and total
    correct_per_enc = correct_flat.sum(dim=1)  # (batch * max_seg,)
    total_per_enc = valid_flat.sum(dim=1)      # (batch * max_seg,)

    # Encounter is correct if all tokens are correct (and has at least 1 token)
    enc_correct = (correct_per_enc == total_per_enc) & (total_per_enc > 0) & encounter_mask_flat
    enc_total = encounter_mask_flat & (total_per_enc > 0)

    if enc_total.sum() > 0:
        encounter_accuracy = enc_correct.sum().float() / enc_total.sum().float()
    else:
        encounter_accuracy = torch.tensor(0.0, device=logits.device)

    metrics = {
        'token_accuracy': token_accuracy.item(),
        'encounter_accuracy': encounter_accuracy.item(),
        'num_masked_encounters': encounter_mask.sum().item(),
        'num_masked_tokens': valid_mask.sum().item(),
    }

    return metrics
