"""Evaluation metrics for EHR Foundation Model."""

import torch


def topk_accuracy(logits: torch.Tensor, labels: torch.Tensor, k: int = 1) -> torch.Tensor:
    """
    Compute top-k accuracy for masked token prediction.

    Args:
        logits: (batch, max_seg, max_seq_len, vocab_size) or flattened
        labels: (batch, max_seg, max_seq_len) or flattened, -100 for non-masked positions
        k: number of top predictions to consider

    Returns:
        Top-k accuracy (scalar tensor)
    """
    # Flatten if needed
    if logits.dim() > 2:
        logits = logits.view(-1, logits.size(-1))
        labels = labels.view(-1)

    # Only consider masked positions (labels != -100)
    masked_position = labels != -100
    if not masked_position.any():
        return torch.tensor(0.0, device=logits.device)

    masked_logits = logits[masked_position]  # (num_masked, vocab_size)
    masked_labels = labels[masked_position]  # (num_masked,)

    # Get top-k predictions
    topk_indices = torch.topk(masked_logits, k=k, dim=-1).indices  # (num_masked, k)

    # Check if true label is in top-k
    correct = (masked_labels.unsqueeze(-1) == topk_indices).any(dim=-1)  # (num_masked,)

    return correct.float().mean()


def _get_last_segment_predictions_and_targets(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    segment_attention_mask: torch.Tensor,
    k: int,
):
    """
    Helper function to extract predictions and targets for the last segment.

    Args:
        logits: (batch, max_seg, max_seq_len, vocab_size)
        input_ids: (batch, max_seg, max_seq_len) - original input IDs
        segment_attention_mask: (batch, max_seg) - which segments are valid
        k: number of predictions to extract

    Returns:
        p_tokens: (batch, k) - predicted token IDs
        t_tokens: (batch, k) - target token IDs from last segment
    """
    batch_size = logits.size(0)
    device = logits.device

    # Find the last valid segment for each batch
    # segment_attention_mask: (batch, max_seg), sum gives count of valid segments
    last_seg_idx = segment_attention_mask.sum(dim=1).long() - 1  # (batch,)
    last_seg_idx = last_seg_idx.clamp(min=0)

    batch_indices = torch.arange(batch_size, device=device)

    # Extract target tokens from last segment (skip CLS at position 0)
    # input_ids[batch_indices, last_seg_idx]: (batch, max_seq_len)
    last_seg_tokens = input_ids[batch_indices, last_seg_idx]  # (batch, max_seq_len)
    t_tokens = last_seg_tokens[:, 1:k+1]  # Skip CLS, take k tokens: (batch, k)

    # Extract predictions from last segment
    # logits[batch_indices, last_seg_idx]: (batch, max_seq_len, vocab_size)
    last_seg_logits = logits[batch_indices, last_seg_idx]  # (batch, max_seq_len, vocab_size)
    # Use position 1 (after CLS) to predict tokens
    p_tokens = last_seg_logits[:, 1].topk(k=k, dim=-1).indices  # (batch, k)

    return p_tokens, t_tokens


def recall_at_k(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    segment_attention_mask: torch.Tensor,
    k: int = 10,
) -> torch.Tensor:
    """
    Compute Recall@K for last segment prediction.

    Measures what fraction of actual tokens in the last segment
    are captured in the top-k predictions.

    Args:
        logits: (batch, max_seg, max_seq_len, vocab_size)
        input_ids: (batch, max_seg, max_seq_len)
        segment_attention_mask: (batch, max_seg)
        k: number of top predictions to consider

    Returns:
        Recall@K (scalar tensor)
    """
    device = logits.device
    batch_size = logits.size(0)
    vocab_size = logits.size(-1)

    p_tokens, t_tokens = _get_last_segment_predictions_and_targets(
        logits, input_ids, segment_attention_mask, k
    )

    # Count valid target tokens (exclude special tokens 0-3: PAD, UNK, CLS, MASK)
    t_valid_mask = t_tokens > 3
    t_setsize = t_valid_mask.sum(dim=-1)  # (batch,)

    # Convert to sets using one-hot encoding
    p_sets = torch.zeros(batch_size, vocab_size, dtype=torch.bool, device=device)
    t_sets = torch.zeros(batch_size, vocab_size, dtype=torch.bool, device=device)
    p_sets.scatter_(1, p_tokens, True)
    t_sets.scatter_(1, t_tokens, True)

    # Compute intersection
    intersection_sets = p_sets & t_sets
    intersection_size = intersection_sets.sum(dim=-1)  # (batch,)

    # Recall = intersection / target_size
    recall = intersection_size.float() / t_setsize.float().clamp(min=1)

    # Only average over samples with valid targets
    valid_samples = t_setsize > 0
    if valid_samples.sum() > 0:
        return recall[valid_samples].mean()
    return torch.tensor(0.0, device=device)


def ndcg_at_k(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    segment_attention_mask: torch.Tensor,
    k: int = 10,
) -> torch.Tensor:
    """
    Compute NDCG@K (Normalized Discounted Cumulative Gain) for last segment prediction.

    Measures ranking quality - rewards correct predictions at higher ranks.

    Args:
        logits: (batch, max_seg, max_seq_len, vocab_size)
        input_ids: (batch, max_seg, max_seq_len)
        segment_attention_mask: (batch, max_seg)
        k: number of top predictions to consider

    Returns:
        NDCG@K (scalar tensor)
    """
    device = logits.device

    p_tokens, t_tokens = _get_last_segment_predictions_and_targets(
        logits, input_ids, segment_attention_mask, k
    )

    # Count valid target tokens
    t_valid_mask = t_tokens > 3
    t_setsize = t_valid_mask.sum(dim=-1)  # (batch,)

    # DCG discount terms: 1/log2(rank+1) for ranks 1 to k
    # ranks are 1-indexed, so discount[i] = 1/log2(i+2)
    discount_terms = 1.0 / torch.log2(
        torch.arange(1, k + 1, dtype=torch.float32, device=device) + 1
    )  # (k,)

    # Check which predictions match any target
    # p_tokens: (batch, k), t_tokens: (batch, k)
    # is_present_matrix[b, i, j] = (p_tokens[b, i] == t_tokens[b, j])
    is_present_matrix = p_tokens.unsqueeze(2) == t_tokens.unsqueeze(1)  # (batch, k, k)
    is_present = is_present_matrix.any(dim=2)  # (batch, k) - is prediction i correct?

    # DCG = sum of discounts for correct predictions
    dcg = (discount_terms.unsqueeze(0) * is_present.float()).sum(dim=-1)  # (batch,)

    # Ideal DCG = sum of first |target| discount terms
    # Precompute cumulative sum of discount terms
    idcg_cumsum = torch.zeros(k + 1, dtype=torch.float32, device=device)
    idcg_cumsum[1:] = torch.cumsum(discount_terms, dim=0)

    # idcg[b] = idcg_cumsum[min(t_setsize[b], k)]
    t_setsize_clamped = t_setsize.clamp(max=k)
    idcg = idcg_cumsum[t_setsize_clamped]  # (batch,)

    # NDCG = DCG / IDCG
    ndcg = dcg / idcg.clamp(min=1e-8)

    # Only average over samples with valid targets
    valid_samples = t_setsize > 0
    if valid_samples.sum() > 0:
        return ndcg[valid_samples].mean()
    return torch.tensor(0.0, device=device)
