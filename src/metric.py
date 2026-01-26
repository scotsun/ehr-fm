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

    The model uses position 1 (MASK token) to predict ALL tokens in the segment
    as a set prediction task. So targets should be the entire segment's tokens,
    not just position 1-k.

    Args:
        logits: (batch, max_seg, max_seq_len, vocab_size)
        input_ids: (batch, max_seg, max_seq_len) - original input IDs
        segment_attention_mask: (batch, max_seg) - which segments are valid
        k: number of predictions to extract

    Returns:
        p_tokens: (batch, k) - predicted token IDs from position 1
        t_tokens: (batch, max_seq_len) - ALL target token IDs from last segment
    """
    batch_size = logits.size(0)
    device = logits.device

    # Find the last valid segment for each batch
    # segment_attention_mask: (batch, max_seg), sum gives count of valid segments
    last_seg_idx = segment_attention_mask.sum(dim=1).long() - 1  # (batch,)
    last_seg_idx = last_seg_idx.clamp(min=0)

    batch_indices = torch.arange(batch_size, device=device)

    # Extract ALL target tokens from last segment (the entire segment is the target set)
    # input_ids[batch_indices, last_seg_idx]: (batch, max_seq_len)
    t_tokens = input_ids[batch_indices, last_seg_idx]  # (batch, max_seq_len)

    # Extract predictions from last segment
    # logits[batch_indices, last_seg_idx]: (batch, max_seq_len, vocab_size)
    last_seg_logits = logits[batch_indices, last_seg_idx]  # (batch, max_seq_len, vocab_size)
    # Use position 1 (MASK token) to predict the entire segment's token set
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

    # p_tokens: (batch, k), t_tokens: (batch, max_seq_len)
    p_tokens, t_tokens = _get_last_segment_predictions_and_targets(
        logits, input_ids, segment_attention_mask, k
    )

    # Valid targets: exclude special tokens 0-3 (PAD, UNK, CLS, MASK)
    t_valid_mask = t_tokens > 3  # (batch, max_seq_len)
    t_setsize = t_valid_mask.sum(dim=-1)  # (batch,)

    # Check which predictions hit any valid target
    # p_tokens: (batch, k) -> (batch, k, 1)
    # t_tokens: (batch, max_seq_len) -> (batch, 1, max_seq_len)
    # hit_matrix[b, i, j] = (p_tokens[b, i] == t_tokens[b, j]) & t_valid_mask[b, j]
    hit_matrix = (p_tokens.unsqueeze(2) == t_tokens.unsqueeze(1)) & t_valid_mask.unsqueeze(1)
    # For each prediction, did it match any valid target?
    num_hits = hit_matrix.any(dim=2).sum(dim=-1)  # (batch,)

    # Recall = num_hits / target_size
    valid_samples = t_setsize > 0
    if valid_samples.sum() > 0:
        recall = num_hits[valid_samples].float() / t_setsize[valid_samples].float()
        return recall.mean()
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

    # p_tokens: (batch, k), t_tokens: (batch, max_seq_len)
    p_tokens, t_tokens = _get_last_segment_predictions_and_targets(
        logits, input_ids, segment_attention_mask, k
    )

    # Valid targets: exclude special tokens 0-3 (PAD, UNK, CLS, MASK)
    t_valid_mask = t_tokens > 3  # (batch, max_seq_len)
    t_setsize = t_valid_mask.sum(dim=-1)  # (batch,)

    # DCG discount terms: 1/log2(rank+1) for ranks 1 to k
    # ranks are 1-indexed, so discount[i] = 1/log2(i+2)
    discount_terms = 1.0 / torch.log2(
        torch.arange(1, k + 1, dtype=torch.float32, device=device) + 1
    )  # (k,)

    # Check which predictions match any valid target
    # p_tokens: (batch, k) -> (batch, k, 1)
    # t_tokens: (batch, max_seq_len) -> (batch, 1, max_seq_len)
    # hit_matrix[b, i, j] = (p_tokens[b, i] == t_tokens[b, j]) & t_valid_mask[b, j]
    hit_matrix = (p_tokens.unsqueeze(2) == t_tokens.unsqueeze(1)) & t_valid_mask.unsqueeze(1)
    is_hit = hit_matrix.any(dim=2)  # (batch, k) - is prediction i a hit?

    # DCG = sum of discounts for correct predictions
    dcg = (discount_terms.unsqueeze(0) * is_hit.float()).sum(dim=-1)  # (batch,)

    # Ideal DCG = sum of first min(|target|, k) discount terms
    # Precompute cumulative sum of discount terms
    idcg_cumsum = torch.zeros(k + 1, dtype=torch.float32, device=device)
    idcg_cumsum[1:] = torch.cumsum(discount_terms, dim=0)

    # idcg[b] = idcg_cumsum[min(t_setsize[b], k)]
    t_setsize_clamped = t_setsize.clamp(max=k)
    idcg = idcg_cumsum[t_setsize_clamped]  # (batch,)

    # NDCG = DCG / IDCG
    valid_samples = t_setsize > 0
    if valid_samples.sum() > 0:
        ndcg = dcg[valid_samples] / idcg[valid_samples].clamp(min=1e-8)
        return ndcg.mean()
    return torch.tensor(0.0, device=device)
