import torch


def topk_accuracy(logits: torch.Tensor, labels: torch.Tensor, k=1):
    masked_position = labels != -100
    if not masked_position.any():
        return 0.0
    masked_logits = logits[masked_position]  # (num_masked, vocab_size)
    masked_labels = labels[masked_position]  # (num_masked,)

    topk_indices = torch.topk(masked_logits, k=k, dim=-1).indices  # (num_masked, k)

    correct = (masked_labels.unsqueeze(-1) == topk_indices).any(dim=-1)  # (num_masked,)

    return correct.float().mean()


def select_last_set(set_attention_mask: torch.Tensor):
    last_set_id = set_attention_mask.sum(dim=1) - 1
    set_select_mask = torch.zeros_like(set_attention_mask, dtype=torch.bool)
    set_select_mask[range(len(last_set_id)), last_set_id] = True
    return set_select_mask


def pred_and_target_sets_1d(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    mask_pos: torch.Tensor,
    k: int = 10,
    max_set_size: int = 32,
):
    B = logits.shape[0]

    t_tokens = torch.full((B, max_set_size), 0, dtype=torch.long, device=logits.device)
    for i in range(B):
        _tokens = input_ids[i, mask_pos[i, 1] :]
        t_tokens[i, : min(max_set_size, _tokens.size(0))] = _tokens[
            : min(max_set_size, _tokens.size(0))
        ]
    p_tokens = logits[torch.arange(B), mask_pos[:, 1], :].topk(k=k, dim=-1).indices
    # p_tokens: (B, k); t_tokens: (B, max_set_size)
    return p_tokens, t_tokens


def pred_and_target_sets(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    set_select_mask: torch.Tensor,
    k: int = 10,
):
    # one-mask-for-all-query
    p_tokens = logits[set_select_mask, 1].topk(k=k, dim=-1).indices
    t_tokens = input_ids[set_select_mask]
    # p_tokens: (B, k); t_tokens: (B, max_set_size)
    return p_tokens, t_tokens


def recall_at_k(p_tokens: torch.Tensor, t_tokens: torch.Tensor):
    # p_tokens: (batch_size, k); t_tokens: (batch_size, n)
    _valid = t_tokens > 3  # get rid of special tokens
    denom = _valid.sum(dim=-1)

    hit = (p_tokens.unsqueeze(2) == t_tokens.unsqueeze(1)) & _valid.unsqueeze(1)
    num_hits = hit.any(dim=2).sum(dim=-1)
    return (num_hits[denom > 0] / denom[denom > 0]).mean()


def ndcg_at_k(p_tokens: torch.Tensor, t_tokens: torch.Tensor):
    k = p_tokens.size(1)
    device = p_tokens.device
    # p_tokens: (batch_size, k); t_tokens: (batch_size, n)

    _valid = t_tokens > 3  # get rid of special tokens
    t_setsize = _valid.sum(dim=-1).clamp(max=k)

    hits = (p_tokens.unsqueeze(2) == t_tokens.unsqueeze(1)) & _valid.unsqueeze(1)
    hits = hits.any(dim=2)  # (B, k)

    terms = 1 / torch.log2(
        torch.arange(1, k + 1, dtype=torch.float32, device=device) + 1
    )
    dcg_at_k = (terms.unsqueeze(0) * hits).sum(dim=-1)

    idcg_sums = torch.zeros(k + 1, device=device)
    idcg_sums[1:] = torch.cumsum(terms, dim=0)
    idcg_at_k = idcg_sums[t_setsize]

    return (dcg_at_k[idcg_at_k > 0] / idcg_at_k[idcg_at_k > 0]).mean()
