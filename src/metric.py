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


def pred_and_target_sets(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    set_attention_mask: torch.Tensor,
    k: int,
):
    last_set_id = set_attention_mask.sum(dim=1) - 1
    t_tokens = input_ids[range(len(last_set_id)), last_set_id, 1 : k + 1]
    # p_tokens = (
    #     logits[range(len(last_set_id)), last_set_id, 1]
    #     .topk(
    #         k=k,
    #         dim=-1,
    #     )
    #     .indices
    # )
    p_tokens = logits[range(len(last_set_id)), last_set_id, 1 : k + 1].argmax(dim=-1)
    return p_tokens, t_tokens


def recall_at_k(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    set_attention_mask: torch.Tensor,
    k=10,
):
    device = logits.device
    batch_size, vocab_size = logits.size(0), logits.size(-1)

    p_tokens, t_tokens = pred_and_target_sets(logits, input_ids, set_attention_mask, k)
    # (batch_size, k)

    t_setsize = (t_tokens > 3).sum(dim=-1)

    p_sets = torch.zeros(batch_size, vocab_size, dtype=torch.bool, device=device)
    t_sets = torch.zeros(batch_size, vocab_size, dtype=torch.bool, device=device)
    p_sets.scatter_(1, p_tokens, True)
    t_sets.scatter_(1, t_tokens, True)

    intersection_sets = p_sets & t_sets
    intersection_size = intersection_sets.sum(dim=-1)

    out = intersection_size / t_setsize
    return out[torch.isfinite(out)].mean()


def ndcg_at_k(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    set_attention_mask: torch.Tensor,
    k=10,
):
    device = logits.device

    p_tokens, t_tokens = pred_and_target_sets(logits, input_ids, set_attention_mask, k)
    # (batch_size, k)

    t_setsize = (t_tokens > 3).sum(dim=-1)

    terms = 1 / torch.log2(
        torch.arange(1, k + 1, dtype=torch.float32, device=device) + 1
    )

    is_present_matrix = p_tokens.unsqueeze(2) == t_tokens.unsqueeze(1)
    is_present = is_present_matrix.any(dim=2)
    dcg_at_k = (terms.unsqueeze(0) * is_present).sum(dim=-1)

    idcg_sums = torch.zeros(k + 1, dtype=torch.float32, device=device)
    idcg_sums[1:] = torch.cumsum(terms, dim=0)
    idcg_at_k = idcg_sums[t_setsize]

    out = dcg_at_k / idcg_at_k
    return out[torch.isfinite(out)].mean()
