import torch


def topk_accuracy(logits, labels, k=1):
    masked_position = labels != -100
    if not masked_position.any():
        return 0.0
    masked_logits = logits[masked_position]  # (num_masked, vocab_size)
    masked_labels = labels[masked_position]  # (num_masked,)

    topk_indices = torch.topk(masked_logits, k=k, dim=-1).indices  # (num_masked, k)

    correct = (masked_labels.unsqueeze(-1) == topk_indices).any(dim=-1)  # (num_masked,)

    return correct.float().mean()
