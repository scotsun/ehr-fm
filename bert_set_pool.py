import torch


def segment_mean_pool_and_tokens(
    last_hidden_state: torch.Tensor,  # [B, L, H]
    input_ids: torch.Tensor,  # [B, L]
    attention_mask: torch.Tensor,  # [B, L]
    cls_token_id: int,
    pad_token_id: int,
):
    """
    Segments are defined as: [CLS] + following context tokens until next [CLS] or [PAD].
    Returns:
      - pooled: list (len=B) of [S_i, H] tensors (mean over tokens in each segment, incl [CLS])
      - tokens: list (len=B) of lists (len=S_i) of [T_ij, H] tensors (token embeddings per segment)
      - token_indices: list (len=B) of lists of 1D LongTensors with positions for each segment
    """
    B, L, H = last_hidden_state.shape
    device = last_hidden_state.device

    pooled_per_batch = []
    tokens_per_batch = []
    idxs_per_batch = []

    for b in range(B):
        ids = input_ids[b]  # [L]
        mask = attention_mask[b].bool()  # [L]
        hs = last_hidden_state[b]  # [L, H]

        # Stop at first PAD (or end). Also respect attention_mask.
        valid = mask & (ids != pad_token_id)
        valid_positions = torch.nonzero(valid, as_tuple=False).squeeze(-1)
        if valid_positions.numel() == 0:
            pooled_per_batch.append(torch.empty((0, H), device=device))
            tokens_per_batch.append([])
            idxs_per_batch.append([])
            continue

        last_valid_pos = int(valid_positions[-1].item())
        ids_v = ids[: last_valid_pos + 1]
        hs_v = hs[: last_valid_pos + 1]

        cls_pos = torch.nonzero(ids_v == cls_token_id, as_tuple=False).squeeze(-1)
        if cls_pos.numel() == 0:
            pooled_per_batch.append(torch.empty((0, H), device=device))
            tokens_per_batch.append([])
            idxs_per_batch.append([])
            continue

        # Segment boundaries: [cls_pos[i], cls_pos[i+1]) and last to end.
        seg_starts = cls_pos.tolist()
        seg_ends = seg_starts[1:] + [ids_v.shape[0]]

        pooled_list = []
        token_list = []
        idx_list = []

        for s, e in zip(seg_starts, seg_ends):
            # segment includes CLS and all following tokens until next CLS (exclusive)
            seg_idxs = torch.arange(s, e, device=device, dtype=torch.long)
            seg_tokens = hs_v[seg_idxs]  # [T, H]

            seg_mean = seg_tokens.mean(dim=0)  # [H]

            pooled_list.append(seg_mean)
            token_list.append(seg_tokens)
            idx_list.append(seg_idxs)

        pooled_per_batch.append(torch.stack(pooled_list, dim=0))  # [S, H]
        tokens_per_batch.append(token_list)  # list of [T, H]
        idxs_per_batch.append(idx_list)  # list of [T]

    return pooled_per_batch, tokens_per_batch, idxs_per_batch


# Example usage:
# outputs = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
# pooled, seg_tokens, seg_pos = segment_mean_pool_and_tokens(
#     outputs.last_hidden_state,
#     input_ids,
#     attention_mask,
#     cls_token_id=tokenizer.cls_token_id,
#     pad_token_id=tokenizer.pad_token_id,
# )
#
# pooled[b].shape -> [num_segments, hidden]
# seg_tokens[b][k].shape -> [segment_len, hidden]
