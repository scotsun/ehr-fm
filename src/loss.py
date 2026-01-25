import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftCLT(nn.Module):
    """adapted from softclt github repo"""

    def __init__(self, model, temperature):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1, z2, mask1, mask2):
        # the mask is the set_attention_mask
        pass

    def masked_max_pool1d(self, z, set_attention_mask, kernel_size):
        # z (batch_size, seq_len, d_model)
        # set_attention_mask (batch_size, seq_len)
        B, C, T = z.shape
        z = z.transpose(1, 2)  # (B, T, C)
        set_attention_mask = set_attention_mask.unsqueeze(1).expand(
            -1, C, -1
        )  # (B, C, T)
        z_masked = z.masked_fill(~set_attention_mask, -float("inf"))
        z_pooled = F.max_pool1d(z_masked, kernel_size=kernel_size)
        z_pooled = z_pooled.transpose(1, 2)  # (B, T // kernel_size, C)
        z_pooled = z_pooled.masked_fill(torch.isinf(z_pooled), 0.0)
        return z_pooled



########################################################################################################
## 1. Soft Contrastive Losses (masked-friendly)
########################################################################################################
# The functions below `inst_CL_soft`, `temp_CL_soft`, and `hier_CL_soft`
# are updated to support optional boolean masks for padded/invalid timesteps.
# Input tensors:
#   - z1, z2: (B, T, C)
#   - mask1, mask2: (B, T) boolean tensors (True = valid, False = pad). Optional.
# Behaviour:
#   - If mask is None, functions behave exactly like the original versions.
#   - If mask provided, pairwise similarities involving any invalid timestep are ignored
#     (their loss-weights set to zero and logits masked out before softmax).
#   - Downsampling of masks is handled consistently with max-pooling (kernel_size=2).
# Notes:
#   - We use a large negative filler (neg_inf) for masked logits so softmax effectively
#     assigns zero probability to those entries.
#   - Normalization is adapted to count only valid anchor-target pairs.

neg_inf = -1e9


def _downsample_mask(mask):
    # mask: (B, T) bool -> returns (B, T//2) bool after max-pooling window=2
    if mask is None:
        return None
    # Convert to float and apply max_pool1d to emulate logical OR in pairs
    m = mask.float().unsqueeze(1)  # B x 1 x T
    m_pooled = F.max_pool1d(m, kernel_size=2)  # B x 1 x T//2
    m_pooled = m_pooled.squeeze(1) > 0.5
    return m_pooled


def inst_CL_soft(z1, z2, soft_labels_L, soft_labels_R, mask1=None, mask2=None):
    """
    Instance-wise soft contrastive loss with optional masking.

    Args:
        z1, z2: (B, T, C)
        soft_labels_L, soft_labels_R: T x B x (2B-1) soft label weight tensors (as produced by dup_matrix)
        mask1, mask2: (B, T) boolean masks (True=valid). If None, no masking applied.

    Returns:
        scalar loss (torch.tensor)
    """
    B, T = z1.size(0), z1.size(1)
    if B == 1:
        return z1.new_tensor(0.)

    # Stack instances: 2B x T x C -> transpose -> T x 2B x C
    z = torch.cat([z1, z2], dim=0)  # 2B x T x C
    z = z.transpose(0, 1)  # T x 2B x C
    sim = torch.matmul(z, z.transpose(1, 2))  # T x 2B x 2B

    # Build per-time valid matrix: T x 2B
    if mask1 is None or mask2 is None:
        valid_t = torch.ones((T, 2 * B), dtype=torch.bool, device=z1.device)
    else:
        valid_mat = torch.cat([mask1, mask2], dim=0)    # 2B x T
        valid_t = valid_mat.t()                         # T x 2B

    # Build pairwise valid mask: T x 2B x 2B
    pair_mask = valid_t.unsqueeze(2) & valid_t.unsqueeze(1)  # True if both positions are valid

    # Build squeezed pair mask in same ordering as logits (remove diagonal column)
    mask_squeezed = torch.tril(pair_mask, diagonal=-1)[:, :, :-1] | torch.triu(pair_mask, diagonal=1)[:, :, 1:]
    # Build logits in the same squeezed form
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1] + torch.triu(sim, diagonal=1)[:, :, 1:]  # T x 2B x (2B-1)

    # Mask logits for invalid anchor-target pairs
    logits = logits.masked_fill(~mask_squeezed, neg_inf)

    # Compute negative log-softmax as in original implementation
    logits = -F.log_softmax(logits, dim=-1)

    # If soft labels are provided, ensure they are on same device
    if soft_labels_L is not None and soft_labels_R is not None:
        soft_labels_L = soft_labels_L.to(z1.device)
        soft_labels_R = soft_labels_R.to(z1.device)
        # Apply mask to soft labels so invalid targets get weight 0
        left_mask = mask_squeezed[:, :B, :]   # T x B x (2B-1)
        right_mask = mask_squeezed[:, B:, :]  # T x B x (2B-1)
        soft_labels_L = soft_labels_L * left_mask.float()
        soft_labels_R = soft_labels_R * right_mask.float()

        i = torch.arange(B, device=z1.device)
        loss_left = torch.sum(logits[:, i] * soft_labels_L)
        loss_right = torch.sum(logits[:, B + i] * soft_labels_R)
        loss = loss_left + loss_right

        # Normalize by the number of valid anchor-target pairs used
        n_valid = mask_squeezed.sum().clamp_min(1.0)  # total valid entries across T x 2B x (2B-1)
        loss = loss / n_valid
        return loss
    else:
        # If soft labels are None, fallback to original hard behavior cannot be performed here.
        # Return zero to indicate no soft-instance loss can be computed.
        return z1.new_tensor(0.)


def temp_CL_soft(z1, z2, timelag_L, timelag_R, mask1=None, mask2=None):
    """
    Temporal soft contrastive loss with optional masking.

    Args:
        z1, z2: (B, T, C)
        timelag_L, timelag_R: B x T x (2T-1) or T x (2T-1) depending on dup_matrix output.
                             We will make sure they are tensors on correct device.
        mask1, mask2: (B, T) boolean masks (True=valid). If None, no masking applied.

    Returns:
        scalar loss (torch.tensor)
    """
    B, T = z1.size(0), z1.size(1)
    if T == 1:
        return z1.new_tensor(0.)

    # Concatenate along temporal dimension: B x 2T x C
    z = torch.cat([z1, z2], dim=1)  # B x 2T x C
    sim = torch.matmul(z, z.transpose(1, 2))  # B x 2T x 2T

    # Build per-sample valid vector: B x 2T
    if mask1 is None or mask2 is None:
        valid = torch.ones((B, 2 * T), dtype=torch.bool, device=z1.device)
    else:
        valid = torch.cat([mask1, mask2], dim=1)  # B x 2T

    # Build pairwise valid mask: B x 2T x 2T
    pair_mask = valid.unsqueeze(1) & valid.unsqueeze(2)

    # Build squeezed mask and squeezed logits (remove diagonal column for each row)
    mask_squeezed = torch.tril(pair_mask, diagonal=-1)[:, :, :-1] | torch.triu(pair_mask, diagonal=1)[:, :, 1:]
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1] + torch.triu(sim, diagonal=1)[:, :, 1:]  # B x 2T x (2T-1)

    # Mask logits for invalid anchor-target pairs
    logits = logits.masked_fill(~mask_squeezed, neg_inf)

    # Negative log-softmax
    logits = -F.log_softmax(logits, dim=-1)

    # Ensure timelag tensors are on same device
    timelag_L = timelag_L.to(z1.device)
    timelag_R = timelag_R.to(z1.device)

    t = torch.arange(T, device=z1.device)

    # Apply mask to timelag weights so invalid targets contribute zero
    left_mask = mask_squeezed[:, :T, :]   # B x T x (2T-1)
    right_mask = mask_squeezed[:, T:, :]  # B x T x (2T-1)
    timelag_L = timelag_L * left_mask.float()
    timelag_R = timelag_R * right_mask.float()

    loss = torch.sum(logits[:, t] * timelag_L)
    loss += torch.sum(logits[:, T + t] * timelag_R)

    # Normalize by number of valid anchor-target pairs used
    n_valid = mask_squeezed.sum().clamp_min(1.0)  # total valid entries across B x 2T x (2T-1)
    loss = loss / n_valid
    return loss


#------------------------------------------------------------------------------------------#
# (3) Hierarchical CL = Instance CL + Temporal CL (masked-capable)
#------------------------------------------------------------------------------------------#
def hier_CL_soft(z1, z2, soft_labels=None, tau_temp=2, lambda_=0.5, temporal_unit=0,
                 soft_temporal=False, soft_instance=False, temporal_hierarchy=True,
                 mask1=None, mask2=None):
    """
    Hierarchical soft contrastive loss with optional masks for padded timesteps.

    Args:
      z1, z2: (B, T, C)
      soft_labels: soft label structure expected by dup_matrix (None or array/tensor)
      tau_temp, lambda_, temporal_unit, soft_temporal, soft_instance, temporal_hierarchy:
        same meaning as original function
      mask1, mask2: (B, T) boolean masks (True=valid). If None, no masking applied.

    Returns:
      scalar loss (torch.tensor)
    """
    if soft_labels is not None:
        soft_labels = torch.tensor(soft_labels, device=z1.device)
        soft_labels_L, soft_labels_R = dup_matrix(soft_labels)
    else:
        soft_labels_L = soft_labels_R = None

    loss = torch.tensor(0., device=z1.device)
    d = 0

    # Local copies of masks so we can downsample them alongside z1/z2
    cur_mask1 = mask1
    cur_mask2 = mask2

    while z1.size(1) > 1:
        if lambda_ != 0:
            if soft_instance:
                loss += lambda_ * inst_CL_soft(z1, z2, soft_labels_L, soft_labels_R, mask1=cur_mask1, mask2=cur_mask2)
            else:
                loss += lambda_ * inst_CL_hard(z1, z2)

        if d >= temporal_unit:
            if 1 - lambda_ != 0:
                if soft_temporal:
                    if temporal_hierarchy:
                        timelag = timelag_sigmoid(z1.shape[1], tau_temp * (2 ** d))
                    else:
                        timelag = timelag_sigmoid(z1.shape[1], tau_temp)
                    timelag = torch.tensor(timelag, device=z1.device)
                    timelag_L, timelag_R = dup_matrix(timelag)
                    loss += (1 - lambda_) * temp_CL_soft(z1, z2, timelag_L, timelag_R, mask1=cur_mask1, mask2=cur_mask2)
                else:
                    loss += (1 - lambda_) * temp_CL_hard(z1, z2)
        d += 1

        # Downsample z1, z2 via max pooling (as original)
        z1 = F.max_pool1d(z1.transpose(1, 2), kernel_size=2).transpose(1, 2)
        z2 = F.max_pool1d(z2.transpose(1, 2), kernel_size=2).transpose(1, 2)

        # Downsample masks in the same manner (logical OR over pooling window)
        cur_mask1 = _downsample_mask(cur_mask1)
        cur_mask2 = _downsample_mask(cur_mask2)

    if z1.size(1) == 1:
        if lambda_ != 0:
            if soft_instance:
                loss += lambda_ * inst_CL_soft(z1, z2, soft_labels_L, soft_labels_R, mask1=cur_mask1, mask2=cur_mask2)
            else:
                loss += lambda_ * inst_CL_hard(z1, z2)
        d += 1

    # Avoid division by zero (d should be >=1 in normal use)
    if d == 0:
        return loss
    return loss / d
