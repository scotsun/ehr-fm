import torch
import torch.nn as nn
import torch.nn.functional as F

from tslearn.metrics import soft_dtw


NEG_INF = -1e9


class SoftCLT(nn.Module):
    """adapted from softclt github repo"""

    def __init__(self, tau_temp, tau_inst, lambda_, alpha):
        super().__init__()
        self.tau_temp = tau_temp
        self.tau_inst = tau_inst
        self.lambda_ = lambda_
        self.alpha = alpha

    def forward(self, z1, z2, mask, x):
        dist_mat = self.soft_dtw_mat(x, mask)  # (B, B)
        soft_labels = 2 * self.alpha * F.sigmoid(dist_mat / self.tau_inst)  # (B, B)
        out = self.hier_CL_soft(
            z1=z1,
            z2=z2,
            mask1=mask,
            mask2=mask,
            soft_labels=soft_labels,
            tau_temp=self.tau_temp,
            lambda_=self.lambda_,
        )
        return out

    def soft_dtw_mat(self, x, mask):
        # h: (batch, max_seq, max_set_size, hidden_size)
        # mask: (batch, max_seq)
        B = x.shape[0]
        out = torch.zeros((B, B)).to(x.device)
        for i in range(B):
            for j in range(i, B):
                out[i, j] = soft_dtw(
                    x[i][mask[i]],
                    x[j][mask[j]],
                    gamma=0.1,
                    be="pytorch",
                    compute_with_backend=True,
                )
                out[j, i] = out[i, j]
        return out

    def masked_max_pool1d(self, z, mask, kernel_size):
        # z (batch_size, seq_len, d_model)
        # mask (batch_size, seq_len) <- set_attention_mask
        _, _, C = z.shape
        z = z.transpose(1, 2)  # (B, C, T)
        mask = mask.unsqueeze(1).expand(-1, C, -1)  # (B, C, T)
        z_masked = z.masked_fill(~mask, -float("inf"))
        z_pooled = F.max_pool1d(z_masked, kernel_size=kernel_size)
        z_pooled = z_pooled.transpose(1, 2)  # (B, T // kernel_size, C)
        z_pooled = z_pooled.masked_fill(torch.isinf(z_pooled), 0.0)
        return z_pooled

    def inst_CL_soft(
        self,
        z1,
        z2,
        soft_labels_L,
        soft_labels_R,
        mask1,
        mask2,
    ):
        """
        Instance-wise soft contrastive loss with masking.

        Args:
            z1, z2: (B, T, C)
            soft_labels_L, soft_labels_R: T x B x (2B-1) soft label weight tensors (as produced by dup_matrix)
            mask1, mask2: (B, T) boolean masks (True=valid). If None, no masking applied.

        Returns:
            scalar loss (torch.tensor)
        """
        B = z1.size(0)
        if B == 1:
            return z1.new_tensor(0.0)

        # Stack instances: 2B x T x C -> transpose -> T x 2B x C
        z = torch.cat([z1, z2], dim=0)  # 2B x T x C
        z = z.transpose(0, 1)  # T x 2B x C
        sim = torch.matmul(z, z.transpose(1, 2))  # T x 2B x 2B

        # Build per-time valid matrix: T x 2B
        valid_mat = torch.cat([mask1, mask2], dim=0)  # 2B x T
        valid_t = valid_mat.t()  # T x 2B

        # Build pairwise valid mask: T x 2B x 2B, True if both positions are valid
        pair_mask = valid_t.unsqueeze(2) & valid_t.unsqueeze(1)

        # Build squeezed pair mask in same ordering as logits (remove diagonal column)
        mask_squeezed = (
            torch.tril(pair_mask, diagonal=-1)[:, :, :-1]
            | torch.triu(pair_mask, diagonal=1)[:, :, 1:]
        )
        # Build logits in the same squeezed form (T x 2B x (2B-1))
        logits = (
            torch.tril(sim, diagonal=-1)[:, :, :-1]
            + torch.triu(sim, diagonal=1)[:, :, 1:]
        )

        # Mask logits for invalid anchor-target pairs
        logits = logits.masked_fill(~mask_squeezed, -1e9)
        # Compute negative log-softmax
        logits = -F.log_softmax(logits, dim=-1)

        # Apply mask to soft labels so invalid targets get weight 0
        left_mask = mask_squeezed[:, :B, :]  # T x B x (2B-1)
        right_mask = mask_squeezed[:, B:, :]  # T x B x (2B-1)
        soft_labels_L = soft_labels_L * left_mask.float()
        soft_labels_R = soft_labels_R * right_mask.float()

        i = torch.arange(B, device=z1.device)
        loss = torch.sum(logits[:, i] * soft_labels_L)
        loss += torch.sum(logits[:, B + i] * soft_labels_R)

        # Normalize by the number of valid anchor-target pairs used
        # total valid entries across T x 2B x (2B-1)
        n_valid = mask_squeezed.sum().clamp_min(1.0)
        loss = loss / n_valid
        return loss

    def temp_CL_soft(
        self,
        z1,
        z2,
        timelag_L,
        timelag_R,
        mask1,
        mask2,
    ):
        """
        Temporal soft contrastive loss with masking.

        Args:
            z1, z2: (B, T, C)
            timelag_L, timelag_R: T x B x (2T-1) timelag weight tensors (as produced by dup_matrix)
            mask1, mask2: (B, T) boolean masks (True=valid). If None, no masking applied.

        Returns:
            scalar loss (torch.tensor)
        """
        T = z1.size(1)
        if T == 1:
            return z1.new_tensor(0.0)

        # Concatenate along temporal dimension: B x 2T x C
        z = torch.cat([z1, z2], dim=1)  # B x 2T x C
        sim = torch.matmul(z, z.transpose(1, 2))  # B x 2T x 2T

        # Build per-sample valid vector: B x 2T
        valid = torch.cat([mask1, mask2], dim=1)  # B x 2T

        # Build pairwise valid mask: B x 2T x 2T
        pair_mask = valid.unsqueeze(1) & valid.unsqueeze(2)

        # Build squeezed mask and squeezed logits (remove diagonal column for each row)
        mask_squeezed = (
            torch.tril(pair_mask, diagonal=-1)[:, :, :-1]
            | torch.triu(pair_mask, diagonal=1)[:, :, 1:]
        )
        logits = (
            torch.tril(sim, diagonal=-1)[:, :, :-1]
            + torch.triu(sim, diagonal=1)[:, :, 1:]
        )  # B x 2T x (2T-1)

        # Mask logits for invalid anchor-target pairs
        logits = logits.masked_fill(~mask_squeezed, -1e9)
        # Negative log-softmax
        logits = -F.log_softmax(logits, dim=-1)

        # Apply mask to timelag weights so invalid targets contribute zero
        left_mask = mask_squeezed[:, :T, :]  # B x T x (2T-1)
        right_mask = mask_squeezed[:, T:, :]  # B x T x (2T-1)
        timelag_L = timelag_L * left_mask.float()
        timelag_R = timelag_R * right_mask.float()

        t = torch.arange(T, device=z1.device)
        loss = torch.sum(logits[:, t] * timelag_L)
        loss += torch.sum(logits[:, T + t] * timelag_R)

        # Normalize by number of valid anchor-target pairs used
        # total valid entries across B x 2T x (2T-1)
        n_valid = mask_squeezed.sum().clamp_min(1.0)
        loss = loss / n_valid
        return loss

    def hier_CL_soft(
        self,
        z1,
        z2,
        mask1,
        mask2,
        soft_labels,
        tau_temp=2,
        lambda_=0.5,
        temporal_unit=0,
        temporal_hierarchy=True,
    ):
        """
        Hierarchical soft contrastive loss with optional masks for padded timesteps.

        Args:
        z1, z2: (B, T, C)
        mask1, mask2: (B, T) boolean masks (True=valid)
        soft_labels (B, B): soft label structure expected by dup_matrix (None or array/tensor)
        tau_temp, lambda_, temporal_unit, soft_temporal, soft_instance, temporal_hierarchy:
            same meaning as original function

        Returns:
        scalar loss (torch.tensor)
        """
        soft_labels_L, soft_labels_R = dup_matrix(soft_labels)

        loss = torch.tensor(0.0, device=z1.device)
        d = 0

        # Local copies of masks so we can downsample them alongside z1/z2
        cur_mask1 = mask1
        cur_mask2 = mask2

        while z1.size(1) > 1:
            if lambda_ != 0:
                loss += lambda_ * self.inst_CL_soft(
                    z1,
                    z2,
                    soft_labels_L,
                    soft_labels_R,
                    mask1=cur_mask1,
                    mask2=cur_mask2,
                )

            if d >= temporal_unit:
                if 1 - lambda_ != 0:
                    if temporal_hierarchy:
                        timelag = timelag_sigmoid(
                            z1.shape[1], z1.device, tau_temp * (2**d)
                        )
                    else:
                        timelag = timelag_sigmoid(z1.shape[1], z1.device, tau_temp)
                    timelag_L, timelag_R = dup_matrix(timelag)
                    loss += (1 - lambda_) * self.temp_CL_soft(
                        z1,
                        z2,
                        timelag_L,
                        timelag_R,
                        mask1=cur_mask1,
                        mask2=cur_mask2,
                    )
            d += 1

            # Downsample z1, z2 via max pooling (as original)
            z1 = self.masked_max_pool1d(z1, cur_mask1, kernel_size=2)
            z2 = self.masked_max_pool1d(z2, cur_mask2, kernel_size=2)

            # Downsample masks in the same manner (logical OR over pooling window)
            cur_mask1 = _downsample_mask(cur_mask1)
            cur_mask2 = _downsample_mask(cur_mask2)

        if z1.size(1) == 1:
            if lambda_ != 0:
                loss += lambda_ * self.inst_CL_soft(
                    z1,
                    z2,
                    soft_labels_L,
                    soft_labels_R,
                    mask1=cur_mask1,
                    mask2=cur_mask2,
                )
            d += 1

        # Avoid division by zero (d should be >=1 in normal use)
        if d == 0:
            return loss
        return loss / d


####################
# helper functions #
####################


def _downsample_mask(mask):
    # mask: (B, T) bool -> returns (B, T//2) bool after max-pooling window=2
    if mask is None:
        return None
    # Convert to float and apply max_pool1d to emulate logical OR in pairs
    m = mask.float().unsqueeze(1)  # B x 1 x T
    m_pooled = F.max_pool1d(m, kernel_size=2)  # B x 1 x T//2
    m_pooled = m_pooled.squeeze(1) > 0.5
    return m_pooled


def dup_matrix(mat):
    mat0 = torch.tril(mat, diagonal=-1)[:, :-1]
    mat0 += torch.triu(mat, diagonal=1)[:, 1:]
    mat1 = torch.cat([mat0, mat], dim=1)
    mat2 = torch.cat([mat, mat0], dim=1)
    return mat1, mat2


def timelag_sigmoid(T, device, sigma=1):
    dist = torch.arange(T).to(device)
    dist = torch.abs(dist - dist[:, None])
    matrix = 2 / (1 + torch.exp(dist.float() * sigma))
    matrix = torch.where(matrix < 1e-6, 0, matrix)  # set very small values to 0
    return matrix


if __name__ == "__main__":
    softclt = SoftCLT(tau_inst=1, tau_temp=0.1, lambda_=0.5, alpha=0.5)
    x = torch.randn(2, 10, 128, requires_grad=True).cuda()
    x.retain_grad()
    z1 = torch.randn(2, 10, 128).cuda()
    z2 = z1 + 0.01
    mask = torch.ones(2, 10, dtype=bool).cuda()

    print(softclt.soft_dtw_mat(x, mask))
    loss = softclt(z1, z2, mask, x)
    loss.backward()
    print(x.grad)
