 ## Goal analyze if [CLS] embedding captures context tokens {x1, x2,...,x3}'s representation

Let [CLS] embedding (`last_hidden_states`) be c and context embeddings be H

Normalize the c to make it a unit vector: `c* = c / norm(c)`

now treat c as the loading in PCA, calculate project scores, s, aka "PC": `s = Hc*`

`total var(H) = tr(Sigma) = sum(eigen)`

`var(s) = var(Hc*) = c*.T @ Sigma @ c*`


in PCA, a PC's contribution is `eigen_k / sum(eigen)`, if c* is a "good" loading, `c*.T @ Sigma @ c* -> eigen_1`.

so we can calculate `var(s) / tr(var(H))` as the contribution


we can start with linear assumption for PCA; if there isn't obvious partern, we can try RBF kernels



```python
import torch

def metric(cls_reprs, token_reprs):
    scores = []

    for c_i, H_i in zip(cls_reprs, token_reprs):
        # Normalize c
        c = c_i / torch.norm(c_i)

        # Projection of tokens onto c
        s = H_i @ c  # shape: (num_tokens,)

        # Explained variance (unbiased)
        explained_var = torch.var(s, unbiased=True)

        # Total variance across all dimensions (sum of per-dim variances)
        total_var = torch.var(H_i, dim=0, unbiased=True).sum()

        # Ratio
        if total_var > 0:
            scores.append(explained_var / total_var)
        else:
            scores.append(torch.tensor(0.0, device=H_i.device))

    return torch.stack(scores).mean()
```