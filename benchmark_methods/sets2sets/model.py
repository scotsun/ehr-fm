import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class EncoderRNN(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.gru = nn.GRU(d_model, d_model, num_layers, batch_first=True)

    def _masked_avg_pool(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = x * mask.unsqueeze(-1)
        return x.sum(dim=-2) / mask.sum(dim=-1, keepdim=True).clamp(min=1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        set_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        emb = self.embedding(input_ids)  # (batch, seq_len, set_size, d_model)
        set_emb = self._masked_avg_pool(
            emb, attention_mask
        )  # (batch, seq_len, d_model)

        lengths = set_attention_mask.sum(dim=1).cpu()  # (batch)
        set_emb = pack_padded_sequence(
            set_emb, lengths, batch_first=True, enforce_sorted=False
        )
        output, hidden = self.gru(set_emb)
        output, _ = pad_packed_sequence(output, batch_first=True)

        return output, hidden

    def initHidden(self, batch_size: int = 1):
        return torch.zeros(
            self.gru.num_layers,
            batch_size,
            self.gru.hidden_size,
            device=next(self.parameters()).device,
        )


class AttnDecoderRNN(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, max_length, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Classic (Bahdanau) attention over encoder outputs
        self.attn = nn.Linear(d_model + d_model, max_length)
        self.attn_combine = nn.Linear(d_model + d_model, d_model)

        # Element-level attention (Between history_context and hidden)
        self.attn1 = nn.Linear(d_model + vocab_size, d_model)

        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(d_model, d_model, num_layers, batch_first=True)
        self.out = nn.Linear(d_model, vocab_size)

        # Output gating/mask layer for the output set
        self.attn_combine5 = nn.Linear(vocab_size, vocab_size)

    def _masked_avg_pool(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = x * mask.unsqueeze(-1)
        return x.sum(dim=-2) / mask.sum(dim=-1, keepdim=True).clamp(min=1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        hidden: torch.Tensor,
        encoder_outputs: torch.Tensor,
        history_context: torch.Tensor,
    ):
        """
        input_ids:       [batch, set_size] (last observed set)
        attention_mask:  [batch, set_size]
        hidden:          [num_layers, batch, d_model]
        encoder_outputs: [batch, seq_len, d_model]
        history_context: [batch, output_size]   (e.g. user frequency vector)
        Returns:
            output:      [batch, output_size]
            hidden:      [num_layers, batch, d_model]
            attn_weights: [batch, seq_len]
        """
        batch, set_size = input_ids.shape

        # 1. Embed input set and pool (average)
        embedded = self.embedding(input_ids)  # [batch, set_size, d_model]
        pooled = self._masked_avg_pool(embedded, attention_mask)  # [batch, 1, d_model]
        pooled = self.dropout(pooled)

        # 2. Compute attention weights for encoder outputs
        last_hidden = hidden[-1].unsqueeze(1)  # [batch, 1, hidden_size]
        attn_input = torch.cat((pooled, last_hidden), dim=2)  # [batch, 1, d_model*2]
        attn_scores = self.attn(attn_input).squeeze(1)  # [batch, seq_len]
        # TODO: add mask to attn_scores
        attn_weights = F.softmax(attn_scores, dim=1)  # [batch, seq_len]

        context = torch.bmm(
            attn_weights.unsqueeze(1), encoder_outputs
        )  # [batch, 1, d_model]

        # 3. Fuse pooled input set and context vector for GRU input
        gru_in = torch.cat((pooled, context), dim=2)  # [batch, 1, d_model*2]
        gru_in = F.relu(self.attn_combine(gru_in))  # [batch, 1, d_model]

        # 4. Run GRU
        output, hidden = self.gru(gru_in, hidden)  # output:[batch, 1, d_model]
        output = output.squeeze(1)  # [batch, d_model]

        # 5. Compute preliminary output
        logits = self.out(output)  # [batch, output_size]

        # 6. Output gating/mask (applies elementwise mask based on history_context)
        beta = history_context.clone()
        beta[history_context != 0] = 1
        gate_mask = torch.sigmoid(
            self.attn_combine5(history_context)
        )  # [batch, output_size]

        # 7. Combine logits and mask/gate (element-wise mix; can customize as needed)
        gated_logits = logits * (1 - beta * gate_mask) + history_context * gate_mask

        return gated_logits, hidden, attn_weights

    def initHidden(self, batch_size=1):
        return torch.zeros(
            self.num_layers,
            batch_size,
            self.hidden_size,
            device=next(self.parameters()).device,
        )


class Sets2Sets(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, max_length, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.enc = EncoderRNN(vocab_size, d_model, num_layers)
        self.dec = AttnDecoderRNN(vocab_size, d_model, num_layers, max_length, dropout)

    def forward(self, input_ids, attention_mask, set_attention_mask):
        batch_size = input_ids.size(0)
        # 1. Encode input set
        encoder_outputs, hidden = self.enc(
            input_ids, attention_mask, set_attention_mask
        )
        # 2. Decode output set
        lengths = set_attention_mask.sum(dim=1)
        dec_input_ids = input_ids[range(batch_size), lengths - 1, :]
        dec_attention_mask = attention_mask[range(batch_size), lengths - 1, :]
        # TODO:
        history_context = torch.zeros(batch_size, self.vocab_size)

        output_logits, hidden, attn_weights = self.dec(
            dec_input_ids,
            dec_attention_mask,
            hidden,
            encoder_outputs,
            history_context,
        )


class Sets2SetsLoss(nn.Module):
    def __init__(self, vocab_size: int, lambda_reg: float = 10.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.lambda_reg = lambda_reg

    def forward(self, pred, target, weights):
        """
        pred:   [1, vocab_size] or [batch, vocab_size]
        target: [1, set_size] or [batch, set_size]
        weights: [1, vocab_size] or [batch, vocab_size]
        """
        batch_size, vocab_size = pred.size(0), self.vocab_size
        # Convert target token id format to multi-hot
        multi_hot = torch.zeros(batch_size, vocab_size, dtype=torch.float32)
        multi_hot.scatter_(1, target, 1.0)

        # Weighted MSE loss, averaged over batch
        mseloss = (weights.view(-1, 1) * (pred - multi_hot).pow(2)).sum() / batch_size

        # Set loss per instance
        set_losses = []
        for i in range(batch_size):
            p = pred[i]  # [vocab_size]
            t = multi_hot[i]  # [vocab_size]

            pos_pred = p[t == 1]
            neg_pred = p[t == 0]

            diff = pos_pred.unsqueeze(1) - neg_pred.unsqueeze(0)  # [num_pos, num_neg]
            exp_loss = torch.exp(-diff).sum()
            norm_loss = exp_loss / (pos_pred.numel() * neg_pred.numel())
            set_losses.append(norm_loss)

        set_loss = torch.stack(set_losses).mean()  # mean over batch

        return mseloss + self.lambda_reg * set_loss
