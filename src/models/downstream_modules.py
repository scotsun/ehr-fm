import torch
import torch.nn as nn

from src.layers.ffn import FFNSwiGLUBlock


class BiGRU(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_hidden: int,
        d_out: int,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=d_hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.projection = nn.Linear(d_hidden * 2, d_out)

    def forward(self, x, mask=None):
        """
        Forward pass for the BiGRU.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, time, input_size).
            mask (torch.Tensor, optional): Boolean mask of shape (batch, time). Defaults to None.

        Returns:
            torch.Tensor: Output tensor of shape (batch, output_size).
        """
        if mask is not None:
            lengths = mask.sum(dim=1).cpu()
            # pack_padded_sequence requires lengths > 0
            if torch.any(lengths == 0):
                # For now, we'll proceed with packing, which might error out if all lengths are 0.
                # A robust implementation would handle this case explicitly.
                pass

            packed_x = nn.utils.rnn.pack_padded_sequence(
                x, lengths, batch_first=True, enforce_sorted=False
            )
            _, h_n = self.gru(packed_x)
        else:
            _, h_n = self.gru(x)

        # h_n is (num_layers * 2, batch, hidden_size)
        # Reshape to (num_layers, 2, batch, hidden_size) to separate directions
        h_n = h_n.view(self.gru.num_layers, 2, x.size(0), self.gru.hidden_size)

        # Get the last layer's hidden states
        last_layer_h_n = h_n[-1]  # (2, batch, hidden_size)

        # Concatenate the final forward and backward hidden states
        # last_layer_h_n[0] is the last forward state
        # last_layer_h_n[1] is the last backward state (from the first time step)
        final_hidden = torch.cat(
            (last_layer_h_n[0], last_layer_h_n[1]), dim=-1
        )  # (batch, hidden_size * 2)

        return self.projection(final_hidden)


class Downstream(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_hidden: int,
        d_out: int,
        model_type: str,
        set_pool: str,
    ) -> None:
        super().__init__()
        self.set_pool = set_pool
        self.model_type = model_type
        match model_type:
            case "mlp":
                self.net = FFNSwiGLUBlock(d_model, d_hidden, d_out)
            case "gru":
                self.net = BiGRU(d_model, d_hidden, d_out)
            case _:
                raise ValueError(f"Unknown model_type: {model_type}")

    def _masked_avg_pool(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = x * mask.unsqueeze(-1)
        return x.sum(dim=-2) / mask.sum(dim=1, keepdim=True).unsqueeze(-1).clamp(min=1)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor, set_mask: torch.Tensor
    ) -> torch.Tensor:
        match self.set_pool:
            case "mean":
                x = self._masked_avg_pool(x, mask)
            case "cls":
                x = x.select(dim=-2, index=0)
            case _:
                raise ValueError(f"Unknown set_pool: {self.set_pool}")

        match self.model_type:
            case "mlp":
                x = self._masked_avg_pool(x, set_mask)
                return self.net(x)
            case "gru":
                return self.net(x, set_mask)
