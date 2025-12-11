import torch
import torch.nn as nn
import torch.nn.functional as F


class SimCSE(nn.Module):
    def __init__(self, model, temperature):
        super().__init__()
        self.model = model
        self.temperature = temperature

    def forward(self, input_ids, attention_mask, segment_attention_mask):
        h1 = self.model.encode(input_ids, attention_mask, segment_attention_mask)[
            :, :, 0, :
        ]
        h2 = self.model.encode(input_ids, attention_mask, segment_attention_mask)[
            :, :, 0, :
        ]
        h1, h2 = h1.view(-1, h1.shape[-1]), h2.view(-1, h2.shape[-1])
        # h (2*batch_size, d_model)
        h = F.normalize(torch.cat([h1, h2], dim=0), dim=1)
        # sim (2*batch_size, 2*batch_size)
        sim = torch.einsum("nd,md->nm", h, h)
        mask = torch.eye(sim.shape[0], device=sim.device).bool()
        # Use -1e4 instead of -1e9 to avoid FP16 overflow in AMP
        sim = sim.masked_fill(mask, -1e4)

        targets = torch.arange(sim.shape[0] // 2, device=sim.device) + sim.shape[0] // 2
        targets = torch.cat([targets, targets - sim.shape[0] // 2], dim=0)

        loss = F.cross_entropy(sim / self.temperature, targets)
        return loss
