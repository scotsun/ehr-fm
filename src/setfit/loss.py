# https://github.com/huggingface/sentence-transformers/blob/main/sentence_transformers/losses/CosineSimilarityLoss.py
# https://github.com/huggingface/sentence-transformers/blob/main/sentence_transformers/losses/CoSENTLoss.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class CosineSimilarityLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.cosine_similarity = nn.CosineSimilarity(dim=-1)

    def forward(
        self, embedding_a: torch.Tensor, embedding_b: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        sim = self.cosine_similarity(embedding_a, embedding_b)
        loss = F.mse_loss(sim, labels)
        return loss


class CoSENTLoss(nn.Module):
    def __init__(self, scale: float = 20):
        super().__init__()
        self.scale = scale
        self.cosine_similarity = nn.CosineSimilarity(dim=-1)

    def forward(
        self, embedding_a: torch.Tensor, embedding_b: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        scores = self.cosine_similarity(embedding_a, embedding_b)
        scores = scores * self.scale
        scores = scores[:, None] - scores[None, :]

        # label matrix indicating which pairs are relevant
        labels = labels[:, None] < labels[None, :]
        labels = labels.float()

        # mask out irrelevant pairs so they are negligible after exp()
        scores = scores - (1 - labels) * 1e12

        # append a zero as e^0 = 1
        scores = torch.cat((torch.zeros(1).to(scores.device), scores.view(-1)), dim=0)
        loss = torch.logsumexp(scores, dim=0)

        return loss
