import torch
from torch import nn
from torch.nn import functional as F


class LDAMLoss(nn.Module):
    """Single-label class-count margin, followed by weighted cross entropy."""

    def __init__(self, counts, max_margin=0.5, scale=10.0):
        super().__init__()
        counts = torch.as_tensor(counts, dtype=torch.float32)
        if (counts <= 0).any():
            raise ValueError("Every class needs training examples")
        margin = counts.pow(-0.25)
        self.register_buffer("margins", margin * (max_margin / margin.max()))
        self.register_buffer("weights", counts.sum() / (len(counts) * counts))
        self.scale = scale

    def forward(self, logits, target):
        adjusted = logits - F.one_hot(target, logits.shape[1]) * self.margins
        return F.cross_entropy(adjusted * self.scale, target, weight=self.weights)


def make_loss(name, counts):
    if name == "ldam":
        return LDAMLoss(counts)
    if name == "weighted_ce":
        counts = torch.tensor(counts, dtype=torch.float32)
        return nn.CrossEntropyLoss(weight=counts.sum() / (len(counts) * counts))
    raise ValueError(f"Unknown loss {name}")
