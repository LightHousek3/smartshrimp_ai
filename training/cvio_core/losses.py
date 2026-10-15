# SPDX-License-Identifier: AGPL-3.0-only
# Adapted from CVIO fd64876cd2fd3acb24640b729d4f019996ebb39b; see NOTICE.
"""Single-label ASL with an LDAM true-class margin."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _reduce(values: torch.Tensor, reduction: str) -> torch.Tensor:
    if reduction == "mean":
        return values.mean()
    if reduction == "sum":
        return values.sum()
    if reduction == "none":
        return values
    raise ValueError(f"Unsupported reduction: {reduction}")


class ASLSingleLabel(nn.Module):
    """Official-style asymmetric loss adapted to single-label classification."""

    def __init__(
        self,
        gamma_pos: float = 0.0,
        gamma_neg: float = 4.0,
        label_smoothing: float = 0.1,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma_pos = float(gamma_pos)
        self.gamma_neg = float(gamma_neg)
        self.label_smoothing = float(label_smoothing)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 2:
            raise ValueError(f"Expected logits [B,C], got {tuple(logits.shape)}")
        targets = targets.long().view(-1).to(logits.device)
        work_logits = logits.float()
        log_probs = F.log_softmax(work_logits, dim=1)
        probs = log_probs.exp()
        one_hot = F.one_hot(targets, num_classes=logits.shape[1]).to(work_logits.dtype)
        anti_targets = 1.0 - one_hot
        xs_pos = probs * one_hot
        xs_neg = (1.0 - probs) * anti_targets
        gamma = self.gamma_pos * one_hot + self.gamma_neg * anti_targets
        weights = torch.clamp(1.0 - xs_pos - xs_neg, min=1e-8).pow(gamma)
        if self.label_smoothing > 0:
            one_hot = one_hot * (1.0 - self.label_smoothing)
            one_hot = one_hot + self.label_smoothing / logits.shape[1]
        loss = -(one_hot * log_probs * weights).sum(dim=1)
        return _reduce(loss, self.reduction)


class ASLLDAMLoss(nn.Module):
    """Apply an LDAM margin and scale before single-label ASL.

    Defaults match the recorded seed-42 paper configuration.
    """

    def __init__(
        self,
        class_counts: Sequence[int | float] = (282, 139, 229, 154),
        gamma_pos: float = 0.0,
        gamma_neg: float = 4.0,
        label_smoothing: float = 0.1,
        max_m: float = 0.5,
        scale: float = 30.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        counts = torch.as_tensor(class_counts, dtype=torch.float32)
        if counts.ndim != 1 or torch.any(counts <= 0):
            raise ValueError("class_counts must contain positive values")
        margins = 1.0 / torch.sqrt(torch.sqrt(counts))
        margins = margins * (float(max_m) / margins.max().clamp_min(1e-12))
        self.register_buffer("margins", margins)
        self.scale = float(scale)
        self.asl = ASLSingleLabel(
            gamma_pos=gamma_pos,
            gamma_neg=gamma_neg,
            label_smoothing=label_smoothing,
            reduction=reduction,
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 2 or logits.shape[1] != self.margins.numel():
            raise ValueError(
                f"Expected logits [B,{self.margins.numel()}], got {tuple(logits.shape)}"
            )
        targets = targets.long().view(-1).to(logits.device)
        margins = self.margins.to(device=logits.device, dtype=logits.dtype)
        one_hot = F.one_hot(targets, num_classes=logits.shape[1]).to(logits.dtype)
        adjusted = self.scale * (logits - one_hot * margins.view(1, -1))
        return self.asl(adjusted, targets)
