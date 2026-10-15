# SPDX-License-Identifier: AGPL-3.0-only
# Adapted from CVIO fd64876cd2fd3acb24640b729d4f019996ebb39b; see NOTICE.
"""SimAM gated residual with disease-contrast feature recalibration."""

from __future__ import annotations

import torch
import torch.nn as nn


class SimAMDCFR(nn.Module):
    """Parameter-light SimAM plus depthwise texture recalibration."""

    def __init__(self, channels: int, e_lambda: float = 1e-4) -> None:
        super().__init__()
        self.e_lambda = float(e_lambda)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.texture = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                groups=max(1, channels),
                bias=False,
            ),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected NCHW tensor, got {tuple(x.shape)}")
        height, width = x.shape[-2:]
        n = max(1, height * width - 1)
        squared = (x - x.mean(dim=(2, 3), keepdim=True)).pow(2)
        variance = squared.sum(dim=(2, 3), keepdim=True) / n
        simam = x * torch.sigmoid(squared / (4.0 * (variance + self.e_lambda)) + 0.5)
        texture = x * self.texture(x)
        return x + self.gate(x) * (simam + texture) * 0.5
