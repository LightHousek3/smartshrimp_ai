"""Model factory; no external research-repository code or weights."""

import torch
from torch import nn
from torchvision import models


class SimAM(nn.Module):
    """Parameter-free spatial energy attention, independently expressed."""

    def forward(self, x):
        centered = x - x.mean(dim=(2, 3), keepdim=True)
        square = centered.square()
        variance = square.sum(dim=(2, 3), keepdim=True) / max(x.shape[2] * x.shape[3] - 1, 1)
        return x * torch.sigmoid(0.5 + square / (4 * (variance + 1e-4)))


def conv(cin, cout, stride=1):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, stride, 1, bias=False), nn.BatchNorm2d(cout), nn.SiLU()
    )


class CSPStage(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.down = conv(cin, cout, 2)
        self.branch = nn.Sequential(conv(cout // 2, cout // 2), conv(cout // 2, cout // 2))
        self.merge = conv(cout, cout)

    def forward(self, x):
        a, b = self.down(x).chunk(2, dim=1)
        return self.merge(torch.cat((a, b + self.branch(b)), dim=1))


class Classifier(nn.Module):
    def __init__(self, features, channels, attention):
        super().__init__()
        self.features = features
        self.attention = SimAM() if attention else nn.Identity()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(channels, 4)

    def forward(self, x):
        return self.head(self.pool(self.attention(self.features(x))).flatten(1))


def create_model(backbone: str, simam=False, pretrained=False) -> nn.Module:
    if backbone == "efficientnet_b0":
        base = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        )
        return Classifier(base.features, 1280, simam)
    if backbone == "mobilenet_v3_large":
        base = models.mobilenet_v3_large(
            weights=models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        )
        return Classifier(base.features, 960, simam)
    if pretrained:
        raise ValueError("Pretrained weights are supported only for torchvision backbones")
    if backbone == "yolo_csp_cls":
        features = nn.Sequential(
            conv(3, 16, 2), CSPStage(16, 32), CSPStage(32, 64), CSPStage(64, 128)
        )
        return Classifier(features, 128, simam)
    if backbone == "tiny":
        return Classifier(nn.Sequential(conv(3, 8, 4), conv(8, 16, 4)), 16, simam)
    raise ValueError(f"Unknown backbone: {backbone}")
