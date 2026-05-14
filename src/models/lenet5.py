"""
LeNet-5 (LeCun et al., 1998) — classic convolutional network for visual recognition.

Original paper used 32×32 grayscale MNIST digits. Here we use the same *layer pattern*
(conv → pool → conv → pool → fully connected stack) adapted to **RGB CIFAR-10**
(32×32×3), which matches common course assignments and keeps the architectural story
intact: local receptive fields, shared weights, and spatial subsampling.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LeNet5CIFAR(nn.Module):
    """
    LeNet-5-style CNN for 32×32 RGB inputs (e.g. CIFAR-10).

    Layout (mapped to classic naming where applicable):
      C1: 3 → 6 channels, 5×5 conv (learns low-level edges / color blobs)
      S2: 2×2 average pooling (subsampling / translation tolerance)
      C3: 6 → 16 channels, 5×5 conv (combinations of earlier features)
      S4: 2×2 average pooling
      F5: flatten then Linear(400 → 120) — “fully connected” stage begins
      F6: Linear(120 → 84)
      Out: Linear(84 → num_classes)

    Tensor shapes for 32×32 input:
      [B, 3, 32, 32] → conv1 → [B, 6, 28, 28] → pool → [B, 6, 14, 14]
      → conv2 → [B, 16, 10, 10] → pool → [B, 16, 5, 5] → 400-dim vector.
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        # Feature extractor: two conv blocks + pooling (the “LeNet” trunk).
        self.features = nn.Sequential(
            # C1: preserve a bit more context than strict historical padding;
            # we use *valid* conv (no padding) so spatial sizes match the textbook derivation.
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        # Classifier head: three linear layers like the original LeNet-5 FC stages.
        self.classifier = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(inplace=True),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Linear(84, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
