"""
model.py - Network architecture for the Oxford-IIIT Pet classifier.

Architecture is inspired by:
    He, K., Zhang, X., Ren, S., & Sun, J. (2016).
    Deep Residual Learning for Image Recognition. CVPR 2016.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """A residual block with two 3x3 convolutions and a shortcut connection.

    The shortcut is an identity mapping when the input/output shapes match,
    or a 1x1 projection (with stride) when they do not.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()

        self.conv_a = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1, bias=False,
        )
        self.norm_a = nn.BatchNorm2d(out_channels)

        self.conv_b = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=False,
        )
        self.norm_b = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        y = F.relu(self.norm_a(self.conv_a(x)))
        y = self.norm_b(self.conv_b(y))
        y = y + residual
        return F.relu(y)


class PetClassifier(nn.Module):
    """Custom CNN classifier for the Oxford-IIIT Pet dataset (37 classes)."""

    def __init__(self, num_classes: int = 37):
        super().__init__()
        # TODO: stack BasicBlocks into a full ResNet-18 architecture
        pass

    def forward(self, x):
        raise NotImplementedError
