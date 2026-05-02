"""
model.py - Network architecture for the Oxford-IIIT Pet classifier.

Architecture is inspired by:
    He, K., Zhang, X., Ren, S., & Sun, J. (2016).
    Deep Residual Learning for Image Recognition. CVPR 2016.

The implementation here is written from scratch using only PyTorch primitives
(no torchvision.models, no pretrained weights). Variable names and module
structure differ from any reference implementation.
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

        # First convolution can subsample (stride > 1) and change channels.
        self.conv_a = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1, bias=False,
        )
        self.norm_a = nn.BatchNorm2d(out_channels)

        # Second convolution preserves shape.
        self.conv_b = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=False,
        )
        self.norm_b = nn.BatchNorm2d(out_channels)

        # Projection shortcut when shape changes; identity otherwise.
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
        y = F.relu(self.norm_a(self.conv_a(x)), inplace=True)
        y = self.norm_b(self.conv_b(y))
        y = y + residual
        return F.relu(y, inplace=True)


class PetClassifier(nn.Module):
    """ResNet-18-style network for 37-class fine-grained pet classification.

    Input:  3 x 224 x 224 image tensor.
    Output: 37-dim logits (softmax applied implicitly via CrossEntropyLoss).

    Stage layout (after the stem):
        stage 1: 2 x BasicBlock,  64 channels, output  56 x 56
        stage 2: 2 x BasicBlock, 128 channels, output  28 x 28
        stage 3: 2 x BasicBlock, 256 channels, output  14 x 14
        stage 4: 2 x BasicBlock, 512 channels, output   7 x  7

    A small dropout layer is applied to the pooled features before the
    classifier as an extra regulariser (helps reduce overfitting on a
    small dataset with high model capacity).
    """

    def __init__(self, num_classes: int = 37, dropout_p: float = 0.2):
        super().__init__()

        # Stem: aggressive downsampling to reach 56x56 spatial size quickly.
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # Four residual stages.
        self.stage_one   = self._make_stage(in_ch=64,  out_ch=64,  blocks=2, first_stride=1)
        self.stage_two   = self._make_stage(in_ch=64,  out_ch=128, blocks=2, first_stride=2)
        self.stage_three = self._make_stage(in_ch=128, out_ch=256, blocks=2, first_stride=2)
        self.stage_four  = self._make_stage(in_ch=256, out_ch=512, blocks=2, first_stride=2)

        # Head: global average pool, dropout, then linear classifier.
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=dropout_p)
        self.classifier = nn.Linear(512, num_classes)

        self._init_weights()

    @staticmethod
    def _make_stage(in_ch: int, out_ch: int, blocks: int, first_stride: int) -> nn.Sequential:
        layers = [BasicBlock(in_ch, out_ch, stride=first_stride)]
        for _ in range(blocks - 1):
            layers.append(BasicBlock(out_ch, out_ch, stride=1))
        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        # Kaiming init for conv layers (good fit for ReLU networks).
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage_one(x)
        x = self.stage_two(x)
        x = self.stage_three(x)
        x = self.stage_four(x)
        x = self.gap(x).flatten(1)
        x = self.dropout(x)
        return self.classifier(x)


def count_parameters(model: nn.Module) -> int:
    """Return the total number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # Sanity check: build the model and print parameter count.
    net = PetClassifier(num_classes=37)
    print(net)
    print(f'\nTotal trainable parameters: {count_parameters(net):,}')
    dummy = torch.randn(2, 3, 224, 224)
    out = net(dummy)
    print(f'Output shape for a batch of 2: {tuple(out.shape)}')
