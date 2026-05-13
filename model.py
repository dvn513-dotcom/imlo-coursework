"""ResNet-18-style architecture for the Oxford-IIIT Pet classifier.

Inspired by He et al., "Deep Residual Learning for Image Recognition" (2016).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """ResNet basic block: two 3x3 convs with a skip connection."""

    def __init__(self, in_channels, out_channels, stride=1):
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

    def forward(self, x):
        residual = self.shortcut(x)
        y = F.relu(self.norm_a(self.conv_a(x)), inplace=True)
        y = self.norm_b(self.conv_b(y))
        y = y + residual
        return F.relu(y, inplace=True)


class PetClassifier(nn.Module):
    """ResNet-18-style classifier for Oxford-IIIT Pet (37 classes)."""

    def __init__(self, num_classes=37, dropout_p=0.2):
        super().__init__()

        # stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # 4 residual stages
        self.stage_one   = self._make_stage(64,  64,  blocks=2, first_stride=1)
        self.stage_two   = self._make_stage(64,  128, blocks=2, first_stride=2)
        self.stage_three = self._make_stage(128, 256, blocks=2, first_stride=2)
        self.stage_four  = self._make_stage(256, 512, blocks=2, first_stride=2)

        # head
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=dropout_p)
        self.classifier = nn.Linear(512, num_classes)

        self._init_weights()

    @staticmethod
    def _make_stage(in_ch, out_ch, blocks, first_stride):
        layers = [BasicBlock(in_ch, out_ch, stride=first_stride)]
        for _ in range(blocks - 1):
            layers.append(BasicBlock(out_ch, out_ch, stride=1))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage_one(x)
        x = self.stage_two(x)
        x = self.stage_three(x)
        x = self.stage_four(x)
        x = self.gap(x).flatten(1)
        x = self.dropout(x)
        return self.classifier(x)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    net = PetClassifier(num_classes=37)
    print(net)
    print(f'\nTotal trainable parameters: {count_parameters(net):,}')
    out = net(torch.randn(1, 3, 224, 224))
    print(f'Output: {tuple(out.shape)}')
