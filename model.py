"""
model.py - Network architecture for the Oxford-IIIT Pet classifier.

To be implemented.
"""

import torch.nn as nn


class PetClassifier(nn.Module):
    """Custom CNN classifier for the Oxford-IIIT Pet dataset (37 classes)."""

    def __init__(self, num_classes: int = 37):
        super().__init__()
        # TODO: define network layers
        pass

    def forward(self, x):
        # TODO: implement forward pass
        raise NotImplementedError
