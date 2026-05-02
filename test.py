"""
test.py - Evaluate the trained PetClassifier on the official Oxford-IIIT
Pet 'test' split.

Usage:
    python test.py

Requires the trained weights file 'pet_classifier.pth' produced by train.py
to be present in the working directory.

Test-time pre-processing is restricted to operations that are strictly
necessary to feed the images into the network:
    * Resize to 224x224 (the network's expected input size),
    * ToTensor (necessary for tensor input),
    * Normalize with the same mean/std used at training time.
No augmentation, cropping, or other modification is applied.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet

from model import PetClassifier


BATCH_SIZE  = 64
NUM_CLASSES = 37
IMG_SIZE    = 224
NUM_WORKERS = 0
MODEL_PATH  = 'pet_classifier.pth'

# Must match the normalisation used during training.
IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD  = [0.229, 0.224, 0.225]

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader) -> float:
    model.eval()
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        preds = model(images).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return 100.0 * correct / total


def main():
    print(f'Device: {DEVICE}')

    test_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMG_MEAN, IMG_STD),
    ])

    test_set = OxfordIIITPet(
        root='./data',
        split='test',
        target_types='category',
        transform=test_transform,
        download=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )
    print(f'Test images: {len(test_set)}')

    model = PetClassifier(num_classes=NUM_CLASSES).to(DEVICE)
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state)

    accuracy = evaluate(model, test_loader)
    print(f'\nTest accuracy: {accuracy:.2f}%')


if __name__ == '__main__':
    main()
