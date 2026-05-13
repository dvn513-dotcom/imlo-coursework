"""Evaluate trained PetClassifier on the Oxford-IIIT Pet test split."""

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

# match training normalisation
IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD  = [0.229, 0.224, 0.225]

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
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
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

    accuracy = evaluate(model, test_loader)
    print(f'\nTest accuracy: {accuracy:.2f}%')


if __name__ == '__main__':
    main()
