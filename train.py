"""
train.py - Train the PetClassifier on the Oxford-IIIT Pet 'trainval' split.

Usage:
    python train.py
"""

import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet

from model import PetClassifier


# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Hyperparameters
EPOCHS         = 30
BATCH_SIZE     = 64
LEARNING_RATE  = 0.01
MOMENTUM       = 0.9
WEIGHT_DECAY   = 5e-4
NUM_CLASSES    = 37
IMG_SIZE       = 224
NUM_WORKERS    = 4

IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD  = [0.229, 0.224, 0.225]

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = 'pet_classifier.pth'


def main():
    print(f'Device: {DEVICE}')

    # Training data with basic augmentation
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(IMG_MEAN, IMG_STD),
    ])

    train_set = OxfordIIITPet(
        root='./data',
        split='trainval',
        target_types='category',
        transform=train_transform,
        download=True,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )
    print(f'Training images: {len(train_set)}')

    model = PetClassifier(num_classes=NUM_CLASSES).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
    )

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for images, labels in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            logits = model(images)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            preds = logits.argmax(dim=1)
            running_loss += loss.item() * labels.size(0)
            running_correct += (preds == labels).sum().item()
            running_total += labels.size(0)

        avg_loss = running_loss / running_total
        train_acc = 100.0 * running_correct / running_total
        print(f'Epoch {epoch + 1:02d}/{EPOCHS} | loss={avg_loss:.4f} | train-acc={train_acc:.2f}%')

    torch.save(model.state_dict(), MODEL_PATH)
    print(f'Saved trained model to "{MODEL_PATH}".')


if __name__ == '__main__':
    main()
