"""
train.py - Train the PetClassifier from scratch on the Oxford-IIIT Pet
'trainval' split.

Usage:
    python train.py

No command-line arguments are required.

Recipe (CPU-friendly):
    * Architecture: custom ResNet-18-style network (see model.py).
    * Loss: standard CrossEntropyLoss.
    * Optimiser: AdamW (lr=1e-3, weight_decay=1e-4).
    * Schedule: 3-epoch linear warmup then cosine decay over the
      remaining 27 epochs.
    * Augmentation: RandomResizedCrop, RandomHorizontalFlip,
      ColorJitter - all PyTorch built-ins.
    * Training data: full official 'trainval' split (no validation split).
"""

import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet

from model import PetClassifier, count_parameters


# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# ---------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------
EPOCHS         = 30
BATCH_SIZE     = 64
INITIAL_LR     = 1e-3
WEIGHT_DECAY   = 1e-4
WARMUP_EPOCHS  = 3
NUM_CLASSES    = 37
IMG_SIZE       = 224
NUM_WORKERS    = 0

IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD  = [0.229, 0.224, 0.225]

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = 'pet_classifier.pth'


# ---------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------
def build_train_loader():
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
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
        pin_memory=False,
        drop_last=True,
    )
    return train_loader, train_set


def build_eval_train_loader():
    eval_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMG_MEAN, IMG_STD),
    ])
    eval_set = OxfordIIITPet(
        root='./data',
        split='trainval',
        target_types='category',
        transform=eval_transform,
        download=False,
    )
    eval_loader = DataLoader(
        eval_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )
    return eval_loader


# ---------------------------------------------------------------------
# Scheduler: linear warmup then cosine decay
# ---------------------------------------------------------------------
def make_scheduler(optimizer, total_epochs: int, warmup_epochs: int):
    def lr_factor(epoch: int) -> float:
        if epoch < warmup_epochs:
            return float(epoch + 1) / max(1, warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1.0 + float(np.cos(np.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_factor)


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------
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


# ---------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------
def main():
    print(f'Device: {DEVICE}')

    train_loader, train_set = build_train_loader()
    eval_train_loader = build_eval_train_loader()
    print(f'Training images (trainval): {len(train_set)}')

    model = PetClassifier(num_classes=NUM_CLASSES).to(DEVICE)
    print(f'Total trainable parameters: {count_parameters(model):,}')

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=INITIAL_LR,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = make_scheduler(optimizer, EPOCHS, WARMUP_EPOCHS)

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for images, labels in train_loader:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            logits = model(images)
            loss = criterion(logits, labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            preds = logits.argmax(dim=1)
            running_loss += loss.item() * labels.size(0)
            running_correct += (preds == labels).sum().item()
            running_total += labels.size(0)

        scheduler.step()

        avg_loss = running_loss / max(1, running_total)
        train_acc = 100.0 * running_correct / max(1, running_total)
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch {epoch + 1:02d}/{EPOCHS} | lr={current_lr:.5f} '
              f'| loss={avg_loss:.4f} | train-acc={train_acc:.2f}%')

    # Final clean training-set accuracy (no augmentation)
    final_train_acc = evaluate(model, eval_train_loader)
    print(f'\nFinal training-set accuracy (un-augmented): {final_train_acc:.2f}%')

    torch.save(model.state_dict(), MODEL_PATH)
    print(f'Saved trained model weights to "{MODEL_PATH}".')


if __name__ == '__main__':
    main()
