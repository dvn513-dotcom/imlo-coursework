"""
train.py - Train the PetClassifier from scratch on the Oxford-IIIT Pet
'trainval' split.

Usage:
    python train.py

Key choices:
    * Architecture: custom ResNet-18-style network (see model.py).
    * Loss: CrossEntropyLoss with label smoothing 0.1.
    * Regularisation: Mixup (alpha=0.1) - reduced from 0.2 to ease training.
    * Optimiser: SGD with Nesterov momentum 0.9, weight decay 5e-4.
    * Schedule: 5-epoch linear warmup followed by 25-epoch cosine decay.
    * LR reduced from 0.1 to 0.03 since 0.1 was unstable for this dataset size.
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
EPOCHS           = 30
BATCH_SIZE       = 64
INITIAL_LR       = 0.03      # reduced from 0.1
MOMENTUM         = 0.9
WEIGHT_DECAY     = 5e-4
WARMUP_EPOCHS    = 5
LABEL_SMOOTHING  = 0.1
MIXUP_ALPHA      = 0.1       # reduced from 0.2
NUM_CLASSES      = 37
IMG_SIZE         = 224
NUM_WORKERS      = 4

IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD  = [0.229, 0.224, 0.225]

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = 'pet_classifier.pth'


# ---------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------
def build_train_loader() -> DataLoader:
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.TrivialAugmentWide(),
        transforms.ToTensor(),
        transforms.Normalize(IMG_MEAN, IMG_STD),
        transforms.RandomErasing(p=0.25),
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
        persistent_workers=(NUM_WORKERS > 0),
    )
    return train_loader, train_set


def build_eval_train_loader() -> DataLoader:
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
        pin_memory=True,
    )
    return eval_loader


# ---------------------------------------------------------------------
# Mixup
# ---------------------------------------------------------------------
def mixup_batch(images: torch.Tensor, labels: torch.Tensor, alpha: float):
    if alpha <= 0.0:
        return images, labels, labels, 1.0
    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(images.size(0), device=images.device)
    mixed = lam * images + (1.0 - lam) * images[perm]
    return mixed, labels, labels[perm], lam


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

    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=INITIAL_LR,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
        nesterov=True,
    )
    scheduler = make_scheduler(optimizer, EPOCHS, WARMUP_EPOCHS)

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        running_correct = 0.0
        running_total = 0

        for images, labels in train_loader:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            mixed, y_a, y_b, lam = mixup_batch(images, labels, MIXUP_ALPHA)
            logits = model(mixed)
            loss = lam * criterion(logits, y_a) + (1.0 - lam) * criterion(logits, y_b)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            preds = logits.argmax(dim=1)
            running_loss += loss.item() * labels.size(0)
            running_correct += (lam * (preds == y_a).sum().item()
                                + (1.0 - lam) * (preds == y_b).sum().item())
            running_total += labels.size(0)

        scheduler.step()

        avg_loss = running_loss / max(1, running_total)
        approx_acc = 100.0 * running_correct / max(1, running_total)
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch {epoch + 1:02d}/{EPOCHS} | lr={current_lr:.4f} '
              f'| loss={avg_loss:.4f} | approx-train-acc={approx_acc:.2f}%')

    final_train_acc = evaluate(model, eval_train_loader)
    print(f'\nFinal training-set accuracy (un-augmented): {final_train_acc:.2f}%')

    torch.save(model.state_dict(), MODEL_PATH)
    print(f'Saved trained model weights to "{MODEL_PATH}".')


if __name__ == '__main__':
    main()
