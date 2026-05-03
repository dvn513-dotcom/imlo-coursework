# Oxford-IIIT Pet Classifier — IMLO Submission

A custom ResNet-18-style CNN trained from scratch on the Oxford-IIIT Pet
dataset (37 cat/dog breeds), implemented in PyTorch.

## Files

| File | Purpose |
| --- | --- |
| `model.py` | Custom ResNet-18-style network (`PetClassifier`) implemented from scratch, with dropout for regularisation. |
| `train.py` | Loads the official `trainval` split, trains the model for 30 epochs, saves weights to `pet_classifier.pth`. Takes no arguments. |
| `test.py` | Loads `pet_classifier.pth`, evaluates on the official `test` split, prints accuracy. Takes no arguments. |
| `answers.md` | Answers to the 15 form questions. |

## How to run

```bash
# 1. Train (downloads Oxford-IIIT Pet on first run; needs internet once)
python train.py

# 2. Evaluate on the official test split
python test.py
```

## Recipe summary

- **Architecture:** custom ResNet-18 (~11.2M parameters, 18 weight layers) with dropout (p=0.2) before the classifier.
- **Loss:** CrossEntropyLoss with label smoothing 0.1.
- **Regularisation:** light Mixup (alpha=0.1) in the training loop, plus weight decay 1e-4.
- **Optimiser:** AdamW (lr=1e-3).
- **Learning-rate schedule:** 3-epoch linear warmup then 27-epoch cosine decay.
- **Batch size:** 64. **Epochs:** 30. **Input size:** 224x224.
- **Train augmentation:** RandomResizedCrop, RandomHorizontalFlip, ColorJitter, ToTensor, Normalize.
- **Test augmentation:** Resize, ToTensor, Normalize only.

## Notes on rules compliance

- No pretrained weights or external image data are used.
- All augmentations come from `torchvision.transforms`; Mixup is implemented with plain tensor ops.
- Test-time pre-processing is limited to operations strictly needed.
- Full official `trainval` split for training; `test` split only for final evaluation.
- Architecture is hand-written from scratch.
