# Oxford-IIIT Pet Classifier — IMLO Coursework

Custom CNN trained from scratch on Oxford-IIIT Pet (37 cat/dog breeds), in PyTorch.

## Files
- `model.py` — ResNet-18-style network, written from scratch
- `train.py` — training script (no args)
- `test.py` — evaluation script (no args)
- `pet_classifier.pth` — trained model weights
- `answers.md` — answers for the submission form

## Running

```
python train.py
python test.py
```

`train.py` will download the dataset on first run.

## Setup

Activate the `imlo-coursework` conda environment:
```
conda activate imlo-coursework
```

## Recipe

- ResNet-18-style CNN (~11.2M params) with dropout (p=0.2) before the classifier
- Loss: CrossEntropyLoss with label smoothing 0.1
- Optimiser: AdamW (lr=1e-3, weight_decay=1e-4)
- Schedule: 3-epoch warmup then cosine decay over the remaining 27 epochs
- Batch size 64, 30 epochs, 224×224 input
- Train aug: RandomResizedCrop, RandomHorizontalFlip, ColorJitter, plus Mixup (α=0.1)
- Test aug: Resize, ToTensor, Normalize only

## Results
- Training accuracy: 84.13%
- Test accuracy: 55.63%

## Hardware
Trained on a laptop CPU. Took about 2 hours.
