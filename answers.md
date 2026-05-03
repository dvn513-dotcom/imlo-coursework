# Answers to Submission-Form Questions

## Q1. How many layers in your network?
**18**

(Standard ResNet-18 convention — counting weight-bearing layers only:
1 stem `Conv2d` + 16 `Conv2d` inside the residual blocks + 1 final
`Linear`. Three additional 1×1 projection convs sit on the shortcut
paths in stages 2, 3, and 4; these are not counted in the 18, matching
the original ResNet paper's convention.)

## Q2. List the type of each layer used in your network.
**Conv2d, Conv2d, Conv2d, Conv2d, Conv2d, Conv2d, Conv2d, Conv2d, Conv2d, Conv2d, Conv2d, Conv2d, Conv2d, Conv2d, Conv2d, Conv2d, Conv2d, Linear**

(Between these weight layers there are also `BatchNorm2d` after every
`Conv2d`, a `MaxPool2d` after the stem, an `AdaptiveAvgPool2d` before
the classifier, a `Dropout(p=0.2)` between the global average pool and
the classifier, and three 1×1 `Conv2d` projection shortcuts in stages
2–4.)

## Q3. List how many units/kernels were used in each layer of your network.
**64, 64, 64, 64, 64, 128, 128, 128, 128, 256, 256, 256, 256, 512, 512, 512, 512, 37**

(One entry per weight layer in the order listed in Q2: 64 kernels in the
stem and stage 1, then 128 in stage 2, 256 in stage 3, 512 in stage 4,
and finally 37 output units in the classifier — one per pet breed.)

## Q4. List the activation functions used in each layer of your architecture.
**ReLU, ReLU, ReLU, ReLU, ReLU, ReLU, ReLU, ReLU, ReLU, ReLU, ReLU, ReLU, ReLU, ReLU, ReLU, ReLU, ReLU, Softmax**

(ReLU is applied after every conv layer's BatchNorm. The final `Linear`
layer outputs raw logits; the softmax is applied implicitly inside
`nn.CrossEntropyLoss` at training time and via `argmax` at inference.)

## Q5. What is the total number of weights/biases in your architecture?
**11,195,493**

(Computed by `sum(p.numel() for p in model.parameters() if p.requires_grad)`.
This counts all conv weights, BN gamma/beta, the 1×1 shortcut projections,
and the final linear weights and bias. Dropout has no learnable parameters
so it does not affect this count.)

## Q6. What loss function did you use to evaluate your network?
**Cross-Entropy Loss (with label smoothing 0.1)**

(In code: `nn.CrossEntropyLoss(label_smoothing=0.1)`.)

## Q7. What optimisation algorithm did you use to train your network?
**AdamW**

(`torch.optim.AdamW` with `lr=1e-3`, `weight_decay=1e-4`, and PyTorch's
default β values.)

## Q8. What learning rate did you use?
**1.0e-3**

(Initial learning rate. Schedule: 3-epoch linear warmup from 0 to 1e-3,
then 27-epoch cosine decay from 1e-3 down to 0.)

## Q9. For how many epochs did you train the network?
**30**

## Q10. What batch size did you use?
**64**

## Q11. List the training dataset augmentations/transforms used.
**RandomResizedCrop(224, scale=(0.6, 1.0)), RandomHorizontalFlip(p=0.5), ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), ToTensor, Normalize(ImageNet mean/std), Mixup(alpha=0.1)**

(All `transforms.*` are PyTorch built-ins from `torchvision.transforms`;
Mixup is implemented in the training loop using plain tensor operations.
ImageNet mean/std refers to `[0.485, 0.456, 0.406]` /
`[0.229, 0.224, 0.225]` — used purely as a fixed centring/scaling, no
external data is loaded.)

## Q12. Total number of images in your training dataset.
**3680**

(The full official `trainval` split of Oxford-IIIT Pet.)

## Q13. Total number of images in your validation dataset.
**0**

(No validation split is used. The entire `trainval` split is used for
training.)

## Q14. Accuracy of the trained model on the Oxford-IIIT Pet training dataset.
**84.13%**

(Final un-augmented training-set accuracy reported by `train.py` after
30 epochs of training.)

## Q15. Accuracy of the trained model on the Oxford-IIIT Pet test dataset.
**55.63%**

(Reported by `test.py` on the official Oxford-IIIT Pet `test` split of
3,669 images.)
