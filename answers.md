# Submission form answers

## Q1
8

## Q2
Conv2d, BatchNorm2d, ReLU, MaxPool2d, AdaptiveAvgPool2d, Flatten, Dropout, Linear

## Q3
64, 64, N/A, 3, 1, N/A, N/A, 37

(Conv2d kernel counts and BatchNorm2d channel counts vary across stages: 64 in stem and stage 1, 128 in stage 2, 256 in stage 3, 512 in stage 4. Listed values above are the stem sizes.)

## Q4
None, None, ReLU, None, None, None, None, Softmax

(Softmax is handled by CrossEntropyLoss.)

## Q5
11195493

## Q6
CrossEntropyLoss (with label smoothing 0.1)

## Q7
AdamW

## Q8
1.0e-3

(Warmup over 3 epochs, then cosine decay.)

## Q9
30

## Q10
64

## Q11
RandomResizedCrop, RandomHorizontalFlip, ColorJitter, ToTensor, Normalize, Mixup

## Q12
3680

## Q13
0

## Q14
84.13%

## Q15
55.63%
