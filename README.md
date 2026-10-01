# CIFAR-10 Image Classification with a Shake-Shake ResNet

A residual network with **Shake-Shake regularization**, trained from scratch on CIFAR-10. It reaches **93.6% test accuracy** in 50 epochs.

NYU Deep Learning (Spring 2025), Mini Project 1.

## Results

| Model | Params | Epochs | Test accuracy |
|---|---|---|---|
| Shake-Shake ResNet-26 (2×64d) | 11.65M | 50 | **93.64%** |

## What's in the model

- **Architecture:** a ResNet-26 where each residual block has two parallel convolutional branches. They are mixed with random weights in the forward pass and independent random weights in the backward pass (Shake-Shake, [Gastaldi 2017](https://arxiv.org/abs/1705.07485)). Implemented as a custom `torch.autograd.Function`.
- **Augmentation:** AutoAugment (CIFAR-10 policy) plus random **Mixup / CutMix** per batch
- **Regularization:** label-smoothing cross-entropy (ε = 0.1) and weight decay
- **Optimization:** SGD with momentum and a cosine-annealing learning-rate schedule
- **Engineering:** batch size 256, pinned-memory data loaders, GPU training on Kaggle

## Run it

Open [`cifar10_shakeshake_resnet.ipynb`](cifar10_shakeshake_resnet.ipynb) in Kaggle or Colab with a GPU and run all cells. CIFAR-10 downloads automatically through `torchvision`.

## Tech

Python · PyTorch · torchvision · NumPy · Kaggle GPU
