# Reparameterized Attention for Convolutional Neural Networks

This repository contains the code to reproduce the experiments carried out in `Reparameterized Attention for Convolutional Neural Networks`.

![Reparameterized Attention](docs/arch.pdf)

# Install

Please see [INSTALL.md](docs/INSTALL.md)

# Training and validation

Please see [TRAIN.md](docs/TRAIN.md)

# Results on ImageNet-100

Our training logs (tensorboard) are in `logs`.

| Backbone            | Our Strategy | Top-1 (val) | Path                |
| ------------------- | ------------ | ----------- | ------------------- |
| SENet               | No           | 81.70       | logs/SENet          |
|                     | Yes          | 82.94       | logs/SENet_rep      |
| GENet50             | No           | 81.31       | logs/GENet50        |
|                     | Yes          | 82.48       | logs/GENet50_rep    |
| Coord Attention(50) | No           | 82.05       | logs/CoordAtt50     |
|                     | Yes          | 83.32       | logs/CoordAtt50_rep |
| CBAM(50)            | No           | 81.11       | logs/cbam_baseline  |
|                     | Yes          | 82.13       | logs/cbam_rep       |

Checkpoints are in `Release`.