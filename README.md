# Reparameterized Attention for Convolutional Neural Networks

This repository contains the code to reproduce the experiments carried out in `Reparameterized Attention for Convolutional Neural Networks`.

![Reparameterized Attention](docs/arch.pdf)

# Install

Please see [INSTALL.md](docs/INSTALL.md)

# Training and validation

Please see [TRAIN.md](docs/TRAIN.md)

# Results on ImageNet-100

Our training logs (tensorboard) are in `logs`.

| Backbone            | Our Strategy | Top-1 (val) | Path                   |
| ------------------- | ------------ | ----------- | ---------------------- |
| SENet               | No           | 81.70       | logs/se_baseline       |
|                     | Yes          | 82.94       | logs/se_rep            |
| GENet50             | No           | 81.31       | logs/ge_baseline       |
|                     | Yes          | 82.48       | logs/ge_rep            |
| Coord Attention(50) | No           | 82.05       | logs/coordatt_baseline |
|                     | Yes          | 83.32       | logs/coordatt_rep      |
| CBAM(50)            | No           | 81.11       | logs/cbam_baseline     |
|                     | Yes          | 82.13       | logs/cbam_rep          |

Checkpoints are in `Release`.