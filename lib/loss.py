import torch
from torch import nn


class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.
    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """

    def __init__(self, num_classes=1000, epsilon=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward_v1(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size(), device=targets.device).scatter_(
            1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + \
            self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss

    def forward_v2(self, inputs, targets):
        probs = self.logsoftmax(inputs)
        targets = torch.zeros(probs.size(), device=targets.device).scatter_(
            1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + \
            self.epsilon / self.num_classes
        loss = nn.KLDivLoss()(probs, targets)
        return loss

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        return self.forward_v1(inputs, targets)

    def test(self):

        inputs = torch.randn(2, 5)
        targets = torch.randint(0, 5, [2])
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size(), device=targets.device).scatter_(
            1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + \
            self.epsilon / self.num_classes

        print((targets * torch.log(targets) - targets * log_probs).sum(-1).mean())
        print(nn.KLDivLoss(reduce='mean')(log_probs, targets))
