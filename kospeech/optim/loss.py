import torch
import torch.nn as nn
from torch import Tensor


class LabelSmoothingLoss(nn.Module):
    """
    Provides Label-Smoothing loss.
    Copied from https://github.com/pytorch/pytorch/issues/7455

    Args:
        num_classes (int): the number of classfication
        ignore_index (int): Indexes that are ignored when calculating loss
        smoothing (float): ratio of smoothing (confidence = 1.0 - smoothing)
        dim (int): dimention of calculation loss

    Inputs: logit, target
        logit (torch.Tensor): probability distribution value from model and it has a logarithm shape
        target (torch.Tensor): ground-thruth encoded to integers which directly point a word in label

    Returns: label_smoothed
        - **label_smoothed** (float): sum of loss
    """
    def __init__(self, num_classes: int, ignore_index: int, smoothing: float = 0.1, dim: int = -1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.num_classes = num_classes
        self.dim = dim
        self.ignore_index = ignore_index

    def forward(self, logit: Tensor, target: Tensor):
        with torch.no_grad():
            label_smoothed = torch.zeros_like(logit)
            label_smoothed.fill_(self.smoothing / (self.num_classes - 1))
            label_smoothed.scatter_(1, target.data.unsqueeze(1), self.confidence)
            label_smoothed[target == self.ignore_index, :] = 0

        return torch.sum(-label_smoothed * logit)
