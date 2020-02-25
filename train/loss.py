import torch
import torch.nn as nn

class LabelSmoothingLoss(nn.Module):
    """
    Provides Label-Smoothing loss.

    Reference:
        https://github.com/pytorch/pytorch/issues/7455
    """
    def __init__(self, vocab_size, ignore_index, smoothing=0.1, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.vocab_size = vocab_size
        self.dim = dim
        self.ignore_index = ignore_index

    def forward(self, logit, target):
        with torch.no_grad():
            label_smoothed = torch.zeros_like(logit)
            label_smoothed.fill_(self.smoothing / (self.vocab_size - 1))
            label_smoothed.scatter_(1, target.data.unsqueeze(1), self.confidence)
            label_smoothed[target == self.ignore_index, :] = 0
        return torch.sum(-label_smoothed * logit)