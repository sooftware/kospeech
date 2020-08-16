import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class CrossEntropyWithSmoothingLoss(nn.Module):
    """
    A Cross Entropy with Label Smoothing Loss.

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
    def __init__(self, num_classes: int, ignore_index: int, smoothing: float = 0.1, dim: int = -1, reduction='sum'):
        super(CrossEntropyWithSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.num_classes = num_classes
        self.dim = dim
        self.ignore_index = ignore_index
        self.reduction = reduction.lower()

        if self.reduction == 'sum':
            self.reduction_method = torch.sum
        elif self.reduction == 'mean':
            self.reduction_method = torch.mean
        else:
            raise ValueError("Unsupported reduction method {0}".format(reduction))

    def forward(self, logit: Tensor, target: Tensor):
        if self.smoothing > 0.0:
            with torch.no_grad():
                label_smoothed = torch.zeros_like(logit)
                label_smoothed.fill_(self.smoothing / (self.num_classes - 1))
                label_smoothed.scatter_(1, target.data.unsqueeze(1), self.confidence)
                label_smoothed[target == self.ignore_index, :] = 0
            return self.reduction_method(-label_smoothed * logit)

        return F.cross_entropy(logit, target, ignore_index=self.ignore_index, reduction=self.reduction)


def cal_performance(pred, gold, smoothing=0.0, ignore_id=0):
    """Calculate cross entropy loss, apply label smoothing if needed.
    Args:
        pred: N x T x C, score before softmax
        gold: N x T
    """

    pred = pred.view(-1, pred.size(2))
    gold = gold.contiguous().view(-1)

    loss = cal_loss(pred, gold, smoothing)

    pred = pred.max(1)[1]
    non_pad_mask = gold.ne(ignore_id)
    n_correct = pred.eq(gold)
    n_correct = n_correct.masked_select(non_pad_mask).sum().item()

    return loss, n_correct


def cal_loss(pred, gold, smoothing=0.0, ignore_id=0):
    """Calculate cross entropy loss, apply label smoothing if needed.
    """

    if smoothing > 0.0:
        eps = smoothing
        n_class = pred.size(1)

        # Generate one-hot matrix: N x C.
        # Only label position is 1 and all other positions are 0
        # gold include -1 value (IGNORE_ID) and this will lead to assert error
        gold_for_scatter = gold.ne(ignore_id).long() * gold
        one_hot = torch.zeros_like(pred).scatter(1, gold_for_scatter.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / n_class
        log_prb = F.log_softmax(pred, dim=1)

        non_pad_mask = gold.ne(ignore_id)
        n_word = non_pad_mask.sum().item()
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).sum() / n_word
    else:
        loss = F.cross_entropy(pred, gold,
                               ignore_index=ignore_id,
                               reduction='mean')

    return loss