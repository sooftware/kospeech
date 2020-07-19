import torch
import torch.nn as nn
import torch.nn.functional as F


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
    def __init__(self, num_classes, ignore_index, smoothing=0.1, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.num_classes = num_classes
        self.dim = dim
        self.ignore_index = ignore_index

    def forward(self, logit, target):
        with torch.no_grad():
            label_smoothed = torch.zeros_like(logit)
            label_smoothed.fill_(self.smoothing / (self.num_classes - 1))
            label_smoothed.scatter_(1, target.data.unsqueeze(1), self.confidence)
            label_smoothed[target == self.ignore_index, :] = 0

        return torch.sum(-label_smoothed * logit)


class TransformerLoss(nn.Module):
    def __init__(self, num_classes, ignore_index, smoothing=0.1):
        super(TransformerLoss, self).__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.smoothing = smoothing

    def forward(self, y_hats, targets):
        """Calculate cross entropy loss, apply label smoothing if needed.
        Args:
            y_hats: N x T x C, score before softmax
            targets: N x T
        """

        if self.smoothing > 0.0:
            eps = self.smoothing

            # Generate one-hot matrix: N x C.
            # Only label position is 1 and all other positions are 0
            # gold include -1 value (IGNORE_ID) and this will lead to assert error
            target_scatter = targets.ne(self.ignore_index).long() * targets
            one_hot = torch.zeros_like(y_hats).scatter(1, target_scatter.view(-1, 1), 1)
            one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / self.num_classes

            log_prb = F.log_softmax(y_hats, dim=1)

            non_pad_mask = targets.ne(self.ignore_index)
            n_word = non_pad_mask.sum().item()
            loss = -(one_hot * log_prb).sum(dim=1)
            loss = loss.masked_select(non_pad_mask).sum() / n_word
        else:
            loss = F.cross_entropy(y_hats, targets,
                                   ignore_index=self.ignore_index,
                                   reduction='sum')

        return loss
