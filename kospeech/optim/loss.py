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
        dim (int): dimension of calculation loss
        reduction (str): reduction method [sum, mean] (default: sum)
        architecture (str): speech model`s architecture [seq2seq, transformer] (default: seq2seq)

    Inputs: logit, target
        logit (torch.Tensor): probability distribution value from model and it has a logarithm shape
        target (torch.Tensor): ground-thruth encoded to integers which directly point a word in label

    Returns: label_smoothed
        - **label_smoothed** (float): sum of loss
    """
    def __init__(
            self,
            num_classes: int,           # the number of classfication
            ignore_index: int,          # indexes that are ignored when calcuating loss
            smoothing: float = 0.1,     # ratio of smoothing (confidence = 1.0 - smoothing)
            dim: int = -1,              # dimension of caculation loss
            reduction='sum',            # reduction method [sum, mean]
            architecture='seq2seq'      # speech model`s architecture [seq2seq, transformer]
    ) -> None:
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

        if architecture.lower() == 'seq2seq':
            self.forward_method = self.get_seq2seq_loss
        elif architecture.lower() == 'transformer':
            self.forward_method = self.get_transformer_loss
        else:
            raise ValueError("Unsupported architecture : {0}".format(architecture))

    def forward(self, logit: Tensor, target: Tensor):
        return self.forward_method(logit, target)

    def get_seq2seq_loss(self, logit: Tensor, target: Tensor):
        if self.smoothing > 0.0:
            with torch.no_grad():
                label_smoothed = torch.zeros_like(logit)
                label_smoothed.fill_(self.smoothing / (self.num_classes - 1))
                label_smoothed.scatter_(1, target.data.unsqueeze(1), self.confidence)
                label_smoothed[target == self.ignore_index, :] = 0
            return self.reduction_method(-label_smoothed * logit)

        return F.cross_entropy(logit, target, ignore_index=self.ignore_index, reduction=self.reduction)

    def get_transformer_loss(self, logit: Tensor, target: Tensor):
        """
        Args:
             logit: B x T x C, score before softmax
             target: B x T
        """
        if self.smoothing > 0.0:
            with torch.no_grad():
                target_for_scatter = target.ne(self.ignore_index).long() * target
                one_hot = torch.zeros_like(logit).scatter(1, target_for_scatter.view(-1, 1), 1)
                one_hot = one_hot * (1 - self.smoothing) + (1 - one_hot) * self.smoothing / self.num_classes
                log_prob = F.log_softmax(logit, dim=1)

                non_pad_mask = target.ne(self.ignore_index)
                num_words = non_pad_mask.sum().item()
                loss = -(one_hot * log_prob).sum(dim=1)
                loss = loss.masked_select(non_pad_mask).sum() / num_words
            return loss

        return F.cross_entropy(logit, target, ignore_index=self.ignore_index, reduction=self.reduction)
