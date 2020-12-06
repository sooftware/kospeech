# -*- coding: utf-8 -*-
# Soohwan Kim, Seyoung Bae, Cheolhwang Won.
# @ArXiv : KoSpeech: Open-Source Toolkit for End-to-End Korean Speech Recognition
# This source code is licensed under the Apache 2.0 License license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class JointCTCAttentionLoss(nn.Module):
    """
    Label smoothed cross entropy loss function.

    Args:
        num_classes (int): the number of classfication
        ignore_index (int): Indexes that are ignored when calculating loss
        smoothing (float): ratio of smoothing (confidence = 1.0 - smoothing)
        dim (int): dimension of calculation loss
        reduction (str): reduction method [sum, mean] (default: sum)
        ctc_weight (float): weight of ctc loss
        cross_entropy_weight (float): weight of cross entropy loss

    Inputs: logit, target
        logit (torch.Tensor): probability distribution value from model and it has a logarithm shape
        target (torch.Tensor): ground-thruth encoded to integers which directly point a word in label

    Returns: label_smoothed
        - **label_smoothed** (Tensor): sum of loss
    """
    def __init__(
            self,
            num_classes: int,                     # the number of classfication
            ignore_index: int,                    # indexes that are ignored when calcuating loss
            smoothing: float = 0.1,               # ratio of smoothing (confidence = 1.0 - smoothing)
            dim: int = -1,                        # dimension of caculation loss
            reduction='mean',                     # reduction method [sum, mean]
            ctc_weight: float = 0.5,              # weight of ctc loss
            cross_entropy_weight: float = 0.5,    # weight of cross entropy loss
            blank_id: int = None
    ) -> None:
        super(JointCTCAttentionLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.num_classes = num_classes
        self.dim = dim
        self.ignore_index = ignore_index
        self.reduction = reduction.lower()
        self.ctc_weight = ctc_weight
        self.cross_entropy_weight = cross_entropy_weight

        if self.reduction == 'sum':
            self.reduction_method = torch.sum
        elif self.reduction == 'mean':
            self.reduction_method = torch.mean
        else:
            raise ValueError("Unsupported reduction method {0}".format(reduction))

        self.ctc_loss = nn.CTCLoss(blank=blank_id, reduction=self.reduction)

    def get_ctc_loss(self, logits: Tensor, input_lengths: Tensor, targets: Tensor, target_lengths: Tensor) -> float:
        return self.ctc_loss(logits.log_softmax(dim=2), targets, input_lengths, target_lengths)

    def get_cross_entropy_loss(self, logits: Tensor, targets: Tensor):
        if self.smoothing > 0.0:
            with torch.no_grad():
                label_smoothed = torch.zeros_like(logits)
                label_smoothed.fill_(self.smoothing / (self.num_classes - 1))
                label_smoothed.scatter_(1, targets.data.unsqueeze(1), self.confidence)
                label_smoothed[targets == self.ignore_index, :] = 0
            return self.reduction_method(-label_smoothed * logits)

        return F.cross_entropy(logits, targets, ignore_index=self.ignore_index, reduction=self.reduction)

    def forward(
            self,
            cross_entropy_logits: Tensor,
            ctc_logits: Tensor,
            input_lengths: Tensor,
            targets: Tensor,
            target_lengths: Tensor
    ) -> Tensor:
        ctc_loss = self.get_ctc_loss(ctc_logits, input_lengths, targets, target_lengths)
        cross_entropy_loss = self.get_cross_entropy_loss(cross_entropy_logits, targets)
        loss = cross_entropy_loss * self.cross_entropy_weight + ctc_loss * self.ctc_weight
        return loss
