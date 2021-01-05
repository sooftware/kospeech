# -*- coding: utf-8 -*-
# Soohwan Kim, Seyoung Bae, Cheolhwang Won.
# @ArXiv : KoSpeech: Open-Source Toolkit for End-to-End Korean Speech Recognition
# This source code is licensed under the Apache 2.0 License license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn

from typing import Tuple
from torch import Tensor


class JointCTCCrossEntropyLoss(nn.Module):
    """
    Label smoothed cross entropy loss function.

    Args:
        num_classes (int): the number of classfication
        ignore_index (int): Indexes that are ignored when calculating loss
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
            dim: int = -1,                        # dimension of caculation loss
            reduction='mean',                     # reduction method [sum, mean]
            ctc_weight: float = 0.3,              # weight of ctc loss
            cross_entropy_weight: float = 0.7,    # weight of cross entropy loss
            blank_id: int = None,                 # identification of blank token
            architecture: str = 'las',            # architecture of model to train
    ) -> None:
        super(JointCTCCrossEntropyLoss, self).__init__()
        self.num_classes = num_classes
        self.dim = dim
        self.ignore_index = ignore_index
        self.reduction = reduction.lower()
        self.ctc_weight = ctc_weight
        self.cross_entropy_weight = cross_entropy_weight
        self.architecture = architecture
        self.ctc_loss = nn.CTCLoss(blank=blank_id, reduction=self.reduction, zero_infinity=True)
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction=self.reduction, ignore_index=self.ignore_index)

    def forward(
            self,
            encoder_log_probs: Tensor,
            decoder_log_probs: Tensor,
            output_lengths: Tensor,
            targets: Tensor,
            target_lengths: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        ctc_loss = self.ctc_loss(encoder_log_probs, targets, output_lengths, target_lengths)
        if self.architecture == 'las':
            targets = targets[:, 1:]
        cross_entropy_loss = self.cross_entropy_loss(decoder_log_probs, targets.contiguous().view(-1))
        loss = cross_entropy_loss * self.cross_entropy_weight + ctc_loss * self.ctc_weight
        return loss, ctc_loss, cross_entropy_loss
