# Copyright (c) 2020, Soohwan Kim. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch.nn as nn
from typing import Tuple
from torch import Tensor

from kospeech.criterion import LabelSmoothedCrossEntropyLoss


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
            smoothing: float = 0.1,               # ratio of smoothing (confidence = 1.0 - smoothing)
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
        if smoothing > 0.0:
            self.cross_entropy_loss = LabelSmoothedCrossEntropyLoss(
                num_classes=num_classes,
                ignore_index=ignore_index,
                smoothing=smoothing,
                reduction=reduction,
                architecture=architecture,
                dim=-1,
            )
        else:
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
