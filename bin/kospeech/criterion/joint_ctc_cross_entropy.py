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
    Privides Joint CTC-CrossEntropy Loss function

    Args:
        num_classes (int): the number of classification
        ignore_index (int): indexes that are ignored when calculating loss
        dim (int): dimension of calculation loss
        reduction (str): reduction method [sum, mean] (default: mean)
        ctc_weight (float): weight of ctc loss
        cross_entropy_weight (float): weight of cross entropy loss
        blank_id (int): identification of blank for ctc
    """
    
    def __init__(
            self,
            num_classes: int,                     
            ignore_index: int,                    
            dim: int = -1,                        
            reduction='mean',                     
            ctc_weight: float = 0.3,              
            cross_entropy_weight: float = 0.7,    
            blank_id: int = None,                 
            smoothing: float = 0.1,               
    ) -> None:
        super(JointCTCCrossEntropyLoss, self).__init__()
        self.num_classes = num_classes
        self.dim = dim
        self.ignore_index = ignore_index
        self.reduction = reduction.lower()
        self.ctc_weight = ctc_weight
        self.cross_entropy_weight = cross_entropy_weight
        self.ctc_loss = nn.CTCLoss(blank=blank_id, reduction=self.reduction, zero_infinity=True)
        if smoothing > 0.0:
            self.cross_entropy_loss = LabelSmoothedCrossEntropyLoss(
                num_classes=num_classes,
                ignore_index=ignore_index,
                smoothing=smoothing,
                reduction=reduction,
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
            target_lengths: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        ctc_loss = self.ctc_loss(encoder_log_probs, targets, output_lengths, target_lengths)
        cross_entropy_loss = self.cross_entropy_loss(decoder_log_probs, targets.contiguous().view(-1))
        loss = cross_entropy_loss * self.cross_entropy_weight + ctc_loss * self.ctc_weight
        return loss, ctc_loss, cross_entropy_loss
