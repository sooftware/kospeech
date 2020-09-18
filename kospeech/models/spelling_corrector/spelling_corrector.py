# -*- coding: utf-8 -*-
# Soohwan Kim, Seyoung Bae, Cheolhwang Won.
# @ArXiv : KoSpeech: Open-Source Toolkit for End-to-End Korean Speech Recognition
# This source code is licensed under the Apache 2.0 License license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn
from torch import Tensor
from typing import Optional, Any


class SpellingCorrector(nn.Module):
    """
    Implementation of paper "A Spelling Correction Model for End-to-End Speech Recognition"
    - arXiv : https://arxiv.org/pdf/1902.07178.pdf
    """
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, inputs: Tensor, targets: Optional[Any] = None):
        output = self.encoder(inputs)
        output = self.decoder(targets, output)

        return output
