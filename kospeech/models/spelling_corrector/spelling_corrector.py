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
