import torch.nn as nn
from torch import Tensor
from typing import Optional, Any


class LanguageSeq2seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, inputs: Tensor, targets: Optional[Any] = None, teacher_forcing_ratio: float = 1.0):
        output, hidden = self.encoder(inputs)
        output = self.decoder(targets, output, teacher_forcing_ratio)

        return output

    def flatten_parameters(self):
        self.encoder.rnn.flatten_parameters()
        self.decoder.rnn.flatten_parameters()

    def set_decoder(self, decoder: nn.Module):
        self.decoder = decoder
