# Copyright (c) 2021, Soohwan Kim. All rights reserved.
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

import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple, Optional


class EncoderInterface(nn.Module):
    def __init__(self):
        super(EncoderInterface, self).__init__()

    def count_parameters(self):
        return sum([p.numel for p in self.parameters()])

    def update_dropout(self, dropout_p: float) -> None:
        for name, child in self.named_children():
            if isinstance(child, nn.Dropout):
                child.p = dropout_p

    def forward(self, inputs: Tensor, input_lengths: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        raise NotImplementedError


class DecoderInterface(nn.Module):
    def __init__(self):
        super(DecoderInterface, self).__init__()

    def count_parameters(self):
        return sum([p.numel for p in self.parameters()])

    def update_dropout(self, dropout_p):
        for name, child in self.named_children():
            if isinstance(child, nn.Dropout):
                child.p = dropout_p

    def forward(self, targets: Tensor, encoder_outputs: Tensor, **kwargs):
        raise NotImplementedError

    @torch.no_grad()
    def decode(self, encoder_outputs: Tensor, *args) -> Tensor:
        raise NotImplementedError


class EncoderDecoderModelInterface(nn.Module):
    """
    Interface of KoSpeech's Encoder-Decoder Models.
    """
    def __init__(self, encoder: nn.Module, decoder: nn.Module) -> None:
        super(EncoderDecoderModelInterface, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def set_encoder(self, encoder):
        self.encoder = encoder

    def set_decoder(self, decoder):
        self.decoder = decoder

    def count_parameters(self):
        num_encoder_parameters = self.encoder.count_parameters()
        num_decoder_parameters = self.decoder.count_parameters()
        return num_encoder_parameters + num_decoder_parameters

    def update_dropout(self, dropout_p):
        self.encoder.update_dropout(dropout_p)
        self.decoder.update_dropout(dropout_p)

    def forward(
            self,
            inputs: Tensor,
            input_lengths: Tensor,
            targets: Tensor,
            *args,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        raise NotImplementedError

    @torch.no_grad()
    def recognize(self, inputs: Tensor, input_lengths: Tensor) -> Tensor:
        encoder_outputs, encoder_output_lengths, _ = self.encoder(inputs, input_lengths)
        return self.decoder.decode(encoder_outputs, encoder_output_lengths)


class CTCModelInterface(nn.Module):
    """
    Interface of KoSpeech's Speech-To-Text Models.
    for a simple, generic encoder / decoder or encoder only model.
    """
    def __init__(self):
        super(CTCModelInterface, self).__init__()
        self.decoder = None

    def set_decoder(self, decoder):
        self.decoder = decoder

    def count_parameters(self):
        return sum([p.numel for p in self.parameters()])

    def update_dropout(self, dropout_p: float) -> None:
        for name, child in self.named_children():
            if isinstance(child, nn.Dropout):
                child.p = dropout_p

    def forward(self, inputs: Tensor, input_lengths: Tensor) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError

    @torch.no_grad()
    def decode(self, predicted_log_probs: Tensor) -> Tensor:
        return predicted_log_probs.max(-1)[1]

    @torch.no_grad()
    def recognize(self, inputs: Tensor, input_lengths: Tensor) -> Tensor:
        predicted_log_probs, _ = self.forward(inputs, input_lengths)
        if self.decoder is not None:
            return self.decoder.decode(predicted_log_probs)
        return self.decode(predicted_log_probs)
