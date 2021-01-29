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
        raise NotImplementedError

    def update_dropout(self, dropout):
        raise NotImplementedError

    def forward(self, inputs: Tensor, input_lengths: Tensor) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        raise NotImplementedError


class DecoderInterface(nn.Module):
    def __init__(self):
        super(DecoderInterface, self).__init__()

    def count_parameters(self):
        raise NotImplementedError

    def update_dropout(self, dropout):
        raise NotImplementedError

    def forward(self, targets: Tensor, encoder_outputs: Tensor, **kwargs):
        raise NotImplementedError

    @torch.no_grad()
    def decode(self, encoder_outputs: Tensor, *args: Tensor) -> Tensor:
        raise NotImplementedError


class EncoderDecoderModelInterface(nn.Module):
    def __init__(self):
        super(EncoderDecoderModelInterface, self).__init__()

    def set_encoder(self, encoder):
        raise NotImplementedError

    def set_decoder(self, decoder):
        raise NotImplementedError

    def count_parameters(self):
        raise NotImplementedError

    def update_dropout(self, dropout_p):
        raise NotImplementedError

    def forward(self, inputs: Tensor, input_lengths: Tensor, targets: Tensor, *args) -> Tuple[Tensor, Tensor, Tensor]:
        raise NotImplementedError

    @torch.no_grad()
    def recognize(self, inputs: Tensor, input_lengths: Tensor) -> Tensor:
        raise NotImplementedError


class CTCModelInterface(nn.Module):
    """
    Interface of KoSpeech's Speech-To-Text Models.
    for a simple, generic encoder / decoder or encoder only model.
    """
    def __init__(self):
        super(CTCModelInterface, self).__init__()

    def set_decoder(self, decoder):
        raise NotImplementedError

    def count_parameters(self):
        raise NotImplementedError

    def update_dropout(self, dropout):
        raise NotImplementedError

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    @torch.no_grad()
    def decode(self, predicted_log_probs: Tensor) -> Tensor:
        raise NotImplementedError

    @torch.no_grad()
    def recognize(self, inputs: Tensor, input_lengths: Tensor) -> Tensor:
        raise NotImplementedError
