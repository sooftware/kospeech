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
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple


class BaseModel(nn.Module):
    """
    Interface of KoSpeech's Speech-To-Text Models.
    for a simple, generic encoder / decoder or encoder only model.
    """
    def __init__(self):
        super(BaseModel, self).__init__()

    def count_parameters(self):
        raise NotImplementedError

    def update_dropout(self, dropout):
        raise NotImplementedError

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    @torch.no_grad()
    def recognize(self, inputs: Tensor, input_lengths: Tensor) -> Tensor:
        raise NotImplementedError


class CTCModel(BaseModel):
    """
    Interface of KoSpeech's CTC based Models.
    """
    def __init__(self):
        super(CTCModel, self).__init__()

    def count_parameters(self):
        return sum([p.numel for p in self.parameters()])

    def update_dropout(self, dropout_p: float) -> None:
        for name, child in self.named_children():
            if isinstance(child, torch.nn.Dropout):
                child.p = dropout_p

    def forward(self, inputs: Tensor, input_lengths: Tensor) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError

    def get_normalized_probs(self, net_outputs: Tensor):
        assert hasattr(self, "fc"), "self.fc should be defined"
        outputs = self.fc(net_outputs)
        outputs = F.log_softmax(outputs, dim=-1)
        return outputs

    @torch.no_grad()
    def recognize(self, inputs: Tensor, input_lengths: Tensor) -> Tensor:
        outputs, output_lengths = self.forward(inputs, input_lengths)
        return outputs.max(-1)[1]


class EncoderDecoderModel(BaseModel):
    """
    Interface of KoSpeech's Encoder-Decoder Models.
    """
    def __init__(self, encoder: nn.Module, decoder: nn.Module) -> None:
        super(EncoderDecoderModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def count_parameters(self):
        num_encoder_parameters = sum([p.numel for p in self.encoder.parameters()])
        num_decoder_parameters = sum([p.numel for p in self.decoder.parameters()])
        if hasattr(self, 'fc'):
            num_fc_parameters = sum([p.numel for p in self.fc.parameters()])
            return num_encoder_parameters, num_decoder_parameters, num_fc_parameters
        return num_encoder_parameters, num_decoder_parameters

    def update_dropout(self, dropout_p):
        self.encoder.update_dropout(dropout_p)
        self.decoder.update_dropout(dropout_p)

    def forward(self, inputs: Tensor, input_lengths: Tensor, targets: Tensor, **kwargs) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError

    @torch.no_grad()
    def recognize(self, inputs: Tensor, input_lengths: Tensor) -> Tensor:
        raise NotImplementedError
