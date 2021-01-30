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

from kospeech.models.modules import Linear


class JointNet(nn.Module):
    def __init__(self, input_dim: int, inner_dim: int, num_classes: int):
        super(JointNet, self).__init__()
        self.fc = nn.Sequential(
            Linear(input_dim, inner_dim, bias=True),
            nn.Tanh(),
            Linear(inner_dim, num_classes, bias=False),
        )

    def forward(self, encoder_outputs: Tensor, decoder_outputs: Tensor) -> Tensor:
        input_length = encoder_outputs.size(1)
        target_length = decoder_outputs.size(1)

        encoder_outputs = encoder_outputs.unsqueeze(2)  # B T 1 D
        decoder_outputs = decoder_outputs.unsqueeze(1)  # B 1 t D

        encoder_outputs = encoder_outputs.repeat([1, 1, target_length, 1])    # B T 1 D => B T t D
        decoder_outputs = decoder_outputs.repeat([1, input_length, 1, 1])     # B 1 t D => B T t D

        outputs = torch.cat((encoder_outputs, decoder_outputs), dim=-1)
        outputs = self.fc(outputs)

        return outputs
