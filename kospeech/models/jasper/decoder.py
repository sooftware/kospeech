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
from kospeech.models.jasper import JasperDecoderConfig
from kospeech.models.jasper.sublayers import JasperSubBlock


class JasperDecoder(nn.Module):
    """
    Jasper Encoder consists of three post-processing blocks.

    Args:
        config (JasperDecoderConfig): configurations of Jasper Decoder
        device (torch.device): device - 'cuda' or 'cpu'

    Inputs: inputs, input_lengths, residual
        - **inputs**: tensor contains input sequence vector
        - **input_lengths**: tensor contains sequence lengths

    Returns: output, output_lengths
        - **output**: tensor contains output sequence vector
        - **output**: tensor contains output sequence lengths
    """

    def __init__(self, config: JasperDecoderConfig, device: torch.device = 'cuda') -> None:
        super(JasperDecoder, self).__init__()
        self.config = config
        self.device = device
        self.layers = nn.ModuleList([
            JasperSubBlock(
                in_channels=config.block['in_channels'][i],
                out_channels=config.block['out_channels'][i],
                kernel_size=config.block['kernel_size'][i],
                dilation=config.block['dilation'][i],
                dropout_p=config.block['dropout_p'][i],
                activation='relu',
                bias=True if i == 2 else False
            ).to(self.device) for i in range(3)
        ])

    def forward(self, encoder_outputs: Tensor, encoder_output_lengths: Tensor) -> Tuple[Tensor, Tensor]:
        """
        encoder_outputs (torch.FloatTensor): (batch_size, dimension, sequence_length)
        encoder_output_lengths (torch.LongTensor): (batch_size)
        """
        outputs, output_lengths = encoder_outputs, encoder_output_lengths

        for i, layer in enumerate(self.layers):
            outputs, output_lengths = layer(outputs, output_lengths)

        outputs = F.log_softmax(outputs.transpose(1, 2), dim=-1)
        del encoder_outputs, encoder_output_lengths

        return outputs, output_lengths
