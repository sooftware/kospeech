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

from torch import Tensor
from typing import Tuple
from kospeech.models.jasper.sublayers import JasperSubBlock


class Jasper10x5DecoderConfig:
    def __init__(self, num_classes):
        self.block = {
            'in_channels': (768, 896, 1024),
            'out_channels': (896, 1024, num_classes),
            'kernel_size': (29, 1, 1),
            'dilation': (2, 1, 1),
            'dropout_p': (0.4, 0.4, 0.0)
        }


class JasperDecoder(nn.Module):
    """
    Jasper Encoder consists of three post-processing blocks.

    Args:
        num_classes (int): number of classification
        version (str): version of jasper. Marked as BxR: B - number of blocks, R - number of sub-blocks

    Inputs: inputs, input_lengths, residual
        - **inputs**: tensor contains input sequence vector
        - **input_lengths**: tensor contains sequence lengths

    Returns: output, output_lengths
        - **output**: tensor contains output sequence vector
        - **output**: tensor contains output sequence lengths
    """
    def __init__(self, num_classes: int, version: str):
        super(JasperDecoder, self).__init__()
        assert version.lower() in ['10x5'], "Unsupported Version: {}".format(version)

        if version.lower() == '10x5':
            self.num_blocks = 10
            self.num_sub_blocks = 5
            self.config = Jasper10x5DecoderConfig(num_classes)

        self.layers = nn.ModuleList([
            JasperSubBlock(
                in_channels=self.config.block['in_channels'][i],
                out_channels=self.config.block['out_channels'][i],
                kernel_size=self.config.block['kernel_size'][i],
                dilation=self.config.block['dilation'][i],
                dropout_p=self.config.block['dropout_p'][i],
                bias=True
            ) for i in range(3)
        ])

    def forward(self, encoder_outputs: Tensor, encoder_output_lengths: Tensor) -> Tuple[Tensor, Tensor]:
        output, output_lengths = None, None

        for layer in self.layers:
            if output is None:
                output, output_lengths = layer(encoder_outputs, encoder_output_lengths)
            else:
                output, output_lengths = layer(output, output_lengths)

        del encoder_outputs, encoder_output_lengths

        return output, output_lengths
