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
from kospeech.models.jasper import JasperEncoderConfig
from kospeech.models.modules import MaskConv1d
from kospeech.models.jasper.sublayers import (
    JasperSubBlock, 
    JasperBlock,
)


class JasperEncoder(nn.Module):
    """
    Jasper Encoder consists of one pre-processing blocks and B Jasper blocks.

    Args:
        config (JasperEncoderConfig): configurations of Jasper Encoder

    Inputs: inputs, input_lengths, residual
        - **inputs**: tensor contains input sequence vector
        - **input_lengths**: tensor contains sequence lengths

    Returns: output, output_lengths
        - **output**: tensor contains output sequence vector
        - **output**: tensor contains output sequence lengths
    """

    def __init__(self, config: JasperEncoderConfig) -> None:
        super(JasperEncoder, self).__init__()
        self.config = config
        layers = list()
        layers.append(JasperSubBlock(
            in_channels=self.config.preprocess_block['in_channels'],
            out_channels=self.config.preprocess_block['out_channels'],
            kernel_size=self.config.preprocess_block['kernel_size'],
            stride=self.config.preprocess_block['stride'],
            dilation=self.config.preprocess_block['dilation'],
            dropout_p=self.config.preprocess_block['dropout_p'],
            activation='relu',
            bias=False,
        ))
        for i in range(config.num_blocks):
            layers.append(JasperBlock(
                num_sub_blocks=self.config.num_sub_blocks,
                in_channels=self.config.block['in_channels'][i],
                out_channels=self.config.block['out_channels'][i],
                kernel_size=self.config.block['kernel_size'][i],
                dilation=self.config.block['dilation'][i],
                dropout_p=self.config.block['dropout_p'][i],
                activation='relu',
                bias=False,
            ))
        self.layers = nn.ModuleList(layers)
        self.total_residual_layers = self._create_residual_layers(self.config.num_blocks)

    def forward(self, inputs: Tensor, input_lengths: Tensor) -> Tuple[Tensor, Tensor]:
        prev_outputs, prev_output_lengths = list(), list()
        residual = None
        output, output_lengths = inputs.transpose(1, 2), input_lengths

        for i, layer in enumerate(self.layers):
            output, output_lengths = layer(output, output_lengths, residual)
            prev_outputs.append(output)
            prev_output_lengths.append(output_lengths)
            residual = self._get_residual(prev_outputs, prev_output_lengths, i)

        return output, output_lengths

    def _get_residual(self, prev_outputs: list, prev_output_lengths: list, index: int):
        residual = None
        for item in zip(prev_outputs, prev_output_lengths, self.total_residual_layers[index]):
            prev_output, prev_output_length, residual_layers = item
            conv, batch_norm = residual_layers
            if residual is None:
                residual = conv(prev_output, prev_output_length)[0]
            else:
                residual += conv(prev_output, prev_output_length)[0]
            residual = batch_norm(residual)

        return residual

    def _create_residual_layers(self, num_blocks: int):
        total_residual_layers = list()

        for i in range(num_blocks):
            residual_layers = list()
            for j in range(i + 1):
                residual_layers.append([
                    MaskConv1d(self.config.block['in_channels'][j], self.config.block['out_channels'][i], kernel_size=1),
                    nn.BatchNorm1d(self.config.block['out_channels'][i], eps=1e-3, momentum=0.1)
                ])
            total_residual_layers.append(residual_layers)

        return total_residual_layers
