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
from kospeech.models.modules import MaskConv1d, BatchNorm1d
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
        layers = list()
        layers.append(JasperSubBlock(
            in_channels=config.preprocess_block['in_channels'],
            out_channels=config.preprocess_block['out_channels'],
            kernel_size=config.preprocess_block['kernel_size'],
            stride=config.preprocess_block['stride'],
            dilation=config.preprocess_block['dilation'],
            dropout_p=config.preprocess_block['dropout_p'],
            activation='relu',
            bias=False,
        ))
        for i in range(config.num_blocks):
            layers.append(JasperBlock(
                num_sub_blocks=config.num_sub_blocks,
                in_channels=config.block['in_channels'][i],
                out_channels=config.block['out_channels'][i],
                kernel_size=config.block['kernel_size'][i],
                dilation=config.block['dilation'][i],
                dropout_p=config.block['dropout_p'][i],
                activation='relu',
                bias=False,
            ))
        self.layers = nn.ModuleList(layers)
        self.total_residual_layers = self._get_residual_layers(config.num_blocks)

    def forward(self, inputs: Tensor, input_lengths: Tensor) -> Tuple[Tensor, Tensor]:
        prev_outputs, prev_output_lengths = list(), list()
        residual = None
        output, output_lengths = inputs, input_lengths

        for layer in self.layerss:
            output, output_lengths = layer(output, output_lengths, residual)
            prev_outputs.append(output)
            prev_output_lengths.append(output_lengths)

            for item in zip(prev_outputs, prev_output_lengths, self.total_residual_layers):
                prev_output, prev_output_length, residual_layers = item
                for residual_layer in residual_layers:
                    residual, _ = residual_layer(prev_output, prev_output_length)

        return inputs, input_lengths

    def _get_residual(self, prev_outputs: list, prev_output_lengths: list):
        residual = None

        for item in zip(prev_outputs, prev_output_lengths, self.total_residual_layers):
            prev_output, prev_output_length, residual_layers = item
            for residual_layer in residual_layers:
                if residual is None:
                    residual = residual_layer(prev_output, prev_output_length)[0]
                else:
                    residual += residual_layer(prev_output, prev_output_length)[0]

        return residual

    def _get_residual_layers(self, num_blocks: int):
        total_residual_layers = list()

        for i in range(num_blocks - 1):
            residual_layers = list()
            for j in range(i + 1):
                residual_layers.append(nn.Sequential(
                    MaskConv1d(self.config.block['in_channels'][j], self.config.block['out_channels'][i], kernel_size=1),
                    BatchNorm1d(self.config.block['out_channels'][i], eps=1e-3, momentum=0.1)
                ))
            total_residual_layers.append(residual_layers)

        return total_residual_layers
