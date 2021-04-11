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

from kospeech.models.convolution import MaskConv1d
from kospeech.models.model import EncoderModel
from kospeech.models.jasper.sublayers import (
    JasperSubBlock,
    JasperBlock,
)
from kospeech.models.jasper.configs import (
    Jasper10x5Config,
    Jasper5x3Config,
)


class Jasper(EncoderModel):
    """
    Jasper: An End-to-End Convolutional Neural Acoustic Model
    Jasper (Just Another Speech Recognizer), an ASR model comprised of 54 layers proposed by NVIDIA.
    Jasper achieved sub 3 percent word error rate (WER) on the LibriSpeech dataset.
    More details: https://arxiv.org/pdf/1904.03288.pdf

    Args:
        num_classes (int): number of classification
        version (str): version of jasper. Marked as BxR: B - number of blocks, R - number of sub-blocks
        device (torch.device): device - 'cuda' or 'cpu'

    Inputs: inputs, input_lengths, residual
        - **inputs**: tensor contains input sequence vector
        - **input_lengths**: tensor contains sequence lengths

    Returns: output, output_lengths
        - **output**: tensor contains output sequence vector
        - **output**: tensor contains output sequence lengths
    """

    def __init__(self, num_classes: int, version: str = '10x5', device: torch.device = 'cuda') -> None:
        super(Jasper, self).__init__()
        supported_versions = {
            '10x5': Jasper10x5Config(num_classes, num_blocks=10, num_sub_blocks=5),
            '5x3': Jasper5x3Config(num_classes, num_blocks=5, num_sub_blocks=3),
        }
        assert version.lower() in supported_versions.keys(), "Unsupported Version: {}".format(version)

        self.config = supported_versions[version]
        self.device = device
        self.layers = nn.ModuleList()
        self.layers.append(
            JasperSubBlock(
                in_channels=self.config.preprocess_block['in_channels'],
                out_channels=self.config.preprocess_block['out_channels'],
                kernel_size=self.config.preprocess_block['kernel_size'],
                stride=self.config.preprocess_block['stride'],
                dilation=self.config.preprocess_block['dilation'],
                dropout_p=self.config.preprocess_block['dropout_p'],
                activation='relu',
                bias=False,
            ).to(self.device)
        )
        self.layers.extend([
            JasperBlock(
                num_sub_blocks=self.config.num_sub_blocks,
                in_channels=self.config.block['in_channels'][i],
                out_channels=self.config.block['out_channels'][i],
                kernel_size=self.config.block['kernel_size'][i],
                dilation=self.config.block['dilation'][i],
                dropout_p=self.config.block['dropout_p'][i],
                activation='relu',
                bias=False,
            ).to(self.device) for i in range(self.config.num_blocks)
        ])
        self.postprocess_layers = nn.ModuleList([
            JasperSubBlock(
                in_channels=self.config.postprocess_block['in_channels'][i],
                out_channels=self.config.postprocess_block['out_channels'][i],
                kernel_size=self.config.postprocess_block['kernel_size'][i],
                dilation=self.config.postprocess_block['dilation'][i],
                dropout_p=self.config.postprocess_block['dropout_p'][i],
                activation='relu',
                bias=True if i == 2 else False,
            ).to(self.device) for i in range(3)
        ])
        self.residual_connections = self._create_jasper_dense_residual_connections(self.config.num_blocks)

    def forward(self, inputs: Tensor, input_lengths: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Forward propagate a `inputs` for  ctc training.

        Args:
            inputs (torch.FloatTensor): A input sequence passed to encoder. Typically for inputs this will be a padded
                `FloatTensor` of size ``(batch, seq_length, dimension)``.
            input_lengths (torch.LongTensor): The length of input tensor. ``(batch)``

        Returns:
            (Tensor, Tensor):

            * predicted_log_prob (torch.FloatTensor)s: Log probability of model predictions.
            * output_lengths (torch.LongTensor): The length of output tensor ``(batch)``
        """
        residual, prev_outputs, prev_output_lengths = None, list(), list()
        inputs = inputs.transpose(1, 2)

        for i, layer in enumerate(self.layers[:-1]):
            inputs, input_lengths = layer(inputs, input_lengths, residual)
            prev_outputs.append(inputs)
            prev_output_lengths.append(input_lengths)
            residual = self._get_jasper_dencse_residual(prev_outputs, prev_output_lengths, i)

        outputs, output_lengths = self.layers[-1](inputs, input_lengths, residual)

        for i, layer in enumerate(self.postprocess_layers):
            outputs, output_lengths = layer(outputs, output_lengths)

        outputs = F.log_softmax(outputs.transpose(1, 2), dim=-1)

        return outputs, output_lengths

    def _get_jasper_dencse_residual(self, prev_outputs: list, prev_output_lengths: list, index: int):
        residual = None

        for item in zip(prev_outputs, prev_output_lengths, self.residual_connections[index]):
            prev_output, prev_output_length, residual_modules = item
            conv1x1, batch_norm = residual_modules

            if residual is None:
                residual = conv1x1(prev_output, prev_output_length)[0]
            else:
                residual += conv1x1(prev_output, prev_output_length)[0]

            residual = batch_norm(residual)

        return residual

    def _create_jasper_dense_residual_connections(self, num_blocks: int) -> nn.ModuleList:
        residual_connections = nn.ModuleList()

        for i in range(num_blocks):
            residual_modules = nn.ModuleList()
            for j in range(i + 1):
                residual_modules.append(nn.ModuleList([
                    MaskConv1d(self.config.block['in_channels'][j], self.config.block['out_channels'][i], kernel_size=1),
                    nn.BatchNorm1d(self.config.block['out_channels'][i], eps=1e-03, momentum=0.1),
                ]))
            residual_connections.append(residual_modules)

        return residual_connections
