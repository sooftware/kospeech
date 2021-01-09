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
from kospeech.models.jasper.sublayers import (
    JasperSubBlock, 
    JasperBlock,
)
from kospeech.models.modules import MaskConv1d


class Jasper10x5EncoderConfig:
    preprocess_block = {
        'in_channels': 1,
        'out_channels': 256,
        'kernel_size': 11,
        'stride': 2,
        'dilation': 1,
        'dropout_p': 0.2,
    }
    block = {
        'in_channels': (256, 256, 256, 384, 384, 512, 512, 640, 640, 768),
        'out_channels': (256, 256, 384, 384, 512, 512, 640, 640, 768, 768),
        'kernel_size': (11, 11, 13, 13, 17, 17, 21, 21, 25, 25),
        'dilation': [1] * 10,
        'dropout_p': (0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3, 0.3),
    }


class JasperEncoder(nn.Module):
    supported_versions = {
        '10x5': {
            'num_blocks': 10,
            'num_sub_blocks': 5,
            'config': Jasper10x5EncoderConfig()
        }
    }

    def __init__(self, version: str = '10x5'):
        super(JasperEncoder, self).__init__()
        assert version.lower() in ['10x5'], "Unsupported Version: {}".format(version)
        
        num_blocks = self.supported_versions[version]['num_blocks']
        num_sub_blocks = self.supported_versions[version]['num_sub_blocks']
        config = self.supported_versions[version]['config']

        layers = list()
        layers.append(JasperSubBlock(
            in_channels=config.preprocess_block['in_channels'],
            out_channels=config.preprocess_block['out_channels'],
            kernel_size=config.preprocess_block['kernel_size'],
            stride=config.preprocess_block['stride'],
            dilation=config.preprocess_block['dilation'],
            dropout_p=config.preprocess_block['dropout_p'],
            bias=True,
        ))
        for i in range(num_blocks):
            layers.append(JasperBlock(
                num_sub_blocks=num_sub_blocks,
                in_channels=config.block['in_channels'][i],
                out_channels=config.block['out_channels'][i],
                kernel_size=config.block['kernel_size'][i],
                dilation=config.block['dilation'][i],
                dropout_p=config.block['dropout_p'][i],
                bias=True
            ))
        self.layers = nn.ModuleList(layers)

        self.residual_conv_layers = [
            MaskConv1d(
                in_channels=self.config.block['in_channels'][i],
                out_channels=self.config.block['out_channels'][i],
                kernel_size=1,
                bias=True,
            ) for i in range(num_blocks)
        ]
        self.residual_bn_layers = [
            nn.BatchNorm1d(self.config.block['out_channels'][i]) for i in range(num_blocks)
        ]

    def forward(self, inputs: Tensor, input_lengths: Tensor) -> Tuple[Tensor, Tensor]:
        # TODO: Residual DenseNet
        # 구조 다시 생각해봐야 할듯
        # 누적애서 Residual 하려면 어떻게 해야할지?
        # 어떻게 해야 깔끔할지?
        residuals, residual_lengths = list(), list()
        residuals.append(inputs)
        residual_lengths.append(input_lengths)

        for layer, conv, bn in zip(self.layers, self.residual_conv_layers, self.residual_bn_layers):
            residual = bn(conv(residual, input_lengths)[0])
            output, output_lengths = layer(residuals[-1], input_lengths, residual)

            outputs.append(output)
            output_lengths.append(output_lengths)

        return inputs, input_lengths
