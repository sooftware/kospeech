# -*- coding: utf-8 -*-
# Soohwan Kim, Seyoung Bae, Cheolhwang Won.
# @ArXiv : KoSpeech: Open-Source Toolkit for End-to-End Korean Speech Recognition
# This source code is licensed under the Apache 2.0 License license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn
from torch import Tensor


class Jasper(nn.Module):
    def __init__(
            self,
            num_blocks: int = 10,
            num_sub_blocks: int = 5,
    ) -> None:
        super(Jasper, self).__init__()
        self.num_blocks = num_blocks
        self.num_sub_blocks = num_sub_blocks

        self.preprocess_block = JasperSubBlock(
            in_channels=1,
            out_channels=256,
            kernel_size=11,
            stride=2,
            bias=True,
            dropout_p=0.2,
        )
        self.block1 = [
            JasperBlock(
                num_sub_blocks=num_sub_blocks,
                in_channels=256,
                out_channels=256,
                kernel_size=11,
                dropout_p=0.2,
            ) for _ in range(2)
        ]
        self.block2 = [
            JasperBlock(
                num_sub_blocks=num_sub_blocks,
                in_channels=256,
                out_channels=384,
                kernel_size=13,
                dropout_p=0.2,
            ) for _ in range(2)
        ]
        self.block3 = [
            JasperBlock(
                num_sub_blocks=num_sub_blocks,
                in_channels=384,
                out_channels=512,
                kernel_size=17,
                dropout_p=0.2,
            ) for _ in range(2)
        ]
        self.block4 = [
            JasperBlock(
                num_sub_blocks=num_sub_blocks,
                in_channels=512,
                out_channels=640,
                kernel_size=21,
                dropout_p=0.3,
            ) for _ in range(2)
        ]
        self.block5 = [
            JasperBlock(
                num_sub_blocks=num_sub_blocks,
                in_channels=640,
                out_channels=768,
                kernel_size=25,
                dropout_p=0.3,
            ) for _ in range(2)
        ]

        # TODO: post-process block


class JasperBlock(nn.Module):
    def __init__(
            self,
            num_sub_blocks: int,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int = 1,
            dilation: int = 1,
            bias: bool = True,
            dropout_p: float = 0.2,
    ) -> None:
        super(JasperBlock, self).__init__()
        self.num_sub_blocks = num_sub_blocks
        self.sub_blocks = [
            JasperSubBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                bias=bias,
                dropout_p=dropout_p
            ) for _ in range(num_sub_blocks)
        ]


class JasperSubBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int = 1,
            dilation: int = 1,
            bias: bool = True,
            dropout_p: float = 0.2,
    ) -> None:
        super(JasperSubBlock, self).__init__()

        # TODO: MaskConv1d

        self.sequential = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, bias=bias, dilation=dilation),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
        )

    def forward(self, inputs: Tensor):
        return self.sequential(inputs)
