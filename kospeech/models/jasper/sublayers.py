import torch.nn as nn

from typing import Optional, Tuple
from torch import Tensor
from kospeech.models.modules import MaskConv1d


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
        self.layers = [
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

    def forward(self, inputs: Tensor, input_lengths: Tensor, residual: Tensor) -> Tuple[Tensor, Tensor]:
        for layer in self.layers[:-1]:
            inputs, input_lengths = layer(inputs, input_lengths)

        output, output_lengths = self.layers[-1](inputs, input_lengths, residual)

        return output, output_lengths


class JasperSubBlock(nn.Module):
    supported_activations = {
        'hardtanh': nn.Hardtanh(0, 20, inplace=True),
        'relu': nn.ReLU(inplace=True),
        'elu': nn.ELU(inplace=True),
        'leaky_relu': nn.LeakyReLU(inplace=True),
        'gelu': nn.GELU()
    }

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int = 1,
            dilation: int = 1,
            bias: bool = True,
            dropout_p: float = 0.2,
            activation: str = 'relu',
    ) -> None:
        super(JasperSubBlock, self).__init__()

        self.conv = MaskConv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            bias=bias,
            dilation=dilation,
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.activation = self.supported_activations[activation]
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, inputs: Tensor, input_lengths: Tensor, residual: Optional = None) -> Tuple[Tensor, Tensor]:
        output, output_lengths = self.conv(inputs, input_lengths)
        output = self.bn(output)

        if residual is not None:
            output += residual

        output = self.dropout(self.activation(output))
        del input_lengths

        return output, output_lengths
