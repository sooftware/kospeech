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
