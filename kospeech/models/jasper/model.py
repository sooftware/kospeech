# -*- coding: utf-8 -*-
# Soohwan Kim, Seyoung Bae, Cheolhwang Won.
# @ArXiv : KoSpeech: Open-Source Toolkit for End-to-End Korean Speech Recognition
# This source code is licensed under the Apache 2.0 License license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn

from torch import Tensor
from typing import Tuple
from kospeech.models.jasper.decoder import JasperDecoder
from kospeech.models.jasper.encoder import JasperEncoder


class Jasper(nn.Module):
    """
    Jasper: An End-to-End Convolutional Neural Acoustic Model
    Jasper (Just Another Speech Recognizer), an ASR model comprised of 54 layers proposed by NVIDIA.
    Jasper achieved sub 3 percent word error rate (WER) on the LibriSpeech dataset.
    """
    def __init__(self, num_classes: int, version: str = '10x5') -> None:
        super(Jasper, self).__init__()
        assert version.lower() in ['10x5'], "Unsupported Version: {}".format(version)

        self.encoder = JasperEncoder(version)
        self.decoder = JasperDecoder(num_classes, version)

    def forward(self, inputs: Tensor, input_lengths: Tensor) -> Tuple[Tensor, Tensor]:
        encoder_outputs, output_lengths = self.encoder(inputs, input_lengths)
        output, output_lengths = self.decoder(encoder_outputs, output_lengths)
        return output, output_lengths
