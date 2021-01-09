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
        """
        inputs: BxTxD
        input_lengths: B
        """
        encoder_outputs, output_lengths = self.encoder(inputs, input_lengths)
        output, output_lengths = self.decoder(encoder_outputs, output_lengths)
        return output, output_lengths
