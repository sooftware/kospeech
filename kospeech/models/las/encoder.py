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
from typing import Tuple, Optional

from kospeech.models.encoder import EncoderRNN
from kospeech.models.modules import Linear, Transpose
from kospeech.models.conv import (
    VGGExtractor,
    DeepSpeech2Extractor,
    Conv2dSubsampling,
)


class Listener(EncoderRNN):
    """
    Converts low level speech signals into higher level features

    Args:
        input_dim (int): dimension of input vector
        hidden_state_dim (int): the number of features in the hidden state `h`
        num_layers (int, optional): number of recurrent layers (default: 1)
        bidirectional (bool, optional): if True, becomes a bidirectional encoder (defulat: False)
        rnn_type (str, optional): type of RNN cell (default: lstm)
        dropout_p (float, optional): dropout probability (default: 0.3)
        extractor (str): type of CNN extractor (default: vgg)
        activation (str): type of activation function (default: hardtanh)

    Inputs: inputs, input_lengths
        - **inputs**: list of sequences, whose length is the batch size and within which each sequence is list of tokens
        - **input_lengths**: list of sequence lengths

    Returns: encoder_outputs, encoder_log__probs, output_lengths
        - **encoder_outputs**: tensor containing the encoded features of the input sequence
        - **encoder_log__probs**: tensor containing log probability for ctc loss
        - **output_lengths**: list of sequence lengths produced by Listener
    """
    def __init__(
            self,
            input_dim: int,                          # size of input
            num_classes: int = None,                 # number of class
            hidden_state_dim: int = 512,             # dimension of RNN`s hidden state
            dropout_p: float = 0.3,                  # dropout probability
            num_layers: int = 3,                     # number of RNN layers
            bidirectional: bool = True,              # if True, becomes a bidirectional encoder
            rnn_type: str = 'lstm',                  # type of RNN cell
            extractor: str = 'vgg',                  # type of CNN extractor
            activation: str = 'hardtanh',            # type of activation function
            joint_ctc_attention: bool = False,       # Use CTC Loss & Cross Entropy Joint Learning
    ) -> None:
        if extractor.lower() == 'vgg':
            conv = VGGExtractor(input_dim, activation=activation)
        elif extractor.lower() == 'ds2':
            conv = DeepSpeech2Extractor(input_dim, activation=activation)
        elif extractor.lower() == 'conv2d':
            conv = Conv2dSubsampling(input_dim, 1, hidden_state_dim, activation=activation)
        else:
            raise ValueError("Unsupported Extractor : {0}".format(extractor))

        super(Listener, self).__init__(
            conv.get_output_dim(), hidden_state_dim, num_layers, rnn_type, dropout_p, bidirectional,
        )

        self.conv = conv
        self.joint_ctc_attention = joint_ctc_attention

        if self.joint_ctc_attention:
            self.fc = nn.Sequential(
                nn.BatchNorm1d(self.hidden_state_dim << 1),
                Transpose(shape=(1, 2)),
                nn.Dropout(dropout_p),
                Linear(self.hidden_state_dim << 1, num_classes, bias=False),
            )

    def forward(self, inputs: Tensor, input_lengths: Tensor) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        """
        inputs (torch.FloatTensor): (batch_size, sequence_length, dimension)
        input_lengths (torch.LongTensor): (batch_size)
        """
        encoder_log_probs = None
        conv_outputs, output_lengths = self.conv(inputs, input_lengths)
        encoder_outputs = self._forward(conv_outputs, output_lengths)

        if self.joint_ctc_attention:
            encoder_log_probs = self.get_normalized_probs(encoder_outputs.transpose(1, 2))
        return encoder_outputs, encoder_log_probs, output_lengths
