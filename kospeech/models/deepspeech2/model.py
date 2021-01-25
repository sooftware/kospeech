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

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple
from kospeech.models.conv import DeepSpeech2Extractor
from kospeech.models.modules import Linear, BNReluRNN


class DeepSpeech2(nn.Module):
    """
    Deep Speech2 model with configurable encoder and decoder.
    Paper: https://arxiv.org/abs/1512.02595

    Args:
        input_size (int): size of input
        num_classes (int): number of classfication
        rnn_type (str, optional): type of RNN cell (default: gru)
        num_rnn_layers (int, optional): number of recurrent layers (default: 5)
        rnn_hidden_dim (int): the number of features in the hidden state `h`
        dropout_p (float, optional): dropout probability (default: 0.1)
        bidirectional (bool, optional): if True, becomes a bidirectional encoder (defulat: True)
        activation (str): type of activation function (default: hardtanh)
        device (torch.device): device - 'cuda' or 'cpu'

    Inputs: inputs, input_lengths
        - **inputs**: list of sequences, whose length is the batch size and within which each sequence is list of tokens
        - **input_lengths**: list of sequence lengths

    Returns: output
        - **output**: tensor containing the encoded features of the input sequence
    """
    def __init__(
            self,
            input_size: int,                        # size of input
            num_classes: int,                       # number of classfication
            rnn_type='gru',                         # type of RNN cell
            num_rnn_layers: int = 5,                # number of RNN layers
            rnn_hidden_dim: int = 512,              # dimension of RNN`s hidden state
            dropout_p: float = 0.1,                 # dropout probability
            bidirectional: bool = True,             # if True, becomes a bidirectional rnn
            activation: str = 'hardtanh',           # type of activation function
            device: torch.device = 'cuda'           # device - 'cuda' or 'cpu'
    ):
        super(DeepSpeech2, self).__init__()
        self.rnn_layers = nn.ModuleList()
        self.device = device

        input_size = int(math.floor(input_size + 2 * 20 - 41) / 2 + 1)
        input_size = int(math.floor(input_size + 2 * 10 - 21) / 2 + 1)
        input_size <<= 5
        rnn_output_size = rnn_hidden_dim << 1 if bidirectional else rnn_hidden_dim

        self.conv = DeepSpeech2Extractor(activation, mask_conv=True)

        for idx in range(num_rnn_layers):
            self.rnn_layers.append(BNReluRNN(
                input_size=input_size if idx == 0 else rnn_output_size,
                hidden_dim=rnn_hidden_dim,
                rnn_type=rnn_type,
                bidirectional=bidirectional,
                dropout_p=dropout_p,
                device=device
            ))

        self.fc = nn.Sequential(
            Linear(rnn_output_size, rnn_hidden_dim),
            nn.ReLU(),
            Linear(rnn_hidden_dim, num_classes, bias=False)
        )

    def forward(self, inputs: Tensor, input_lengths: Tensor) -> Tuple[Tensor, Tensor]:
        """
        inputs (torch.FloatTensor): (batch_size, sequence_length, dimension)
        input_lengths (torch.LongTensor): (batch_size)
        """
        inputs = inputs.unsqueeze(1).permute(0, 1, 3, 2)
        outputs, output_lengths = self.conv(inputs, input_lengths)

        batch_size, num_channels, hidden_dim, seq_length = outputs.size()
        outputs = outputs.view(batch_size, num_channels * hidden_dim, seq_length).permute(2, 0, 1).contiguous()

        for rnn_layer in self.rnn_layers:
            rnn_layer.to(self.device)
            outputs = rnn_layer(outputs, output_lengths)

        outputs = outputs.transpose(0, 1)
        outputs = self.fc(outputs)
        outputs = F.log_softmax(outputs, dim=-1)

        return outputs, output_lengths

    def greedy_search(self, inputs: Tensor, input_lengths: Tensor, device: str):
        with torch.no_grad():
            outputs, output_lengths = self.forward(inputs, input_lengths)
            return outputs.max(-1)[1]
