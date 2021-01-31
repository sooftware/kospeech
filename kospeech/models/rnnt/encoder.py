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

import torch.nn as nn
from torch import Tensor
from typing import Tuple

from kospeech.models.encoder import TransducerEncoder
from kospeech.models.modules import Linear


class EncoderRNNT(TransducerEncoder):
    """
    Encoder of RNN-Transducer.

    Args:
        input_dim (int): dimension of input vector
        hidden_state_dim (int, optional): hidden state dimension of encoder (default: 320)
        output_dim (int, optional): output dimension of encoder and decoder (default: 512)
        num_layers (int, optional): number of encoder layers (default: 4)
        rnn_type (str, optional): type of rnn cell (default: lstm)
        bidirectional (bool, optional): if True, becomes a bidirectional encoder (default: True)

    Inputs: inputs, input_lengths
        inputs (torch.FloatTensor): A input sequence passed to encoder. Typically for inputs this will be a padded
            `FloatTensor` of size ``(batch, seq_length, dimension)``.
        input_lengths (torch.LongTensor): The length of input tensor. ``(batch)``

    Returns:
        (Tensor, Tensor)

        * outputs (torch.FloatTensor): A output sequence of encoder. `FloatTensor` of size
            ``(batch, seq_length, dimension)``
        * hidden_states (torch.FloatTensor): A hidden state of encoder. `FloatTensor` of size
            ``(batch, seq_length, dimension)``
    """
    supported_rnns = {
        'lstm': nn.LSTM,
        'gru': nn.GRU,
        'rnn': nn.RNN,
    }

    def __init__(
            self,
            input_dim: int,
            hidden_state_dim: int,
            output_dim: int,
            num_layers: int,
            rnn_type: str = 'lstm',
            dropout_p: float = 0.2,
            bidirectional: bool = True,
    ):
        super(EncoderRNNT, self).__init__()
        self.hidden_state_dim = hidden_state_dim
        rnn_cell = self.supported_rnns[rnn_type.lower()]
        self.rnn = rnn_cell(
            input_size=input_dim,
            hidden_size=hidden_state_dim,
            num_layers=num_layers,
            bias=True,
            batch_first=True,
            dropout=dropout_p,
            bidirectional=bidirectional,
        )
        self.out_proj = Linear(hidden_state_dim << 1 if bidirectional else hidden_state_dim, output_dim)

    def forward(self, inputs: Tensor, input_lengths: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Forward propagate a `inputs` for  encoder training.

        Args:
            inputs (torch.FloatTensor): A input sequence passed to encoder. Typically for inputs this will be a padded
                `FloatTensor` of size ``(batch, seq_length, dimension)``.
            input_lengths (torch.LongTensor): The length of input tensor. ``(batch)``

        Returns:
            (Tensor, Tensor)

            * outputs (torch.FloatTensor): A output sequence of encoder. `FloatTensor` of size
                ``(batch, seq_length, dimension)``
            * output_lengths (torch.LongTensor): The length of output tensor. ``(batch)``
        """
        inputs = nn.utils.rnn.pack_padded_sequence(inputs.transpose(0, 1), input_lengths.cpu())
        outputs, hidden_states = self.rnn(inputs)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)
        outputs = self.out_proj(outputs.transpose(0, 1))
        return outputs, input_lengths
