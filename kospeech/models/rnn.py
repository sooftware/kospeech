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
import torch.nn.functional as F
from torch import Tensor


class BaseRNN(nn.Module):
    """
    Applies a multi-layer RNN to an input sequence.

    Note:
        Do not use this class directly, use one of the sub classes.

    Args:
        input_size (int): size of input
        hidden_dim (int): dimension of RNN`s hidden state vector
        num_layers (int, optional): number of RNN layers (default: 1)
        bidirectional (bool, optional): if True, becomes a bidirectional RNN (defulat: False)
        rnn_type (str, optional): type of RNN cell (default: gru)
        dropout_p (float, optional): dropout probability (default: 0)

    Attributes:
          supported_rnns = Dictionary of supported rnns
    """
    supported_rnns = {
        'lstm': nn.LSTM,
        'gru': nn.GRU,
        'rnn': nn.RNN,
    }

    def __init__(
            self,
            input_size: int,                       # size of input
            hidden_state_dim: int = 512,           # dimension of RNN`s hidden state vector
            num_layers: int = 1,                   # number of recurrent layers
            rnn_type: str = 'lstm',                # number of RNN layers
            dropout_p: float = 0.3,                # dropout probability
            bidirectional: bool = True,            # if True, becomes a bidirectional rnn
    ) -> None:
        super(BaseRNN, self).__init__()
        rnn_cell = self.supported_rnns[rnn_type]
        self.rnn = rnn_cell(input_size, hidden_state_dim, num_layers, True, True, dropout_p, bidirectional)
        self.hidden_state_dim = hidden_state_dim

    def forward(self, *args, **kwargs):
        raise NotImplementedError


class BNReluRNN(BaseRNN):
    """
    Recurrent neural network with batch normalization layer & ReLU activation function.

    Args:
        input_size (int): size of input
        hidden_state_dim (int): the number of features in the hidden state `h`
        rnn_type (str, optional): type of RNN cell (default: gru)
        bidirectional (bool, optional): if True, becomes a bidirectional encoder (defulat: True)
        dropout_p (float, optional): dropout probability (default: 0.1)

    Inputs: inputs, input_lengths
        - **inputs** (batch, time, dim): Tensor containing input vectors
        - **input_lengths**: Tensor containing containing sequence lengths

    Returns: outputs
        - **outputs**: Tensor produced by the BNReluRNN module
    """
    def __init__(
            self,
            input_size: int,                    # size of input
            hidden_state_dim: int = 512,        # dimension of RNN`s hidden state
            rnn_type: str = 'gru',              # type of RNN cell
            bidirectional: bool = True,         # if True, becomes a bidirectional rnn
            dropout_p: float = 0.1,             # dropout probability
    ):
        super(BNReluRNN, self).__init__(input_size=input_size, hidden_state_dim=hidden_state_dim, num_layers=1,
                                        rnn_type=rnn_type, dropout_p=dropout_p, bidirectional=bidirectional)
        self.batch_norm = nn.BatchNorm1d(input_size)

    def forward(self, inputs: Tensor, input_lengths: Tensor):
        total_length = inputs.size(0)

        inputs = F.relu(self.batch_norm(inputs.transpose(1, 2)))
        inputs = inputs.transpose(1, 2)

        outputs = nn.utils.rnn.pack_padded_sequence(inputs, input_lengths.cpu())
        outputs, hidden_states = self.rnn(outputs)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, total_length=total_length)

        return outputs
