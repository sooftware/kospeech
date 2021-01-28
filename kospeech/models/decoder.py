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
from typing import Tuple


class BaseDecoder(nn.Module):
    def __init__(self):
        super(BaseDecoder, self).__init__()

    def update_dropout(self, dropout_p: float) -> None:
        for name, child in self.named_children():
            if isinstance(child, nn.Dropout):
                child.p = dropout_p

    def get_normalized_probs(self, net_outputs: Tensor):
        assert hasattr(self, "fc"), "self.fc should be defined"
        outputs = self.fc(net_outputs)
        outputs = F.log_softmax(outputs, dim=-1)
        return outputs

    def forward(self, *args, **kwargs):
        raise NotImplementedError


class DecoderRNN(BaseDecoder):
    supported_rnns = {
        'lstm': nn.LSTM,
        'gru': nn.GRU,
        'rnn': nn.RNN,
    }

    def __init__(
            self,
            input_dim: int,                   # size of input
            hidden_state_dim: int = 512,      # dimension of RNN`s hidden state vector
            num_layers: int = 1,              # number of recurrent layers
            rnn_type: str = 'lstm',           # number of RNN layers
            dropout_p: float = 0.3,           # dropout probability
            bidirectional: bool = True,       # if True, becomes a bidirectional rnn
    ):
        super(DecoderRNN, self).__init__()
        rnn_cell = self.supported_rnns[rnn_type]
        self.rnn = rnn_cell(input_dim, hidden_state_dim, num_layers, True, True, dropout_p, bidirectional)
        self.hidden_state_dim = hidden_state_dim

    def _forward(self, inputs: Tensor, hidden_states: Tensor) -> Tuple[Tensor, Tensor]:
        if self.training:
            self.rnn.flatten_parameters()
        return self.rnn(inputs, hidden_states)

    def forward(self, *args, **kwargs):
        raise NotImplementedError
