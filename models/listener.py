"""
Copyright 2020- Kai.Lib
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import torch.nn as nn
import torch

class PyramidalRNN(nn.Module):
    """ Pyramidal RNN for time resolution reduction """
    def __init__(self, rnn_cell, input_size, hidden_size, dropout_p):
        super(PyramidalRNN, self).__init__()
        self.rnn_cell = nn.LSTM if rnn_cell.lower() == 'lstm' else nn.GRU if rnn_cell.lower() == 'gru' else nn.RNN
        self.rnn = self.rnn_cell(input_size = input_size * 2, hidden_size = hidden_size, num_layers=1,
                                 bidirectional = True, bias = True, batch_first = True, dropout = dropout_p)

    def forward(self, inputs):
        batch_size = inputs.size(0)
        seq_len = inputs.size(1)
        input_size = inputs.size(2)
        if seq_len % 2:
            zeros = torch.zeros((inputs.size(0), 1, inputs.size(2)))
            inputs = torch.cat([inputs, zeros], dim = 1)
            seq_len += 1
        inputs = inputs.contiguous().view(batch_size, int(seq_len / 2), input_size * 2)
        output, hidden = self.rnn(inputs)
        return output, hidden

    def flatten_parameters(self):
        self.rnn.flatten_parameters()

class Listener(nn.Module):
    """
    Converts low level speech signals into higher level features

    Args:
        hidden_size (int): the number of features in the hidden state `h`
        dropout_p (float, optional): dropout probability for the output sequence (default: 0)
        layer_size (int, optional): number of recurrent layers (default: 1)
        bidirectional (bool, optional): if True, becomes a bidirectional encoder (defulat False)
        rnn_cell (str, optional): type of RNN cell (default: gru)

    Outputs: output, hidden
        - **output** (batch, seq_len, hidden_size): tensor containing the encoded features of the input sequence
        - **hidden** (num_layers * num_directions, batch, hidden_size): tensor containing the features in the hidden state `h`
    """

    def __init__(self, feat_size, hidden_size, dropout_p=0.5, layer_size=5, bidirectional=True, rnn_cell='gru', use_pyramidal = False):
        super(Listener, self).__init__()
        self.use_pyramidal = use_pyramidal
        self.rnn_cell = nn.LSTM if rnn_cell.lower() == 'lstm' else nn.GRU if rnn_cell.lower() == 'gru' else nn.RNN
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1, bias=True),
            nn.Hardtanh(0, 20, inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True),
            nn.Hardtanh(0, 20, inplace=True),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=True),
            nn.Hardtanh(0, 20, inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=True),
            nn.Hardtanh(0, 20, inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=True),
            nn.Hardtanh(0, 20, inplace=True),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2, 2)
        )

        """ math :: feat_size = (in_channel * out_channel) / maxpool_layer_num """
        if feat_size % 2:
            feat_size = (feat_size-1) * 64
        else: feat_size *= 64

        if use_pyramidal:
            self.bottom_layer_size = layer_size - 2
            self.bottom_rnn = self.rnn_cell(input_size=feat_size, hidden_size=hidden_size, num_layers=self.bottom_layer_size,
                                            bias=True, batch_first=True, bidirectional=bidirectional, dropout=dropout_p)
            self.middle_rnn = PyramidalRNN(rnn_cell=rnn_cell, input_size=hidden_size * 2, hidden_size=hidden_size, dropout_p=dropout_p)
            self.top_rnn = PyramidalRNN(rnn_cell=rnn_cell, input_size=hidden_size * 2, hidden_size=hidden_size, dropout_p=dropout_p)
        else:
            self.rnn = self.rnn_cell(input_size=feat_size, hidden_size=hidden_size, num_layers=layer_size,
                                     bias=True, batch_first=True, bidirectional=bidirectional, dropout=dropout_p)


    def forward(self, inputs):
        """ Applies a multi-layer RNN to an input sequence. """
        x = self.conv(inputs.unsqueeze(1))
        x = x.transpose(1, 2)
        x = x.contiguous().view(x.size(0), x.size(1), x.size(2) * x.size(3))

        if self.use_pyramidal:
            if self.training:
                self.bottom_rnn.flatten_parameters()
                self.middle_rnn.flatten_parameters()
                self.top_rnn.flatten_parameters()
            bottom_output = self.bottom_rnn(x)[0]
            middle_output = self.middle_rnn(bottom_output)[0]
            output, hidden = self.top_rnn(middle_output)
        else:
            if self.training:
                self.rnn.flatten_parameters()
            output, hidden = self.rnn(x)

        return output, hidden