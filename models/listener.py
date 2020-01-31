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

class Listener(nn.Module):
    """
    Applies a multi-layer RNN to an input sequence.

    Args:
        hidden_size (int): the number of features in the hidden state `h`
        dropout_p (float, optional): dropout probability for the output sequence (default: 0)
        layer_size (int, optional): number of recurrent layers (default: 1)
        bidirectional (bool, optional): if True, becomes a bidirectional encoder (defulat False)
        rnn_cell (str, optional): type of RNN cell (default: gru)
        update_embedding (bool, optional): If the embedding should be updated during training (default: False).

    Inputs: inputs, input_lengths
        - **inputs**: list of sequences, whose length is the batch size and within which each sequence is a list of token IDs.
        - **input_lengths** (list of int, optional): list that contains the lengths of sequences
            in the mini-batch, it must be provided when using variable length RNN (default: `None`)

    Outputs: output, hidden
        - **output** (batch, seq_len, hidden_size): tensor containing the encoded features of the input sequence
        - **hidden** (num_layers * num_directions, batch, hidden_size): tensor containing the features in the hidden state `h`
    """

    def __init__(self, feature_size, hidden_size, dropout_p=0.5, layer_size=5, bidirectional=True, rnn_cell='gru', use_pyramidal = False):
        super(Listener, self).__init__()
        if rnn_cell.lower() != 'gru' and rnn_cell.lower() != 'lstm':
            raise ValueError("Unsupported RNN Cell: %s" % rnn_cell)
        self.use_pyramidal = use_pyramidal
        self.rnn_cell = nn.GRU if rnn_cell.lower() == 'gru' else nn.LSTM
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1, bias=False),
            nn.Hardtanh(0, 20, inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.Hardtanh(0, 20, inplace=True),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.Hardtanh(0, 20, inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.Hardtanh(0, 20, inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.Hardtanh(0, 20, inplace=True),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2, 2)
        )
        if feature_size % 2: feature_size = (feature_size-1) * 64
        else: feature_size *= 64
        if use_pyramidal:
            self.bottom_layer_size = layer_size - 2
            self.bottom_rnn = self.rnn_cell(feature_size, hidden_size, self.bottom_layer_size, batch_first=True, bidirectional=bidirectional, dropout=dropout_p)
            self.middle_rnn = self.rnn_cell(hidden_size * 4, hidden_size, 1, batch_first=True, bidirectional=bidirectional, dropout=dropout_p)
            self.top_rnn = self.rnn_cell(hidden_size * 4, hidden_size, 1, batch_first=True, bidirectional=bidirectional, dropout=dropout_p)
        else:
            self.rnn = self.rnn_cell(feature_size, hidden_size, layer_size, batch_first=True, bidirectional=bidirectional, dropout=dropout_p)


    def forward(self, inputs):
        """
        Applies a multi-layer RNN to an input sequence.

        Args:
            inputs (batch, seq_len): tensor containing the features of the input sequence.

        Returns: output, hidden
            - **output** (batch, seq_len, hidden_size): variable containing the encoded features of the input sequence
                          => Ex (32, 257, 512)
            - **hidden** (num_layers * num_directions, batch, hidden_size): variable containing the features in the hidden state h
                          => Ex (16, 32, 512)
        """

        # Before : (batch_size, seq_len, feature_size)
        # After  : (batch_size, 1(in_channel), seq_len, feature_size)
        inputs = inputs.unsqueeze(1)
        # Before : (batch_size, 1, seq_len, feature_size)
        # After  : (batch_size, out_channel, seq_len / 4 , feature_size / 4) 4는 MaxPool2d x 2번
        x = self.conv(inputs)
        # Before : (batch_size, out_channel, seq_len, feature_size)
        # After  : (batch_size, seq_len, out_channel, feature_size)
        x = x.transpose(1, 2)
        # 메모리에 contiguous 하게 저장 ( torch.view() 사용시 필요 )
        x = x.contiguous()
        # x`s shape
        sizes = x.size()
        # Dimenstion Synchronization
        # Before : (batch_size, seq_len, out_channel, feature_size)
        # After  : (batch_size, seq_len, out_channel * feature_size)
        x = x.view(sizes[0], sizes[1], sizes[2] * sizes[3])

        if self.training:
            if self.use_pyramidal:
                self.bottom_rnn.flatten_parameters()
                self.middle_rnn.flatten_parameters()
                self.top_rnn.flatten_parameters()
            else:
                self.rnn.flatten_parameters()
        # Apply pBLSTM
        if self.use_pyramidal:
            bottom_outputs, _ = self.bottom_rnn(x)
            middle_inputs = self._make_pyramid(bottom_outputs)
            middle_outputs, _ = self.middle_rnn(middle_inputs)
            top_inputs = self._make_pyramid(middle_outputs)
            outputs, hiddens = self.top_rnn(top_inputs)
            del bottom_outputs, middle_inputs, middle_outputs, top_inputs
        else:
            outputs, hiddens = self.rnn(x)
        return outputs, hiddens

    def _make_pyramid(self, h_outputs):
        """
        Inputs:
            - **h_outputs**: (batch, seq_len, hidden_size * direction)
        Outputs:
            - **output**: (batch, seq_len / 2, hidden_size * direction * 2)
        """
        if h_outputs.size(1) % 2:
            zeros = torch.zeros((h_outputs.size(0), 1, h_outputs.size(2)))
            h_outputs = torch.cat([h_outputs, zeros], 1)
        return torch.cat([h_outputs[:, 0::2], h_outputs[:, 1::2]], 2)