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
    Converts low level speech signals into higher level features

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

    def __init__(self, feat_size, hidden_size, dropout_p=0.5, layer_size=5, bidirectional=True, rnn_cell='gru', use_pyramidal = False):
        super(Listener, self).__init__()
        self.use_pyramidal = use_pyramidal
        self.rnn_cell = nn.LSTM if rnn_cell.lower() == 'lstm' else nn.GRU if rnn_cell.lower() == 'gru' else nn.RNN
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

        """ math :: feat_size = (in_channel * out_channel) / maxpool_layer_num """
        if feat_size % 2: feat_size = (feat_size-1) * 64
        else: feat_size *= 64

        if use_pyramidal:
            self.bottom_layer_size = layer_size - 2
            self.bottom_rnn = self.rnn_cell(input_size=feat_size, hidden_size=hidden_size, num_layers=self.bottom_layer_size,
                                            bias=True, batch_first=True, bidirectional=bidirectional, dropout=dropout_p)
            self.middle_rnn = self.rnn_cell(input_size=hidden_size * 2 * (2 if bidirectional else 1), hidden_size=hidden_size,
                                            num_layers=1, bias=True, batch_first=True, bidirectional=bidirectional, dropout=dropout_p)
            self.top_rnn = self.rnn_cell(input_size=hidden_size * 2 * (2 if bidirectional else 1), hidden_size=hidden_size,
                                         num_layers=1, bias=True, batch_first=True, bidirectional=bidirectional, dropout=dropout_p)
        else:
            self.rnn = self.rnn_cell(input_size=feat_size, hidden_size=hidden_size, num_layers=layer_size,
                                     bias=True, batch_first=True, bidirectional=bidirectional, dropout=dropout_p)


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
        x = self.conv(inputs.unsqueeze(1))
        x = x.transpose(1, 2)
        x = x.contiguous().view(x.size(0), x.size(1), x.size(2) * x.size(3))

        if self.use_pyramidal:
            if self.training:
                self.bottom_rnn.flatten_parameters()
                self.middle_rnn.flatten_parameters()
                self.top_rnn.flatten_parameters()
            bottom_outputs = self.bottom_rnn(x)[0]
            middle_inputs = self._cat_consecutive(bottom_outputs)
            middle_outputs = self.middle_rnn(middle_inputs)[0]
            top_inputs = self._cat_consecutive(middle_outputs)
            outputs, hiddens = self.top_rnn(top_inputs)
        else:
            if self.training:
                self.rnn.flatten_parameters()
            outputs, hiddens = self.rnn(x)

        return outputs, hiddens

    def _cat_consecutive(self, prev_layer_outputs):
        """concatenate the outputs at consecutive steps of  each layer before feeding it to the next layer"""
        if prev_layer_outputs.size(1) % 2:
            """if prev_layer_outputs`s seq_len is odd, concatenate zeros"""
            zeros = torch.zeros((prev_layer_outputs.size(0), 1, prev_layer_outputs.size(2)))
            prev_layer_outputs = torch.cat([prev_layer_outputs, zeros], 1)
        return torch.cat([prev_layer_outputs[:, 0::2], prev_layer_outputs[:, 1::2]], 2)