# -*- coding: utf-8 -*-
# Soohwan Kim, Seyoung Bae, Cheolhwang Won.
# @ArXiv : KoSpeech: Open-Source Toolkit for End-to-End Korean Speech Recognition
# This source code is licensed under the Apache 2.0 License license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from kospeech.models.extractor import DeepSpeech2Extractor
from kospeech.models.modules import BaseRNN, Linear


class BatchNormRNN(BaseRNN):
    def __init__(
            self,
            input_size: int,
            hidden_dim: int = 512,
            rnn_type: str = 'gru',
            bidirectional: bool = True,
            dropout_p: float = 0.1,
            device: str = 'cuda'
    ):
        super(BatchNormRNN, self).__init__(
            input_size=input_size,
            hidden_dim=hidden_dim,
            num_layers=1,
            rnn_type=rnn_type,
            dropout_p=dropout_p,
            bidirectional=bidirectional,
            device=device
        )
        self.batch_norm = nn.BatchNorm1d(input_size)

    def forward(self, inputs: Tensor, seq_lengths: Tensor):
        inputs = self.batch_norm(inputs)
        inputs = nn.utils.rnn.pack_padded_sequence(inputs, seq_lengths)

        output, hidden = self.rnn(inputs)
        output, _ = nn.utils.rnn.pad_packed_sequence(output)
        output = output.transpose(0, 1)

        return output


class DeepSpeech2(nn.Module):
    """
    Deep Speech2 architecture with configurable encoder and decoder.
    Paper: https://arxiv.org/abs/1512.02595
    """
    def __init__(
            self,
            input_size: int,
            num_classes: int,
            rnn_type='gru',
            num_rnn_layers: int = 5,
            rnn_hidden_dim: int = 512,
            dropout_p: float = 0.1,
            bidirectional: bool = True,
            activation: str = 'hardtanh',
            device: str = 'cuda'
    ):
        super(DeepSpeech2, self).__init__()

        input_size = int(math.floor(input_size + 2 * 20 - 41) / 2 + 1)
        input_size = int(math.floor(input_size + 2 * 10 - 21) / 2 + 1)
        input_size <<= 5
        self.conv = DeepSpeech2Extractor(activation, mask_conv=True)
        self.rnn_layers = list()

        for idx in range(num_rnn_layers):
            self.rnn_layers.append(BatchNormRNN(
                input_size=input_size if idx == 0 else rnn_hidden_dim << 1,
                hidden_dim=rnn_hidden_dim,
                rnn_type=rnn_type,
                bidirectional=bidirectional,
                dropout_p=dropout_p,
                device=device
            ))

        self.fc = nn.Sequential(
            nn.BatchNorm1d(rnn_hidden_dim << 1),
            Linear(rnn_hidden_dim << 1, rnn_hidden_dim, bias=False),
            nn.ReLU(),
            Linear(rnn_hidden_dim, num_classes)
        )

    def forward(self, inputs: Tensor, input_lengths: Tensor):
        inputs = inputs.unsqueeze(1).permute(0, 1, 3, 2)
        output, seq_lengths = self.conv(inputs, input_lengths)

        batch_size, num_channels, hidden_dim, seq_length = output.size()
        output = output.view(batch_size, num_channels * hidden_dim, seq_length).permute(2, 0, 1).contiguous()

        for rnn_layer in self.rnn_layers:
            output = rnn_layer(output, seq_lengths)

        output = F.log_softmax(self.fc(output), dim=-1)

        return output
