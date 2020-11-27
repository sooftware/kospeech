# -*- coding: utf-8 -*-
# Soohwan Kim, Seyoung Bae, Cheolhwang Won.
# @ArXiv : KoSpeech: Open-Source Toolkit for End-to-End Korean Speech Recognition
# This source code is licensed under the Apache 2.0 License license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from kospeech.models.extractor import DeepSpeech2Extractor
from kospeech.models.modules import BaseRNN, Linear


class BatchNormRNN(BaseRNN):
    """
    Recurrent neural network with batch normalization layer.

    Args:
        input_size (int): size of input
        hidden_dim (int): the number of features in the hidden state `h`
        rnn_type (str, optional): type of RNN cell (default: gru)
        bidirectional (bool, optional): if True, becomes a bidirectional encoder (defulat: True)
        dropout_p (float, optional): dropout probability (default: 0.1)
        device (torch.device): device - 'cuda' or 'cpu'

    Inputs: inputs, seq_lengths
        - **inputs**: list of sequences, whose length is the batch size and within which each sequence is list of tokens
        - **seq_lengths**: list of sequence lengths
    """
    def __init__(
            self,
            input_size: int,                    # size of input
            hidden_dim: int = 512,              # dimension of RNN`s hidden state
            rnn_type: str = 'gru',              # type of RNN cell
            bidirectional: bool = True,         # if True, becomes a bidirectional rnn
            dropout_p: float = 0.1,             # dropout probability
            device: str = 'cuda'                # device - 'cuda' or 'cpu'
    ):
        super(BatchNormRNN, self).__init__(input_size=input_size, hidden_dim=hidden_dim, num_layers=1, rnn_type=rnn_type,
                                           dropout_p=dropout_p, bidirectional=bidirectional, device=device)
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
            device: str = 'cuda'                    # device - 'cuda' or 'cpu'
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

    def inference(self, inputs: Tensor, input_lengths: Tensor, blank_label: int):
        hypothesis = list()

        with torch.no_grad():
            output = self.forward(inputs, input_lengths)
            argmax_indices = torch.argmax(output, dim=-1)

            for i, argmax_index in enumerate(argmax_indices):
                decode_result = list()
                for j, token_id in enumerate(argmax_index):
                    if token_id != blank_label:
                        if j != 0 and token_id == argmax_index[j - 1]:
                            continue
                        decode_result.append(token_id.item())
                hypothesis.append(decode_result)
        return hypothesis
