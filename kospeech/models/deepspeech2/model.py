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
from kospeech.models.modules import Linear, BNReluRNN


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
        self.rnn_layers = list()
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

    def forward(self, inputs: Tensor, input_lengths: Tensor):
        inputs = inputs.unsqueeze(1).permute(0, 1, 3, 2)
        output, output_lengths = self.conv(inputs, input_lengths)

        batch_size, num_channels, hidden_dim, seq_length = output.size()
        output = output.view(batch_size, num_channels * hidden_dim, seq_length).permute(2, 0, 1).contiguous()

        for rnn_layer in self.rnn_layers:
            rnn_layer.to(self.device)
            output = rnn_layer(output, output_lengths)

        output = output.transpose(0, 1)
        output = self.fc(output)
        output = F.log_softmax(output, dim=-1)

        return output, output_lengths

    def decode(self, output, blank_label: int) -> Tensor:
        decode_results = list()
        max_prob_indices = torch.argmax(output, dim=-1)

        for i, max_prob_index in enumerate(max_prob_indices):
            decode_result = list()
            for j, label_index in enumerate(max_prob_index):
                if label_index != blank_label:
                    decode_result.append(label_index.item())
            decode_results.append(decode_result)
        return torch.as_tensor(decode_results)

    def inference(self, inputs: Tensor, input_lengths: Tensor, device: str):
        with torch.no_grad():
            output = self.forward(inputs, input_lengths)
            logit = torch.stack(output, dim=1).to(device)
            return logit.max(-1)[1]
