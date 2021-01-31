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

import torch.nn as nn
from torch import Tensor
from typing import Tuple, Optional

from kospeech.models.encoder import BaseEncoder


class EncoderRNN(BaseEncoder):
    """
    Converts low level speech signals into higher level features

    Args:
        input_dim (int): dimension of input vector
        num_classes (int): number of classification
        hidden_state_dim (int): the number of features in the encoder hidden state `h`
        num_layers (int, optional): number of recurrent layers (default: 3)
        bidirectional (bool, optional): if True, becomes a bidirectional encoder (default: False)
        extractor (str): type of CNN extractor (default: vgg)
        activation (str): type of activation function (default: hardtanh)
        rnn_type (str, optional): type of RNN cell (default: lstm)
        dropout_p (float, optional): dropout probability of encoder (default: 0.2)
        joint_ctc_attention (bool, optional): flag indication joint ctc attention or not

    Inputs: inputs, input_lengths
        - **inputs**: list of sequences, whose length is the batch size and within which each sequence is list of tokens
        - **input_lengths**: list of sequence lengths

    Returns: encoder_outputs, encoder_log__probs, output_lengths
        - **encoder_outputs**: tensor containing the encoded features of the input sequence
        - **encoder_log__probs**: tensor containing log probability for ctc loss
        - **output_lengths**: list of sequence lengths produced by Listener
    """
    supported_rnns = {
        'lstm': nn.LSTM,
        'gru': nn.GRU,
        'rnn': nn.RNN,
    }

    def __init__(
            self,
            input_dim: int,
            num_classes: int = None,
            hidden_state_dim: int = 512,
            dropout_p: float = 0.3,
            num_layers: int = 3,
            bidirectional: bool = True,
            rnn_type: str = 'lstm',
            extractor: str = 'vgg',
            activation: str = 'hardtanh',
            joint_ctc_attention: bool = False,
    ) -> None:
        super(EncoderRNN, self).__init__(input_dim=input_dim, extractor=extractor, d_model=hidden_state_dim << 1,
                                         num_classes=num_classes, dropout_p=dropout_p, activation=activation,
                                         joint_ctc_attention=joint_ctc_attention)
        self.hidden_state_dim = hidden_state_dim
        rnn_cell = self.supported_rnns[rnn_type.lower()]
        self.rnn = rnn_cell(
            input_size=self.conv_output_dim,
            hidden_size=hidden_state_dim,
            num_layers=num_layers,
            bias=True,
            batch_first=True,
            dropout=dropout_p,
            bidirectional=bidirectional,
        )

    def forward(self, inputs: Tensor, input_lengths: Tensor) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        """
        Forward propagate a `inputs` for  encoder training.

        Args:
            inputs (torch.FloatTensor): A input sequence passed to encoder. Typically for inputs this will be a padded
                `FloatTensor` of size ``(batch, seq_length, dimension)``.
            input_lengths (torch.LongTensor): The length of input tensor. ``(batch)``

        Returns:
            (Tensor, Tensor, Tensor):

            * encoder_outputs: A output sequence of encoder. `FloatTensor` of size ``(batch, seq_length, dimension)``
            * encoder_output_lengths: The length of encoder outputs. ``(batch)``
            * encoder_log_probs: Log probability of encoder outputs will be passed to CTC Loss.
                If joint_ctc_attention is False, return None.
        """
        encoder_log_probs = None
        features, output_lengths = self.conv(inputs, input_lengths)

        features = nn.utils.rnn.pack_padded_sequence(features.transpose(0, 1), output_lengths.cpu())
        encoder_outputs, hidden_states = self.rnn(features)
        encoder_outputs, _ = nn.utils.rnn.pad_packed_sequence(encoder_outputs)
        encoder_outputs = encoder_outputs.transpose(0, 1)

        if self.joint_ctc_attention:
            encoder_log_probs = self.fc(encoder_outputs.transpose(1, 2)).log_softmax(dim=2)

        return encoder_outputs, output_lengths, encoder_log_probs
