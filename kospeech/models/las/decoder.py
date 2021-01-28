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

import random
import torch
import torch.nn as nn
from torch import Tensor, LongTensor
from typing import Optional, Any, Tuple

from kospeech.models.decoder import DecoderRNN
from kospeech.models.modules import Linear, View
from kospeech.models.attention import (
    LocationAwareAttention,
    MultiHeadAttention,
    AdditiveAttention,
    ScaledDotProductAttention,
)


class Speller(DecoderRNN):
    """
    Converts higher level features (from encoder) into output utterances
    by specifying a probability distribution over sequences of characters.

    Args:
        num_classes (int): number of classification
        max_length (int): a maximum allowed length for the sequence to be processed
        hidden_state_dim (int): dimension of RNN`s hidden state vector
        sos_id (int): index of the start of sentence symbol
        eos_id (int): index of the end of sentence symbol
        attn_mechanism (str): type of attention mechanism (default: dot)
        num_heads (int): number of attention heads. (default: 4)
        num_layers (int, optional): number of recurrent layers (default: 1)
        rnn_type (str, optional): type of RNN cell (default: lstm)
        dropout_p (float, optional): dropout probability (default: 0.3)
        device (torch.device): device - 'cuda' or 'cpu'

    Inputs: inputs, encoder_outputs, teacher_forcing_ratio, return_decode_dict
        - **inputs** (batch, seq_len, input_size): list of sequences, whose length is the batch size and within which
          each sequence is a list of token IDs.  It is used for teacher forcing when provided. (default `None`)
        - **encoder_outputs** (batch, seq_len, hidden_dim): tensor with containing the outputs of the encoder.
          Used for attention mechanism (default is `None`).
        - **teacher_forcing_ratio** (float): The probability that teacher forcing will be used. A random number is
          drawn uniformly from 0-1 for every decoding token, and if the sample is smaller than the given value,
          teacher forcing would be used (default is 0).
        - **return_decode_dict** (dict): dictionary which contains decode informations.

    Returns: predicted_log_probs
        - **predicted_log_probs**: list contains decode result (log probability)
    """

    def __init__(
            self,
            num_classes: int,                        # number of classfication
            max_length: int = 150,                   # a maximum allowed length for the sequence to be processed
            hidden_state_dim: int = 1024,            # dimension of RNN`s hidden state vector
            pad_id: int = 0,                         # pad token`s id
            sos_id: int = 1,                         # start of sentence token`s id
            eos_id: int = 2,                         # end of sentence token`s id
            attn_mechanism: str = 'multi-head',      # type of attention mechanism
            num_heads: int = 4,                      # number of attention heads
            num_layers: int = 2,                     # number of RNN layers
            rnn_type: str = 'lstm',                  # type of RNN cell
            dropout_p: float = 0.3,                  # dropout probability
    ) -> None:
        super(Speller, self).__init__(hidden_state_dim, hidden_state_dim, num_layers, rnn_type, dropout_p, False)
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_length = max_length
        self.eos_id = eos_id
        self.sos_id = sos_id
        self.pad_id = pad_id
        self.attn_mechanism = attn_mechanism.lower()
        self.embedding = nn.Embedding(num_classes, hidden_state_dim)
        self.input_dropout = nn.Dropout(dropout_p)

        if self.attn_mechanism == 'loc':
            self.attention = LocationAwareAttention(decoder_dim=hidden_state_dim, attn_dim=hidden_state_dim, smoothing=False)
        elif self.attn_mechanism == 'multi-head':
            self.attention = MultiHeadAttention(d_model=hidden_state_dim, num_heads=num_heads)
        elif self.attn_mechanism == 'additive':
            self.attention = AdditiveAttention(hidden_state_dim)
        elif self.attn_mechanism == 'scaled-dot':
            self.attention = ScaledDotProductAttention(dim=hidden_state_dim)
        else:
            raise ValueError("Unsupported attention: %s".format(attn_mechanism))

        self.fc = nn.Sequential(
            Linear(hidden_state_dim << 1, hidden_state_dim),
            nn.Tanh(),
            View(shape=(-1, self.hidden_state_dim), contiguous=True),
            Linear(hidden_state_dim, num_classes),
        )

    def forward_step(
            self,
            input_var: Tensor,                      # tensor of sequences whose contains target variables
            hidden_states: Optional[Tensor],        # tensor containing hidden state vector of RNN
            encoder_outputs: Tensor,                # tensor with containing the outputs of the encoder
            attn: Optional[Tensor] = None,          # tensor containing attention distribution
    ) -> Tuple[Tensor, Tensor, Tensor]:
        batch_size, output_lengths = input_var.size(0), input_var.size(1)

        embedded = self.embedding(input_var)
        embedded = self.input_dropout(embedded)

        outputs, hidden_states = self._forward(embedded, hidden_states)

        if self.attn_mechanism == 'loc':
            context, attn = self.attention(outputs, encoder_outputs, attn)
        else:
            context, attn = self.attention(outputs, encoder_outputs, encoder_outputs)

        outputs = torch.cat((outputs, context), dim=2)

        step_outputs = self.get_normalized_probs(outputs.view(-1, self.hidden_state_dim << 1))
        step_outputs = step_outputs.view(batch_size, output_lengths, -1).squeeze(1)

        return step_outputs, hidden_states, attn

    def forward(
            self,
            inputs: Tensor,                         # tensor of sequences whose contains target variables
            encoder_outputs: Tensor,                # tensor with containing the outputs of the encoder
            teacher_forcing_ratio: float = 1.0,     # probability that teacher forcing will be used.
    ) -> list:

        hidden_states, attn = None, None
        predicted_log_probs = list()

        inputs, batch_size, max_length = self._validate_args(inputs, encoder_outputs, teacher_forcing_ratio)
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        if use_teacher_forcing:
            inputs = inputs[inputs != self.eos_id].view(batch_size, -1)

            if self.attn_mechanism == 'loc' or self.attn_mechanism == 'additive':
                for di in range(inputs.size(1)):
                    input_var = inputs[:, di].unsqueeze(1)
                    step_outputs, hidden_states, attn = self.forward_step(input_var, hidden_states, encoder_outputs, attn)
                    predicted_log_probs.append(step_outputs)

            else:
                step_outputs, hidden_states, attn = self.forward_step(inputs, hidden_states, encoder_outputs, attn)

                for di in range(step_outputs.size(1)):
                    step_output = step_outputs[:, di, :]
                    predicted_log_probs.append(step_output)

        else:
            input_var = inputs[:, 0].unsqueeze(1)

            for di in range(max_length):
                step_outputs, hidden_states, attn = self.forward_step(input_var, hidden_states, encoder_outputs, attn)
                predicted_log_probs.append(step_outputs)
                input_var = predicted_log_probs[-1].topk(1)[1]

        return predicted_log_probs

    def _validate_args(
            self,
            inputs: Optional[Any] = None,           # tensor of sequences whose contains target variables
            encoder_outputs: Tensor = None,         # tensor with containing the outputs of the encoder
            teacher_forcing_ratio: float = 1.0,     # the probability that teacher forcing will be used
    ) -> Tuple[Tensor, int, int]:
        """ Validate arguments """
        assert encoder_outputs is not None
        batch_size = encoder_outputs.size(0)

        if inputs is None:  # inference
            inputs = LongTensor([self.sos_id] * batch_size).view(batch_size, 1)
            max_length = self.max_length

            if torch.cuda.is_available():
                inputs = inputs.cuda()

            if teacher_forcing_ratio > 0:
                raise ValueError("Teacher forcing has to be disabled (set 0) when no inputs is provided.")

        else:
            max_length = inputs.size(1) - 1  # minus the start of sequence symbol

        return inputs, batch_size, max_length
