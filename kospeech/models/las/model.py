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

import torch
from torch import Tensor
from typing import Optional, Tuple

from kospeech.models import EncoderRNN, DecoderRNN
from kospeech.models.model import EncoderDecoderModel


class ListenAttendSpell(EncoderDecoderModel):
    """
    Listen, Attend and Spell model with configurable encoder and decoder.

    Args:
        input_dim (int): dimension of input vector
        num_classes (int): number of classification
        encoder_hidden_state_dim (int): the number of features in the encoder hidden state `h`
        decoder_hidden_state_dim (int): the number of features in the decoder hidden state `h`
        num_encoder_layers (int, optional): number of recurrent layers (default: 3)
        num_decoder_layers (int, optional): number of recurrent layers (default: 2)
        bidirectional (bool, optional): if True, becomes a bidirectional encoder (default: False)
        extractor (str): type of CNN extractor (default: vgg)
        activation (str): type of activation function (default: hardtanh)
        rnn_type (str, optional): type of RNN cell (default: lstm)
        encoder_dropout_p (float, optional): dropout probability of encoder (default: 0.2)
        decoder_dropout_p (float, optional): dropout probability of decoder (default: 0.2)
        pad_id (int, optional): index of the pad symbol (default: 0)
        sos_id (int, optional): index of the start of sentence symbol (default: 1)
        eos_id (int, optional): index of the end of sentence symbol (default: 2)
        attn_mechanism (str, optional): type of attention mechanism (default: multi-head)
        num_heads (int, optional): number of attention heads. (default: 4)
        max_length (int, optional): max decoding step (default: 400)
        joint_ctc_attention (bool, optional): flag indication joint ctc attention or not

    Inputs: inputs, input_lengths, targets, teacher_forcing_ratio
        - **inputs** (torch.Tensor): tensor of sequences, whose length is the batch size and within which
          each sequence is a list of token IDs. This information is forwarded to the encoder.
        - **input_lengths** (torch.Tensor): tensor of sequences, whose contains length of inputs.
        - **targets** (torch.Tensor): tensor of sequences, whose length is the batch size and within which
          each sequence is a list of token IDs. This information is forwarded to the decoder.
        - **teacher_forcing_ratio** (float): The probability that teacher forcing will be used. A random number
          is drawn uniformly from 0-1 for every decoding token, and if the sample is smaller than the given value,
          teacher forcing would be used (default is 0.90)

    Returns:
        (Tensor, Tensor, Tensor)

        * predicted_log_probs (torch.FloatTensor): Log probability of model predictions.
        * encoder_output_lengths: The length of encoder outputs. ``(batch)``
        * encoder_log_probs: Log probability of encoder outputs will be passed to CTC Loss.
            If joint_ctc_attention is False, return None.
    """
    def __init__(
            self,
            input_dim: int,
            num_classes: int,
            encoder_hidden_state_dim: int = 512,
            decoder_hidden_state_dim: int = 1024,
            num_encoder_layers: int = 3,
            num_decoder_layers: int = 2,
            bidirectional: bool = True,
            extractor: str = "vgg",
            activation: str = "hardtanh",
            rnn_type: str = "lstm",
            max_length: int = 400,
            pad_id: int = 0,
            sos_id: int = 1,
            eos_id: int = 2,
            attn_mechanism: str = "multi-head",
            num_heads: int = 4,
            encoder_dropout_p: int = 0.2,
            decoder_dropout_p: int = 0.2,
            joint_ctc_attention: bool = False,
    ) -> None:
        encoder = EncoderRNN(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_state_dim=encoder_hidden_state_dim,
            dropout_p=encoder_dropout_p,
            num_layers=num_encoder_layers,
            bidirectional=bidirectional,
            extractor=extractor,
            activation=activation,
            rnn_type=rnn_type,
            joint_ctc_attention=joint_ctc_attention,
        )
        decoder = DecoderRNN(
            num_classes=num_classes,
            max_length=max_length,
            pad_id=pad_id,
            sos_id=sos_id,
            eos_id=eos_id,
            hidden_state_dim=decoder_hidden_state_dim,
            num_layers=num_decoder_layers,
            rnn_type=rnn_type,
            dropout_p=decoder_dropout_p,
            num_heads=num_heads,
            attn_mechanism=attn_mechanism,
        )
        super(ListenAttendSpell, self).__init__(encoder, decoder)

    def forward(
            self,
            inputs: Tensor,
            input_lengths: Tensor,
            targets: Optional[Tensor] = None,
            teacher_forcing_ratio: float = 1.0,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Forward propagate a `inputs` and `targets` pair for training.

        Args:
            inputs (torch.FloatTensor): A input sequence passed to encoder. Typically for inputs this will be a padded
                `FloatTensor` of size ``(batch, seq_length, dimension)``.
            input_lengths (torch.LongTensor): The length of input tensor. ``(batch)``
            targets (torch.LongTensr): A target sequence passed to decoder. `IntTensor` of size ``(batch, seq_length)``
            teacher_forcing_ratio (float): ratio of teacher forcing

        Returns:
            (Tensor, Tensor, Tensor)

            * predicted_log_probs (torch.FloatTensor): Log probability of model predictions.
            * encoder_output_lengths: The length of encoder outputs. ``(batch)``
            * encoder_log_probs: Log probability of encoder outputs will be passed to CTC Loss.
                If joint_ctc_attention is False, return None.
        """
        encoder_outputs, encoder_output_lengths, encoder_log_probs = self.encoder(inputs, input_lengths)
        predicted_log_probs = self.decoder(targets, encoder_outputs, teacher_forcing_ratio)
        return predicted_log_probs, encoder_output_lengths, encoder_log_probs

    def flatten_parameters(self) -> None:
        self.encoder.rnn.flatten_parameters()
        self.decoder.rnn.flatten_parameters()
