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

import torch
from torch import Tensor
from typing import Tuple, Optional

from kospeech.models.conformer.encoder import ConformerEncoder
from kospeech.models.model import TransducerModel
from kospeech.models.rnnt.decoder import DecoderRNNT


class Conformer(TransducerModel):
    """
    Conformer: Convolution-augmented Transformer for Speech Recognition
    The paper used a one-lstm Transducer decoder, currently still only implemented
    the conformer encoder shown in the paper.

    Args:
        num_classes (int): Number of classification classes
        input_dim (int, optional): Dimension of input vector
        encoder_dim (int, optional): Dimension of conformer encoder
        decoder_dim (int, optional): Dimension of conformer decoder
        num_encoder_layers (int, optional): Number of conformer blocks
        num_decoder_layers (int, optional): Number of decoder layers
        decoder_rnn_type (str, optional): type of RNN cell
        num_attention_heads (int, optional): Number of attention heads
        feed_forward_expansion_factor (int, optional): Expansion factor of feed forward module
        conv_expansion_factor (int, optional): Expansion factor of conformer convolution module
        feed_forward_dropout_p (float, optional): Probability of feed forward module dropout
        attention_dropout_p (float, optional): Probability of attention module dropout
        conv_dropout_p (float, optional): Probability of conformer convolution module dropout
        decoder_dropout_p (float, optional): Probability of conformer decoder dropout
        conv_kernel_size (int or tuple, optional): Size of the convolving kernel
        half_step_residual (bool): Flag indication whether to use half step residual or not
        device (torch.device): torch device (cuda or cpu)
        decoder (str): If decoder is None, train with CTC decoding

    Inputs: inputs
        - **inputs** (batch, time, dim): Tensor containing input vector
        - **input_lengths** (batch): list of sequence input lengths

    Returns:
        * predictions (torch.FloatTensor): Result of model predictions.
    """
    def __init__(
            self,
            num_classes: int,
            input_dim: int = 80,
            encoder_dim: int = 512,
            decoder_dim: int = 640,
            num_encoder_layers: int = 17,
            num_decoder_layers: int = 1,
            decoder_rnn_type: str = 'lstm',
            num_attention_heads: int = 8,
            feed_forward_expansion_factor: int = 4,
            conv_expansion_factor: int = 2,
            input_dropout_p: float = 0.1,
            feed_forward_dropout_p: float = 0.1,
            attention_dropout_p: float = 0.1,
            conv_dropout_p: float = 0.1,
            decoder_dropout_p: float = 0.1,
            conv_kernel_size: int = 31,
            half_step_residual: bool = True,
            device: torch.device = 'cuda',
            decoder: str = None,
    ) -> None:
        encoder = ConformerEncoder(
            input_dim=input_dim,
            encoder_dim=encoder_dim,
            num_layers=num_encoder_layers,
            num_attention_heads=num_attention_heads,
            feed_forward_expansion_factor=feed_forward_expansion_factor,
            conv_expansion_factor=conv_expansion_factor,
            input_dropout_p=input_dropout_p,
            feed_forward_dropout_p=feed_forward_dropout_p,
            attention_dropout_p=attention_dropout_p,
            conv_dropout_p=conv_dropout_p,
            conv_kernel_size=conv_kernel_size,
            half_step_residual=half_step_residual,
            device=device,
        )
        if decoder == 'rnnt':
            decoder = DecoderRNNT(
                num_classes=num_classes,
                hidden_state_dim=decoder_dim,
                output_dim=encoder_dim,
                num_layers=num_decoder_layers,
                rnn_type=decoder_rnn_type,
                dropout_p=decoder_dropout_p,
            )
        else:
            decoder = None
        super(Conformer, self).__init__(encoder, decoder, encoder_dim >> 1, num_classes)

    def forward(
            self,
            inputs: Tensor,
            input_lengths: Tensor,
            targets: Tensor,
            target_lengths: Tensor,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Forward propagate a `inputs` and `targets` pair for training.

        Args:
            inputs (torch.FloatTensor): A input sequence passed to encoder. Typically for inputs this will be a padded
                `FloatTensor` of size ``(batch, seq_length, dimension)``.
            input_lengths (torch.LongTensor): The length of input tensor. ``(batch)``
            targets (torch.LongTensr): A target sequence passed to decoder. `IntTensor` of size ``(batch, seq_length)``
            target_lengths (torch.LongTensor): The length of target tensor. ``(batch)``

        Returns:
            * predictions (torch.FloatTensor): Result of model predictions.
        """
        if self.decoder is not None:
            return super().forward(inputs, input_lengths, targets, target_lengths)
        encoder_outputs, output_lengths = self.encoder(inputs, input_lengths)
        outputs = self.fc(encoder_outputs).log_softmax(dim=-1)
        return outputs, output_lengths

    @torch.no_grad()
    def decode(self, encoder_outputs: Tensor, max_length: int = None) -> Tensor:
        """
        Decode `encoder_outputs`.

        Args:
            encoder_outputs (torch.FloatTensor): A output sequence of encoder. `FloatTensor` of size
                ``(seq_length, dimension)``
            max_length (int): max decoding time step

        Returns:
            * predicted_log_probs (torch.FloatTensor): Log probability of model predictions.
        """
        if self.decoder is not None:
            return super().decode(encoder_outputs, max_length)
        return encoder_outputs.max(-1)[1]

    @torch.no_grad()
    def recognize(self, inputs: Tensor, input_lengths: Tensor) -> Tensor:
        """
        Recognize input speech.

        Args:
            inputs (torch.FloatTensor): A input sequence passed to encoder. Typically for inputs this will be a padded
                `FloatTensor` of size ``(batch, seq_length, dimension)``.
            input_lengths (torch.LongTensor): The length of input tensor. ``(batch)``

        Returns:
            * predictions (torch.FloatTensor): Result of model predictions.
        """
        if self.decoder is not None:
            return super().recognize(inputs, input_lengths)
        encoder_outputs, _ = self.encoder(inputs, input_lengths)
        predicted_log_probs = self.fc(encoder_outputs).log_softmax(dim=-1)
        return self.decode(predicted_log_probs)
