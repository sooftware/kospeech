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
import torch.nn as nn
from torch import Tensor
from typing import Tuple


class EncoderInterface(nn.Module):
    """ ASR Encoder Interface for KoSpeech model implementation """
    def __init__(self):
        super(EncoderInterface, self).__init__()

    def count_parameters(self) -> int:
        """ Count parameters of encoder """
        return sum([p.numel for p in self.parameters()])

    def update_dropout(self, dropout_p: float) -> None:
        """ Update dropout probability of encoder """
        for name, child in self.named_children():
            if isinstance(child, nn.Dropout):
                child.p = dropout_p

    def forward(self, inputs: Tensor, input_lengths: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
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
        raise NotImplementedError


class DecoderInterface(nn.Module):
    """ ASR Decoder Interface for KoSpeech model implementation """
    def __init__(self):
        super(DecoderInterface, self).__init__()

    def count_parameters(self) -> int:
        """ Count parameters of encoder """
        return sum([p.numel for p in self.parameters()])

    def update_dropout(self, dropout_p) -> None:
        """ Update dropout probability of decoder """
        for name, child in self.named_children():
            if isinstance(child, nn.Dropout):
                child.p = dropout_p

    def forward(self, targets: Tensor, encoder_outputs: Tensor, **kwargs) -> Tensor:
        """
        Forward propagate a `encoder_outputs` for training.
        Args:
            targets (torch.LongTensr): A target sequence passed to decoder. `IntTensor` of size ``(batch, seq_length)``
            encoder_outputs (torch.FloatTensor): A output sequence of encoder. `FloatTensor` of size
                ``(batch, seq_length, dimension)``
        Returns:
            * predicted_log_probs (torch.FloatTensor): Log probability of model predictions.
        """
        raise NotImplementedError

    @torch.no_grad()
    def decode(self, encoder_outputs: Tensor, *args) -> Tensor:
        """
        Decode encoder_outputs.
        Args:
            encoder_outputs (torch.FloatTensor): A output sequence of encoder. `FloatTensor` of size
                ``(batch, seq_length, dimension)``
        Returns:
            * predicted_log_probs (torch.FloatTensor): Log probability of model predictions.
        """
        raise NotImplementedError


class EncoderDecoderModelInterface(nn.Module):
    """ Interface of KoSpeech's Encoder-Decoder Models """
    def __init__(self, encoder: EncoderInterface, decoder: DecoderInterface) -> None:
        super(EncoderDecoderModelInterface, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        
    def set_encoder(self, encoder):
        """ Setter for encoder """
        self.encoder = encoder
        
    def set_decoder(self, decoder):
        """ Setter for decoder """
        self.decoder = decoder

    def count_parameters(self) -> int:
        """ Count parameters of encoder """
        num_encoder_parameters = self.encoder.count_parameters()
        num_decoder_parameters = self.decoder.count_parameters()
        return num_encoder_parameters + num_decoder_parameters

    def update_dropout(self, dropout_p) -> None:
        """ Update dropout probability of model """
        self.encoder.update_dropout(dropout_p)
        self.decoder.update_dropout(dropout_p)

    def forward(
            self,
            inputs: Tensor,
            input_lengths: Tensor,
            targets: Tensor,
            *args,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Forward propagate a `inputs` and `targets` pair for training.
        Args:
            inputs (torch.FloatTensor): A input sequence passed to encoder. Typically for inputs this will be a padded
                `FloatTensor` of size ``(batch, seq_length, dimension)``.
            input_lengths (torch.LongTensor): The length of input tensor. ``(batch)``
            targets (torch.LongTensr): A target sequence passed to decoder. `IntTensor` of size ``(batch, seq_length)``
        Returns:
            * predictions (torch.FloatTensor): Result of model predictions.
        """
        raise NotImplementedError

    @torch.no_grad()
    def recognize(self, inputs: Tensor, input_lengths: Tensor) -> Tensor:
        """
        Recognize input speech. This method consists of the forward of the encoder and the decode() of the decoder.
        Args:
            inputs (torch.FloatTensor): A input sequence passed to encoder. Typically for inputs this will be a padded
                `FloatTensor` of size ``(batch, seq_length, dimension)``.
            input_lengths (torch.LongTensor): The length of input tensor. ``(batch)``
        Returns:
            * predictions (torch.FloatTensor): Result of model predictions.
        """
        encoder_outputs, encoder_output_lengths, _ = self.encoder(inputs, input_lengths)
        return self.decoder.decode(encoder_outputs, encoder_output_lengths)


class CTCModelInterface(nn.Module):
    """
    Interface of KoSpeech's Speech-To-Text Models.
    for a simple, generic encoder / decoder or encoder only model.
    """
    def __init__(self):
        super(CTCModelInterface, self).__init__()
        self.decoder = None
        
    def set_decoder(self, decoder):
        self.decoder = decoder

    def count_parameters(self) -> int:
        """ Count parameters of encoder """
        return sum([p.numel for p in self.parameters()])

    def update_dropout(self, dropout_p: float) -> None:
        """ Update dropout probability of model """
        for name, child in self.named_children():
            if isinstance(child, nn.Dropout):
                child.p = dropout_p

    def forward(self, inputs: Tensor, input_lengths: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Forward propagate a `inputs` for  ctc training.
        Args:
            inputs (torch.FloatTensor): A input sequence passed to encoder. Typically for inputs this will be a padded
                `FloatTensor` of size ``(batch, seq_length, dimension)``.
            input_lengths (torch.LongTensor): The length of input tensor. ``(batch)``
        Returns:
            (Tensor, Tensor):
            * predicted_log_prob (torch.FloatTensor)s: Log probability of model predictions.
            * output_lengths (torch.LongTensor): The length of output tensor ``(batch)``
        """
        raise NotImplementedError

    @torch.no_grad()
    def decode(self, predicted_log_probs: Tensor) -> Tensor:
        """
        Decode encoder_outputs.
        Args:
            predicted_log_probs (torch.FloatTensor):Log probability of model predictions. `FloatTensor` of size
                ``(batch, seq_length, dimension)``
        Returns:
            * predictions (torch.FloatTensor): Result of model predictions.
        """
        return predicted_log_probs.max(-1)[1]

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
        predicted_log_probs, _ = self.forward(inputs, input_lengths)
        if self.decoder is not None:
            return self.decoder.decode(predicted_log_probs)
        return self.decode(predicted_log_probs)
