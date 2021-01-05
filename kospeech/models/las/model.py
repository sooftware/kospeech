# -*- coding: utf-8 -*-
# Soohwan Kim, Seyoung Bae, Cheolhwang Won.
# @ArXiv : KoSpeech: Open-Source Toolkit for End-to-End Korean Speech Recognition
# This source code is licensed under the Apache 2.0 License license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from torch import Tensor
from kospeech.models import TopKDecoder
from typing import Optional, Tuple


class ListenAttendSpell(nn.Module):
    """
    Listen, Attend and Spell model with configurable encoder and decoder.

    Args:
        encoder (torch.nn.Module): encoder of las
        decoder (torch.nn.Module): decoder of las

    Inputs: inputs, input_lengths, targets, teacher_forcing_ratio
        - **inputs** (torch.Tensor): tensor of sequences, whose length is the batch size and within which
          each sequence is a list of token IDs. This information is forwarded to the encoder.
        - **input_lengths** (torch.Tensor): tensor of sequences, whose contains length of inputs.
        - **targets** (torch.Tensor): tensor of sequences, whose length is the batch size and within which
          each sequence is a list of token IDs. This information is forwarded to the decoder.
        - **teacher_forcing_ratio** (float): The probability that teacher forcing will be used. A random number
          is drawn uniformly from 0-1 for every decoding token, and if the sample is smaller than the given value,
          teacher forcing would be used (default is 0.90)

    Returns: output
        - **output** (seq_len, batch_size, num_classes): list of tensors containing
          the outputs of the decoding function.
    """
    def __init__(self, encoder: nn.Module, decoder: nn.Module) -> None:
        super(ListenAttendSpell, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(
            self,
            inputs: Tensor,                             # tensor contains audio's features
            input_lengths: Tensor,                      # tensor contains feature's lengths
            targets: Optional[Tensor] = None,           # tensor contains target sentences
            teacher_forcing_ratio: float = 1.0,         # ratio of teacher forcing
    ) -> Tuple[dict, Tensor, Tensor]:
        encoder_outputs, encoder_log_probs, encoder_output_lengths = self.encoder(inputs, input_lengths)

        if isinstance(self.decoder, TopKDecoder):
            return self.decoder(targets, encoder_outputs)
        decoder_outputs = self.decoder(targets, encoder_outputs, teacher_forcing_ratio)

        return decoder_outputs, encoder_log_probs, encoder_output_lengths

    def greedy_search(self, inputs: Tensor, input_lengths: Tensor, device: str) -> Tensor:
        with torch.no_grad():
            self.flatten_parameters()
            decoder_outputs, _, _ = self.forward(inputs, input_lengths, teacher_forcing_ratio=0.0)
            output = torch.stack(decoder_outputs['decoder_log_probs'], dim=1).to(device)
            return output.max(-1)[1]

    def flatten_parameters(self) -> None:
        self.encoder.rnn.flatten_parameters()
        self.decoder.rnn.flatten_parameters()

    def set_decoder(self, decoder: nn.Module) -> None:
        self.decoder = decoder
