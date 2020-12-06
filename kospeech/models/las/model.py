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
    Listen, Attend and Spell architecture with configurable encoder and decoder.

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
    def __init__(
            self,
            encoder: nn.Module,
            decoder: nn.Module,
            joint_ctc: bool = False,
            blank_id: int = None
    ) -> None:
        super(ListenAttendSpell, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

        self.joint_ctc = joint_ctc
        if self.joint_ctc:
            assert blank_id is not None, "If use joint learning, blank_id should not be None"
            self.ctc_loss = nn.CTCLoss(blank=blank_id, reduction='mean')

    def forward(
            self,
            inputs: Tensor,                              # tensor of sequences whose contains input variables
            input_lengths: Tensor,                       # tensor of sequences whose contains lengths of inputs
            targets: Optional[Tensor] = None,            # tensor of sequences whose contains target variables
            target_lengths: Optional[Tensor] = None,     # tensor of sequences whose contains lengths of targets
            teacher_forcing_ratio: float = 1.0           # the probability that teacher forcing will be used
    ) -> Tuple[dict, float]:
        ctc_loss = None
        encoder_outputs, ctc_output, seq_lengths = self.encoder(inputs, input_lengths)

        if self.joint_ctc:
            assert targets is not None and target_lengths is not None, "targets, target_lengths is None (Joint-CTC)"
            ctc_loss = self.get_ctc_loss(ctc_output, seq_lengths, targets, target_lengths)

        if isinstance(self.decoder, TopKDecoder):
            return self.decoder(targets, encoder_outputs)
        decoder_outputs = self.decoder(targets, encoder_outputs, teacher_forcing_ratio)

        return decoder_outputs, ctc_loss

    def greedy_decode(self, inputs: Tensor, input_lengths: Tensor, device: str):
        with torch.no_grad():
            self.flatten_parameters()
            output = self.forward(inputs, input_lengths, teacher_forcing_ratio=0.0)
            logit = torch.stack(output['decoder_outputs'], dim=1).to(device)
            return logit.max(-1)[1]

    def get_ctc_loss(self, ctc_output: Tensor, input_lengths: Tensor, targets: Tensor, target_lengths: Tensor):
        ctc_output = ctc_output.transpose(0, 1)  # B x T x D => T x B x D
        ctc_loss = self.ctc_loss(ctc_output.log_softmax(dim=2), targets, input_lengths, target_lengths)
        return ctc_loss

    def flatten_parameters(self):
        self.encoder.rnn.flatten_parameters()
        self.decoder.rnn.flatten_parameters()

    def set_decoder(self, decoder: nn.Module):
        self.decoder = decoder
