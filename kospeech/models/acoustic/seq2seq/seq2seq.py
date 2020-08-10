import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Any
from kospeech.models.acoustic.seq2seq.decoder import SpeechTopKDecoder


class SpeechSeq2seq(nn.Module):
    """
    Speech sequence-to-sequence architecture with configurable encoder and decoder.

    Args:
        encoder (torch.nn.Module): encoder of seq2seq
        decoder (torch.nn.Module): decoder of seq2seq

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
        super(SpeechSeq2seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, inputs: Tensor, input_lengths: Tensor, targets: Optional[Any] = None,
                teacher_forcing_ratio: float = 1.0, return_decode_dict: bool = False):
        output, hidden = self.encoder(inputs, input_lengths)

        if isinstance(self.decoder, SpeechTopKDecoder):
            result = self.decoder(targets, output)
        else:
            result = self.decoder(targets, output, teacher_forcing_ratio, return_decode_dict)

        return result

    def flatten_parameters(self):
        self.encoder.rnn.flatten_parameters()
        self.decoder.rnn.flatten_parameters()

    def set_decoder(self, decoder: nn.Module):
        self.decoder = decoder
