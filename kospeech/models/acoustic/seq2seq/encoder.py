import math
import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple
from kospeech.models.modules import BaseRNN
from kospeech.models.acoustic.seq2seq.sublayers import (
    VGGExtractor,
    DeepSpeech2Extractor
)


class SpeechEncoderRNN(BaseRNN):
    """
    Converts low level speech signals into higher level features

    Args:
        input_size (int): size of input
        hidden_dim (int): the number of features in the hidden state `h`
        num_layers (int, optional): number of recurrent layers (default: 1)
        bidirectional (bool, optional): if True, becomes a bidirectional encoder (defulat: False)
        rnn_type (str, optional): type of RNN cell (default: gru)
        dropout_p (float, optional): dropout probability (default: 0.3)
        extractor (str): type of CNN extractor (default: vgg)
        device (torch.device): device - 'cuda' or 'cpu'
        activation (str): type of activation function (default: hardtanh)
        mask_conv (bool): flag indication whether apply mask convolution or not

    Inputs: inputs, input_lengths
        - **inputs**: list of sequences, whose length is the batch size and within which each sequence is list of tokens
        - **input_lengths**: list of sequence lengths

    Returns: output, hidden
        - **output**: tensor containing the encoded features of the input sequence
        - **hidden**: variable containing the features in the hidden state h
    """

    def __init__(self,
                 input_size: int,                       # size of input
                 hidden_dim: int = 512,                 # dimension of RNN`s hidden state
                 device: str = 'cuda',                  # device - 'cuda' or 'cpu'
                 dropout_p: float = 0.3,                # dropout probability
                 num_layers: int = 3,                   # number of RNN layers
                 bidirectional: bool = True,            # if True, becomes a bidirectional encoder
                 rnn_type: str = 'lstm',                # type of RNN cell
                 extractor: str = 'vgg',                # type of CNN extractor
                 activation: str = 'hardtanh',          # type of activation function
                 mask_conv: bool = False) -> None:      # flag indication whether apply mask convolution or not
        self.mask_conv = mask_conv
        self.extractor = extractor.lower()
        if self.extractor == 'vgg':
            input_size = (input_size - 1) << 5 if input_size % 2 else input_size << 5
            super(SpeechEncoderRNN, self).__init__(input_size, hidden_dim, num_layers,
                                                   rnn_type, dropout_p, bidirectional, device)
            self.conv = VGGExtractor(activation, mask_conv)

        elif self.extractor == 'ds2':
            input_size = int(math.floor(input_size + 2 * 20 - 41) / 2 + 1)
            input_size = int(math.floor(input_size + 2 * 10 - 21) / 2 + 1)
            input_size <<= 5
            super(SpeechEncoderRNN, self).__init__(input_size, hidden_dim, num_layers,
                                                   rnn_type, dropout_p, bidirectional, device)
            self.conv = DeepSpeech2Extractor(activation, mask_conv)

        else:
            raise ValueError("Unsupported Extractor : {0}".format(extractor))

    def forward(self, inputs: Tensor, input_lengths: Tensor) -> Tuple[Tensor, Tensor]:
        if self.mask_conv:
            inputs = inputs.unsqueeze(1).permute(0, 1, 3, 2)
            conv_feat, seq_lengths = self.conv(inputs, input_lengths)

            batch_size, num_channels, hidden_dim, seq_length = conv_feat.size()
            conv_feat = conv_feat.view(batch_size, num_channels * hidden_dim, seq_length).permute(2, 0, 1).contiguous()

            conv_feat = nn.utils.rnn.pack_padded_sequence(conv_feat, seq_lengths)
            output, hidden = self.rnn(conv_feat)
            output, _ = nn.utils.rnn.pad_packed_sequence(output)
            output = output.transpose(0, 1)

        else:
            conv_feat = self.conv(inputs.unsqueeze(1), input_lengths).to(self.device)
            conv_feat = conv_feat.transpose(1, 2)

            batch_size, num_channels, seq_length, hidden_dim = conv_feat.size()
            conv_feat = conv_feat.contiguous().view(batch_size, num_channels, seq_length * hidden_dim)

            if self.training:
                self.rnn.flatten_parameters()

            output, hidden = self.rnn(conv_feat)

        return output, hidden
