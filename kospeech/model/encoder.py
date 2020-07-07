import math
import torch
import torch.nn as nn
from kospeech.model.base_rnn import BaseRNN
from kospeech.model.conv import VGGExtractor, DeepSpeech2Extractor


class Listener(BaseRNN):
    """Converts low level speech signals into higher level features

    Args:
        input_size (int): size of input
        hidden_dim (int): the number of features in the hidden state `h`
        num_layers (int, optional): number of recurrent layers (default: 1)
        bidirectional (bool, optional): if True, becomes a bidirectional encoder (defulat: False)
        rnn_type (str, optional): type of RNN cell (default: gru)
        dropout_p (float, optional): dropout probability (default: 0)
        device (torch.device): device - 'cuda' or 'cpu'

    Inputs: inputs, hidden
        - **inputs**: list of sequences, whose length is the batch size and within which each sequence is list of tokens
        - **hidden**: variable containing the features in the hidden state h

    Returns: output
        - **output**: tensor containing the encoded features of the input sequence
    """

    def __init__(self, input_size: int, hidden_dim: int = 256, device: str = 'cuda', dropout_p: float = 0.3,
                 num_layers: int = 3, bidirectional: bool = True, rnn_type: str = 'lstm', extractor: str = 'vgg',
                 activation: str = 'hardtanh', mask_conv: bool = False):
        self.mask_conv = mask_conv

        if extractor.lower() == 'vgg':
            input_size = (input_size - 1) << 5 if input_size % 2 else input_size << 5
            super(Listener, self).__init__(input_size, hidden_dim, num_layers, rnn_type, dropout_p, bidirectional, device)
            self.conv_extractor = VGGExtractor(in_channels=1, activation=activation, mask_conv=mask_conv)

        elif extractor.lower() == 'ds2':
            input_size = int(math.floor(input_size + 2 * 20 - 41) / 2 + 1)
            input_size = int(math.floor(input_size + 2 * 10 - 21) / 2 + 1)
            input_size <<= 5
            super(Listener, self).__init__(input_size, hidden_dim, num_layers, rnn_type, dropout_p, bidirectional, device)
            self.conv_extractor = DeepSpeech2Extractor(in_channels=1, activation=activation, mask_conv=mask_conv)

        else:
            raise ValueError("Unsupported Extractor : {0}".format(extractor))

    def forward(self, inputs: torch.Tensor, input_lengths: torch.Tensor):
        if self.mask_conv:
            inputs = inputs.unsqueeze(1).permute(0, 1, 3, 2)
            conv_feat, seq_lengths = self.conv_extractor(inputs, input_lengths)

            batch_size, num_channels, hidden_dim, seq_length = conv_feat.size()
            conv_feat = conv_feat.view(batch_size, num_channels * hidden_dim, seq_length).permute(2, 0, 1).contiguous()

            inputs = nn.utils.rnn.pack_padded_sequence(conv_feat, seq_lengths)
            output, hidden = self.rnn(inputs)
            output, _ = nn.utils.rnn.pad_packed_sequence(output)
            output = output.transpose(0, 1)

        else:
            conv_feat = self.conv_extractor(inputs.unsqueeze(1), input_lengths).to(self.device)
            conv_feat = conv_feat.transpose(1, 2)

            batch_size, num_channels, seq_length, hidden_dim = conv_feat.size()
            conv_feat = conv_feat.contiguous().view(batch_size, num_channels, seq_length * hidden_dim)

            if self.training:
                self.rnn.flatten_parameters()

            output, hidden = self.rnn(conv_feat)

        return output
