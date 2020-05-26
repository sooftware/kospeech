import torch
import torch.nn as nn
from e2e.model.module import BaseRNN, MaskCNN, VGGExtractor


class Listener(BaseRNN):
    r"""Converts low level speech signals into higher level features

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
    def __init__(self, input_size, hidden_dim, device, dropout_p=0.5, num_layers=1, bidirectional=True, rnn_type='gru'):
        input_size = (input_size - 1) << 5 if input_size % 2 else input_size << 5
        super(Listener, self).__init__(input_size, hidden_dim, num_layers, rnn_type, dropout_p, bidirectional, device)
        self.extractor = VGGExtractor(in_channel=1)

    def forward(self, inputs, input_lengths):
        inputs = inputs.unsqueeze(1).permute(0, 1, 3, 2)
        conv_feat, seq_lengths = self.extractor(inputs, input_lengths)

        batch_size, channel, hidden_dim, seq_length = conv_feat.size()
        conv_feat = conv_feat.view(batch_size, channel * hidden_dim, seq_length).permute(2, 0, 1).contiguous()

        inputs = nn.utils.rnn.pack_padded_sequence(conv_feat, seq_lengths)
        output, hidden = self.rnn(inputs)
        output, _ = nn.utils.rnn.pad_packed_sequence(output)
        output = output.transpose(0, 1)   # (batch_size, seq_len, hidden_dim)

        return output, hidden
