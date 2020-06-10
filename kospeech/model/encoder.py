import math
import torch.nn as nn
from kospeech.model.convolutional import VGGExtractor, DeepSpeech2Extractor


class BaseRNN(nn.Module):
    """
    Applies a multi-layer RNN to an input sequence.

    Note:
        Do not use this class directly, use one of the sub classes.

    Args:
        input_size (int): size of input
        hidden_dim (int): the number of features in the hidden state `h`
        num_layers (int, optional): number of recurrent layers (default: 1)
        bidirectional (bool, optional): if True, becomes a bidirectional encoder (defulat: False)
        rnn_type (str, optional): type of RNN cell (default: gru)
        dropout_p (float, optional): dropout probability (default: 0)
        device (torch.device): device - 'cuda' or 'cpu'

    Attributes:
          supported_rnns = Dictionary of supported rnns
    """
    supported_rnns = {
        'lstm': nn.LSTM,
        'gru': nn.GRU,
        'rnn': nn.RNN
    }

    def __init__(self, input_size, hidden_dim, num_layers, rnn_type, dropout_p, bidirectional, device):
        super(BaseRNN, self).__init__()
        rnn_cell = self.supported_rnns[rnn_type]
        self.rnn = rnn_cell(input_size, hidden_dim, num_layers, True, True, dropout_p, bidirectional)
        self.hidden_dim = hidden_dim
        self.device = device

    def forward(self, *args, **kwargs):
        raise NotImplementedError


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

    def __init__(self, input_size, hidden_dim, device, dropout_p=0.3, num_layers=3,
                 bidirectional=True, rnn_type='lstm', extractor='vgg', activation='hardtanh'):
        if extractor.lower() == 'vgg':
            input_size = (input_size - 1) << 5 if input_size % 2 else input_size << 5
            super(Listener, self).__init__(input_size, hidden_dim, num_layers, rnn_type, dropout_p, bidirectional, device)
            self.extractor = VGGExtractor(in_channels=1, activation=activation)

        elif extractor.lower() == 'ds2':
            input_size = int(math.floor(input_size + 2 * 20 - 41) / 2 + 1)
            input_size = int(math.floor(input_size + 2 * 10 - 21) / 2 + 1)
            input_size <<= 5
            super(Listener, self).__init__(input_size, hidden_dim, num_layers, rnn_type, dropout_p, bidirectional, device)
            self.extractor = DeepSpeech2Extractor(in_channels=1, activation=activation)

        else:
            raise ValueError("Unsupported Extractor : {0}".format(extractor))

    def forward(self, inputs, input_lengths):
        inputs = inputs.unsqueeze(1).permute(0, 1, 3, 2)
        conv_feat, seq_lengths = self.extractor(inputs, input_lengths)

        batch_size, channel, hidden_dim, seq_length = conv_feat.size()
        conv_feat = conv_feat.view(batch_size, channel * hidden_dim, seq_length).permute(2, 0, 1).contiguous()

        inputs = nn.utils.rnn.pack_padded_sequence(conv_feat, seq_lengths)
        output, hidden = self.rnn(inputs)
        output, _ = nn.utils.rnn.pad_packed_sequence(output)
        output = output.transpose(0, 1)   # (batch_size, seq_len, hidden_dim)

        del inputs, input_lengths, hidden, _, batch_size, channel, hidden_dim, seq_length, seq_lengths, conv_feat
        return output
