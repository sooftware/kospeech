import math
import torch.nn as nn

supported_rnns = {
    'lstm': nn.LSTM,
    'gru': nn.GRU,
    'rnn': nn.RNN
}


class Listener(nn.Module):
    r"""Converts low level speech signals into higher level features

    Args:
        rnn_type (str, optional): type of RNN cell (default: gru)
        hidden_dim (int): the number of features in the hidden state `h`
        num_layers (int, optional): number of recurrent layers (default: 1)
        bidirectional (bool, optional): if True, becomes a bidirectional encoder (defulat: False)
        dropout_p (float, optional): dropout probability for the output sequence (default: 0)

    Inputs: inputs, h_state
        - **inputs**: list of sequences, whose length is the batch size and within which each sequence is list of tokens
        - **h_state**: variable containing the features in the hidden state h

    Returns: output
        - **output**: tensor containing the encoded features of the input sequence

    Examples::

        >>> listener = Listener(80, hidden_dim=256, dropout_p=0.5, num_layers=5)
        >>> output = listener(inputs)
    """

    def __init__(self, in_features, hidden_dim, device, dropout_p=0.5, num_layers=5, bidirectional=True,
                 rnn_type='gru', conv_type='custom'):
        super(Listener, self).__init__()
        assert rnn_type.lower() in supported_rnns.keys(), 'RNN type not supported.'

        rnn_cell = supported_rnns[rnn_type]
        self.device = device

        if conv_type == 'custom':
            self.conv = nn.Sequential(
                nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
                nn.Hardtanh(0, 20, inplace=True),
                nn.BatchNorm2d(num_features=64),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                nn.Hardtanh(0, 20, inplace=True),
                nn.MaxPool2d(2, stride=2),
                nn.BatchNorm2d(num_features=64),
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.Hardtanh(0, 20, inplace=True),
                nn.BatchNorm2d(num_features=128),
                nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                nn.Hardtanh(0, 20, inplace=True),
                nn.MaxPool2d(2, stride=2)
            )
            in_features = (in_features - 1) << 5 if in_features % 2 else in_features << 5

        elif conv_type == 'deepspeech2':
            self.conv = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=(41, 11), stride=(2, 2), padding=(20, 5)),
                nn.BatchNorm2d(32),
                nn.Hardtanh(0, 20, inplace=True),
                nn.Conv2d(32, 32, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5)),
                nn.Hardtanh(0, 20, inplace=True),
                nn.MaxPool2d(2, stride=2),
                nn.BatchNorm2d(num_features=32),
                nn.Conv2d(32, 64, kernel_size=(41, 11), stride=(2, 2), padding=(20, 5)),
                nn.BatchNorm2d(64),
                nn.Hardtanh(0, 20, inplace=True),
                nn.Conv2d(64, 64, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5)),
                nn.BatchNorm2d(64),
                nn.Hardtanh(0, 20, inplace=True),
                nn.MaxPool2d(2, stride=2)
            )
            in_features = int(math.floor(in_features + 2 * 20 - 41) / 2 + 1)
            in_features = int(math.floor(in_features + 2 * 10 - 21) / 2 + 1)
            in_features *= 16

        else:
            raise ValueError("conv type should be one of [custom, deepspeech2]")

        self.rnn = rnn_cell(
            input_size=in_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout_p
        )

    def forward(self, inputs):
        x = self.conv(inputs.unsqueeze(1)).to(self.device)
        x = x.transpose(1, 2)
        x = x.contiguous().view(x.size(0), x.size(1), x.size(2) * x.size(3)).to(self.device)

        if self.training:
            self.flatten_parameters()

        output, h_state = self.rnn(x)

        return output, h_state

    def flatten_parameters(self):
        self.rnn.flatten_parameters()
