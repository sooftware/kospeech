import torch.nn as nn
import torch


class Listener(nn.Module):
    r"""
    Converts low level speech signals into higher level features

    Args:
        rnn_cell (str, optional): type of RNN cell (default: gru)
        hidden_size (int): the number of features in the hidden state `h`
        n_layers (int, optional): number of recurrent layers (default: 1)
        bidirectional (bool, optional): if True, becomes a bidirectional encoder (defulat: False)
        use_pyramidal (bool): flag indication whether to use pyramidal rnn for time resolution (default: True)
        dropout_p (float, optional): dropout probability for the output sequence (default: 0)

    Inputs: inputs, hidden
        - **inputs**: list of sequences, whose length is the batch size and within which each sequence is a list of token IDs.
        - **hidden**: variable containing the features in the hidden state h

    Returns: output
        - **output**: tensor containing the encoded features of the input sequence

    Examples::

        >>> listener = Listener(in_features, hidden_size, dropout_p=0.5, n_layers=5)
        >>> output = listener(inputs)
    """

    def __init__(self, in_features, hidden_size, device, dropout_p=0.5, n_layers=5,
                 bidirectional=True, rnn_cell='gru', use_pyramidal = True):

        super(Listener, self).__init__()
        assert rnn_cell.lower() == 'lstm' or rnn_cell.lower() == 'gru' or rnn_cell.lower() == 'rnn'
        assert n_layers > 1, "n_layers should be bigger than 1"
        if use_pyramidal:
            assert n_layers > 4, "Pyramidal Listener`s n_layers should be bigger than 4"

        self.use_pyramidal = use_pyramidal
        self.rnn_cell = nn.LSTM if rnn_cell.lower() == 'lstm' else nn.GRU if rnn_cell.lower() == 'gru' else nn.RNN
        self.device = device
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1),
            nn.Hardtanh(0, 20, inplace=True),
            nn.BatchNorm2d(num_features=64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.Hardtanh(0, 20, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(num_features=64),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.Hardtanh(0, 20, inplace=True),
            nn.BatchNorm2d(num_features=128),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.Hardtanh(0, 20, inplace=True),
            nn.BatchNorm2d(num_features=128),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.Hardtanh(0, 20, inplace=True),
            nn.BatchNorm2d(num_features=256),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        in_features = (in_features-1) << 6 if in_features % 2 else in_features << 6

        if use_pyramidal:
            self.bottom_rnn = self.rnn_cell(
                in_features = in_features,
                hidden_size = hidden_size,
                num_layers = 2,
                batch_first = True,
                bidirectional = bidirectional,
                dropout = dropout_p
            )
            self.middle_rnn = PyramidalRNN(
                rnn_cell = rnn_cell,
                in_features = hidden_size << 1 if bidirectional else 0,
                hidden_size = hidden_size,
                dropout_p = dropout_p,
                n_layers = 2,
                device = device
            )
            self.top_rnn = PyramidalRNN(
                rnn_cell = rnn_cell,
                in_features = hidden_size << 1 if bidirectional else 0,
                hidden_size = hidden_size,
                dropout_p = dropout_p,
                n_layers = n_layers-4,
                device = device
            )

        else:
            self.rnn = self.rnn_cell(
                input_size = in_features,
                hidden_size = hidden_size,
                num_layers = n_layers,
                batch_first = True,
                bidirectional = bidirectional,
                dropout = dropout_p
            )


    def forward(self, inputs):
        """
        Applies a multi-layer RNN to an input sequence.

        Args:
            inputs (batch, seq_len): tensor containing the features of the input sequence.

        Returns: output, hidden
            - **output** (batch, seq_len, hidden_size): variable containing the encoded features of the input sequence
            - **hidden** (num_layers * num_directions, batch, hidden_size): variable containing the features in the hidden state h
        """
        x = self.conv(inputs.unsqueeze(1)).to(self.device)
        x = x.transpose(1, 2)
        x = x.contiguous().view(x.size(0), x.size(1), x.size(2) * x.size(3)).to(self.device)

        if self.training:
            self.flatten_parameters()

        if self.use_pyramidal:
            output, hidden = self.bottom_rnn(x, None)
            output, hidden = self.middle_rnn(output, hidden)
            output, hidden = self.top_rnn(output, hidden)

        else:
            output, hidden = self.rnn(x)

        return output, hidden


    def flatten_parameters(self):
        """ flatten parameters for fast training """
        if self.use_pyramidal:
            self.bottom_rnn.flatten_parameters()
            self.middle_rnn.flatten_parameters()
            self.top_rnn.flatten_parameters()

        else:
            self.rnn.flatten_parameters()



class PyramidalRNN(nn.Module):
    r"""
    Pyramidal RNN for time resolution reduction

    Args:
        rnn_cell (str, optional): type of RNN cell (default: gru)
        hidden_size (int): the number of features in the hidden state `h`
        n_layers (int, optional): number of recurrent layers (default: 1)
        input_size (int): size of input
        dropout_p (float, optional): dropout probability for the output sequence (default: 0)

    Inputs: inputs
        - **inputs**: list of sequences, whose length is the batch size and within which each sequence is a list of token IDs.

    Returns: output, hidden
        - **output** (batch, seq_len, hidden_size): tensor containing the encoded features of the input sequence
        - **hidden** (num_layers * num_directions, batch, hidden_size): tensor containing the features in the hidden state `h`

    Examples::
        >>> rnn = PyramidalRNN(rnn_cell, input_size, hidden_size, dropout_p)
        >>> output, hidden = rnn(inputs)
    """
    def __init__(self, rnn_cell, in_features, hidden_size, dropout_p, device, n_layers=2):
        super(PyramidalRNN, self).__init__()

        assert rnn_cell.lower() == 'lstm' or rnn_cell.lower() == 'gru' or rnn_cell.lower() == 'rnn'

        self.rnn_cell = nn.LSTM if rnn_cell.lower() == 'lstm' else nn.GRU if rnn_cell.lower() == 'gru' else nn.RNN
        self.rnn = self.rnn_cell(
            input_size = in_features << 1,
            hidden_size = hidden_size,
            num_layers = n_layers,
            bidirectional = True,
            batch_first = True,
            dropout = dropout_p
        )
        self.device = device


    def forward(self, inputs, hidden):
        """
        Applies a multi-layer RNN to an input sequence.

        Args:
            inputs (batch, seq_len): tensor containing the features of the input sequence.

        Returns: output
            - **output** (batch, seq_len, hidden_size): variable containing the encoded features of the input sequence
        """
        batch_size = inputs.size(0)
        seq_len = inputs.size(1)
        input_size = inputs.size(2)

        if seq_len % 2:
            zeros = torch.zeros((inputs.size(0), 1, inputs.size(2))).to(self.device)
            inputs = torch.cat([inputs, zeros], dim = 1)
            seq_len += 1

        inputs = inputs.contiguous().view(batch_size, int(seq_len / 2), input_size * 2)

        if hidden is None:
            output, hidden = self.rnn(inputs)

        else:
            output, hidden = self.rnn(inputs, hidden)

        return output, hidden


    def flatten_parameters(self):
        self.rnn.flatten_parameters()