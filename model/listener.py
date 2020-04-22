import torch.nn as nn

supported_rnns = {
    'lstm': nn.LSTM,
    'gru': nn.GRU,
    'rnn': nn.RNN
}


class Listener(nn.Module):
    r"""
    Converts low level speech signals into higher level features

    Args:
        rnn_type (str, optional): type of RNN cell (default: gru)
        hidden_dim (int): the number of features in the hidden state `h`
        n_layers (int, optional): number of recurrent layers (default: 1)
        bidirectional (bool, optional): if True, becomes a bidirectional encoder (defulat: False)
        dropout_p (float, optional): dropout probability for the output sequence (default: 0)

    Inputs: inputs, h_state
        - **inputs**: list of sequences, whose length is the batch size and within which each sequence is list of tokens
        - **h_state**: variable containing the features in the hidden state h

    Returns: output
        - **output**: tensor containing the encoded features of the input sequence

    Examples::

        >>> listener = Listener(80, hidden_dim=256, dropout_p=0.5, n_layers=5)
        >>> output = listener(inputs)
    """

    def __init__(self, in_features, hidden_dim, device, dropout_p=0.5, n_layers=5, bidirectional=True, rnn_type='gru'):
        super(Listener, self).__init__()
        assert rnn_type.lower() in supported_rnns.keys(), 'RNN type not supported.'
        assert n_layers > 1, 'n_layers should be bigger than 1'

        rnn_cell = supported_rnns[rnn_type]
        self.device = device
        self.conv = nn.Sequential(
            # Refered to VGGNet
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

        in_features = (in_features - 1) << 6 if in_features % 2 else in_features << 6
        self.rnn = rnn_cell(
            input_size=in_features,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout_p
        )

    def forward(self, inputs):
        """
        Applies a multi-layer RNN to an input sequence.

        Args: inputs
            inputs (batch, seq_len): tensor containing the features of the input sequence.

        Returns: output, h_state
            - **output** (batch, seq_len, hidden_dim): variable containing the encoded features of the input sequence
            - **h_state** (num_layers * directions, batch, hidden_dim): variable containing the features in the hidden
        """
        x = self.conv(inputs.unsqueeze(1)).to(self.device)
        x = x.transpose(1, 2)
        x = x.contiguous().view(x.size(0), x.size(1), x.size(2) * x.size(3)).to(self.device)

        if self.training:
            self.flatten_parameters()

        output, h_state = self.rnn(x)

        return output, h_state

    def flatten_parameters(self):
        self.rnn.flatten_parameters()
