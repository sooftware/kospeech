import torch
import torch.nn as nn


class MaskCNN(nn.Module):
    """
    Masking Convolutional Neural Network

    Adds padding to the output of the module based on the given lengths.
    This is to ensure that the results of the model do not change when batch sizes change during inference.
    Input needs to be in the shape of (batch_size, channel, hidden_dim, seq_len)

    Args:
        sequential (torch.nn): sequential list of convolution layer

    Inputs: inputs, input_lengths
        - **inputs**: The input of size BxCxHxS
        - **input_lengths**: The actual length of each sequence in the batch

    Returns: output, output_lengths
        - **output**: Masked output from the module
        - **output_lengths**: Length of output from the module
    """
    def __init__(self, sequential):
        super(MaskCNN, self).__init__()
        self.sequential = sequential

    def forward(self, inputs, input_lengths):
        output = None
        output_lengths = None

        for module in self.sequential:
            output = module(inputs)
            mask = torch.BoolTensor(output.size()).fill_(0)

            if output.is_cuda:
                mask = mask.cuda()

            output_lengths = self.get_output_lengths(input_lengths, module)

            for i, length in enumerate(output_lengths):
                length = length.item()

                if (mask[i].size(2) - length) > 0:
                    mask[i].narrow(dim=2, start=length, length=mask[i].size(2) - length).fill_(1)

            output = output.masked_fill(mask, 0)

            inputs = output
            input_lengths = output_lengths

        return output, output_lengths

    def get_output_lengths(self, lengths, m):
        """ Calculate convolutional neural network receptive formula """
        if type(m) == nn.modules.conv.Conv2d:
            lengths = (lengths + 2 * m.padding[1] - m.dilation[1] * (m.kernel_size[1] - 1) - 1) / m.stride[1] + 1

        elif type(m) == nn.modules.MaxPool2d:
            lengths /= 2

        return lengths


class Listener(nn.Module):
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
    supported_rnns = {
        'lstm': nn.LSTM,
        'gru': nn.GRU,
        'rnn': nn.RNN
    }

    def __init__(self, input_size, hidden_dim, device, dropout_p=0.5, num_layers=1, bidirectional=True, rnn_type='gru'):
        super(Listener, self).__init__()
        input_size = (input_size - 1) << 5 if input_size % 2 else input_size << 5
        rnn_cell = self.supported_rnns[rnn_type]
        self.device = device
        self.cnn = MaskCNN(
            nn.Sequential(
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
        )
        self.rnn = rnn_cell(
            input_size=input_size,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout_p
        )

    def forward(self, inputs, input_lengths):
        inputs = inputs.unsqueeze(1).permute(0, 1, 3, 2)
        output, output_lengths = self.cnn(inputs, input_lengths)

        B, C, H, S = output.size()

        output = output.view(B, C * H, S)
        output = output.transpose(1, 2).transpose(0, 1).contiguous()

        inputs = nn.utils.rnn.pack_padded_sequence(output, output_lengths)
        output, hidden = self.rnn(inputs)
        output, _ = nn.utils.rnn.pad_packed_sequence(output)

        output = output.transpose(0, 1)   # (batch_size, seq_len, hidden_dim)

        return output, hidden

    def flatten_parameters(self):
        self.rnn.flatten_parameters()
